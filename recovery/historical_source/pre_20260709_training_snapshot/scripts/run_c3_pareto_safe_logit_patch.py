from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    RSCDSurfaceDataset,
    build_class_map,
    build_model,
    collate,
    factor_confusion_summary,
    load_config,
    write_hard_pair_metrics,
    write_wcs_diagnosis,
)
from friction_affordance.rscd_factors import build_rscd_factor_spec, canonical_class_label  # noqa: E402
from friction_affordance.transforms import build_transforms  # noqa: E402
from friction_affordance.utils import resolve_device, set_seed  # noqa: E402


@dataclass(frozen=True)
class PatchRule:
    source: int
    target: int
    topk: int
    margin: float
    delta: float

    def name(self, idx_to_class: dict[int, str]) -> str:
        return (
            f"{idx_to_class[self.source]}->{idx_to_class[self.target]}"
            f"|top{self.topk}|m<={self.margin:.2f}|d={self.delta:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-constrained RSCD hard-edge logit patch for C3-FaRNet. "
            "It accepts a rule only if validation Top-1/Macro-F1 do not regress "
            "and protected per-class F1 drops stay within a small tolerance."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-val-samples-per-class", type=int, default=None)
    parser.add_argument("--max-test-samples-per-class", type=int, default=None)
    parser.add_argument("--min-confusion-count", type=int, default=3)
    parser.add_argument("--weak-f1-threshold", type=float, default=0.86)
    parser.add_argument("--max-rules", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--safety-fraction", type=float, default=0.5)
    parser.add_argument("--min-target-f1-gain-pp", type=float, default=0.01)
    parser.add_argument("--min-macro-gain-pp", type=float, default=0.0)
    parser.add_argument("--min-top1-gain-pp", type=float, default=0.0)
    parser.add_argument("--max-protected-f1-drop-pp", type=float, default=0.02)
    parser.add_argument("--protect-all-classes", action="store_true")
    parser.add_argument(
        "--blocked-source-classes",
        nargs="*",
        default=[],
        help="Validation-only safety filter: do not generate rules that move predictions out of these source classes.",
    )
    parser.add_argument(
        "--max-rule-delta",
        type=float,
        default=None,
        help="Optional upper bound for candidate rule delta; useful for conservative no-harm sweeps.",
    )
    parser.add_argument(
        "--min-source-f1-for-rules",
        type=float,
        default=None,
        help="Optional full-validation source-class F1 floor. Candidate rules can move predictions out of a source class only when the base validation F1 of that source is at least this value.",
    )
    parser.add_argument(
        "--final-full-val-noharm",
        action="store_true",
        help="After greedy tune/safety selection, keep only rules that also satisfy constraints on the full validation set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    if args.max_val_samples_per_class is not None:
        cfg["eval"]["max_val_samples_per_class"] = int(args.max_val_samples_per_class)
    if args.max_test_samples_per_class is not None:
        cfg["eval"]["max_test_samples_per_class"] = int(args.max_test_samples_per_class)
    if args.batch_size is not None:
        cfg["eval"]["batch_size"] = int(args.batch_size)
    cfg["train"]["num_workers"] = int(args.num_workers)

    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = build_model(cfg, class_to_idx).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    val_cache = args.output_dir / "val_logits.npz"
    test_cache = args.output_dir / "test_logits.npz"
    val = collect_or_load(
        cache_path=val_cache,
        manifest=Path(data["val_manifest"]),
        cfg=cfg,
        class_to_idx=class_to_idx,
        model=model,
        device=device,
        split="val",
        force_cache=bool(args.force_cache),
    )
    test = collect_or_load(
        cache_path=test_cache,
        manifest=Path(data["test_manifest"]),
        cfg=cfg,
        class_to_idx=class_to_idx,
        model=model,
        device=device,
        split="test",
        force_cache=bool(args.force_cache),
    )

    result = run_patch_search(
        val=val,
        test=test,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        min_confusion_count=int(args.min_confusion_count),
        weak_f1_threshold=float(args.weak_f1_threshold),
        max_rules=int(args.max_rules),
        max_candidates=int(args.max_candidates),
        safety_fraction=float(args.safety_fraction),
        min_target_gain=float(args.min_target_f1_gain_pp) / 100.0,
        min_macro_gain=float(args.min_macro_gain_pp) / 100.0,
        min_top1_gain=float(args.min_top1_gain_pp) / 100.0,
        max_protected_drop=float(args.max_protected_f1_drop_pp) / 100.0,
        protect_all_classes=bool(args.protect_all_classes),
        seed=int(args.seed),
        final_full_val_noharm=bool(args.final_full_val_noharm),
        blocked_source_classes=[str(v) for v in args.blocked_source_classes],
        max_rule_delta=args.max_rule_delta,
        min_source_f1_for_rules=args.min_source_f1_for_rules,
    )

    result["protocol"] = {
        "method": "RSCD Pareto-safe hard-edge logit patch",
        "claim_boundary": (
            "Post-hoc validation-constrained diagnostic. It is RSCD-specific: "
            "rules are restricted to factor-neighbor hard edges and accepted only "
            "when validation Top-1, Macro-F1, and protected class F1 constraints hold."
        ),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "val_cache": str(val_cache),
        "test_cache": str(test_cache),
        "max_val_samples_per_class": cfg["eval"].get("max_val_samples_per_class"),
        "max_test_samples_per_class": cfg["eval"].get("max_test_samples_per_class"),
        "min_confusion_count": int(args.min_confusion_count),
        "weak_f1_threshold": float(args.weak_f1_threshold),
        "max_rules": int(args.max_rules),
        "max_candidates": int(args.max_candidates),
        "safety_fraction": float(args.safety_fraction),
        "min_target_f1_gain_pp": float(args.min_target_f1_gain_pp),
        "min_macro_gain_pp": float(args.min_macro_gain_pp),
        "min_top1_gain_pp": float(args.min_top1_gain_pp),
        "max_protected_f1_drop_pp": float(args.max_protected_f1_drop_pp),
        "protect_all_classes": bool(args.protect_all_classes),
        "blocked_source_classes": [str(v) for v in args.blocked_source_classes],
        "max_rule_delta": args.max_rule_delta,
        "min_source_f1_for_rules": args.min_source_f1_for_rules,
        "final_full_val_noharm": bool(args.final_full_val_noharm),
    }

    (args.output_dir / "pareto_safe_logit_patch.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_metric_outputs(args.output_dir, result, idx_to_class, class_to_idx)
    print(
        json.dumps(
            {
                "val_base": result["val_base"]["summary"],
                "val_patched": result["val_patched"]["summary"],
                "test_base": result["test_base"]["summary"],
                "test_patched": result["test_patched"]["summary"],
                "accepted_rules": len(result["accepted_rules"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


@torch.no_grad()
def collect_or_load(
    *,
    cache_path: Path,
    manifest: Path,
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    force_cache: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force_cache:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    image_size = int(cfg["data"].get("image_size", 192))
    transform = build_transforms(
        image_size,
        train=False,
        aug_cfg={"resize_mode": str(cfg["data"].get("eval_resize_mode", "letterbox"))},
    )
    dataset = RSCDSurfaceDataset(
        manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        max_samples_per_class=cfg["eval"].get(f"max_{split}_samples_per_class"),
        max_samples=cfg["eval"].get(f"max_{split}_samples"),
        seed=int(cfg.get("seed", 79)) + (1 if split == "val" else 2),
    )
    num_workers = int(cfg["train"].get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg["eval"].get("batch_size", cfg["train"].get("batch_size", 8))),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)

    logits_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    path_rows: list[str] = []
    for batch in tqdm(loader, desc=f"collect-{split}", leave=False, ascii=True):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(image, return_aux=True)
            logits = out["logits"]
        logits_rows.append(logits.detach().float().cpu().numpy().astype(np.float32))
        label_rows.append(batch["label"].detach().cpu().numpy().astype(np.int64))
        path_rows.extend([str(p) for p in batch["image_path"]])

    payload = {
        "logits": np.concatenate(logits_rows, axis=0),
        "label": np.concatenate(label_rows, axis=0),
        "image_path": np.asarray(path_rows, dtype=object),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def run_patch_search(
    *,
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    min_confusion_count: int,
    weak_f1_threshold: float,
    max_rules: int,
    max_candidates: int,
    safety_fraction: float,
    min_target_gain: float,
    min_macro_gain: float,
    min_top1_gain: float,
    max_protected_drop: float,
    protect_all_classes: bool,
    seed: int,
    final_full_val_noharm: bool,
    blocked_source_classes: list[str],
    max_rule_delta: float | None,
    min_source_f1_for_rules: float | None,
) -> dict[str, Any]:
    val_logits = val["logits"].astype(np.float32).copy()
    test_logits = test["logits"].astype(np.float32).copy()
    y_val = val["label"].astype(np.int64)
    y_test = test["label"].astype(np.int64)
    labels = list(range(len(idx_to_class)))

    base_val = metric_bundle(y_val, val_logits.argmax(axis=1), idx_to_class, class_to_idx)
    tune_idx, safety_idx = stratified_safety_split(y_val, safety_fraction=float(safety_fraction), seed=int(seed))
    tune_logits = val_logits[tune_idx].copy()
    safety_logits = val_logits[safety_idx].copy()
    y_tune = y_val[tune_idx]
    y_safety = y_val[safety_idx]
    weak_classes = {
        int(class_to_idx[name])
        for name, row in base_val["classification_report"].items()
        if name in class_to_idx and float(row.get("f1-score", 0.0)) <= float(weak_f1_threshold)
    }
    weak_classes.update(
        int(class_to_idx[name])
        for name in [
            "water_concrete_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_severe",
            "dry_concrete_slight",
            "dry_asphalt_severe",
        ]
        if name in class_to_idx
    )

    candidates = build_candidates(
        y_val=y_tune,
        logits=tune_logits,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        weak_classes=weak_classes,
        min_confusion_count=int(min_confusion_count),
        max_candidates=int(max_candidates),
        blocked_source_classes={
            canonical_class_label(name)
            for name in blocked_source_classes
        },
        max_rule_delta=max_rule_delta,
        source_f1_floor=min_source_f1_for_rules,
        full_val_source_f1=np.asarray(base_val["per_class_f1"], dtype=np.float64),
    )
    current_tune_logits = tune_logits.copy()
    current_safety_logits = safety_logits.copy()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    current_tune_metrics = fast_metric_bundle(y_tune, current_tune_logits.argmax(axis=1), idx_to_class)
    current_safety_metrics = fast_metric_bundle(y_safety, current_safety_logits.argmax(axis=1), idx_to_class)

    for _ in range(max_rules):
        best: tuple[
            float,
            PatchRule,
            np.ndarray,
            np.ndarray,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ] | None = None
        for rule in candidates:
            if any(same_rule(rule, accepted_rule["rule_raw"], idx_to_class) for accepted_rule in accepted):
                continue
            patched_tune = apply_rule(current_tune_logits, rule)
            patched_safety = apply_rule(current_safety_logits, rule)
            tune_metrics = fast_metric_bundle(y_tune, patched_tune.argmax(axis=1), idx_to_class)
            safety_metrics = fast_metric_bundle(y_safety, patched_safety.argmax(axis=1), idx_to_class)
            ok_tune, tune_checks = passes_constraints(
                base=current_tune_metrics,
                new=tune_metrics,
                target=rule.target,
                labels=labels,
                weak_classes=weak_classes,
                min_target_gain=min_target_gain,
                min_macro_gain=min_macro_gain,
                min_top1_gain=min_top1_gain,
                max_protected_drop=max_protected_drop,
                protect_all_classes=protect_all_classes,
            )
            ok_safety, safety_checks = passes_constraints(
                base=current_safety_metrics,
                new=safety_metrics,
                target=rule.target,
                labels=labels,
                weak_classes=weak_classes,
                min_target_gain=0.0,
                min_macro_gain=0.0,
                min_top1_gain=0.0,
                max_protected_drop=max_protected_drop,
                protect_all_classes=protect_all_classes,
            )
            score = (
                float(tune_metrics["summary"]["macro_f1"]) - float(current_tune_metrics["summary"]["macro_f1"])
                + float(safety_metrics["summary"]["macro_f1"]) - float(current_safety_metrics["summary"]["macro_f1"])
                + 0.5 * (float(tune_metrics["summary"]["top1"]) - float(current_tune_metrics["summary"]["top1"]))
                + 0.5 * (float(safety_metrics["summary"]["top1"]) - float(current_safety_metrics["summary"]["top1"]))
                + tune_checks.get("target_f1_gain", 0.0)
            )
            record = {
                "rule": rule.name(idx_to_class),
                "rule_raw": rule_to_dict(rule, idx_to_class),
                "tune_checks": tune_checks,
                "safety_checks": safety_checks,
                "score": float(score),
            }
            if not (ok_tune and ok_safety):
                if len(rejected) < 200:
                    rejected.append(record)
                continue
            if best is None or score > best[0]:
                best = (score, rule, patched_tune, patched_safety, tune_metrics, safety_metrics, tune_checks, safety_checks)
        if best is None:
            break
        score, rule, current_tune_logits, current_safety_logits, current_tune_metrics, current_safety_metrics, tune_checks, safety_checks = best
        accepted.append(
            {
                "rule": rule.name(idx_to_class),
                "rule_raw": rule_to_dict(rule, idx_to_class),
                "tune_checks": tune_checks,
                "safety_checks": safety_checks,
                "score": float(score),
                "tune_summary_after": current_tune_metrics["summary"],
                "safety_summary_after": current_safety_metrics["summary"],
            }
        )

    patched_val_logits = val_logits.copy()
    for item in accepted:
        patched_val_logits = apply_rule(patched_val_logits, dict_to_rule(item["rule_raw"], class_to_idx))

    if final_full_val_noharm and accepted:
        accepted, full_val_rejected = enforce_full_validation_noharm(
            accepted=accepted,
            val_logits=val_logits,
            y_val=y_val,
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
            labels=labels,
            weak_classes=weak_classes,
            min_target_gain=min_target_gain,
            min_macro_gain=min_macro_gain,
            min_top1_gain=min_top1_gain,
            max_protected_drop=max_protected_drop,
            protect_all_classes=protect_all_classes,
        )
        rejected.extend(full_val_rejected[: max(0, 200 - len(rejected))])
        patched_val_logits = val_logits.copy()
        for item in accepted:
            patched_val_logits = apply_rule(patched_val_logits, dict_to_rule(item["rule_raw"], class_to_idx))
    patched_test_logits = test_logits.copy()
    for item in accepted:
        patched_test_logits = apply_rule(patched_test_logits, dict_to_rule(item["rule_raw"], class_to_idx))

    base_test = metric_bundle(y_test, test_logits.argmax(axis=1), idx_to_class, class_to_idx)
    patched_test = metric_bundle(y_test, patched_test_logits.argmax(axis=1), idx_to_class, class_to_idx)
    return {
        "val_base": base_val,
        "val_patched": metric_bundle(y_val, patched_val_logits.argmax(axis=1), idx_to_class, class_to_idx),
        "val_tune_base": metric_bundle(y_tune, tune_logits.argmax(axis=1), idx_to_class, class_to_idx),
        "val_tune_patched": current_tune_metrics,
        "val_safety_base": metric_bundle(y_safety, safety_logits.argmax(axis=1), idx_to_class, class_to_idx),
        "val_safety_patched": current_safety_metrics,
        "test_base": base_test,
        "test_patched": patched_test,
        "accepted_rules": accepted,
        "rejected_rule_examples": rejected,
        "candidate_count": int(len(candidates)),
        "validation_split": {"tune": int(len(tune_idx)), "safety": int(len(safety_idx))},
        "weak_classes": [idx_to_class[i] for i in sorted(weak_classes)],
        "patched_test_pred": patched_test_logits.argmax(axis=1).astype(int).tolist(),
        "base_test_pred": test_logits.argmax(axis=1).astype(int).tolist(),
        "test_true": y_test.astype(int).tolist(),
        "test_image_path": [str(p) for p in test["image_path"].tolist()],
    }


def build_candidates(
    *,
    y_val: np.ndarray,
    logits: np.ndarray,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
        weak_classes: set[int],
        min_confusion_count: int,
        max_candidates: int,
        blocked_source_classes: set[str],
        max_rule_delta: float | None,
        source_f1_floor: float | None,
        full_val_source_f1: np.ndarray,
) -> list[PatchRule]:
    spec = build_rscd_factor_spec(class_to_idx)
    hard_edges = {tuple(sorted((int(pair.left), int(pair.right)))) for pair in spec.hard_pairs}
    pred = logits.argmax(axis=1)
    rows: list[PatchRule] = []
    top_order = np.argsort(-logits, axis=1)
    topk_grid = [2, 3]
    margin_grid = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
    delta_grid = [0.05, 0.10, 0.16, 0.22, 0.30]

    confusion: dict[tuple[int, int], int] = {}
    for true_idx, pred_idx in zip(y_val.tolist(), pred.tolist(), strict=True):
        if int(true_idx) == int(pred_idx):
            continue
        key = (int(true_idx), int(pred_idx))
        confusion[key] = confusion.get(key, 0) + 1

    for (target, source), count in sorted(confusion.items(), key=lambda x: (-x[1], x[0])):
        if count < int(min_confusion_count):
            continue
        if canonical_class_label(idx_to_class[int(source)]) in blocked_source_classes:
            continue
        if source_f1_floor is not None and float(full_val_source_f1[int(source)]) < float(source_f1_floor):
            continue
        if target not in weak_classes:
            continue
        if tuple(sorted((int(target), int(source)))) not in hard_edges:
            continue
        for topk in topk_grid:
            in_topk = np.any(top_order[:, :topk] == int(target), axis=1)
            source_mask = pred == int(source)
            if int((source_mask & in_topk).sum()) < int(min_confusion_count):
                continue
            for margin in margin_grid:
                margin_mask = (logits[:, int(source)] - logits[:, int(target)]) <= float(margin)
                support = int((source_mask & in_topk & margin_mask).sum())
                if support < int(min_confusion_count):
                    continue
                for delta in delta_grid:
                    if max_rule_delta is not None and float(delta) > float(max_rule_delta):
                        continue
                    rows.append(PatchRule(source=int(source), target=int(target), topk=int(topk), margin=float(margin), delta=float(delta)))

    uniq: dict[tuple[int, int, int, float, float], PatchRule] = {}
    for rule in rows:
        uniq[(rule.source, rule.target, rule.topk, rule.margin, rule.delta)] = rule
    return list(uniq.values())[: int(max_candidates)]


def stratified_safety_split(y: np.ndarray, *, safety_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    safety_fraction = float(np.clip(safety_fraction, 0.0, 0.9))
    if safety_fraction <= 0.0:
        idx = np.arange(len(y), dtype=np.int64)
        return idx, idx
    rng = np.random.default_rng(int(seed))
    tune: list[int] = []
    safety: list[int] = []
    for label in sorted(set(y.astype(int).tolist())):
        label_idx = np.flatnonzero(y == int(label))
        label_idx = rng.permutation(label_idx)
        n_safety = int(round(len(label_idx) * safety_fraction))
        if len(label_idx) > 1:
            n_safety = min(max(n_safety, 1), len(label_idx) - 1)
        else:
            n_safety = 0
        safety.extend(label_idx[:n_safety].astype(int).tolist())
        tune.extend(label_idx[n_safety:].astype(int).tolist())
    return np.asarray(tune, dtype=np.int64), np.asarray(safety, dtype=np.int64)


def enforce_full_validation_noharm(
    *,
    accepted: list[dict[str, Any]],
    val_logits: np.ndarray,
    y_val: np.ndarray,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    labels: list[int],
    weak_classes: set[int],
    min_target_gain: float,
    min_macro_gain: float,
    min_top1_gain: float,
    max_protected_drop: float,
    protect_all_classes: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    current_logits = val_logits.copy()
    current_metrics = fast_metric_bundle(y_val, current_logits.argmax(axis=1), idx_to_class)
    for item in accepted:
        rule = dict_to_rule(item["rule_raw"], class_to_idx)
        candidate_logits = apply_rule(current_logits, rule)
        candidate_metrics = fast_metric_bundle(y_val, candidate_logits.argmax(axis=1), idx_to_class)
        ok, checks = passes_constraints(
            base=current_metrics,
            new=candidate_metrics,
            target=rule.target,
            labels=labels,
            weak_classes=weak_classes,
            min_target_gain=min_target_gain,
            min_macro_gain=min_macro_gain,
            min_top1_gain=min_top1_gain,
            max_protected_drop=max_protected_drop,
            protect_all_classes=protect_all_classes,
        )
        if ok:
            updated = dict(item)
            updated["full_val_checks"] = checks
            kept.append(updated)
            current_logits = candidate_logits
            current_metrics = candidate_metrics
        else:
            rejected.append(
                {
                    "rule": item.get("rule"),
                    "rule_raw": item.get("rule_raw"),
                    "full_val_checks": checks,
                    "reason": "failed_final_full_val_noharm",
                }
            )
    return kept, rejected


def apply_rule(logits: np.ndarray, rule: PatchRule) -> np.ndarray:
    out = logits.copy()
    pred = out.argmax(axis=1)
    order = np.argsort(-out, axis=1)
    in_topk = np.any(order[:, : int(rule.topk)] == int(rule.target), axis=1)
    close = (out[:, int(rule.source)] - out[:, int(rule.target)]) <= float(rule.margin)
    mask = (pred == int(rule.source)) & in_topk & close
    if bool(mask.any()):
        out[mask, int(rule.target)] += float(rule.delta)
        out[mask, int(rule.source)] -= float(rule.delta) * 0.25
    return out


def passes_constraints(
    *,
    base: dict[str, Any],
    new: dict[str, Any],
    target: int,
    labels: list[int],
    weak_classes: set[int],
    min_target_gain: float,
    min_macro_gain: float,
    min_top1_gain: float,
    max_protected_drop: float,
    protect_all_classes: bool,
) -> tuple[bool, dict[str, float]]:
    base_summary = base["summary"]
    new_summary = new["summary"]
    base_f1 = np.asarray(base["per_class_f1"], dtype=np.float64)
    new_f1 = np.asarray(new["per_class_f1"], dtype=np.float64)
    drops = base_f1 - new_f1
    protected = np.ones(len(labels), dtype=bool) if protect_all_classes else np.asarray([i not in weak_classes for i in labels], dtype=bool)
    protected[int(target)] = False
    max_drop = float(drops[protected].max()) if bool(protected.any()) else 0.0
    checks = {
        "top1_gain": float(new_summary["top1"] - base_summary["top1"]),
        "macro_f1_gain": float(new_summary["macro_f1"] - base_summary["macro_f1"]),
        "target_f1_gain": float(new_f1[int(target)] - base_f1[int(target)]),
        "max_protected_f1_drop": max_drop,
    }
    ok = (
        checks["top1_gain"] + 1e-12 >= float(min_top1_gain)
        and checks["macro_f1_gain"] + 1e-12 >= float(min_macro_gain)
        and checks["target_f1_gain"] + 1e-12 >= float(min_target_gain)
        and checks["max_protected_f1_drop"] <= float(max_protected_drop) + 1e-12
    )
    return bool(ok), checks


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
) -> dict[str, Any]:
    labels = list(range(len(idx_to_class)))
    names = [idx_to_class[i] for i in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
    per_class_f1 = [float(report[name]["f1-score"]) for name in names]
    factor_summary = factor_confusion_summary(y_true.astype(int).tolist(), y_pred.astype(int).tolist(), build_rscd_factor_spec(class_to_idx), idx_to_class)
    summary = {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
        "num_classes": int(len(labels)),
        "weakest_class": names[int(np.argmin(per_class_f1))],
        "weakest_f1": float(np.min(per_class_f1)),
        "water_concrete_slight_f1": float(report.get("water_concrete_slight", {}).get("f1-score", 0.0)),
        "num_errors": int(np.sum(y_true != y_pred)),
    }
    summary.update(factor_summary["summary"])
    return {
        "summary": summary,
        "classification_report": report,
        "factor_confusion_summary": factor_summary,
        "per_class_f1": per_class_f1,
    }


def fast_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, Any]:
    num_classes = len(idx_to_class)
    per_class_f1: list[float] = []
    for idx in range(num_classes):
        true_pos = (y_true == idx)
        pred_pos = (y_pred == idx)
        tp = float(np.sum(true_pos & pred_pos))
        fp = float(np.sum(~true_pos & pred_pos))
        fn = float(np.sum(true_pos & ~pred_pos))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        per_class_f1.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    per_class = np.asarray(per_class_f1, dtype=np.float64)
    return {
        "summary": {
            "top1": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
            "macro_f1": float(per_class.mean()) if len(per_class) else 0.0,
            "weakest_class": idx_to_class[int(np.argmin(per_class))] if len(per_class) else "",
            "weakest_f1": float(per_class.min()) if len(per_class) else 0.0,
            "num_samples": int(len(y_true)),
            "num_errors": int(np.sum(y_true != y_pred)),
        },
        "per_class_f1": per_class_f1,
    }


def write_metric_outputs(out_dir: Path, result: dict[str, Any], idx_to_class: dict[int, str], class_to_idx: dict[str, int]) -> None:
    y_true = np.asarray(result["test_true"], dtype=int)
    base_pred = np.asarray(result["base_test_pred"], dtype=int)
    patched_pred = np.asarray(result["patched_test_pred"], dtype=int)
    names = [idx_to_class[i] for i in range(len(idx_to_class))]

    with (out_dir / "per_class_test_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "base_f1", "patched_f1", "delta_f1", "base_recall", "patched_recall", "delta_recall"])
        base_report = result["test_base"]["classification_report"]
        patched_report = result["test_patched"]["classification_report"]
        for name in names:
            base_row = base_report[name]
            patched_row = patched_report[name]
            w.writerow(
                [
                    name,
                    float(base_row["f1-score"]),
                    float(patched_row["f1-score"]),
                    float(patched_row["f1-score"]) - float(base_row["f1-score"]),
                    float(base_row["recall"]),
                    float(patched_row["recall"]),
                    float(patched_row["recall"]) - float(base_row["recall"]),
                ]
            )
    pd.DataFrame(result["accepted_rules"]).to_csv(out_dir / "accepted_rules.csv", index=False, encoding="utf-8-sig")
    pred_rows = [
        {
            "image_path": str(path),
            "true_label": idx_to_class[int(t)],
            "base_pred_label": idx_to_class[int(b)],
            "patched_pred_label": idx_to_class[int(p)],
            "changed": bool(int(b) != int(p)),
        }
        for path, t, b, p in zip(result["test_image_path"], y_true.tolist(), base_pred.tolist(), patched_pred.tolist(), strict=True)
    ]
    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions_test_patched.csv", index=False, encoding="utf-8-sig")

    metrics_like = {
        "classification_report": result["test_patched"]["classification_report"],
        "factor_confusion_summary": result["test_patched"]["factor_confusion_summary"],
        "summary": result["test_patched"]["summary"],
        "y_true": y_true.tolist(),
        "y_pred": patched_pred.tolist(),
    }
    write_hard_pair_metrics(out_dir / "hard_pair_metrics_patched.csv", metrics_like, idx_to_class)
    write_wcs_diagnosis(out_dir / "water_concrete_slight_diagnosis_patched.json", metrics_like, idx_to_class)


def rule_to_dict(rule: PatchRule, idx_to_class: dict[int, str]) -> dict[str, Any]:
    return {
        "source": idx_to_class[int(rule.source)],
        "target": idx_to_class[int(rule.target)],
        "topk": int(rule.topk),
        "margin": float(rule.margin),
        "delta": float(rule.delta),
    }


def dict_to_rule(payload: dict[str, Any], class_to_idx: dict[str, int]) -> PatchRule:
    return PatchRule(
        source=int(class_to_idx[canonical_class_label(str(payload["source"]))]),
        target=int(class_to_idx[canonical_class_label(str(payload["target"]))]),
        topk=int(payload["topk"]),
        margin=float(payload["margin"]),
        delta=float(payload["delta"]),
    )


def same_rule(rule: PatchRule, payload: dict[str, Any], idx_to_class: dict[int, str]) -> bool:
    return (
        idx_to_class[int(rule.source)] == str(payload["source"])
        and idx_to_class[int(rule.target)] == str(payload["target"])
        and int(rule.topk) == int(payload["topk"])
        and abs(float(rule.margin) - float(payload["margin"])) < 1e-9
        and abs(float(rule.delta) - float(payload["delta"])) < 1e-9
    )


if __name__ == "__main__":
    main()
