from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
from run_c3_pareto_safe_logit_patch import (  # noqa: E402
    fast_metric_bundle,
    metric_bundle,
    stratified_safety_split,
)


@dataclass(frozen=True)
class RouterRule:
    source: int
    target: int
    anchor_margin: float
    specialist_margin: float
    anchor_conf: float
    any_source: bool = False

    def name(self, idx_to_class: dict[int, str]) -> str:
        source = "ANY" if bool(self.any_source) else idx_to_class[self.source]
        return (
            f"{source}->{idx_to_class[self.target]}"
            f"|am<={self.anchor_margin:.2f}|sm>={self.specialist_margin:.2f}"
            f"|ac<={self.anchor_conf:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RSCD selective anchor-router. The anchor model is the default. "
            "A specialist prediction is accepted only on validation-safe hard-edge "
            "regions, which implements a reject-to-anchor version of expert routing."
        )
    )
    parser.add_argument("--anchor-config", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-config", type=Path, required=True)
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-val-samples-per-class", type=int, default=None)
    parser.add_argument("--max-test-samples-per-class", type=int, default=None)
    parser.add_argument("--min-switch-count", type=int, default=3)
    parser.add_argument("--weak-f1-threshold", type=float, default=0.86)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--rule-family", choices=("pair", "target", "both"), default="pair")
    parser.add_argument("--safety-fraction", type=float, default=0.5)
    parser.add_argument("--min-target-f1-gain-pp", type=float, default=0.0)
    parser.add_argument("--min-macro-gain-pp", type=float, default=0.0)
    parser.add_argument("--min-top1-gain-pp", type=float, default=0.0)
    parser.add_argument("--safety-min-target-f1-gain-pp", type=float, default=0.0)
    parser.add_argument("--safety-min-macro-gain-pp", type=float, default=0.0)
    parser.add_argument("--safety-min-top1-gain-pp", type=float, default=0.0)
    parser.add_argument("--max-protected-f1-drop-pp", type=float, default=0.02)
    parser.add_argument("--protect-all-classes", action="store_true")
    parser.add_argument(
        "--shuffle-safety-split",
        action="store_true",
        help="Shuffle samples inside each class before the tune/safety split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor_cfg = load_config(args.anchor_config)
    specialist_cfg = load_config(args.specialist_config)
    for cfg in (anchor_cfg, specialist_cfg):
        if args.max_val_samples_per_class is not None:
            cfg["eval"]["max_val_samples_per_class"] = int(args.max_val_samples_per_class)
        if args.max_test_samples_per_class is not None:
            cfg["eval"]["max_test_samples_per_class"] = int(args.max_test_samples_per_class)
        if args.batch_size is not None:
            cfg["eval"]["batch_size"] = int(args.batch_size)
        cfg["train"]["num_workers"] = int(args.num_workers)

    data = anchor_cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    anchor_model = load_model(anchor_cfg, args.anchor_checkpoint, class_to_idx, device)
    specialist_model = load_model(specialist_cfg, args.specialist_checkpoint, class_to_idx, device)

    anchor_val = collect_logits(
        cache_path=args.output_dir / "anchor_val_logits.npz",
        manifest=Path(data["val_manifest"]),
        cfg=anchor_cfg,
        class_to_idx=class_to_idx,
        model=anchor_model,
        device=device,
        split="val",
        force_cache=bool(args.force_cache),
    )
    anchor_test = collect_logits(
        cache_path=args.output_dir / "anchor_test_logits.npz",
        manifest=Path(data["test_manifest"]),
        cfg=anchor_cfg,
        class_to_idx=class_to_idx,
        model=anchor_model,
        device=device,
        split="test",
        force_cache=bool(args.force_cache),
    )
    specialist_val = collect_logits(
        cache_path=args.output_dir / "specialist_val_logits.npz",
        manifest=Path(data["val_manifest"]),
        cfg=specialist_cfg,
        class_to_idx=class_to_idx,
        model=specialist_model,
        device=device,
        split="val",
        force_cache=bool(args.force_cache),
    )
    specialist_test = collect_logits(
        cache_path=args.output_dir / "specialist_test_logits.npz",
        manifest=Path(data["test_manifest"]),
        cfg=specialist_cfg,
        class_to_idx=class_to_idx,
        model=specialist_model,
        device=device,
        split="test",
        force_cache=bool(args.force_cache),
    )

    if not np.array_equal(anchor_val["label"], specialist_val["label"]):
        raise ValueError("anchor and specialist validation labels are not aligned")
    if not np.array_equal(anchor_test["label"], specialist_test["label"]):
        raise ValueError("anchor and specialist test labels are not aligned")

    result = run_router_search(
        val_anchor=anchor_val,
        val_specialist=specialist_val,
        test_anchor=anchor_test,
        test_specialist=specialist_test,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        min_switch_count=int(args.min_switch_count),
        weak_f1_threshold=float(args.weak_f1_threshold),
        max_rules=int(args.max_rules),
        max_candidates=int(args.max_candidates),
        rule_family=str(args.rule_family),
        safety_fraction=float(args.safety_fraction),
        min_target_gain=float(args.min_target_f1_gain_pp) / 100.0,
        min_macro_gain=float(args.min_macro_gain_pp) / 100.0,
        min_top1_gain=float(args.min_top1_gain_pp) / 100.0,
        safety_min_target_gain=float(args.safety_min_target_f1_gain_pp) / 100.0,
        safety_min_macro_gain=float(args.safety_min_macro_gain_pp) / 100.0,
        safety_min_top1_gain=float(args.safety_min_top1_gain_pp) / 100.0,
        max_protected_drop=float(args.max_protected_f1_drop_pp) / 100.0,
        protect_all_classes=bool(args.protect_all_classes),
        shuffle_safety_split=bool(args.shuffle_safety_split),
        seed=int(args.seed),
    )

    result["protocol"] = {
        "method": "RSCD selective anchor-router",
        "claim_boundary": (
            "Validation-constrained reject-to-anchor expert router. It is not a "
            "generic ensemble: switches are restricted to RSCD factor-neighbor "
            "hard edges and accepted only if tune/safety validation metrics do not "
            "regress under protected-class F1 constraints."
        ),
        "anchor_config": str(args.anchor_config),
        "anchor_checkpoint": str(args.anchor_checkpoint),
        "specialist_config": str(args.specialist_config),
        "specialist_checkpoint": str(args.specialist_checkpoint),
        "max_val_samples_per_class": anchor_cfg["eval"].get("max_val_samples_per_class"),
        "max_test_samples_per_class": anchor_cfg["eval"].get("max_test_samples_per_class"),
        "min_switch_count": int(args.min_switch_count),
        "weak_f1_threshold": float(args.weak_f1_threshold),
        "max_rules": int(args.max_rules),
        "max_candidates": int(args.max_candidates),
        "rule_family": str(args.rule_family),
        "safety_fraction": float(args.safety_fraction),
        "safety_min_target_f1_gain_pp": float(args.safety_min_target_f1_gain_pp),
        "safety_min_macro_gain_pp": float(args.safety_min_macro_gain_pp),
        "safety_min_top1_gain_pp": float(args.safety_min_top1_gain_pp),
        "max_protected_f1_drop_pp": float(args.max_protected_f1_drop_pp),
        "protect_all_classes": bool(args.protect_all_classes),
        "shuffle_safety_split": bool(args.shuffle_safety_split),
    }
    write_outputs(args.output_dir, result, idx_to_class, class_to_idx)
    print(
        json.dumps(
            {
                "val_anchor": result["val_anchor"]["summary"],
                "val_routed": result["val_routed"]["summary"],
                "test_anchor": result["test_anchor"]["summary"],
                "test_specialist": result["test_specialist"]["summary"],
                "test_routed": result["test_routed"]["summary"],
                "accepted_rules": len(result["accepted_rules"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def load_model(
    cfg: dict[str, Any],
    checkpoint: Path,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(cfg, class_to_idx).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    payload = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(payload, strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_logits(
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
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    logits_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    path_rows: list[str] = []
    for batch in tqdm(loader, desc=f"collect-{split}", leave=False, ascii=True):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(image, return_aux=True)
            logits = out["logits"] if isinstance(out, dict) else out
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


def stratified_safety_split_random(
    y: np.ndarray,
    *,
    safety_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    safety_fraction = float(np.clip(safety_fraction, 0.0, 0.9))
    if safety_fraction <= 0.0:
        idx = np.arange(len(y), dtype=np.int64)
        return idx, idx
    rng = np.random.default_rng(int(seed))
    tune: list[int] = []
    safety: list[int] = []
    for label in sorted(set(y.astype(int).tolist())):
        label_idx = np.flatnonzero(y == int(label)).astype(np.int64)
        shuffled = rng.permutation(label_idx)
        n_safety = int(round(len(shuffled) * safety_fraction))
        if len(shuffled) > 1:
            n_safety = min(max(n_safety, 1), len(shuffled) - 1)
        else:
            n_safety = 0
        safety.extend(shuffled[:n_safety].astype(int).tolist())
        tune.extend(shuffled[n_safety:].astype(int).tolist())
    return np.asarray(tune, dtype=np.int64), np.asarray(safety, dtype=np.int64)


def run_router_search(
    *,
    val_anchor: dict[str, np.ndarray],
    val_specialist: dict[str, np.ndarray],
    test_anchor: dict[str, np.ndarray],
    test_specialist: dict[str, np.ndarray],
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    min_switch_count: int,
    weak_f1_threshold: float,
    max_rules: int,
    max_candidates: int,
    rule_family: str,
    safety_fraction: float,
    min_target_gain: float,
    min_macro_gain: float,
    min_top1_gain: float,
    safety_min_target_gain: float,
    safety_min_macro_gain: float,
    safety_min_top1_gain: float,
    max_protected_drop: float,
    protect_all_classes: bool,
    shuffle_safety_split: bool,
    seed: int,
) -> dict[str, Any]:
    y_val = val_anchor["label"].astype(np.int64)
    y_test = test_anchor["label"].astype(np.int64)
    val_a = val_anchor["logits"].astype(np.float32)
    val_s = val_specialist["logits"].astype(np.float32)
    test_a = test_anchor["logits"].astype(np.float32)
    test_s = test_specialist["logits"].astype(np.float32)
    labels = list(range(len(idx_to_class)))

    base_val = metric_bundle(y_val, val_a.argmax(axis=1), idx_to_class, class_to_idx)
    weak_classes = {
        int(class_to_idx[name])
        for name, row in base_val["classification_report"].items()
        if name in class_to_idx and float(row.get("f1-score", 0.0)) <= float(weak_f1_threshold)
    }
    weak_classes.update(
        int(class_to_idx[name])
        for name in (
            "water_concrete_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_severe",
            "water_asphalt_slight",
            "dry_concrete_slight",
            "dry_concrete_severe",
            "dry_asphalt_severe",
        )
        if name in class_to_idx
    )
    if bool(shuffle_safety_split):
        tune_idx, safety_idx = stratified_safety_split_random(
            y_val,
            safety_fraction=float(safety_fraction),
            seed=int(seed),
        )
    else:
        tune_idx, safety_idx = stratified_safety_split(y_val, safety_fraction=float(safety_fraction))
    candidates = build_candidates(
        y=y_val[tune_idx],
        anchor_logits=val_a[tune_idx],
        specialist_logits=val_s[tune_idx],
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        weak_classes=weak_classes,
        min_switch_count=int(min_switch_count),
        max_candidates=int(max_candidates),
        rule_family=str(rule_family),
    )

    tune_current = val_a[tune_idx].copy()
    safety_current = val_a[safety_idx].copy()
    tune_specialist = val_s[tune_idx]
    safety_specialist = val_s[safety_idx]
    y_tune = y_val[tune_idx]
    y_safety = y_val[safety_idx]
    tune_metrics = fast_metric_bundle(y_tune, tune_current.argmax(axis=1), idx_to_class)
    safety_metrics = fast_metric_bundle(y_safety, safety_current.argmax(axis=1), idx_to_class)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for _ in range(int(max_rules)):
        best: tuple[float, RouterRule, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, float], dict[str, float]] | None = None
        for rule in candidates:
            if any(same_rule(rule, item["rule_raw"], idx_to_class) for item in accepted):
                continue
            patched_tune = apply_router_rule(tune_current, tune_specialist, rule)
            patched_safety = apply_router_rule(safety_current, safety_specialist, rule)
            new_tune = fast_metric_bundle(y_tune, patched_tune.argmax(axis=1), idx_to_class)
            new_safety = fast_metric_bundle(y_safety, patched_safety.argmax(axis=1), idx_to_class)
            ok_tune, tune_checks = passes_router_constraints(
                base=tune_metrics,
                new=new_tune,
                target=rule.target,
                labels=labels,
                weak_classes=weak_classes,
                min_target_gain=min_target_gain,
                min_macro_gain=min_macro_gain,
                min_top1_gain=min_top1_gain,
                max_protected_drop=max_protected_drop,
                protect_all_classes=protect_all_classes,
            )
            ok_safety, safety_checks = passes_router_constraints(
                base=safety_metrics,
                new=new_safety,
                target=rule.target,
                labels=labels,
                weak_classes=weak_classes,
                min_target_gain=safety_min_target_gain,
                min_macro_gain=safety_min_macro_gain,
                min_top1_gain=safety_min_top1_gain,
                max_protected_drop=max_protected_drop,
                protect_all_classes=protect_all_classes,
            )
            score = (
                float(new_tune["summary"]["macro_f1"]) - float(tune_metrics["summary"]["macro_f1"])
                + float(new_safety["summary"]["macro_f1"]) - float(safety_metrics["summary"]["macro_f1"])
                + 0.5 * (float(new_tune["summary"]["top1"]) - float(tune_metrics["summary"]["top1"]))
                + 0.5 * (float(new_safety["summary"]["top1"]) - float(safety_metrics["summary"]["top1"]))
                + tune_checks["target_f1_gain"]
            )
            record = {
                "rule": rule.name(idx_to_class),
                "rule_raw": rule_to_dict(rule, idx_to_class),
                "score": float(score),
                "tune_checks": tune_checks,
                "safety_checks": safety_checks,
            }
            if not (ok_tune and ok_safety):
                if len(rejected) < 200:
                    rejected.append(record)
                continue
            if float(score) <= 1e-12:
                continue
            if best is None or score > best[0]:
                best = (score, rule, patched_tune, patched_safety, new_tune, new_safety, tune_checks, safety_checks)
        if best is None:
            break
        score, rule, tune_current, safety_current, tune_metrics, safety_metrics, tune_checks, safety_checks = best
        accepted.append(
            {
                "rule": rule.name(idx_to_class),
                "rule_raw": rule_to_dict(rule, idx_to_class),
                "score": float(score),
                "tune_checks": tune_checks,
                "safety_checks": safety_checks,
                "tune_summary_after": tune_metrics["summary"],
                "safety_summary_after": safety_metrics["summary"],
            }
        )

    routed_val = val_a.copy()
    routed_test = test_a.copy()
    for item in accepted:
        rule = dict_to_rule(item["rule_raw"], class_to_idx)
        routed_val = apply_router_rule(routed_val, val_s, rule)
        routed_test = apply_router_rule(routed_test, test_s, rule)

    return {
        "val_anchor": base_val,
        "val_specialist": metric_bundle(y_val, val_s.argmax(axis=1), idx_to_class, class_to_idx),
        "val_routed": metric_bundle(y_val, routed_val.argmax(axis=1), idx_to_class, class_to_idx),
        "val_tune_anchor": metric_bundle(y_tune, val_a[tune_idx].argmax(axis=1), idx_to_class, class_to_idx),
        "val_tune_routed": tune_metrics,
        "val_safety_anchor": metric_bundle(y_safety, val_a[safety_idx].argmax(axis=1), idx_to_class, class_to_idx),
        "val_safety_routed": safety_metrics,
        "test_anchor": metric_bundle(y_test, test_a.argmax(axis=1), idx_to_class, class_to_idx),
        "test_specialist": metric_bundle(y_test, test_s.argmax(axis=1), idx_to_class, class_to_idx),
        "test_routed": metric_bundle(y_test, routed_test.argmax(axis=1), idx_to_class, class_to_idx),
        "accepted_rules": accepted,
        "rejected_rule_examples": rejected,
        "candidate_count": int(len(candidates)),
        "weak_classes": [idx_to_class[i] for i in sorted(weak_classes)],
        "test_true": y_test.astype(int).tolist(),
        "test_anchor_pred": test_a.argmax(axis=1).astype(int).tolist(),
        "test_specialist_pred": test_s.argmax(axis=1).astype(int).tolist(),
        "test_routed_pred": routed_test.argmax(axis=1).astype(int).tolist(),
        "test_image_path": [str(p) for p in test_anchor["image_path"].tolist()],
    }


def build_candidates(
    *,
    y: np.ndarray,
    anchor_logits: np.ndarray,
    specialist_logits: np.ndarray,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    weak_classes: set[int],
    min_switch_count: int,
    max_candidates: int,
    rule_family: str,
) -> list[RouterRule]:
    spec = build_rscd_factor_spec(class_to_idx)
    hard_edges = {tuple(sorted((int(pair.left), int(pair.right)))) for pair in spec.hard_pairs}
    a_pred = anchor_logits.argmax(axis=1)
    s_pred = specialist_logits.argmax(axis=1)
    a_prob = softmax_np(anchor_logits)
    a_conf = a_prob.max(axis=1)
    s_order = np.argsort(-specialist_logits, axis=1)
    s_top = specialist_logits[np.arange(len(specialist_logits)), s_order[:, 0]]
    s_second = specialist_logits[np.arange(len(specialist_logits)), s_order[:, 1]]
    s_margin_any = s_top - s_second

    anchor_margin_grid = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.35]
    pair_specialist_margin_grid = [-0.20, -0.05, 0.00, 0.10, 0.20, 0.35]
    target_specialist_margin_grid = [0.00, 0.10, 0.20, 0.35, 0.50, 0.75]
    anchor_conf_grid = [0.60, 0.70, 0.80, 0.90, 1.01]
    rows: list[RouterRule] = []

    if str(rule_family) in {"pair", "both"}:
        pairs: dict[tuple[int, int], int] = {}
        for true_idx, source, target in zip(y.tolist(), a_pred.tolist(), s_pred.tolist(), strict=True):
            if int(source) == int(target) or int(true_idx) != int(target):
                continue
            if int(target) not in weak_classes:
                continue
            if tuple(sorted((int(source), int(target)))) not in hard_edges:
                continue
            key = (int(source), int(target))
            pairs[key] = pairs.get(key, 0) + 1

        for (source, target), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0])):
            if count < int(min_switch_count):
                continue
            source = int(source)
            target = int(target)
            base_mask = (a_pred == source) & (s_pred == target)
            if int(base_mask.sum()) < int(min_switch_count):
                continue
            anchor_gap = anchor_logits[:, source] - anchor_logits[:, target]
            specialist_gap = specialist_logits[:, target] - specialist_logits[:, source]
            for anchor_margin in anchor_margin_grid:
                for specialist_margin in pair_specialist_margin_grid:
                    for anchor_conf in anchor_conf_grid:
                        mask = (
                            base_mask
                            & (anchor_gap <= float(anchor_margin))
                            & (specialist_gap >= float(specialist_margin))
                            & (a_conf <= float(anchor_conf))
                        )
                        if int(mask.sum()) >= int(min_switch_count):
                            rows.append(
                                RouterRule(
                                    source=source,
                                    target=target,
                                    anchor_margin=float(anchor_margin),
                                    specialist_margin=float(specialist_margin),
                                    anchor_conf=float(anchor_conf),
                                    any_source=False,
                                )
                            )

    if str(rule_family) in {"target", "both"}:
        targets: dict[int, int] = {}
        for true_idx, source, target in zip(y.tolist(), a_pred.tolist(), s_pred.tolist(), strict=True):
            target = int(target)
            source = int(source)
            if source == target or int(true_idx) != target:
                continue
            if target not in weak_classes:
                continue
            if tuple(sorted((source, target))) not in hard_edges:
                continue
            targets[target] = targets.get(target, 0) + 1

        for target, count in sorted(targets.items(), key=lambda item: (-item[1], item[0])):
            if count < int(min_switch_count):
                continue
            target = int(target)
            neighbor_mask = np.asarray(
                [tuple(sorted((int(source), target))) in hard_edges for source in a_pred.tolist()],
                dtype=bool,
            )
            base_mask = (a_pred != target) & (s_pred == target) & neighbor_mask
            if int(base_mask.sum()) < int(min_switch_count):
                continue
            anchor_gap = anchor_logits[np.arange(len(anchor_logits)), a_pred] - anchor_logits[:, target]
            for anchor_margin in anchor_margin_grid:
                for specialist_margin in target_specialist_margin_grid:
                    for anchor_conf in anchor_conf_grid:
                        mask = (
                            base_mask
                            & (anchor_gap <= float(anchor_margin))
                            & (s_margin_any >= float(specialist_margin))
                            & (a_conf <= float(anchor_conf))
                        )
                        if int(mask.sum()) >= int(min_switch_count):
                            rows.append(
                                RouterRule(
                                    source=-1,
                                    target=target,
                                    anchor_margin=float(anchor_margin),
                                    specialist_margin=float(specialist_margin),
                                    anchor_conf=float(anchor_conf),
                                    any_source=True,
                                )
                            )
    return rows[: int(max_candidates)]


def apply_router_rule(anchor_logits: np.ndarray, specialist_logits: np.ndarray, rule: RouterRule) -> np.ndarray:
    out = anchor_logits.copy()
    a_pred = out.argmax(axis=1)
    s_pred = specialist_logits.argmax(axis=1)
    a_prob = softmax_np(out)
    a_conf = a_prob.max(axis=1)
    if bool(rule.any_source):
        s_order = np.argsort(-specialist_logits, axis=1)
        s_top = specialist_logits[np.arange(len(specialist_logits)), s_order[:, 0]]
        s_second = specialist_logits[np.arange(len(specialist_logits)), s_order[:, 1]]
        specialist_gap = s_top - s_second
        anchor_gap = out[np.arange(len(out)), a_pred] - out[:, int(rule.target)]
        mask = (
            (a_pred != int(rule.target))
            & (s_pred == int(rule.target))
            & (anchor_gap <= float(rule.anchor_margin))
            & (specialist_gap >= float(rule.specialist_margin))
            & (a_conf <= float(rule.anchor_conf))
        )
    else:
        anchor_gap = out[:, int(rule.source)] - out[:, int(rule.target)]
        specialist_gap = specialist_logits[:, int(rule.target)] - specialist_logits[:, int(rule.source)]
        mask = (
            (a_pred == int(rule.source))
            & (s_pred == int(rule.target))
            & (anchor_gap <= float(rule.anchor_margin))
            & (specialist_gap >= float(rule.specialist_margin))
            & (a_conf <= float(rule.anchor_conf))
        )
    if bool(mask.any()):
        out[mask] = specialist_logits[mask]
    return out


def passes_router_constraints(
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
    protected = np.ones(len(labels), dtype=bool) if protect_all_classes else np.asarray(
        [i not in weak_classes for i in labels],
        dtype=bool,
    )
    protected[int(target)] = False
    drops = base_f1 - new_f1
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


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def rule_to_dict(rule: RouterRule, idx_to_class: dict[int, str]) -> dict[str, Any]:
    return {
        "source": "__any__" if bool(rule.any_source) else idx_to_class[int(rule.source)],
        "target": idx_to_class[int(rule.target)],
        "anchor_margin": float(rule.anchor_margin),
        "specialist_margin": float(rule.specialist_margin),
        "anchor_conf": float(rule.anchor_conf),
        "any_source": bool(rule.any_source),
    }


def dict_to_rule(payload: dict[str, Any], class_to_idx: dict[str, int]) -> RouterRule:
    any_source = bool(payload.get("any_source", False)) or str(payload.get("source")) == "__any__"
    return RouterRule(
        source=-1 if any_source else int(class_to_idx[canonical_class_label(str(payload["source"]))]),
        target=int(class_to_idx[canonical_class_label(str(payload["target"]))]),
        anchor_margin=float(payload["anchor_margin"]),
        specialist_margin=float(payload["specialist_margin"]),
        anchor_conf=float(payload["anchor_conf"]),
        any_source=any_source,
    )


def same_rule(rule: RouterRule, payload: dict[str, Any], idx_to_class: dict[int, str]) -> bool:
    return (
        ("__any__" if bool(rule.any_source) else idx_to_class[int(rule.source)]) == str(payload["source"])
        and idx_to_class[int(rule.target)] == str(payload["target"])
        and abs(float(rule.anchor_margin) - float(payload["anchor_margin"])) < 1e-9
        and abs(float(rule.specialist_margin) - float(payload["specialist_margin"])) < 1e-9
        and abs(float(rule.anchor_conf) - float(payload["anchor_conf"])) < 1e-9
        and bool(rule.any_source) == (bool(payload.get("any_source", False)) or str(payload.get("source")) == "__any__")
    )


def write_outputs(
    out_dir: Path,
    result: dict[str, Any],
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selective_anchor_router.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    with (out_dir / "accepted_rules.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "score", "tune_top1_gain", "safety_top1_gain", "tune_macro_gain", "safety_macro_gain"])
        for item in result["accepted_rules"]:
            writer.writerow(
                [
                    item["rule"],
                    item["score"],
                    item["tune_checks"]["top1_gain"],
                    item["safety_checks"]["top1_gain"],
                    item["tune_checks"]["macro_f1_gain"],
                    item["safety_checks"]["macro_f1_gain"],
                ]
            )
    with (out_dir / "per_class_test_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "anchor_f1", "specialist_f1", "routed_f1", "delta_vs_anchor", "anchor_recall", "routed_recall"])
        anchor_report = result["test_anchor"]["classification_report"]
        specialist_report = result["test_specialist"]["classification_report"]
        routed_report = result["test_routed"]["classification_report"]
        for name in names:
            writer.writerow(
                [
                    name,
                    float(anchor_report[name]["f1-score"]),
                    float(specialist_report[name]["f1-score"]),
                    float(routed_report[name]["f1-score"]),
                    float(routed_report[name]["f1-score"]) - float(anchor_report[name]["f1-score"]),
                    float(anchor_report[name]["recall"]),
                    float(routed_report[name]["recall"]),
                ]
            )
    with (out_dir / "predictions_test_routed.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "anchor_pred", "specialist_pred", "routed_pred", "changed"])
        for path, true, anchor, specialist, routed in zip(
            result["test_image_path"],
            result["test_true"],
            result["test_anchor_pred"],
            result["test_specialist_pred"],
            result["test_routed_pred"],
            strict=True,
        ):
            writer.writerow(
                [
                    path,
                    idx_to_class[int(true)],
                    idx_to_class[int(anchor)],
                    idx_to_class[int(specialist)],
                    idx_to_class[int(routed)],
                    bool(int(anchor) != int(routed)),
                ]
            )
    metrics_like = {
        "classification_report": result["test_routed"]["classification_report"],
        "factor_confusion_summary": result["test_routed"]["factor_confusion_summary"],
        "summary": result["test_routed"]["summary"],
        "y_true": [int(x) for x in result["test_true"]],
        "y_pred": [int(x) for x in result["test_routed_pred"]],
    }
    write_hard_pair_metrics(out_dir / "hard_pair_metrics_routed.csv", metrics_like, idx_to_class)
    write_wcs_diagnosis(out_dir / "water_concrete_slight_diagnosis_routed.json", metrics_like, idx_to_class)
    factor_summary = factor_confusion_summary(
        [int(x) for x in result["test_true"]],
        [int(x) for x in result["test_routed_pred"]],
        build_rscd_factor_spec(class_to_idx),
        idx_to_class,
    )
    (out_dir / "factor_confusion_summary_routed.json").write_text(
        json.dumps(factor_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
