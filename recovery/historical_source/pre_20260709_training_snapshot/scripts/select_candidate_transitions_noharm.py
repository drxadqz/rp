from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class TransitionRule:
    source: str
    target: str
    min_candidate_conf: float
    max_anchor_conf: float

    def key(self) -> str:
        return (
            f"{self.source}->{self.target}"
            f"|cand>={self.min_candidate_conf:.2f}|anchor<={self.max_anchor_conf:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select candidate-model prediction transitions only when validation "
            "tune/safety splits prove no protected class-level F1 regression."
        )
    )
    parser.add_argument("--anchor-val", type=Path, required=True)
    parser.add_argument("--candidate-val", type=Path, required=True)
    parser.add_argument("--anchor-test", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safety-fraction", type=float, default=0.5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=16)
    parser.add_argument("--min-target-f1-gain-pp", type=float, default=0.0)
    parser.add_argument("--min-macro-gain-pp", type=float, default=0.0)
    parser.add_argument("--min-top1-gain-pp", type=float, default=0.0)
    parser.add_argument("--max-protected-f1-drop-pp", type=float, default=0.02)
    parser.add_argument("--protect-all-classes", action="store_true")
    parser.add_argument("--focus-class", action="append", default=[])
    parser.add_argument("--no-default-focus", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    focus_classes = {str(x) for x in args.focus_class}
    if not bool(args.no_default_focus):
        focus_classes.update(
            {
                "water_concrete_slight",
                "water_concrete_severe",
                "wet_concrete_slight",
                "wet_concrete_severe",
                "water_asphalt_slight",
                "water_asphalt_severe",
                "dry_concrete_slight",
                "dry_gravel",
            }
        )

    val = load_aligned(args.anchor_val, args.candidate_val)
    test = load_aligned(args.anchor_test, args.candidate_test)
    class_names = sorted(set(val["true_label"]).union(test["true_label"]).union(val["anchor_pred"]).union(val["candidate_pred"]))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    y_val = encode(val["true_label"], class_to_idx)
    anchor_val_pred = encode(val["anchor_pred"], class_to_idx)
    cand_val_pred = encode(val["candidate_pred"], class_to_idx)
    y_test = encode(test["true_label"], class_to_idx)
    anchor_test_pred = encode(test["anchor_pred"], class_to_idx)
    cand_test_pred = encode(test["candidate_pred"], class_to_idx)

    tune_idx, safety_idx = stratified_safety_split(y_val, safety_fraction=float(args.safety_fraction))
    accepted, rejected, val_selected_pred = greedy_select_rules(
        frame=val,
        y=y_val,
        anchor_pred=anchor_val_pred,
        candidate_pred=cand_val_pred,
        tune_idx=tune_idx,
        safety_idx=safety_idx,
        class_names=class_names,
        focus_classes=focus_classes,
        min_support=int(args.min_support),
        max_rules=int(args.max_rules),
        min_target_gain=float(args.min_target_f1_gain_pp) / 100.0,
        min_macro_gain=float(args.min_macro_gain_pp) / 100.0,
        min_top1_gain=float(args.min_top1_gain_pp) / 100.0,
        max_protected_drop=float(args.max_protected_f1_drop_pp) / 100.0,
        protect_all_classes=bool(args.protect_all_classes),
    )
    test_selected_pred = apply_rules_to_split(
        test,
        anchor_test_pred.copy(),
        cand_test_pred,
        accepted,
        class_to_idx,
    )

    result = {
        "protocol": {
            "method": "validation-selected candidate transition no-harm filter",
            "claim_boundary": (
                "Diagnostic only. It uses both anchor and candidate predictions "
                "to identify candidate transitions that can be distilled later "
                "into a single-model gate."
            ),
            "anchor_val": str(args.anchor_val),
            "candidate_val": str(args.candidate_val),
            "anchor_test": str(args.anchor_test),
            "candidate_test": str(args.candidate_test),
            "safety_fraction": float(args.safety_fraction),
            "min_support": int(args.min_support),
            "max_rules": int(args.max_rules),
            "min_target_f1_gain_pp": float(args.min_target_f1_gain_pp),
            "min_macro_gain_pp": float(args.min_macro_gain_pp),
            "min_top1_gain_pp": float(args.min_top1_gain_pp),
            "max_protected_f1_drop_pp": float(args.max_protected_f1_drop_pp),
            "protect_all_classes": bool(args.protect_all_classes),
            "focus_classes": sorted(focus_classes),
        },
        "val_anchor": metric_bundle(y_val, anchor_val_pred, class_names),
        "val_candidate": metric_bundle(y_val, cand_val_pred, class_names),
        "val_selected": metric_bundle(y_val, val_selected_pred, class_names),
        "test_anchor": metric_bundle(y_test, anchor_test_pred, class_names),
        "test_candidate": metric_bundle(y_test, cand_test_pred, class_names),
        "test_selected": metric_bundle(y_test, test_selected_pred, class_names),
        "accepted_rules": [rule_record(rule, class_to_idx) for rule in accepted],
        "rejected_rule_examples": rejected[:200],
        "validation_split": {"tune": int(len(tune_idx)), "safety": int(len(safety_idx))},
    }
    (args.output_dir / "transition_noharm_selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_per_class_comparison(args.output_dir, y_test, anchor_test_pred, cand_test_pred, test_selected_pred, class_names)
    write_predictions(args.output_dir / "predictions_test_selected.csv", test, test_selected_pred, class_names)
    print(
        json.dumps(
            {
                "val_anchor": result["val_anchor"]["summary"],
                "val_candidate": result["val_candidate"]["summary"],
                "val_selected": result["val_selected"]["summary"],
                "test_anchor": result["test_anchor"]["summary"],
                "test_candidate": result["test_candidate"]["summary"],
                "test_selected": result["test_selected"]["summary"],
                "accepted_rules": len(result["accepted_rules"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def load_aligned(anchor_path: Path, candidate_path: Path) -> pd.DataFrame:
    anchor = pd.read_csv(anchor_path, dtype=str)
    candidate = pd.read_csv(candidate_path, dtype=str)
    required = {"image_path", "true_label", "pred_label", "confidence"}
    if not required.issubset(anchor.columns) or not required.issubset(candidate.columns):
        raise ValueError(f"prediction CSVs must contain {sorted(required)}")
    merged = anchor.merge(
        candidate,
        on=["image_path", "true_label"],
        suffixes=("_anchor", "_candidate"),
        how="inner",
    )
    if len(merged) != len(anchor) or len(merged) != len(candidate):
        raise ValueError(
            f"prediction CSVs are not perfectly aligned: anchor={len(anchor)} "
            f"candidate={len(candidate)} merged={len(merged)}"
        )
    out = pd.DataFrame(
        {
            "image_path": merged["image_path"].astype(str),
            "true_label": merged["true_label"].astype(str),
            "anchor_pred": merged["pred_label_anchor"].astype(str),
            "candidate_pred": merged["pred_label_candidate"].astype(str),
            "anchor_conf": merged["confidence_anchor"].astype(float),
            "candidate_conf": merged["confidence_candidate"].astype(float),
        }
    )
    return out


def encode(values: pd.Series | list[str], class_to_idx: dict[str, int]) -> np.ndarray:
    return np.asarray([class_to_idx[str(v)] for v in values], dtype=np.int64)


def greedy_select_rules(
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    anchor_pred: np.ndarray,
    candidate_pred: np.ndarray,
    tune_idx: np.ndarray,
    safety_idx: np.ndarray,
    class_names: list[str],
    focus_classes: set[str],
    min_support: int,
    max_rules: int,
    min_target_gain: float,
    min_macro_gain: float,
    min_top1_gain: float,
    max_protected_drop: float,
    protect_all_classes: bool,
) -> tuple[list[TransitionRule], list[dict[str, Any]], np.ndarray]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    focus_idx = {class_to_idx[name] for name in focus_classes if name in class_to_idx}
    rules = build_transition_rules(frame, min_support=min_support)
    current = anchor_pred.copy()
    accepted: list[TransitionRule] = []
    rejected: list[dict[str, Any]] = []
    for _ in range(max_rules):
        best: tuple[float, TransitionRule, np.ndarray, dict[str, float], dict[str, float]] | None = None
        for rule in rules:
            if rule in accepted:
                continue
            proposed = apply_rules_to_split(frame, current.copy(), candidate_pred, [rule], class_to_idx)
            tune_base = fast_metric_bundle(y[tune_idx], current[tune_idx], len(class_names))
            tune_new = fast_metric_bundle(y[tune_idx], proposed[tune_idx], len(class_names))
            safety_base = fast_metric_bundle(y[safety_idx], current[safety_idx], len(class_names))
            safety_new = fast_metric_bundle(y[safety_idx], proposed[safety_idx], len(class_names))
            target_idx = class_to_idx[rule.target]
            ok_tune, tune_checks = passes_constraints(
                base=tune_base,
                new=tune_new,
                target=target_idx,
                focus_idx=focus_idx,
                min_target_gain=min_target_gain,
                min_macro_gain=min_macro_gain,
                min_top1_gain=min_top1_gain,
                max_protected_drop=max_protected_drop,
                protect_all_classes=protect_all_classes,
            )
            ok_safety, safety_checks = passes_constraints(
                base=safety_base,
                new=safety_new,
                target=target_idx,
                focus_idx=focus_idx,
                min_target_gain=0.0,
                min_macro_gain=0.0,
                min_top1_gain=0.0,
                max_protected_drop=max_protected_drop,
                protect_all_classes=protect_all_classes,
            )
            score = (
                tune_checks["macro_f1_gain"]
                + safety_checks["macro_f1_gain"]
                + 0.5 * tune_checks["top1_gain"]
                + 0.5 * safety_checks["top1_gain"]
                + tune_checks["target_f1_gain"]
            )
            record = {
                "rule": rule.key(),
                "support": int(rule_support(frame, rule)),
                "tune_checks": tune_checks,
                "safety_checks": safety_checks,
                "score": float(score),
            }
            if not (ok_tune and ok_safety):
                if len(rejected) < 200:
                    rejected.append(record)
                continue
            if best is None or score > best[0]:
                best = (float(score), rule, proposed, tune_checks, safety_checks)
        if best is None:
            break
        _, rule, current, _, _ = best
        accepted.append(rule)
    return accepted, rejected, current


def build_transition_rules(frame: pd.DataFrame, *, min_support: int) -> list[TransitionRule]:
    changed = frame[frame["anchor_pred"] != frame["candidate_pred"]]
    conf_thresholds = [0.0, 0.55, 0.65, 0.75, 0.85, 0.92]
    anchor_max_thresholds = [1.0, 0.95, 0.85, 0.75]
    rules: dict[tuple[str, str, float, float], TransitionRule] = {}
    for (source, target), group in changed.groupby(["anchor_pred", "candidate_pred"], sort=True):
        for min_conf in conf_thresholds:
            for max_anchor in anchor_max_thresholds:
                support = int(((group["candidate_conf"] >= min_conf) & (group["anchor_conf"] <= max_anchor)).sum())
                if support >= min_support:
                    rules[(source, target, float(min_conf), float(max_anchor))] = TransitionRule(
                        source=str(source),
                        target=str(target),
                        min_candidate_conf=float(min_conf),
                        max_anchor_conf=float(max_anchor),
                    )
    return list(rules.values())


def rule_support(frame: pd.DataFrame, rule: TransitionRule) -> int:
    mask = rule_mask(frame, rule)
    return int(mask.sum())


def rule_mask(frame: pd.DataFrame, rule: TransitionRule) -> np.ndarray:
    return (
        (frame["anchor_pred"].to_numpy(dtype=str) == rule.source)
        & (frame["candidate_pred"].to_numpy(dtype=str) == rule.target)
        & (frame["candidate_conf"].to_numpy(dtype=float) >= float(rule.min_candidate_conf))
        & (frame["anchor_conf"].to_numpy(dtype=float) <= float(rule.max_anchor_conf))
    )


def apply_rules_to_split(
    frame: pd.DataFrame,
    pred: np.ndarray,
    candidate_pred: np.ndarray,
    rules: list[TransitionRule],
    class_to_idx: dict[str, int],
) -> np.ndarray:
    out = pred.copy()
    for rule in rules:
        mask = rule_mask(frame, rule)
        out[mask] = candidate_pred[mask]
    return out


def passes_constraints(
    *,
    base: dict[str, Any],
    new: dict[str, Any],
    target: int,
    focus_idx: set[int],
    min_target_gain: float,
    min_macro_gain: float,
    min_top1_gain: float,
    max_protected_drop: float,
    protect_all_classes: bool,
) -> tuple[bool, dict[str, float]]:
    base_f1 = np.asarray(base["per_class_f1"], dtype=np.float64)
    new_f1 = np.asarray(new["per_class_f1"], dtype=np.float64)
    drops = base_f1 - new_f1
    if protect_all_classes:
        protected = np.ones_like(drops, dtype=bool)
    else:
        protected = np.ones_like(drops, dtype=bool)
        for idx in focus_idx:
            protected[int(idx)] = False
    protected[int(target)] = False
    checks = {
        "top1_gain": float(new["summary"]["top1"] - base["summary"]["top1"]),
        "macro_f1_gain": float(new["summary"]["macro_f1"] - base["summary"]["macro_f1"]),
        "target_f1_gain": float(new_f1[int(target)] - base_f1[int(target)]),
        "max_protected_f1_drop": float(drops[protected].max()) if bool(protected.any()) else 0.0,
    }
    ok = (
        checks["top1_gain"] + 1e-12 >= min_top1_gain
        and checks["macro_f1_gain"] + 1e-12 >= min_macro_gain
        and checks["target_f1_gain"] + 1e-12 >= min_target_gain
        and checks["max_protected_f1_drop"] <= max_protected_drop + 1e-12
    )
    return bool(ok), checks


def stratified_safety_split(y: np.ndarray, *, safety_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    safety_fraction = float(np.clip(safety_fraction, 0.0, 0.9))
    tune: list[int] = []
    safety: list[int] = []
    for label in sorted(set(y.astype(int).tolist())):
        idx = np.flatnonzero(y == int(label))
        n_safety = int(round(len(idx) * safety_fraction))
        if len(idx) > 1:
            n_safety = min(max(n_safety, 1), len(idx) - 1)
        else:
            n_safety = 0
        safety.extend(idx[:n_safety].astype(int).tolist())
        tune.extend(idx[n_safety:].astype(int).tolist())
    return np.asarray(tune, dtype=np.int64), np.asarray(safety, dtype=np.int64)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    per_class_f1 = [float(report[name]["f1-score"]) for name in class_names]
    summary = {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "num_samples": int(len(y_true)),
        "num_errors": int(np.sum(y_true != y_pred)),
        "weakest_class": class_names[int(np.argmin(per_class_f1))],
        "weakest_f1": float(np.min(per_class_f1)),
        "water_concrete_slight_f1": float(report.get("water_concrete_slight", {}).get("f1-score", 0.0)),
    }
    return {"summary": summary, "classification_report": report, "per_class_f1": per_class_f1}


def fast_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    per_class_f1 = []
    for idx in range(num_classes):
        true_pos = y_true == idx
        pred_pos = y_pred == idx
        tp = float(np.sum(true_pos & pred_pos))
        fp = float(np.sum(~true_pos & pred_pos))
        fn = float(np.sum(true_pos & ~pred_pos))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        per_class_f1.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    arr = np.asarray(per_class_f1, dtype=np.float64)
    return {
        "summary": {
            "top1": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
            "macro_f1": float(arr.mean()) if len(arr) else 0.0,
            "num_errors": int(np.sum(y_true != y_pred)),
        },
        "per_class_f1": per_class_f1,
    }


def rule_record(rule: TransitionRule, class_to_idx: dict[str, int]) -> dict[str, Any]:
    return {
        "rule": rule.key(),
        "source": rule.source,
        "target": rule.target,
        "source_idx": int(class_to_idx[rule.source]),
        "target_idx": int(class_to_idx[rule.target]),
        "min_candidate_conf": float(rule.min_candidate_conf),
        "max_anchor_conf": float(rule.max_anchor_conf),
    }


def write_per_class_comparison(
    out_dir: Path,
    y_true: np.ndarray,
    anchor_pred: np.ndarray,
    cand_pred: np.ndarray,
    selected_pred: np.ndarray,
    class_names: list[str],
) -> None:
    anchor = metric_bundle(y_true, anchor_pred, class_names)
    cand = metric_bundle(y_true, cand_pred, class_names)
    selected = metric_bundle(y_true, selected_pred, class_names)
    with (out_dir / "per_class_transition_selection.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "anchor_f1", "candidate_f1", "selected_f1", "cand_delta", "selected_delta"])
        for idx, name in enumerate(class_names):
            a = float(anchor["per_class_f1"][idx])
            c = float(cand["per_class_f1"][idx])
            s = float(selected["per_class_f1"][idx])
            w.writerow([name, a, c, s, c - a, s - a])


def write_predictions(path: Path, frame: pd.DataFrame, selected_pred: np.ndarray, class_names: list[str]) -> None:
    out = frame.copy()
    out["selected_pred"] = [class_names[int(i)] for i in selected_pred.tolist()]
    out.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
