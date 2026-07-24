"""Validation-constrained router for a base classifier plus one specialist.

The router is intentionally post-hoc and conservative: candidate switch rules
are selected only on validation predictions, then applied once to test
predictions.  It is useful for checking whether an expert has a reliable
competence region without retraining the backbone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    mask_fn: Callable[[pd.DataFrame], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-base", type=Path, required=True)
    parser.add_argument("--val-expert", type=Path, required=True)
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--test-expert", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-val-correct-gain", type=int, default=1)
    parser.add_argument("--min-macro-gain-pp", type=float, default=0.01)
    parser.add_argument("--max-class-recall-drop-pp", type=float, default=0.0)
    parser.add_argument("--max-rules", type=int, default=30)
    return parser.parse_args()


def read_pair(base_path: Path, expert_path: Path) -> pd.DataFrame:
    base = pd.read_csv(base_path)
    expert = pd.read_csv(expert_path)
    required = {"image_path", "true_label", "pred_label", "confidence"}
    missing_base = required.difference(base.columns)
    missing_expert = required.difference(expert.columns)
    if missing_base or missing_expert:
        raise ValueError(f"Missing columns: base={missing_base}, expert={missing_expert}")
    if len(base) != len(expert):
        raise ValueError(f"Length mismatch: {len(base)} vs {len(expert)}")
    if not base["image_path"].equals(expert["image_path"]):
        merged = base.merge(
            expert,
            on=["image_path", "true_label"],
            suffixes=("_base", "_expert"),
            how="inner",
        )
    else:
        merged = pd.DataFrame(
            {
                "image_path": base["image_path"],
                "true_label": base["true_label"],
                "base_pred": base["pred_label"],
                "base_conf": base["confidence"].astype(float),
                "expert_pred": expert["pred_label"],
                "expert_conf": expert["confidence"].astype(float),
            }
        )
    if len(merged) != len(base):
        raise ValueError(f"Aligned rows changed after merge: {len(base)} -> {len(merged)}")
    merged["dconf"] = merged["expert_conf"] - merged["base_conf"]
    merged["base_correct"] = merged["base_pred"].eq(merged["true_label"])
    merged["expert_correct"] = merged["expert_pred"].eq(merged["true_label"])
    return merged


def label_parts(label: str) -> tuple[str, str, str]:
    parts = label.split("_")
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    return label, "", ""


def add_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("base", "expert"):
        parts = out[f"{prefix}_pred"].map(label_parts)
        out[f"{prefix}_friction"] = parts.map(lambda x: x[0])
        out[f"{prefix}_material"] = parts.map(lambda x: x[1])
        out[f"{prefix}_roughness"] = parts.map(lambda x: x[2])
    true_parts = out["true_label"].map(label_parts)
    out["true_friction"] = true_parts.map(lambda x: x[0])
    out["true_material"] = true_parts.map(lambda x: x[1])
    out["true_roughness"] = true_parts.map(lambda x: x[2])
    return out


def macro_f1(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    vals = []
    true = y_true.to_numpy()
    pred = y_pred.to_numpy()
    for label in labels:
        tp = np.sum((true == label) & (pred == label))
        fp = np.sum((true != label) & (pred == label))
        fn = np.sum((true == label) & (pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def per_class_recall(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    true = y_true.to_numpy()
    pred = y_pred.to_numpy()
    out = {}
    for label in labels:
        idx = true == label
        out[label] = float(np.mean(pred[idx] == true[idx])) if np.any(idx) else 0.0
    return out


def metrics(df: pd.DataFrame, pred_col: str, labels: list[str]) -> dict[str, object]:
    pred = df[pred_col]
    return {
        "n": int(len(df)),
        "top1": float(np.mean(pred.eq(df["true_label"]))),
        "macro_f1": macro_f1(df["true_label"], pred, labels),
        "per_class_recall": per_class_recall(df["true_label"], pred, labels),
    }


def build_rules(val: pd.DataFrame, min_support: int) -> list[Rule]:
    labels = sorted(val["true_label"].unique())
    dconf_thresholds = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    base_conf_highs = [0.50, 0.60, 0.70, 0.80, 0.90]
    expert_conf_lows = [0.40, 0.50, 0.60, 0.70, 0.80]
    rules: list[Rule] = []

    def add(name: str, fn: Callable[[pd.DataFrame], np.ndarray]) -> None:
        if int(fn(val).sum()) >= min_support:
            rules.append(Rule(name, fn))

    for tau in dconf_thresholds:
        add(
            f"global_dconf_ge_{tau:+.2f}",
            lambda df, tau=tau: df["dconf"].to_numpy() >= tau,
        )
        add(
            f"disagree_dconf_ge_{tau:+.2f}",
            lambda df, tau=tau: (df["base_pred"].to_numpy() != df["expert_pred"].to_numpy())
            & (df["dconf"].to_numpy() >= tau),
        )

    for bp in sorted(val["base_pred"].unique()):
        for tau in dconf_thresholds:
            add(
                f"base={bp}|dconf_ge_{tau:+.2f}",
                lambda df, bp=bp, tau=tau: (df["base_pred"].to_numpy() == bp)
                & (df["dconf"].to_numpy() >= tau),
            )
    for ep in sorted(val["expert_pred"].unique()):
        for tau in dconf_thresholds:
            add(
                f"expert={ep}|dconf_ge_{tau:+.2f}",
                lambda df, ep=ep, tau=tau: (df["expert_pred"].to_numpy() == ep)
                & (df["dconf"].to_numpy() >= tau),
            )

    for field in ("friction", "material", "roughness"):
        for value in sorted(set(val[f"base_{field}"].unique()) | set(val[f"expert_{field}"].unique())):
            for tau in dconf_thresholds:
                add(
                    f"base_{field}={value}|dconf_ge_{tau:+.2f}",
                    lambda df, field=field, value=value, tau=tau: (
                        df[f"base_{field}"].to_numpy() == value
                    )
                    & (df["dconf"].to_numpy() >= tau),
                )
                add(
                    f"expert_{field}={value}|dconf_ge_{tau:+.2f}",
                    lambda df, field=field, value=value, tau=tau: (
                        df[f"expert_{field}"].to_numpy() == value
                    )
                    & (df["dconf"].to_numpy() >= tau),
                )

    for high in base_conf_highs:
        for low in expert_conf_lows:
            add(
                f"lowbase_le_{high:.2f}|experthigh_ge_{low:.2f}",
                lambda df, high=high, low=low: (df["base_conf"].to_numpy() <= high)
                & (df["expert_conf"].to_numpy() >= low),
            )

    # Pair-specific rules capture local confusion edges without using true labels.
    pairs = val.groupby(["base_pred", "expert_pred"]).size()
    for bp, ep in pairs[pairs >= min_support].index:
        for tau in dconf_thresholds:
            add(
                f"edge={bp}->{ep}|dconf_ge_{tau:+.2f}",
                lambda df, bp=bp, ep=ep, tau=tau: (df["base_pred"].to_numpy() == bp)
                & (df["expert_pred"].to_numpy() == ep)
                & (df["dconf"].to_numpy() >= tau),
            )

    # De-duplicate by name while keeping order.
    seen: set[str] = set()
    uniq = []
    for rule in rules:
        if rule.name not in seen:
            seen.add(rule.name)
            uniq.append(rule)
    return uniq


def apply_mask(df: pd.DataFrame, switch: np.ndarray) -> pd.Series:
    return pd.Series(np.where(switch, df["expert_pred"], df["base_pred"]), index=df.index)


def evaluate_rule(
    df: pd.DataFrame,
    rule: Rule,
    current_switch: np.ndarray,
    labels: list[str],
    base_recall: dict[str, float],
    min_val_correct_gain: int,
    min_macro_gain: float,
    max_class_drop: float,
) -> dict[str, object] | None:
    mask = rule.mask_fn(df) & ~current_switch
    support = int(mask.sum())
    if support == 0:
        return None
    gain = int((df.loc[mask, "expert_correct"].astype(int) - df.loc[mask, "base_correct"].astype(int)).sum())
    if gain < min_val_correct_gain:
        return None

    current_pred = apply_mask(df, current_switch)
    new_switch = current_switch | mask
    new_pred = apply_mask(df, new_switch)
    current_macro = macro_f1(df["true_label"], current_pred, labels)
    new_macro = macro_f1(df["true_label"], new_pred, labels)
    if (new_macro - current_macro) < min_macro_gain:
        return None

    recalls = per_class_recall(df["true_label"], new_pred, labels)
    drops = {label: recalls[label] - base_recall[label] for label in labels}
    worst_drop = min(drops.values())
    if worst_drop < -max_class_drop:
        return None

    return {
        "name": rule.name,
        "support": support,
        "correct_gain": gain,
        "macro_gain": float(new_macro - current_macro),
        "worst_class_recall_delta": float(worst_drop),
        "switch": new_switch,
    }


def greedy_select(
    val: pd.DataFrame,
    rules: list[Rule],
    labels: list[str],
    min_val_correct_gain: int,
    min_macro_gain: float,
    max_class_drop: float,
    max_rules: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    current_switch = np.zeros(len(val), dtype=bool)
    base_recall = per_class_recall(val["true_label"], val["base_pred"], labels)
    selected: list[dict[str, object]] = []

    for _ in range(max_rules):
        candidates = []
        for rule in rules:
            result = evaluate_rule(
                val,
                rule,
                current_switch,
                labels,
                base_recall,
                min_val_correct_gain,
                min_macro_gain,
                max_class_drop,
            )
            if result is not None:
                candidates.append(result)
        if not candidates:
            break
        candidates.sort(
            key=lambda x: (
                float(x["macro_gain"]),
                int(x["correct_gain"]),
                -int(x["support"]),
                str(x["name"]),
            ),
            reverse=True,
        )
        best = candidates[0]
        current_switch = best.pop("switch")  # type: ignore[assignment]
        selected.append(best)
    return selected, current_switch


def apply_selected_rules(df: pd.DataFrame, rules_by_name: dict[str, Rule], selected: list[dict[str, object]]) -> np.ndarray:
    switch = np.zeros(len(df), dtype=bool)
    for item in selected:
        rule = rules_by_name[str(item["name"])]
        switch = switch | (rule.mask_fn(df) & ~switch)
    return switch


def delta_summary(base: dict[str, object], routed: dict[str, object]) -> dict[str, float]:
    base_rec = base["per_class_recall"]
    routed_rec = routed["per_class_recall"]
    assert isinstance(base_rec, dict) and isinstance(routed_rec, dict)
    return {label: float(routed_rec[label] - base_rec[label]) for label in sorted(base_rec)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val = add_factor_columns(read_pair(args.val_base, args.val_expert))
    test = add_factor_columns(read_pair(args.test_base, args.test_expert))
    labels = sorted(val["true_label"].unique())
    if labels != sorted(test["true_label"].unique()):
        raise ValueError("Validation/test labels do not match")

    val["routed_pred"] = val["base_pred"]
    test["routed_pred"] = test["base_pred"]
    rules = build_rules(val, args.min_support)
    selected, val_switch = greedy_select(
        val,
        rules,
        labels,
        args.min_val_correct_gain,
        args.min_macro_gain_pp / 100.0,
        args.max_class_recall_drop_pp / 100.0,
        args.max_rules,
    )
    rules_by_name = {rule.name: rule for rule in rules}
    test_switch = apply_selected_rules(test, rules_by_name, selected)

    val["routed_pred"] = apply_mask(val, val_switch)
    test["routed_pred"] = apply_mask(test, test_switch)

    payload = {
        "paths": {
            "val_base": str(args.val_base),
            "val_expert": str(args.val_expert),
            "test_base": str(args.test_base),
            "test_expert": str(args.test_expert),
        },
        "settings": {
            "min_support": args.min_support,
            "min_val_correct_gain": args.min_val_correct_gain,
            "min_macro_gain_pp": args.min_macro_gain_pp,
            "max_class_recall_drop_pp": args.max_class_recall_drop_pp,
            "max_rules": args.max_rules,
            "candidate_rule_count": len(rules),
        },
        "selected_rules": selected,
        "val": {
            "base": metrics(val, "base_pred", labels),
            "expert": metrics(val, "expert_pred", labels),
            "routed": metrics(val, "routed_pred", labels),
            "switch_rate": float(np.mean(val_switch)),
            "per_class_recall_delta_vs_base": delta_summary(
                metrics(val, "base_pred", labels), metrics(val, "routed_pred", labels)
            ),
        },
        "test": {
            "base": metrics(test, "base_pred", labels),
            "expert": metrics(test, "expert_pred", labels),
            "routed": metrics(test, "routed_pred", labels),
            "switch_rate": float(np.mean(test_switch)),
            "per_class_recall_delta_vs_base": delta_summary(
                metrics(test, "base_pred", labels), metrics(test, "routed_pred", labels)
            ),
        },
    }
    out_json = args.output_dir / "validation_constrained_router.json"
    out_csv = args.output_dir / "test_routed_predictions.csv"
    out_md = args.output_dir / "validation_constrained_router.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    test[["image_path", "true_label", "base_pred", "expert_pred", "routed_pred", "base_conf", "expert_conf", "dconf"]].to_csv(
        out_csv, index=False, encoding="utf-8"
    )

    def pct(x: float) -> str:
        return f"{100.0 * x:.3f}%"

    val_base = payload["val"]["base"]
    val_expert = payload["val"]["expert"]
    val_routed = payload["val"]["routed"]
    test_base = payload["test"]["base"]
    test_expert = payload["test"]["expert"]
    test_routed = payload["test"]["routed"]
    lines = [
        "# Validation-constrained expert router",
        "",
        "Rules are selected only on validation predictions and then applied once to test predictions.",
        "",
        "| split/model | top-1 | macro-F1 | switch rate |",
        "|---|---:|---:|---:|",
        f"| val/base | {pct(float(val_base['top1']))} | {pct(float(val_base['macro_f1']))} | 0.000% |",
        f"| val/expert | {pct(float(val_expert['top1']))} | {pct(float(val_expert['macro_f1']))} | 100.000% |",
        f"| val/routed | {pct(float(val_routed['top1']))} | {pct(float(val_routed['macro_f1']))} | {pct(float(payload['val']['switch_rate']))} |",
        f"| test/base | {pct(float(test_base['top1']))} | {pct(float(test_base['macro_f1']))} | 0.000% |",
        f"| test/expert | {pct(float(test_expert['top1']))} | {pct(float(test_expert['macro_f1']))} | 100.000% |",
        f"| test/routed | {pct(float(test_routed['top1']))} | {pct(float(test_routed['macro_f1']))} | {pct(float(payload['test']['switch_rate']))} |",
        "",
        f"Selected rules: {len(selected)} / {len(rules)} candidates.",
    ]
    if selected:
        lines.extend(["", "## Selected Rules", ""])
        for item in selected:
            lines.append(
                f"- `{item['name']}`: support={item['support']}, "
                f"correct_gain={item['correct_gain']}, "
                f"val_macro_gain={100.0 * float(item['macro_gain']):.3f} pp, "
                f"worst_val_class_recall_delta={100.0 * float(item['worst_class_recall_delta']):.3f} pp"
            )
    else:
        lines.extend(["", "No validation-safe rule was selected under the requested constraints."])

    test_delta = payload["test"]["per_class_recall_delta_vs_base"]
    assert isinstance(test_delta, dict)
    worst = sorted(test_delta.items(), key=lambda kv: kv[1])[:8]
    best = sorted(test_delta.items(), key=lambda kv: kv[1], reverse=True)[:8]
    lines.extend(["", "## Test Per-class Recall Delta vs Base", "", "Worst:"])
    for label, delta in worst:
        lines.append(f"- {label}: {100.0 * float(delta):+.3f} pp")
    lines.append("")
    lines.append("Best:")
    for label, delta in best:
        lines.append(f"- {label}: {100.0 * float(delta):+.3f} pp")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
