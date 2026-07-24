from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


KEY_CLASSES = {
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_severe",
}
FULL_TRAIN_SAMPLES = 958_941
FULL_VAL_SAMPLES = 19_860


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def is_empty(value: Any) -> bool:
    return value is None or value == ""


def count_manifest_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def full_training_protocol_evidence(run_dir: Path, full_test_samples: int) -> dict[str, Any]:
    config_path = run_dir / "config_resolved.yaml"
    cfg = read_yaml(config_path)
    if cfg is None:
        return {
            "passed": False,
            "config_path": str(config_path),
            "reason": "config_resolved.yaml missing or unreadable",
            "checks": {"config_exists": False},
        }
    data = cfg.get("data") or {}
    train = cfg.get("train") or {}
    eval_cfg = cfg.get("eval") or {}
    manifests = {
        "train": Path(str(data.get("train_manifest") or "")),
        "val": Path(str(data.get("val_manifest") or "")),
        "test": Path(str(data.get("test_manifest") or "")),
    }
    counts = {split: count_manifest_rows(path) for split, path in manifests.items()}
    checks = {
        "config_exists": True,
        "train_manifest_full": counts["train"] == FULL_TRAIN_SAMPLES,
        "val_manifest_full": counts["val"] == FULL_VAL_SAMPLES,
        "test_manifest_full": counts["test"] == int(full_test_samples),
        "train_samples_per_epoch_uncapped": int(train.get("samples_per_epoch") or 0) == 0,
        "train_max_samples_per_class_uncapped": is_empty(train.get("max_train_samples_per_class")),
        "train_max_samples_uncapped": is_empty(train.get("max_train_samples")),
        "eval_val_uncapped": is_empty(eval_cfg.get("max_val_samples_per_class")),
        "eval_test_uncapped": is_empty(eval_cfg.get("max_test_samples_per_class")),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "config_path": str(config_path),
        "manifest_paths": {split: str(path) for split, path in manifests.items()},
        "manifest_counts": counts,
        "checks": checks,
        "train_protocol": {
            "samples_per_epoch": train.get("samples_per_epoch"),
            "max_train_samples_per_class": train.get("max_train_samples_per_class"),
            "max_train_samples": train.get("max_train_samples"),
        },
        "eval_protocol": {
            "max_val_samples_per_class": eval_cfg.get("max_val_samples_per_class"),
            "max_test_samples_per_class": eval_cfg.get("max_test_samples_per_class"),
        },
    }


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    payload = read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    return dict(payload.get("summary", payload))


def read_per_class(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "per_class_metrics.csv"
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(name).lstrip("\ufeff") for name in (reader.fieldnames or [])]
        class_field = "class" if "class" in fields else (fields[0] if fields else "class")
        for row in reader:
            if class_field not in row and f"\ufeff{class_field}" in row:
                class_field = f"\ufeff{class_field}"
            name = str(row.get(class_field, ""))
            if not name:
                continue
            rows[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return rows


def read_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "predictions_test.csv"
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row["image_path"])] = {
                "true_label": str(row["true_label"]),
                "pred_label": str(row["pred_label"]),
                "confidence": float(row.get("confidence") or 0.0),
            }
    return rows


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def summary_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ["top1", "macro_f1", "mean_precision", "mean_recall", "weighted_f1"]
    return {key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def per_class_delta(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(set(candidate) | set(baseline)):
        c = candidate.get(name, {})
        b = baseline.get(name, {})
        rows.append(
            {
                "class": name,
                "candidate_f1": float(c.get("f1", 0.0)),
                "baseline_f1": float(b.get("f1", 0.0)),
                "delta_f1": float(c.get("f1", 0.0)) - float(b.get("f1", 0.0)),
                "candidate_precision": float(c.get("precision", 0.0)),
                "baseline_precision": float(b.get("precision", 0.0)),
                "delta_precision": float(c.get("precision", 0.0)) - float(b.get("precision", 0.0)),
                "candidate_recall": float(c.get("recall", 0.0)),
                "baseline_recall": float(b.get("recall", 0.0)),
                "delta_recall": float(c.get("recall", 0.0)) - float(b.get("recall", 0.0)),
                "support": float(c.get("support", b.get("support", 0.0))),
                "key_class": name in KEY_CLASSES,
            }
        )
    return rows


def prediction_transfer(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(candidate) & set(baseline))
    fixed = 0
    worsened = 0
    changed = 0
    fixed_by_class: dict[str, int] = {}
    worsened_by_class: dict[str, int] = {}
    for path in common:
        c = candidate[path]
        b = baseline[path]
        true_label = str(c["true_label"])
        if true_label != str(b["true_label"]):
            continue
        candidate_ok = str(c["pred_label"]) == true_label
        baseline_ok = str(b["pred_label"]) == true_label
        if str(c["pred_label"]) != str(b["pred_label"]):
            changed += 1
        if candidate_ok and not baseline_ok:
            fixed += 1
            fixed_by_class[true_label] = fixed_by_class.get(true_label, 0) + 1
        elif baseline_ok and not candidate_ok:
            worsened += 1
            worsened_by_class[true_label] = worsened_by_class.get(true_label, 0) + 1
    return {
        "common_predictions": len(common),
        "changed_predictions": changed,
        "fixed": fixed,
        "worsened": worsened,
        "net_fixed": fixed - worsened,
        "fixed_by_class": dict(sorted(fixed_by_class.items(), key=lambda item: item[1], reverse=True)),
        "worsened_by_class": dict(sorted(worsened_by_class.items(), key=lambda item: item[1], reverse=True)),
    }


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _binom_tail_ge(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    if k <= n / 2:
        return 1.0
    if n > 1000:
        z = ((float(k) - 0.5) - 0.5 * float(n)) / math.sqrt(0.25 * float(n))
        return float(min(max(_normal_sf(z), 0.0), 1.0))
    prob = 2.0 ** (-n)
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * prob
    return float(min(max(total, 0.0), 1.0))


def paired_improvement_stats(transfer: dict[str, Any]) -> dict[str, Any]:
    fixed = int(transfer.get("fixed", 0) or 0)
    worsened = int(transfer.get("worsened", 0) or 0)
    n = int(transfer.get("common_predictions", 0) or 0)
    discordant = fixed + worsened
    net_fixed = fixed - worsened
    p_one_sided = _binom_tail_ge(fixed, discordant) if fixed >= worsened else 1.0
    if discordant > 0:
        z = (abs(net_fixed) - 1.0) / math.sqrt(float(discordant))
        z = max(z, 0.0)
        mcnemar_p_two_sided = min(2.0 * _normal_sf(z), 1.0)
    else:
        mcnemar_p_two_sided = 1.0
    net_rate = float(net_fixed) / float(n) if n else 0.0
    if n > 1:
        mean = net_rate
        ties = max(n - discordant, 0)
        ss = fixed * (1.0 - mean) ** 2 + worsened * (-1.0 - mean) ** 2 + ties * (0.0 - mean) ** 2
        variance = ss / float(n - 1)
        se = math.sqrt(max(variance, 0.0) / float(n))
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
    else:
        se = 0.0
        ci_low = net_rate
        ci_high = net_rate
    return {
        "fixed": fixed,
        "worsened": worsened,
        "discordant": discordant,
        "net_fixed": net_fixed,
        "net_fixed_rate": net_rate,
        "net_fixed_rate_ci95": [ci_low, ci_high],
        "paired_sign_test_p_one_sided": p_one_sided,
        "mcnemar_p_two_sided_approx": mcnemar_p_two_sided,
        "paired_improvement_significant": bool(net_fixed > 0 and p_one_sided <= 0.05),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def promotion_decision(
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    class_rows: list[dict[str, Any]],
    transfer: dict[str, Any],
    args: argparse.Namespace,
    candidate_full_training_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta = summary_delta(candidate_summary, baseline_summary)
    paired_stats = paired_improvement_stats(transfer)
    candidate_samples = int(float(candidate_summary.get("num_samples", 0) or 0))
    baseline_samples = int(float(baseline_summary.get("num_samples", 0) or 0))
    common_predictions = int(transfer.get("common_predictions", 0) or 0)
    candidate_worst = min((row["candidate_f1"] for row in class_rows), default=0.0)
    baseline_worst = min((row["baseline_f1"] for row in class_rows), default=0.0)
    key_drops = [row for row in class_rows if row["key_class"] and row["delta_f1"] < -float(args.max_key_drop)]
    large_drops = [row for row in class_rows if row["delta_f1"] < -float(args.max_any_drop)]
    checks = {
        "protocol_sample_count_match": candidate_samples == baseline_samples and candidate_samples > 0,
        "prediction_row_alignment": common_predictions == candidate_samples == baseline_samples,
        "top1_delta": delta["top1"] >= float(args.min_top1_delta),
        "macro_f1_delta": delta["macro_f1"] >= float(args.min_macro_f1_delta),
        "worst_class_noncollapse": candidate_worst + float(args.max_worst_drop) >= baseline_worst,
        "key_class_no_large_drop": not key_drops,
        "any_class_no_large_drop": not large_drops,
        "prediction_net_nonnegative": int(transfer.get("net_fixed", 0)) >= int(args.min_net_fixed),
    }
    if bool(args.require_sota):
        checks["candidate_full_test_samples"] = candidate_samples == int(args.full_test_samples)
        checks["baseline_full_test_samples"] = baseline_samples == int(args.full_test_samples)
        checks["candidate_complete_train_val_test_protocol"] = bool(
            (candidate_full_training_protocol or {}).get("passed")
        )
        checks["top1_beats_sota"] = float(candidate_summary.get("top1", 0.0)) >= float(args.sota_top1)
        checks["macro_f1_beats_sota"] = float(candidate_summary.get("macro_f1", 0.0)) >= float(args.sota_macro_f1)
        checks["paired_improvement_significant"] = (
            int(paired_stats["net_fixed"]) > 0
            and float(paired_stats["paired_sign_test_p_one_sided"]) <= float(args.paired_alpha)
        )
    passed = all(bool(value) for value in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "summary_delta": delta,
        "candidate_samples": candidate_samples,
        "baseline_samples": baseline_samples,
        "common_predictions": common_predictions,
        "paired_improvement": paired_stats,
        "candidate_worst_f1": candidate_worst,
        "baseline_worst_f1": baseline_worst,
        "candidate_full_training_protocol": candidate_full_training_protocol,
        "key_class_large_drops": key_drops,
        "any_class_large_drops": large_drops[:20],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    candidate = payload["candidate_summary"]
    baseline = payload["baseline_summary"]
    decision = payload["decision"]
    transfer = payload["prediction_transfer"]
    paired = decision.get("paired_improvement", {})
    lines = [
        "# RSCD Candidate Promotion Audit",
        "",
        f"- Candidate: `{payload['candidate_name']}`",
        f"- Candidate dir: `{payload['candidate_dir']}`",
        f"- Baseline: `{payload['baseline_name']}`",
        f"- Baseline dir: `{payload['baseline_dir']}`",
        f"- Protocol match: `{payload['protocol_match']}`",
        f"- Candidate / baseline samples: `{decision.get('candidate_samples')}` / `{decision.get('baseline_samples')}`",
        f"- Common prediction rows: `{decision.get('common_predictions')}`",
        f"- Promotion passed: `{decision['passed']}`",
        "",
        "## Metric Gate",
        "",
        "| Metric | Candidate | Baseline | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "weighted_f1"]:
        lines.append(
            f"| {key} | {pct(float(candidate.get(key, 0.0)))} | "
            f"{pct(float(baseline.get(key, 0.0)))} | {pp(decision['summary_delta'].get(key, 0.0))} |"
        )
    lines.extend(
        [
            "",
            "## Decision Checks",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for key, value in decision["checks"].items():
        lines.append(f"| {key} | {value} |")
    protocol = decision.get("candidate_full_training_protocol")
    if protocol:
        counts = protocol.get("manifest_counts") or {}
        train_protocol = protocol.get("train_protocol") or {}
        eval_protocol = protocol.get("eval_protocol") or {}
        lines.extend(
            [
                "",
                "## Full Training Protocol",
                "",
                f"- Passed: `{protocol.get('passed')}`",
                f"- Config: `{protocol.get('config_path')}`",
                "",
                "| Split | Manifest rows |",
                "|---|---:|",
                f"| train | {counts.get('train')} |",
                f"| val | {counts.get('val')} |",
                f"| test | {counts.get('test')} |",
                "",
                "| Protocol field | Value |",
                "|---|---:|",
                f"| samples_per_epoch | {train_protocol.get('samples_per_epoch')} |",
                f"| max_train_samples_per_class | {train_protocol.get('max_train_samples_per_class')} |",
                f"| max_train_samples | {train_protocol.get('max_train_samples')} |",
                f"| max_val_samples_per_class | {eval_protocol.get('max_val_samples_per_class')} |",
                f"| max_test_samples_per_class | {eval_protocol.get('max_test_samples_per_class')} |",
            ]
        )
    lines.extend(
        [
            "",
            "## No-Spill Transfer",
            "",
            f"- Common prediction rows: `{transfer['common_predictions']}`",
            f"- Fixed / worsened / net: `{transfer['fixed']}` / `{transfer['worsened']}` / `{transfer['net_fixed']}`",
            f"- Paired sign-test p-value, one-sided: `{paired.get('paired_sign_test_p_one_sided', '-')}`",
            f"- McNemar p-value, two-sided approximate: `{paired.get('mcnemar_p_two_sided_approx', '-')}`",
            f"- Net fixed rate and 95% CI: `{paired.get('net_fixed_rate', '-')}` / `{paired.get('net_fixed_rate_ci95', '-')}`",
            "",
            "## Worst And Key Classes",
            "",
            f"- Candidate worst F1: `{pct(decision['candidate_worst_f1'])}`",
            f"- Baseline worst F1: `{pct(decision['baseline_worst_f1'])}`",
            "",
            "| Key class | Candidate F1 | Baseline F1 | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted([r for r in payload["per_class_delta"] if r["key_class"]], key=lambda item: item["class"]):
        lines.append(
            f"| {row['class']} | {pct(row['candidate_f1'])} | {pct(row['baseline_f1'])} | {pp(row['delta_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest F1 Drops",
            "",
            "| Class | Candidate F1 | Baseline F1 | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(payload["per_class_delta"], key=lambda item: item["delta_f1"])[:12]:
        lines.append(
            f"| {row['class']} | {pct(row['candidate_f1'])} | {pct(row['baseline_f1'])} | {pp(row['delta_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A candidate is promoted only when it improves the target metrics without hiding regressions in key RSCD coupled classes. "
            "For S136 this is important because the route is a new custom backbone; a successful screen must prove that early coupling helps globally, not only on a cherry-picked weak boundary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether an RSCD candidate should be promoted.")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--min-top1-delta", type=float, default=0.0)
    parser.add_argument("--min-macro-f1-delta", type=float, default=0.0)
    parser.add_argument("--max-key-drop", type=float, default=0.005)
    parser.add_argument("--max-any-drop", type=float, default=0.015)
    parser.add_argument("--max-worst-drop", type=float, default=0.005)
    parser.add_argument("--min-net-fixed", type=int, default=0)
    parser.add_argument("--require-sota", action="store_true")
    parser.add_argument("--sota-top1", type=float, default=0.9286)
    parser.add_argument("--sota-macro-f1", type=float, default=0.8949)
    parser.add_argument("--full-test-samples", type=int, default=49500)
    parser.add_argument("--paired-alpha", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_summary = read_summary(args.candidate_dir)
    baseline_summary = read_summary(args.baseline_dir)
    candidate_class = read_per_class(args.candidate_dir)
    baseline_class = read_per_class(args.baseline_dir)
    candidate_pred = read_predictions(args.candidate_dir)
    baseline_pred = read_predictions(args.baseline_dir)
    missing = []
    for run_dir, summary, class_rows, pred_rows in [
        (args.candidate_dir, candidate_summary, candidate_class, candidate_pred),
        (args.baseline_dir, baseline_summary, baseline_class, baseline_pred),
    ]:
        if summary is None:
            missing.append(str(run_dir / "test_metrics.json"))
        if not class_rows:
            missing.append(str(run_dir / "per_class_metrics.csv"))
        if not pred_rows:
            missing.append(str(run_dir / "predictions_test.csv"))
    if missing:
        payload = {"ok": False, "missing": missing}
        (args.output_dir / "promotion_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    assert candidate_summary is not None
    assert baseline_summary is not None
    class_rows = per_class_delta(candidate_class, baseline_class)
    transfer = prediction_transfer(candidate_pred, baseline_pred)
    candidate_full_training_protocol = (
        full_training_protocol_evidence(args.candidate_dir, int(args.full_test_samples))
        if bool(args.require_sota)
        else None
    )
    decision = promotion_decision(
        candidate_summary,
        baseline_summary,
        class_rows,
        transfer,
        args,
        candidate_full_training_protocol,
    )
    protocol_match = int(float(candidate_summary.get("num_samples", 0) or 0)) == int(
        float(baseline_summary.get("num_samples", 0) or 0)
    )
    payload = {
        "ok": True,
        "candidate_name": args.candidate_name,
        "baseline_name": args.baseline_name,
        "candidate_dir": str(args.candidate_dir),
        "baseline_dir": str(args.baseline_dir),
        "protocol_match": protocol_match,
        "candidate_summary": candidate_summary,
        "baseline_summary": baseline_summary,
        "per_class_delta": class_rows,
        "prediction_transfer": transfer,
        "candidate_full_training_protocol": candidate_full_training_protocol,
        "decision": decision,
    }
    write_csv(args.output_dir / "per_class_delta.csv", class_rows)
    (args.output_dir / "promotion_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, args.output_dir / "promotion_audit.md")
    print(json.dumps({"ok": True, "passed": decision["passed"], "report": str(args.output_dir / "promotion_audit.md")}, ensure_ascii=False))
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
