from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pair_key(base_pred: str, cand_pred: str) -> str:
    return f"{base_pred}->{cand_pred}"


def learn_accept_pairs(
    baseline_val: list[dict[str, str]],
    candidate_val: list[dict[str, str]],
    *,
    min_changed: int,
    min_net_gain: int,
    min_confidence_delta: float,
    max_baseline_confidence: float | None,
) -> tuple[set[str], list[dict[str, Any]]]:
    base = {row["image_path"]: row for row in baseline_val}
    cand = {row["image_path"]: row for row in candidate_val}
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(set(base) & set(cand)):
        b = base[path]
        c = cand[path]
        if b["pred_label"] == c["pred_label"]:
            continue
        base_conf = float(b.get("confidence", 0.0) or 0.0)
        cand_conf = float(c.get("confidence", 0.0) or 0.0)
        if cand_conf - base_conf < float(min_confidence_delta):
            continue
        if max_baseline_confidence is not None and base_conf > float(max_baseline_confidence):
            continue
        key = pair_key(b["pred_label"], c["pred_label"])
        true = b["true_label"]
        was_ok = b["pred_label"] == true
        now_ok = c["pred_label"] == true
        if now_ok and not was_ok:
            stats[key]["fixed"] += 1
        elif was_ok and not now_ok:
            stats[key]["regressed"] += 1
        else:
            stats[key]["neutral"] += 1
        stats[key]["changed"] += 1
    rows = []
    accept: set[str] = set()
    for key, item in sorted(stats.items()):
        fixed = int(item["fixed"])
        regressed = int(item["regressed"])
        changed = int(item["changed"])
        net = fixed - regressed
        use = changed >= int(min_changed) and net >= int(min_net_gain)
        if use:
            accept.add(key)
        rows.append(
            {
                "pair": key,
                "changed": changed,
                "fixed": fixed,
                "regressed": regressed,
                "neutral": int(item["neutral"]),
                "net": net,
                "accepted": int(use),
            }
        )
    rows.sort(key=lambda row: (int(row["accepted"]), int(row["net"]), int(row["changed"])), reverse=True)
    return accept, rows


def apply_arbitration(
    baseline_test: list[dict[str, str]],
    candidate_test: list[dict[str, str]],
    accept_pairs: set[str],
    *,
    min_confidence_delta: float,
    max_baseline_confidence: float | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    base = {row["image_path"]: row for row in baseline_test}
    cand = {row["image_path"]: row for row in candidate_test}
    rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    for path in sorted(set(base) & set(cand)):
        b = base[path]
        c = cand[path]
        key = pair_key(b["pred_label"], c["pred_label"])
        base_conf = float(b.get("confidence", 0.0) or 0.0)
        cand_conf = float(c.get("confidence", 0.0) or 0.0)
        confidence_ok = cand_conf - base_conf >= float(min_confidence_delta)
        if max_baseline_confidence is not None:
            confidence_ok = confidence_ok and base_conf <= float(max_baseline_confidence)
        accept = b["pred_label"] != c["pred_label"] and key in accept_pairs and confidence_ok
        pred = c["pred_label"] if accept else b["pred_label"]
        decision_counts["accepted" if accept else "baseline"] += 1
        if b["pred_label"] != c["pred_label"]:
            decision_counts["changed_seen"] += 1
        rows.append(
            {
                "image_path": path,
                "true_label": b["true_label"],
                "pred_label": pred,
                "baseline_pred": b["pred_label"],
                "candidate_pred": c["pred_label"],
                "accepted_candidate": int(accept),
                "pair": key if b["pred_label"] != c["pred_label"] else "",
                "baseline_confidence": b.get("confidence", ""),
                "candidate_confidence": c.get("confidence", ""),
                "confidence_delta": cand_conf - base_conf,
            }
        )
    return rows, decision_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-learned pair acceptance arbitration.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-val-dir", type=Path, required=True)
    parser.add_argument("--candidate-val-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-changed", type=int, default=1)
    parser.add_argument("--min-net-gain", type=int, default=1)
    parser.add_argument("--min-confidence-delta", type=float, default=0.0)
    parser.add_argument("--max-baseline-confidence", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accept_pairs, pair_rows = learn_accept_pairs(
        read_csv(args.baseline_val_dir / "predictions_val.csv"),
        read_csv(args.candidate_val_dir / "predictions_val.csv"),
        min_changed=int(args.min_changed),
        min_net_gain=int(args.min_net_gain),
        min_confidence_delta=float(args.min_confidence_delta),
        max_baseline_confidence=args.max_baseline_confidence,
    )
    arb_rows, decision_counts = apply_arbitration(
        read_csv(args.baseline_dir / "predictions_test.csv"),
        read_csv(args.candidate_dir / "predictions_test.csv"),
        accept_pairs,
        min_confidence_delta=float(args.min_confidence_delta),
        max_baseline_confidence=args.max_baseline_confidence,
    )
    labels = sorted({row["true_label"] for row in arb_rows} | {row["pred_label"] for row in arb_rows})
    y_true = [row["true_label"] for row in arb_rows]
    y_pred = [row["pred_label"] for row in arb_rows]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(report["macro avg"]["f1-score"])
    errors = int(sum(t != p for t, p in zip(y_true, y_pred)))
    per_class = []
    for label in labels:
        item = report[label]
        per_class.append(
            {
                "class": label,
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1-score"],
                "support": item["support"],
            }
        )
    write_rows(args.out_dir / "pair_acceptance_val.csv", pair_rows)
    write_rows(args.out_dir / "predictions_test.csv", arb_rows)
    write_rows(args.out_dir / "per_class_metrics.csv", per_class)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    write_rows(
        args.out_dir / "confusion_matrix.csv",
        [
            {"class": label, **{labels[j]: int(value) for j, value in enumerate(matrix[i])}}
            for i, label in enumerate(labels)
        ],
    )
    payload = {
        "summary": {
            "top1": acc,
            "macro_f1": macro_f1,
            "num_samples": len(arb_rows),
            "num_classes": len(labels),
            "num_errors": errors,
            "accepted_pairs": sorted(accept_pairs),
            "decision_counts": dict(decision_counts),
            "min_confidence_delta": float(args.min_confidence_delta),
            "max_baseline_confidence": args.max_baseline_confidence,
        },
        "classification_report": report,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
