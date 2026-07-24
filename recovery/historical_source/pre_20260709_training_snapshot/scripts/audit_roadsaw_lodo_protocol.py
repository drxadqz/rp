from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/experiments/paper_protocol/lodo_roadsaw_full_faf.yaml")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
REQUIRED_COLUMNS = {
    "image_path",
    "split",
    "dataset",
    "class_label",
    "domain_id",
    "friction_label",
    "risk_label",
    "mu_low",
    "mu_high",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "roadsaw_lodo_protocol_audit.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "roadsaw_lodo_protocol_audit.json")
    args = parser.parse_args()

    report = audit(args.config)
    md = render_markdown(report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def audit(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    splits = {
        "train": [Path(item) for item in data.get("train_manifests", [])],
        "val": [Path(item) for item in data.get("val_manifests", [])],
        "test": [Path(item) for item in data.get("test_manifests", [])],
    }
    manifest_rows: dict[str, list[dict[str, Any]]] = {
        split: [_read_manifest(path) for path in paths]
        for split, paths in splits.items()
    }
    split_summaries = {
        split: _summarize_split(split, rows)
        for split, rows in manifest_rows.items()
    }
    checks: list[dict[str, Any]] = []
    _check_manifest_paths(checks, splits)
    _check_columns(checks, manifest_rows)
    _check_dataset_partition(checks, split_summaries)
    _check_path_overlap(checks, manifest_rows)
    _check_mu_ranges(checks, manifest_rows)
    _check_split_columns(checks, split_summaries)
    _check_label_coverage(checks, split_summaries)

    verdict = "pass"
    if any(item["level"] == "block" for item in checks):
        verdict = "fail"
    elif any(item["level"] == "warn" for item in checks):
        verdict = "pass_with_warnings"

    return {
        "config": str(config_path),
        "output_dir": str(cfg.get("output_dir", "")),
        "verdict": verdict,
        "checks": checks,
        "splits": split_summaries,
        "policy": [
            "The held-out RoadSaW LODO run must exclude roadsaw rows from train and validation manifests.",
            "The test split must contain only roadsaw rows.",
            "Train/validation/test image paths must be disjoint.",
            "Weak friction interval labels are audited as public-label-derived affordance intervals, not measured tire-road friction.",
        ],
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "columns": [],
            "rows": [],
            "num_rows": 0,
        }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
        columns = list(reader.fieldnames or [])
    return {
        "path": str(path),
        "exists": True,
        "columns": columns,
        "rows": rows,
        "num_rows": len(rows),
    }


def _summarize_split(split: str, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    friction_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    paths: set[str] = set()
    duplicate_paths = 0
    num_rows = 0
    min_mu_low: float | None = None
    max_mu_high: float | None = None
    for manifest in manifests:
        for row in manifest.get("rows", []):
            num_rows += 1
            path = _norm(row.get("image_path"))
            if path:
                if path in paths:
                    duplicate_paths += 1
                paths.add(path)
            dataset_counter[_norm(row.get("dataset"))] += 1
            split_counter[_norm(row.get("split"))] += 1
            class_counter[_norm(row.get("class_label"))] += 1
            friction_counter[_norm(row.get("friction_label"))] += 1
            risk_counter[_norm(row.get("risk_label"))] += 1
            domain_counter[_norm(row.get("domain_id"))] += 1
            low = _num(row.get("mu_low"))
            high = _num(row.get("mu_high"))
            if low is not None:
                min_mu_low = low if min_mu_low is None else min(min_mu_low, low)
            if high is not None:
                max_mu_high = high if max_mu_high is None else max(max_mu_high, high)
    return {
        "split": split,
        "manifest_paths": [item.get("path") for item in manifests],
        "num_manifests": len(manifests),
        "num_rows": num_rows,
        "datasets": dict(sorted(dataset_counter.items())),
        "split_values": dict(sorted(split_counter.items())),
        "domains": dict(sorted(domain_counter.items())),
        "num_unique_paths": len(paths),
        "duplicate_paths_within_split": duplicate_paths,
        "class_counts_top": _top_counts(class_counter),
        "friction_counts": dict(sorted(friction_counter.items())),
        "risk_counts": dict(sorted(risk_counter.items())),
        "min_mu_low": min_mu_low,
        "max_mu_high": max_mu_high,
    }


def _check_manifest_paths(checks: list[dict[str, Any]], splits: dict[str, list[Path]]) -> None:
    for split, paths in splits.items():
        if not paths:
            _add(checks, "block", "missing_manifest_list", f"{split} manifest list is empty.", split=split)
        for path in paths:
            if not path.exists():
                _add(checks, "block", "manifest_missing", f"{split} manifest does not exist: {path}", split=split, path=str(path))


def _check_columns(checks: list[dict[str, Any]], manifest_rows: dict[str, list[dict[str, Any]]]) -> None:
    for split, manifests in manifest_rows.items():
        for manifest in manifests:
            missing = sorted(REQUIRED_COLUMNS - set(manifest.get("columns", [])))
            if missing:
                _add(
                    checks,
                    "block",
                    "manifest_required_columns",
                    f"{split} manifest is missing required columns.",
                    split=split,
                    path=manifest.get("path"),
                    missing=missing,
                )


def _check_dataset_partition(checks: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> None:
    for split in ("train", "val"):
        datasets = set(summaries[split].get("datasets", {}))
        if "roadsaw" in datasets:
            _add(checks, "block", "roadsaw_leakage", f"RoadSaW appears in {split}; held-out protocol is invalid.", split=split)
        expected = {"rscd", "roadsc"}
        if datasets != expected:
            _add(
                checks,
                "block",
                "lodo_train_val_dataset_set",
                f"{split} should contain exactly RSCD and RoadSC.",
                split=split,
                observed=sorted(datasets),
                expected=sorted(expected),
            )
    test_datasets = set(summaries["test"].get("datasets", {}))
    if test_datasets != {"roadsaw"}:
        _add(
            checks,
            "block",
            "lodo_test_dataset_set",
            "Test split should contain only RoadSaW.",
            observed=sorted(test_datasets),
            expected=["roadsaw"],
        )


def _check_path_overlap(checks: list[dict[str, Any]], manifest_rows: dict[str, list[dict[str, Any]]]) -> None:
    paths_by_split = {}
    for split, manifests in manifest_rows.items():
        paths: set[str] = set()
        for manifest in manifests:
            for row in manifest.get("rows", []):
                path = _norm(row.get("image_path"))
                if path:
                    paths.add(path)
        paths_by_split[split] = paths
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(paths_by_split[left] & paths_by_split[right])
        if overlap:
            _add(
                checks,
                "block",
                "image_path_overlap",
                f"{left} and {right} image paths overlap.",
                left=left,
                right=right,
                num_overlap=len(overlap),
                examples=overlap[:5],
            )


def _check_mu_ranges(checks: list[dict[str, Any]], manifest_rows: dict[str, list[dict[str, Any]]]) -> None:
    bad: list[dict[str, Any]] = []
    missing = 0
    for split, manifests in manifest_rows.items():
        for manifest in manifests:
            for idx, row in enumerate(manifest.get("rows", []), start=2):
                low = _num(row.get("mu_low"))
                high = _num(row.get("mu_high"))
                if low is None or high is None:
                    missing += 1
                    continue
                if low > high or low < 0.0 or high > 1.3:
                    bad.append(
                        {
                            "split": split,
                            "manifest": manifest.get("path"),
                            "line": idx,
                            "mu_low": low,
                            "mu_high": high,
                        }
                    )
    if missing:
        _add(checks, "block", "missing_mu_interval", "Some rows are missing weak friction interval endpoints.", num_rows=missing)
    if bad:
        _add(checks, "block", "invalid_mu_interval", "Some weak friction intervals are invalid.", examples=bad[:10], num_rows=len(bad))


def _check_split_columns(checks: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> None:
    for split, summary in summaries.items():
        observed = set(summary.get("split_values", {}))
        if observed != {split}:
            _add(
                checks,
                "block",
                "split_column_mismatch",
                f"{split} manifest rows should have split={split}.",
                split=split,
                observed=sorted(observed),
            )


def _check_label_coverage(checks: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> None:
    for split, summary in summaries.items():
        if not summary.get("risk_counts"):
            _add(checks, "block", "missing_risk_labels", f"{split} split has no risk labels.", split=split)
        if not summary.get("friction_counts"):
            _add(checks, "block", "missing_friction_labels", f"{split} split has no friction labels.", split=split)
        if summary.get("num_rows", 0) <= 0:
            _add(checks, "block", "empty_split", f"{split} split has no rows.", split=split)
        if summary.get("duplicate_paths_within_split", 0) > 0:
            _add(
                checks,
                "warn",
                "duplicate_paths_within_split",
                f"{split} split contains duplicate image paths.",
                split=split,
                duplicates=summary.get("duplicate_paths_within_split"),
            )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RoadSaW LODO Protocol Audit",
        "",
        f"Config: `{report['config']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Split Summary",
        "",
        "| Split | Rows | Datasets | Split values | Domains | mu range | Top classes |",
        "|---|---:|---|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        summary = report["splits"][split]
        mu_range = _fmt_range(summary.get("min_mu_low"), summary.get("max_mu_high"))
        lines.append(
            "| {split} | {rows} | `{datasets}` | `{split_values}` | `{domains}` | {mu} | `{classes}` |".format(
                split=split,
                rows=summary.get("num_rows"),
                datasets=_compact_dict(summary.get("datasets", {})),
                split_values=_compact_dict(summary.get("split_values", {})),
                domains=_compact_dict(summary.get("domains", {})),
                mu=mu_range,
                classes=_compact_dict(summary.get("class_counts_top", {})),
            )
        )
    lines.extend(["", "## Checks", "", "| Level | Check | Message |", "|---|---|---|"])
    for item in report.get("checks", []):
        lines.append(f"| {item.get('level')} | `{item.get('name')}` | {item.get('message')} |")
    if not report.get("checks"):
        lines.append("| pass | `all_checks` | All RoadSaW held-out protocol checks passed. |")
    lines.extend(["", "## Policy", ""])
    for idx, item in enumerate(report.get("policy", []), start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")
    return "\n".join(lines)


def _top_counts(counter: Counter[str], n: int = 6) -> dict[str, int]:
    return dict(counter.most_common(n))


def _compact_dict(data: dict[str, Any]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in data.items()) or "-"


def _fmt_range(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"`{float(low):.3f}-{float(high):.3f}`"


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\\", "/").lower()


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add(checks: list[dict[str, Any]], level: str, name: str, message: str, **details: Any) -> None:
    checks.append({"level": level, "name": name, "message": message, **details})


if __name__ == "__main__":
    main()
