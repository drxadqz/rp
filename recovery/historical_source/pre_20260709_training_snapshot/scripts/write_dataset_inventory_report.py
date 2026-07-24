from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml


DEFAULT_LOCAL_PATHS = Path("configs/data/local_paths.yaml")
DEFAULT_MANIFEST_DIR = Path("data/manifests_full")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_DISKS = ("C:\\", "D:\\", "E:\\")

DATASET_LABELS = {
    "rscd": "RSCD",
    "roadsaw": "RoadSaW",
    "roadsc": "RoadSC",
}

LABEL_COLUMNS = (
    "friction",
    "risk",
    "wetness",
    "snow",
    "material",
    "unevenness",
    "mu_low",
    "mu_high",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-paths", type=Path, default=DEFAULT_LOCAL_PATHS)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--compute-dir-size",
        action="store_true",
        help="Recursively compute dataset directory sizes. Disabled by default because RSCD has many files.",
    )
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    out_md = args.out_md or args.summary_dir / "dataset_inventory_report.md"
    out_json = args.out_json or args.summary_dir / "dataset_inventory_report.json"

    report = build_report(args.local_paths, args.manifest_dir, compute_dir_size=args.compute_dir_size)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_markdown(report))


def build_report(local_paths: Path, manifest_dir: Path, *, compute_dir_size: bool = False) -> dict[str, Any]:
    cfg = _load_yaml(local_paths)
    datasets = _dataset_paths(cfg, compute_dir_size=compute_dir_size)
    manifests = _manifest_summaries(manifest_dir)
    disks = _disk_summaries()
    checks = _checks(datasets, manifests, disks)
    return {
        "local_paths": str(local_paths),
        "manifest_dir": str(manifest_dir),
        "datasets": datasets,
        "manifests": manifests,
        "disks": disks,
        "checks": checks,
        "verdict": "pass" if not any(c["level"] == "block" for c in checks) else "block",
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded if isinstance(loaded, dict) else {}


def _dataset_paths(cfg: dict[str, Any], *, compute_dir_size: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in sorted((cfg.get("datasets") or {}).items()):
        if not isinstance(item, dict):
            continue
        dataset_name = str(item.get("dataset_name") or key).lower()
        root = item.get("root") or item.get("labels_csv")
        path = Path(str(root)) if root else None
        exists = bool(path and path.exists())
        rows.append(
            {
                "key": key,
                "dataset": dataset_name,
                "label": DATASET_LABELS.get(dataset_name, dataset_name),
                "path": str(path) if path else "",
                "exists": exists,
                "kind": item.get("type", ""),
                "size_gb": _dir_size_gb(path) if compute_dir_size and exists and path else None,
                "size_scan": "recursive" if compute_dir_size else "skipped",
            }
        )
    return rows


def _manifest_summaries(manifest_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not manifest_dir.exists():
        return rows
    for path in sorted(manifest_dir.glob("*.csv")):
        rows.append(_manifest_summary(path))
    return rows


def _manifest_summary(path: Path) -> dict[str, Any]:
    rows = 0
    datasets: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    present: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    invalid_mu = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            rows += 1
            dataset = _norm(row.get("dataset"))
            split = _norm(row.get("split"))
            if dataset:
                datasets[dataset] += 1
            if split:
                splits[split] += 1
            for col in LABEL_COLUMNS:
                value = row.get(col)
                if value is None or str(value).strip() == "":
                    missing[col] += 1
                else:
                    present[col] += 1
            try:
                low = float(row.get("mu_low", "nan"))
                high = float(row.get("mu_high", "nan"))
                if not low < high:
                    invalid_mu += 1
            except ValueError:
                invalid_mu += 1
    return {
        "name": path.name,
        "path": str(path),
        "rows": rows,
        "columns": fields,
        "datasets": dict(sorted(datasets.items())),
        "splits": dict(sorted(splits.items())),
        "label_present": dict(sorted(present.items())),
        "label_missing": dict(sorted(missing.items())),
        "invalid_mu_rows": invalid_mu,
    }


def _disk_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for disk in DEFAULT_DISKS:
        try:
            usage = shutil.disk_usage(disk)
        except FileNotFoundError:
            rows.append({"disk": disk, "exists": False})
            continue
        rows.append(
            {
                "disk": disk,
                "exists": True,
                "free_gb": round(usage.free / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "total_gb": round(usage.total / (1024**3), 2),
            }
        )
    return rows


def _checks(
    datasets: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    disks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    dataset_names = {row["dataset"] for row in datasets if row.get("exists")}
    for required in ("rscd", "roadsaw", "roadsc"):
        if required in dataset_names:
            _add(checks, "pass", f"{required}_local_data", f"{DATASET_LABELS[required]} local path exists.")
        else:
            _add(checks, "block", f"{required}_local_data", f"{DATASET_LABELS[required]} local path is missing.")
    by_name = {row["name"]: row for row in manifests}
    for dataset in ("rscd_prepared", "roadsaw", "roadsc"):
        for split in ("train", "val", "test"):
            name = f"{dataset}_{split}.csv"
            if name in by_name and by_name[name]["rows"] > 0:
                _add(checks, "pass", f"{name}_manifest", f"{name} has {by_name[name]['rows']} rows.")
            else:
                _add(checks, "block", f"{name}_manifest", f"{name} is missing or empty.")
    invalid = sum(int(row.get("invalid_mu_rows") or 0) for row in manifests)
    if invalid:
        _add(checks, "block", "invalid_mu_intervals", f"{invalid} manifest rows have invalid mu intervals.")
    else:
        _add(checks, "pass", "valid_mu_intervals", "All audited manifest rows have valid mu interval endpoints.")
    low_disks = [row for row in disks if row.get("exists") and float(row.get("free_gb", 0.0)) < 10.0]
    if low_disks:
        names = ", ".join(f"{row['disk']}={row['free_gb']}GB" for row in low_disks)
        _add(checks, "warn", "low_disk_free", f"Some tracked disks have low free space: {names}.")
    else:
        _add(checks, "pass", "disk_free", "All tracked disks have at least 10GB free.")
    return checks


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Inventory Report",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Local Datasets",
        "",
        "| Key | Dataset | Exists | Size GB | Path |",
        "|---|---|---:|---:|---|",
    ]
    for row in report["datasets"]:
        lines.append(
            "| {key} | {label} | {exists} | {size} | `{path}` |".format(
                key=row["key"],
                label=row["label"],
                exists="yes" if row["exists"] else "no",
                size=_fmt(row.get("size_gb")),
                path=row.get("path", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Manifests",
            "",
            "| Manifest | Rows | Datasets | Splits | Invalid mu rows |",
            "|---|---:|---|---|---:|",
        ]
    )
    for row in report["manifests"]:
        lines.append(
            "| `{name}` | {rows} | `{datasets}` | `{splits}` | {invalid} |".format(
                name=row["name"],
                rows=row["rows"],
                datasets=_compact(row.get("datasets", {})),
                splits=_compact(row.get("splits", {})),
                invalid=row.get("invalid_mu_rows", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Disk",
            "",
            "| Disk | Free GB | Used GB | Total GB |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["disks"]:
        if not row.get("exists"):
            lines.append(f"| `{row['disk']}` | - | - | - |")
        else:
            lines.append(
                "| `{disk}` | {free:.2f} | {used:.2f} | {total:.2f} |".format(
                    disk=row["disk"],
                    free=row["free_gb"],
                    used=row["used_gb"],
                    total=row["total_gb"],
                )
            )
    lines.extend(["", "## Checks", "", "| Level | Check | Message |", "|---|---|---|"])
    for check in report["checks"]:
        lines.append(f"| {check['level']} | `{check['name']}` | {check['message']} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- These datasets provide public visual road-condition labels and weak friction-affordance intervals.",
            "- They do not provide synchronized measured tire-road friction coefficients for each image.",
        ]
    )
    return "\n".join(lines) + "\n"


def _add(checks: list[dict[str, str]], level: str, name: str, message: str) -> None:
    checks.append({"level": level, "name": name, "message": message})


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _compact(value: dict[str, Any]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(value.items())) or "-"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _dir_size_gb(path: Path | None) -> float | None:
    if path is None:
        return None
    if path.is_file():
        return round(path.stat().st_size / (1024**3), 2)
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return round(total / (1024**3), 2)


if __name__ == "__main__":
    main()
