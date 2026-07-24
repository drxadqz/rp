from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num-bootstrap", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-incomplete-main",
        action="store_true",
        help=(
            "Also evaluate safety-selected checkpoints before the loss-selected "
            "main test/calibration artifacts exist. By default safety backfill "
            "waits for the main row so supplemental evidence cannot block P0."
        ),
    )
    args = parser.parse_args()

    generated = 0
    skipped_waiting_for_main = 0
    for run_dir in sorted(path for path in args.root.glob("*") if path.is_dir()):
        ckpt = run_dir / "best_safety.pt"
        cfg = run_dir / "config.json"
        if not ckpt.exists() or not cfg.exists():
            continue
        if not args.include_incomplete_main and not _main_ready(run_dir):
            skipped_waiting_for_main += 1
            continue
        safety_dir = run_dir / "safety_selected"
        if _ready(safety_dir) and not args.force:
            continue
        safety_dir.mkdir(parents=True, exist_ok=True)
        _run_safety_eval(cfg, ckpt, safety_dir, args.num_bootstrap)
        generated += 1
    print(f"safety-selected evaluations generated: {generated}")
    if skipped_waiting_for_main:
        print(f"safety-selected evaluations waiting for main artifacts: {skipped_waiting_for_main}")


def _run_safety_eval(config: Path, checkpoint: Path, out_dir: Path, num_bootstrap: int) -> None:
    _run(
        [
            sys.executable,
            "-u",
            "scripts/evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--out",
            str(out_dir / "evaluate_test.json"),
        ]
    )
    detailed = out_dir / "detailed_test.json"
    _run(
        [
            sys.executable,
            "-u",
            "scripts/evaluate_detailed.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--out",
            str(detailed),
        ]
    )
    for task in ["friction", "risk"]:
        _confusion(detailed, out_dir, task)
    if detailed.exists() and '"roadsaw"' in detailed.read_text(encoding="utf-8", errors="ignore"):
        for task in ["friction", "risk"]:
            _confusion(detailed, out_dir, task, dataset="roadsaw")
    _run(
        [
            sys.executable,
            "-u",
            "scripts/calibrate_intervals.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--target-coverage",
            "0.90",
            "--out",
            str(out_dir / "interval_calibration_90.json"),
        ]
    )
    _run(
        [
            sys.executable,
            "-u",
            "scripts/bootstrap_metrics.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--num-bootstrap",
            str(num_bootstrap),
            "--out-json",
            str(out_dir / "bootstrap_metrics.json"),
            "--out-md",
            str(out_dir / "bootstrap_metrics.md"),
        ]
    )
    cfg = _load_json(config)
    if isinstance(cfg, dict) and int(cfg.get("model", {}).get("num_domains", 0)) > 1:
        _run(
            [
                sys.executable,
                "-u",
                "scripts/dataset_id_diagnostic.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(out_dir / "dataset_id_diagnostic.json"),
            ]
        )
    if isinstance(cfg, dict) and cfg.get("model", {}).get("use_evidence_field"):
        _run(
            [
                sys.executable,
                "-u",
                "scripts/analyze_evidence_field.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--max-samples",
                "3000",
                "--out-json",
                str(out_dir / "evidence_field_audit.json"),
                "--out-md",
                str(out_dir / "evidence_field_audit.md"),
            ]
        )
        _run(
            [
                sys.executable,
                "-u",
                "scripts/export_evidence_maps.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--selection",
                "mixed",
                "--max-samples",
                "16",
                "--out-dir",
                str(out_dir / "evidence_maps"),
                "--clean",
            ]
        )
    _write_summary(out_dir, checkpoint)


def _confusion(detailed: Path, out_dir: Path, task: str, dataset: str | None = None) -> None:
    suffix = f"_{dataset}" if dataset else "_overall"
    cmd = [
        sys.executable,
        "scripts/summarize_confusions.py",
        "--detailed",
        str(detailed),
        "--task",
        task,
        "--out-csv",
        str(out_dir / f"confusion_{task}{suffix}.csv"),
        "--out-md",
        str(out_dir / f"confusion_{task}{suffix}.md"),
    ]
    if dataset:
        cmd.extend(["--dataset", dataset])
    _run(cmd)


def _write_summary(out_dir: Path, checkpoint: Path) -> None:
    detailed = _load_json(out_dir / "detailed_test.json") or {}
    calib = _load_json(out_dir / "interval_calibration_90.json") or {}
    bootstrap = _load_json(out_dir / "bootstrap_metrics.json") or {}
    risk = detailed.get("tasks", {}).get("risk", {})
    friction = detailed.get("tasks", {}).get("friction", {})
    low = detailed.get("low_friction_detection", {})
    mu = detailed.get("mu_interval", {})
    cal = calib.get("test_split", {})
    payload = {
        "checkpoint": str(checkpoint),
        "friction_macro_f1": friction.get("macro_f1"),
        "risk_macro_f1": risk.get("macro_f1"),
        "low_friction_recall": low.get("recall"),
        "raw_interval_coverage": mu.get("coverage"),
        "raw_interval_width": mu.get("width_mean"),
        "calibrated_coverage": cal.get("calibrated_coverage"),
        "calibrated_width": cal.get("calibrated_width"),
        "risk_macro_f1_ci": _ci(bootstrap, ["classification", "risk", "macro_f1"]),
        "low_friction_recall_ci": _ci(bootstrap, ["low_friction_detection", "recall"]),
        "calibrated_coverage_ci": _ci(bootstrap, ["mu_interval", "calibrated_coverage"]),
    }
    (out_dir / "safety_selected_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Safety-Selected Checkpoint Summary",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| friction macro-F1 | {_fmt_pct(payload['friction_macro_f1'])} |",
        f"| risk macro-F1 | {_fmt_pct(payload['risk_macro_f1'])} |",
        f"| low-friction recall | {_fmt_pct(payload['low_friction_recall'])} |",
        f"| raw interval coverage | {_fmt_pct(payload['raw_interval_coverage'])} |",
        f"| raw interval width | {_fmt_abs(payload['raw_interval_width'])} |",
        f"| calibrated coverage | {_fmt_pct(payload['calibrated_coverage'])} |",
        f"| calibrated width | {_fmt_abs(payload['calibrated_width'])} |",
        "",
    ]
    (out_dir / "safety_selected_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _ci(payload: dict, keys: list[str]) -> dict | None:
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if not isinstance(cur, dict):
        return None
    return {"low": cur.get("ci_low"), "high": cur.get("ci_high")}


def _ready(out_dir: Path) -> bool:
    return all(
        (out_dir / name).exists()
        for name in [
            "evaluate_test.json",
            "detailed_test.json",
            "interval_calibration_90.json",
            "bootstrap_metrics.json",
            "safety_selected_summary.json",
        ]
    )


def _main_ready(run_dir: Path) -> bool:
    return all(
        (run_dir / name).exists()
        for name in [
            "evaluate_test.json",
            "detailed_test.json",
            "interval_calibration_90.json",
            "bootstrap_metrics.json",
        ]
    )


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt_pct(value) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def _fmt_abs(value) -> str:
    return "-" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
