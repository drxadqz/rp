from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.utils import load_yaml  # noqa: E402


DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_QUEUE_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--queue-python", type=Path, default=DEFAULT_QUEUE_PYTHON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "gpu_protocol_audit.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "gpu_protocol_audit.json")
    args = parser.parse_args()

    report = build_report(args.config_dir, args.queue_python)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(config_dir: Path, queue_python: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(config_dir.glob("*.yaml")):
        cfg = load_yaml(path)
        rows.append(_inspect_config(path, cfg))
    failures = []
    warnings = []
    if not queue_python.exists():
        failures.append(f"queue python is missing: {queue_python}")
    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is false for the current audit interpreter")
    for row in rows:
        if row["device"] not in {"auto", "cuda"}:
            failures.append(f"{row['run']}: device is {row['device']}")
        if not row["amp"]:
            warnings.append(f"{row['run']}: AMP is disabled")
        if row["batch_size"] > 16:
            warnings.append(f"{row['run']}: batch_size {row['batch_size']} may exceed the RTX 3050 safety envelope")
        if row["batch_size"] <= 8 and row["grad_accum_steps"] < 4:
            warnings.append(f"{row['run']}: small batch has grad_accum_steps {row['grad_accum_steps']}")
        if row["image_size"] > 224:
            warnings.append(f"{row['run']}: image_size {row['image_size']} may increase VRAM pressure")
    verdict = "pass" if not failures else "fail"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_dir": str(config_dir),
        "queue_python": str(queue_python),
        "queue_python_exists": queue_python.exists(),
        "audit_python": sys.executable,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        },
        "environment_policy": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
            "TEMP": os.environ.get("TEMP"),
            "TMP": os.environ.get("TMP"),
            "FAF_ALLOW_WINDOWS_DATALOADER_WORKERS": os.environ.get("FAF_ALLOW_WINDOWS_DATALOADER_WORKERS"),
            "windows_dataloader_worker_policy": (
                "DataLoader workers are disabled on Windows unless FAF_ALLOW_WINDOWS_DATALOADER_WORKERS is set. "
                "This favors long-run stability over maximum throughput."
            ),
        },
        "num_configs": len(rows),
        "rows": rows,
        "failures": failures,
        "warnings": warnings,
        "verdict": verdict,
    }


def _inspect_config(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data", {})
    optim = cfg.get("optim", {})
    return {
        "run": path.stem,
        "config": str(path),
        "output_dir": cfg.get("output_dir"),
        "device": str(cfg.get("device", "auto")),
        "image_size": int(data.get("image_size", 0) or 0),
        "batch_size": int(data.get("batch_size", 0) or 0),
        "num_workers_configured": int(data.get("num_workers", 0) or 0),
        "balanced_num_samples_per_epoch": data.get("balanced_num_samples_per_epoch"),
        "epochs": int(optim.get("epochs", 0) or 0),
        "grad_accum_steps": int(optim.get("grad_accum_steps", 1) or 1),
        "effective_batch": int(data.get("batch_size", 0) or 0) * int(optim.get("grad_accum_steps", 1) or 1),
        "amp": bool(optim.get("amp", False)),
        "early_stop_patience": optim.get("early_stop_patience"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    torch_info = report["torch"]
    env = report["environment_policy"]
    lines = [
        "# GPU Protocol Audit",
        "",
        f"Generated at: {report['generated_at']}",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Runtime",
        "",
        f"- Queue Python: `{report['queue_python']}`; exists: `{report['queue_python_exists']}`.",
        f"- Audit Python: `{report['audit_python']}`.",
        f"- Torch: `{torch_info['version']}`; CUDA build: `{torch_info['cuda_build']}`; CUDA available: `{torch_info['cuda_available']}`.",
        f"- GPU: `{torch_info['device_name']}`; capability: `{torch_info['device_capability']}`.",
        "",
        "## Environment Policy",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- `{key}`: {value if value is not None else '-'}")
    lines.extend(
        [
            "",
            "## Config Envelope",
            "",
            "| Run | device | image | batch | accum | effective | AMP | epochs | samples/epoch |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {run} | {device} | {image} | {batch} | {accum} | {effective} | {amp} | {epochs} | {samples} |".format(
                run=row["run"],
                device=row["device"],
                image=row["image_size"],
                batch=row["batch_size"],
                accum=row["grad_accum_steps"],
                effective=row["effective_batch"],
                amp="yes" if row["amp"] else "no",
                epochs=row["epochs"],
                samples=row["balanced_num_samples_per_epoch"],
            )
        )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for item in report["failures"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        for item in report["warnings"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
