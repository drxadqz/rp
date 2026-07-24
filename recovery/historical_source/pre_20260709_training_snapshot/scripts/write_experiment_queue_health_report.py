from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


SUMMARY = Path("reports/paper_protocol_summary")
OUT_JSON = SUMMARY / "experiment_queue_health_report.json"
OUT_MD = SUMMARY / "experiment_queue_health_report.md"
ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
LOG_DIR = Path("outputs/rscd_surface_formal_queue")
SKIP_MARKERS = {
    ROOT / "fast_dinov2_physics_texture_rscd" / "evaluate_test.json": ROOT
    / "fast_dinov2_physics_texture_rscd"
    / "skipped_after_global_failure.json",
    ROOT / "fast_physics_texture_lg_224_patch_factor_aux" / "evaluate_test.json": ROOT
    / "fast_physics_texture_lg_224_patch_factor_aux"
    / "skipped_after_lg_failure.json",
    ROOT / "fast_physics_texture_local_residual_224_region" / "evaluate_test.json": ROOT
    / "fast_physics_texture_local_residual_224_region"
    / "skipped_after_local_residual_failure.json",
}

STAGES = [
    {
        "name": "formal_seed_runs",
        "watchers": [],
        "process_keywords": ["formal_convnext_tiny_b12e20_resume", "formal_physics_texture_quality_b12e20_parallel"],
        "required_outputs": [
            ROOT / "formal_convnext_tiny_b12e20_resume" / "evaluate_test.json",
            ROOT / "formal_physics_texture_quality_b12e20_parallel" / "evaluate_test.json",
        ],
        "depends_on": [],
    },
    {
        "name": "standard_fast_candidates",
        "watchers": ["run_directional_rscd_fast_after_formal", "run_hierarchical_rscd_fast_after_directional"],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "fast_physics_directional_texture_quality" / "evaluate_test.json",
            ROOT / "fast_physics_texture_hier_smoothing" / "evaluate_test.json",
            ROOT / "fast_physics_directional_hier_smoothing" / "evaluate_test.json",
            ROOT / "fast_physics_directional_gated_hier_smoothing" / "evaluate_test.json",
        ],
        "depends_on": ["formal_seed_runs"],
    },
    {
        "name": "high_priority_texture_candidates",
        "watchers": [
            "run_high_priority_texture_candidates_after_formal",
            "run_high_priority_formal_promotion_after_fast",
        ],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "fast_physics_texture_residual_adapter" / "evaluate_test.json",
            ROOT / "fast_physics_directional_residual_adapter" / "evaluate_test.json",
            ROOT / "fast_physics_texture_film" / "evaluate_test.json",
            ROOT / "fast_physics_directional_film_gate_hier" / "evaluate_test.json",
            ROOT / "fast_physics_wavelet_film" / "evaluate_test.json",
            ROOT / "fast_physics_wavelet_directional_film_gate_hier" / "evaluate_test.json",
        ],
        "depends_on": ["formal_seed_runs"],
    },
    {
        "name": "physics_attention_candidates",
        "watchers": [
            "run_physics_attention_rscd_fast_after_priority",
            "run_physics_attention_formal_promotion_after_fast",
        ],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "fast_physics_attention_film" / "evaluate_test.json",
            ROOT / "fast_physics_attention_wavelet_film_gate_hier" / "evaluate_test.json",
        ],
        "depends_on": ["high_priority_texture_candidates"],
    },
    {
        "name": "standard_formal_promotion",
        "watchers": ["run_promoted_rscd_formal_after_fast"],
        "process_keywords": [],
        "required_outputs": [],
        "depends_on": ["standard_fast_candidates"],
    },
    {
        "name": "hard_condition_candidates",
        "watchers": ["run_hard_condition_rscd_fast_after_queue", "run_hard_condition_formal_promotion_after_fast"],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "fast_physics_texture_hard_condition_boost035" / "evaluate_test.json",
            ROOT / "fast_physics_texture_hard_condition_hier_boost035" / "evaluate_test.json",
        ],
        "depends_on": ["standard_formal_promotion"],
    },
    {
        "name": "direct_visual_friction_route",
        "watchers": ["run_direct_visual_friction_after_rscd_queue"],
        "process_keywords": [],
        "required_outputs": [
            SUMMARY / "direct_visual_friction_report.md",
        ],
        "depends_on": ["hard_condition_candidates"],
    },
    {
        "name": "residual_adapter_candidates",
        "watchers": [
            "run_residual_adapter_rscd_fast_after_queue",
            "run_residual_adapter_formal_promotion_after_fast",
        ],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "fast_physics_texture_residual_adapter" / "evaluate_test.json",
            ROOT / "fast_physics_directional_residual_adapter" / "evaluate_test.json",
        ],
        "depends_on": ["direct_visual_friction_route"],
    },
    {
        "name": "texture_film_candidates",
        "watchers": [
            "run_texture_film_rscd_fast_after_queue",
            "run_wavelet_texture_rscd_fast_after_queue",
            "run_texture_film_formal_promotion_after_fast",
        ],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "fast_physics_texture_film" / "evaluate_test.json",
            ROOT / "fast_physics_directional_film_gate_hier" / "evaluate_test.json",
            ROOT / "fast_physics_wavelet_film" / "evaluate_test.json",
            ROOT / "fast_physics_wavelet_directional_film_gate_hier" / "evaluate_test.json",
        ],
        "depends_on": ["residual_adapter_candidates"],
    },
    {
        "name": "texture_film_formal_promotion",
        "watchers": ["run_texture_film_formal_promotion_after_fast"],
        "process_keywords": ["formal_physics_wavelet_directional_film_gate_hier"],
        "required_outputs": [
            ROOT / "formal_physics_wavelet_directional_film_gate_hier" / "evaluate_test.json",
        ],
        "depends_on": ["texture_film_candidates"],
    },
    {
        "name": "tta_ensemble_after_formal",
        "watchers": ["run_tta_ensemble_after_texture_wavelet"],
        "process_keywords": [],
        "required_outputs": [
            ROOT / "tta_ensemble_physics_texture_formal_hflip" / "evaluate_test.json",
        ],
        "depends_on": ["texture_film_formal_promotion"],
    },
    {
        "name": "material_conditioned_gate_fast",
        "watchers": ["run_material_conditioned_gate_fast_after_formal"],
        "process_keywords": ["fast_physics_material_gate_patch_quality"],
        "required_outputs": [
            ROOT / "fast_physics_material_gate_patch_quality" / "evaluate_test.json",
        ],
        "depends_on": ["tta_ensemble_after_formal"],
    },
    {
        "name": "retinex_texture_fast",
        "watchers": ["run_retinex_texture_fast_after_queue"],
        "process_keywords": [
            "fast_physics_retinex_texture_quality",
            "fast_physics_retinex_film_gate_hier",
        ],
        "required_outputs": [
            ROOT / "fast_physics_retinex_texture_quality" / "evaluate_test.json",
            ROOT / "fast_physics_retinex_film_gate_hier" / "evaluate_test.json",
        ],
        "depends_on": ["material_conditioned_gate_fast"],
    },
    {
        "name": "foundation_probe_fast",
        "watchers": ["run_foundation_rscd_fast_after_queue"],
        "process_keywords": [
            "fast_physics_texture_quality_patch_stats",
            "fast_physics_texture_quality_patch_stats_224",
            "fast_dinov2_global_rscd",
            "fast_dinov2_physics_texture_rscd",
        ],
        "required_outputs": [
            ROOT / "fast_physics_texture_quality_patch_stats" / "evaluate_test.json",
            ROOT / "fast_physics_texture_quality_patch_stats_224" / "evaluate_test.json",
            ROOT / "fast_dinov2_global_rscd" / "evaluate_test.json",
            ROOT / "fast_dinov2_physics_texture_rscd" / "evaluate_test.json",
        ],
        "depends_on": ["retinex_texture_fast"],
    },
    {
        "name": "resolution_224_fast",
        "watchers": ["run_resolution_rscd_fast_queue"],
        "process_keywords": [
            "rscd_surface_classification\\fast_convnext_tiny_224 --",
            "rscd_surface_classification\\fast_physics_texture_quality_224_region --",
        ],
        "required_outputs": [
            ROOT / "fast_convnext_tiny_224" / "evaluate_test.json",
            ROOT / "fast_physics_texture_quality_224_region" / "evaluate_test.json",
        ],
        "depends_on": ["foundation_probe_fast"],
    },
    {
        "name": "factor_aux_224_fast",
        "watchers": ["run_factor_aux_rscd_fast_after_resolution"],
        "process_keywords": [
            "fast_convnext_tiny_224_factor_aux",
            "fast_physics_texture_quality_224_patch_factor_aux",
        ],
        "required_outputs": [
            ROOT / "fast_convnext_tiny_224_factor_aux" / "evaluate_test.json",
            ROOT / "fast_physics_texture_quality_224_patch_factor_aux" / "evaluate_test.json",
        ],
        "depends_on": ["resolution_224_fast"],
    },
    {
        "name": "local_global_224_fast",
        "watchers": ["run_local_global_rscd_fast_after_factor"],
        "process_keywords": [
            "fast_convnext_tiny_lg_224",
            "fast_physics_texture_lg_224_patch_factor_aux",
        ],
        "required_outputs": [
            ROOT / "fast_convnext_tiny_lg_224" / "evaluate_test.json",
            ROOT / "fast_physics_texture_lg_224_patch_factor_aux" / "evaluate_test.json",
        ],
        "depends_on": ["factor_aux_224_fast"],
    },
    {
        "name": "local_residual_224_fast",
        "watchers": ["run_local_residual_rscd_fast"],
        "process_keywords": [
            "fast_convnext_tiny_local_residual_224",
            "fast_physics_texture_local_residual_224_region",
        ],
        "required_outputs": [
            ROOT / "fast_convnext_tiny_local_residual_224" / "evaluate_test.json",
            ROOT / "fast_physics_texture_local_residual_224_region" / "evaluate_test.json",
        ],
        "depends_on": ["local_global_224_fast"],
    },
    {
        "name": "strong_backbone_224_fast",
        "watchers": ["run_strong_backbone_rscd_fast"],
        "process_keywords": [
            "fast_timm_convnext_tiny_in22k_224",
            "fast_timm_convnext_tiny_in22k_physics_224",
            "fast_timm_convnextv2_tiny_224",
        ],
        "required_outputs": [
            ROOT / "fast_timm_convnext_tiny_in22k_224" / "evaluate_test.json",
            ROOT / "fast_timm_convnext_tiny_in22k_physics_224" / "evaluate_test.json",
            ROOT / "fast_timm_convnextv2_tiny_224" / "evaluate_test.json",
        ],
        "depends_on": ["local_residual_224_fast"],
    },
    {
        "name": "final_postprocess",
        "watchers": ["run_final_project_postprocess_watcher"],
        "process_keywords": [],
        "required_outputs": [
            SUMMARY / "rscd_decision_dashboard.md",
            SUMMARY / "goal_completion_audit.md",
        ],
        "depends_on": ["foundation_probe_fast"],
    },
]


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    processes = _active_processes()
    rows = [_stage_status(stage, processes) for stage in STAGES]
    report = {
        "claim_boundary": (
            "Queue-health report for local long-running RSCD/direct-friction experiments. "
            "It verifies scheduling health only; it is not model-performance evidence."
        ),
        "active_relevant_processes": processes,
        "stages": rows,
        "deadlock_assessment": _deadlock_assessment(rows),
        "logs": _log_tails(),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(OUT_MD)


def _active_processes() -> list[dict[str, Any]]:
    pattern = (
        "run_rscd_surface_classification.py|run_directional_rscd_fast_after_formal|"
        "run_hierarchical_rscd_fast_after_directional|run_promoted_rscd_formal_after_fast|"
        "run_hard_condition|run_direct_visual_friction_after_rscd_queue|"
        "run_rscd_postprocess_watcher|run_residual_adapter|run_texture_film|"
        "run_wavelet_texture|run_high_priority_texture_candidates_after_formal|"
        "run_high_priority_formal_promotion_after_fast|"
        "run_physics_attention_rscd_fast_after_priority|"
        "run_physics_attention_formal_promotion_after_fast|"
        "run_tta_ensemble_after_texture_wavelet|"
        "run_material_conditioned_gate_fast_after_formal|"
        "run_retinex_texture_fast_after_queue|"
        "fast_physics_retinex_texture_quality|fast_physics_retinex_film_gate_hier|"
        "run_foundation_rscd_fast_after_queue|fast_dinov2_global_rscd|"
        "fast_dinov2_physics_texture_rscd|fast_physics_texture_quality_patch_stats|"
        "fast_physics_texture_quality_patch_stats_224|"
        "run_resolution_rscd_fast_queue|fast_convnext_tiny_224|"
        "fast_physics_texture_quality_224_region|"
        "run_factor_aux_rscd_fast_after_resolution|fast_convnext_tiny_224_factor_aux|"
        "fast_physics_texture_quality_224_patch_factor_aux|"
        "run_local_global_rscd_fast_after_factor|fast_convnext_tiny_lg_224|"
        "fast_physics_texture_lg_224_patch_factor_aux|"
        "run_local_residual_rscd_fast|fast_convnext_tiny_local_residual_224|"
        "fast_physics_texture_local_residual_224_region|"
        "run_strong_backbone_rscd_fast|fast_timm_convnext_tiny_in22k_224|"
        "fast_timm_convnext_tiny_in22k_physics_224|fast_timm_convnextv2_tiny_224|"
        "formal_physics_wavelet_directional_film_gate_hier|"
        "run_final_project_postprocess"
    )
    script = (
        "$ErrorActionPreference='SilentlyContinue'; "
        f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    text = proc.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = [payload]
    rows = []
    for item in payload:
        command = str(item.get("CommandLine") or "")
        if "write_experiment_queue_health_report.py" in command:
            continue
        if "Get-CimInstance Win32_Process" in command and "ConvertTo-Json" in command:
            continue
        rows.append(
            {
                "pid": int(item.get("ProcessId")),
                "name": item.get("Name"),
                "command": command,
            }
        )
    return rows


def _stage_status(stage: dict[str, Any], processes: list[dict[str, Any]]) -> dict[str, Any]:
    required = list(stage["required_outputs"])
    existing = [str(path) for path in required if path.exists()]
    skipped = [str(path) for path in required if not path.exists() and _skip_marker_for(path).exists()]
    missing = [str(path) for path in required if not path.exists() and not _skip_marker_for(path).exists()]
    active = []
    for proc in processes:
        command = str(proc.get("command") or "")
        if any(keyword in command for keyword in stage["watchers"] + stage["process_keywords"]):
            active.append(proc)
    if active:
        status = "running_or_waiting"
    elif required and not missing:
        status = "complete_with_skips" if skipped else "complete"
    elif not required:
        status = "waiting_or_decision_stage"
    else:
        status = "pending"
    return {
        "name": stage["name"],
        "status": status,
        "depends_on": stage["depends_on"],
        "active_pids": [item["pid"] for item in active],
        "required_outputs_existing": existing,
        "required_outputs_skipped": skipped,
        "required_outputs_missing": missing,
    }


def _skip_marker_for(path: Path) -> Path:
    return SKIP_MARKERS.get(path, Path("__no_skip_marker__"))


def _deadlock_assessment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row["active_pids"]]
    done_statuses = {"complete", "complete_with_skips", "waiting_or_decision_stage"}
    incomplete = [row for row in rows if row["status"] not in done_statuses]
    if active:
        return {
            "status": "healthy_waiting_or_running",
            "message": (
                "At least one stage has an active process. Current dependencies are serialized; "
                "missing downstream outputs are expected until upstream formal jobs finish."
            ),
            "active_stage_count": len(active),
            "incomplete_stage_count": len(incomplete),
        }
    if incomplete:
        return {
            "status": "needs_attention",
            "message": "Some outputs are missing but no relevant watcher/training process is active.",
            "active_stage_count": 0,
            "incomplete_stage_count": len(incomplete),
        }
    return {
        "status": "complete",
        "message": "All tracked queue outputs exist.",
        "active_stage_count": 0,
        "incomplete_stage_count": 0,
    }


def _log_tails() -> list[dict[str, Any]]:
    if not LOG_DIR.exists():
        return []
    rows = []
    for path in sorted(LOG_DIR.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)[:12]:
        try:
            tail = _read_log_lines(path)[-5:]
        except OSError:
            tail = []
        rows.append(
            {
                "name": path.name,
                "last_write_time": path.stat().st_mtime,
                "size": path.stat().st_size,
                "tail": tail,
            }
        )
    return rows


def _read_log_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    if b"\x00" in raw[:512]:
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    return [line for line in text.splitlines() if line.strip()]


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Queue Health Report",
        "",
        report["claim_boundary"],
        "",
        "## Deadlock Assessment",
        "",
        f"- Status: `{report['deadlock_assessment']['status']}`",
        f"- Message: {report['deadlock_assessment']['message']}",
        f"- Active tracked stages: {report['deadlock_assessment']['active_stage_count']}",
        f"- Incomplete tracked stages: {report['deadlock_assessment']['incomplete_stage_count']}",
        "",
        "## Stage Status",
        "",
        "| stage | status | active PIDs | existing outputs | missing outputs |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["stages"]:
        lines.append(
            "| `{name}` | `{status}` | {pids} | {existing} | {missing} |".format(
                name=row["name"],
                status=row["status"],
                pids=", ".join(str(pid) for pid in row["active_pids"]) or "-",
                existing=len(row["required_outputs_existing"]) + len(row.get("required_outputs_skipped", [])),
                missing=len(row["required_outputs_missing"]),
            )
        )
    lines += [
        "",
        "## Missing Outputs",
        "",
    ]
    for row in report["stages"]:
        if row["required_outputs_missing"]:
            lines.append(f"### {row['name']}")
            lines.extend(f"- `{item}`" for item in row["required_outputs_missing"])
            lines.append("")
    skipped_rows = [row for row in report["stages"] if row.get("required_outputs_skipped")]
    if skipped_rows:
        lines += [
            "## Skipped Outputs",
            "",
        ]
        for row in skipped_rows:
            lines.append(f"### {row['name']}")
            lines.extend(f"- `{item}`" for item in row["required_outputs_skipped"])
            lines.append("")
    lines += [
        "## Recent Logs",
        "",
    ]
    for row in report["logs"]:
        lines.append(f"### {row['name']}")
        if row["tail"]:
            lines.extend(f"- {line}" for line in row["tail"])
        else:
            lines.append("- empty")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
