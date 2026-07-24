from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from friction_affordance.utils import load_yaml
from paper_protocol_progress import ROWS, inspect_run


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")

BASE_REQUIRED = [
    "config.json",
    "manifest_stats_train.json",
    "best.pt",
    "evaluate_test.json",
    "detailed_test.json",
    "interval_calibration_90.json",
    "bootstrap_metrics.json",
    "bootstrap_metrics.md",
    "topvenue_result_audit.json",
    "topvenue_result_audit.md",
]
CONFUSION_REQUIRED = [
    "confusion_friction_overall.csv",
    "confusion_friction_overall.md",
    "confusion_risk_overall.csv",
    "confusion_risk_overall.md",
]
ROADSAW_CONFUSION_REQUIRED = [
    "confusion_friction_roadsaw.csv",
    "confusion_friction_roadsaw.md",
    "confusion_risk_roadsaw.csv",
    "confusion_risk_roadsaw.md",
]
DATASET_ID_REQUIRED = ["dataset_id_diagnostic.json"]
EVIDENCE_REQUIRED = [
    "evidence_maps",
    "evidence_field_audit.json",
    "evidence_field_audit.md",
]
TRAINING_TRACE_RECOMMENDED = [
    "metrics_history.json",
    "metrics_history.csv",
    "training_state.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "artifact_contract_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "artifact_contract_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir, args.config_dir, args.log_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(root: Path, summary_dir: Path, config_dir: Path, log_dir: Path) -> dict[str, Any]:
    rows = [_inspect_contract(root, config_dir, log_dir, name) for name in ROWS]
    groups = _group_rows(rows)
    hard_groups = {
        "p0_ablation": ["v0_", "v1_", "v2_", "v3_", "v4_", "v5_"],
        "lodo": ["lodo_"],
        "single_dataset_fair": ["single_", "baseline_single_"],
        "p1_candidates": ["v6_", "v7_", "v8_", "v9_", "v10_", "v11_", "v12_", "v13_", "v14_", "v15_", "v16_", "v17_", "v18_", "v19_", "v20_", "v21_", "v22_", "v23_", "v24_"],
        "final_method": ["final_lodo_", "final_single_"],
    }
    hard_status = {
        name: _hard_group_status(rows, prefixes)
        for name, prefixes in hard_groups.items()
    }
    invalid_complete_like = [
        row
        for row in rows
        if row["progress_status"] in {"complete", "partial_ci_missing"}
        and row["contract_status"] != "complete"
    ]
    stale_rows = [row for row in rows if row["stale_artifacts"]]
    completed_rows = [row for row in rows if row["contract_status"] == "complete"]
    incomplete_rows = [row for row in rows if row["contract_status"] != "complete"]
    verdict = "pass"
    if invalid_complete_like or stale_rows:
        verdict = "block"
    elif incomplete_rows:
        verdict = "in_progress"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "summary_dir": str(summary_dir),
        "config_dir": str(config_dir),
        "verdict": verdict,
        "num_runs": len(rows),
        "num_contract_complete": len(completed_rows),
        "num_contract_incomplete": len(incomplete_rows),
        "num_invalid_complete_like": len(invalid_complete_like),
        "num_stale_rows": len(stale_rows),
        "rows": rows,
        "groups": groups,
        "hard_status": hard_status,
        "claim_policy": [
            "A row is paper-table complete only when checkpoint, test metrics, detailed metrics, conformal calibration, bootstrap CI, and top-venue audit artifacts exist.",
            "Multi-dataset, LODO, and P1/final cross-dataset rows require dataset-ID diagnostic artifacts; strict single-dataset and baseline rows do not.",
            "EvidenceField rows require quantitative evidence-field audit and exported evidence maps before interpretability claims.",
            "RoadSaW-containing test rows require RoadSaW-specific confusion artifacts for friction and risk.",
            "Weak friction intervals are public-label-derived proxies, not synchronized tire-dynamics friction measurements.",
        ],
    }


def _inspect_contract(root: Path, config_dir: Path, log_dir: Path, name: str) -> dict[str, Any]:
    run_dir = root / name
    progress = inspect_run(run_dir, log_dir=log_dir)
    planned_config = _load_config(config_dir / f"{name}.yaml")
    run_config = _load_json(run_dir / "config.json")
    config = run_config if isinstance(run_config, dict) else planned_config
    scope = _scope_for(name)
    uses_evidence = _uses_evidence(config, name)
    requires_dataset_id = _requires_dataset_id(name, config)
    touches_roadsaw = _touches_roadsaw(config, run_dir)

    required = list(BASE_REQUIRED)
    required.extend(CONFUSION_REQUIRED)
    if touches_roadsaw:
        required.extend(ROADSAW_CONFUSION_REQUIRED)
    if requires_dataset_id:
        required.extend(DATASET_ID_REQUIRED)
    if uses_evidence:
        required.extend(EVIDENCE_REQUIRED)

    artifact_rows = [_artifact_row(run_dir, artifact) for artifact in required]
    recommended_rows = [_artifact_row(run_dir, artifact) for artifact in TRAINING_TRACE_RECOMMENDED]
    missing = [row["artifact"] for row in artifact_rows if not row["exists"] or row["bytes"] == 0]
    stale = _stale_artifacts(run_dir, artifact_rows, progress, config)
    contract_status = "complete" if not missing and not stale else ("missing" if not run_dir.exists() else "incomplete")

    return {
        "name": name,
        "scope": scope,
        "path": str(run_dir),
        "config_path": str(config_dir / f"{name}.yaml"),
        "config_exists": (config_dir / f"{name}.yaml").exists(),
        "progress_status": progress.get("status"),
        "contract_status": contract_status,
        "epoch": progress.get("epoch"),
        "epochs": progress.get("epochs"),
        "active_epoch": progress.get("active_epoch"),
        "active_epochs": progress.get("active_epochs"),
        "active_step": progress.get("active_step"),
        "active_steps": progress.get("active_steps"),
        "uses_evidence_field": uses_evidence,
        "requires_dataset_id_diagnostic": requires_dataset_id,
        "touches_roadsaw_test": touches_roadsaw,
        "missing_required_artifacts": missing,
        "stale_artifacts": stale,
        "required_artifacts": artifact_rows,
        "recommended_trace_artifacts": recommended_rows,
        "next_action": _next_action(name, progress, missing, stale),
    }


def _artifact_row(run_dir: Path, artifact: str) -> dict[str, Any]:
    path = run_dir / artifact
    exists = path.exists()
    if exists and path.is_dir():
        children = [item for item in path.rglob("*") if item.is_file()]
        size = sum(item.stat().st_size for item in children)
        mtime = max((item.stat().st_mtime for item in children), default=path.stat().st_mtime)
    elif exists:
        size = path.stat().st_size
        mtime = path.stat().st_mtime
    else:
        size = 0
        mtime = None
    return {
        "artifact": artifact,
        "exists": exists,
        "bytes": size,
        "path": str(path),
        "mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S") if mtime else None,
    }


def _stale_artifacts(
    run_dir: Path,
    artifacts: list[dict[str, Any]],
    progress: dict[str, Any],
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    best = run_dir / "best.pt"
    if not best.exists():
        return []
    best_mtime = best.stat().st_mtime
    stale: list[dict[str, Any]] = []
    eval_artifacts = {
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "dataset_id_diagnostic.json",
        "evidence_field_audit.json",
        "topvenue_result_audit.json",
    }
    for row in artifacts:
        if row["artifact"] not in eval_artifacts or not row["exists"] or row["mtime"] is None:
            continue
        path = Path(str(row["path"]))
        if path.exists() and path.stat().st_mtime + 1 < best_mtime:
            stale.append(
                {
                    "artifact": row["artifact"],
                    "reason": "best_checkpoint_newer_than_artifact",
                    "best_mtime": datetime.fromtimestamp(best_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "artifact_mtime": row["mtime"],
                }
            )
    last = run_dir / "last.pt"
    if last.exists() and progress.get("contract_status") == "complete":
        stale.append({"artifact": "last.pt", "reason": "resume_checkpoint_left_after_complete_contract"})
    if last.exists() and not _training_complete(run_dir, config):
        detailed = run_dir / "detailed_test.json"
        if detailed.exists() and last.stat().st_mtime > detailed.stat().st_mtime + 1:
            stale.append({"artifact": "detailed_test.json", "reason": "training_checkpoint_newer_than_evaluation"})
    return stale


def _training_complete(run_dir: Path, config: dict[str, Any] | None) -> bool:
    state = _load_json(run_dir / "training_state.json")
    if not isinstance(state, dict):
        return False
    epochs = int(_dig(config or {}, ["optim", "epochs"], state.get("epochs") or 0) or 0)
    patience = _dig(config or {}, ["optim", "early_stop_patience"])
    reached_epochs = int(state.get("epoch", 0) or 0) >= epochs
    reached_patience = patience is not None and int(state.get("stale_epochs", 0) or 0) >= int(patience)
    return reached_epochs or reached_patience


def _requires_dataset_id(name: str, config: dict[str, Any] | None) -> bool:
    if name.startswith(("single_", "baseline_single_", "final_single_")):
        return False
    if not isinstance(config, dict):
        return True
    num_domains = _dig(config, ["model", "num_domains"])
    if num_domains is not None:
        return int(num_domains) > 1
    train_manifests = _dig(config, ["data", "train_manifests"], [])
    return len(_datasets_from_manifests(train_manifests)) > 1


def _uses_evidence(config: dict[str, Any] | None, name: str) -> bool:
    if isinstance(config, dict):
        return bool(_dig(config, ["model", "use_evidence_field"], False))
    text = name.lower()
    return "faf" in text or "evidence" in text or "lean" in text


def _touches_roadsaw(config: dict[str, Any] | None, run_dir: Path) -> bool:
    detailed = _load_json(run_dir / "detailed_test.json")
    if isinstance(detailed, dict) and "roadsaw" in json.dumps(detailed).lower():
        return True
    if not isinstance(config, dict):
        return False
    for key in ["test_manifests", "eval_manifests", "val_manifests", "train_manifests"]:
        manifests = _dig(config, ["data", key], [])
        if any("roadsaw" in str(item).lower() for item in manifests):
            return key == "test_manifests" or run_dir.name.startswith(("v", "single_roadsaw", "baseline_single_roadsaw", "final_single_roadsaw"))
    return False


def _datasets_from_manifests(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    out: set[str] = set()
    for item in items:
        text = str(item).replace("\\", "/").lower()
        if "roadsaw" in text:
            out.add("roadsaw")
        if "roadsc" in text:
            out.add("roadsc")
        if "rscd" in text:
            out.add("rscd")
    return out


def _scope_for(name: str) -> str:
    if name.startswith("baseline_single_"):
        return "single_dataset_baseline"
    if name.startswith("single_"):
        return "single_dataset_faf"
    if name.startswith("final_single_"):
        return "final_single_dataset"
    if name.startswith("final_lodo_"):
        return "final_lodo"
    if name.startswith("lodo_"):
        return "lodo"
    if name.startswith(("v0_", "v1_", "v2_", "v3_", "v4_", "v5_")):
        return "p0_ablation"
    if name.startswith("v"):
        return "p1_candidate"
    return "unknown"


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["scope"]), []).append(row)
    out: dict[str, Any] = {}
    for name, items in groups.items():
        out[name] = {
            "num_runs": len(items),
            "num_complete": sum(1 for item in items if item["contract_status"] == "complete"),
            "num_incomplete": sum(1 for item in items if item["contract_status"] != "complete"),
            "missing_or_incomplete": [
                item["name"]
                for item in items
                if item["contract_status"] != "complete"
            ],
        }
    return out


def _hard_group_status(rows: list[dict[str, Any]], prefixes: list[str]) -> dict[str, Any]:
    members = [
        row
        for row in rows
        if any(str(row["name"]).startswith(prefix) for prefix in prefixes)
    ]
    missing = [row["name"] for row in members if row["contract_status"] != "complete"]
    return {
        "complete": not missing and bool(members),
        "num_runs": len(members),
        "num_complete": len(members) - len(missing),
        "missing": missing,
    }


def _next_action(name: str, progress: dict[str, Any], missing: list[str], stale: list[dict[str, Any]]) -> str:
    if not missing and not stale:
        return "ready_for_tables"
    if progress.get("status") == "missing":
        return "run_training_and_full_eval_pipeline"
    if progress.get("status") == "running_or_partial" and progress.get("active_epoch") is not None:
        return "finish_current_training_then_postprocess"
    if any(item.endswith(".pt") for item in missing):
        return "finish_or_resume_training"
    if stale:
        return "rerun_eval_pipeline_from_current_best_checkpoint"
    if missing:
        return "run_missing_postprocess_artifacts"
    return f"inspect_{name}"


def _load_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = load_yaml(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dig(items: Any, keys: list[str], default: Any = None) -> Any:
    cur = items
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Artifact Contract Report",
        "",
        f"Generated at: {report['generated_at']}",
        f"Root: `{report['root']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Summary",
        "",
        f"- Runs: `{report['num_runs']}`.",
        f"- Contract complete: `{report['num_contract_complete']}`.",
        f"- Contract incomplete: `{report['num_contract_incomplete']}`.",
        f"- Complete-looking but contract-incomplete rows: `{report['num_invalid_complete_like']}`.",
        f"- Rows with stale artifacts: `{report['num_stale_rows']}`.",
        "",
        "## Hard Groups",
        "",
        "| Group | Complete | Runs | Complete Runs | Missing |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in report["hard_status"].items():
        missing = ", ".join(f"`{row}`" for row in item["missing"]) if item["missing"] else "-"
        lines.append(
            f"| {name} | {item['complete']} | {item['num_runs']} | {item['num_complete']} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Run Contract",
            "",
            "| # | Run | Scope | Progress | Contract | Missing Required | Stale | Next Action |",
            "|---:|---|---|---|---|---|---|---|",
        ]
    )
    for idx, row in enumerate(report["rows"], start=1):
        missing = ", ".join(f"`{item}`" for item in row["missing_required_artifacts"]) or "-"
        stale = ", ".join(f"`{item['artifact']}:{item['reason']}`" for item in row["stale_artifacts"]) or "-"
        lines.append(
            "| {idx} | `{name}` | {scope} | `{progress}` | `{contract}` | {missing} | {stale} | {action} |".format(
                idx=idx,
                name=row["name"],
                scope=row["scope"],
                progress=row["progress_status"],
                contract=row["contract_status"],
                missing=missing,
                stale=stale,
                action=row["next_action"],
            )
        )
    lines.extend(["", "## Claim Policy", ""])
    for item in report["claim_policy"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
