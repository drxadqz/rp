from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "next_queue_readiness_report.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "next_queue_readiness_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a reviewer-facing readiness and pruning report for the next "
            "formal runs in the paper protocol queue."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--limit", type=int, default=18)
    args = parser.parse_args()

    report = build_report(args.summary_dir, limit=args.limit)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path, limit: int = 18) -> dict[str, Any]:
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    protocol = _load_json(summary_dir / "protocol_config_audit.json") or {}
    gpu = _load_json(summary_dir / "gpu_protocol_audit.json") or {}
    modules = _load_json(summary_dir / "algorithm_module_audit.json") or {}
    cv_playbook = _load_json(summary_dir / "cv_subfield_migration_playbook.json") or {}
    retention = _load_json(summary_dir / "module_retention_report.json") or {}

    protocol_by_run = {str(row.get("run")): row for row in protocol.get("runs", []) if isinstance(row, dict)}
    gpu_by_run = {str(row.get("run")): row for row in gpu.get("rows", []) if isinstance(row, dict)}
    module_by_run = {str(row.get("run")): row for row in modules.get("rows", []) if isinstance(row, dict)}

    queue_rows = queue.get("queue_order", []) if isinstance(queue.get("queue_order"), list) else []
    incomplete = [row for row in queue_rows if row.get("status") != "complete"]
    selected = incomplete[: max(int(limit), 1)]
    rows = [
        _inspect_run(
            idx=idx,
            row=row,
            protocol=protocol_by_run.get(str(row.get("name")), {}),
            gpu=gpu_by_run.get(str(row.get("name")), {}),
            modules=module_by_run.get(str(row.get("name")), {}),
        )
        for idx, row in enumerate(selected, start=1)
    ]
    blocks = _blocks(rows, protocol, gpu)
    verdict = "ready_waiting_for_queue" if not blocks else "readiness_gaps_present"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "active": queue.get("active_rows", []),
        "queue_counts": {
            "total": queue.get("num_total"),
            "complete": queue.get("num_complete"),
            "partial": queue.get("num_partial"),
            "missing": queue.get("num_missing"),
        },
        "protocol_verdict": protocol.get("verdict"),
        "gpu_verdict": gpu.get("verdict"),
        "cv_route_verdict": cv_playbook.get("verdict"),
        "module_retention_verdict": retention.get("verdict"),
        "blocks": blocks,
        "rows": rows,
        "pruning_policy": pruning_policy(),
    }


def _inspect_run(
    *,
    idx: int,
    row: dict[str, Any],
    protocol: dict[str, Any],
    gpu: dict[str, Any],
    modules: dict[str, Any],
) -> dict[str, Any]:
    name = str(row.get("name"))
    role = _role(name)
    module_flags = modules.get("modules", {}) if isinstance(modules.get("modules"), dict) else {}
    train_datasets = protocol.get("train_datasets", [])
    test_datasets = protocol.get("test_datasets", [])
    gpu_ready = bool(gpu) and int(gpu.get("effective_batch", 0) or 0) > 0 and bool(gpu.get("amp", False))
    protocol_ready = bool(protocol) and bool(protocol.get("train_manifests")) and bool(protocol.get("test_manifests"))
    module_ready = bool(modules) and modules.get("status") == "ok"
    risks = _risk_notes(name, module_flags, train_datasets, test_datasets)
    return {
        "queue_index": idx,
        "run": name,
        "status": row.get("status"),
        "progress": _progress(row),
        "role": role["role"],
        "cv_transfer": role["cv_transfer"],
        "claim_use": role["claim_use"],
        "protocol_ready": protocol_ready,
        "gpu_ready": gpu_ready,
        "module_ready": module_ready,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "batch_size": gpu.get("batch_size"),
        "grad_accum_steps": gpu.get("grad_accum_steps"),
        "effective_batch": gpu.get("effective_batch"),
        "samples_per_epoch": gpu.get("balanced_num_samples_per_epoch"),
        "key_modules": _key_modules(module_flags),
        "risk_notes": risks,
        "decision_rule": role["decision_rule"],
    }


def _role(name: str) -> dict[str, str]:
    if name.startswith("baseline_single_"):
        dataset = name.replace("baseline_single_", "").replace("_global_convnext", "")
        return {
            "role": "fair_single_dataset_baseline",
            "cv_transfer": "none_baseline",
            "claim_use": f"Required fair comparator for the {dataset} single-dataset FAF result.",
            "decision_rule": "Must complete before any claim that FAF beats a standard ConvNeXt image-level baseline.",
        }
    if name.startswith("v6") or name.startswith("v15") or name.startswith("v16") or name.startswith("v18"):
        return {
            "role": "style_shortcut_suppression_candidate",
            "cv_transfer": "domain_adaptive_segmentation_style_robustness",
            "claim_use": "Tests whether style/camera shortcuts can be reduced without erasing road-state signal.",
            "decision_rule": "Keep only if dataset-ID shortcut drops while risk F1, low-friction recall, and worst-dataset F1 do not regress.",
        }
    if name.startswith("v7"):
        return {
            "role": "domain_adversarial_candidate",
            "cv_transfer": "domain_adversarial_learning",
            "claim_use": "Tests a stronger anti-domain-shortcut loss after Fourier augmentation.",
            "decision_rule": "Prune fast if safety metrics regress like earlier DG losses or dataset-ID probe stays high.",
        }
    if name.startswith("v8") or name.startswith("v12") or name.startswith("v14"):
        return {
            "role": "road_roi_attention_candidate",
            "cv_transfer": "segmentation_roi_prior",
            "claim_use": "Tests whether EvidenceField attention should be constrained to plausible road/contact regions.",
            "decision_rule": "Keep only if attention-on-road and conditional interval coverage improve without low-friction recall loss.",
        }
    if name.startswith("v9"):
        return {
            "role": "roadsaw_hard_case_candidate",
            "cv_transfer": "rare_condition_sampling",
            "claim_use": "Targets damp/wet/very-wet RoadSaW confusion and near-white slices.",
            "decision_rule": "Keep only if RoadSaW wet-state F1 or low-friction recall improves without overfitting RoadSaW.",
        }
    if name.startswith("v10"):
        return {
            "role": "weak_view_consistency_candidate",
            "cv_transfer": "semi_supervised_segmentation_consistency",
            "claim_use": "Tests weak-to-strong consistency for logits, intervals, and attention.",
            "decision_rule": "Keep only if robustness improves without oversmoothing hard wet/snow states.",
        }
    if name.startswith("v11"):
        return {
            "role": "domain_adapter_candidate",
            "cv_transfer": "shared_semantics_with_domain_style_adapter",
            "claim_use": "Allows small dataset-style offsets while preserving shared friction semantics.",
            "decision_rule": "Keep only if single-dataset and LODO evidence both improve or shortcut drops cleanly.",
        }
    if name.startswith("v13"):
        return {
            "role": "lean_core_candidate",
            "cv_transfer": "module_pruning",
            "claim_use": "Tests whether removing unstable FrictionSet/DG components improves the final route.",
            "decision_rule": "Keep if it matches or beats full model with fewer modules and better safety/generalization.",
        }
    if name.startswith("v17") or name.startswith("v21"):
        return {
            "role": "visual_quality_uncertainty_candidate",
            "cv_transfer": "material_texture_physical_vision_uncertainty",
            "claim_use": "Targets wet, bright, specular, low-texture, and snowy visual ambiguity.",
            "decision_rule": "Keep if quality-slice coverage improves at bounded interval width and low-friction recall is preserved.",
        }
    if name.startswith("v19") or name.startswith("v20") or name.startswith("v22"):
        return {
            "role": "ordered_state_alignment_candidate",
            "cv_transfer": "ordinal_learning_and_cross_domain_contrast",
            "claim_use": "Tests whether weak friction ordering and state alignment help generalization.",
            "decision_rule": "Keep only if worst-domain behavior or low-friction recall improves without interval-width inflation.",
        }
    if name.startswith("v23"):
        return {
            "role": "segmentation_region_mixture_candidate",
            "cv_transfer": "semantic_segmentation_region_reasoning",
            "claim_use": "Main segmentation-transfer route: local material-mixture evidence expands uncertainty where the road surface is spatially mixed.",
            "decision_rule": "Compare against v21/v22; keep only if wet/snow/near-white slices improve at bounded width.",
        }
    if name.startswith("final_"):
        return {
            "role": "final_method_verification",
            "cv_transfer": "selected_final_route",
            "claim_use": "Required final evidence after module pruning.",
            "decision_rule": "Run only for the selected final method and report both matched single-dataset and LODO results.",
        }
    return {
        "role": "active_or_general_protocol_run",
        "cv_transfer": "current_protocol",
        "claim_use": "Protocol continuation.",
        "decision_rule": "Complete and postprocess before making final claims.",
    }


def _risk_notes(
    name: str,
    modules: dict[str, Any],
    train_datasets: Any,
    test_datasets: Any,
) -> list[str]:
    notes: list[str] = []
    if modules.get("dann"):
        notes.append("DANN can erase real road-state differences; compare safety metrics before retaining.")
    if modules.get("friction_set"):
        notes.append("FrictionSet previously hurt worst-dataset F1; requires rescue evidence.")
    if modules.get("region_mixture_evidence"):
        notes.append("Region-mixture must improve conditional slices, not merely widen all intervals.")
    if modules.get("visual_quality_weighted_coverage"):
        notes.append("Visual-quality weighting must not become a dataset-brightness shortcut.")
    if modules.get("feature_mixstyle"):
        notes.append("MixStyle should reduce style shortcut without damaging low-friction recall.")
    if isinstance(train_datasets, list) and isinstance(test_datasets, list):
        if set(train_datasets).isdisjoint(set(test_datasets)) and name.startswith("final_lodo"):
            notes.append("LODO row is a stress test; poor result should be reported as generalization failure evidence.")
    if name.startswith("baseline_single_"):
        notes.append("Baseline must use same split and metric as corresponding single-dataset FAF run.")
    return notes


def _key_modules(modules: dict[str, Any]) -> list[str]:
    order = [
        "physics_texture",
        "physics_quality_cues",
        "evidence_field",
        "evidence_final_mix",
        "road_likelihood_prior",
        "region_mixture_evidence",
        "pseudo_road_mask_supervision",
        "roi_attention_constraint",
        "weak_view_consistency",
        "mask_aware_consistency",
        "fourier_style_jitter",
        "bottom_square_input_canonicalization",
        "gray_world_color_constancy",
        "feature_mixstyle",
        "dann",
        "domain_adapter",
        "interval_order_consistency",
        "state_contrastive_alignment",
        "visual_quality_weighted_coverage",
        "safety_weighted_coverage",
        "wetness_ordinal_loss",
    ]
    return [name for name in order if modules.get(name)]


def _progress(row: dict[str, Any]) -> str:
    if row.get("active_epoch") is not None:
        return f"epoch {row.get('active_epoch')}/{row.get('active_epochs')}, {row.get('active_phase')} {row.get('active_step')}/{row.get('active_steps')}"
    if row.get("epoch") is not None:
        return f"epoch {row.get('epoch')}/{row.get('epochs')}"
    return "-"


def _blocks(rows: list[dict[str, Any]], protocol: dict[str, Any], gpu: dict[str, Any]) -> list[str]:
    blocks = []
    if protocol.get("verdict") != "pass":
        blocks.append("protocol_config_audit_not_pass")
    if gpu.get("verdict") != "pass":
        blocks.append("gpu_protocol_audit_not_pass")
    for row in rows:
        if not row["protocol_ready"]:
            blocks.append(f"{row['run']}:missing_protocol_config")
        if not row["gpu_ready"]:
            blocks.append(f"{row['run']}:missing_gpu_envelope")
        if not row["module_ready"]:
            blocks.append(f"{row['run']}:missing_module_audit")
    return blocks


def pruning_policy() -> list[dict[str, str]]:
    return [
        {
            "scope": "fair baselines",
            "keep_condition": "Same split, same metric, same compute envelope; complete before superiority claims.",
            "drop_condition": "Never drop; these are required controls.",
        },
        {
            "scope": "style/domain candidates",
            "keep_condition": "Dataset-ID probe decreases and risk F1, low-friction recall, worst-dataset F1 do not regress.",
            "drop_condition": "Shortcut remains high or safety metrics collapse.",
        },
        {
            "scope": "EvidenceField/ROI/mask candidates",
            "keep_condition": "Attention diagnostics improve and conditional coverage-width tradeoff improves.",
            "drop_condition": "Only visual maps look nicer, or intervals widen without conditional gains.",
        },
        {
            "scope": "visual-quality uncertainty candidates",
            "keep_condition": "RoadSaW near-white/wet and RoadSC low-texture snow slices improve at bounded width.",
            "drop_condition": "Model learns brightness as dataset ID or coverage improves only by broadening all intervals.",
        },
        {
            "scope": "ordinal/order/contrast candidates",
            "keep_condition": "Worst-domain or low-friction recall improves without width inflation.",
            "drop_condition": "Noisy weak intervals over-constrain the model.",
        },
        {
            "scope": "final method",
            "keep_condition": "Beats matched ConvNeXt on single-dataset tests and has honest LODO stress-test evidence.",
            "drop_condition": "If LODO fails, report the failure and restrict claims to single-dataset or revise algorithm.",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Next Queue Readiness Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        f"Protocol config audit: `{report.get('protocol_verdict')}`; GPU protocol audit: `{report.get('gpu_verdict')}`",
        f"CV route verdict: `{report.get('cv_route_verdict')}`; module retention: `{report.get('module_retention_verdict')}`",
        "",
        "## Queue Counts",
        "",
        "| Total | Complete | Running/partial | Missing |",
        "|---:|---:|---:|---:|",
        "| {total} | {complete} | {partial} | {missing} |".format(**report["queue_counts"]),
        "",
        "## Readiness Rows",
        "",
        "| # | Run | Status | Role | Train -> Test | GPU envelope | Ready | Key modules | Decision rule |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        ready = "yes" if row["protocol_ready"] and row["gpu_ready"] and row["module_ready"] else "no"
        lines.append(
            "| {idx} | `{run}` | `{status}` | {role} | {train} -> {test} | bs={bs}, accum={accum}, eff={eff}, n/epoch={n} | {ready} | {mods} | {rule} |".format(
                idx=row["queue_index"],
                run=row["run"],
                status=row["status"],
                role=row["role"],
                train=",".join(row.get("train_datasets") or ["-"]),
                test=",".join(row.get("test_datasets") or ["-"]),
                bs=row.get("batch_size", "-"),
                accum=row.get("grad_accum_steps", "-"),
                eff=row.get("effective_batch", "-"),
                n=row.get("samples_per_epoch", "-"),
                ready=ready,
                mods=", ".join(row.get("key_modules") or ["none"]),
                rule=row["decision_rule"],
            )
        )
    lines.extend(["", "## Risk Notes", ""])
    for row in report["rows"]:
        if row.get("risk_notes"):
            for note in row["risk_notes"]:
                lines.append(f"- `{row['run']}`: {note}")
    if report["blocks"]:
        lines.extend(["", "## Blocks", ""])
        lines.extend(f"- `{item}`" for item in report["blocks"])
    lines.extend(["", "## Pruning Policy", ""])
    lines.extend(["| Scope | Keep condition | Drop condition |", "|---|---|---|"])
    for item in report["pruning_policy"]:
        lines.append(
            f"| {item['scope']} | {item['keep_condition']} | {item['drop_condition']} |"
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    main()
