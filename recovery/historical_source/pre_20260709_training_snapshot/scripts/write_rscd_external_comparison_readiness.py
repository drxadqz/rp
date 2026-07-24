from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "rscd_external_comparison_readiness.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "rscd_external_comparison_readiness.json"
DEFAULT_TRAIN = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_VAL = Path("data/manifests_full/rscd_prepared_val.csv")
DEFAULT_TEST = Path("data/manifests_full/rscd_prepared_test.csv")
DEFAULT_FAST_OUT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\fast_convnext_tiny")
DEFAULT_FORMAL_OUT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\formal_convnext_tiny")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the local RSCD-27 class-label protocol is ready for "
            "fair RSCD-style external comparison, distinct from the weak "
            "friction-affordance interval protocol."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--fast-output-dir", type=Path, default=DEFAULT_FAST_OUT)
    parser.add_argument("--formal-output-dir", type=Path, default=DEFAULT_FORMAL_OUT)
    args = parser.parse_args()

    report = build_report(args)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {
        "train": summarize_manifest(args.train_manifest),
        "val": summarize_manifest(args.val_manifest),
        "test": summarize_manifest(args.test_manifest),
    }
    runner = Path("scripts/run_rscd_surface_classification.py")
    fast = summarize_result(args.fast_output_dir)
    formal = summarize_result(args.formal_output_dir)
    manifest_ok = all(item.get("exists") and item.get("classes") == 27 for item in manifests.values())
    runner_ok = runner.exists()
    has_formal_result = formal.get("status") == "complete"
    has_fast_result = fast.get("status") == "complete"
    if manifest_ok and runner_ok and has_formal_result:
        verdict = "formal_result_ready_for_local_rscd_context"
    elif manifest_ok and runner_ok:
        verdict = "protocol_ready_results_pending"
    else:
        verdict = "protocol_incomplete"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "claim_boundary": (
            "RSCD-27 class-label classification is a separate protocol for "
            "RSCD-style road-surface classification papers. It is not the main "
            "weak friction/risk/interval affordance protocol and does not prove "
            "measured tire-road friction estimation."
        ),
        "manifests": manifests,
        "runner": {
            "path": str(runner),
            "exists": runner_ok,
            "default_model": "ConvNeXt-Tiny + 27-way class head",
            "metrics": ["top1", "macro_f1", "weighted_f1", "balanced_accuracy"],
        },
        "results": {
            "fast": fast,
            "formal": formal,
        },
        "external_references": external_references(),
        "comparison_rules": comparison_rules(),
        "commands": commands(args.fast_output_dir, args.formal_output_dir),
        "decision": decision(verdict, formal),
    }


def summarize_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path, usecols=["image_path", "class_label"], dtype=str, low_memory=False)
    canonical = df["class_label"].map(canonical_class_label)
    counts = canonical.value_counts().sort_index()
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "classes": int(counts.size),
        "min_class_rows": int(counts.min()) if not counts.empty else 0,
        "max_class_rows": int(counts.max()) if not counts.empty else 0,
        "labels": counts.index.tolist(),
    }


def summarize_result(output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "evaluate_test.json"
    protocol_path = output_dir / "protocol.json"
    if not result_path.exists():
        return {
            "status": "missing",
            "output_dir": str(output_dir),
            "evaluate_test": str(result_path),
            "protocol": str(protocol_path),
        }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unreadable",
            "output_dir": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "top1": _float(summary.get("top1")),
        "macro_f1": _float(summary.get("macro_f1")),
        "weighted_f1": _float(summary.get("weighted_f1")),
        "balanced_accuracy": _float(summary.get("balanced_accuracy")),
        "num_samples": summary.get("num_samples"),
        "num_classes": summary.get("num_classes"),
        "sota_gap_top1_vs_roadformer": _delta(_float(summary.get("top1")), 0.9252),
        "sota_gap_macro_f1_vs_roadformer": _delta(_float(summary.get("macro_f1")), 0.8442),
    }


def external_references() -> list[dict[str, Any]]:
    return [
        {
            "name": "RSCD official page",
            "url": "https://thu-rsxd.com/rscd/",
            "relevance": "Defines RSCD as a public 1M-sample road-surface classification dataset with material, friction, and unevenness labels.",
            "numeric_role": "data_provenance",
        },
        {
            "name": "RSCD Data in Brief dataset paper",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9343931/",
            "relevance": "Documents bonnet-mounted camera acquisition and cropped road-surface patches.",
            "numeric_role": "data_provenance",
        },
        {
            "name": "RoadFormer",
            "url": "https://arxiv.org/abs/2506.02358",
            "reported": "RSCD Top-1 92.52%; Mean-F1 84.42%; simple-RSCD Top-1 96.50%.",
            "relevance": "Recent RSCD-style local-global feature fusion reference.",
            "numeric_role": "context_or_reimplementation_target",
        },
        {
            "name": "A Comprehensive Implementation of Road Surface Classification for Vehicle Driving Assistance",
            "url": "https://doi.org/10.1109/TITS.2023.3264588",
            "reported": "Original RSCD/T-ITS road-surface classification and fusion reference.",
            "relevance": "Useful for dataset/model context, but numeric comparison requires exact split/label/metric matching.",
            "numeric_role": "context_or_protocol_target",
        },
    ]


def comparison_rules() -> list[dict[str, str]]:
    return [
        {
            "rule": "same_label_space",
            "requirement": "Use canonical 27-class RSCD class_label targets, not friction/risk interval targets.",
        },
        {
            "rule": "same_split",
            "requirement": "External numeric comparison is fair only when the train/val/test split is identical or exactly reproduced.",
        },
        {
            "rule": "same_metric",
            "requirement": "Report Top-1, macro/mean-F1, weighted-F1, balanced accuracy, and per-class confusion in the RSCD-27 protocol.",
        },
        {
            "rule": "same_preprocessing_boundary",
            "requirement": "State image size, letterbox/crop policy, label canonicalization, backbone, and pretrained weights.",
        },
    ]


def commands(fast_output_dir: Path, formal_output_dir: Path) -> dict[str, str]:
    python = r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
    return {
        "fast_check": (
            f"{python} scripts\\run_rscd_surface_classification.py "
            f"--output-dir {fast_output_dir} --epochs 4 --samples-per-epoch 10800 "
            "--max-train-samples-per-class 600 --max-val-samples-per-class 200 "
            "--max-test-samples-per-class 300"
        ),
        "formal": (
            f"{python} scripts\\run_rscd_surface_classification.py "
            f"--output-dir {formal_output_dir} --epochs 20 --samples-per-epoch 36000"
        ),
        "formal_eval_only": (
            f"{python} scripts\\run_rscd_surface_classification.py "
            f"--output-dir {formal_output_dir} --eval-only"
        ),
    }


def decision(verdict: str, formal: dict[str, Any]) -> dict[str, Any]:
    if verdict == "formal_result_ready_for_local_rscd_context":
        top_gap = formal.get("sota_gap_top1_vs_roadformer")
        f1_gap = formal.get("sota_gap_macro_f1_vs_roadformer")
        return {
            "allowed_claim": "Local RSCD-27 class-label comparison may be reported as secondary context under the documented local protocol.",
            "sota_claim": (
                "possible_only_if_split_label_metric_match_external_protocol"
                if top_gap is not None and f1_gap is not None and top_gap >= 0 and f1_gap >= 0
                else "not_supported_by_current_local_result"
            ),
            "next_action": "If a numeric RoadFormer-style claim is desired, reproduce the exact external split/metric or clearly label this as local-protocol context.",
        }
    if verdict == "protocol_ready_results_pending":
        return {
            "allowed_claim": "Protocol is implemented, but no RSCD-27 local result can be claimed yet.",
            "sota_claim": "not_allowed_until_result_exists_and_protocol_matches",
            "next_action": "Run fast check after the current GPU queue is idle, then the formal RSCD-27 ConvNeXt baseline.",
        }
    return {
        "allowed_claim": "No RSCD external-comparison claim is allowed.",
        "sota_claim": "not_allowed",
        "next_action": "Fix missing manifests or runner before scheduling RSCD-27 experiments.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RSCD External Comparison Readiness",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Local RSCD-27 Protocol",
        "",
        "| Split | Rows | Classes | Min/class | Max/class | Manifest |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split, item in report["manifests"].items():
        lines.append(
            "| {split} | {rows} | {classes} | {min_rows} | {max_rows} | `{path}` |".format(
                split=split,
                rows=item.get("rows", "-"),
                classes=item.get("classes", "-"),
                min_rows=item.get("min_class_rows", "-"),
                max_rows=item.get("max_class_rows", "-"),
                path=item.get("path", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## Runner",
            "",
            f"- Path: `{report['runner']['path']}`",
            f"- Exists: `{report['runner']['exists']}`",
            f"- Model: {report['runner']['default_model']}",
            f"- Metrics: {', '.join(report['runner']['metrics'])}",
            "",
            "## Results",
            "",
            "| Row | Status | Top-1 | Macro-F1 | Weighted-F1 | Balanced Acc | RoadFormer Top-1 Gap | RoadFormer Macro-F1 Gap |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in report["results"].items():
        lines.append(
            "| {key} | `{status}` | {top1} | {macro} | {weighted} | {bal} | {gap_top} | {gap_f1} |".format(
                key=key,
                status=item.get("status", "-"),
                top1=_pct(item.get("top1")),
                macro=_pct(item.get("macro_f1")),
                weighted=_pct(item.get("weighted_f1")),
                bal=_pct(item.get("balanced_accuracy")),
                gap_top=_pct(item.get("sota_gap_top1_vs_roadformer"), signed=True),
                gap_f1=_pct(item.get("sota_gap_macro_f1_vs_roadformer"), signed=True),
            )
        )
    lines.extend(["", "## External References", ""])
    lines.extend(["| Source | Role | Link |", "|---|---|---|"])
    for source in report["external_references"]:
        lines.append(
            f"| {source['name']} | {source.get('numeric_role', '-')} | {source['url']} |"
        )
    lines.extend(["", "## Comparison Rules", ""])
    for item in report["comparison_rules"]:
        lines.append(f"- `{item['rule']}`: {item['requirement']}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Allowed claim: {report['decision']['allowed_claim']}",
            f"- SOTA claim: `{report['decision']['sota_claim']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
            "## Commands",
            "",
        ]
    )
    for name, command in report["commands"].items():
        lines.append(f"- `{name}`: `{command}`")
    return "\n".join(lines) + "\n"


def canonical_class_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(value: float | None, reference: float) -> float | None:
    if value is None:
        return None
    return float(value) - float(reference)


def _pct(value: Any, *, signed: bool = False) -> str:
    number = _float(value)
    if number is None:
        return "-"
    sign = "+" if signed and number >= 0 else ""
    return f"{sign}{number * 100:.2f}"


if __name__ == "__main__":
    main()
