from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = Path("reports/paper_protocol_summary/formal_candidate_protocol_controls_20260709")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_formal_candidate_queue import CANDIDATES  # noqa: E402
from friction_affordance.c3_experiment import load_config  # noqa: E402


CONTROL_OF = {
    "S7": "current formal baseline / anchor control",
    "S12": "mechanism candidate; same-budget control is S11, isolating factor-graph pair sampling",
    "S13": "mechanism candidate; control is S12, isolating early family-router tensor coupling",
    "S14": "mechanism candidate; control is S13, isolating RSCD-specific PCGrad/no-harm protection on top of early family-router coupling",
    "S11": "mechanism candidate; control is S7, isolating factor-graph metric loss",
    "S8": "boundary-route candidate; control is S7, isolating WCS incoming route",
    "S9": "boundary-route candidate; control is S7, isolating dry-concrete ordinal + WCS routes",
    "S10": "boundary-route candidate; control is S7, isolating graph-control stable route set",
}


def _noneish(value: Any) -> bool:
    return value in (None, "", "null", "None")


def _list_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return str(value)


def _cfg_row(candidate: dict[str, str]) -> dict[str, Any]:
    config_path = ROOT / candidate["config"]
    run_script = ROOT / candidate["run_script"]
    cfg = load_config(config_path)
    train = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    loss = cfg.get("loss", {}) or {}
    model = cfg.get("model", {}) or {}

    full_train = train.get("samples_per_epoch") in (0, None)
    full_test = _noneish(eval_cfg.get("max_test_samples_per_class"))
    full_val = _noneish(eval_cfg.get("max_val_samples_per_class"))
    no_tta_or_ensemble = not bool(eval_cfg.get("tta", False)) and not bool(eval_cfg.get("ensemble", False))
    fair_full_protocol = bool(full_train and full_test and no_tta_or_ensemble)

    trainable_prefixes = train.get("trainable_prefixes", [])
    return {
        "id": candidate["id"],
        "name": candidate["name"],
        "role": candidate["role"],
        "control_relation": CONTROL_OF.get(candidate["id"], ""),
        "config": candidate["config"],
        "run_script": candidate["run_script"],
        "config_exists": config_path.exists(),
        "run_script_exists": run_script.exists(),
        "output_dir": cfg.get("output_dir", ""),
        "resume_from": train.get("resume_from", ""),
        "teacher_checkpoint": train.get("teacher_checkpoint", ""),
        "epochs": train.get("epochs"),
        "batch_size": train.get("batch_size"),
        "grad_accum_steps": train.get("grad_accum_steps"),
        "lr": train.get("lr"),
        "weight_decay": train.get("weight_decay"),
        "samples_per_epoch": train.get("samples_per_epoch"),
        "max_test_samples_per_class": eval_cfg.get("max_test_samples_per_class"),
        "max_val_samples_per_class": eval_cfg.get("max_val_samples_per_class"),
        "full_train": full_train,
        "full_test": full_test,
        "full_val": full_val,
        "no_tta_or_ensemble": no_tta_or_ensemble,
        "fair_full_protocol": fair_full_protocol,
        "trainable_prefixes": _list_text(trainable_prefixes),
        "backbone": model.get("backbone", ""),
        "family_mechanism_router_weight": loss.get("family_mechanism_router_weight", 0.0),
        "factor_graph_metric_weight": loss.get("factor_graph_metric_weight", 0.0),
        "factor_graph_pair_sampling": train.get("factor_graph_pair_sampling", False),
        "rscd_pcgrad_enabled": loss.get("rscd_pcgrad_enabled", False),
        "anchor_nonregression_weight": loss.get("anchor_nonregression_weight", 0.0),
        "source_reliable_boundary_routes": len(model.get("source_reliable_boundary_routes", []) or []),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_cfg_row(candidate) for candidate in CANDIDATES]
    _write_csv(SUMMARY_DIR / "formal_candidate_protocol_controls.csv", rows)
    data = {
        "all_fair_full_protocol": all(bool(row["fair_full_protocol"]) for row in rows),
        "candidate_count": len(rows),
        "rows": rows,
    }
    (SUMMARY_DIR / "formal_candidate_protocol_controls.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# Formal Candidate Protocol and Control Audit",
        "",
        f"- Candidate count: {len(rows)}",
        f"- All candidates full/fair protocol: {data['all_fair_full_protocol']}",
        "",
        "| id | fair full protocol | control relation | epochs | batch x accum | LR | trainable prefixes | key mechanism |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        key_mechanism = []
        if float(row["factor_graph_metric_weight"] or 0.0) > 0:
            key_mechanism.append(f"factor_graph_metric={row['factor_graph_metric_weight']}")
        if bool(row["factor_graph_pair_sampling"]):
            key_mechanism.append("factor_graph_pair_sampling")
        if float(row["family_mechanism_router_weight"] or 0.0) > 0:
            key_mechanism.append(f"family_router={row['family_mechanism_router_weight']}")
        if bool(row.get("rscd_pcgrad_enabled")):
            key_mechanism.append("RSCD-PCGrad no-harm")
        if float(row.get("anchor_nonregression_weight") or 0.0) > 0:
            key_mechanism.append(f"nonregression={row['anchor_nonregression_weight']}")
        if "family_router" in str(row["backbone"]):
            key_mechanism.append(str(row["backbone"]))
        routes = int(row["source_reliable_boundary_routes"] or 0)
        if routes:
            key_mechanism.append(f"SRBR_routes={routes}")
        md.append(
            "| {id} | {fair} | {control} | {epochs} | {batch}x{accum} | {lr} | {prefixes} | {mechanism} |".format(
                id=row["id"],
                fair=row["fair_full_protocol"],
                control=row["control_relation"],
                epochs=row["epochs"],
                batch=row["batch_size"],
                accum=row["grad_accum_steps"],
                lr=row["lr"],
                prefixes=row["trainable_prefixes"],
                mechanism=", ".join(key_mechanism),
            )
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `S12` is paired with `S11` as its same-budget control. The isolated mechanism is graph-neighbor batch construction over one-axis RSCD factor contrasts.",
            "- `S13` is paired with `S12` as its control. The isolated mechanism is early family-router tensor coupling before final classification.",
            "- `S14` is paired with `S13` as its control. The isolated mechanism is RSCD-specific PCGrad/no-harm protection for teacher-correct stable classes while retaining early family-router coupling.",
            "- `S11` is paired with `S7` to isolate factor-graph metric representation shaping.",
            "- `S8`, `S9`, and `S10` are SRBR route-set candidates controlled against `S7`.",
            "- A candidate remains only a queued hypothesis until its complete full-test artifacts and promotion gate exist.",
        ]
    )
    (SUMMARY_DIR / "formal_candidate_protocol_controls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out_dir": str(SUMMARY_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
