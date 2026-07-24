from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.engine import _balanced_sampling_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = PROJECT_ROOT / "reports" / "paper_protocol_summary"
DEFAULT_CONFIGS = [
    "configs/experiments/paper_protocol/v14_lean_road_roi_safety.yaml",
    "configs/experiments/paper_protocol/v17_lean_quality_physics_safety.yaml",
    "configs/experiments/paper_protocol/v18_lean_mixstyle_quality_safety.yaml",
    "configs/experiments/paper_protocol/v19_lean_state_contrast_quality_safety.yaml",
    "configs/experiments/paper_protocol/v20_lean_interval_order_quality_safety.yaml",
    "configs/experiments/paper_protocol/v21_lean_quality_uncertainty_safety.yaml",
    "configs/experiments/paper_protocol/v22_lean_quality_order_contrast_safety.yaml",
    "configs/experiments/paper_protocol/v23_lean_region_mixture_evidence_safety.yaml",
]
MISSING_LABELS = {"", "nan", "none", "null", "unknown", "-1"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only audit for whether candidate losses can actually activate under "
            "the configured sampler. This helps discard routes whose loss terms are "
            "mostly empty before spending GPU time."
        )
    )
    parser.add_argument("--config", action="append", type=Path, default=None)
    parser.add_argument("--batches", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument(
        "--max-rows-per-group",
        type=int,
        default=512,
        help=(
            "Build a small audit pool by sampling at most this many rows per "
            "dataset/class group. The training sampler is group-balanced, so this "
            "keeps activation estimates fast without scanning million-row groups."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=SUMMARY_DIR / "candidate_loss_activation_audit.json")
    parser.add_argument("--md-out", type=Path, default=SUMMARY_DIR / "candidate_loss_activation_audit.md")
    args = parser.parse_args()

    configs = args.config or [PROJECT_ROOT / item for item in DEFAULT_CONFIGS]
    manifest_cache: dict[tuple[str, ...], pd.DataFrame] = {}
    report = {
        "batches": int(args.batches),
        "seed": int(args.seed),
        "claim_boundary": (
            "This audit checks batch-level availability of supervised pairs and "
            "ordered weak-friction intervals. It does not prove that a candidate "
            "will improve task metrics."
        ),
        "runs": [
            _audit_config(
                path,
                batches=args.batches,
                seed=args.seed,
                max_rows_per_group=args.max_rows_per_group,
                manifest_cache=manifest_cache,
            )
            for path in configs
        ],
    }
    report["decisions"] = [_decision(row) for row in report["runs"]]

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.md_out.write_text(_to_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


def _audit_config(
    path: Path,
    *,
    batches: int,
    seed: int,
    max_rows_per_group: int,
    manifest_cache: dict[tuple[str, ...], pd.DataFrame],
) -> dict[str, Any]:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    loss = cfg.get("loss", {})
    batch_size = int(data.get("batch_size", 16))
    train_manifests = tuple(str(item) for item in data["train_manifests"])
    df = _read_train_manifests(
        train_manifests,
        manifest_cache,
        seed=seed,
        max_rows_per_group=max_rows_per_group,
    )
    weights = _sampler_weights(df, data)
    total = max(int(batches), 1) * batch_size
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.multinomial(torch.as_tensor(weights, dtype=torch.double), total, replacement=True, generator=generator).tolist()

    rows = {
        "run": path.stem,
        "config": str(path.relative_to(PROJECT_ROOT)),
        "batch_size": batch_size,
        "train_rows": int(df.attrs.get("full_rows", len(df))),
        "audit_pool_rows": int(len(df)),
        "sampled_dataset_mass": _value_counts(df.iloc[indices]["dataset"]),
        "loss_terms_enabled": _enabled_terms(loss),
    }
    stats = _batch_stats(df, indices, batch_size)
    rows.update(stats)
    return rows


def _read_train_manifests(
    manifests: tuple[str, ...],
    manifest_cache: dict[tuple[str, ...], pd.DataFrame],
    *,
    seed: int,
    max_rows_per_group: int,
) -> pd.DataFrame:
    cache_key = (*manifests, f"max_rows_per_group={int(max_rows_per_group)}", f"seed={int(seed)}")
    if cache_key in manifest_cache:
        return manifest_cache[cache_key]
    columns = [
        "dataset",
        "class_label",
        "risk_label",
        "friction_label",
        "wetness_label",
        "snow_label",
        "mu_low",
        "mu_high",
    ]
    frames = []
    for item in manifests:
        path = Path(item)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        frames.append(pd.read_csv(path, dtype=str, usecols=lambda col: col in columns, low_memory=False))
    full_rows = sum(len(frame) for frame in frames)
    df = pd.concat(frames, ignore_index=True)
    if max_rows_per_group > 0 and {"dataset", "class_label"}.issubset(df.columns):
        sampled = []
        for _, group in df.groupby(["dataset", "class_label"], dropna=False, sort=False):
            n = min(int(max_rows_per_group), len(group))
            sampled.append(group.sample(n=n, random_state=int(seed)) if len(group) > n else group)
        df = pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    df.attrs["full_rows"] = int(full_rows)
    manifest_cache[cache_key] = df
    return df


def _sampler_weights(df: pd.DataFrame, data: dict[str, Any]) -> list[float]:
    if not bool(data.get("balanced_sampling", False)):
        return [1.0] * len(df)
    return _balanced_sampling_weights(
        df,
        data.get("balanced_group_columns", ["dataset", "class_label"]),
        dataset_first=bool(data.get("balanced_dataset_first", True)),
        overrides=data.get("balanced_weight_overrides"),
    )


def _batch_stats(df: pd.DataFrame, indices: list[int], batch_size: int) -> dict[str, Any]:
    n_batches = max(len(indices) // batch_size, 1)
    counters = {
        "multi_domain_batches": 0,
        "risk_cross_domain_positive_batches": 0,
        "friction_cross_domain_positive_batches": 0,
        "wetness_cross_domain_positive_batches": 0,
        "snow_cross_domain_positive_batches": 0,
        "interval_order_active_batches": 0,
    }
    domain_counts = []
    interval_pair_counts = []
    for start in range(0, n_batches * batch_size, batch_size):
        part = df.iloc[indices[start : start + batch_size]]
        domain_count = int(part["dataset"].astype(str).nunique()) if "dataset" in part.columns else 0
        domain_counts.append(domain_count)
        if domain_count >= 2:
            counters["multi_domain_batches"] += 1
        for column, key in [
            ("risk_label", "risk_cross_domain_positive_batches"),
            ("friction_label", "friction_cross_domain_positive_batches"),
            ("wetness_label", "wetness_cross_domain_positive_batches"),
            ("snow_label", "snow_cross_domain_positive_batches"),
        ]:
            if _has_cross_domain_positive(part, column):
                counters[key] += 1
        ordered_pairs = _interval_order_pairs(part)
        interval_pair_counts.append(ordered_pairs)
        if ordered_pairs > 0:
            counters["interval_order_active_batches"] += 1

    out = {f"{key}_rate": value / float(n_batches) for key, value in counters.items()}
    out["mean_domains_per_batch"] = sum(domain_counts) / float(n_batches)
    out["mean_interval_order_pairs_per_batch"] = sum(interval_pair_counts) / float(n_batches)
    return out


def _has_cross_domain_positive(batch: pd.DataFrame, column: str) -> bool:
    if column not in batch.columns or "dataset" not in batch.columns:
        return False
    for label, group in batch.groupby(column, dropna=False):
        if _missing_label(label):
            continue
        if len(group) >= 2 and int(group["dataset"].astype(str).nunique()) >= 2:
            return True
    return False


def _interval_order_pairs(batch: pd.DataFrame, min_gap: float = 0.02) -> int:
    lows = pd.to_numeric(batch.get("mu_low"), errors="coerce")
    highs = pd.to_numeric(batch.get("mu_high"), errors="coerce")
    if lows is None or highs is None:
        return 0
    lows = lows.to_numpy()
    highs = highs.to_numpy()
    count = 0
    for i in range(len(lows)):
        for j in range(len(lows)):
            if pd.isna(lows[j]) or pd.isna(highs[i]):
                continue
            if float(lows[j]) - float(highs[i]) >= float(min_gap):
                count += 1
    return count


def _missing_label(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_LABELS


def _value_counts(series: pd.Series) -> dict[str, float]:
    counts = series.astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def _enabled_terms(loss: dict[str, Any]) -> list[str]:
    terms = []
    for key in [
        "risk_state_contrastive_weight",
        "friction_state_contrastive_weight",
        "wetness_state_contrastive_weight",
        "interval_order_weight",
        "coverage_near_white_weight",
        "coverage_low_texture_weight",
        "coverage_specular_weight",
        "risk_conditional_coral_weight",
        "wetness_conditional_coral_weight",
        "aug_consistency_weight",
    ]:
        try:
            if float(loss.get(key, 0.0)) > 0:
                terms.append(key)
        except (TypeError, ValueError):
            pass
    return terms


def _decision(row: dict[str, Any]) -> dict[str, str]:
    enabled = set(row.get("loss_terms_enabled") or [])
    problems = []
    if enabled & {
        "risk_state_contrastive_weight",
        "friction_state_contrastive_weight",
        "wetness_state_contrastive_weight",
    }:
        if float(row.get("risk_cross_domain_positive_batches_rate", 0.0)) < 0.35:
            problems.append("risk cross-domain positives are sparse")
        if float(row.get("friction_cross_domain_positive_batches_rate", 0.0)) < 0.25:
            problems.append("friction cross-domain positives are sparse")
    if "interval_order_weight" in enabled:
        if float(row.get("interval_order_active_batches_rate", 0.0)) < 0.65:
            problems.append("interval-order pairs are sparse")
    if float(row.get("multi_domain_batches_rate", 0.0)) < 0.90:
        problems.append("sampler does not consistently create multi-domain batches")

    if problems:
        status = "revise_or_drop_before_gpu"
        action = "; ".join(problems)
    else:
        status = "loss_activation_ok_for_fast_screen"
        action = "The configured sampler provides enough batch-level signal for the enabled candidate losses."
    return {"run": str(row.get("run")), "status": status, "action": action}


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Loss Activation Audit",
        "",
        f"Batches simulated per run: `{report['batches']}`.",
        "",
        f"Boundary: {report['claim_boundary']}",
        "",
        "## Activation Table",
        "",
        "| run | terms | dataset mass | multi-domain | risk pos | friction pos | wetness pos | interval order | ordered pairs/batch | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    decisions = {row["run"]: row for row in report.get("decisions", [])}
    for row in report.get("runs", []):
        decision = decisions.get(row["run"], {})
        mass = ", ".join(f"{k}:{v:.2f}" for k, v in sorted((row.get("sampled_dataset_mass") or {}).items()))
        terms = ", ".join(row.get("loss_terms_enabled") or ["none"])
        lines.append(
            "| {run} | {terms} | {mass} | {multi:.1%} | {risk:.1%} | {fric:.1%} | {wet:.1%} | {order:.1%} | {pairs:.1f} | {status} |".format(
                run=row["run"],
                terms=terms,
                mass=mass,
                multi=float(row.get("multi_domain_batches_rate", 0.0)),
                risk=float(row.get("risk_cross_domain_positive_batches_rate", 0.0)),
                fric=float(row.get("friction_cross_domain_positive_batches_rate", 0.0)),
                wet=float(row.get("wetness_cross_domain_positive_batches_rate", 0.0)),
                order=float(row.get("interval_order_active_batches_rate", 0.0)),
                pairs=float(row.get("mean_interval_order_pairs_per_batch", 0.0)),
                status=decision.get("status", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## Route Rule",
            "",
            "- If a candidate loss is mostly inactive under its sampler, revise or drop it before GPU training.",
            "- Passing this audit only means the loss has usable batch signal; promotion still depends on fast-screen task metrics, coverage, dataset-ID leakage, and quality slices.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
