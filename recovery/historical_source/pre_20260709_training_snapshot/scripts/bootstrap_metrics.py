from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

import evaluate_detailed
from friction_affordance.engine import build_model
from friction_affordance.ontology import RISK
from friction_affordance.utils import load_yaml, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--num-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--min-group-calibration", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="Override config device, e.g. cpu for smoke checks.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    # Bootstrap is already memory-heavy because it keeps test/calibration
    # predictions in RAM. On Windows, worker-spawned Torch imports can exhaust
    # the page file, so keep this evaluation single-process.
    data_cfg = cfg.setdefault("data", {})
    data_cfg["num_workers"] = 0 if args.num_workers is None else int(args.num_workers)
    if args.max_val_samples is not None:
        data_cfg["max_val_samples"] = int(args.max_val_samples)
    if args.max_test_samples is not None:
        data_cfg["max_test_samples"] = int(args.max_test_samples)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loader = evaluate_detailed._build_loader(cfg, args.split)
    test_raw = evaluate_detailed._collect(model, test_loader, device, progress_label=args.split)
    calib_loader = evaluate_detailed._build_loader(cfg, "val")
    calib_raw = evaluate_detailed._collect(model, calib_loader, device, progress_label="val")

    report = build_report(
        test_raw,
        calib_raw,
        num_bootstrap=max(1, int(args.num_bootstrap)),
        seed=int(args.seed),
        target_coverage=float(args.target_coverage),
        min_group_calibration=int(args.min_group_calibration),
    )
    report["checkpoint"] = str(args.checkpoint)
    report["config"] = str(args.config)
    report["split"] = args.split

    out_json = args.out_json or args.checkpoint.parent / "bootstrap_metrics.json"
    out_md = args.out_md or args.checkpoint.parent / "bootstrap_metrics.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote: {out_json}")


def build_report(
    test_raw: dict[str, Any],
    calib_raw: dict[str, Any],
    *,
    num_bootstrap: int,
    seed: int,
    target_coverage: float,
    min_group_calibration: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "num_bootstrap": int(num_bootstrap),
        "seed": int(seed),
        "target_coverage": float(target_coverage),
        "min_group_calibration": int(min_group_calibration),
        "classification": {},
        "low_friction_detection": {},
        "mu_interval": {},
    }

    for task in ["friction", "risk"]:
        raw = test_raw["tasks"][task]
        y_true = np.asarray(raw["y_true"], dtype=int)
        y_pred = np.asarray(raw["y_pred"], dtype=int)
        dataset = np.asarray(raw["dataset"])
        if len(y_true) == 0:
            continue
        report["classification"][task] = {
            "macro_f1": _bootstrap_metric(
                len(y_true),
                lambda idx, yt=y_true, yp=y_pred: _macro_f1(yt[idx], yp[idx]),
                lambda yt=y_true, yp=y_pred: _macro_f1(yt, yp),
                rng,
                num_bootstrap,
            ),
            "worst_dataset_macro_f1": _bootstrap_metric(
                len(y_true),
                lambda idx, yt=y_true, yp=y_pred, ds=dataset: _worst_dataset_f1(
                    yt[idx], yp[idx], ds[idx]
                ),
                lambda yt=y_true, yp=y_pred, ds=dataset: _worst_dataset_f1(yt, yp, ds),
                rng,
                num_bootstrap,
            ),
            "by_dataset_macro_f1": _dataset_metric_table(y_true, y_pred, dataset, rng, num_bootstrap),
        }

    risk_raw = test_raw["tasks"]["risk"]
    risk_true = np.asarray(risk_raw["y_true"], dtype=int)
    risk_pred = np.asarray(risk_raw["y_pred"], dtype=int)
    if len(risk_true):
        high_idx = RISK.index("high")
        y_true_low = risk_true >= high_idx
        y_pred_low = risk_pred >= high_idx
        num_positive = int(y_true_low.sum())
        report["low_friction_detection"]["positive_definition"] = "risk in {high, very_high}"
        report["low_friction_detection"]["num_positive"] = num_positive
        report["low_friction_detection"]["num_pred_positive"] = int(y_pred_low.sum())
        report["low_friction_detection"]["applicable"] = num_positive > 0
        if num_positive > 0:
            report["low_friction_detection"]["recall"] = _bootstrap_metric(
                len(y_true_low),
                lambda idx: float(recall_score(y_true_low[idx], y_pred_low[idx], zero_division=0)),
                lambda: float(recall_score(y_true_low, y_pred_low, zero_division=0)),
                rng,
                num_bootstrap,
            )
        else:
            report["low_friction_detection"]["recall"] = None

    test_mu = _mu_arrays(test_raw["mu"])
    calib_mu = _mu_arrays(calib_raw["mu"])
    if len(test_mu["target"]):
        radius = _conformal_radius(_mu_scores(calib_mu), target_coverage)
        hierarchy_radii, hierarchy_meta = _hierarchical_radii(
            calib_mu,
            test_mu,
            target_coverage=target_coverage,
            min_group_calibration=min_group_calibration,
        )
        report["mu_interval"] = {
            "conformal_radius": float(radius),
            "raw_coverage": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _coverage(test_mu, idx=idx, radius=0.0),
                lambda: _coverage(test_mu, radius=0.0),
                rng,
                num_bootstrap,
            ),
            "raw_width": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _width(test_mu, idx=idx, radius=0.0),
                lambda: _width(test_mu, radius=0.0),
                rng,
                num_bootstrap,
            ),
            "calibrated_coverage": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _coverage(test_mu, idx=idx, radius=radius),
                lambda: _coverage(test_mu, radius=radius),
                rng,
                num_bootstrap,
            ),
            "calibrated_width": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _width(test_mu, idx=idx, radius=radius),
                lambda: _width(test_mu, radius=radius),
                rng,
                num_bootstrap,
            ),
            "hierarchical_conformal_policy": {
                "radius_rule": (
                    "per sample, use the maximum available conformal radius among "
                    "global, dataset, dataset::state, risk, and dataset::state+risk"
                ),
                "min_group_calibration": int(min_group_calibration),
                "claim_boundary": (
                    "Use only with coverage-width reporting; wider intervals are not an automatic win."
                ),
                **hierarchy_meta,
            },
            "hierarchical_calibrated_coverage": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _coverage_variable_radius(test_mu, hierarchy_radii, idx=idx),
                lambda: _coverage_variable_radius(test_mu, hierarchy_radii),
                rng,
                num_bootstrap,
            ),
            "hierarchical_calibrated_width": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _width_variable_radius(test_mu, hierarchy_radii, idx=idx),
                lambda: _width_variable_radius(test_mu, hierarchy_radii),
                rng,
                num_bootstrap,
            ),
            "hierarchical_worst_dataset_core_coverage": _bootstrap_metric(
                len(test_mu["target"]),
                lambda idx: _worst_group_coverage_variable_radius(
                    test_mu, hierarchy_radii, group_key="group_key", idx=idx
                ),
                lambda: _worst_group_coverage_variable_radius(
                    test_mu, hierarchy_radii, group_key="group_key"
                ),
                rng,
                num_bootstrap,
            ),
        }
    return report


def _bootstrap_metric(
    n: int,
    bootstrap_fn: Callable[[np.ndarray], float],
    point_fn: Callable[[], float],
    rng: np.random.Generator,
    num_bootstrap: int,
) -> dict[str, float | int]:
    point = float(point_fn())
    if n <= 1:
        return {"point": point, "ci_low": point, "ci_high": point, "num_samples": int(n)}
    values = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        values[i] = float(bootstrap_fn(idx))
    return {
        "point": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "num_samples": int(n),
        "num_bootstrap": int(num_bootstrap),
    }


def _dataset_metric_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset: np.ndarray,
    rng: np.random.Generator,
    num_bootstrap: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in sorted(set(dataset.tolist())):
        keep = dataset == name
        yt = y_true[keep]
        yp = y_pred[keep]
        if len(yt) == 0:
            continue
        out[str(name)] = _bootstrap_metric(
            len(yt),
            lambda idx, a=yt, b=yp: _macro_f1(a[idx], b[idx]),
            lambda a=yt, b=yp: _macro_f1(a, b),
            rng,
            num_bootstrap,
        )
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _worst_dataset_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset: np.ndarray,
) -> float:
    scores = []
    for name in sorted(set(dataset.tolist())):
        keep = dataset == name
        if keep.any():
            scores.append(_macro_f1(y_true[keep], y_pred[keep]))
    return float(min(scores)) if scores else 0.0


def _mu_arrays(raw: dict[str, list]) -> dict[str, np.ndarray]:
    n = len(raw.get("pred_low", []) or [])
    return {
        "pred": np.stack(
            [
                np.asarray(raw.get("pred_low", []), dtype=float),
                np.asarray(raw.get("pred_high", []), dtype=float),
            ],
            axis=1,
        )
        if raw.get("pred_low")
        else np.empty((0, 2), dtype=float),
        "target": np.stack(
            [
                np.asarray(raw.get("target_low", []), dtype=float),
                np.asarray(raw.get("target_high", []), dtype=float),
            ],
            axis=1,
        )
        if raw.get("target_low")
        else np.empty((0, 2), dtype=float),
        "dataset": np.asarray(raw.get("dataset", [""] * n), dtype=str),
        "group_key": np.asarray(raw.get("group_key", [""] * n), dtype=str),
        "risk": np.asarray([str(item) for item in raw.get("risk", [""] * n)], dtype=str),
    }


def _mu_scores(items: dict[str, np.ndarray]) -> np.ndarray:
    if len(items["target"]) == 0:
        return np.zeros(0, dtype=float)
    pred = items["pred"]
    target = items["target"]
    return np.maximum.reduce(
        [
            pred[:, 0] - target[:, 0],
            target[:, 1] - pred[:, 1],
            np.zeros(len(target), dtype=float),
        ]
    )


def _conformal_radius(scores: np.ndarray, target_coverage: float) -> float:
    if len(scores) == 0:
        return 0.0
    sorted_scores = np.sort(scores)
    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    idx = int(np.ceil((len(sorted_scores) + 1) * target_coverage) - 1)
    idx = min(max(idx, 0), len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def _coverage(items: dict[str, np.ndarray], *, radius: float, idx: np.ndarray | None = None) -> float:
    pred = items["pred"] if idx is None else items["pred"][idx]
    target = items["target"] if idx is None else items["target"][idx]
    low = np.clip(pred[:, 0] - radius, 0.0, 1.2)
    high = np.clip(pred[:, 1] + radius, 0.0, 1.2)
    return float(((low <= target[:, 0]) & (high >= target[:, 1])).mean())


def _width(items: dict[str, np.ndarray], *, radius: float, idx: np.ndarray | None = None) -> float:
    pred = items["pred"] if idx is None else items["pred"][idx]
    low = np.clip(pred[:, 0] - radius, 0.0, 1.2)
    high = np.clip(pred[:, 1] + radius, 0.0, 1.2)
    return float((high - low).mean())


def _hierarchical_radii(
    calib: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    target_coverage: float,
    min_group_calibration: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(test["target"])
    if n == 0:
        return np.zeros(0, dtype=float), {"global_radius": 0.0, "radius_source_counts": {}}
    global_radius = _conformal_radius(_mu_scores(calib), target_coverage)
    scores = _mu_scores(calib)
    dataset_radii = _group_radii(scores, calib["dataset"], target_coverage, min_group_calibration)
    core_radii = _group_radii(scores, calib["group_key"], target_coverage, min_group_calibration)
    risk_radii = _group_radii(scores, calib["risk"], target_coverage, min_group_calibration)
    calib_core_risk = _combine_keys(calib["group_key"], calib["risk"])
    test_core_risk = _combine_keys(test["group_key"], test["risk"])
    core_risk_radii = _group_radii(scores, calib_core_risk, target_coverage, min_group_calibration)

    radii = np.full(n, global_radius, dtype=float)
    source_counts: dict[str, int] = {}
    for i in range(n):
        candidates = [("global", global_radius)]
        dataset = str(test["dataset"][i])
        core = str(test["group_key"][i])
        risk = str(test["risk"][i])
        core_risk = str(test_core_risk[i])
        if dataset in dataset_radii:
            candidates.append(("dataset", dataset_radii[dataset]))
        if core in core_radii:
            candidates.append(("dataset_core", core_radii[core]))
        if risk in risk_radii:
            candidates.append(("risk", risk_radii[risk]))
        if core_risk in core_risk_radii:
            candidates.append(("dataset_core_risk", core_risk_radii[core_risk]))
        source, radius = max(candidates, key=lambda item: item[1])
        radii[i] = radius
        source_counts[source] = source_counts.get(source, 0) + 1
    return radii, {
        "global_radius": float(global_radius),
        "mean_radius": float(np.mean(radii)),
        "max_radius": float(np.max(radii)),
        "radius_source_counts": source_counts,
        "num_dataset_radii": len(dataset_radii),
        "num_dataset_core_radii": len(core_radii),
        "num_risk_radii": len(risk_radii),
        "num_dataset_core_risk_radii": len(core_risk_radii),
    }


def _group_radii(
    scores: np.ndarray,
    keys: np.ndarray,
    target_coverage: float,
    min_group_calibration: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in sorted(set(keys.tolist())):
        mask = keys == group
        if int(mask.sum()) < int(min_group_calibration):
            continue
        out[str(group)] = _conformal_radius(scores[mask], target_coverage)
    return out


def _combine_keys(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray([f"{a}::risk={b}" for a, b in zip(left.tolist(), right.tolist())], dtype=str)


def _coverage_variable_radius(
    items: dict[str, np.ndarray],
    radii: np.ndarray,
    *,
    idx: np.ndarray | None = None,
) -> float:
    pred = items["pred"] if idx is None else items["pred"][idx]
    target = items["target"] if idx is None else items["target"][idx]
    rr = radii if idx is None else radii[idx]
    low = np.clip(pred[:, 0] - rr, 0.0, 1.2)
    high = np.clip(pred[:, 1] + rr, 0.0, 1.2)
    return float(((low <= target[:, 0]) & (high >= target[:, 1])).mean())


def _width_variable_radius(
    items: dict[str, np.ndarray],
    radii: np.ndarray,
    *,
    idx: np.ndarray | None = None,
) -> float:
    pred = items["pred"] if idx is None else items["pred"][idx]
    rr = radii if idx is None else radii[idx]
    low = np.clip(pred[:, 0] - rr, 0.0, 1.2)
    high = np.clip(pred[:, 1] + rr, 0.0, 1.2)
    return float((high - low).mean())


def _worst_group_coverage_variable_radius(
    items: dict[str, np.ndarray],
    radii: np.ndarray,
    *,
    group_key: str,
    idx: np.ndarray | None = None,
) -> float:
    groups = items[group_key] if idx is None else items[group_key][idx]
    if len(groups) == 0:
        return 0.0
    values = []
    local_items = items
    local_radii = radii
    if idx is not None:
        local_items = {
            "pred": items["pred"][idx],
            "target": items["target"][idx],
            group_key: groups,
        }
        local_radii = radii[idx]
    for group in sorted(set(groups.tolist())):
        keep = groups == group
        if keep.any():
            values.append(
                _coverage_variable_radius(
                    {"pred": local_items["pred"][keep], "target": local_items["target"][keep]},
                    local_radii[keep],
                )
            )
    return float(min(values)) if values else 0.0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bootstrap Metric Confidence Intervals",
        "",
        f"Config: `{report.get('config', '-')}`",
        f"Checkpoint: `{report.get('checkpoint', '-')}`",
        f"Bootstrap samples: `{report['num_bootstrap']}`",
        "",
        "| Metric | Point | 95% CI | N |",
        "|---|---:|---:|---:|",
    ]
    for task, metrics in report.get("classification", {}).items():
        for name in ["macro_f1", "worst_dataset_macro_f1"]:
            item = metrics.get(name)
            if item:
                lines.append(_metric_row(f"{task} {name}", item, percent=True))
    low_info = report.get("low_friction_detection", {})
    item = low_info.get("recall")
    if item:
        lines.append(_metric_row("low-friction recall", item, percent=True))
    elif low_info.get("applicable") is False:
        lines.append(
            "| low-friction recall | N/A | no high/very_high risk positives in this split | "
            f"{low_info.get('num_positive', 0)} |"
        )
    for name, item in report.get("mu_interval", {}).items():
        if isinstance(item, dict) and "point" in item:
            lines.append(_metric_row(f"mu {name}", item, percent="coverage" in name))
    policy = report.get("mu_interval", {}).get("hierarchical_conformal_policy")
    if isinstance(policy, dict):
        lines.append("")
        lines.append(
            "Hierarchical conformal policy: max-radius over global/dataset/dataset::state/risk/"
            "dataset::state+risk; source counts "
            f"`{policy.get('radius_source_counts', {})}`."
        )
    lines.append("")
    lines.append("Dataset-specific macro-F1 CIs are stored in the JSON file.")
    lines.append("")
    return "\n".join(lines)


def _metric_row(name: str, item: dict[str, Any], *, percent: bool) -> str:
    point = _fmt(item["point"], percent=percent)
    low = _fmt(item["ci_low"], percent=percent)
    high = _fmt(item["ci_high"], percent=percent)
    return f"| {name} | {point} | [{low}, {high}] | {item.get('num_samples', '-')} |"


def _fmt(value: float, *, percent: bool) -> str:
    if percent:
        return f"{100.0 * float(value):.2f}"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
