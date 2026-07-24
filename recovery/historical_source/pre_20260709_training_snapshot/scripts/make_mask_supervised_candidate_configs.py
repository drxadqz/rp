from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs" / "experiments" / "paper_protocol" / "v23_lean_region_mixture_evidence_safety.yaml"
DEFAULT_OUT_DIR = ROOT / "configs" / "experiments" / "segmentation_transfer"
DEFAULT_RUN_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\segmentation_transfer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate optional mask-supervised EvidenceField configs from cached "
            "road_mask_path manifests. These are not inserted into the main paper "
            "queue until mask audit evidence justifies a formal ablation."
        )
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cached-manifest-dir", type=Path, required=True)
    parser.add_argument("--backend", type=str, default="clipseg")
    parser.add_argument("--resize-mode", type=str, default="bottom_square")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    manifests = _cached_manifests(args.cached_manifest_dir, args.backend, args.resize_mode, int(args.image_size))
    prefix = args.prefix or f"{args.backend}_mask_supervised_evidence"
    cfg = _make_config(base, manifests, args, prefix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{prefix}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(out_path)


def _cached_manifests(
    manifest_dir: Path,
    backend: str,
    resize_mode: str,
    image_size: int,
) -> dict[str, list[str]]:
    suffix = f"__{backend}_{resize_mode}_{image_size}.csv"
    split_map = {
        "train": [
            f"rscd_prepared_train{suffix}",
            f"roadsaw_train{suffix}",
            f"roadsc_train{suffix}",
        ],
        "val": [
            f"rscd_prepared_val{suffix}",
            f"roadsaw_val{suffix}",
            f"roadsc_val{suffix}",
        ],
        "test": [
            f"rscd_prepared_test{suffix}",
            f"roadsaw_test{suffix}",
            f"roadsc_test{suffix}",
        ],
    }
    out: dict[str, list[str]] = {}
    missing = []
    for split, names in split_map.items():
        paths = [manifest_dir / name for name in names]
        for path in paths:
            if not path.exists():
                missing.append(str(path))
        out[split] = [str(path) for path in paths]
    if missing:
        raise FileNotFoundError(
            "Missing cached mask manifests. Run scripts/cache_external_road_masks.py first.\n"
            + "\n".join(missing)
        )
    return out


def _make_config(
    base: dict[str, Any],
    manifests: dict[str, list[str]],
    args: argparse.Namespace,
    name: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(cfg.get("seed", 79))
    cfg["output_dir"] = str(args.run_root / name)
    data = cfg.setdefault("data", {})
    data["train_manifests"] = manifests["train"]
    data["val_manifests"] = manifests["val"]
    data["test_manifests"] = manifests["test"]
    data["image_size"] = int(args.image_size)
    data["load_road_masks"] = True
    data["road_mask_pretransformed"] = True
    if args.max_train_samples is not None:
        data["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        data["max_val_samples"] = int(args.max_val_samples)
    if args.max_test_samples is not None:
        data["max_test_samples"] = int(args.max_test_samples)
    augmentation = data.setdefault("augmentation", {})
    augmentation["resize_mode"] = str(args.resize_mode)
    augmentation["random_resized_crop"] = False
    augmentation["horizontal_flip_p"] = 0.0
    model = cfg.setdefault("model", {})
    model["use_evidence_field"] = True
    model["evidence_road_likelihood_prior_strength"] = max(
        float(model.get("evidence_road_likelihood_prior_strength", 0.0)),
        0.75,
    )
    model["evidence_region_mixture_cues"] = bool(model.get("evidence_region_mixture_cues", True))
    model["evidence_region_mixture_expansion"] = max(
        float(model.get("evidence_region_mixture_expansion", 0.0)),
        0.05,
    )
    loss = cfg.setdefault("loss", {})
    loss["evidence_attention_pseudo_road_weight"] = max(
        float(loss.get("evidence_attention_pseudo_road_weight", 0.0)),
        0.08,
    )
    loss["evidence_pseudo_road_min_mass"] = max(
        float(loss.get("evidence_pseudo_road_min_mass", 0.0)),
        0.76,
    )
    loss["evidence_attention_region_weight"] = max(
        float(loss.get("evidence_attention_region_weight", 0.0)),
        0.05,
    )
    if not args.formal:
        data["balanced_num_samples_per_epoch"] = min(int(data.get("balanced_num_samples_per_epoch", 36000)), 1200)
        optim = cfg.setdefault("optim", {})
        optim["epochs"] = min(int(optim.get("epochs", 20)), 2)
        optim["early_stop_patience"] = min(int(optim.get("early_stop_patience", 5)), 2)
        loss["log_every_steps"] = min(int(loss.get("log_every_steps", 100)), 50)
    cfg["experiment_note"] = (
        "Optional segmentation-transfer candidate: cached road/contact pseudo masks "
        f"from {args.backend} are used as soft EvidenceField attention supervision. "
        "This config is valid only for manifests containing road_mask_path and is "
        "kept outside the formal paper queue until the external mask audit passes."
    )
    return cfg


if __name__ == "__main__":
    main()
