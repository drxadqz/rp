from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import build_class_map, build_model, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check C3-FaRNet ablation YAML configs.")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs" / "c3_farnet" / "ablations")
    parser.add_argument("--pattern", default="c3_ablation_*.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--forward", action="store_true", help="Run a batch=1 forward pass for every config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = sorted(Path(args.config_dir).glob(str(args.pattern)))
    if not configs:
        raise FileNotFoundError(f"No configs matched {args.config_dir / args.pattern}")

    reference_cfg = load_config(configs[0])
    manifests = [
        Path(reference_cfg["data"]["train_manifest"]),
        Path(reference_cfg["data"]["val_manifest"]),
        Path(reference_cfg["data"]["test_manifest"]),
    ]
    class_to_idx = build_class_map(manifests)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"checking {len(configs)} configs on {device}; classes={len(class_to_idx)}", flush=True)

    checked: list[dict[str, object]] = []
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        model = build_model(cfg, class_to_idx)
        logits_shape: tuple[int, ...] | None = None
        if bool(args.forward):
            model = model.to(device).eval()
            image_size = int(cfg["data"].get("image_size", 192))
            sample = torch.rand(1, 3, image_size, image_size, device=device)
            with torch.no_grad():
                out = model(sample, return_aux=True)
            logits = out["logits"] if isinstance(out, dict) else out
            logits_shape = tuple(int(v) for v in logits.shape)
            if logits_shape != (1, len(class_to_idx)):
                raise RuntimeError(f"{cfg_path}: expected logits {(1, len(class_to_idx))}, got {logits_shape}")
            del sample, out, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
        train_cfg = cfg.get("train", {})
        checked.append(
            {
                "file": cfg_path.name,
                "head_type": cfg["model"].get("head_type"),
                "boundary_use_physics_feature": cfg["model"].get("boundary_use_physics_feature"),
                "dryvor": cfg["model"].get("use_dry_concrete_roughness_vor_residual"),
                "factor_weight": cfg["loss"].get("factor_weight"),
                "tournament_weight": cfg["loss"].get("tournament_weight"),
                "counterfactual_weight": cfg["loss"].get("counterfactual_weight"),
                "reliability_weight": cfg["loss"].get("reliability_weight"),
                "epochs": train_cfg.get("epochs"),
                "samples_per_epoch": train_cfg.get("samples_per_epoch"),
                "logits_shape": logits_shape,
            }
        )
        del model

    for item in checked:
        print(item, flush=True)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
