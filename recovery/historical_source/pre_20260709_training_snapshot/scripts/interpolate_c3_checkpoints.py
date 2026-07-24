from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"checkpoint has no model state: {path}")
    return state


def interpolate(anchor_path: Path, specialist_path: Path, output_path: Path, alpha: float) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    anchor = _load(anchor_path)
    specialist = _load(specialist_path)
    anchor_model = anchor["model"]
    specialist_model = specialist["model"]
    if not isinstance(anchor_model, dict) or not isinstance(specialist_model, dict):
        raise ValueError("model states must be dictionaries")

    merged_model: dict[str, torch.Tensor] = {}
    interpolated = 0
    copied_anchor = 0
    copied_specialist_only = 0
    copied_specialist_shape = 0
    for key, anchor_tensor in anchor_model.items():
        specialist_tensor = specialist_model.get(key)
        if (
            isinstance(anchor_tensor, torch.Tensor)
            and isinstance(specialist_tensor, torch.Tensor)
            and anchor_tensor.shape == specialist_tensor.shape
            and anchor_tensor.is_floating_point()
            and specialist_tensor.is_floating_point()
        ):
            merged_model[key] = (1.0 - alpha) * anchor_tensor + alpha * specialist_tensor
            interpolated += 1
        elif (
            isinstance(anchor_tensor, torch.Tensor)
            and isinstance(specialist_tensor, torch.Tensor)
            and anchor_tensor.shape != specialist_tensor.shape
        ):
            # The current C3 configs may widen physics-conditioned expert layers
            # relative to the older anchor. Keep the specialist-shaped tensor so
            # the merged checkpoint remains loadable by the current model.
            merged_model[key] = specialist_tensor
            copied_specialist_shape += 1
        else:
            merged_model[key] = anchor_tensor
            copied_anchor += 1

    for key, specialist_tensor in specialist_model.items():
        if key not in merged_model:
            merged_model[key] = specialist_tensor
            copied_specialist_only += 1

    merged = dict(anchor)
    merged["model"] = merged_model
    merged["epoch"] = int(specialist.get("epoch", anchor.get("epoch", 0)))
    merged["interpolation"] = {
        "anchor": str(anchor_path),
        "specialist": str(specialist_path),
        "alpha": float(alpha),
        "interpolated_tensors": int(interpolated),
        "copied_anchor_tensors": int(copied_anchor),
        "copied_specialist_only_tensors": int(copied_specialist_only),
        "copied_specialist_shape_tensors": int(copied_specialist_shape),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    return merged["interpolation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate compatible C3-FaRNet checkpoints.")
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--specialist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    args = parser.parse_args()
    info = interpolate(args.anchor, args.specialist, args.output, args.alpha)
    print(info)


if __name__ == "__main__":
    main()
