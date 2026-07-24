from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from friction_affordance.models.coupled_factor_backbone import RSCDCoupledFactorClassifier, count_parameters


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU smoke test for the S136 coupled-factor backbone prototype.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=27)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = RSCDCoupledFactorClassifier(num_classes=args.num_classes).cpu().eval()
    x = torch.rand(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        logits, aux = model(x, return_aux=True)
    weights = aux["coupling_weights"]
    payload = {
        "ok": True,
        "logits_shape": list(logits.shape),
        "evidence_shape": list(aux["evidence"].shape),
        "coupling_weights_shape": list(weights.shape),
        "coupling_weight_row_sums": weights.sum(dim=1).tolist(),
        "param_count": count_parameters(model),
    }
    (args.output_dir / "s136_coupled_factor_backbone_smoke.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
