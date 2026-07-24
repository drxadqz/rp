from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from friction_affordance.engine import build_loaders, build_model, move_batch
from friction_affordance.losses import compute_total_loss
from friction_affordance.utils import load_yaml, resolve_device, set_seed


def main() -> None:
    base = load_yaml(Path("configs/experiments/topvenue_v4_evidencefield.yaml"))
    base["data"]["num_workers"] = 0
    base["data"]["max_train_samples_per_class"] = 60
    base["data"]["max_val_samples_per_class"] = 20
    base["data"]["balanced_num_samples_per_epoch"] = 120
    base["optim"]["amp"] = True

    loss_cfg = dict(base.get("loss", {}))
    loss_cfg["grad_clip_norm"] = 5.0
    loss_cfg["grad_accum_steps"] = 1

    candidates = [
        ("mobilenet_v3_large", 192, 16, 512, 64, 48),
        ("mobilenet_v3_large", 224, 12, 512, 64, 48),
        ("mobilenet_v3_large", 160, 12, 512, 64, 48),
        ("mobilenet_v3_large", 176, 12, 512, 64, 48),
        ("mobilenet_v3_large", 192, 10, 512, 64, 48),
        ("convnext_tiny", 160, 12, 768, 96, 64),
        ("convnext_tiny", 176, 12, 768, 96, 64),
        ("convnext_tiny", 192, 12, 768, 96, 64),
        ("convnext_tiny", 192, 16, 768, 96, 64),
        ("convnext_tiny", 224, 8, 768, 96, 64),
        ("convnext_tiny", 160, 4, 768, 96, 64),
        ("convnext_tiny", 176, 3, 768, 96, 64),
        ("convnext_tiny", 192, 2, 768, 96, 64),
    ]

    set_seed(79)
    device = resolve_device("auto")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    for backbone, image_size, batch_size, embedding_dim, evidence_dim, evidence_hidden_dim in candidates:
        cfg = copy.deepcopy(base)
        cfg["data"]["image_size"] = image_size
        cfg["data"]["batch_size"] = batch_size
        cfg["model"]["backbone"] = backbone
        cfg["model"]["embedding_dim"] = embedding_dim
        cfg["model"]["evidence_dim"] = evidence_dim
        cfg["model"]["evidence_hidden_dim"] = evidence_hidden_dim
        cfg["model"]["physics_dim"] = 64 if backbone.startswith("mobile") else 96
        cfg["model"]["pretrained"] = True
        try:
            torch.cuda.empty_cache()
            gc.collect()
            train_loader, _ = build_loaders(cfg)
            cfg["model"]["num_domains"] = int(getattr(train_loader.dataset, "num_domains", 0))
            cfg["data"]["num_groups"] = int(getattr(train_loader.dataset, "num_groups", 0))
            model = build_model(cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            batch = move_batch(next(iter(train_loader)), device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(batch["image"])
                loss, _ = compute_total_loss(outputs, batch, loss_cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
            print(
                "OK "
                f"backbone={backbone} image_size={image_size} batch_size={batch_size} "
                f"peak_alloc_gb={peak_gb:.2f} peak_reserved_gb={reserved_gb:.2f} "
                f"loss={float(loss.detach().cpu()):.4f}"
            )
        except Exception as exc:
            print(
                "FAIL "
                f"backbone={backbone} image_size={image_size} batch_size={batch_size}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
