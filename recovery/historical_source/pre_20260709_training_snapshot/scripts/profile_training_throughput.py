from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from friction_affordance.engine import build_loaders, build_model, move_batch
from friction_affordance.losses import compute_total_loss
from friction_affordance.utils import load_yaml, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-step", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = copy.deepcopy(cfg)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False

    set_seed(int(cfg.get("seed", 79)))
    device = resolve_device(cfg.get("device", "auto"))
    train_loader, _ = build_loaders(cfg)
    print(
        f"config={args.config} batch_size={cfg['data']['batch_size']} "
        f"num_workers={cfg['data'].get('num_workers', 0)} steps={args.steps}"
    )
    print(
        f"device={device} train_batches={len(train_loader)} "
        f"effective_num_workers={train_loader.num_workers}"
    )

    model = None
    optimizer = None
    scaler = None
    loss_cfg = None
    if args.train_step:
        cfg["model"]["num_domains"] = int(getattr(train_loader.dataset, "num_domains", 0))
        cfg["data"]["num_groups"] = int(getattr(train_loader.dataset, "num_groups", 0))
        model = build_model(cfg).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"].get("lr", 1e-4)))
        use_amp = bool(cfg["optim"].get("amp", False)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        loss_cfg = dict(cfg.get("loss", {}))
        loss_cfg["grad_clip_norm"] = float(cfg["optim"].get("grad_clip_norm", 5.0))
        loss_cfg["grad_accum_steps"] = 1

    load_times: list[float] = []
    step_times: list[float] = []
    images = 0
    iterator = iter(train_loader)
    last = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        now = time.perf_counter()
        load_times.append(now - last)
        images += int(batch["image"].shape[0])

        if args.train_step and model is not None and optimizer is not None and scaler is not None and loss_cfg is not None:
            t0 = time.perf_counter()
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(batch["image"])
                loss, _ = compute_total_loss(outputs, batch, loss_cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
        last = time.perf_counter()

    elapsed_load = sum(load_times)
    elapsed_step = sum(step_times)
    total = elapsed_load + elapsed_step
    print(f"images={images}")
    print(f"mean_load_s={mean(load_times):.4f}")
    if step_times:
        print(f"mean_train_step_s={mean(step_times):.4f}")
    print(f"throughput_images_per_s={images / max(total, 1e-9):.2f}")
    if device.type == "cuda":
        print(f"cuda_peak_alloc_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
        print(f"cuda_peak_reserved_gb={torch.cuda.max_memory_reserved() / 1024**3:.2f}")


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
