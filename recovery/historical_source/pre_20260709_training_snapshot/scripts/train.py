from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.tensorboard import SummaryWriter

from friction_affordance.engine import build_loaders, build_model, evaluate, save_checkpoint, train_one_epoch
from friction_affordance.utils import load_yaml, resolve_device, set_seed


_RUN_LOCK_PATH: Path | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg.get("output_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _register_run_lock(out_dir, args.config)
    set_seed(int(cfg.get("seed", 7)))

    device = resolve_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_loaders(cfg)
    cfg.setdefault("model", {})
    cfg["model"]["num_domains"] = int(getattr(train_loader.dataset, "num_domains", 0))
    cfg.setdefault("data", {})
    cfg["data"]["num_groups"] = int(getattr(train_loader.dataset, "num_groups", 0))
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    if cfg["model"]["num_domains"] > 1:
        print(f"Domains: {cfg['model']['num_domains']} ({getattr(train_loader.dataset, 'domain_to_idx', {})})")
    if int(getattr(train_loader.dataset, "num_groups", 0)) > 1:
        print(f"Groups: {getattr(train_loader.dataset, 'num_groups', 0)}")

    model = build_model(cfg).to(device)
    _maybe_freeze_backbone(model, cfg.get("model", {}))
    total_params, trainable_params = _count_parameters(model)
    print(f"Parameters: trainable={trainable_params:,} total={total_params:,}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["optim"].get("lr", 1e-4)),
        weight_decay=float(cfg["optim"].get("weight_decay", 1e-4)),
    )
    use_amp = bool(cfg["optim"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("AMP: enabled")

    start_epoch = 1
    best_metric = float("inf")
    best_safety_metric = float("-inf")
    resumed_state = _load_training_state(out_dir / "training_state.json") if args.resume else {}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("metrics", {}).get("loss", best_metric))
        best_safety_metric = float(resumed_state.get("best_safety_metric", best_safety_metric))
        best_path = out_dir / "best.pt"
        if best_path.exists() and best_path.resolve() != args.resume.resolve():
            best_ckpt = torch.load(best_path, map_location="cpu")
            best_metric = min(best_metric, float(best_ckpt.get("metrics", {}).get("loss", best_metric)))
        best_safety_path = out_dir / "best_safety.pt"
        if best_safety_path.exists() and best_safety_path.resolve() != args.resume.resolve():
            best_safety_ckpt = torch.load(best_safety_path, map_location="cpu")
            best_safety_metric = max(
                best_safety_metric,
                _safety_proxy(best_safety_ckpt.get("metrics", {})),
            )
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    history_path = out_dir / "metrics_history.json"
    history = _load_history(history_path)
    epochs = int(cfg["optim"].get("epochs", 1))
    patience = cfg["optim"].get("early_stop_patience")
    patience = int(patience) if patience is not None else None
    min_delta = float(cfg["optim"].get("early_stop_min_delta", 0.0))
    safety_min_delta = float(cfg["optim"].get("safety_proxy_min_delta", 0.0))
    stale_epochs = int(resumed_state.get("stale_epochs", 0)) if args.resume else 0
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg["grad_clip_norm"] = float(cfg["optim"].get("grad_clip_norm", 5.0))
    loss_cfg["grad_accum_steps"] = int(cfg["optim"].get("grad_accum_steps", 1))
    loss_cfg["log_every_steps"] = int(cfg["optim"].get("log_every_steps", 250))

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}", flush=True)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_cfg, scaler=scaler, use_amp=use_amp)
        val_metrics = evaluate(model, val_loader, device, loss_cfg)
        print("  train:", _fmt(train_metrics), flush=True)
        print("  val  :", _fmt(val_metrics), flush=True)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, val_metrics, cfg)
        should_stop = False
        saved_best = False
        saved_best_safety = False
        if val_metrics["loss"] < best_metric - min_delta:
            best_metric = val_metrics["loss"]
            stale_epochs = 0
            save_checkpoint(out_dir / "best.pt", model, None, epoch, val_metrics, cfg)
            saved_best = True
            print(f"  saved best checkpoint: {out_dir / 'best.pt'}", flush=True)
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                print(f"  early stopping: no val loss improvement for {stale_epochs} epochs", flush=True)
                should_stop = True
        safety_metric = _safety_proxy(val_metrics)
        if safety_metric > best_safety_metric + safety_min_delta:
            best_safety_metric = safety_metric
            save_checkpoint(out_dir / "best_safety.pt", model, None, epoch, val_metrics, cfg)
            saved_best_safety = True
            print(
                f"  saved best safety checkpoint: {out_dir / 'best_safety.pt'} "
                f"(proxy={best_safety_metric:.4f})",
                flush=True,
            )
        with open(out_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "epochs": epochs,
                    "best_metric": best_metric,
                    "best_safety_metric": best_safety_metric,
                    "stale_epochs": stale_epochs,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        _append_history(
            history,
            history_path,
            {
                "epoch": epoch,
                "epochs": epochs,
                "best_metric": best_metric,
                "best_safety_metric": best_safety_metric,
                "stale_epochs": stale_epochs,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "saved_best": saved_best,
                "saved_best_safety": saved_best_safety,
                "safety_proxy": safety_metric,
            },
        )
        if should_stop:
            break

    writer.close()


def _fmt(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))


def _safety_proxy(metrics: dict) -> float:
    risk = float(metrics.get("acc_risk", 0.0) or 0.0)
    friction = float(metrics.get("acc_friction", 0.0) or 0.0)
    coverage = float(metrics.get("mu_interval_coverage", 0.0) or 0.0)
    width = float(metrics.get("mu_interval_width", 0.0) or 0.0)
    return risk + 0.5 * friction + 0.5 * coverage - 0.1 * width


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except json.JSONDecodeError:
        return []


def _load_training_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except json.JSONDecodeError:
        return {}


def _register_run_lock(out_dir: Path, config: Path) -> None:
    """Prevent two training processes from writing the same run directory."""
    global _RUN_LOCK_PATH
    lock_path = out_dir / "train.lock"
    payload = {
        "pid": os.getpid(),
        "config": str(config),
        "argv": sys.argv,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            _RUN_LOCK_PATH = lock_path
            atexit.register(_release_run_lock)
            return
        except FileExistsError:
            current = _read_lock(lock_path)
            pid = _as_int(current.get("pid")) if isinstance(current, dict) else None
            if pid is not None and _pid_alive(pid):
                raise RuntimeError(
                    f"Refusing to start duplicate training for {out_dir}; "
                    f"active lock is held by PID {pid}. Lock file: {lock_path}"
                )
            lock_path.unlink(missing_ok=True)
    raise RuntimeError(f"Could not acquire training lock: {lock_path}")


def _release_run_lock() -> None:
    global _RUN_LOCK_PATH
    lock_path = _RUN_LOCK_PATH
    if lock_path is None or not lock_path.exists():
        return
    current = _read_lock(lock_path)
    if isinstance(current, dict) and _as_int(current.get("pid")) == os.getpid():
        lock_path.unlink(missing_ok=True)
    _RUN_LOCK_PATH = None


def _read_lock(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return ctypes.get_last_error() == 5
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _append_history(history: list[dict], path: Path, row: dict) -> None:
    history[:] = [item for item in history if int(item.get("epoch", -1)) != int(row["epoch"])]
    history.append(row)
    history.sort(key=lambda item: int(item.get("epoch", 0)))
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_freeze_backbone(model: torch.nn.Module, model_cfg: dict) -> None:
    if not bool(model_cfg.get("freeze_backbone", False)):
        return
    encoder = getattr(model, "encoder", None)
    feature_extractor = getattr(encoder, "model", None)
    target = feature_extractor if feature_extractor is not None else encoder
    if target is None:
        return
    for param in target.parameters():
        param.requires_grad = False
    print("Frozen backbone feature extractor; projection and task heads remain trainable.")


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    main()
