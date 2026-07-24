from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from friction_affordance.models.backbone import build_backbone
from friction_affordance.utils import resolve_device, set_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_TRAIN = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_VAL = Path("data/manifests_full/rscd_prepared_val.csv")
DEFAULT_OUT = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\pretrain_physics_evidence"
)


def canonical_class_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


class RSCDEvidencePretrainDataset(Dataset):
    """Return a mildly perturbed view and a clean view of the same RSCD patch.

    The target evidence fields are computed from the clean view in the training
    loop. Geometry is shared between the input and target so the task is
    evidence-preserving denoising rather than arbitrary masked classification.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        image_size: int,
        train: bool,
        max_samples: int | None = None,
        max_samples_per_class: int | None = None,
        seed: int = 79,
    ) -> None:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        df = pd.read_csv(manifest, dtype=str, low_memory=False)
        if "image_path" not in df.columns or "class_label" not in df.columns:
            raise ValueError(f"{manifest} must contain image_path and class_label columns.")
        df["class_label_canonical"] = df["class_label"].map(canonical_class_label)
        if max_samples_per_class:
            parts = []
            for _, group in df.groupby("class_label_canonical", sort=True):
                if len(group) > int(max_samples_per_class):
                    group = group.sample(n=int(max_samples_per_class), random_state=int(seed))
                parts.append(group)
            df = pd.concat(parts, ignore_index=True)
        if max_samples:
            df = df.sample(n=min(int(max_samples), len(df)), random_state=int(seed)).reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.color_jitter = transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.08,
            hue=0.015,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self._warned_bad_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        start_idx = int(idx)
        last_error: Exception | None = None
        for offset in range(min(50, len(self.df))):
            row_idx = (start_idx + offset) % len(self.df)
            row = self.df.iloc[row_idx]
            path = Path(str(row["image_path"]))
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()
                image = self.resize(image)
                if self.train and random.random() < 0.5:
                    image = transforms.functional.hflip(image)
                clean = image
                view = image
                if self.train:
                    view = self.color_jitter(view)
                    if random.random() < 0.04:
                        view = transforms.functional.rgb_to_grayscale(view, num_output_channels=3)
                clean_tensor = self.normalize(self.to_tensor(clean))
                view_tensor = self.normalize(self.to_tensor(view))
                return {
                    "image": view_tensor,
                    "clean_image": clean_tensor,
                    "image_path": str(path),
                    "class_label": str(row["class_label_canonical"]),
                }
            except (OSError, SyntaxError, ValueError) as exc:
                last_error = exc
                path_text = str(path)
                if path_text not in self._warned_bad_paths:
                    self._warned_bad_paths.add(path_text)
                    print(f"WARNING: skipped unreadable image: {path_text} ({type(exc).__name__}: {exc})", flush=True)
                continue
        raise RuntimeError(f"Could not load a valid image after retries near index {start_idx}: {last_error}")


class PhysicsEvidenceTarget(nn.Module):
    """Analytic RSCD evidence fields used as public-data self-supervision."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer(
            "laplace",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
        )

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        lo = flat.amin(dim=2).view(x.shape[0], x.shape[1], 1, 1)
        hi = flat.amax(dim=2).view(x.shape[0], x.shape[1], 1, 1)
        return (x - lo) / (hi - lo).clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std.to(dtype=x.dtype, device=x.device) + self.mean.to(dtype=x.dtype, device=x.device)).clamp(
            0.0,
            1.0,
        )
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x.to(dtype=x.dtype, device=x.device), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=x.dtype, device=x.device), padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace.to(dtype=x.dtype, device=x.device), padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        grad_norm = self._normalize_map(grad)
        lap_norm = self._normalize_map(lap)
        contrast_norm = self._normalize_map(local_contrast)
        rough_base = torch.clamp(0.42 * grad_norm + 0.34 * lap_norm + 0.24 * contrast_norm, 0.0, 1.0)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        snow_phase = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)

        obstruction = torch.clamp(
            0.40 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure,
            0.0,
            1.0,
        )
        visibility = 1.0 - obstruction
        visible_rough = rough_base * visibility * (1.0 - snow_phase) * (1.0 - marking)
        hidden_rough = rough_base * obstruction * (1.0 - snow_phase)
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.28 - saturation) * 10.0)
            * (1.0 - snow_phase)
            * (1.0 - marking)
        )
        dry_rough = rough_base * (1.0 - obstruction) * concrete_like
        masked_concrete_rough = hidden_rough * concrete_like
        film_rough_coupling = thin_film * rough_base
        granular = (
            torch.sigmoid((local_contrast - 0.040) * 35.0)
            * torch.sigmoid((saturation - 0.045) * 8.0)
            * (1.0 - marking)
        )
        granular_wet = granular * torch.clamp(thin_film + dark_water, 0.0, 1.0)

        return torch.cat(
            [
                obstruction,
                visible_rough,
                hidden_rough,
                thin_film,
                texture_erasure,
                dry_rough,
                masked_concrete_rough,
                film_rough_coupling,
                granular_wet,
            ],
            dim=1,
        ).clamp(0.0, 1.0)


class PhysicsEvidencePretrainer(nn.Module):
    """Backbone encoder trained to predict RSCD physical evidence fields."""

    _STAGE_CHANNELS = {
        "convnext_tiny": {"early": 96, "mid": 192, "late": 384, "final": 768},
        "road_mechanism_tiny": {"early": 64, "mid": 128, "late": 256, "final": 384},
    }

    def __init__(
        self,
        *,
        backbone: str = "convnext_tiny",
        embedding_dim: int = 256,
        pretrained: bool = True,
        num_fields: int = 9,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone)
        self.backbone = build_backbone(self.backbone_name, embedding_dim, pretrained=pretrained)
        if self.backbone_name not in self._STAGE_CHANNELS:
            raise ValueError(
                f"Unsupported evidence-pretrain backbone: {self.backbone_name}. "
                f"Known: {sorted(self._STAGE_CHANNELS)}"
            )
        channels = self._STAGE_CHANNELS[self.backbone_name]
        self.early_head = self._map_head(channels["early"], num_fields)
        self.mid_head = self._map_head(channels["mid"], num_fields)
        self.late_head = self._map_head(channels["late"], num_fields)
        self.final_head = self._map_head(channels["final"], num_fields)
        self.summary_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_fields * 2),
        )

    @staticmethod
    def _map_head(channels: int, num_fields: int) -> nn.Sequential:
        hidden = max(32, min(128, int(channels) // 2))
        return nn.Sequential(
            nn.Conv2d(int(channels), hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, int(num_fields), kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.backbone(x)
        stage_maps = getattr(self.backbone, "stage_feature_maps", {})
        early = stage_maps.get("early")
        mid = stage_maps.get("mid")
        late = stage_maps.get("late")
        final = stage_maps.get("final")
        if early is None or mid is None or late is None or final is None:
            raise RuntimeError(
                f"Backbone {self.backbone_name} did not expose early/mid/late/final stage_feature_maps."
            )
        return {
            "early": self.early_head(early),
            "mid": self.mid_head(mid),
            "late": self.late_head(late),
            "final": self.final_head(final),
            "summary": self.summary_head(embedding),
        }


def apply_evidence_preserving_mask(x: torch.Tensor, *, mask_ratio: float, max_rectangles: int = 2) -> torch.Tensor:
    if mask_ratio <= 0:
        return x
    out = x.clone()
    b, _, h, w = out.shape
    target_area = int(float(mask_ratio) * h * w)
    if target_area <= 0:
        return out
    for sample_idx in range(b):
        covered = 0
        rectangles = random.randint(1, max(1, int(max_rectangles)))
        for _ in range(rectangles):
            remaining = max(target_area - covered, 1)
            rect_area = max(remaining // max(rectangles, 1), 1)
            aspect = math.exp(random.uniform(math.log(0.45), math.log(2.2)))
            rect_h = max(4, min(h, int(round(math.sqrt(rect_area / aspect)))))
            rect_w = max(4, min(w, int(round(math.sqrt(rect_area * aspect)))))
            top = random.randint(0, max(h - rect_h, 0))
            left = random.randint(0, max(w - rect_w, 0))
            out[sample_idx, :, top : top + rect_h, left : left + rect_w] = 0.0
            covered += rect_h * rect_w
            if covered >= target_area:
                break
    return out


def evidence_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}
    weights = {"early": 1.0, "mid": 0.75, "late": 0.45, "final": 0.30}
    total = target.new_tensor(0.0)
    for key, weight in weights.items():
        pred = torch.sigmoid(outputs[key])
        tgt = F.interpolate(target, size=pred.shape[-2:], mode="area")
        loss = F.smooth_l1_loss(pred, tgt)
        losses[key] = loss
        total = total + float(weight) * loss
    mean = target.mean(dim=(2, 3))
    std = target.std(dim=(2, 3))
    summary_target = torch.cat([mean, std], dim=1)
    summary_loss = F.smooth_l1_loss(outputs["summary"], summary_target)
    total = total + 0.20 * summary_loss
    losses["summary"] = summary_loss
    logs = {name: float(value.detach().cpu()) for name, value in losses.items()}
    logs["loss"] = float(total.detach().cpu())
    return total, logs


def train_one_epoch(
    model: PhysicsEvidencePretrainer,
    target_builder: PhysicsEvidenceTarget,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    mask_ratio: float,
    log_every_steps: int,
) -> dict[str, float]:
    model.train()
    total_logs: dict[str, float] = {}
    steps = 0
    progress = tqdm(loader, desc="train", dynamic_ncols=True)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        clean = batch["clean_image"].to(device, non_blocking=True)
        image = apply_evidence_preserving_mask(image, mask_ratio=mask_ratio)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            target = target_builder(clean)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(image)
            loss, logs = evidence_loss(outputs, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        for key, value in logs.items():
            total_logs[key] = total_logs.get(key, 0.0) + value
        if log_every_steps > 0 and steps % int(log_every_steps) == 0:
            progress.set_postfix({key: f"{value / steps:.4f}" for key, value in total_logs.items() if key == "loss"})
    return {key: value / max(steps, 1) for key, value in total_logs.items()}


@torch.no_grad()
def evaluate(
    model: PhysicsEvidencePretrainer,
    target_builder: PhysicsEvidenceTarget,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_logs: dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc="val", dynamic_ncols=True):
        image = batch["image"].to(device, non_blocking=True)
        clean = batch["clean_image"].to(device, non_blocking=True)
        target = target_builder(clean)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(image)
            _, logs = evidence_loss(outputs, target)
        steps += 1
        for key, value in logs.items():
            total_logs[key] = total_logs.get(key, 0.0) + value
    return {key: value / max(steps, 1) for key, value in total_logs.items()}


def build_loader(ds: Dataset, *, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(train),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": bool(train),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSCD physics-evidence reconstruction pretraining.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backbone", default="convnext_tiny", choices=["convnext_tiny", "road_mechanism_tiny"])
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--mask-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-samples-per-class", type=int, default=None)
    parser.add_argument("--max-val-samples-per-class", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every-steps", type=int, default=50)
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_ds = RSCDEvidencePretrainDataset(
        args.train_manifest,
        image_size=int(args.image_size),
        train=True,
        max_samples=args.max_train_samples,
        max_samples_per_class=args.max_train_samples_per_class,
        seed=int(args.seed),
    )
    val_ds = RSCDEvidencePretrainDataset(
        args.val_manifest,
        image_size=int(args.image_size),
        train=False,
        max_samples=args.max_val_samples,
        max_samples_per_class=args.max_val_samples_per_class,
        seed=int(args.seed) + 1,
    )
    print(f"Dataset sizes: train={len(train_ds)} val={len(val_ds)}", flush=True)
    train_loader = build_loader(train_ds, batch_size=int(args.batch_size), num_workers=int(args.num_workers), train=True)
    val_loader = build_loader(val_ds, batch_size=int(args.batch_size), num_workers=int(args.num_workers), train=False)

    model = PhysicsEvidencePretrainer(
        backbone=str(args.backbone),
        embedding_dim=int(args.embedding_dim),
        pretrained=bool(args.pretrained),
    ).to(device)
    target_builder = PhysicsEvidenceTarget().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = vars(args).copy()
    config["train_manifest"] = str(args.train_manifest)
    config["val_manifest"] = str(args.val_manifest)
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_loss = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            target_builder,
            train_loader,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
            mask_ratio=float(args.mask_ratio),
            log_every_steps=int(args.log_every_steps),
        )
        val_metrics = evaluate(model, target_builder, val_loader, device, use_amp=use_amp)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "config": config}, args.output_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "config": config},
                args.output_dir / "best.pt",
            )
            print(f"  saved best checkpoint: val_loss={best_loss:.6f}", flush=True)
    (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "evaluate_val.json").write_text(
        json.dumps({"summary": history[-1]["val"] if history else {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
