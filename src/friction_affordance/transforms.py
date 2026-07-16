from __future__ import annotations

import random

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transforms(image_size: int, train: bool = True, aug_cfg: dict | None = None):
    aug_cfg = aug_cfg or {}
    ops = []
    resize_mode = str(aug_cfg.get("resize_mode", "stretch")).lower()
    if train and bool(aug_cfg.get("random_resized_crop", False)):
        scale = tuple(aug_cfg.get("crop_scale", [0.75, 1.0]))
        ratio = tuple(aug_cfg.get("crop_ratio", [0.9, 1.1]))
        ops.append(transforms.RandomResizedCrop(image_size, scale=scale, ratio=ratio))
    elif resize_mode in {"letterbox", "pad", "aspect_pad"}:
        ops.append(LetterboxResize(image_size))
    elif resize_mode in {"bottom_square", "bottom_center_square", "road_bottom_square"}:
        ops.append(BottomSquareCropResize(image_size))
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    if train:
        horizontal_flip_p = float(aug_cfg.get("horizontal_flip_p", 0.5))
        jitter = aug_cfg.get("color_jitter", {})
        brightness = float(jitter.get("brightness", 0.15))
        contrast = float(jitter.get("contrast", 0.15))
        saturation = float(jitter.get("saturation", 0.08))
        hue = float(jitter.get("hue", 0.02))
        if horizontal_flip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=horizontal_flip_p))
        ops.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )
        grayscale_p = float(aug_cfg.get("random_grayscale_p", 0.0))
        if grayscale_p > 0:
            ops.append(transforms.RandomGrayscale(p=grayscale_p))
        blur_p = float(aug_cfg.get("gaussian_blur_p", 0.0))
        if blur_p > 0:
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=tuple(aug_cfg.get("blur_sigma", [0.1, 1.2])))],
                    p=blur_p,
                )
            )
    ops.extend(
        [
            transforms.ToTensor(),
        ]
    )
    gray_world_alpha = float(aug_cfg.get("gray_world_alpha", 0.0))
    if gray_world_alpha > 0:
        ops.append(GrayWorldColorConstancy(alpha=gray_world_alpha))
    line_erase_p = float(aug_cfg.get("line_erasing_p", 0.0)) if train else 0.0
    if line_erase_p > 0:
        ops.append(
            RandomLineErasing(
                p=line_erase_p,
                num_lines=tuple(aug_cfg.get("line_erasing_num_lines", [1, 3])),
                length=tuple(aug_cfg.get("line_erasing_length", [0.35, 0.95])),
                width=tuple(aug_cfg.get("line_erasing_width", [0.015, 0.055])),
                orientations=tuple(aug_cfg.get("line_erasing_orientations", ["horizontal", "vertical"])),
            )
        )
    fourier_p = float(aug_cfg.get("fourier_low_freq_jitter_p", 0.0)) if train else 0.0
    if fourier_p > 0:
        ops.append(
            FourierLowFrequencyJitter(
                p=fourier_p,
                beta=float(aug_cfg.get("fourier_beta", 0.08)),
                strength=tuple(aug_cfg.get("fourier_strength", [0.75, 1.25])),
            )
        )
    ops.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    erase_p = float(aug_cfg.get("random_erasing_p", 0.0)) if train else 0.0
    if erase_p > 0:
        ops.append(
            transforms.RandomErasing(
                p=erase_p,
                scale=tuple(aug_cfg.get("erase_scale", [0.02, 0.08])),
                ratio=tuple(aug_cfg.get("erase_ratio", [0.3, 3.3])),
                value="random",
            )
        )
    return transforms.Compose(ops)


def build_mask_transforms(
    image_size: int,
    aug_cfg: dict | None = None,
    *,
    pretransformed: bool = False,
):
    """Build the deterministic geometry transform for cached soft road masks.

    Cached road masks are supervision targets, so they must stay aligned with
    the image geometry. This helper mirrors only the deterministic resize/crop
    stage from ``build_transforms`` and intentionally excludes random flips,
    color jitter, Fourier jitter, normalization, and erasing.
    """
    aug_cfg = aug_cfg or {}
    if bool(aug_cfg.get("random_resized_crop", False)):
        raise ValueError(
            "road_mask supervision requires deterministic geometry; disable "
            "random_resized_crop or precompute masks in the final image frame."
        )
    ops = []
    if pretransformed:
        ops.append(transforms.Resize((int(image_size), int(image_size)), interpolation=InterpolationMode.BILINEAR))
    else:
        resize_mode = str(aug_cfg.get("resize_mode", "stretch")).lower()
        if resize_mode in {"letterbox", "pad", "aspect_pad"}:
            resampling = getattr(Image, "Resampling", Image)
            ops.append(LetterboxResize(image_size, method=resampling.BILINEAR, fill=0))
        elif resize_mode in {"bottom_square", "bottom_center_square", "road_bottom_square"}:
            ops.append(BottomSquareCropResize(image_size, interpolation=InterpolationMode.BILINEAR))
        else:
            ops.append(transforms.Resize((int(image_size), int(image_size)), interpolation=InterpolationMode.BILINEAR))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class LetterboxResize:
    """Resize while preserving aspect ratio and padding to a square canvas.

    This is disabled by default. It is useful as a controlled candidate when
    native dataset aspect ratios are a suspected shortcut, because it avoids
    stretching a vertical RSCD frame into the same square geometry as RoadSaW.
    """

    def __init__(
        self,
        image_size: int,
        fill: tuple[int, int, int] | int = (0, 0, 0),
        method=None,
    ) -> None:
        self.image_size = int(image_size)
        self.fill = fill
        if method is None:
            resampling = getattr(Image, "Resampling", Image)
            method = resampling.BILINEAR
        self.method = method

    def __call__(self, image):
        return ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=self.method,
            color=self.fill,
            centering=(0.5, 0.5),
        )


class BottomSquareCropResize:
    """Crop a bottom-centered square before resizing.

    The transform removes a strong native aspect-ratio cue while biasing the
    image toward the road region closest to the vehicle. It is intentionally
    deterministic so train/val/test use the same geometric canonicalization.
    """

    def __init__(self, image_size: int, interpolation=InterpolationMode.BILINEAR) -> None:
        self.image_size = int(image_size)
        self.interpolation = interpolation

    def __call__(self, image):
        width, height = image.size
        side = min(width, height)
        left = max((width - side) // 2, 0)
        top = max(height - side, 0)
        cropped = image.crop((left, top, left + side, top + side))
        return transforms.functional.resize(cropped, [self.image_size, self.image_size], interpolation=self.interpolation)


class GrayWorldColorConstancy:
    """Apply a soft gray-world color-constancy correction.

    Public road datasets often differ by camera pipeline, file format, and
    illumination statistics. This deterministic transform reduces global color
    cast while keeping road texture and wetness reflections in the image.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-6) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.eps = float(eps)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0 or image.ndim != 3 or image.size(0) != 3:
            return image
        channel_mean = image.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        gray_mean = channel_mean.mean(dim=0, keepdim=True)
        corrected = (image * (gray_mean / channel_mean)).clamp(0.0, 1.0)
        if self.alpha >= 1.0:
            return corrected
        return ((1.0 - self.alpha) * image + self.alpha * corrected).clamp(0.0, 1.0)


class RandomLineErasing:
    """Erase thin elongated regions to discourage marking/grate shortcuts.

    RSCD patches often contain lane markings, arrows, drains, and manholes. This
    transform targets those thin high-salience structures while leaving most of
    the road texture visible. It operates before normalization, so erased pixels
    stay in the valid image range.
    """

    def __init__(
        self,
        p: float = 0.15,
        num_lines: tuple[int, int] = (1, 3),
        length: tuple[float, float] = (0.35, 0.95),
        width: tuple[float, float] = (0.015, 0.055),
        orientations: tuple[str, ...] = ("horizontal", "vertical"),
    ) -> None:
        self.p = float(p)
        self.num_lines = (int(num_lines[0]), int(num_lines[1]))
        self.length = (float(length[0]), float(length[1]))
        self.width = (float(width[0]), float(width[1]))
        valid = {"horizontal", "vertical"}
        self.orientations = tuple(item for item in orientations if str(item).lower() in valid) or ("horizontal", "vertical")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() > self.p or image.ndim != 3:
            return image
        _, height, width = image.shape
        if height <= 1 or width <= 1:
            return image
        out = image.clone()
        n_min, n_max = self.num_lines
        num_lines = random.randint(max(1, n_min), max(max(1, n_min), n_max))
        fill = out.mean(dim=(-2, -1), keepdim=True)
        for _ in range(num_lines):
            orientation = random.choice(self.orientations)
            if orientation == "horizontal":
                erase_h = max(1, int(round(random.uniform(*self.width) * height)))
                erase_w = max(1, int(round(random.uniform(*self.length) * width)))
                top = random.randint(0, max(height - erase_h, 0))
                left = random.randint(0, max(width - erase_w, 0))
                out[:, top : top + erase_h, left : left + erase_w] = fill
            else:
                erase_h = max(1, int(round(random.uniform(*self.length) * height)))
                erase_w = max(1, int(round(random.uniform(*self.width) * width)))
                top = random.randint(0, max(height - erase_h, 0))
                left = random.randint(0, max(width - erase_w, 0))
                out[:, top : top + erase_h, left : left + erase_w] = fill
        return out.clamp(0.0, 1.0)


class FourierLowFrequencyJitter:
    """Randomize low-frequency amplitude to reduce dataset style shortcuts.

    The transform is a lightweight domain-generalization augmentation inspired by
    Fourier-domain adaptation. It keeps phase intact, so geometry and labels are
    preserved, while low-frequency color/illumination style is perturbed.
    """

    def __init__(
        self,
        p: float = 0.25,
        beta: float = 0.08,
        strength: tuple[float, float] = (0.75, 1.25),
    ) -> None:
        self.p = float(p)
        self.beta = float(beta)
        self.strength = (float(strength[0]), float(strength[1]))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() > self.p:
            return image
        if image.ndim != 3:
            return image
        _, height, width = image.shape
        radius_h = max(int(height * self.beta), 1)
        radius_w = max(int(width * self.beta), 1)
        cy, cx = height // 2, width // 2

        fft = torch.fft.fft2(image, dim=(-2, -1))
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)
        amplitude = torch.fft.fftshift(amplitude, dim=(-2, -1))

        scale = torch.empty((image.size(0), 1, 1), device=image.device, dtype=image.dtype)
        scale.uniform_(self.strength[0], self.strength[1])
        amplitude[:, cy - radius_h : cy + radius_h + 1, cx - radius_w : cx + radius_w + 1] *= scale
        amplitude = torch.fft.ifftshift(amplitude, dim=(-2, -1))

        perturbed = torch.fft.ifft2(amplitude * torch.exp(1j * phase), dim=(-2, -1)).real
        return perturbed.clamp(0.0, 1.0)
