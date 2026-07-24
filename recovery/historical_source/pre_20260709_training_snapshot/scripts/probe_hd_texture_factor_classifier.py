"""High-definition texture probe for RSCD factor bottlenecks.

This probe tests whether camera-visible micro/meso texture statistics contain
factor information that the current ConvNeXt+PhysicsTexture model still misses.
It is deliberately offline: the output decides whether an HD texture teacher is
worth distilling into the main model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


FACTOR_NAMES = ("friction", "material", "unevenness")


def label_to_factors(label: str) -> dict[str, str]:
    text = str(label).replace("-", "_")
    if text in {"fresh_snow", "melted_snow", "ice"}:
        return {"friction": text, "material": "winter", "unevenness": "winter"}
    parts = text.split("_")
    if len(parts) == 2:
        return {"friction": parts[0], "material": parts[1], "unevenness": "granular"}
    if len(parts) >= 3:
        return {"friction": parts[0], "material": parts[1], "unevenness": parts[2]}
    return {"friction": "unknown", "material": "unknown", "unevenness": "unknown"}


def sample_per_class(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    chunks = []
    for _, group in df.groupby("class_label", sort=True):
        chunks.append(group.sample(n=min(int(n), len(group)), random_state=seed))
    return pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def frame_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for _, row in df.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_batch(paths: list[str], device: torch.device, image_size: tuple[int, int]) -> torch.Tensor:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((image_size[1], image_size[0]), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(images, dim=0).to(device=device)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    lo = flat.quantile(0.05, dim=2, keepdim=True).view(x.shape[0], x.shape[1], 1, 1)
    hi = flat.quantile(0.95, dim=2, keepdim=True).view(x.shape[0], x.shape[1], 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-4)).clamp(0.0, 1.0)


def channel_stats(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    q10 = flat.quantile(0.10, dim=2)
    q50 = flat.quantile(0.50, dim=2)
    q90 = flat.quantile(0.90, dim=2)
    return torch.cat(
        [
            x.mean(dim=(2, 3)),
            x.std(dim=(2, 3)),
            q10,
            q50,
            q90,
            x.amax(dim=(2, 3)),
        ],
        dim=1,
    )


def grid_stats(x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, grid).flatten(1)


def soft_connectedness(field: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool2d(field, kernel_size=5, stride=1, padding=2)
    return (field * pooled).mean(dim=(2, 3))


def extract_hd_texture_features(batch: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
    maxc = batch.max(dim=1, keepdim=True).values
    minc = batch.min(dim=1, keepdim=True).values
    value = maxc
    saturation = (maxc - minc) / maxc.clamp_min(1e-4)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, 1, 3, 3) / 8.0
    laplace = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
    lap = F.conv2d(gray, laplace, padding=1).abs()
    local_mean9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    local_mean31 = F.avg_pool2d(gray, kernel_size=31, stride=1, padding=15)
    local_mean63 = F.avg_pool2d(gray, kernel_size=63, stride=1, padding=31)
    local_contrast = F.avg_pool2d((gray - local_mean9).abs(), kernel_size=9, stride=1, padding=4)

    micro_band = (gray - local_mean9).abs()
    meso_band = (local_mean9 - local_mean31).abs()
    macro_band = (local_mean31 - local_mean63).abs()

    grad_norm = normalize_map(grad)
    lap_norm = normalize_map(lap)
    contrast_norm = normalize_map(local_contrast)
    rough_energy = torch.clamp(0.40 * grad_norm + 0.35 * lap_norm + 0.25 * contrast_norm, 0.0, 1.0)

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
    marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)

    orientation_balance = (gx.abs() - gy.abs()) / (gx.abs() + gy.abs() + 1e-4)
    diagonal_balance = (gx + gy).abs() / (gx.abs() + gy.abs() + 1e-4)

    fields = [
        gray,
        value,
        saturation,
        grad,
        lap,
        local_contrast,
        micro_band,
        meso_band,
        macro_band,
        rough_energy,
        low_texture,
        low_contrast,
        specular,
        dark_water,
        thin_film,
        texture_erasure,
        marking,
        orientation_balance.abs(),
        diagonal_balance,
    ]
    stack = torch.cat(fields, dim=1)
    feature_parts = [channel_stats(stack)]
    for grid in ((2, 3), (4, 6), (6, 9)):
        feature_parts.append(grid_stats(stack, grid))
    topology_fields = [rough_energy, low_texture, specular, dark_water, thin_film, texture_erasure]
    feature_parts.append(torch.cat([soft_connectedness(x) for x in topology_fields], dim=1))
    return torch.cat(feature_parts, dim=1)


def extract_for_df(df: pd.DataFrame, device: torch.device, batch_size: int, image_size: tuple[int, int]) -> np.ndarray:
    paths = df["image_path"].astype(str).tolist()
    chunks: list[np.ndarray] = []
    for batch_paths in tqdm(list(batched(paths, batch_size)), desc="extract"):
        batch = load_batch(batch_paths, device, image_size)
        with torch.no_grad():
            feat = extract_hd_texture_features(batch).float().cpu().numpy()
        chunks.append(feat)
    return np.concatenate(chunks, axis=0)


def train_and_eval(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    if model_name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260702,
            n_jobs=-1,
        )
    elif model_name == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=600, class_weight="balanced", n_jobs=-1, solver="saga", C=2.0),
        )
    else:
        raise ValueError(f"unknown classifier: {model_name}")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    report = classification_report(y_test, pred, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_train.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=240)
    parser.add_argument("--test-per-class", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = sample_per_class(pd.read_csv(args.train_manifest), int(args.train_per_class), int(args.seed))
    test_df = sample_per_class(pd.read_csv(args.test_manifest), int(args.test_per_class), int(args.seed))
    for frame in (train_df, test_df):
        factors = frame["class_label"].map(label_to_factors)
        for name in FACTOR_NAMES:
            frame[f"factor_{name}"] = factors.map(lambda item, n=name: item[n])
        frame["class_label_norm"] = frame["class_label"].astype(str).str.replace("-", "_", regex=False)

    train_df.to_csv(args.output_dir / "train_sample.csv", index=False)
    test_df.to_csv(args.output_dir / "test_sample.csv", index=False)

    image_size = (int(args.image_height), int(args.image_width))
    x_train = extract_for_df(train_df, device, int(args.batch_size), image_size)
    x_test = extract_for_df(test_df, device, int(args.batch_size), image_size)
    np.save(args.output_dir / "x_train.npy", x_train)
    np.save(args.output_dir / "x_test.npy", x_test)

    metrics: dict[str, object] = {
        "device": str(device),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "feature_dim": int(x_train.shape[1]),
        "image_size": list(image_size),
    }
    rows = []
    targets = {"class27": "class_label_norm"}
    targets.update({name: f"factor_{name}" for name in FACTOR_NAMES})
    for target_name, column in targets.items():
        y_train = train_df[column].astype(str).to_numpy()
        y_test = test_df[column].astype(str).to_numpy()
        for model_name in ("extra_trees", "logreg"):
            result = train_and_eval(x_train, x_test, y_train, y_test, model_name)
            metrics[f"{target_name}_{model_name}"] = result
            rows.append(
                {
                    "target": target_name,
                    "classifier": model_name,
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "weighted_f1": result["weighted_f1"],
                }
            )

    with open(args.output_dir / "hd_texture_probe_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "hd_texture_probe_summary.csv", index=False)
    md = [
        "# HD Texture Factor Probe",
        "",
        f"- train samples: {len(train_df)} ({args.train_per_class}/class cap)",
        f"- test samples: {len(test_df)} ({args.test_per_class}/class cap)",
        f"- feature dim: {x_train.shape[1]}",
        f"- device: `{device}`",
        "",
        frame_to_markdown(summary),
        "",
        "Decision rule: promote this direction only if the probe shows strong roughness/factor signal and suggests a distillable teacher rather than another late classifier patch.",
    ]
    (args.output_dir / "hd_texture_probe_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
