from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_MANIFEST_DIR = Path("data/manifests_full")
DEFAULT_OUT_DIR = Path("data/quality_flags")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-image quality/style flags that can be joined with "
            "model predictions without modifying the original manifests."
        )
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--max-images-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    manifests = args.manifest or sorted(args.manifest_dir.glob("*.csv"))
    if not manifests:
        raise FileNotFoundError(f"No manifest CSVs found under {args.manifest_dir}")

    out_csv = args.out_csv or args.out_dir / "image_quality_flags.csv"
    out_json = args.out_json or args.summary_dir / "image_quality_flags_summary.json"
    out_md = args.out_md or args.summary_dir / "image_quality_flags_summary.md"

    df = load_manifest_rows(manifests, set(args.dataset))
    records = build_quality_records(
        df,
        max_images_per_dataset=int(args.max_images_per_dataset),
        seed=int(args.seed),
    )
    report = summarize(records, manifests)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report, out_csv), encoding="utf-8")
    print(render_markdown(report, out_csv))
    print(f"wrote: {out_csv}")


def load_manifest_rows(manifests: list[Path], datasets: set[str]) -> pd.DataFrame:
    frames = []
    for path in manifests:
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        frame["manifest_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if datasets:
        keep = df["dataset"].astype(str).isin(datasets)
        df = df[keep].copy()
    if df.empty:
        raise ValueError("No manifest rows remain after dataset filtering.")
    df["image_path_norm"] = df["image_path"].astype(str).str.strip()
    df = df.drop_duplicates("image_path_norm").reset_index(drop=True)
    return df


def build_quality_records(
    df: pd.DataFrame,
    *,
    max_images_per_dataset: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    frames = []
    for dataset, sub in df.groupby("dataset", sort=True):
        sub = sub.copy()
        if max_images_per_dataset > 0 and len(sub) > max_images_per_dataset:
            indices = rng.choice(sub.index.to_numpy(), size=max_images_per_dataset, replace=False)
            sub = sub.loc[np.sort(indices)]
        frames.append(sub)
    work = pd.concat(frames, ignore_index=True)

    records: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        row_dict = row._asdict()
        image_path = str(row_dict.get("image_path_norm") or row_dict.get("image_path") or "").strip()
        base = {
            "image_path": image_path,
            "dataset": _clean(row_dict.get("dataset")),
            "split": _clean(row_dict.get("split")),
            "class_label": _clean(row_dict.get("class_label")),
            "friction_label": _clean(row_dict.get("friction_label")),
            "material_label": _clean(row_dict.get("material_label")),
            "wetness_label": _clean(row_dict.get("wetness_label")),
            "snow_label": _clean(row_dict.get("snow_label")),
            "risk_label": _clean(row_dict.get("risk_label")),
        }
        try:
            metrics = image_quality_metrics(Path(image_path))
            records.append({**base, **metrics})
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            records.append(
                {
                    **base,
                    "decode_ok": False,
                    "decode_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records


def image_quality_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    gray = arr.mean(axis=2)
    height, width = gray.shape
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = _mean_saturation(arr)
    white_pixel_frac = float(((arr[:, :, 0] > 0.95) & (arr[:, :, 1] > 0.95) & (arr[:, :, 2] > 0.95)).mean())
    black_pixel_frac = float(((arr[:, :, 0] < 0.05) & (arr[:, :, 1] < 0.05) & (arr[:, :, 2] < 0.05)).mean())
    specular_frac = float(((arr.max(axis=2) > 0.94) & (_pixel_saturation(arr) < 0.18)).mean())
    edge_strength = _edge_strength(gray)
    texture_energy = float(contrast + edge_strength)
    near_white_score = float(brightness + white_pixel_frac + 0.5 * specular_frac - contrast - 0.5 * saturation)

    near_white_flag = bool(
        brightness >= 0.82
        and white_pixel_frac >= 0.25
        and contrast <= 0.16
        and saturation <= 0.20
    )
    overexposed_flag = bool(white_pixel_frac >= 0.45 or (brightness >= 0.88 and contrast <= 0.12))
    low_contrast_flag = bool(contrast <= 0.08 and edge_strength <= 0.035)
    low_texture_flag = bool(texture_energy <= 0.13)
    dark_flag = bool(brightness <= 0.16)
    suspicious_quality_flag = bool(near_white_flag or overexposed_flag or low_contrast_flag or dark_flag)

    return {
        "decode_ok": True,
        "decode_error": "",
        "width": int(width),
        "height": int(height),
        "aspect": float(width / max(1, height)),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "white_pixel_frac": white_pixel_frac,
        "black_pixel_frac": black_pixel_frac,
        "specular_highlight_frac": specular_frac,
        "edge_strength": edge_strength,
        "texture_energy": texture_energy,
        "near_white_score": near_white_score,
        "near_white_flag": near_white_flag,
        "overexposed_flag": overexposed_flag,
        "low_contrast_flag": low_contrast_flag,
        "low_texture_flag": low_texture_flag,
        "dark_flag": dark_flag,
        "suspicious_quality_flag": suspicious_quality_flag,
    }


def summarize(records: list[dict[str, Any]], manifests: list[Path]) -> dict[str, Any]:
    decoded = [row for row in records if row.get("decode_ok")]
    by_dataset = {}
    for dataset in sorted({str(row.get("dataset")) for row in records}):
        rows = [row for row in records if str(row.get("dataset")) == dataset]
        ok_rows = [row for row in rows if row.get("decode_ok")]
        by_dataset[dataset] = _group_summary(ok_rows, total_rows=len(rows))
        by_dataset[dataset]["near_white_by_class"] = _flag_by_key(ok_rows, "class_label", "near_white_flag")
        by_dataset[dataset]["suspicious_by_class"] = _flag_by_key(ok_rows, "class_label", "suspicious_quality_flag")
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "manifests": [str(path) for path in manifests],
        "num_records": int(len(records)),
        "num_decoded": int(len(decoded)),
        "claim_boundary": (
            "Quality flags are low-level visual diagnostics. They do not prove "
            "that an image has an incorrect label or a measured friction value."
        ),
        "overall": _group_summary(decoded, total_rows=len(records)),
        "by_dataset": by_dataset,
    }


def _group_summary(rows: list[dict[str, Any]], *, total_rows: int) -> dict[str, Any]:
    return {
        "total_rows": int(total_rows),
        "decoded_rows": int(len(rows)),
        "decode_error_rows": int(total_rows - len(rows)),
        "near_white_count": int(sum(bool(row.get("near_white_flag")) for row in rows)),
        "near_white_rate": _rate(rows, "near_white_flag"),
        "overexposed_rate": _rate(rows, "overexposed_flag"),
        "low_contrast_rate": _rate(rows, "low_contrast_flag"),
        "low_texture_rate": _rate(rows, "low_texture_flag"),
        "suspicious_quality_rate": _rate(rows, "suspicious_quality_flag"),
        "brightness": _num_summary([row.get("brightness") for row in rows]),
        "contrast": _num_summary([row.get("contrast") for row in rows]),
        "saturation": _num_summary([row.get("saturation") for row in rows]),
        "white_pixel_frac": _num_summary([row.get("white_pixel_frac") for row in rows]),
        "texture_energy": _num_summary([row.get("texture_energy") for row in rows]),
        "dimension_top": dict(Counter(f"{row.get('width')}x{row.get('height')}" for row in rows).most_common(8)),
    }


def _flag_by_key(rows: list[dict[str, Any]], key: str, flag: str) -> dict[str, dict[str, Any]]:
    out = {}
    for value in sorted({str(row.get(key)) for row in rows}):
        group = [row for row in rows if str(row.get(key)) == value]
        count = int(sum(bool(row.get(flag)) for row in group))
        out[value] = {"count": count, "total": int(len(group)), "rate": float(count / len(group)) if group else None}
    return out


def render_markdown(report: dict[str, Any], out_csv: Path) -> str:
    lines = [
        "# Image Quality Flags",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Records: `{report['num_records']}`; decoded: `{report['num_decoded']}`",
        f"- CSV: `{out_csv}`",
        f"- Boundary: {report['claim_boundary']}",
        "",
        "## Dataset Summary",
        "",
        "| dataset | decoded | near-white | suspicious | median brightness | median contrast | main dimensions |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, row in report["by_dataset"].items():
        dims = ", ".join(f"{key}:{value}" for key, value in row.get("dimension_top", {}).items()) or "-"
        lines.append(
            "| {dataset} | {decoded} | {near_white} | {suspicious} | {brightness} | {contrast} | {dims} |".format(
                dataset=dataset,
                decoded=row["decoded_rows"],
                near_white=_fmt_pct(row.get("near_white_rate")),
                suspicious=_fmt_pct(row.get("suspicious_quality_rate")),
                brightness=_fmt_num(row.get("brightness", {}).get("median")),
                contrast=_fmt_num(row.get("contrast", {}).get("median")),
                dims=dims,
            )
        )
    lines += ["", "## RoadSaW Near-White By Class", ""]
    roadsaw = report.get("by_dataset", {}).get("roadsaw", {})
    rows = roadsaw.get("near_white_by_class", {})
    lines += ["| class | count | total | rate |", "|---|---:|---:|---:|"]
    for label, row in sorted(rows.items(), key=lambda item: item[1].get("rate") or 0.0, reverse=True):
        lines.append(f"| {label} | {row['count']} | {row['total']} | {_fmt_pct(row.get('rate'))} |")
    return "\n".join(lines) + "\n"


def _mean_saturation(arr: np.ndarray) -> float:
    return float(_pixel_saturation(arr).mean())


def _pixel_saturation(arr: np.ndarray) -> np.ndarray:
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    return np.where(mx <= 1e-6, 0.0, (mx - mn) / np.maximum(mx, 1e-6))


def _edge_strength(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return float(0.5 * (dx + dy))


def _num_summary(values: list[Any]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return {"min": None, "p05": None, "median": None, "mean": None, "p95": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _rate(rows: list[dict[str, Any]], flag: str) -> float | None:
    if not rows:
        return None
    return float(sum(bool(row.get(flag)) for row in rows) / len(rows))


def _fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}%"


def _fmt_num(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
