from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_high_error_feature_values import canonical, extract_feature_vector, write_rows


def _as_bool(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--strict-col", default="strict_pred")
    parser.add_argument("--candidate-col", default="wet_pred")
    parser.add_argument("--one-correct-only", action="store_true")
    parser.add_argument("--no-fft", action="store_true")
    args = parser.parse_args()

    with args.comparison.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    selected: list[dict[str, str]] = []
    for row in rows:
        if str(row.get(args.strict_col, "")) == str(row.get(args.candidate_col, "")):
            continue
        if args.one_correct_only and (_as_bool(row.get("strict_ok")) == _as_bool(row.get("wet_ok"))):
            continue
        selected.append(row)

    out_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    for index, row in enumerate(selected, start=1):
        feats = extract_feature_vector(row["image_path"], args.image_size, include_fft=not args.no_fft)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        out: dict[str, object] = {
            "image_path": row["image_path"],
            "label": canonical(row.get("true_label", "")),
            "pred_label": canonical(row.get(args.strict_col, "")),
            "candidate_label": canonical(row.get(args.candidate_col, "")),
            "strict_ok": row.get("strict_ok", ""),
            "candidate_ok": row.get("wet_ok", ""),
            "case": row.get("case", ""),
        }
        for key in feature_names:
            out[key] = feats[key]
        out_rows.append(out)
        if index % 100 == 0:
            print(f"extracted {index}/{len(selected)} feature rows", flush=True)

    write_rows(args.out_csv, out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
