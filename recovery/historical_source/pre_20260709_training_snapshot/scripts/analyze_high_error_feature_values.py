from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-6


def canonical(label: str) -> str:
    return str(label).strip().replace("-", "_")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def image_blur(gray: np.ndarray, radius: int) -> np.ndarray:
    img = Image.fromarray(np.uint8(np.clip(gray, 0.0, 1.0) * 255.0), mode="L")
    return np.asarray(img.filter(ImageFilter.BoxBlur(radius=radius)), dtype=np.float32) / 255.0


def connectedness(mask: np.ndarray) -> float:
    mask_u8 = np.uint8(np.clip(mask, 0.0, 1.0) * 255.0)
    pooled = np.asarray(Image.fromarray(mask_u8, mode="L").filter(ImageFilter.MaxFilter(size=9)), dtype=np.float32) / 255.0
    return float((mask * pooled).mean())


def safe_ratio(a: float, b: float) -> float:
    return float(a / (b + EPS))


def add_basic_stats(out: dict[str, float], name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    out[f"{name}_mean"] = float(arr.mean())
    out[f"{name}_std"] = float(arr.std())
    out[f"{name}_p90"] = float(np.quantile(arr, 0.90))
    out[f"{name}_cover_0p5"] = float((arr > 0.5).mean())


def radial_frequency_features(gray: np.ndarray) -> dict[str, float]:
    h, w = gray.shape
    centered = gray - float(gray.mean())
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (float(rr.max()) + EPS)
    total = float(spectrum.sum() + EPS)
    low = float(spectrum[rr <= 0.15].sum() / total)
    mid = float(spectrum[(rr > 0.15) & (rr <= 0.35)].sum() / total)
    high = float(spectrum[rr > 0.35].sum() / total)
    return {
        "fft_low_ratio": low,
        "fft_mid_ratio": mid,
        "fft_high_ratio": high,
        "fft_high_mid_ratio": safe_ratio(high, mid),
        "fft_mid_low_ratio": safe_ratio(mid, low),
    }


def gradient_orientation_features(gx: np.ndarray, gy: np.ndarray, grad: np.ndarray) -> dict[str, float]:
    w = grad.reshape(-1) + EPS
    gx_f = gx.reshape(-1)
    gy_f = gy.reshape(-1)
    cxx = float((w * gx_f * gx_f).sum() / w.sum())
    cyy = float((w * gy_f * gy_f).sum() / w.sum())
    cxy = float((w * gx_f * gy_f).sum() / w.sum())
    trace = cxx + cyy + EPS
    det_term = math.sqrt(max((cxx - cyy) ** 2 + 4.0 * cxy * cxy, 0.0))
    lam1 = 0.5 * (trace + det_term)
    lam2 = 0.5 * (trace - det_term)
    return {
        "grad_anisotropy": float((lam1 - lam2) / (lam1 + lam2 + EPS)),
        "grad_orientation_energy": float(lam1 + lam2),
        "grad_horizontal_ratio": safe_ratio(float(np.abs(gx).mean()), float(np.abs(gy).mean())),
    }


def extract_feature_vector(image_path: str, image_size: int, *, include_fft: bool = True) -> dict[str, float]:
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    value = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = (value - minc) / np.maximum(value, 1e-4)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy + EPS)
    pad = np.pad(gray, 1, mode="edge")
    lap = np.abs(pad[0:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, 0:-2] + pad[1:-1, 2:] - 4.0 * gray) / 4.0
    blur3 = image_blur(gray, 1)
    blur9 = image_blur(gray, 4)
    blur21 = image_blur(gray, 10)
    contrast = np.abs(gray - blur9)
    micro = np.abs(gray - blur3)
    meso = np.abs(blur3 - blur9)
    macro = np.abs(blur9 - blur21)

    specular = sigmoid((value - 0.80) * 14.0) * sigmoid((0.28 - saturation) * 12.0)
    low_texture = sigmoid((0.055 - grad) * 30.0)
    low_contrast = sigmoid((0.050 - contrast) * 30.0)
    dark_water = sigmoid((0.42 - value) * 10.0) * sigmoid((0.34 - saturation) * 10.0) * low_texture
    wet = np.clip(specular + 0.65 * dark_water, 0.0, 1.0)
    rough = np.clip(0.42 * grad + 0.30 * lap + 0.28 * contrast, 0.0, 1.0)
    film_erasure = low_texture * low_contrast
    texture_erasure = sigmoid((0.050 - grad) * 32.0) * sigmoid((0.035 - contrast) * 40.0)
    snow_like = sigmoid((value - 0.72) * 12.0) * sigmoid((0.30 - saturation) * 12.0)
    ice_like = specular * low_texture * sigmoid((0.22 - saturation) * 12.0)
    marking_like = sigmoid((value - 0.76) * 15.0) * sigmoid((grad - 0.08) * 18.0)
    concrete_like = sigmoid((value - 0.52) * 8.0) * sigmoid((0.34 - saturation) * 8.0)
    asphalt_like = sigmoid((0.58 - value) * 8.0) * sigmoid((micro + contrast - 0.065) * 18.0)
    visible_roughness = sigmoid((grad - 0.060) * 22.0) * sigmoid((lap - 0.052) * 20.0)
    paved_boundary = sigmoid((visible_roughness + film_erasure + specular + dark_water - 0.85) * 5.0)

    out: dict[str, float] = {}
    for name, arr in [
        ("r", r),
        ("g", g),
        ("b", b),
        ("gray", gray),
        ("value", value),
        ("saturation", saturation),
        ("grad", grad),
        ("lap", lap),
        ("contrast", contrast),
        ("micro", micro),
        ("meso", meso),
        ("macro", macro),
        ("specular", specular),
        ("dark_water", dark_water),
        ("wet", wet),
        ("rough", rough),
        ("film_erasure", film_erasure),
        ("texture_erasure", texture_erasure),
        ("snow_like", snow_like),
        ("ice_like", ice_like),
        ("marking_like", marking_like),
        ("concrete_like", concrete_like),
        ("asphalt_like", asphalt_like),
        ("visible_roughness", visible_roughness),
        ("paved_boundary", paved_boundary),
    ]:
        add_basic_stats(out, name, arr)

    h = gray.shape[0]
    bottom = slice(int(h * 2 / 3), h)
    top = slice(0, int(h / 3))
    for name, arr in [
        ("wet", wet),
        ("rough", rough),
        ("concrete_like", concrete_like),
        ("asphalt_like", asphalt_like),
        ("grad", grad),
        ("marking_like", marking_like),
    ]:
        b_mean = float(arr[bottom, :].mean())
        t_mean = float(arr[top, :].mean())
        out[f"{name}_bottom_minus_top"] = b_mean - t_mean
        out[f"{name}_bottom_top_ratio"] = safe_ratio(b_mean, t_mean)

    out["wet_connectedness"] = connectedness(wet)
    out["rough_connectedness"] = connectedness(visible_roughness)
    out["marking_connectedness"] = connectedness(marking_like)
    out["rough_under_wet_mean"] = float((rough * wet).mean())
    out["rough_without_wet_mean"] = float((rough * (1.0 - wet)).mean())
    out["concrete_wet_mean"] = float((concrete_like * wet).mean())
    out["asphalt_wet_mean"] = float((asphalt_like * wet).mean())
    out["concrete_rough_mean"] = float((concrete_like * rough).mean())
    out["asphalt_rough_mean"] = float((asphalt_like * rough).mean())
    out["wet_rough_ratio"] = safe_ratio(float(wet.mean()), float(rough.mean()))
    out["concrete_asphalt_ratio"] = safe_ratio(float(concrete_like.mean()), float(asphalt_like.mean()))
    if include_fft:
        out.update(radial_frequency_features(gray))
    out.update(gradient_orientation_features(gx, gy, grad))
    return out


def select_targets(
    metrics_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    bottom_n: int,
    pair_count: int,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    ranked = sorted(
        [
            {
                "class": canonical(row["class"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "support": float(row["support"]),
            }
            for row in metrics_rows
        ],
        key=lambda item: item["f1"],
    )
    bottom = ranked[:bottom_n]
    bottom_set = {str(row["class"]) for row in bottom}
    conf = Counter()
    for row in prediction_rows:
        t = canonical(row["true_label"])
        p = canonical(row["pred_label"])
        if t != p and (t in bottom_set or p in bottom_set):
            conf[(t, p)] += 1
    pair_rows = [
        {"true": pair[0], "pred": pair[1], "count": count}
        for pair, count in conf.most_common(pair_count)
    ]
    target_set = set(bottom_set)
    for row in pair_rows:
        target_set.add(str(row["true"]))
        target_set.add(str(row["pred"]))
    targets = sorted(target_set)
    return targets, bottom, pair_rows


def sample_manifest_rows(rows: list[dict[str, str]], target_classes: set[str], samples_per_class: int, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = canonical(row["class_label"])
        if label in target_classes:
            row = dict(row)
            row["class_label"] = label
            by_class[label].append(row)
    sampled: list[dict[str, str]] = []
    for label, items in sorted(by_class.items()):
        if len(items) > samples_per_class:
            idx = rng.choice(len(items), size=samples_per_class, replace=False)
            sampled.extend(items[int(i)] for i in idx)
        else:
            sampled.extend(items)
    return sampled


def prediction_target_rows(
    rows: list[dict[str, str]],
    target_classes: set[str],
    include_true: bool,
    *,
    samples_per_true_class: int | None = None,
    seed: int = 1701,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        true_label = canonical(row["true_label"])
        pred_label = canonical(row["pred_label"])
        if pred_label in target_classes or (include_true and true_label in target_classes):
            selected.append(
                {
                    "image_path": row["image_path"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": row.get("confidence", ""),
                }
            )
    if samples_per_true_class is not None and samples_per_true_class > 0:
        rng = np.random.default_rng(seed)
        by_true: dict[str, list[dict[str, str]]] = defaultdict(list)
        extra: list[dict[str, str]] = []
        for row in selected:
            true_label = canonical(row["true_label"])
            if true_label in target_classes:
                by_true[true_label].append(row)
            else:
                extra.append(row)
        capped: list[dict[str, str]] = []
        for label, items in sorted(by_true.items()):
            if len(items) > samples_per_true_class:
                idx = rng.choice(len(items), size=samples_per_true_class, replace=False)
                capped.extend(items[int(i)] for i in idx)
            else:
                capped.extend(items)
        # Keep a small deterministic sample of non-target true labels whose base
        # prediction fell into a target class; they are useful for rerank damage
        # checks without making the quick diagnostic crawl through the full test set.
        if extra:
            cap = min(len(extra), max(samples_per_true_class, 100))
            idx = rng.choice(len(extra), size=cap, replace=False)
            capped.extend(extra[int(i)] for i in idx)
        selected = capped
    return selected


def feature_rows_from_records(
    records: list[dict[str, str]],
    image_size: int,
    label_key: str,
    pred_key: str | None = None,
    *,
    include_fft: bool = True,
) -> tuple[list[dict[str, object]], list[str]]:
    out_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    for i, row in enumerate(records, start=1):
        feats = extract_feature_vector(row["image_path"], image_size, include_fft=include_fft)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        out: dict[str, object] = {
            "image_path": row["image_path"],
            "label": canonical(row[label_key]),
        }
        if pred_key is not None:
            out["pred_label"] = canonical(row[pred_key])
            out["base_confidence"] = row.get("confidence", "")
        for key in feature_names:
            out[key] = feats[key]
        out_rows.append(out)
        if i % 1000 == 0:
            print(f"extracted {i}/{len(records)} feature rows", flush=True)
    return out_rows, feature_names or []


def rows_to_matrix(rows: list[dict[str, object]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y_labels = [str(row["label"]) for row in rows]
    classes = sorted(set(y_labels))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y = np.asarray([class_to_idx[name] for name in y_labels], dtype=np.int64)
    return X, y, classes


def cohen_abs(a: np.ndarray, b: np.ndarray) -> float:
    va = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    vb = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled = math.sqrt(max(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1), EPS))
    return abs(float(a.mean()) - float(b.mean())) / pooled


def auc_1d(a: np.ndarray, b: np.ndarray) -> float:
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    n1 = float((labels == 1).sum())
    n0 = float((labels == 0).sum())
    rank_sum = float(ranks[labels == 1].sum())
    auc = (rank_sum - n1 * (n1 + 1.0) / 2.0) / max(n1 * n0, EPS)
    return max(auc, 1.0 - auc)


def pair_feature_differences(
    rows: list[dict[str, object]],
    feature_names: list[str],
    pairs: Iterable[tuple[str, str]],
    top_k: int,
) -> list[dict[str, object]]:
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["label"])].append(row)
    output: list[dict[str, object]] = []
    seen = set()
    for a_label, b_label in pairs:
        a_label = canonical(a_label)
        b_label = canonical(b_label)
        if a_label == b_label:
            continue
        key = tuple(sorted([a_label, b_label]))
        if key in seen:
            continue
        seen.add(key)
        a_rows = by_class.get(a_label, [])
        b_rows = by_class.get(b_label, [])
        if len(a_rows) < 8 or len(b_rows) < 8:
            continue
        scored = []
        for feat in feature_names:
            av = np.asarray([float(row[feat]) for row in a_rows], dtype=np.float32)
            bv = np.asarray([float(row[feat]) for row in b_rows], dtype=np.float32)
            d = cohen_abs(av, bv)
            scored.append(
                {
                    "class_a": a_label,
                    "class_b": b_label,
                    "feature": feat,
                    "n_a": len(a_rows),
                    "n_b": len(b_rows),
                    "mean_a": float(av.mean()),
                    "mean_b": float(bv.mean()),
                    "delta_b_minus_a": float(bv.mean() - av.mean()),
                    "cohen_abs": d,
                    "auc_abs": auc_1d(av, bv),
                }
            )
        output.extend(sorted(scored, key=lambda row: (row["cohen_abs"], row["auc_abs"]), reverse=True)[:top_k])
    return output


def feature_importance_targets(pair_rows: list[dict[str, object]], feature_names: list[str], max_features: int) -> list[str]:
    score = Counter()
    for row in pair_rows:
        score[str(row["feature"])] += float(row["cohen_abs"]) + max(float(row["auc_abs"]) - 0.5, 0.0)
    selected = [feat for feat, _ in score.most_common(max_features)]
    if not selected:
        selected = feature_names[:max_features]
    return selected


def feature_space_augment(
    X: np.ndarray,
    y: np.ndarray,
    selected_idx: list[int],
    copies: int,
    jitter_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    chunks = [X]
    labels = [y]
    class_ids = sorted(set(int(v) for v in y.tolist()))
    class_std = {}
    for cls in class_ids:
        x_cls = X[y == cls]
        class_std[cls] = x_cls.std(axis=0) + 1e-4
    for _ in range(copies):
        X_aug = X.copy()
        for cls in class_ids:
            mask = y == cls
            noise = rng.normal(0.0, jitter_scale, size=(int(mask.sum()), len(selected_idx))).astype(np.float32)
            std = class_std[cls][selected_idx].reshape(1, -1)
            X_aug[np.ix_(mask, selected_idx)] += noise * std
        chunks.append(X_aug)
        labels.append(y.copy())
    return np.vstack(chunks), np.concatenate(labels)


def evaluate_feature_classifiers(
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, str]],
    target_classes: list[str],
    feature_names: list[str],
    pair_diff_rows: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    target_set = set(target_classes)
    train_target = [row for row in train_rows if str(row["label"]) in target_set]
    test_true_target = [row for row in test_rows if str(row["label"]) in target_set]
    X_train, y_train, classes = rows_to_matrix(train_target, feature_names)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    test_true_target = [row for row in test_true_target if str(row["label"]) in class_to_idx]
    X_test, y_test, _ = rows_to_matrix(test_true_target, feature_names)
    y_test = np.asarray([class_to_idx[str(row["label"])] for row in test_true_target], dtype=np.int64)

    selected_features = feature_importance_targets(pair_diff_rows, feature_names, max_features=36)
    selected_idx = [feature_names.index(name) for name in selected_features if name in feature_names]
    X_aug, y_aug = feature_space_augment(X_train, y_train, selected_idx, copies=2, jitter_scale=0.12, seed=seed)

    models = {
        "logistic_raw": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1600, class_weight="balanced", C=1.0, solver="lbfgs"),
        ),
        "logistic_value_augmented": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1600, class_weight="balanced", C=0.8, solver="lbfgs"),
        ),
        "random_forest_values": RandomForestClassifier(
            n_estimators=360,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_values": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            random_state=seed,
        ),
    }
    results: dict[str, object] = {
        "target_classes": target_classes,
        "selected_augmented_features": selected_features,
        "num_train": int(len(y_train)),
        "num_test_true_target": int(len(y_test)),
        "class_order": classes,
    }
    trained = {}
    for name, model in models.items():
        if name == "logistic_value_augmented":
            model.fit(X_aug, y_aug)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "target_true_accuracy": float(accuracy_score(y_test, pred)),
            "target_true_macro_f1": float(f1_score(y_test, pred, average="macro")),
            "classification_report": classification_report(y_test, pred, target_names=classes, output_dict=True, zero_division=0),
        }
        trained[name] = model

    # Rerank only samples whose base prediction belongs to target classes. This is
    # an inference-feasible check because it does not use the true label to decide
    # whether the feature classifier is allowed to intervene.
    best_name = max(
        models,
        key=lambda name: float(results[name]["target_true_macro_f1"]),  # type: ignore[index]
    )
    best_model = trained[best_name]
    pred_rows_by_path = {row["image_path"]: row for row in prediction_rows}
    test_by_path = {str(row["image_path"]): row for row in test_rows}
    full_true = []
    full_pred = []
    full_rerank = []
    target_pred_paths = []
    for row in prediction_rows:
        true_label = canonical(row["true_label"])
        pred_label = canonical(row["pred_label"])
        full_true.append(true_label)
        full_pred.append(pred_label)
        full_rerank.append(pred_label)
        if pred_label in target_set and row["image_path"] in test_by_path:
            target_pred_paths.append(row["image_path"])
    if target_pred_paths and hasattr(best_model, "predict_proba"):
        X_pred_target = np.asarray(
            [[float(test_by_path[path][feat]) for feat in feature_names] for path in target_pred_paths],
            dtype=np.float32,
        )
        prob = best_model.predict_proba(X_pred_target)
        pred_idx = prob.argmax(axis=1)
        pred_conf = prob.max(axis=1)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        path_to_suggestion = {
            path: (idx_to_class[int(cls_idx)], float(conf))
            for path, cls_idx, conf in zip(target_pred_paths, pred_idx, pred_conf, strict=True)
        }
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
        rerank_results = []
        for threshold in thresholds:
            changed = 0
            cand = list(full_pred)
            for i, row in enumerate(prediction_rows):
                item = path_to_suggestion.get(row["image_path"])
                if item is None:
                    continue
                suggested, conf = item
                if conf >= threshold and suggested != cand[i]:
                    cand[i] = suggested
                    changed += 1
            rerank_results.append(
                {
                    "threshold": threshold,
                    "changed": changed,
                    "full_accuracy": float(accuracy_score(full_true, cand)),
                    "full_macro_f1": float(f1_score(full_true, cand, average="macro")),
                }
            )
        results["rerank_best_model"] = best_name
        results["rerank_base_full_accuracy"] = float(accuracy_score(full_true, full_pred))
        results["rerank_base_full_macro_f1"] = float(f1_score(full_true, full_pred, average="macro"))
        results["rerank_thresholds"] = rerank_results
    else:
        results["rerank_note"] = "No target-predicted test rows or classifier lacks probabilities."
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--per-class-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bottom-n", type=int, default=10)
    parser.add_argument("--pair-count", type=int, default=24)
    parser.add_argument("--train-samples-per-class", type=int, default=900)
    parser.add_argument("--test-samples-per-class", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = read_csv(args.per_class_metrics)
    prediction_rows = read_csv(args.predictions)
    targets, bottom_rows, confusion_rows = select_targets(metrics_rows, prediction_rows, args.bottom_n, args.pair_count)
    target_set = set(targets)
    write_rows(args.out_dir / "bottom_f1_classes.csv", bottom_rows)
    write_rows(args.out_dir / "top_confusions_involving_bottom_classes.csv", confusion_rows)
    (args.out_dir / "target_classes.json").write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"target classes ({len(targets)}): {', '.join(targets)}", flush=True)

    train_manifest = read_csv(args.train_manifest)
    train_records = sample_manifest_rows(train_manifest, target_set, args.train_samples_per_class, args.seed)
    test_records = prediction_target_rows(
        prediction_rows,
        target_set,
        include_true=True,
        samples_per_true_class=args.test_samples_per_class if args.test_samples_per_class > 0 else None,
        seed=args.seed,
    )
    print(f"train target records: {len(train_records)}; test target/pred records: {len(test_records)}", flush=True)

    train_features, feature_names = feature_rows_from_records(
        train_records,
        args.image_size,
        "class_label",
        include_fft=not args.no_fft,
    )
    test_features, _ = feature_rows_from_records(
        test_records,
        args.image_size,
        "true_label",
        pred_key="pred_label",
        include_fft=not args.no_fft,
    )
    write_rows(args.out_dir / "train_target_feature_values.csv", train_features)
    write_rows(args.out_dir / "test_target_or_pred_feature_values.csv", test_features)
    (args.out_dir / "feature_names.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")

    pairs = [(str(row["true"]), str(row["pred"])) for row in confusion_rows]
    # Also compare every bottom class against its nearest selected target class.
    bottom_classes = [str(row["class"]) for row in bottom_rows]
    for i, a in enumerate(bottom_classes):
        for b in bottom_classes[i + 1 :]:
            pairs.append((a, b))
    pair_diffs = pair_feature_differences(train_features, feature_names, pairs, top_k=10)
    write_rows(args.out_dir / "pair_feature_differences_top10.csv", pair_diffs)

    clf_results = evaluate_feature_classifiers(
        train_features,
        test_features,
        prediction_rows,
        targets,
        feature_names,
        pair_diffs,
        args.seed,
    )
    (args.out_dir / "feature_classifier_results.json").write_text(
        json.dumps(clf_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# High-error RSCD feature-value diagnosis",
        "",
        f"Target classes: {len(targets)}",
        "",
        "## Bottom F1 classes",
        "",
    ]
    for row in bottom_rows:
        summary_lines.append(f"- {row['class']}: F1={float(row['f1'])*100:.2f}%, P={float(row['precision'])*100:.2f}%, R={float(row['recall'])*100:.2f}%")
    summary_lines += ["", "## Feature classifier check", ""]
    for name in ["logistic_raw", "logistic_value_augmented", "random_forest_values", "hist_gradient_values"]:
        item = clf_results.get(name, {})
        if isinstance(item, dict):
            summary_lines.append(
                f"- {name}: target true Acc={float(item['target_true_accuracy'])*100:.2f}%, "
                f"Macro-F1={float(item['target_true_macro_f1'])*100:.2f}%"
            )
    summary_lines += ["", "## Full-prediction feature rerank check", ""]
    summary_lines.append(
        f"- Base full Acc={float(clf_results.get('rerank_base_full_accuracy', 0.0))*100:.2f}%, "
        f"Macro-F1={float(clf_results.get('rerank_base_full_macro_f1', 0.0))*100:.2f}%"
    )
    for row in clf_results.get("rerank_thresholds", []):
        summary_lines.append(
            f"- threshold {row['threshold']:.2f}: changed={row['changed']}, "
            f"Acc={row['full_accuracy']*100:.2f}%, Macro-F1={row['full_macro_f1']*100:.2f}%"
        )
    summary_lines += ["", "## Top discriminative feature values", ""]
    for row in sorted(pair_diffs, key=lambda item: float(item["cohen_abs"]), reverse=True)[:30]:
        summary_lines.append(
            f"- {row['class_a']} vs {row['class_b']}: {row['feature']} "
            f"d={float(row['cohen_abs']):.2f}, AUC={float(row['auc_abs']):.3f}, "
            f"mean_a={float(row['mean_a']):.4f}, mean_b={float(row['mean_b']):.4f}"
        )
    (args.out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
