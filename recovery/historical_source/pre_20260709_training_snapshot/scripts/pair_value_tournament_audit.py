from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def canonical(label: str) -> str:
    return str(label).strip().replace("-", "_")


def rows_to_matrix(
    rows: list[dict[str, str]], feature_names: list[str], class_order: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = [canonical(row["label"]) for row in rows]
    classes = class_order or sorted(set(labels))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    kept = [i for i, label in enumerate(labels) if label in class_to_idx]
    x = np.asarray([[float(rows[i][name]) for name in feature_names] for i in kept], dtype=np.float32)
    y = np.asarray([class_to_idx[labels[i]] for i in kept], dtype=np.int64)
    return x, y, classes


def cohen_abs(a: np.ndarray, b: np.ndarray) -> float:
    va = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    vb = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled = math.sqrt(max(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1), EPS))
    return abs(float(a.mean()) - float(b.mean())) / pooled


def auc_abs(a: np.ndarray, b: np.ndarray) -> float:
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    n0 = float((labels == 0).sum())
    n1 = float((labels == 1).sum())
    rank_sum = float(ranks[labels == 1].sum())
    auc = (rank_sum - n1 * (n1 + 1.0) / 2.0) / max(n0 * n1, EPS)
    return max(auc, 1.0 - auc)


def top_pair_features(
    train_rows: list[dict[str, str]], a: str, b: str, feature_names: list[str], top_k: int
) -> list[dict[str, object]]:
    a_rows = [row for row in train_rows if canonical(row["label"]) == a]
    b_rows = [row for row in train_rows if canonical(row["label"]) == b]
    scored: list[dict[str, object]] = []
    for feat in feature_names:
        av = np.asarray([float(row[feat]) for row in a_rows], dtype=np.float32)
        bv = np.asarray([float(row[feat]) for row in b_rows], dtype=np.float32)
        if len(av) == 0 or len(bv) == 0:
            continue
        mean_a = float(av.mean())
        mean_b = float(bv.mean())
        scored.append(
            {
                "feature": feat,
                "cohen_d_abs": cohen_abs(av, bv),
                "auc_abs": auc_abs(av, bv),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "larger_class": b if mean_b > mean_a else a,
                "delta_b_minus_a": mean_b - mean_a,
            }
        )
    return sorted(scored, key=lambda row: (float(row["cohen_d_abs"]), float(row["auc_abs"])), reverse=True)[:top_k]


def augment_pair_values(
    x: np.ndarray,
    y: np.ndarray,
    *,
    copies: int,
    jitter_scale: float,
    directional_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    x0 = x[y == 0]
    x1 = x[y == 1]
    std0 = x0.std(axis=0) + 1e-4
    std1 = x1.std(axis=0) + 1e-4
    direction = x1.mean(axis=0) - x0.mean(axis=0)
    chunks = [x]
    labels = [y]
    for _ in range(copies):
        aug = x.copy()
        mask0 = y == 0
        mask1 = y == 1
        aug[mask0] += rng.normal(0.0, jitter_scale, size=(int(mask0.sum()), x.shape[1])).astype(np.float32) * std0
        aug[mask1] += rng.normal(0.0, jitter_scale, size=(int(mask1.sum()), x.shape[1])).astype(np.float32) * std1
        aug[mask0] -= directional_scale * direction.reshape(1, -1)
        aug[mask1] += directional_scale * direction.reshape(1, -1)
        chunks.append(aug)
        labels.append(y.copy())
    return np.vstack(chunks), np.concatenate(labels)


def make_lr(c: float = 0.8) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1200, class_weight="balanced", C=c, solver="lbfgs"),
    )


def pair_eval(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    a: str,
    b: str,
    feature_names: list[str],
    top_features: list[str],
    seed: int,
) -> dict[str, object] | None:
    pair_train = [row for row in train_rows if canonical(row["label"]) in {a, b}]
    pair_test = [row for row in test_rows if canonical(row["label"]) in {a, b}]
    if len(pair_train) < 12 or len(pair_test) < 12:
        return None
    y_train = np.asarray([0 if canonical(row["label"]) == a else 1 for row in pair_train], dtype=np.int64)
    y_test = np.asarray([0 if canonical(row["label"]) == a else 1 for row in pair_test], dtype=np.int64)

    def run(features: list[str], use_aug: bool) -> tuple[float, float]:
        x_train = np.asarray([[float(row[feat]) for feat in features] for row in pair_train], dtype=np.float32)
        x_test = np.asarray([[float(row[feat]) for feat in features] for row in pair_test], dtype=np.float32)
        if use_aug:
            x_fit, y_fit = augment_pair_values(
                x_train,
                y_train,
                copies=3,
                jitter_scale=0.10,
                directional_scale=0.025,
                seed=seed,
            )
        else:
            x_fit, y_fit = x_train, y_train
        model = make_lr()
        model.fit(x_fit, y_fit)
        pred = model.predict(x_test)
        return float(accuracy_score(y_test, pred)), float(f1_score(y_test, pred, average="macro"))

    top_acc, top_f1 = run(top_features, False)
    aug_acc, aug_f1 = run(top_features, True)
    all_acc, all_f1 = run(feature_names, False)
    return {
        "pair": f"{a} | {b}",
        "test_samples": len(pair_test),
        "topk_lr_acc": top_acc,
        "topk_lr_macro_f1": top_f1,
        "topk_value_aug_acc": aug_acc,
        "topk_value_aug_macro_f1": aug_f1,
        "all_value_lr_acc": all_acc,
        "all_value_lr_macro_f1": all_f1,
        "aug_minus_topk_f1_pp": (aug_f1 - top_f1) * 100.0,
    }


def all_pairs(classes: Iterable[str]) -> list[tuple[str, str]]:
    items = sorted(canonical(cls) for cls in classes)
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def train_pairwise_models(
    train_rows: list[dict[str, str]],
    classes: list[str],
    feature_names: list[str],
    pair_top_features: dict[tuple[str, str], list[str]],
    *,
    use_aug: bool,
    seed: int,
) -> dict[tuple[str, str], object]:
    models: dict[tuple[str, str], object] = {}
    for a, b in all_pairs(classes):
        features = pair_top_features[(a, b)]
        rows = [row for row in train_rows if canonical(row["label"]) in {a, b}]
        x = np.asarray([[float(row[feat]) for feat in features] for row in rows], dtype=np.float32)
        y = np.asarray([0 if canonical(row["label"]) == a else 1 for row in rows], dtype=np.int64)
        if use_aug:
            x_fit, y_fit = augment_pair_values(
                x,
                y,
                copies=3,
                jitter_scale=0.10,
                directional_scale=0.025,
                seed=seed + len(models),
            )
        else:
            x_fit, y_fit = x, y
        model = make_lr()
        model.fit(x_fit, y_fit)
        models[(a, b)] = model
    return models


def tournament_predict(
    models: dict[tuple[str, str], object],
    pair_top_features: dict[tuple[str, str], list[str]],
    test_rows: list[dict[str, str]],
    classes: list[str],
) -> list[str]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    predictions: list[str] = []
    for row in test_rows:
        scores = np.zeros(len(classes), dtype=np.float64)
        for (a, b), model in models.items():
            features = pair_top_features[(a, b)]
            x = np.asarray([[float(row[feat]) for feat in features]], dtype=np.float32)
            prob = model.predict_proba(x)[0]
            pa = float(prob[0])
            pb = float(prob[1])
            margin = abs(pa - pb)
            scores[class_to_idx[a]] += margin * (pa - 0.5)
            scores[class_to_idx[b]] += margin * (pb - 0.5)
        predictions.append(classes[int(scores.argmax())])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    train_rows = read_csv(args.source_dir / "train_target_feature_values.csv")
    test_rows_raw = read_csv(args.source_dir / "test_target_or_pred_feature_values.csv")
    target_classes = [canonical(name) for name in json.loads((args.source_dir / "target_classes.json").read_text(encoding="utf-8"))]
    target_set = set(target_classes)
    test_rows = [row for row in test_rows_raw if canonical(row["label"]) in target_set]
    feature_names = json.loads((args.source_dir / "feature_names.json").read_text(encoding="utf-8"))

    pair_seed_rows = []
    seed_path = args.source_dir / "pairwise_value_augmented_classification.csv"
    if seed_path.exists():
        pair_seed_rows = read_csv(seed_path)
    important_pairs: list[tuple[str, str]] = []
    for row in pair_seed_rows:
        parts = [canonical(part) for part in str(row["pair"]).split("|")]
        if len(parts) == 2 and parts[0] in target_set and parts[1] in target_set:
            important_pairs.append((parts[0], parts[1]))
    if not important_pairs:
        important_pairs = all_pairs(target_classes)[:20]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_feature_rows: list[dict[str, object]] = []
    pair_results: list[dict[str, object]] = []
    pair_top_features: dict[tuple[str, str], list[str]] = {}

    for a, b in all_pairs(target_classes):
        top_rows = top_pair_features(train_rows, a, b, feature_names, args.top_k)
        pair_top_features[(a, b)] = [str(row["feature"]) for row in top_rows]
        if (a, b) in important_pairs or (b, a) in important_pairs:
            for rank, item in enumerate(top_rows, start=1):
                row = {"pair": f"{a} | {b}", "rank": rank}
                row.update(item)
                pair_feature_rows.append(row)
            result = pair_eval(train_rows, test_rows, a, b, feature_names, pair_top_features[(a, b)], args.seed)
            if result is not None:
                pair_results.append(result)

    write_rows(args.out_dir / "important_pair_top_features.csv", pair_feature_rows)
    write_rows(args.out_dir / "important_pair_value_aug_results.csv", pair_results)

    classes = sorted(target_classes)
    y_true = [canonical(row["label"]) for row in test_rows]
    raw_models = train_pairwise_models(train_rows, classes, feature_names, pair_top_features, use_aug=False, seed=args.seed)
    aug_models = train_pairwise_models(train_rows, classes, feature_names, pair_top_features, use_aug=True, seed=args.seed)
    raw_pred = tournament_predict(raw_models, pair_top_features, test_rows, classes)
    aug_pred = tournament_predict(aug_models, pair_top_features, test_rows, classes)

    x_train, y_train, class_order = rows_to_matrix(train_rows, feature_names, classes)
    x_test, y_test, _ = rows_to_matrix(test_rows, feature_names, classes)
    global_lr = make_lr(c=0.8)
    global_lr.fit(x_train, y_train)
    global_pred_idx = global_lr.predict(x_test)
    global_pred = [class_order[int(idx)] for idx in global_pred_idx]

    class_counts = Counter(y_true)
    per_class_rows = []
    for cls in classes:
        idx = [i for i, label in enumerate(y_true) if label == cls]
        if not idx:
            continue
        yt = [y_true[i] for i in idx]
        row = {
            "class": cls,
            "support": len(idx),
            "global_lr_recall": sum(global_pred[i] == yt[j] for j, i in enumerate(idx)) / len(idx),
            "pair_tournament_recall": sum(raw_pred[i] == yt[j] for j, i in enumerate(idx)) / len(idx),
            "pair_aug_tournament_recall": sum(aug_pred[i] == yt[j] for j, i in enumerate(idx)) / len(idx),
        }
        per_class_rows.append(row)
    write_rows(args.out_dir / "tournament_per_class_recall.csv", per_class_rows)

    summary = {
        "target_classes": classes,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "test_class_counts": dict(sorted(class_counts.items())),
        "global_value_lr": {
            "accuracy": float(accuracy_score(y_true, global_pred)),
            "macro_f1": float(f1_score(y_true, global_pred, labels=classes, average="macro", zero_division=0)),
        },
        "pairwise_topk_tournament": {
            "accuracy": float(accuracy_score(y_true, raw_pred)),
            "macro_f1": float(f1_score(y_true, raw_pred, labels=classes, average="macro", zero_division=0)),
        },
        "pairwise_topk_value_aug_tournament": {
            "accuracy": float(accuracy_score(y_true, aug_pred)),
            "macro_f1": float(f1_score(y_true, aug_pred, labels=classes, average="macro", zero_division=0)),
        },
        "important_pairs_mean_aug_gain_pp": float(np.mean([float(row["aug_minus_topk_f1_pp"]) for row in pair_results]))
        if pair_results
        else 0.0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Pair-value tournament audit",
        "",
        f"Target classes: {len(classes)}",
        f"Train rows: {len(train_rows)}",
        f"Test rows: {len(test_rows)}",
        "",
        "## Value-only multiclass checks",
        "",
        f"- Global value LR: Acc={summary['global_value_lr']['accuracy']*100:.2f}%, "
        f"Macro-F1={summary['global_value_lr']['macro_f1']*100:.2f}%",
        f"- Pairwise Top-{args.top_k} tournament: Acc={summary['pairwise_topk_tournament']['accuracy']*100:.2f}%, "
        f"Macro-F1={summary['pairwise_topk_tournament']['macro_f1']*100:.2f}%",
        f"- Pairwise Top-{args.top_k} value-aug tournament: Acc={summary['pairwise_topk_value_aug_tournament']['accuracy']*100:.2f}%, "
        f"Macro-F1={summary['pairwise_topk_value_aug_tournament']['macro_f1']*100:.2f}%",
        "",
        "## Important pair checks",
        "",
    ]
    for row in sorted(pair_results, key=lambda item: float(item["topk_value_aug_macro_f1"]), reverse=True):
        lines.append(
            f"- {row['pair']}: TopK F1={float(row['topk_lr_macro_f1'])*100:.2f}%, "
            f"ValueAug F1={float(row['topk_value_aug_macro_f1'])*100:.2f}%, "
            f"gain={float(row['aug_minus_topk_f1_pp']):+.2f} pp"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
