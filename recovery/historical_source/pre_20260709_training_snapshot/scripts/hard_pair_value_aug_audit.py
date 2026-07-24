from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-8


def canonical(label: str) -> str:
    return str(label).strip().replace("-", "_")


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


def feature_names_from_rows(rows: list[dict[str, str]]) -> list[str]:
    skip = {"image_path", "label", "pred_label", "base_confidence", "confidence"}
    return [key for key in rows[0] if key not in skip]


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


def top_confusion_pairs(prediction_rows: list[dict[str, str]], max_pairs: int) -> list[dict[str, object]]:
    directed = Counter()
    symmetric = Counter()
    for row in prediction_rows:
        true_label = canonical(row["true_label"])
        pred_label = canonical(row["pred_label"])
        if true_label == pred_label:
            continue
        directed[(true_label, pred_label)] += 1
        symmetric[tuple(sorted((true_label, pred_label)))] += 1
    rows = []
    for (a, b), count in symmetric.most_common(max_pairs):
        rows.append(
            {
                "class_a": a,
                "class_b": b,
                "symmetric_errors": int(count),
                "a_to_b_errors": int(directed[(a, b)]),
                "b_to_a_errors": int(directed[(b, a)]),
            }
        )
    return rows


def error_class_rows(per_class_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]], max_classes: int) -> list[dict[str, object]]:
    err = Counter()
    pred_into = Counter()
    for row in prediction_rows:
        true_label = canonical(row["true_label"])
        pred_label = canonical(row["pred_label"])
        if true_label == pred_label:
            continue
        err[true_label] += 1
        pred_into[pred_label] += 1
    rows = []
    for row in per_class_rows:
        cls = canonical(row["class"])
        rows.append(
            {
                "class": cls,
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "support": float(row["support"]),
                "true_errors": int(err[cls]),
                "false_positive_errors": int(pred_into[cls]),
            }
        )
    rows.sort(key=lambda item: (float(item["f1"]), -int(item["true_errors"])))
    return rows[:max_classes]


def rows_by_label(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[canonical(row["label"])].append(row)
    return out


def pair_top_features(
    by_label: dict[str, list[dict[str, str]]],
    a: str,
    b: str,
    feature_names: list[str],
    top_k: int,
) -> list[dict[str, object]]:
    rows_a = by_label.get(a, [])
    rows_b = by_label.get(b, [])
    scored = []
    for feat in feature_names:
        av = np.asarray([float(row[feat]) for row in rows_a], dtype=np.float32)
        bv = np.asarray([float(row[feat]) for row in rows_b], dtype=np.float32)
        if len(av) < 4 or len(bv) < 4:
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
                "delta_b_minus_a": mean_b - mean_a,
                "larger_class": b if mean_b > mean_a else a,
            }
        )
    return sorted(scored, key=lambda item: (float(item["cohen_d_abs"]), float(item["auc_abs"])), reverse=True)[:top_k]


def make_lr(c: float = 0.8) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1600, class_weight="balanced", C=c, solver="lbfgs"),
    )


def augment_values(
    x: np.ndarray,
    y: np.ndarray,
    *,
    copies: int,
    jitter: float,
    directional: float,
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
    q_min = np.quantile(x, 0.005, axis=0)
    q_max = np.quantile(x, 0.995, axis=0)
    chunks = [x]
    labels = [y]
    for _ in range(copies):
        aug = x.copy()
        mask0 = y == 0
        mask1 = y == 1
        aug[mask0] += rng.normal(0.0, jitter, size=(int(mask0.sum()), x.shape[1])).astype(np.float32) * std0
        aug[mask1] += rng.normal(0.0, jitter, size=(int(mask1.sum()), x.shape[1])).astype(np.float32) * std1
        aug[mask0] -= directional * direction.reshape(1, -1)
        aug[mask1] += directional * direction.reshape(1, -1)
        aug = np.clip(aug, q_min, q_max)
        chunks.append(aug.astype(np.float32))
        labels.append(y.copy())
    return np.vstack(chunks), np.concatenate(labels)


def eval_pair(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    a: str,
    b: str,
    top_features: list[str],
    all_features: list[str],
    seed: int,
) -> dict[str, object] | None:
    train_pair = [row for row in train_rows if canonical(row["label"]) in {a, b}]
    test_pair = [row for row in test_rows if canonical(row["label"]) in {a, b}]
    if len(train_pair) < 20 or len(test_pair) < 20:
        return None
    y_train = np.asarray([0 if canonical(row["label"]) == a else 1 for row in train_pair], dtype=np.int64)
    y_test = np.asarray([0 if canonical(row["label"]) == a else 1 for row in test_pair], dtype=np.int64)

    def run(features: list[str], use_aug: bool) -> tuple[float, float, float, float]:
        x_train = np.asarray([[float(row[feat]) for feat in features] for row in train_pair], dtype=np.float32)
        x_test = np.asarray([[float(row[feat]) for feat in features] for row in test_pair], dtype=np.float32)
        if use_aug:
            x_fit, y_fit = augment_values(x_train, y_train, copies=4, jitter=0.12, directional=0.020, seed=seed)
        else:
            x_fit, y_fit = x_train, y_train
        model = make_lr()
        model.fit(x_fit, y_fit)
        pred = model.predict(x_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        return float(accuracy_score(y_test, pred)), float(p), float(r), float(f1)

    top_acc, top_p, top_r, top_f1 = run(top_features, False)
    aug_acc, aug_p, aug_r, aug_f1 = run(top_features, True)
    all_acc, all_p, all_r, all_f1 = run(all_features, False)
    return {
        "pair": f"{a} | {b}",
        "test_samples": len(test_pair),
        "topk_value_acc": top_acc,
        "topk_value_macro_p": top_p,
        "topk_value_macro_r": top_r,
        "topk_value_macro_f1": top_f1,
        "topk_value_aug_acc": aug_acc,
        "topk_value_aug_macro_p": aug_p,
        "topk_value_aug_macro_r": aug_r,
        "topk_value_aug_macro_f1": aug_f1,
        "all_value_acc": all_acc,
        "all_value_macro_p": all_p,
        "all_value_macro_r": all_r,
        "all_value_macro_f1": all_f1,
        "aug_gain_f1_pp": (aug_f1 - top_f1) * 100.0,
    }


def eval_multiclass(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    classes: list[str],
    feature_names: list[str],
    selected_features: list[str],
    seed: int,
) -> dict[str, object]:
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    train = [row for row in train_rows if canonical(row["label"]) in class_to_idx]
    test = [row for row in test_rows if canonical(row["label"]) in class_to_idx]
    y_train = np.asarray([class_to_idx[canonical(row["label"])] for row in train], dtype=np.int64)
    y_test = np.asarray([class_to_idx[canonical(row["label"])] for row in test], dtype=np.int64)

    def run(features: list[str], use_aug: bool) -> dict[str, float]:
        x_train = np.asarray([[float(row[feat]) for feat in features] for row in train], dtype=np.float32)
        x_test = np.asarray([[float(row[feat]) for feat in features] for row in test], dtype=np.float32)
        if use_aug:
            # Multiclass augmentation uses only jitter, because one global
            # direction would collapse different RSCD factor boundaries.
            rng = np.random.default_rng(seed)
            chunks = [x_train]
            labels = [y_train]
            for _ in range(3):
                aug = x_train.copy()
                for cls_idx in sorted(set(y_train.tolist())):
                    mask = y_train == cls_idx
                    std = x_train[mask].std(axis=0) + 1e-4
                    noise = rng.normal(0.0, 0.12, size=(int(mask.sum()), x_train.shape[1])).astype(np.float32)
                    aug[mask] += noise * std
                chunks.append(aug)
                labels.append(y_train.copy())
            x_fit = np.vstack(chunks)
            y_fit = np.concatenate(labels)
        else:
            x_fit, y_fit = x_train, y_train
        model = make_lr()
        model.fit(x_fit, y_fit)
        pred = model.predict(x_test)
        return {
            "acc": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        }

    return {
        "num_train": len(train),
        "num_test": len(test),
        "all_values": run(feature_names, False),
        "selected_values": run(selected_features, False),
        "selected_value_aug": run(selected_features, True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--per-class-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-classes", type=int, default=12)
    parser.add_argument("--max-pairs", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv(args.source_dir / "train_target_feature_values.csv")
    test_rows = read_csv(args.source_dir / "test_target_or_pred_feature_values.csv")
    prediction_rows = read_csv(args.predictions)
    per_class_rows = read_csv(args.per_class_metrics)
    feature_names = json.loads((args.source_dir / "feature_names.json").read_text(encoding="utf-8"))
    if not feature_names:
        feature_names = feature_names_from_rows(train_rows)

    hard_classes = error_class_rows(per_class_rows, prediction_rows, args.max_classes)
    hard_class_set = {str(row["class"]) for row in hard_classes}
    confusion_pairs = top_confusion_pairs(prediction_rows, args.max_pairs)
    confusion_pairs = [
        row
        for row in confusion_pairs
        if str(row["class_a"]) in hard_class_set or str(row["class_b"]) in hard_class_set
    ]

    by_label = rows_by_label(train_rows)
    top_feature_rows: list[dict[str, object]] = []
    pair_eval_rows: list[dict[str, object]] = []
    selected_counter = Counter()
    for pair_idx, pair_row in enumerate(confusion_pairs):
        a = str(pair_row["class_a"])
        b = str(pair_row["class_b"])
        top_rows = pair_top_features(by_label, a, b, feature_names, args.top_k)
        top_features = [str(row["feature"]) for row in top_rows]
        if not top_features:
            continue
        for rank, row in enumerate(top_rows, start=1):
            out = dict(pair_row)
            out["pair"] = f"{a} | {b}"
            out["rank"] = rank
            out.update(row)
            top_feature_rows.append(out)
            selected_counter[str(row["feature"])] += 1
        result = eval_pair(train_rows, test_rows, a, b, top_features, feature_names, args.seed + pair_idx)
        if result:
            combined = dict(pair_row)
            combined.update(result)
            combined["top_features"] = "; ".join(top_features)
            pair_eval_rows.append(combined)

    selected_features = [feat for feat, _ in selected_counter.most_common(36)]
    hard_classes_from_pairs = sorted(hard_class_set | {str(row["class_a"]) for row in confusion_pairs} | {str(row["class_b"]) for row in confusion_pairs})
    multiclass = eval_multiclass(train_rows, test_rows, hard_classes_from_pairs, feature_names, selected_features, args.seed)

    write_rows(args.out_dir / "hard_error_classes.csv", hard_classes)
    write_rows(args.out_dir / "top_confusion_pairs.csv", confusion_pairs)
    write_rows(args.out_dir / "pair_top_feature_differences.csv", top_feature_rows)
    write_rows(args.out_dir / "pair_value_aug_classification.csv", pair_eval_rows)
    (args.out_dir / "selected_features.json").write_text(json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "hard_classes": hard_classes_from_pairs,
        "num_pairs": len(pair_eval_rows),
        "selected_features": selected_features,
        "pair_value_aug_mean_gain_pp": float(np.mean([float(row["aug_gain_f1_pp"]) for row in pair_eval_rows])) if pair_eval_rows else 0.0,
        "pair_value_aug_positive_pairs": int(sum(float(row["aug_gain_f1_pp"]) > 0.0 for row in pair_eval_rows)),
        "pair_value_aug_negative_pairs": int(sum(float(row["aug_gain_f1_pp"]) < 0.0 for row in pair_eval_rows)),
        "multiclass": multiclass,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Hard-pair value comparison and augmentation audit",
        "",
        "## Scope",
        "",
        f"- Hard classes/pair classes: {len(hard_classes_from_pairs)}",
        f"- High-confusion pairs evaluated: {len(pair_eval_rows)}",
        f"- Selected pair-difference features: {len(selected_features)}",
        "",
        "## Multiclass value-only check",
        "",
        f"- All values: Acc={multiclass['all_values']['acc']*100:.2f}%, Macro-F1={multiclass['all_values']['macro_f1']*100:.2f}%",
        f"- Selected pair-difference values: Acc={multiclass['selected_values']['acc']*100:.2f}%, Macro-F1={multiclass['selected_values']['macro_f1']*100:.2f}%",
        f"- Selected values + value jitter: Acc={multiclass['selected_value_aug']['acc']*100:.2f}%, Macro-F1={multiclass['selected_value_aug']['macro_f1']*100:.2f}%",
        "",
        "## Pairwise value augmentation check",
        "",
    ]
    for row in sorted(pair_eval_rows, key=lambda item: float(item["aug_gain_f1_pp"]), reverse=True):
        lines.append(
            f"- {row['pair']}: errors={row['symmetric_errors']}, "
            f"TopK F1={float(row['topk_value_macro_f1'])*100:.2f}%, "
            f"Aug F1={float(row['topk_value_aug_macro_f1'])*100:.2f}%, "
            f"gain={float(row['aug_gain_f1_pp']):+.2f} pp"
        )
    lines.extend(["", "## Largest value gaps", ""])
    for row in sorted(top_feature_rows, key=lambda item: float(item["cohen_d_abs"]), reverse=True)[:40]:
        lines.append(
            f"- {row['pair']} / {row['feature']}: d={float(row['cohen_d_abs']):.2f}, "
            f"AUC={float(row['auc_abs']):.3f}, mean_a={float(row['mean_a']):.4f}, "
            f"mean_b={float(row['mean_b']):.4f}, larger={row['larger_class']}"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "- Value features are useful for specific RSCD hard boundaries, but the multiclass value-only classifier is far below the image model."
    )
    lines.append(
        "- Keep value augmentation as pair-specific boundary evidence, not as a global replacement for ConvNeXt visual features."
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
