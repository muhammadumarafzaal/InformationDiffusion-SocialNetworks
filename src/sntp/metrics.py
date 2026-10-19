"""Evaluation metrics implemented without recommender-library shortcuts.

Definitions used throughout the report
--------------------------------------
* Relevance: a held-out test item is relevant when its observed rating is at least
  ``relevance_threshold`` (default 4). Unobserved sampled items are non-relevant.
* Ranking cut-off: K = 10, as configured in ``recommendation.evaluation.top_k``.
* Candidate set: all test items of the user plus 100 sampled unobserved items.
* Precision/recall use binary relevance; NDCG uses the same binary gains.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    diff = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan")


def mae(prediction: np.ndarray, truth: np.ndarray) -> float:
    diff = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.mean(np.abs(diff))) if diff.size else float("nan")


def precision_at_k(ranked_relevance: np.ndarray, k: int) -> float:
    if k == 0:
        return 0.0
    return float(ranked_relevance[:k].sum() / k)


def recall_at_k(ranked_relevance: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return float("nan")
    return float(ranked_relevance[:k].sum() / n_relevant)


def hit_at_k(ranked_relevance: np.ndarray, k: int) -> float:
    return float(ranked_relevance[:k].any()) if ranked_relevance.size else 0.0


def dcg(gains: np.ndarray) -> float:
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_relevance: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return float("nan")
    top = ranked_relevance[:k]
    ideal = np.ones(min(n_relevant, k))
    denominator = dcg(ideal)
    return float(dcg(top) / denominator) if denominator > 0 else 0.0


def average_precision_at_k(ranked_relevance: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return float("nan")
    top = ranked_relevance[:k]
    if top.sum() == 0:
        return 0.0
    positions = np.flatnonzero(top) + 1
    return float(np.sum(np.arange(1, positions.size + 1) / positions) / min(n_relevant, k))


def ranking_metrics(ranked_relevance: np.ndarray, n_relevant: int, k: int) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(ranked_relevance, k),
        f"recall@{k}": recall_at_k(ranked_relevance, n_relevant, k),
        f"hit@{k}": hit_at_k(ranked_relevance, k),
        f"ndcg@{k}": ndcg_at_k(ranked_relevance, n_relevant, k),
        f"map@{k}": average_precision_at_k(ranked_relevance, n_relevant, k),
    }


# ------------------------------------------------------------------- catalogue
def catalog_coverage(top_k: dict[int, np.ndarray], n_items: int) -> float:
    if not top_k:
        return 0.0
    recommended = np.unique(np.concatenate([np.asarray(v) for v in top_k.values()]))
    return float(recommended.size / n_items)


def novelty(top_k: dict[int, np.ndarray], popularity: np.ndarray) -> float:
    """Mean self-information ``-log2(p(i))`` of recommended items."""
    if not top_k:
        return 0.0
    total = popularity.sum()
    if total <= 0:
        return 0.0
    probabilities = popularity / total
    scores = []
    for items in top_k.values():
        items = np.asarray(items)
        p = probabilities[items]
        p = np.clip(p, 1e-12, None)
        scores.append(np.mean(-np.log2(p)))
    return float(np.mean(scores))


def intra_list_diversity(
    top_k: dict[int, np.ndarray],
    item_features: sparse.csr_matrix,
    n_sample: int = 2000,
    seed: int = 0,
) -> float:
    """``1 - mean pairwise cosine`` between recommended items, averaged over users."""
    if not top_k:
        return 0.0
    rng = np.random.default_rng(seed)
    users = np.fromiter(top_k.keys(), dtype=np.int64)
    if users.size > n_sample:
        users = rng.choice(users, size=n_sample, replace=False)
    features = item_features.tocsr()
    norms = np.sqrt(np.asarray(features.multiply(features).sum(axis=1)).ravel())
    scores = []
    for user in users:
        items = np.asarray(top_k[int(user)])
        if items.size < 2:
            continue
        block = features[items]
        similarity = (block @ block.T).toarray().astype(np.float64)
        local_norms = norms[items]
        denominator = np.outer(local_norms, local_norms)
        similarity = np.divide(
            similarity, denominator, out=np.zeros_like(similarity), where=denominator > 0
        )
        upper = similarity[np.triu_indices(items.size, k=1)]
        scores.append(1.0 - float(np.mean(upper)) if upper.size else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def gini(top_k: dict[int, np.ndarray], n_items: int) -> float:
    """Gini coefficient of the item exposure distribution (0 = uniform)."""
    if not top_k:
        return 0.0
    counts = np.bincount(np.concatenate([np.asarray(v) for v in top_k.values()]), minlength=n_items)
    counts = np.sort(counts.astype(np.float64))
    total = counts.sum()
    if total == 0:
        return 0.0
    n = counts.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * counts).sum()) / (n * total) - (n + 1.0) / n)


def classification_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    """Adoption-style binary classification at a fixed score threshold.

    Positive example: an item the user actually rated (adopted) in the test fold.
    Negative example: an item sampled from the user's unobserved set. Accuracy is
    only reported for this explicitly defined task.
    """
    predicted = scores >= threshold
    labels = labels.astype(bool)
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def aggregate_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Column-wise mean of per-user metric dictionaries (NaNs ignored)."""
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    result = {}
    for key in keys:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[key] = float(finite.mean()) if finite.size else float("nan")
    return result
