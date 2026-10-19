"""Leakage-safe evaluation protocol shared by every experiment.

Protocol
--------
1. Folds are created once on raw interactions (`data.assign_folds`); models only
   ever see fold 0.
2. For each user the candidate list is ``test items + N unobserved items``. The
   unobserved items are drawn with a fixed seed from items the user never rated in
   any fold, so a sampled negative can never be a hidden positive.
3. Relevance is binary: a test item counts as relevant when the observed rating is
   at least the configured threshold. Test items below the threshold stay in the
   candidate list (they were genuinely consumed but disliked) and count as
   non-relevant, which is stricter than dropping them.
4. Ranking metrics use the same score matrix that produces the rating predictions,
   so no metric benefits from a different scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config
from .data import TEST, TRAIN, Dataset
from .metrics import (
    aggregate_metrics,
    catalog_coverage,
    gini,
    intra_list_diversity,
    novelty,
    ranking_metrics,
)
from .recommenders.base import Recommender
from .utils import seeded_rng


@dataclass
class RankingProtocol:
    users: np.ndarray
    candidates: list[np.ndarray]
    relevance: list[np.ndarray]
    n_relevant: np.ndarray
    test_items: dict[int, np.ndarray]
    test_ratings: dict[int, np.ndarray]
    n_negatives: int
    relevance_threshold: float
    seed: int

    def summary(self) -> dict[str, Any]:
        return {
            "users_evaluated": int(self.users.size),
            "n_negatives": int(self.n_negatives),
            "relevance_threshold": float(self.relevance_threshold),
            "mean_candidates": float(np.mean([c.size for c in self.candidates])),
            "mean_relevant": float(np.mean(self.n_relevant)),
            "users_without_relevant": int(np.sum(self.n_relevant == 0)),
            "seed": int(self.seed),
        }


@dataclass
class EvaluationResult:
    model: str
    aggregate: dict[str, float]
    per_user: list[dict[str, Any]] = field(default_factory=list)
    per_pair: list[dict[str, Any]] = field(default_factory=list)
    top_k: dict[int, np.ndarray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def user_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_user)

    def pair_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_pair)


def build_protocol(
    dataset: Dataset,
    n_negatives: int = 100,
    relevance_threshold: float = 4.0,
    seed: int = 0,
    fold: int = TEST,
) -> RankingProtocol:
    """Build candidate sets for a held-out fold.

    ``fold`` defaults to TEST. Passing VALID produces the tuning protocol: model
    selection therefore never looks at the test fold.
    """
    rng = seeded_rng(seed, stream=301)
    observed = dataset.observed_pairs().tocsr()
    test_mask = dataset.fold == fold
    test_users = dataset.rating_user[test_mask]
    test_items = dataset.rating_item[test_mask]
    test_ratings = dataset.ratings[test_mask]

    users = np.unique(test_users)
    items_by_user: dict[int, np.ndarray] = {}
    ratings_by_user: dict[int, np.ndarray] = {}
    order = np.argsort(test_users, kind="stable")
    boundaries = np.flatnonzero(np.diff(test_users[order])) + 1
    for group in np.split(order, boundaries):
        user = int(test_users[group[0]])
        items_by_user[user] = test_items[group]
        ratings_by_user[user] = test_ratings[group]

    candidates: list[np.ndarray] = []
    relevance: list[np.ndarray] = []
    n_relevant = np.zeros(users.size, dtype=np.int64)
    n_items = dataset.n_items

    for position, user in enumerate(users):
        positives = items_by_user[int(user)]
        seen = observed.indices[observed.indptr[user] : observed.indptr[user + 1]]
        negatives = _sample_unobserved(rng, seen, n_items, n_negatives)
        pool = np.concatenate([positives, negatives]).astype(np.int64)
        labels = np.concatenate(
            [
                ratings_by_user[int(user)] >= relevance_threshold,
                np.zeros(negatives.size, dtype=bool),
            ]
        )
        # Shuffle so that a constant-scoring model cannot accidentally rank the
        # held-out positives first through the order of construction.
        shuffle = rng.permutation(pool.size)
        candidates.append(pool[shuffle])
        relevance.append(labels[shuffle])
        n_relevant[position] = int(labels.sum())

    return RankingProtocol(
        users=users,
        candidates=candidates,
        relevance=relevance,
        n_relevant=n_relevant,
        test_items=items_by_user,
        test_ratings=ratings_by_user,
        n_negatives=int(n_negatives),
        relevance_threshold=float(relevance_threshold),
        seed=int(seed),
    )


def _sample_unobserved(
    rng: np.random.Generator, seen: np.ndarray, n_items: int, n_negatives: int
) -> np.ndarray:
    """Uniform sample of items the user never interacted with (rejection sampling)."""
    target = min(n_negatives, n_items - seen.size)
    seen_set = np.zeros(n_items, dtype=bool)
    seen_set[seen] = True
    collected = np.empty(0, dtype=np.int64)
    for _ in range(10):
        if collected.size >= target:
            break
        draw = rng.integers(0, n_items, size=max(2 * (target - collected.size), 16))
        draw = draw[~seen_set[draw]]
        collected = np.unique(np.concatenate([collected, draw]))
    if collected.size < target:
        remaining = np.setdiff1d(np.flatnonzero(~seen_set), collected)
        collected = np.concatenate([collected, remaining[: target - collected.size]])
    return collected[:target]


def evaluate(
    model: Recommender,
    dataset: Dataset,
    protocol: RankingProtocol,
    cfg: Config,
    item_features: sparse.csr_matrix | None = None,
    collect_top_k: bool = True,
    chunk_size: int = 512,
) -> EvaluationResult:
    """Stream users in chunks and compute error + ranking metrics in one pass."""
    settings = cfg.section("recommendation.evaluation")
    k = int(settings["top_k"])
    users = protocol.users
    per_user: list[dict[str, Any]] = []
    per_pair: list[dict[str, Any]] = []
    top_k: dict[int, np.ndarray] = {}

    train_counts = np.bincount(
        dataset.rating_user[dataset.fold == TRAIN], minlength=dataset.n_users
    )
    all_counts = np.bincount(dataset.rating_item, minlength=dataset.n_items)

    for start in range(0, users.size, chunk_size):
        chunk = users[start : start + chunk_size]
        scores = model.predict_matrix(chunk)
        for local, user in enumerate(chunk):
            user = int(user)
            pool = protocol.candidates[start + local]
            labels = protocol.relevance[start + local]
            user_scores = scores[local, pool]
            order = np.argsort(-user_scores, kind="stable")
            ranked_labels = labels[order]
            metrics = ranking_metrics(ranked_labels, int(protocol.n_relevant[start + local]), k)
            metrics["user"] = user
            metrics["train_interactions"] = int(train_counts[user])
            metrics["n_candidates"] = int(pool.size)
            per_user.append(metrics)
            if collect_top_k:
                top_k[user] = pool[order[:k]]

            items = protocol.test_items[user]
            truth = protocol.test_ratings[user]
            predictions = np.clip(scores[local, items], 1.0, 5.0)
            for item, rating, prediction in zip(items, truth, predictions, strict=True):
                per_pair.append(
                    {
                        "user": user,
                        "item": int(item),
                        "rating": float(rating),
                        "prediction": float(prediction),
                        "train_interactions": int(train_counts[user]),
                        "item_interactions": int(all_counts[item]),
                    }
                )

    pair_frame = pd.DataFrame(per_pair)
    aggregate = aggregate_metrics(per_user)
    aggregate["model"] = model.name  # type: ignore[assignment]
    aggregate["mae"] = float(np.mean(np.abs(pair_frame["prediction"] - pair_frame["rating"])))
    aggregate["rmse"] = float(
        np.sqrt(np.mean((pair_frame["prediction"] - pair_frame["rating"]) ** 2))
    )
    aggregate["n_users"] = int(users.size)
    aggregate["n_pairs"] = int(pair_frame.shape[0])
    if collect_top_k:
        aggregate[f"coverage@{k}"] = catalog_coverage(top_k, dataset.n_items)
        aggregate[f"gini@{k}"] = gini(top_k, dataset.n_items)
        if item_features is not None:
            aggregate[f"diversity@{k}"] = intra_list_diversity(top_k, item_features, seed=0)
        popularity = np.bincount(
            dataset.rating_item[dataset.fold == TRAIN], minlength=dataset.n_items
        ).astype(np.float64)
        aggregate[f"novelty@{k}"] = novelty(top_k, popularity)

    return EvaluationResult(
        model=model.name,
        aggregate=aggregate,
        per_user=per_user,
        per_pair=per_pair,
        top_k=top_k,
        extra={"protocol": protocol.summary()},
    )


def adoption_classification(
    model: Recommender,
    protocol: RankingProtocol,
    threshold: float = 3.5,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Binary adoption task on the exact ranking candidate pool.

    Task definition: given a user/item pair that the user has not rated before,
    will the user adopt it (observed rating >= relevance threshold) or not?
    Negatives are the same unobserved sampled items used by the ranking protocol,
    so the task is well defined and the positive rate is known. Because positives
    are rare (~2 per 100 candidates), plain accuracy is dominated by the majority
    class; the all-negative no-skill baseline is therefore reported next to it.
    """
    from .metrics import classification_metrics

    scores_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    users = protocol.users
    for start in range(0, users.size, chunk_size):
        chunk = users[start : start + chunk_size]
        matrix = model.predict_matrix(chunk)
        for local in range(chunk.size):
            pool = protocol.candidates[start + local]
            scores_all.append(np.asarray(matrix[local, pool], dtype=float))
            labels_all.append(protocol.relevance[start + local])
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)
    metrics = classification_metrics(scores, labels, threshold)
    positive_rate = float(labels.mean())
    metrics.update(
        {
            "task": "adopt (rating >= relevance threshold) vs unobserved negative",
            "decision_threshold": float(threshold),
            "positive_rate": positive_rate,
            "no_skill_accuracy": 1.0 - positive_rate,
            "balanced_accuracy": 0.5 * (metrics["tpr"] + (1.0 - metrics["fpr"])),
            "n_examples": float(labels.size),
        }
    )
    return metrics


def history_bucket(counts: np.ndarray, edges: tuple[int, ...] = (20, 30, 50, 100)) -> np.ndarray:
    """Label users by training-history length for cold-start slicing."""
    labels = np.full(counts.shape, f"<{edges[0]}", dtype=object)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        labels[(counts >= low) & (counts < high)] = f"{low}-{high - 1}"
    labels[counts >= edges[-1]] = f">={edges[-1]}"
    return labels


def popularity_bucket(
    counts: np.ndarray, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> np.ndarray:
    """Quartile labels of item popularity (train interactions)."""
    thresholds = np.quantile(counts[counts > 0], quantiles)
    labels = np.full(counts.shape, "Q1 (coldest)", dtype=object)
    labels[counts > thresholds[0]] = "Q2"
    labels[counts > thresholds[1]] = "Q3"
    labels[counts > thresholds[2]] = "Q4 (hottest)"
    return labels


def sliced_report(result: EvaluationResult, dataset: Dataset) -> dict[str, pd.DataFrame]:
    """Per-history and per-popularity breakdowns used in Requirement 2."""
    pairs = result.pair_frame()
    users = result.user_frame()
    train_counts = np.bincount(
        dataset.rating_user[dataset.fold == TRAIN], minlength=dataset.n_users
    )
    user_labels = history_bucket(train_counts)
    users["history_bucket"] = user_labels[users["user"].to_numpy()]
    pairs["history_bucket"] = user_labels[pairs["user"].to_numpy()]
    item_counts = np.bincount(dataset.rating_item, minlength=dataset.n_items)
    pairs["popularity_bucket"] = popularity_bucket(item_counts)[pairs["item"].to_numpy()]

    ndcg_column = next((c for c in users.columns if c.startswith("ndcg@")), None)
    hit_column = next((c for c in users.columns if c.startswith("hit@")), None)

    by_history = users.groupby("history_bucket", observed=True).agg(
        users=("user", "size"),
        ndcg=(ndcg_column, "mean"),
        hit=(hit_column, "mean"),
    )
    pairs["abs_error"] = (pairs["prediction"] - pairs["rating"]).abs()
    pairs["squared_error"] = (pairs["prediction"] - pairs["rating"]) ** 2
    by_popularity = pairs.groupby("popularity_bucket", observed=True).agg(
        pairs=("rating", "size"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
    )
    by_history_pairs = pairs.groupby("history_bucket", observed=True).agg(
        pairs=("rating", "size"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
    )
    return {
        "by_history_ranking": by_history.reset_index(),
        "by_history_error": by_history_pairs.reset_index(),
        "by_popularity_error": by_popularity.reset_index(),
    }
