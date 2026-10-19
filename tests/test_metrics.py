"""Metric definitions must match the formulas stated in the report."""

from __future__ import annotations

import numpy as np
import pytest

from sntp.metrics import (
    aggregate_metrics,
    average_precision_at_k,
    catalog_coverage,
    classification_metrics,
    dcg,
    gini,
    hit_at_k,
    intra_list_diversity,
    mae,
    ndcg_at_k,
    novelty,
    precision_at_k,
    ranking_metrics,
    recall_at_k,
    rmse,
)


def test_error_metrics_are_symmetric_and_zero_for_perfect_predictions():
    truth = np.array([1.0, 2.5, 4.0, 5.0])
    assert rmse(truth, truth) == 0.0
    assert mae(truth, truth) == 0.0
    assert rmse(truth + 1.0, truth) == pytest.approx(1.0)
    assert mae(truth - 2.0, truth) == pytest.approx(2.0)


def test_empty_input_returns_nan_instead_of_raising():
    assert np.isnan(rmse(np.array([]), np.array([])))
    assert np.isnan(mae(np.array([]), np.array([])))


def test_dcg_matches_hand_computed_value():
    gains = np.array([1.0, 1.0, 0.0])
    expected = 1.0 + 1.0 / np.log2(3.0)
    assert dcg(gains) == pytest.approx(expected)
    assert dcg(np.array([])) == 0.0


def test_precision_recall_and_hit():
    ranked = np.array([1, 0, 1, 0, 0], dtype=bool)
    assert precision_at_k(ranked, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, 2, 3) == pytest.approx(1.0)
    assert hit_at_k(ranked, 1) == 1.0
    assert hit_at_k(ranked, 0) == 0.0
    assert precision_at_k(ranked, 0) == 0.0


def test_ndcg_is_one_for_the_ideal_ranking():
    ideal = np.ones(4, dtype=bool)
    assert ndcg_at_k(ideal, 4, 10) == pytest.approx(1.0)
    assert ndcg_at_k(ideal, 4, 2) == pytest.approx(1.0)


def test_ndcg_discounts_late_hits():
    early = np.array([1, 1, 0, 0], dtype=bool)
    late = np.array([0, 0, 1, 1], dtype=bool)
    assert ndcg_at_k(early, 2, 4) > ndcg_at_k(late, 2, 4)


def test_ndcg_without_relevant_items_is_nan():
    assert np.isnan(ndcg_at_k(np.zeros(3, dtype=bool), 0, 10))


def test_average_precision_matches_definition():
    ranked = np.array([1, 0, 1], dtype=bool)
    expected = (1.0 / 1 + 2.0 / 3) / 2
    assert average_precision_at_k(ranked, 2, 3) == pytest.approx(expected)
    assert average_precision_at_k(np.zeros(3, dtype=bool), 2, 3) == 0.0


def test_ranking_metrics_keys_follow_the_cutoff():
    metrics = ranking_metrics(np.array([1, 0], dtype=bool), 1, 10)
    assert set(metrics) == {"precision@10", "recall@10", "hit@10", "ndcg@10", "map@10"}


def test_catalog_coverage_counts_distinct_items():
    top_k = {0: np.array([0, 1]), 1: np.array([1, 2])}
    assert catalog_coverage(top_k, n_items=4) == pytest.approx(3 / 4)
    assert catalog_coverage({}, n_items=4) == 0.0


def test_gini_is_zero_for_a_uniform_exposure_and_grows_with_concentration():
    uniform = {user: np.array([0, 1, 2, 3]) for user in range(5)}
    concentrated = {0: np.array([0, 0, 0, 0])}
    assert gini(uniform, 4) == pytest.approx(0.0, abs=1e-9)
    assert gini(concentrated, 4) > 0.5


def test_novelty_prefers_cold_items():
    popularity = np.array([1000.0, 10.0, 1.0])
    common = {0: np.array([0])}
    rare = {0: np.array([2])}
    assert novelty(rare, popularity) > novelty(common, popularity)
    assert novelty({}, popularity) == 0.0


def test_intra_list_diversity_is_bounded_and_rewards_orthogonal_items():
    from scipy import sparse

    features = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    diverse = {0: np.array([0, 1, 2])}
    identical = {0: np.array([0, 0, 0])}
    assert intra_list_diversity(diverse, features) == pytest.approx(1.0)
    # repeating the same item gives a pairwise cosine of 1, hence no diversity
    assert intra_list_diversity(identical, features) == pytest.approx(0.0)
    assert intra_list_diversity({}, features) == 0.0


def test_classification_metrics_on_a_perfect_classifier():
    scores = np.array([4.0, 4.5, 2.0, 1.0])
    labels = np.array([1, 1, 0, 0])
    metrics = classification_metrics(scores, labels, threshold=3.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0


def test_classification_metrics_on_an_all_negative_decision():
    scores = np.zeros(4)
    labels = np.array([1, 0, 0, 0])
    metrics = classification_metrics(scores, labels, threshold=3.5)
    assert metrics["recall"] == 0.0
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_aggregate_metrics_ignores_nan_columns():
    rows = [
        {"a": 1.0, "b": float("nan")},
        {"a": 3.0, "b": 2.0},
    ]
    out = aggregate_metrics(rows)
    assert out["a"] == pytest.approx(2.0)
    assert out["b"] == pytest.approx(2.0)
    assert aggregate_metrics([]) == {}
