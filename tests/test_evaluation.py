"""Leakage controls and the evaluation harness."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from sntp.data import TEST
from sntp.evaluation import (
    adoption_classification,
    build_protocol,
    evaluate,
    history_bucket,
    popularity_bucket,
    sliced_report,
)
from sntp.recommenders import BiasBaseline, GlobalMeanRecommender


@pytest.fixture(scope="module")
def protocol(dataset):
    return build_protocol(dataset, n_negatives=10, relevance_threshold=4.0, seed=5)


def test_protocol_contains_every_test_item(dataset, protocol):
    for user, items in protocol.test_items.items():
        pool = protocol.candidates[int(np.searchsorted(protocol.users, user))]
        assert set(items.tolist()).issubset(set(pool.tolist()))


def test_sampled_negatives_were_never_observed(dataset, protocol):
    observed = defaultdict(set)
    for user, item in zip(dataset.rating_user.tolist(), dataset.rating_item.tolist(), strict=True):
        observed[user].add(item)
    for position, user in enumerate(protocol.users):
        held_out = set(protocol.test_items[int(user)].tolist())
        negatives = [
            item for item in protocol.candidates[position].tolist() if item not in held_out
        ]
        assert len(negatives) == protocol.n_negatives
        assert not (set(negatives) & observed[int(user)])


def test_relevance_follows_the_configured_threshold(dataset, protocol):
    for position, user in enumerate(protocol.users):
        held_out = protocol.test_items[int(user)]
        ratings = protocol.test_ratings[int(user)]
        labels = dict(
            zip(protocol.candidates[position].tolist(), protocol.relevance[position].tolist())
        )
        for item, rating in zip(held_out, ratings, strict=True):
            assert labels[int(item)] == bool(rating >= 4.0)
        assert protocol.n_relevant[position] == int((ratings >= 4.0).sum())


def test_candidate_lists_are_shuffled(protocol):
    # a constant model must not be able to rank positives first by construction
    shuffled = sum(1 for pool in protocol.candidates if not np.all(np.diff(pool) > 0))
    assert shuffled >= len(protocol.candidates) // 2


def test_evaluate_reports_error_and_ranking_metrics(dataset, cfg, protocol):
    model = GlobalMeanRecommender().fit(dataset)
    result = evaluate(model, dataset, protocol, cfg, collect_top_k=True)
    aggregate = result.aggregate
    assert aggregate["n_users"] == protocol.users.size
    assert aggregate["mae"] > 0
    assert aggregate["rmse"] >= aggregate["mae"]
    assert 0.0 <= aggregate["ndcg@10"] <= 1.0
    assert 0.0 <= aggregate["coverage@10"] <= 1.0
    assert 0.0 <= aggregate["gini@10"] <= 1.0
    assert aggregate["novelty@10"] >= 0.0
    assert result.pair_frame().shape[0] == aggregate["n_pairs"]


def test_a_better_model_has_a_lower_test_error(dataset, cfg, protocol):
    global_mean = evaluate(GlobalMeanRecommender().fit(dataset), dataset, protocol, cfg)
    bias = evaluate(BiasBaseline(reg=5.0).fit(dataset), dataset, protocol, cfg)
    assert bias.aggregate["rmse"] < global_mean.aggregate["rmse"]


def test_adoption_task_reports_its_no_skill_baseline(dataset, protocol):
    model = BiasBaseline(reg=5.0).fit(dataset)
    metrics = adoption_classification(model, protocol, threshold=3.5)
    assert metrics["no_skill_accuracy"] == pytest.approx(1.0 - metrics["positive_rate"])
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["n_examples"] == sum(c.size for c in protocol.candidates)
    assert metrics["decision_threshold"] == 3.5


def test_history_buckets_cover_the_whole_range():
    counts = np.array([0, 5, 20, 29, 30, 49, 50, 99, 100, 250])
    labels = history_bucket(counts)
    assert labels[0] == "<20"
    assert labels[2] == "20-29"
    assert labels[4] == "30-49"
    assert labels[6] == "50-99"
    assert labels[8] == ">=100"


def test_popularity_buckets_are_ordered_by_count():
    counts = np.array([1, 1, 2, 3, 4, 5, 10, 100], dtype=np.float64)
    labels = popularity_bucket(counts)
    assert labels[0] == "Q1 (coldest)"
    assert labels[-1] == "Q4 (hottest)"
    assert len(set(labels.tolist())) == 4


def test_sliced_report_exposes_the_three_breakdowns(dataset, cfg, protocol):
    result = evaluate(BiasBaseline(reg=5.0).fit(dataset), dataset, protocol, cfg)
    slices = sliced_report(result, dataset)
    assert set(slices) == {"by_history_ranking", "by_history_error", "by_popularity_error"}
    ranking = slices["by_history_ranking"]
    assert {"history_bucket", "users", "ndcg", "hit"}.issubset(ranking.columns)
    assert slices["by_popularity_error"]["pairs"].sum() == int((dataset.fold == TEST).sum())
