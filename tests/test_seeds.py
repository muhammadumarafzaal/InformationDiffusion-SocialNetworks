"""Seed-selection contracts used by Requirement 5."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from sntp.diffusion import CampaignSimulator, DiffusionSettings
from sntp.seeds import (
    SeedSelector,
    audience_overlap,
    degree_selection,
    expected_influence_proxy,
    preference_selection,
    prefilter_candidates,
    random_selection,
    rank_agreement,
)


def _graph() -> sparse.csr_matrix:
    # node 0 is the only hub: four outgoing edges, everybody else has at most one
    rows = [0, 0, 0, 0, 1, 2, 3]
    cols = [1, 2, 3, 4, 2, 3, 4]
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(12, 12))


def _simulator() -> CampaignSimulator:
    graph = _graph()
    affinity = np.full(graph.shape[0], 0.6)
    settings = DiffusionSettings(model="ic", p0=0.2, seed_acceptance=False, max_steps=8)
    return CampaignSimulator(graph, affinity, settings)


def test_greedy_returns_distinct_seeds_from_the_candidate_pool():
    selector = SeedSelector(_simulator(), seed=1)
    candidates = np.arange(12)
    selection = selector.greedy(candidates, budget=3, n_mc=3, celf=True)
    assert selection.seeds.size == 3
    assert np.unique(selection.seeds).size == 3
    assert set(selection.seeds.tolist()).issubset(set(candidates.tolist()))
    assert selection.gains.size == 3
    assert selection.simulations > 0


def test_greedy_finds_the_hub_first():
    # only node 0 has a large audience, so a marginal-gain search must pick it
    selection = SeedSelector(_simulator(), seed=2).greedy(np.arange(12), budget=1, n_mc=15)
    assert selection.seeds.tolist() == [0]
    assert selection.gains[0] > 0


def test_degree_selection_picks_the_hub():
    graph = _graph()
    chosen = degree_selection(graph, np.arange(12), budget=1)
    assert chosen.tolist() == [0]


def test_preference_selection_sorts_by_score():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    chosen = preference_selection(scores, np.array([0, 1, 2, 3]), budget=2)
    assert chosen.tolist() == [1, 3]


def test_random_selection_is_reproducible_and_bounded():
    candidates = np.arange(50)
    first = random_selection(candidates, budget=5, seed=7)
    second = random_selection(candidates, budget=5, seed=7)
    other = random_selection(candidates, budget=5, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.unique(first).size == 5


def test_prefilter_keeps_the_requested_pool_size():
    proxy = np.linspace(1.0, 0.0, 100)
    eligible = np.arange(100)
    pool = prefilter_candidates(proxy, eligible, pool_size=10, seed=0)
    assert pool.size == 10
    assert pool.tolist() == list(range(10))
    assert prefilter_candidates(proxy, eligible, 0, seed=0).size == 100


def test_expected_influence_proxy_is_non_negative_and_finite():
    proxy = expected_influence_proxy(_graph(), np.linspace(0.1, 0.9, 12))
    assert np.isfinite(proxy).all()
    assert (proxy >= 0).all()


def test_audience_overlap_detects_identical_and_disjoint_audiences():
    identical = sparse.csr_matrix(
        (np.ones(4, dtype=np.float32), ([0, 0, 1, 1], [2, 3, 2, 3])), shape=(4, 4)
    )
    overlap = audience_overlap(np.array([0, 1]), identical, seed=1)
    # four impressions, two distinct users: half of the delivery is duplicated
    assert overlap["audience_overlap"] == 0.5
    assert overlap["unique_audience"] == 2.0
    assert overlap["mean_pairwise_jaccard"] == 1.0

    disjoint = sparse.csr_matrix((np.ones(2, dtype=np.float32), ([0, 1], [2, 3])), shape=(4, 4))
    separated = audience_overlap(np.array([0, 1]), disjoint, seed=1)
    assert separated["audience_overlap"] == 0.0
    assert separated["sum_audience"] == 2.0


def test_rank_agreement_is_one_for_a_monotone_transformation():
    gains = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_agreement(gains, gains * 2.0 + 5.0) > 0.99
    assert rank_agreement(gains, -gains) < -0.99
