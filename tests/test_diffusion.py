"""Diffusion semantics: events, models, interventions and determinism."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from sntp.diffusion import (
    CampaignSimulator,
    DiffusionSettings,
    affinity_from_scores,
)
from sntp.utils import seeded_rng


def _ring(n: int = 30, extra: int = 40) -> sparse.csr_matrix:
    rng = np.random.default_rng(11)
    rows = list(range(n))
    cols = [(i + 1) % n for i in range(n)]
    rows.extend(rng.integers(0, n, size=extra).tolist())
    cols.extend(rng.integers(0, n, size=extra).tolist())
    keep = np.asarray(rows) != np.asarray(cols)
    return sparse.csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (np.asarray(rows)[keep], np.asarray(cols)[keep]),
        ),
        shape=(n, n),
    )


def _simulator(model: str = "ic", **overrides) -> CampaignSimulator:
    graph = _ring()
    affinity = np.linspace(0.2, 0.9, graph.shape[0])
    settings = DiffusionSettings(
        model=model,
        p0=overrides.pop("p0", 0.08),
        influence_weight=overrides.pop("influence_weight", 0.7),
        seed_acceptance=overrides.pop("seed_acceptance", False),
        recovery_prob=overrides.pop("recovery_prob", 0.35),
        max_steps=overrides.pop("max_steps", 20),
        **overrides,
    )
    return CampaignSimulator(graph, affinity, settings)


def test_seeds_always_adopt_when_acceptance_is_disabled():
    simulator = _simulator()
    result = simulator.run(np.array([3, 7, 9]), seeded_rng(0, stream=1))
    assert result.adopted[[3, 7, 9]].all()
    assert result.seed_adoptions == 3
    assert np.all(result.adoption_time[[3, 7, 9]] == 0)


def test_zero_transmission_leaves_only_the_seeds():
    simulator = _simulator(p0=0.0)
    result = simulator.run(np.array([1, 2]), seeded_rng(0, stream=2))
    assert result.reach == 2
    assert result.recommended.sum() == 2


def test_reach_is_never_below_the_seed_count():
    for model in ("ic", "sir"):
        simulator = _simulator(model=model)
        result = simulator.run(np.array([0, 5, 10, 15]), seeded_rng(3, stream=4))
        assert result.reach >= 4
        assert 0.0 <= result.reach_fraction <= 1.0


def test_higher_transmission_reaches_more_users_on_average():
    weak = _simulator(p0=0.02).simulate(np.array([0, 5]), n_simulations=20, seed=7)
    strong = _simulator(p0=0.25).simulate(np.array([0, 5]), n_simulations=20, seed=7)
    assert strong.reach_fraction_mean > weak.reach_fraction_mean


def test_simulations_are_reproducible_for_a_fixed_seed():
    simulator = _simulator()
    first = simulator.simulate(np.array([2, 11]), n_simulations=5, seed=42)
    second = simulator.simulate(np.array([2, 11]), n_simulations=5, seed=42)
    assert np.allclose(first.adoption_curve, second.adoption_curve)
    assert first.aggregate()["reach_mean"] == second.aggregate()["reach_mean"]


def test_exposures_are_only_recorded_when_requested():
    quiet = _simulator(track_exposures=False)
    tracking = _simulator(track_exposures=True)
    seeds = np.array([0])
    assert quiet.run(seeds, seeded_rng(0, stream=9)).exposures.sum() == 0
    assert tracking.run(seeds, seeded_rng(0, stream=9)).exposures.sum() > 0


def test_zero_exposure_cap_blocks_everything_after_the_seeds():
    open_simulator = _simulator()
    capped = _simulator(exposure_cap=0)
    seeds = np.array([0, 4])
    open_reach = open_simulator.simulate(seeds, n_simulations=20, seed=5).reach_fraction_mean
    capped_reach = capped.simulate(seeds, n_simulations=20, seed=5).reach_fraction_mean
    assert capped_reach == pytest.approx(2 / 30)
    assert capped_reach < open_reach


def test_affinity_gate_cannot_increase_reach():
    open_simulator = _simulator()
    gated = _simulator(affinity_gate=0.8)
    seeds = np.array([0, 6, 12])
    open_reach = open_simulator.simulate(seeds, n_simulations=25, seed=8).reach_fraction_mean
    gated_reach = gated.simulate(seeds, n_simulations=25, seed=8).reach_fraction_mean
    assert gated_reach <= open_reach


def test_sir_cascades_finish_within_the_step_budget():
    simulator = _simulator(model="sir", max_steps=6)
    result = simulator.run(np.array([0, 1, 2]), seeded_rng(0, stream=12))
    assert result.steps <= 6
    assert len(result.frontier_sizes) == result.steps + 1


def test_unknown_model_raises():
    simulator = _simulator(model="bogus")
    with pytest.raises(ValueError):
        simulator.run(np.array([0]), seeded_rng(0, stream=1))


def test_time_to_ninety_percent_is_monotone_in_the_curve():
    simulator = _simulator()
    result = simulator.run(np.array([0, 3]), seeded_rng(1, stream=13))
    assert 0 <= result.time_to_90pct <= result.steps
    assert result.peak_size >= 1


def test_affinity_from_scores_maps_and_clips():
    scores = np.array([0.0, 1.0, 3.0, 5.0, 9.0])
    mapped = affinity_from_scores(scores)
    assert mapped[0] == 0.0
    assert mapped[1] == 0.0
    assert mapped[3] == 1.0
    assert mapped[4] == 1.0
    assert mapped[2] == pytest.approx(0.5)
