"""Group aggregation rules and their declared trade-offs."""

from __future__ import annotations

import numpy as np
import pytest

from sntp.groups import (
    STRATEGIES,
    aggregate_item,
    candidate_items,
    consensus_rank,
    group_satisfaction,
)


def _scores() -> np.ndarray:
    # rows = members, columns = candidate items
    return np.array(
        [
            [5.0, 3.0, 1.0],
            [5.0, 4.0, 1.0],
            [2.0, 4.0, 5.0],
        ]
    )


def test_average_maximises_the_mean():
    scores = _scores()
    assert aggregate_item(scores, "average") == 0
    assert scores[:, 0].mean() > scores[:, 1].mean() > scores[:, 2].mean()


def test_least_misery_protects_the_worst_off_member():
    scores = _scores()
    chosen = aggregate_item(scores, "least_misery")
    assert chosen == 1
    assert scores[:, chosen].min() == 3.0
    assert scores[:, chosen].min() > scores[:, 0].min()


def test_most_pleasure_follows_the_most_enthusiastic_member():
    scores = _scores()
    assert aggregate_item(scores, "most_pleasure") == 0
    assert scores[:, 0].max() == 5.0


def test_consensus_minimises_the_worst_disagreement():
    scores = np.array(
        [
            [5.0, 4.0],
            [5.0, 4.0],
            [1.0, 4.0],
        ]
    )
    assert consensus_rank(scores) == 1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        aggregate_item(_scores(), "dictator")


def test_group_satisfaction_reports_the_spread():
    scores = _scores()
    summary = group_satisfaction(scores, 0)
    assert summary["mean_satisfaction"] == pytest.approx(4.0)
    assert summary["min_satisfaction"] == 2.0
    assert summary["max_satisfaction"] == 5.0
    assert summary["spread_members"] == pytest.approx(3.0)


def test_candidate_items_unions_per_member_top_n_and_global_items():
    scores = np.array(
        [
            [0.1, 0.9, 0.2, 0.3],
            [0.8, 0.2, 0.1, 0.3],
        ]
    )
    candidates = candidate_items(scores, n_per_member=1, global_items=np.array([3]))
    assert candidates.tolist() == [0, 1, 3]
    assert candidates.tolist() == sorted(candidates.tolist())


def test_declared_strategies_are_complete():
    assert STRATEGIES == ("average", "least_misery", "most_pleasure")
