"""The dataset layer must be deterministic and leakage-free."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sntp.data import TEST, TRAIN, VALID, assign_folds, core_filter


def test_core_filter_enforces_the_requested_thresholds():
    frame = pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "item": [10, 11, 12, 10, 11, 10, 11, 12, 13],
            "rating": [5.0] * 9,
        }
    )
    filtered = core_filter(frame, min_user=3, min_item=2)
    assert set(filtered["user"]) == {1, 3}
    assert filtered.groupby("user").size().min() >= 3
    assert filtered.groupby("item").size().min() >= 2


def test_core_filter_is_idempotent():
    frame = pd.DataFrame(
        {
            "user": np.repeat(np.arange(6), 4),
            "item": np.tile(np.arange(4), 6),
            "rating": 3.0,
        }
    )
    once = core_filter(frame, min_user=2, min_item=2)
    twice = core_filter(once, min_user=2, min_item=2)
    assert len(once) == len(twice)


def test_assign_folds_partitions_every_interaction_once():
    user = np.repeat(np.arange(40), 10)
    fold = assign_folds(user, 40, val_fraction=0.1, test_fraction=0.1, seed=3)
    assert fold.shape == user.shape
    assert set(np.unique(fold)) == {TRAIN, VALID, TEST}
    # exactly one validation and one test interaction per user
    for user_id in range(40):
        counts = np.bincount(fold[user == user_id], minlength=3)
        assert counts[TEST] == 1
        assert counts[VALID] == 1
        assert counts.sum() == 10


def test_assign_folds_is_reproducible():
    user = np.repeat(np.arange(30), 8)
    first = assign_folds(user, 30, 0.1, 0.1, seed=99)
    second = assign_folds(user, 30, 0.1, 0.1, seed=99)
    other = assign_folds(user, 30, 0.1, 0.1, seed=100)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_assign_folds_keeps_at_least_one_training_interaction():
    user = np.repeat(np.arange(10), 3)
    fold = assign_folds(user, 10, 0.2, 0.2, seed=1)
    assert (np.bincount(fold[fold == TRAIN], minlength=3)[TRAIN]) > 0
    for user_id in range(10):
        assert (fold[user == user_id] == TRAIN).sum() >= 1


def test_influence_graph_reverses_the_stored_trust_direction(dataset):
    influence = dataset.influence_graph().tocsr()
    stored = dataset.trust_graph().tocsr()
    assert influence.shape == (dataset.n_users, dataset.n_users)
    assert influence.nnz == stored.nnz
    user, target = int(dataset.trust_src[0]), int(dataset.trust_dst[0])
    assert influence[target, user] == pytest.approx(1.0)
    assert stored[user, target] == pytest.approx(1.0)


def test_matrix_accessors_match_the_fold_masks(dataset):
    train = dataset.rating_matrix((TRAIN,))
    everything = dataset.observed_pairs()
    assert train.nnz == int((dataset.fold == TRAIN).sum())
    assert everything.nnz == dataset.n_ratings
    assert train.nnz < everything.nnz
    values = train.data
    assert values.min() >= 1.0 and values.max() <= 5.0
