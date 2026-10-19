"""Shared fixtures for the unit tests.

The tests never touch the real Epinions files: they build a small synthetic
network with latent taste clusters so that every claim a model makes (better
error, better ranking, influence flowing along edges) is verifiable by
construction. A ``Config`` that mirrors the production defaults is used
throughout, with the heavy hyper-parameters shrunk so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from sntp.config import Config
from sntp.data import Dataset, assign_folds


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load(
        None,
        overrides={
            "recommendation.n_factors": 8,
            "recommendation.epochs": 6,
            "recommendation.batch_size": 256,
            "recommendation.social.beta": 0.25,
            "diffusion.n_sims": 20,
            "diffusion.max_steps": 20,
        },
    )


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """80 users, 40 items, 4 taste clusters, 14 ratings per user."""
    rng = np.random.default_rng(2026)
    n_users, n_items, n_clusters = 80, 40, 4
    ratings_per_user = 14

    user_cluster = rng.integers(0, n_clusters, size=n_users)
    item_cluster = np.arange(n_items) % n_clusters
    cluster_quality = rng.uniform(3.0, 4.6, size=(n_clusters, n_clusters))

    users, items, values = [], [], []
    for user in range(n_users):
        items_all = np.arange(n_items)
        weight = np.where(item_cluster == user_cluster[user], 6.0, 1.0)
        picked = rng.choice(
            items_all, size=ratings_per_user, replace=False, p=weight / weight.sum()
        )
        for item in picked:
            base = cluster_quality[user_cluster[user], item_cluster[item]]
            rating = int(np.clip(round(base + rng.normal(0.0, 0.6)), 1, 5))
            users.append(user)
            items.append(int(item))
            values.append(rating)

    rating_user = np.asarray(users, dtype=np.int64)
    rating_item = np.asarray(items, dtype=np.int64)
    ratings = np.asarray(values, dtype=np.float32)
    fold = assign_folds(rating_user, n_users, 0.1, 0.1, seed=7)

    # Each user listens to three users of the same taste cluster.
    src, dst = [], []
    for user in range(n_users):
        peers = np.flatnonzero(user_cluster == user_cluster[user])
        peers = peers[peers != user]
        for peer in rng.choice(peers, size=min(3, peers.size), replace=False):
            src.append(user)
            dst.append(int(peer))

    return Dataset(
        name="synthetic",
        user_ids=np.arange(n_users, dtype=np.int64),
        item_ids=np.arange(n_items, dtype=np.int64),
        rating_user=rating_user,
        rating_item=rating_item,
        ratings=ratings,
        fold=fold,
        trust_src=np.asarray(src, dtype=np.int64),
        trust_dst=np.asarray(dst, dtype=np.int64),
        trust_weight=np.ones(len(src), dtype=np.float32),
        metadata={"name": "synthetic"},
    )
