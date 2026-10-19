"""Non-personalised baselines and the shared bias model.

``BiasBaseline`` (global mean + regularised user/item biases) is both a stand-alone
predictor and the de-biasing step reused by the item-based k-NN recommender.
"""

from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Recommender


class GlobalMeanRecommender(Recommender):
    """Predicts the training mean for everybody (lower bound for error metrics)."""

    name = "global_mean"

    def fit(self, data: Dataset) -> GlobalMeanRecommender:
        _, _, ratings = data.interactions()
        self.mu = float(ratings.mean())
        self.n_users, self.n_items = data.n_users, data.n_items
        self._fitted = True
        return self

    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.full((np.asarray(users).size, self.n_items), self.mu, dtype=np.float32)


class BiasBaseline(Recommender):
    """Regularised bias model: ``mu + b_u + b_i``.

    Biases are estimated by alternating ridge updates on the training fold, which
    is exactly the de-biasing step used before computing item-item residuals.
    """

    name = "bias_baseline"

    def __init__(self, reg: float = 10.0, n_sweeps: int = 10, seed: int = 0) -> None:
        super().__init__(seed)
        self.reg = float(reg)
        self.n_sweeps = int(n_sweeps)
        self.mu = 0.0
        self.bu = np.zeros(0)
        self.bi = np.zeros(0)

    def fit(self, data: Dataset) -> BiasBaseline:
        users, items, ratings = data.interactions()
        self.n_users, self.n_items = data.n_users, data.n_items
        self.mu = float(ratings.mean())
        self.bu = np.zeros(self.n_users, dtype=np.float64)
        self.bi = np.zeros(self.n_items, dtype=np.float64)
        user_counts = np.bincount(users, minlength=self.n_users).astype(np.float64)
        item_counts = np.bincount(items, minlength=self.n_items).astype(np.float64)
        for _ in range(self.n_sweeps):
            residual = ratings - self.mu - self.bi[items]
            self.bu = np.bincount(users, weights=residual, minlength=self.n_users) / (
                user_counts + self.reg
            )
            residual = ratings - self.mu - self.bu[users]
            self.bi = np.bincount(items, weights=residual, minlength=self.n_items) / (
                item_counts + self.reg
            )
        self._fitted = True
        return self

    def residual_matrix(self, data: Dataset):
        """Sparse matrix of bias-corrected training residuals (zeros where unobserved)."""
        users, items, ratings = data.interactions()
        residual = ratings - self.mu - self.bu[users] - self.bi[items]
        from scipy import sparse

        return sparse.csr_matrix(
            (residual.astype(np.float32), (users, items)), shape=(self.n_users, self.n_items)
        )

    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(users, dtype=np.int64)
        return (self.mu + self.bu[users][:, None] + self.bi[None, :]).astype(np.float32)


class PopularityRecommender(BiasBaseline):
    """Same bias model, but ranking ignores the user term (pure item popularity)."""

    name = "popularity"

    def fit(self, data: Dataset) -> PopularityRecommender:
        super().fit(data)
        # shrunk mean rating, used for the ranking side only
        _, items, ratings = data.interactions()
        counts = np.bincount(items, minlength=self.n_items).astype(np.float64)
        sums = np.bincount(items, weights=ratings, minlength=self.n_items)
        self.item_mean = (sums + self.reg * self.mu) / (counts + self.reg)
        self._counts = counts
        return self

    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(users, dtype=np.int64)
        row = self.mu + self.bi
        return np.repeat(row[None, :], users.size, axis=0).astype(np.float32)
