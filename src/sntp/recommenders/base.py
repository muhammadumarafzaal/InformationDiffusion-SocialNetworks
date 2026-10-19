"""Common interface for every recommender implemented in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from ..data import Dataset


class Recommender(ABC):
    """Rating predictor with an explicit dense scoring API.

    ``predict_matrix`` is the primitive used by the evaluation harness: it returns
    a ``(len(users), n_items)`` dense score matrix for the requested users. Ranking
    and error metrics are then computed from the same scores, so the reported
    ranking quality is exactly what the model would serve.

    The central recommendation logic is implemented here from scratch; NumPy and
    SciPy are used only for linear algebra and sparse storage.
    """

    name: str = "recommender"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.n_users: int = 0
        self.n_items: int = 0
        self._fitted = False

    # ------------------------------------------------------------------ fitting
    @abstractmethod
    def fit(self, data: Dataset) -> Recommender:
        """Fit on the training fold of ``data`` only."""

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted first")

    # --------------------------------------------------------------- prediction
    @abstractmethod
    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        """Dense scores for ``users`` (rows) over all items."""

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        out = np.empty(users.size, dtype=np.float64)
        # group by user to reuse one score row per user
        order = np.argsort(users, kind="stable")
        sorted_users = users[order]
        boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
        for group in np.split(order, boundaries):
            user = int(users[group[0]])
            row = self.predict_matrix(np.array([user]))[0]
            out[group] = row[items[group]]
        return out

    def predict_all(self, users: np.ndarray | None = None, chunk: int = 512) -> np.ndarray:
        users = np.arange(self.n_users) if users is None else np.asarray(users)
        chunks = [self.predict_matrix(users[s : s + chunk]) for s in range(0, users.size, chunk)]
        return np.vstack(chunks) if chunks else np.empty((0, self.n_items))

    # ------------------------------------------------------------------ ranking
    def top_n(
        self,
        user: int,
        n: int = 10,
        candidates: np.ndarray | None = None,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``n`` items for one user, optionally restricted to ``candidates``."""
        pool = (
            np.arange(self.n_items)
            if candidates is None
            else np.asarray(candidates, dtype=np.int64)
        )
        scores = self.predict_matrix(np.array([user]))[0][pool]
        if exclude is not None and exclude.size:
            mask = ~np.isin(pool, exclude)
            pool, scores = pool[mask], scores[mask]
        order = np.argsort(-scores, kind="stable")[:n]
        return pool[order], scores[order]

    # ------------------------------------------------------------- side outputs
    def item_embeddings(self) -> np.ndarray | None:
        """Item representation used for diversity metrics, if the model has one."""
        return None

    def item_similarity(self) -> sparse.csr_matrix | None:
        return None

    def describe(self) -> dict[str, float | str]:
        return {"model": self.name}
