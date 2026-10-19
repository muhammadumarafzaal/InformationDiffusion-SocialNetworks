"""Social recommendation models (Requirement 3).

``SocialMF`` adds a SocialMF-style regulariser to biased matrix factorisation:

    L = L_rating + beta * sum_u || p_u - sum_{v in N(u)} s_uv p_v ||^2
    dL/dP (social) = 2 * beta * (E - S^T E),  E = P - S P

``S`` is the row-stochastic influence graph (``S[u, v] > 0`` means u trusts v's
taste). ``beta`` is exposed as a knob so the contribution of social information
can be switched off (beta = 0) or amplified, which is what Requirement 3 asks for.

``NeighbourResidualRecommender`` is a deliberately simple social predictor with no
latent factors: it predicts a user's residual for an item as the trust-weighted
average of the residuals of the people they trust. It is the cleanest test of
whether friends carry preference information at all, before any factorisation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..data import VALID, Dataset
from ..graph import row_normalise
from ..utils import seeded_rng
from .base import Recommender
from .baselines import BiasBaseline


class SocialMF(Recommender):
    name = "social_mf"

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        reg: float = 0.08,
        epochs: int = 150,
        init_std: float = 0.05,
        beta: float = 0.25,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        patience: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self.n_factors = int(n_factors)
        self.learning_rate = float(learning_rate)
        self.reg = float(reg)
        self.epochs = int(epochs)
        self.init_std = float(init_std)
        self.beta = float(beta)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.patience = int(patience)
        self.history: dict[str, list[float]] = {"epoch": [], "train_rmse": [], "valid_rmse": []}

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        data: Dataset,
        graph: sparse.csr_matrix | None = None,
        beta: float | None = None,
    ) -> SocialMF:
        if beta is not None:
            self.beta = float(beta)
        self.n_users, self.n_items = data.n_users, data.n_items
        users, items, ratings = data.interactions()
        self.mu = float(ratings.mean())

        bias = BiasBaseline(reg=10.0).fit(data)
        self.bu, self.bi = bias.bu.copy(), bias.bi.copy()

        rng = seeded_rng(self.seed, stream=202)
        self.P = rng.normal(0.0, self.init_std, size=(self.n_users, self.n_factors))
        self.Q = rng.normal(0.0, self.init_std, size=(self.n_items, self.n_factors))

        if graph is None or self.beta == 0.0:
            influence = None
        else:
            influence = row_normalise(graph)

        states = {
            name: [np.zeros_like(value), np.zeros_like(value)]
            for name, value in {"bu": self.bu, "bi": self.bi, "P": self.P, "Q": self.Q}.items()
        }
        step = 0
        user_counts = np.bincount(users, minlength=self.n_users).astype(np.float64)
        item_counts = np.bincount(items, minlength=self.n_items).astype(np.float64)

        val_mask = data.fold == VALID
        val_users, val_items, val_ratings = (
            data.rating_user[val_mask],
            data.rating_item[val_mask],
            data.ratings[val_mask],
        )
        best = (np.inf, 0, None)
        stale = 0

        for epoch in range(1, self.epochs + 1):
            step += 1
            prediction = (
                self.mu
                + self.bu[users]
                + self.bi[items]
                + np.einsum("ij,ij->i", self.P[users], self.Q[items])
            )
            error = prediction - ratings

            grad_bu = np.bincount(users, weights=error, minlength=self.n_users) + self.reg * self.bu
            grad_bi = np.bincount(items, weights=error, minlength=self.n_items) + self.reg * self.bi
            grad_p = np.zeros_like(self.P)
            np.add.at(grad_p, users, error[:, None] * self.Q[items])
            grad_q = np.zeros_like(self.Q)
            np.add.at(grad_q, items, error[:, None] * self.P[users])
            grad_p += self.reg * self.P
            grad_q += self.reg * self.Q

            if influence is not None:
                spread = influence @ self.P
                edge_error = self.P - spread
                grad_p += 2.0 * self.beta * (edge_error - influence.T @ edge_error)

            for name, param, grad in (
                ("bu", self.bu, grad_bu),
                ("bi", self.bi, grad_bi),
                ("P", self.P, grad_p),
                ("Q", self.Q, grad_q),
            ):
                m, v = states[name]
                m *= self.beta1
                m += (1.0 - self.beta1) * grad
                v *= self.beta2
                v += (1.0 - self.beta2) * (grad**2)
                m_hat = m / (1.0 - self.beta1**step)
                v_hat = v / (1.0 - self.beta2**step)
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            train_rmse = float(np.sqrt(np.mean(error**2)))
            if val_ratings.size:
                val_pred = np.clip(self._raw_predict(val_users, val_items), 1.0, 5.0)
                val_rmse = float(np.sqrt(np.mean((val_pred - val_ratings) ** 2)))
            else:
                val_rmse = train_rmse
            self.history["epoch"].append(epoch)
            self.history["train_rmse"].append(train_rmse)
            self.history["valid_rmse"].append(val_rmse)

            if val_rmse < best[0] - 1e-5:
                best = (val_rmse, epoch, self._snapshot())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best[2] is not None:
            self._restore(best[2])
        self.best_epoch = int(best[1])
        self.best_valid_rmse = float(best[0])
        self._fitted = True
        return self

    def _snapshot(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).copy() for k in ("bu", "bi", "P", "Q")}

    def _restore(self, state: dict[str, np.ndarray]) -> None:
        for key, value in state.items():
            setattr(self, key, value.copy())

    # --------------------------------------------------------------- prediction
    def _raw_predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.einsum("ij,ij->i", self.P[users], self.Q[items])
        )

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        return self._raw_predict(users, items)

    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(users, dtype=np.int64)
        return (
            self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T
        ).astype(np.float32)

    def item_embeddings(self) -> np.ndarray:
        return self.Q

    def describe(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "beta": self.beta,
            "n_factors": self.n_factors,
            "best_epoch": getattr(self, "best_epoch", self.epochs),
            "best_valid_rmse": round(getattr(self, "best_valid_rmse", float("nan")), 4),
        }


class NeighbourResidualRecommender(Recommender):
    """Predict a residual as the weighted mean residual of graph neighbours.

    ``graph[u, v]`` must be an influence edge (u listens to v). No learning happens
    beyond the bias model, so any gain over the bias baseline is direct evidence
    that the neighbours' preferences transfer to the target user.
    """

    name = "neighbour_residual"

    def __init__(self, graph: sparse.csr_matrix, label: str = "trust", seed: int = 0) -> None:
        super().__init__(seed)
        self.raw_graph = graph.tocsr()
        self.label = label

    def fit(self, data: Dataset) -> NeighbourResidualRecommender:
        bias = BiasBaseline(reg=10.0).fit(data)
        self.mu, self.bu, self.bi = bias.mu, bias.bu, bias.bi
        self.n_users, self.n_items = data.n_users, data.n_items
        self._residual = bias.residual_matrix(data)
        self._binary = self._residual.copy()
        self._binary.data = np.ones_like(self._binary.data)
        self._graph = row_normalise(self.raw_graph)
        self.coverage = float((self.raw_graph.getnnz(axis=1) > 0).mean())
        self._fitted = True
        return self

    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(users, dtype=np.int64)
        rows = self._graph[users]
        numerator = (rows @ self._residual).toarray()
        denominator = (rows @ self._binary).toarray()
        social = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9
        )
        base = self.mu + self.bu[users][:, None] + self.bi[None, :]
        return (base + social).astype(np.float32)

    def describe(self) -> dict[str, float | str]:
        return {"model": self.name, "graph": self.label}
