"""Social recommendation models (Requirement 3).

``SocialMF`` adds a SocialMF-style regulariser to biased matrix factorisation:

    L = L_rating + beta * sum_u || p_u - sum_{v in N(u)} s_uv p_v ||^2
    dL/dP (social) = 2 * beta * (E - S^T E),  E = P - S P

``S`` is the row-stochastic influence graph (``S[u, v] > 0`` means u trusts v's
taste). ``beta`` is exposed as a knob so the contribution of social information
can be switched off (beta = 0) or amplified, which is what Requirement 3 asks
for. The gradient is injected through ``BiasMF._epoch_gradient`` so the optimiser
is bit-for-bit the same as plain biased MF; ``beta = 0`` is therefore a genuine
control rather than a re-implementation.

``NeighbourResidualRecommender`` is a deliberately simple social predictor with
no latent factors: it predicts a user's residual for an item as the trust-weighted
average of the residuals of the people they trust. It is the cleanest test of
whether friends carry preference information at all, before any factorisation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..data import Dataset
from ..graph import row_normalise
from .base import Recommender
from .baselines import BiasBaseline
from .bias_mf import BiasMF


class SocialMF(BiasMF):
    name = "social_mf"

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.002,
        reg: float = 0.2,
        epochs: int = 40,
        batch_size: int = 4096,
        init_std: float = 0.02,
        beta: float = 0.25,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        patience: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n_factors=n_factors,
            learning_rate=learning_rate,
            reg=reg,
            epochs=epochs,
            batch_size=batch_size,
            init_std=init_std,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            patience=patience,
            seed=seed,
        )
        self.beta = float(beta)
        self._social: sparse.csr_matrix | None = None
        self.history["social_norm"] = []

    # ------------------------------------------------------------------ fitting
    def fit(  # type: ignore[override]
        self,
        data: Dataset,
        graph: sparse.csr_matrix | None = None,
        beta: float | None = None,
    ) -> SocialMF:
        if beta is not None:
            self.beta = float(beta)
        if graph is None or self.beta == 0.0:
            self._social = None
        else:
            # Row-normalised so that each user is pulled towards the *mean*
            # taste of the people they trust, independent of their out-degree.
            self._social = row_normalise(graph).tocsr()
        self.history["social_norm"] = []
        super().fit(data)
        return self

    def _epoch_gradient(self) -> np.ndarray | None:
        if self._social is None:
            return None
        spread = self._social @ self.P
        edge_error = self.P - spread
        gradient = 2.0 * self.beta * (edge_error - self._social.T @ edge_error)
        self.history["social_norm"].append(float(np.linalg.norm(edge_error)))
        return gradient

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
