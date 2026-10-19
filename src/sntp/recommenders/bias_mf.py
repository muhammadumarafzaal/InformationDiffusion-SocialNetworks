"""Biased matrix factorisation trained with mini-batch Adam (implemented from scratch).

Objective (explicit feedback, ratings in [1, 5]):

    L = sum_(u,i) (r_ui - mu - b_u - b_i - p_u . q_i)^2
        + lambda * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

The optimiser is Adam over sparse row updates: only users/items present in a
mini-batch receive a gradient step, which keeps a full epoch at O(number of
ratings * factors) instead of O(users * factors).
"""

from __future__ import annotations

import numpy as np

from ..data import VALID, Dataset
from ..utils import seeded_rng
from .base import Recommender


def _group_sum(inverse: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Row-wise accumulation of ``values`` into ``n_groups`` groups.

    Equivalent to ``np.add.at``: the batch is sorted once and reduced with
    ``np.add.reduceat`` instead of doing individual scatter-adds in a loop. It is
    neutral at the 1,024-row batch used in training and pulls ahead once the batch
    grows (1.55x at 4,096 rows), which is why both paths are benchmarked in
    ``experiments/rq8_intervention.py`` rather than assumed.
    """
    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_groups)) + 1))
    reduced = np.add.reduceat(values[order], starts, axis=0)
    if reduced.shape[0] == n_groups:
        return reduced
    # defensive: groups absent from the batch keep a zero gradient
    out = np.zeros((n_groups, values.shape[1]))
    out[np.unique(inverse)] = reduced
    return out


class BiasMF(Recommender):
    name = "bias_mf"

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.002,
        reg: float = 0.5,
        epochs: int = 60,
        batch_size: int = 4096,
        init_std: float = 0.02,
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
        self.batch_size = int(batch_size)
        self.init_std = float(init_std)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.patience = int(patience)
        self.mu = 0.0
        self.bu = np.zeros(0)
        self.bi = np.zeros(0)
        self.P = np.zeros((0, 0))
        self.Q = np.zeros((0, 0))
        self.history: dict[str, list[float]] = {"epoch": [], "train_rmse": [], "valid_rmse": []}

    # ------------------------------------------------------------------ helpers
    def _init_params(self, data: Dataset, rng: np.random.Generator, warm_start: bool) -> None:
        self.mu = float(data.ratings[data.fold == 0].mean())
        self.bu = np.zeros(self.n_users, dtype=np.float64)
        self.bi = np.zeros(self.n_items, dtype=np.float64)
        if warm_start:
            # Adam moves each parameter by roughly `learning_rate` per epoch, so
            # biases started at zero would still be far from convergence after a
            # few dozen epochs. Initialising them from the ridge bias model both
            # accelerates convergence and gives the latent factors a clean signal.
            from .baselines import BiasBaseline

            bias = BiasBaseline(reg=10.0).fit(data)
            self.mu = bias.mu
            self.bu = bias.bu.copy()
            self.bi = bias.bi.copy()
        self.P = rng.normal(0.0, self.init_std, size=(self.n_users, self.n_factors))
        self.Q = rng.normal(0.0, self.init_std, size=(self.n_items, self.n_factors))
        self._states = {
            key: [np.zeros_like(value), np.zeros_like(value)]
            for key, value in {
                "bu": self.bu,
                "bi": self.bi,
                "P": self.P,
                "Q": self.Q,
            }.items()
        }
        self._step = 0

    def _adam(self, name: str, param: np.ndarray, idx: np.ndarray, grad: np.ndarray) -> None:
        m, v = self._states[name]
        m[idx] = self.beta1 * m[idx] + (1.0 - self.beta1) * grad
        v[idx] = self.beta2 * v[idx] + (1.0 - self.beta2) * (grad**2)
        m_hat = m[idx] / (1.0 - self.beta1**self._step)
        v_hat = v[idx] / (1.0 - self.beta2**self._step)
        param[idx] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    # --------------------------------------------------------------- extension
    def _epoch_gradient(self) -> np.ndarray | None:
        """Optional dense gradient added to ``P`` on every mini-batch of an epoch.

        Social models override this hook to inject a regulariser that does not
        decompose over individual interactions. Returning ``None`` keeps the
        optimiser identical to plain biased MF, which makes ``beta = 0`` a true
        control condition. The gradient is computed once per epoch and re-used
        within it (a standard stale-gradient approximation).
        """
        return None

    # ------------------------------------------------------------------ fitting
    def fit(self, data: Dataset, validation: bool = True, warm_start: bool = True) -> BiasMF:
        self.n_users, self.n_items = data.n_users, data.n_items
        rng = seeded_rng(self.seed, stream=101)
        self._init_params(data, rng, warm_start)

        users, items, ratings = data.interactions()
        val_mask = data.fold == VALID
        val_users, val_items, val_ratings = (
            data.rating_user[val_mask],
            data.rating_item[val_mask],
            data.ratings[val_mask],
        )
        best = (np.inf, 0, None)
        stale = 0

        for epoch in range(1, self.epochs + 1):
            extra_grad = self._epoch_gradient()
            permutation = rng.permutation(ratings.size)
            squared_error = 0.0
            for start in range(0, ratings.size, self.batch_size):
                batch = permutation[start : start + self.batch_size]
                bu_idx = users[batch]
                bi_idx = items[batch]
                target = ratings[batch]
                u_rows, u_inv = np.unique(bu_idx, return_inverse=True)
                i_rows, i_inv = np.unique(bi_idx, return_inverse=True)

                prediction = (
                    self.mu
                    + self.bu[bu_idx]
                    + self.bi[bi_idx]
                    + np.einsum("ij,ij->i", self.P[bu_idx], self.Q[bi_idx])
                )
                error = prediction - target
                squared_error += float(np.sum(error**2))
                self._step += 1

                grad_bu = np.bincount(u_inv, weights=error, minlength=u_rows.size)
                grad_bu += self.reg * self.bu[u_rows]
                self._adam("bu", self.bu, u_rows, grad_bu)

                grad_bi = np.bincount(i_inv, weights=error, minlength=i_rows.size)
                grad_bi += self.reg * self.bi[i_rows]
                self._adam("bi", self.bi, i_rows, grad_bi)

                q_batch = self.Q[bi_idx]
                p_batch = self.P[bu_idx]
                grad_p = _group_sum(u_inv, error[:, None] * q_batch, u_rows.size)
                if extra_grad is not None:
                    grad_p += extra_grad[u_rows]
                grad_p += self.reg * self.P[u_rows]
                self._adam("P", self.P, u_rows, grad_p)

                grad_q = _group_sum(i_inv, error[:, None] * p_batch, i_rows.size)
                grad_q += self.reg * self.Q[i_rows]
                self._adam("Q", self.Q, i_rows, grad_q)

            train_rmse = float(np.sqrt(squared_error / ratings.size))
            if validation and val_ratings.size:
                val_pred = self._raw_predict(val_users, val_items)
                val_rmse = float(np.sqrt(np.mean((np.clip(val_pred, 1.0, 5.0) - val_ratings) ** 2)))
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
        return {
            "bu": self.bu.copy(),
            "bi": self.bi.copy(),
            "P": self.P.copy(),
            "Q": self.Q.copy(),
        }

    def _restore(self, state: dict[str, np.ndarray]) -> None:
        self.bu = state["bu"].copy()
        self.bi = state["bi"].copy()
        self.P = state["P"].copy()
        self.Q = state["Q"].copy()

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

    def item_scores(self, item: int) -> np.ndarray:
        """Score of one item for every user in O(n_users * factors) without a
        dense (users x items) matrix. Used by the diffusion simulator."""
        self._check_fitted()
        return self.mu + self.bu + self.bi[int(item)] + self.P @ self.Q[int(item)]

    def describe(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "n_factors": self.n_factors,
            "learning_rate": self.learning_rate,
            "reg": self.reg,
            "best_epoch": getattr(self, "best_epoch", self.epochs),
            "best_valid_rmse": round(getattr(self, "best_valid_rmse", float("nan")), 4),
        }
