"""Item-based k-NN collaborative filtering on bias-corrected residuals.

Design choices
--------------
* Similarity: shrunk cosine on residuals ``r_ui - mu - b_u - b_i`` so that item
  popularity cannot masquerade as similarity.
* Neighbourhood: per-item top-K (K = 60) computed with blocked sparse products.
  Exact computation over all item pairs would need O(n_items^2) memory; blocking
  keeps the peak at ``block_size x n_items`` without approximating.
* Shrinkage: ``cos * n_ij / (n_ij + gamma)`` suppresses pairs with few co-ratings.
* Prediction: weighted average of the user's residuals on the item's neighbours.
  Items without neighbours fall back to the bias model, which is also the
  cold-item behaviour we analyse in Requirement 2.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..data import Dataset
from .baselines import BiasBaseline


class ItemKNN(BiasBaseline):
    name = "item_knn"

    def __init__(
        self,
        n_neighbors: int = 60,
        shrinkage: float = 5.0,
        block_size: int = 256,
        reg: float = 10.0,
        confidence_shrinkage: float = 0.0,
        max_correction: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(reg=reg, seed=seed)
        self.n_neighbors = int(n_neighbors)
        self.shrinkage = float(shrinkage)
        self.block_size = int(block_size)
        self.confidence_shrinkage = float(confidence_shrinkage)
        self.max_correction = float(max_correction)
        self._sim: sparse.csr_matrix | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, data: Dataset) -> ItemKNN:
        super().fit(data)
        residual = self.residual_matrix(data)
        self._centered = residual
        squared = residual.multiply(residual)
        column_norm = np.sqrt(np.asarray(squared.sum(axis=0)).ravel())
        inverse = np.divide(
            1.0, column_norm, out=np.zeros_like(column_norm), where=column_norm > 1e-9
        )
        normalised = (residual @ sparse.diags(inverse.astype(np.float32))).tocsc()

        binary = residual.copy()
        binary.data = np.ones_like(binary.data)
        binary = binary.tocsc()

        k = min(self.n_neighbors, self.n_items - 1)
        rows_out, cols_out, vals_out = [], [], []
        for start in range(0, self.n_items, self.block_size):
            stop = min(start + self.block_size, self.n_items)
            block = slice(start, stop)
            similarity = (normalised[:, block].T @ normalised).toarray()
            co_ratings = (binary[:, block].T @ binary).toarray()
            with np.errstate(invalid="ignore", divide="ignore"):
                shrinkage = co_ratings / (co_ratings + self.shrinkage)
            similarity *= shrinkage
            local_rows = np.arange(stop - start)
            similarity[local_rows, np.arange(start, stop)] = -np.inf
            top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
            row_index = np.repeat(local_rows + start, k)
            col_index = top.ravel()
            values = similarity[np.repeat(local_rows, k), col_index]
            keep = np.isfinite(values) & (values > 1e-6)
            rows_out.append(row_index[keep])
            cols_out.append(col_index[keep])
            vals_out.append(values[keep])

        self._sim = sparse.csr_matrix(
            (np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
            shape=(self.n_items, self.n_items),
        )
        self._sim_t = self._sim.T.tocsr()
        self._sim_abs_t = abs(self._sim).T.tocsr()
        self._fitted = True
        return self

    # --------------------------------------------------------------- prediction
    def predict_matrix(self, users: np.ndarray) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(users, dtype=np.int64)
        chunk = self._centered[users]
        numerator = (chunk @ self._sim_t).toarray()
        absolute = chunk.copy()
        absolute.data = np.abs(absolute.data)
        denominator = (absolute @ self._sim_abs_t).toarray()
        residual = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9
        )
        if self.confidence_shrinkage > 0:
            # Low-support items are predicted by few (or unstable) neighbours; shrink
            # their correction towards the bias model instead of trusting noise.
            residual *= denominator / (denominator + self.confidence_shrinkage)
        if self.max_correction > 0:
            # A neighbourhood correction larger than half the rating scale is almost
            # always noise on a sparse core and only inflates MAE/RMSE.
            residual = np.clip(residual, -self.max_correction, self.max_correction)
        base = self.mu + self.bu[users][:, None] + self.bi[None, :]
        return (base + residual).astype(np.float32)

    def item_similarity(self) -> sparse.csr_matrix:
        self._check_fitted()
        return self._sim

    def describe(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "n_neighbors": self.n_neighbors,
            "shrinkage": self.shrinkage,
            "confidence_shrinkage": self.confidence_shrinkage,
            "max_correction": self.max_correction,
            "reg": self.reg,
        }
