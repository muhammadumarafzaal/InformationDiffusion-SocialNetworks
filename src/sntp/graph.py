"""Graph utilities: trust influence graph, similarity pseudo-graphs, random controls."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .utils import seeded_rng


def row_normalise(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-stochastic version of a non-negative sparse matrix (zero rows stay zero)."""
    matrix = matrix.tocsr(copy=True)
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums > 0)
    return sparse.diags(inv.astype(np.float32)) @ matrix


def symmetrise(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Union of both directions, preserving the maximum observed weight."""
    return matrix.maximum(matrix.T).tocsr()


def top_k_neighbours(similarity: np.ndarray, k: int, exclude_self: bool = True) -> tuple[np.ndarray, ...]:
    """Row-wise top-k of a dense similarity block, returned as (rows, cols, values)."""
    n = similarity.shape[0]
    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)
    k = min(k, max(1, n - 1))
    idx = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = similarity[rows, cols]
    keep = np.isfinite(vals) & (vals > 0)
    return rows[keep], cols[keep], vals[keep]


def similarity_graph(
    user_factors: np.ndarray, n_neighbors: int, min_similarity: float = 0.05, block: int = 1024
) -> sparse.csr_matrix:
    """k-NN graph over users using cosine similarity of learned/observed profiles.

    Used to build the *similar-but-unconnected* control graph for Requirement 3:
    the graph has a comparable out-degree to the trust graph but ignores the
    observed trust relations.
    """
    norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    unit = user_factors / np.maximum(norms, 1e-8)
    n = unit.shape[0]
    rows_all, cols_all, vals_all = [], [], []
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = unit[start:stop] @ unit.T
        local = similarity.copy()
        np.fill_diagonal(local[:, start:stop], -np.inf)
        k = min(n_neighbors, n - 1)
        idx = np.argpartition(-local, k - 1, axis=1)[:, :k]
        r = np.repeat(np.arange(start, stop), k)
        c = idx.ravel()
        v = local[np.arange(start, stop).repeat(k), c]
        keep = np.isfinite(v) & (v >= min_similarity)
        rows_all.append(r[keep])
        cols_all.append(c[keep])
        vals_all.append(v[keep])
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)
    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # keep the strongest k per row after thresholding
    return _prune_rows(graph, n_neighbors)


def _prune_rows(graph: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    graph = graph.tocsr()
    rows, cols, vals = [], [], []
    for row in range(graph.shape[0]):
        start, end = graph.indptr[row], graph.indptr[row + 1]
        if start == end:
            continue
        row_vals = graph.data[start:end]
        row_cols = graph.indices[start:end]
        if row_vals.size > k:
            top = np.argpartition(-row_vals, k - 1)[:k]
            row_vals, row_cols = row_vals[top], row_cols[top]
        rows.append(np.full(row_vals.size, row))
        cols.append(row_cols)
        vals.append(row_vals)
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=graph.shape,
    )


def user_similarity_graph(
    residual: sparse.csr_matrix,
    n_neighbors: int,
    exclude: sparse.csr_matrix | None = None,
    min_similarity: float = 0.05,
) -> sparse.csr_matrix:
    """k-NN graph of *similar but socially unconnected* users (Requirement 3).

    Similarity is the cosine between bias-corrected rating profiles, computed from
    the training fold only. Edges that exist in ``exclude`` (the observed trust
    graph) are removed before top-k selection, so the resulting graph answers the
    counterfactual: what if the platform knew the same number of neighbours per
    user, chosen by taste similarity instead of by trust?
    """
    rows = residual.shape[0]
    binary = residual.copy()
    binary.data = np.ones_like(binary.data)
    norms = np.sqrt(np.asarray(residual.multiply(residual).sum(axis=1)).ravel())
    safe = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 1e-9)
    unit = sparse.diags(safe.astype(np.float32)) @ residual
    similarity = (unit @ unit.T).tocsr()

    out_rows, out_cols, out_vals = [], [], []
    block = 512
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        dense = similarity[start:stop].toarray()
        local = np.arange(stop - start)
        dense[local, np.arange(start, stop)] = 0.0
        if exclude is not None:
            blocked = exclude[start:stop]
            dense[blocked.toarray() > 0] = 0.0
        k = min(n_neighbors, rows - 1)
        top = np.argpartition(-dense, k - 1, axis=1)[:, :k]
        row_ids = np.repeat(np.arange(start, stop), k)
        col_ids = top.ravel()
        vals = dense[np.arange(start, stop).repeat(k), col_ids]
        keep = np.isfinite(vals) & (vals >= min_similarity)
        out_rows.append(row_ids[keep])
        out_cols.append(col_ids[keep])
        out_vals.append(vals[keep])
    graph = sparse.csr_matrix(
        (np.concatenate(out_vals), (np.concatenate(out_rows), np.concatenate(out_cols))),
        shape=(rows, rows),
    )
    return graph


def rewired_control(graph: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """Degree-preserving random control: shuffle edge targets, drop self loops.

    Keeps the out-degree sequence of the influence graph but destroys the
    assortative structure of trust, isolating the effect of *who* trusts *whom*.
    """
    rng = seeded_rng(seed, stream=7717)
    graph = graph.tocsr()
    src = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    dst = graph.indices.copy()
    weights = graph.data.copy()
    for _ in range(5):
        perm = rng.permutation(dst.size)
        dst = dst[perm]
        bad = dst == src
        if not bad.any():
            break
        # repair self loops by swapping with a random position
        for pos in np.flatnonzero(bad):
            swap = int(rng.integers(0, dst.size))
            dst[pos], dst[swap] = dst[swap], dst[pos]
    keep = dst != src
    return sparse.csr_matrix(
        (weights[keep], (src[keep], dst[keep])), shape=graph.shape
    )


def graph_statistics(graph: sparse.csr_matrix) -> dict[str, float]:
    graph = graph.tocsr()
    out_degree = np.diff(graph.indptr)
    in_degree = np.asarray(graph.getnnz(axis=0)).ravel()
    n_edges = int(graph.nnz)
    n = graph.shape[0]
    mutual = graph.multiply(graph.T).nnz
    return {
        "nodes": int(n),
        "edges": n_edges,
        "density": float(n_edges / (n * (n - 1))) if n > 1 else 0.0,
        "mean_out_degree": float(out_degree.mean()),
        "median_out_degree": float(np.median(out_degree)),
        "max_out_degree": int(out_degree.max()) if n else 0,
        "isolated_nodes": int((out_degree == 0).sum()),
        "mean_in_degree": float(in_degree.mean()),
        "reciprocity": float(mutual / max(n_edges, 1)),
        "largest_component_fraction": float(_largest_component_fraction(graph)),
    }


def _largest_component_fraction(graph: sparse.csr_matrix) -> float:
    """Size of the largest weakly connected component as a fraction of nodes."""
    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(graph, directed=True, connection="weak")
    if n_components == 0:
        return 0.0
    counts = np.bincount(labels)
    return float(counts.max() / graph.shape[0])
