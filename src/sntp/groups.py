"""Group recommendation under conflicting preferences (Requirement 6).

Three social-choice aggregation rules are compared on the same candidate set:

``average``        maximise the mean predicted satisfaction of the group
                   (utilitarian / "additive" rule);
``least_misery``   maximise the minimum predicted satisfaction
                   (egalitarian, protects the worst-off member);
``most_pleasure``  maximise the maximum predicted satisfaction
                   (dictatorial towards the most enthusiastic member).

Groups are built in two flavours:

``friends``   an anchor user plus a random sample of its trust neighbours;
``conflict``  an anchor plus the most dissimilar users found among a random pool
              (lowest cosine similarity on bias-corrected rating profiles), which
              deliberately creates the conflicting-preference case in the task.

Only predicted satisfaction is used to choose an item; the realised diffusion of
the chosen item is evaluated afterwards, so the two quantities stay independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .data import TRAIN, Dataset
from .graph import symmetrise
from .recommenders.baselines import BiasBaseline
from .utils import seeded_rng

STRATEGIES = ("average", "least_misery", "most_pleasure")


@dataclass
class Group:
    index: int
    members: np.ndarray
    kind: str


def _profile_matrix(data: Dataset) -> sparse.csr_matrix:
    bias = BiasBaseline(reg=10.0).fit(data)
    return bias.residual_matrix(data)


def _cosine_to_rows(
    matrix: sparse.csr_matrix, rows: np.ndarray, target: sparse.csr_matrix
) -> np.ndarray:
    """Cosine between every row in ``rows`` and the single row ``target``."""
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    block = matrix[rows]
    vector = np.asarray(target.todense()).ravel()
    dots = np.asarray(block @ vector).ravel()
    denominator = norms[rows] * float(np.linalg.norm(vector))
    return np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator > 1e-12)


def build_groups(
    data: Dataset,
    size: int = 5,
    n_groups: int = 60,
    kind: str = "friends",
    seed: int = 0,
    pool_size: int = 400,
) -> list[Group]:
    rng = seeded_rng(seed, stream=606)
    train_counts = np.bincount(data.rating_user[data.fold == TRAIN], minlength=data.n_users)
    eligible = np.flatnonzero(train_counts >= 10)

    if kind == "friends":
        graph = symmetrise(data.influence_graph()).tocsr()
        degree = np.diff(graph.indptr)
        anchors = eligible[degree[eligible] >= size - 1]
        groups: list[Group] = []
        for index in range(n_groups):
            anchor = int(rng.choice(anchors))
            neighbours = graph.indices[graph.indptr[anchor] : graph.indptr[anchor + 1]]
            picks = rng.choice(neighbours, size=min(size - 1, neighbours.size), replace=False)
            groups.append(Group(index, np.concatenate(([anchor], picks)).astype(np.int64), kind))
        return groups

    if kind != "conflict":
        raise ValueError(f"unknown group kind '{kind}'")

    residual = _profile_matrix(data)
    residual = residual.tocsr()
    groups = []
    for index in range(n_groups):
        anchor = int(rng.choice(eligible))
        pool = rng.choice(eligible, size=min(pool_size, eligible.size), replace=False)
        pool = np.setdiff1d(pool, np.array([anchor]))
        similarity = _cosine_to_rows(residual, pool, residual.getrow(anchor))
        order = np.argsort(similarity, kind="stable")
        picks = pool[order[: size - 1]]
        groups.append(Group(index, np.concatenate(([anchor], picks)).astype(np.int64), kind))
    return groups


def candidate_items(
    score_matrix: np.ndarray,
    n_per_member: int = 10,
    global_items: np.ndarray | None = None,
) -> np.ndarray:
    """Union of each member's top-N predictions plus global popular items."""
    candidates: set[int] = set()
    for row in score_matrix:
        candidates.update(np.argsort(-row, kind="stable")[:n_per_member].tolist())
    if global_items is not None:
        candidates.update(int(item) for item in global_items)
    return np.array(sorted(candidates), dtype=np.int64)


def aggregate_item(scores: np.ndarray, strategy: str) -> int:
    """Pick one candidate index from a (members x candidates) satisfaction matrix."""
    if strategy == "average":
        return int(np.argmax(scores.mean(axis=0)))
    if strategy == "least_misery":
        return int(np.argmax(scores.min(axis=0)))
    if strategy == "most_pleasure":
        return int(np.argmax(scores.max(axis=0)))
    raise ValueError(f"unknown group strategy '{strategy}'")


def group_satisfaction(scores: np.ndarray, column: int) -> dict[str, float]:
    values = scores[:, column]
    return {
        "mean_satisfaction": float(values.mean()),
        "min_satisfaction": float(values.min()),
        "max_satisfaction": float(values.max()),
        "std_satisfaction": float(values.std(ddof=0)),
        "spread_members": float(values.max() - values.min()),
    }


def consensus_rank(scores: np.ndarray) -> int:
    """Item with the lowest average absolute deviation from each member's ideal.

    A consensus-oriented reference point: it does not look at the mean level, only
    at disagreement, so it shows whether least-misery is really the fairest rule.
    """
    deviations = np.abs(scores - scores.max(axis=1, keepdims=True))
    return int(np.argmin(deviations.mean(axis=0)))
