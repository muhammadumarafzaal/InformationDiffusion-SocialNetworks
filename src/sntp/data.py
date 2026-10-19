"""Dataset construction for the Epinions trust + rating network.

Observed data (from the public Epinions release shipped in ``data/raw``):

* ``ratings_data.txt`` -- ``user_id item_id rating`` with ratings on a 1..5 scale.
* ``trust_data.txt``   -- ``source target trust`` with ``+1`` meaning *source trusts target*.

Everything below either (a) filters/remaps these observed records or (b) derives
evaluation folds. No synthetic preference values are created here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config
from .utils import ensure_dir, write_json

TRAIN, VALID, TEST = 0, 1, 2


@dataclass
class Dataset:
    """Remapped Epinions core plus a per-interaction train/valid/test fold."""

    name: str
    user_ids: np.ndarray  # original Epinions user ids, index = internal id
    item_ids: np.ndarray  # original Epinions item ids
    rating_user: np.ndarray  # internal user id per rating
    rating_item: np.ndarray  # internal item id per rating
    ratings: np.ndarray  # float32 values in [1, 5]
    fold: np.ndarray  # int8: 0 train, 1 valid, 2 test
    trust_src: np.ndarray  # internal id, "src trusts dst"
    trust_dst: np.ndarray
    trust_weight: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def n_users(self) -> int:
        return int(self.user_ids.size)

    @property
    def n_items(self) -> int:
        return int(self.item_ids.size)

    @property
    def n_ratings(self) -> int:
        return int(self.ratings.size)

    def fold_mask(self, fold: int) -> np.ndarray:
        return self.fold == fold

    def interactions(self, folds: tuple[int, ...] = (TRAIN,)) -> tuple[np.ndarray, ...]:
        mask = np.isin(self.fold, folds)
        return self.rating_user[mask], self.rating_item[mask], self.ratings[mask]

    def observed_pairs(self) -> sparse.csr_matrix:
        """Boolean user x item matrix of every observed interaction (all folds)."""
        data = np.ones(self.n_ratings, dtype=np.int8)
        return sparse.csr_matrix(
            (data, (self.rating_user, self.rating_item)), shape=(self.n_users, self.n_items)
        )

    def rating_matrix(self, folds: tuple[int, ...] = (TRAIN,)) -> sparse.csr_matrix:
        users, items, values = self.interactions(folds)
        return sparse.csr_matrix(
            (values.astype(np.float32), (users, items)), shape=(self.n_users, self.n_items)
        )

    def influence_graph(self) -> sparse.csr_matrix:
        """Directed influence graph: edge ``v -> u`` means *u trusts v*.

        Epinions stores ``source target`` = "source trusts target", so influence
        flows in the opposite direction of the stored record.
        """
        return sparse.csr_matrix(
            (self.trust_weight.astype(np.float32), (self.trust_dst, self.trust_src)),
            shape=(self.n_users, self.n_users),
        )

    def trust_graph(self) -> sparse.csr_matrix:
        """Directed graph in the stored direction (src -trusts-> dst)."""
        return sparse.csr_matrix(
            (self.trust_weight.astype(np.float32), (self.trust_src, self.trust_dst)),
            shape=(self.n_users, self.n_users),
        )

    def user_degree(self) -> np.ndarray:
        return np.asarray(self.influence_graph().getnnz(axis=1)).ravel()

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        target = ensure_dir(directory)
        np.savez_compressed(
            target / "dataset.npz",
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            rating_user=self.rating_user,
            rating_item=self.rating_item,
            ratings=self.ratings,
            fold=self.fold,
            trust_src=self.trust_src,
            trust_dst=self.trust_dst,
            trust_weight=self.trust_weight,
        )
        write_json(target / "metadata.json", self.metadata)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> Dataset:
        target = Path(directory)
        archive = np.load(target / "dataset.npz", allow_pickle=False)
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            name=str(metadata.get("name", "epinions")),
            user_ids=archive["user_ids"],
            item_ids=archive["item_ids"],
            rating_user=archive["rating_user"],
            rating_item=archive["rating_item"],
            ratings=archive["ratings"].astype(np.float32),
            fold=archive["fold"].astype(np.int8),
            trust_src=archive["trust_src"],
            trust_dst=archive["trust_dst"],
            trust_weight=archive["trust_weight"].astype(np.float32),
            metadata=metadata,
        )


# --------------------------------------------------------------------- reading
def read_ratings(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["user", "item", "rating"],
        dtype={"user": np.int64, "item": np.int64, "rating": np.float32},
    )
    return frame.drop_duplicates(subset=["user", "item"], keep="first")


def read_trust(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["source", "target", "trust"],
        dtype={"source": np.int64, "target": np.int64, "trust": np.float32},
    )
    frame = frame[frame["source"] != frame["target"]]
    return frame.drop_duplicates(subset=["source", "target"], keep="first")


def raw_statistics(ratings: pd.DataFrame, trust: pd.DataFrame) -> dict[str, Any]:
    users_per_item = ratings.groupby("item").size()
    items_per_user = ratings.groupby("user").size()
    pair_set = set(zip(trust["source"].tolist(), trust["target"].tolist()))
    reciprocal = sum(1 for s, t in pair_set if (t, s) in pair_set) / max(len(pair_set), 1)
    return {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user"].nunique()),
        "n_items": int(ratings["item"].nunique()),
        "density": float(len(ratings) / (ratings["user"].nunique() * ratings["item"].nunique())),
        "rating_counts": {str(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        "rating_mean": float(ratings["rating"].mean()),
        "ratings_per_user_median": float(items_per_user.median()),
        "ratings_per_user_p90": float(items_per_user.quantile(0.90)),
        "ratings_per_item_median": float(users_per_item.median()),
        "trust_edges": int(len(trust)),
        "trust_sources": int(trust["source"].nunique()),
        "trust_targets": int(trust["target"].nunique()),
        "trust_values": {str(k): int(v) for k, v in trust["trust"].value_counts().items()},
        "trust_mean_out_degree": float(len(trust) / max(trust["source"].nunique(), 1)),
        "trust_reciprocity": float(reciprocal),
        "trust_nodes_without_ratings": int(
            len((set(trust["source"]) | set(trust["target"])) - set(ratings["user"]))
        ),
    }


# -------------------------------------------------------------------- filtering
def core_filter(
    ratings: pd.DataFrame, min_user: int, min_item: int, max_rounds: int = 10
) -> pd.DataFrame:
    """Iteratively keep users/items with at least the requested interaction counts.

    A single pass is not enough: dropping sparse items can push users below the
    threshold, so the filter is repeated until the bipartite core is stable.
    """
    current = ratings
    for _ in range(max_rounds):
        users = current.groupby("user").size()
        keep_users = users[users >= min_user].index
        current = current[current["user"].isin(keep_users)]
        items = current.groupby("item").size()
        keep_items = items[items >= min_item].index
        filtered = current[current["item"].isin(keep_items)]
        if len(filtered) == len(current):
            current = filtered
            break
        current = filtered
    return current.reset_index(drop=True)


def _remap(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(values)
    lookup = {int(v): idx for idx, v in enumerate(unique)}
    return np.array([lookup[int(v)] for v in values], dtype=np.int64), unique


def assign_folds(
    rating_user: np.ndarray,
    n_users: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> np.ndarray:
    """Per-user random hold-out (no timestamps exist, so time splits are impossible)."""
    fold = np.full(rating_user.size, TRAIN, dtype=np.int8)
    order = np.argsort(rating_user, kind="stable")
    sorted_users = rating_user[order]
    boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_users.size]))
    rng = np.random.default_rng(seed)
    for start, end in zip(starts, ends, strict=True):
        idx = order[start:end]
        shuffled = rng.permutation(idx)
        n = shuffled.size
        n_test = max(1, int(round(test_fraction * n)))
        n_valid = max(1, int(round(val_fraction * n)))
        n_test = min(n_test, n - 2)
        n_valid = min(n_valid, n - n_test - 1)
        fold[shuffled[:n_test]] = TEST
        fold[shuffled[n_test : n_test + n_valid]] = VALID
    return fold


def prepare_dataset(cfg: Config, force: bool = False) -> Dataset:
    """Build (or load) the processed Epinions core described by ``cfg``."""
    processed_dir = cfg.path("paths.processed_dir") / str(cfg["dataset.name"])
    if (processed_dir / "dataset.npz").exists() and not force:
        return Dataset.load(processed_dir)

    ratings_raw = read_ratings(cfg.path("paths.raw_ratings"))
    trust_raw = read_trust(cfg.path("paths.raw_trust"))
    stats_raw = raw_statistics(ratings_raw, trust_raw)

    min_user = int(cfg["dataset.min_user_ratings"])
    min_item = int(cfg["dataset.min_item_ratings"])
    ratings = core_filter(ratings_raw, min_user, min_item)

    rating_user, user_ids = _remap(ratings["user"].to_numpy())
    rating_item, item_ids = _remap(ratings["item"].to_numpy())
    ratings_values = ratings["rating"].to_numpy(dtype=np.float32)

    user_set = set(user_ids.tolist())
    trust = trust_raw[
        trust_raw["source"].isin(user_set) & trust_raw["target"].isin(user_set)
    ].copy()
    src, _ = _remap_users(trust["source"].to_numpy(), user_ids)
    dst, _ = _remap_users(trust["target"].to_numpy(), user_ids)
    trust_weight = trust["trust"].to_numpy(dtype=np.float32)
    keep = src >= 0
    keep &= dst >= 0
    src, dst, trust_weight = src[keep], dst[keep], trust_weight[keep]
    self_loop = src != dst
    src, dst, trust_weight = src[self_loop], dst[self_loop], trust_weight[self_loop]

    fold = assign_folds(
        rating_user,
        user_ids.size,
        float(cfg["dataset.split.val_fraction"]),
        float(cfg["dataset.split.test_fraction"]),
        int(cfg["dataset.split.seed"]),
    )

    n_users, n_items = user_ids.size, item_ids.size
    metadata = {
        "name": str(cfg["dataset.name"]),
        "source": {
            "ratings": str(cfg["paths.raw_ratings"]),
            "trust": str(cfg["paths.raw_trust"]),
        },
        "filters": {"min_user_ratings": min_user, "min_item_ratings": min_item},
        "raw": stats_raw,
        "core": {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_ratings": int(ratings_values.size),
            "density": float(ratings_values.size / (n_users * n_items)),
            "rating_mean": float(ratings_values.mean()),
            "rating_std": float(ratings_values.std()),
            "rating_counts": {
                str(k): int(v)
                for k, v in zip(*np.unique(ratings_values, return_counts=True), strict=True)
            },
            "trust_edges": int(src.size),
            "trust_weight_mean": float(trust_weight.mean()),
            "mean_out_degree": float(src.size / max(n_users, 1)),
            "median_out_degree": float(np.median(np.bincount(src, minlength=n_users))),
            "max_out_degree": int(np.bincount(src, minlength=n_users).max()),
            "max_in_degree": int(np.bincount(dst, minlength=n_users).max()),
            "users_without_influence_edges": int((np.bincount(src, minlength=n_users) == 0).sum()),
            "reciprocity": float(
                len(set(zip(src.tolist(), dst.tolist(), strict=True))
                    & {(b, a) for a, b in zip(src.tolist(), dst.tolist(), strict=True)})
                / max(src.size, 1)
            ),
        },
        "split": {
            "strategy": "per-user random hold-out (no timestamps in Epinions)",
            "val_fraction": float(cfg["dataset.split.val_fraction"]),
            "test_fraction": float(cfg["dataset.split.test_fraction"]),
            "seed": int(cfg["dataset.split.seed"]),
            "n_train": int((fold == TRAIN).sum()),
            "n_valid": int((fold == VALID).sum()),
            "n_test": int((fold == TEST).sum()),
            "min_train_per_user": int(
                np.bincount(rating_user[fold == TRAIN], minlength=n_users).min()
            ),
            "mean_train_per_user": float(
                np.bincount(rating_user[fold == TRAIN], minlength=n_users).mean()
            ),
        },
    }

    dataset = Dataset(
        name=str(cfg["dataset.name"]),
        user_ids=np.asarray(user_ids, dtype=np.int64),
        item_ids=np.asarray(item_ids, dtype=np.int64),
        rating_user=rating_user,
        rating_item=rating_item,
        ratings=ratings_values,
        fold=fold,
        trust_src=src,
        trust_dst=dst,
        trust_weight=trust_weight,
        metadata=metadata,
    )
    dataset.save(processed_dir)
    write_json(processed_dir / "statistics.json", metadata)
    return dataset


def _remap_users(values: np.ndarray, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {int(v): idx for idx, v in enumerate(user_ids)}
    mapped = np.array([lookup.get(int(v), -1) for v in values], dtype=np.int64)
    return mapped, user_ids
