"""Shared plumbing for the experiment drivers: paths, caching and campaign setup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data import TRAIN, Dataset, prepare_dataset
from ..diffusion import DiffusionSettings, affinity_from_scores
from ..recommenders import build_recommender
from ..recommenders.base import Recommender
from ..utils import ensure_dir, hardware_summary, write_json


@dataclass
class Context:
    cfg: Config
    dataset: Dataset
    results_dir: Path

    # ------------------------------------------------------------------ folders
    @property
    def raw_dir(self) -> Path:
        return ensure_dir(self.results_dir / "raw")

    @property
    def table_dir(self) -> Path:
        return ensure_dir(self.results_dir / "tables")

    @property
    def figure_dir(self) -> Path:
        return ensure_dir(self.results_dir / "figures")

    @property
    def summary_dir(self) -> Path:
        return ensure_dir(self.results_dir / "summary")

    # ------------------------------------------------------------------ outputs
    def save_raw(self, frame: pd.DataFrame, name: str) -> Path:
        target = self.raw_dir / f"{name}.csv"
        frame.to_csv(target, index=False, float_format="%.6f")
        return target

    def save_table(self, frame: pd.DataFrame, name: str, caption: str | None = None) -> Path:
        target = self.table_dir / f"{name}.csv"
        frame.to_csv(target, index=False, float_format="%.4f")
        if caption:
            target.with_suffix(".md").write_text(
                f"**{caption}**\n\n{frame.to_markdown(index=False, floatfmt='.4f')}\n",
                encoding="utf-8",
            )
        return target

    def save_json(self, payload: dict[str, Any], name: str) -> Path:
        target = self.summary_dir / f"{name}.json"
        write_json(target, payload)
        return target

    def load_json(self, name: str) -> dict[str, Any]:
        target = self.summary_dir / f"{name}.json"
        return json.loads(target.read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return (self.summary_dir / f"{name}.json").exists()

    # ------------------------------------------------------------------ models
    def model(self, name: str, seed: int = 0) -> Recommender:
        return build_recommender(name, self.cfg, seed=seed).fit(self.dataset)

    def cached_model(self, name: str, seed: int = 0) -> Recommender:
        """Reuse the seed-11 model produced by the recommendation experiment."""
        cache = self.results_dir / "cache" / f"{name}_seed{seed}.npz"
        if not cache.exists():
            raise FileNotFoundError(
                f"missing cached model {cache}; run `sntp run recommenders` first"
            )
        return load_model(cache)


def build_context(cfg: Config) -> Context:
    """Load (or build) the processed dataset and wrap it with output folders."""
    return Context(
        cfg=cfg,
        dataset=prepare_dataset(cfg),
        results_dir=cfg.path("paths.results_dir"),
    )


def save_model(model: Recommender, path: Path, metadata: dict[str, Any] | None = None) -> Path:
    """Persist the numeric state of a fitted model (no pickle)."""
    ensure_dir(path.parent)
    payload: dict[str, np.ndarray] = {}
    for key in ("mu", "bu", "bi", "P", "Q"):
        if hasattr(model, key):
            payload[key] = np.asarray(getattr(model, key))
    payload["name"] = np.array([model.name])
    metadata = metadata or {}
    payload["metadata"] = np.array([json.dumps(metadata)])
    np.savez_compressed(path, **payload)
    return path


def load_model(path: Path) -> Recommender:
    archive = np.load(path, allow_pickle=False)
    name = str(archive["name"][0])
    if name == "bias_mf":
        from ..recommenders.bias_mf import BiasMF

        model: Recommender = BiasMF()
    elif name == "social_mf":
        from ..recommenders.social import SocialMF

        model = SocialMF()
    else:
        raise ValueError(f"cannot reload model '{name}'")
    for key in ("mu", "bu", "bi", "P", "Q"):
        if key in archive:
            value = archive[key]
            setattr(model, key, float(value) if value.ndim == 0 else value)
    model.n_users = model.P.shape[0]
    model.n_items = model.Q.shape[0]
    model._fitted = True
    return model


# ---------------------------------------------------------------- campaign setup
def select_campaign_item(
    dataset: Dataset, model: Recommender, pool: int = 200, seed: int = 0
) -> int:
    """Choose the item the platform promotes.

    Deterministic rule: among the ``pool`` most-rated training items (a realistic
    mainstream campaign target), take the one whose mean predicted affinity is
    highest. Using a broadly appealing item keeps the campaign interpretable and
    avoids selecting an item only a handful of users could ever adopt.
    """
    counts = np.bincount(dataset.rating_item[dataset.fold == TRAIN], minlength=dataset.n_items)
    top = np.argsort(-counts, kind="stable")[:pool]
    means = np.array([model.item_scores(int(item)).mean() for item in top])
    return int(top[int(np.argmax(means))])


def campaign_affinity(model: Recommender, item: int) -> np.ndarray:
    scores = np.clip(model.item_scores(item), 1.0, 5.0)
    return affinity_from_scores(scores)


def diffusion_settings(ctx: Context, **overrides: Any) -> DiffusionSettings:
    return DiffusionSettings.from_config(ctx.cfg, **overrides)


def eligible_users(dataset: Dataset, min_interactions: int = 1) -> np.ndarray:
    counts = np.bincount(dataset.rating_user[dataset.fold == TRAIN], minlength=dataset.n_users)
    return np.flatnonzero(counts >= min_interactions)


def context_footer(ctx: Context) -> dict[str, Any]:
    return {
        "hardware": hardware_summary(),
        "config": ctx.cfg.to_dict(),
        "dataset": ctx.dataset.metadata,
    }


def snapshot_config(ctx: Context) -> None:
    ctx.cfg.save(ctx.results_dir / "config_snapshot.yaml")
