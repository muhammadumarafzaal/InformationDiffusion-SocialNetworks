"""Recommender implementations and a small factory used by the experiments."""

from __future__ import annotations

from ..config import Config
from .base import Recommender
from .baselines import BiasBaseline, GlobalMeanRecommender, PopularityRecommender
from .bias_mf import BiasMF
from .item_knn import ItemKNN
from .social import NeighbourResidualRecommender, SocialMF

__all__ = [
    "BiasBaseline",
    "BiasMF",
    "GlobalMeanRecommender",
    "ItemKNN",
    "NeighbourResidualRecommender",
    "PopularityRecommender",
    "Recommender",
    "SocialMF",
    "build_recommender",
]


def build_recommender(name: str, cfg: Config, seed: int = 0) -> Recommender:
    """Instantiate a model from the ``recommendation`` config section."""
    settings = cfg.section("recommendation")
    if name == "global_mean":
        return GlobalMeanRecommender(seed=seed)
    if name == "bias_baseline":
        return BiasBaseline(seed=seed)
    if name == "popularity":
        return PopularityRecommender(seed=seed)
    if name == "item_knn":
        return ItemKNN(
            n_neighbors=int(settings["knn_neighbors"]),
            shrinkage=float(settings["knn_shrinkage"]),
            max_correction=float(settings.get("knn_max_correction", 0.5)),
            seed=seed,
        )
    if name == "bias_mf":
        return BiasMF(
            n_factors=int(settings["n_factors"]),
            learning_rate=float(settings["learning_rate"]),
            reg=float(settings["reg"]),
            epochs=int(settings["epochs"]),
            batch_size=int(settings["batch_size"]),
            init_std=float(settings["init_std"]),
            seed=seed,
        )
    if name == "social_mf":
        return SocialMF(
            n_factors=int(settings["n_factors"]),
            learning_rate=float(settings["learning_rate"]),
            reg=float(settings["reg"]),
            init_std=float(settings["init_std"]),
            beta=float(settings["social.beta"]),
            seed=seed,
        )
    raise ValueError(f"unknown recommender '{name}'")
