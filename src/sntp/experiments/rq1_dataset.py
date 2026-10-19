"""Requirement 1 -- justify and quantify the social environment representation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data import TRAIN
from ..graph import graph_statistics, rewired_control, user_similarity_graph
from ..recommenders.baselines import BiasBaseline
from ..utils import current_rss_mb, hardware_summary
from .common import build_context, snapshot_config


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    metadata = ds.metadata
    raw, core, split = metadata["raw"], metadata["core"], metadata["split"]

    rows = [
        ("rating interactions", raw["n_ratings"], core["n_ratings"]),
        ("users", raw["n_users"], core["n_users"]),
        ("items", raw["n_items"], core["n_items"]),
        ("density", raw["density"], core["density"]),
        ("mean rating", raw["rating_mean"], core["rating_mean"]),
        (
            "median ratings per user",
            raw["ratings_per_user_median"],
            float(np.median(np.bincount(ds.rating_user[ds.fold == TRAIN], minlength=ds.n_users))),
        ),
        (
            "median ratings per item",
            raw["ratings_per_item_median"],
            float(np.median(np.bincount(ds.rating_item[ds.fold == TRAIN], minlength=ds.n_items))),
        ),
    ]
    summary = pd.DataFrame(rows, columns=["quantity", "raw_epinions", "analysed_core"])
    ctx.save_table(summary, "table_1_dataset_summary", "Table 1: raw Epinions versus analysed core")

    influence = ds.influence_graph()
    trust = ds.trust_graph()
    stats_trust = graph_statistics(trust)

    bias = BiasBaseline(reg=10.0).fit(ds)
    residual = bias.residual_matrix(ds)
    stats_similarity = graph_statistics(
        user_similarity_graph(residual, n_neighbors=10, exclude=trust, min_similarity=0.05)
    )
    stats_random = graph_statistics(rewired_control(influence, seed=int(cfg["seed"])))
    stats_influence = graph_statistics(influence)

    graph_rows = []
    for label, stats in [
        ("trust (stored direction)", stats_trust),
        ("influence (reversed)", stats_influence),
        ("similarity (taste kNN)", stats_similarity),
        ("rewired control", stats_random),
    ]:
        graph_rows.append({"graph": label, **stats})
    graph_frame = pd.DataFrame(graph_rows)
    ctx.save_table(graph_frame, "table_2_graph_statistics", "Table 2: network variants")

    split_frame = pd.DataFrame(
        [
            {
                "fold": "train",
                "interactions": split["n_train"],
                "share": split["n_train"] / core["n_ratings"],
            },
            {
                "fold": "validation",
                "interactions": split["n_valid"],
                "share": split["n_valid"] / core["n_ratings"],
            },
            {
                "fold": "test",
                "interactions": split["n_test"],
                "share": split["n_test"] / core["n_ratings"],
            },
        ]
    )
    ctx.save_table(split_frame, "table_3_split", "Table 3: interaction folds")

    rating_counts = {float(k): int(v) for k, v in core["rating_counts"].items()}
    payload = {
        "dataset": metadata,
        "graph_statistics": graph_frame.to_dict(orient="records"),
        "rating_histogram": rating_counts,
        "hardware": hardware_summary(),
        "rss_mb": round(current_rss_mb(), 1),
        "assumptions": [
            "One interaction row = one review; duplicate (user, item) rows are removed.",
            "Ratings are explicit preference signals on a 1-5 scale; "
            "no implicit feedback is available.",
            "A trust record 'source target' means source trusts target; "
            "influence therefore flows target -> source.",
            "All trust weights in this release are +1, so distrust edges are absent "
            "and cannot be modelled.",
            "Epinions has no timestamps, so folds are random per user rather than temporal.",
            "Users and items with fewer than 10 interactions are pruned iteratively "
            "to obtain an evaluable core.",
            "Exposure and adoption events are simulated, never observed; "
            "they are reported separately.",
        ],
    }
    ctx.save_json(payload, "dataset")
    print(
        f"[rq1] core: {core['n_users']} users, {core['n_items']} items, "
        f"{core['n_ratings']} ratings, {core['trust_edges']} trust edges"
    )
    return payload


if __name__ == "__main__":
    run(Config.load())
