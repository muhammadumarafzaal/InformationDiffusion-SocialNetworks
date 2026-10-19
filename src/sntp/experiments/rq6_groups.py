"""Requirement 6 -- recommend to a group and compare the outcome with the spread.

Groups of five are built in two flavours: ``friends`` (an anchor plus trust
neighbours) and ``conflict`` (an anchor plus the most dissimilar users found in a
random pool, measured on bias-corrected profiles). Each group gets a shared
candidate slate -- the union of the members' individual top-N predictions plus the
globally most popular items -- with every item any member has already rated in
the training fold removed, so nothing recommended is already consumed.

Three social-choice rules pick one item from that slate:

``average``        maximise the mean predicted satisfaction (utilitarian);
``least_misery``   maximise the worst-off member's satisfaction (egalitarian);
``most_pleasure``  maximise the best-off member's satisfaction (enthusiast rule).

The predicted satisfaction decides the choice; the realised diffusion of the
chosen item (with the group as seeds) is simulated separately afterwards. That
separation is what lets the report answer whether the fairest choice is also the
one that spreads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data import TRAIN
from ..diffusion import CampaignSimulator, DiffusionSettings
from ..groups import STRATEGIES, aggregate_item, build_groups, consensus_rank, group_satisfaction
from ..utils import spearman, write_json
from .common import build_context, campaign_affinity, snapshot_config


def _slate(
    scores: np.ndarray,
    seen: set[int],
    n_per_member: int,
    global_items: np.ndarray,
) -> np.ndarray:
    """Union of member top-N plus global hits, minus anything already consumed."""
    candidates: set[int] = set()
    for row in scores:
        candidates.update(np.argsort(-row, kind="stable")[:n_per_member].tolist())
    candidates.update(int(i) for i in global_items)
    ordered = np.array(sorted(candidates - seen), dtype=np.int64)
    if ordered.size:
        return ordered
    # degenerate fallback: best-scoring unseen items of the whole catalogue
    fallback = np.argsort(-scores.max(axis=0), kind="stable")
    return np.array([i for i in fallback if int(i) not in seen][:n_per_member], dtype=np.int64)


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    settings = cfg.section("groups")
    seed = int(cfg["seed"])
    train_seed = int(cfg["recommendation.training_seeds"][0])
    size = int(settings["size"])
    n_groups = int(settings["n_groups"])
    n_per_member = int(settings["candidate_items"])
    n_simulations = int(settings["n_simulations"])

    model = ctx.cached_model("bias_mf", train_seed)
    counts = np.bincount(ds.rating_item[ds.fold == TRAIN], minlength=ds.n_items)
    global_items = np.argsort(-counts, kind="stable")[: int(settings["global_popular_items"])]
    diffusion = DiffusionSettings.from_config(cfg, model="ic")
    observed = ds.observed_pairs().tocsr()
    influence = ds.influence_graph()

    simulators: dict[int, CampaignSimulator] = {}

    def simulator_for(item: int) -> CampaignSimulator:
        if item not in simulators:
            simulators[item] = CampaignSimulator(
                influence, campaign_affinity(model, item), diffusion
            )
        return simulators[item]

    rows: list[dict] = []
    for kind in ("friends", "conflict"):
        groups = build_groups(ds, size=size, n_groups=n_groups, kind=kind, seed=seed, pool_size=400)
        for group in groups:
            scores = model.predict_matrix(group.members)
            seen: set[int] = set()
            for member in group.members:
                start, end = observed.indptr[member], observed.indptr[member + 1]
                seen.update(int(i) for i in observed.indices[start:end])
            slate = _slate(scores, seen, n_per_member, global_items)
            if slate.size == 0:
                continue
            matrix = np.clip(scores[:, slate], 1.0, 5.0)

            consensus_item = int(slate[consensus_rank(matrix)])
            for strategy in STRATEGIES:
                position = aggregate_item(matrix, strategy)
                item = int(slate[position])
                satisfaction = group_satisfaction(matrix, position)
                summary = simulator_for(item).simulate(
                    group.members, n_simulations=n_simulations, seed=seed + group.index
                )
                rows.append(
                    {
                        "group_kind": kind,
                        "group": group.index,
                        "strategy": strategy,
                        "item": item,
                        "slate_size": int(slate.size),
                        "is_consensus_item": bool(item == consensus_item),
                        **satisfaction,
                        "reach_mean": summary.reach_fraction_mean * ds.n_users,
                        "reach_fraction": summary.reach_fraction_mean,
                        "reach_std": summary.reach_fraction_std * ds.n_users,
                    }
                )
            rows.append(
                {
                    "group_kind": kind,
                    "group": group.index,
                    "strategy": "consensus",
                    "item": consensus_item,
                    "slate_size": int(slate.size),
                    "is_consensus_item": True,
                    **group_satisfaction(matrix, int(np.flatnonzero(slate == consensus_item)[0])),
                    "reach_mean": np.nan,
                    "reach_fraction": np.nan,
                    "reach_std": np.nan,
                }
            )
        print(f"[rq6] {kind}: {len(groups)} groups built")

    raw = pd.DataFrame(rows)
    ctx.save_raw(raw, "groups")

    simulated = raw.dropna(subset=["reach_mean"])
    summary = (
        simulated.groupby(["group_kind", "strategy"], sort=False)
        .agg(
            groups=("group", "size"),
            mean_satisfaction=("mean_satisfaction", "mean"),
            min_satisfaction=("min_satisfaction", "mean"),
            worst_member=("min_satisfaction", "min"),
            std_satisfaction=("std_satisfaction", "mean"),
            reach=("reach_mean", "mean"),
            reach_std=("reach_mean", "std"),
            distinct_items=("item", "nunique"),
            consensus_hits=("is_consensus_item", "mean"),
        )
        .reset_index()
    )
    ctx.save_table(
        summary,
        "table_18_group_strategies",
        f"Table 18: group recommendation by aggregation rule ({size}-person groups, IC diffusion)",
    )

    # ---------------------------------------------------- agreement and trade-off
    agreement_rows = []
    for kind in ("friends", "conflict"):
        wide = simulated[simulated.group_kind == kind].pivot(
            index="group", columns="strategy", values="item"
        )
        for left, right in (
            ("average", "least_misery"),
            ("average", "most_pleasure"),
            ("least_misery", "most_pleasure"),
        ):
            agreement_rows.append(
                {
                    "group_kind": kind,
                    "pair": f"{left} vs {right}",
                    "share_same_item": float((wide[left] == wide[right]).mean()),
                    "groups": int(wide.shape[0]),
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    ctx.save_table(
        agreement,
        "table_19_group_agreement",
        "Table 19: how often two aggregation rules choose the same item",
    )

    tradeoff = []
    for kind in ("friends", "conflict"):
        frame = simulated[simulated.group_kind == kind]
        rho = spearman(frame["mean_satisfaction"].to_numpy(), frame["reach_mean"].to_numpy())
        within = []
        for _, block in frame.groupby("group"):
            if block["item"].nunique() >= 2:
                value = spearman(
                    block["mean_satisfaction"].to_numpy(), block["reach_mean"].to_numpy()
                )
                if np.isfinite(value):
                    within.append(value)
        tradeoff.append(
            {
                "group_kind": kind,
                "spearman_satisfaction_vs_reach": rho,
                "mean_within_group_spearman": float(np.mean(within)) if within else float("nan"),
                "groups_used_within": len(within),
            }
        )
    tradeoff_frame = pd.DataFrame(tradeoff)
    ctx.save_table(
        tradeoff_frame,
        "table_20_satisfaction_vs_spread",
        "Table 20: does the most satisfying item also spread the furthest?",
    )

    payload = {
        "groups": {"size": size, "n_groups": n_groups, "kinds": ["friends", "conflict"]},
        "strategy_summary": summary.to_dict(orient="records"),
        "agreement": agreement.to_dict(orient="records"),
        "satisfaction_vs_reach": tradeoff_frame.to_dict(orient="records"),
        "n_simulations": n_simulations,
        "notes": [
            "Candidate slates exclude every item observed in the training fold for any member.",
            "Satisfaction is the predicted rating clipped to [1, 5].",
            "Realised reach is simulated with the group as the seed set, "
            "separately from the choice.",
            "The consensus reference minimises the mean deviation from each member's "
            "own best item.",
        ],
    }
    write_json(ctx.summary_dir / "groups.json", payload)
    least_misery = agreement[agreement.pair.str.startswith("average vs least")][
        "share_same_item"
    ].mean()
    print(f"[rq6] average vs least-misery agreement: {least_misery:.3f}")
    return payload


if __name__ == "__main__":
    run(Config.load())
