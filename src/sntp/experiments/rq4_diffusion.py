"""Requirement 4 -- model adoption and spread, and separate the three events.

A campaign for one promoted item is simulated with two structurally different
models (Independent Cascade and SIR) over the observed influence graph. The
transmission probability of an edge depends on how well the *receiver* is
predicted to like the item, so the recommender feeds the diffusion model while
remaining a separate component.

Three events are counted separately and reported side by side:

* **recommended** -- the platform pushed the item to the user (the seed set);
* **exposed** -- the user received the item from an adopting neighbour;
* **adopted** -- the user actually took the item up.

Every configuration is repeated ``n_simulations`` times with common random
numbers per (model, p0) cell, and the full distribution (not only the mean) is
kept so the report can discuss uncertainty and near-critical behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..diffusion import CampaignSimulator, DiffusionSettings
from ..seeds import preference_selection
from ..utils import write_json
from .common import (
    build_context,
    campaign_affinity,
    eligible_users,
    select_campaign_item,
    snapshot_config,
)

SUMMARY_KEYS = (
    "reach_mean",
    "reach_std",
    "reach_fraction_mean",
    "steps_mean",
    "peak_size_mean",
    "time_to_90pct_mean",
    "mean_adoption_step_mean",
    "attempts_mean",
    "seed_adoptions_mean",
    "mean_exposures_mean",
    "zero_exposure_users_mean",
)


def _simulate(
    graph,
    affinity: np.ndarray,
    seeds: np.ndarray,
    settings: DiffusionSettings,
    n_simulations: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    simulator = CampaignSimulator(graph, affinity, settings)
    summary = simulator.simulate(seeds, n_simulations=n_simulations, seed=seed)
    aggregate = summary.aggregate()
    aggregate["reach_cv"] = (
        aggregate["reach_std"] / aggregate["reach_mean"] if aggregate["reach_mean"] else 0.0
    )
    return aggregate, summary.adoption_curve


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    seed = int(cfg["seed"])
    train_seed = int(cfg["recommendation.training_seeds"][0])
    n_simulations = int(cfg["diffusion.n_simulations"])

    model = ctx.cached_model("bias_mf", train_seed)
    campaign_item = select_campaign_item(ds, model, pool=200, seed=seed)
    affinity = campaign_affinity(model, campaign_item)
    influence = ds.influence_graph()

    eligible = eligible_users(ds, min_interactions=int(cfg["dataset.min_user_ratings"]))
    launch_size = 20
    seeds = preference_selection(affinity, eligible, launch_size)
    print(
        f"[rq4] campaign item {campaign_item} "
        f"(original id {int(ds.item_ids[campaign_item])}), "
        f"{launch_size} preference-ranked seeds, mean affinity {affinity[seeds].mean():.3f}"
    )

    # ------------------------------------------------------- p0 x model grid
    rows, curves = [], []
    for model_name in ("ic", "sir"):
        for p0 in [float(p) for p in cfg["diffusion.p0_grid"]]:
            settings = DiffusionSettings.from_config(
                cfg, model=model_name, p0=p0, track_exposures=True
            )
            aggregate, curve = _simulate(influence, affinity, seeds, settings, n_simulations, seed)
            rows.append({"model": model_name, "p0": p0, **aggregate})
            for step, value in enumerate(curve):
                curves.append(
                    {
                        "model": model_name,
                        "p0": p0,
                        "step": step,
                        "new_adoptions": float(value),
                        "cumulative": float(curve[: step + 1].sum()),
                    }
                )
            print(
                f"[rq4] {model_name.upper():<3} p0={p0:<5} reach={aggregate['reach_mean']:7.1f} "
                f"+/- {aggregate['reach_std']:6.1f} ({aggregate['reach_fraction_mean']:.3f})"
            )

    grid = pd.DataFrame(rows)
    ctx.save_raw(grid, "diffusion_grid")
    ctx.save_table(
        grid[["model", "p0", *SUMMARY_KEYS]],
        "table_11_diffusion_grid",
        "Table 11: reach and timing across transmission probabilities (100 simulations per cell)",
    )

    curve_frame = pd.DataFrame(curves)
    ctx.save_raw(curve_frame, "diffusion_curves")

    # ------------------------------------------- affinity weight sensitivity
    influence_rows = []
    for weight in [float(w) for w in cfg["diffusion.influence_grid"]]:
        settings = DiffusionSettings.from_config(
            cfg, model="ic", influence_weight=weight, track_exposures=True
        )
        aggregate, _ = _simulate(influence, affinity, seeds, settings, n_simulations, seed)
        influence_rows.append({"influence_weight": weight, **aggregate})
    weight_frame = pd.DataFrame(influence_rows)
    ctx.save_raw(weight_frame, "diffusion_influence_weight")
    ctx.save_table(
        weight_frame[["influence_weight", *SUMMARY_KEYS]],
        "table_12_affinity_sensitivity",
        "Table 12: effect of recommender-driven affinity on transmission (IC)",
    )

    # --------------------------------------- event decomposition at default p0
    p0 = float(cfg["diffusion.p0"])
    events = []
    for model_name in ("ic", "sir"):
        settings = DiffusionSettings.from_config(cfg, model=model_name, p0=p0, track_exposures=True)
        aggregate, _ = _simulate(influence, affinity, seeds, settings, n_simulations, seed)
        accepted = aggregate["seed_adoptions_mean"]
        events.append(
            {
                "model": model_name,
                "p0": p0,
                "recommended": float(seeds.size),
                "seed_adoptions": accepted,
                "seed_acceptance_rate": accepted / seeds.size,
                "adoptions_from_influence": aggregate["reach_mean"] - accepted,
                "reach_mean": aggregate["reach_mean"],
                "exposed_users": aggregate["mean_exposures_mean"],
                "share_never_exposed": aggregate["zero_exposure_users_mean"],
                "conversion_per_exposure": (
                    (aggregate["reach_mean"] - accepted) / aggregate["mean_exposures_mean"]
                    if aggregate["mean_exposures_mean"]
                    else 0.0
                ),
                "attempts": aggregate["attempts_mean"],
            }
        )
    event_frame = pd.DataFrame(events)
    ctx.save_table(
        event_frame,
        "table_13_event_decomposition",
        f"Table 13: recommendation vs exposure vs adoption (p0={p0})",
    )

    payload = {
        "campaign_item": int(campaign_item),
        "campaign_item_original_id": int(ds.item_ids[campaign_item]),
        "seeds": [int(s) for s in seeds],
        "launch_size": launch_size,
        "mean_seed_affinity": float(affinity[seeds].mean()),
        "population_affinity_mean": float(affinity.mean()),
        "n_simulations": n_simulations,
        "grid": grid[["model", "p0", *SUMMARY_KEYS, "reach_cv"]].to_dict(orient="records"),
        "affinity_sensitivity": weight_frame[["influence_weight", *SUMMARY_KEYS]].to_dict(
            orient="records"
        ),
        "events": event_frame.to_dict(orient="records"),
        "notes": [
            "Adoption requires an exposure to succeed; a seed is a platform recommendation.",
            "IC allows one transmission attempt per edge; SIR retries while infectious "
            "and recovers with probability recovery_prob.",
            "influence_weight = 0 reproduces the homogeneous classical models.",
            "All simulations use the observed influence graph; no simulated edge is added.",
        ],
    }
    write_json(ctx.summary_dir / "diffusion.json", payload)
    return payload


if __name__ == "__main__":
    run(Config.load())
