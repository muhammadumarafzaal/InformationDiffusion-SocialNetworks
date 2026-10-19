"""Requirement 5 -- budgeted seed selection against random and network baselines.

The promotion budget is a number of users the platform can contact directly.
Five strategies compete on identical candidate pools and identical simulation
budgets:

``greedy_mc``   Monte-Carlo greedy on *marginal* reach with CELF lazy evaluation;
``proxy``       the cheap two-hop heuristic the greedy search is prefiltered with;
``preference``  the users who like the item most (individual popularity);
``degree``      the highest out-degree users (a purely structural baseline);
``random``      uniform draws, repeated to expose selection variance.

Because the greedy construction is nested (the budget-k set is the prefix of the
budget-(k+1) set), one run up to the largest budget yields every smaller budget
and the marginal-gain curve at the same time. Final seed sets are always
re-evaluated with a much larger simulation budget than the one used during
selection, so reported numbers are not measured on the selection noise.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import Config
from ..diffusion import CampaignSimulator, DiffusionSettings
from ..seeds import (
    SeedSelector,
    audience_overlap,
    degree_selection,
    expected_influence_proxy,
    preference_selection,
    prefilter_candidates,
    random_selection,
)
from ..utils import seeded_rng, spearman, write_json
from .common import (
    build_context,
    campaign_affinity,
    eligible_users,
    select_campaign_item,
    snapshot_config,
)

STRATEGIES = ("greedy_mc", "proxy", "preference", "degree", "random")


def _evaluate(
    simulator: CampaignSimulator, seeds: np.ndarray, n_mc: int, seed: int, stream: int
) -> tuple[float, float, float]:
    """Mean, standard deviation and best of ``n_mc`` independent cascades."""
    rng = seeded_rng(seed, stream=stream)
    values = np.array([simulator.run(seeds, rng).reach for _ in range(n_mc)], dtype=float)
    return float(values.mean()), float(values.std(ddof=1)), float(values.max())


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    seed = int(cfg["seed"])
    train_seed = int(cfg["recommendation.training_seeds"][0])
    settings_influence = cfg.section("influence")

    model = ctx.cached_model("bias_mf", train_seed)
    campaign_item = select_campaign_item(ds, model, pool=200, seed=seed)
    affinity = campaign_affinity(model, campaign_item)
    graph = ds.influence_graph()

    eligible = eligible_users(ds, min_interactions=int(cfg["dataset.min_user_ratings"]))
    proxy = expected_influence_proxy(graph, affinity)
    pool = prefilter_candidates(proxy, eligible, int(settings_influence["candidate_pool"]), seed)
    budgets = [int(b) for b in settings_influence["budgets"]]
    max_budget = max(budgets)
    n_mc_selection = int(settings_influence["mc_samples_selection"])
    n_mc_final = int(settings_influence["mc_samples_final"])
    n_random_draws = 10

    settings = DiffusionSettings.from_config(cfg, model="ic")
    simulator = CampaignSimulator(graph, affinity, settings)
    selector = SeedSelector(simulator, seed=seed)

    print(
        f"[rq5] item={campaign_item} eligible={eligible.size} pool={pool.size} "
        f"budgets={budgets} selection_mc={n_mc_selection}"
    )

    # ------------------------------------------------------- greedy (CELF)
    started = time.perf_counter()
    greedy = selector.greedy(
        pool, max_budget, n_mc=n_mc_selection, celf=bool(settings_influence["celf_enabled"])
    )
    greedy_seconds = time.perf_counter() - started
    print(
        f"[rq5] greedy done: {greedy.simulations} simulations in {greedy_seconds:.1f}s "
        f"({greedy_seconds / max(greedy.simulations, 1) * 1000:.1f} ms/sim)"
    )

    marginal = pd.DataFrame(
        {
            "step": np.arange(1, greedy.seeds.size + 1),
            "user": greedy.seeds,
            "marginal_gain": greedy.gains,
            "marginal_std": greedy.gain_std,
            "cumulative_gain": np.cumsum(greedy.gains),
        }
    )
    ctx.save_raw(marginal, "marginal_gains")

    # ------------------------------------------------------------ strategies
    deterministic = {
        "greedy_mc": greedy.seeds,
        "proxy": preference_selection(proxy, pool, max_budget),
        "preference": preference_selection(affinity, eligible, max_budget),
        "degree": degree_selection(graph, eligible, max_budget),
    }

    rows = []
    for budget in budgets:
        for name, seeds in deterministic.items():
            mean, std, best = _evaluate(
                simulator, seeds[:budget], n_mc_final, seed, stream=5100 + 31 * budget
            )
            rows.append(
                {
                    "strategy": name,
                    "budget": budget,
                    "reach_mean": mean,
                    "reach_std": std,
                    "reach_best": best,
                    "draws": 1,
                    "simulations_per_draw": n_mc_final,
                }
            )
        draws = [
            random_selection(eligible, budget, seed=seed + 97 * d + budget, repetitions=1)
            for d in range(n_random_draws)
        ]
        values = np.array(
            [
                _evaluate(simulator, draw, n_mc_final // 2, seed, stream=7300 + 13 * d)[0]
                for d, draw in enumerate(draws)
            ]
        )
        rows.append(
            {
                "strategy": "random",
                "budget": budget,
                "reach_mean": float(values.mean()),
                "reach_std": float(values.std(ddof=1)),
                "reach_best": float(values.max()),
                "draws": n_random_draws,
                "simulations_per_draw": n_mc_final // 2,
            }
        )
        print(
            f"[rq5] budget={budget:<3} "
            + " ".join(
                f"{r['strategy']}={r['reach_mean']:.0f}" for r in rows if r["budget"] == budget
            )
        )

    runs = pd.DataFrame(rows)
    reference = runs[runs.strategy == "random"].set_index("budget")["reach_mean"]
    runs["lift_vs_random"] = runs.apply(
        lambda r: (
            (r["reach_mean"] / reference.loc[r["budget"]] - 1.0)
            if reference.loc[r["budget"]]
            else 0.0
        ),
        axis=1,
    )
    ctx.save_raw(runs, "influence_runs")
    ctx.save_table(
        runs[["strategy", "budget", "reach_mean", "reach_std", "reach_best", "lift_vs_random"]],
        "table_14_seed_performance",
        "Table 14: realised reach by strategy and budget (Independent Cascade)",
    )

    marginal_table = marginal.copy()
    ctx.save_table(
        marginal_table[["step", "marginal_gain", "marginal_std", "cumulative_gain"]],
        "table_15_marginal_gains",
        "Table 15: marginal reach of each additional seed chosen by the greedy search",
    )

    # ---------------------------------------------------------------- overlap
    overlap_rows = []
    for name, seeds in deterministic.items():
        stats = audience_overlap(seeds[: int(cfg["influence.budgets"][2])], graph, seed=seed)
        overlap_rows.append({"strategy": name, **stats})
    overlap_rows.append({"strategy": "random", **audience_overlap(draws[-1], graph, seed=seed)})
    overlap = pd.DataFrame(overlap_rows)
    ctx.save_table(
        overlap,
        "table_16_audience_overlap",
        "Table 16: one-hop audience overlap of the seed sets at the median budget",
    )

    # ------------------------------------------- measured optimisation: CELF
    small_pool = pool[:30]
    small_budget = 3
    naive_selector = SeedSelector(simulator, seed=seed)
    started = time.perf_counter()
    naive = naive_selector.greedy(small_pool, small_budget, n_mc=8, celf=False)
    naive_seconds = time.perf_counter() - started
    celf_selector = SeedSelector(simulator, seed=seed)
    started = time.perf_counter()
    lazy = celf_selector.greedy(small_pool, small_budget, n_mc=8, celf=True)
    celf_seconds = time.perf_counter() - started
    overlap_between = len(set(naive.seeds.tolist()) & set(lazy.seeds.tolist()))
    optimisation = pd.DataFrame(
        [
            {
                "method": "naive greedy",
                "candidates": small_pool.size,
                "budget": small_budget,
                "simulations": naive.simulations,
                "seconds": round(naive_seconds, 2),
                "selected_overlap_with_celf": overlap_between,
            },
            {
                "method": "CELF lazy greedy",
                "candidates": small_pool.size,
                "budget": small_budget,
                "simulations": lazy.simulations,
                "seconds": round(celf_seconds, 2),
                "selected_overlap_with_celf": overlap_between,
            },
        ]
    )
    optimisation["speedup"] = naive_seconds / max(celf_seconds, 1e-9)
    ctx.save_table(
        optimisation,
        "table_17_selection_cost",
        "Table 17: cost of exact marginal-gain search with and without lazy evaluation",
    )

    # -------------------------------------------------- heuristic vs. true gain
    rng = np.random.default_rng(seed)
    sample = rng.choice(pool, size=min(40, pool.size), replace=False)
    measured = np.array(
        [
            _evaluate(simulator, np.array([int(c)]), 10, seed, 9100 + i)[0]
            for i, c in enumerate(sample)
        ]
    )
    agreement = spearman(proxy[sample], measured)
    tie_in_degree = spearman(np.diff(graph.indptr)[sample], measured)

    # ----------------------------------------------------------------- digest
    best_row = runs.loc[runs.groupby("budget")["reach_mean"].idxmax()]
    greedy_at_default = runs[
        (runs.strategy == "greedy_mc") & (runs.budget == int(cfg["intervention.budget"]))
    ]
    payload = {
        "campaign_item": int(campaign_item),
        "eligible_users": int(eligible.size),
        "candidate_pool": int(pool.size),
        "budgets": budgets,
        "selection_simulations": int(greedy.simulations),
        "selection_seconds": greedy_seconds,
        "greedy_mean_marginal_gain": float(greedy.gains.mean()),
        "greedy_last_marginal_gain": float(greedy.gains[-1]),
        "runs": runs.to_dict(orient="records"),
        "best_strategy_per_budget": best_row[["budget", "strategy", "reach_mean"]].to_dict(
            orient="records"
        ),
        "marginal_gains": marginal[["step", "marginal_gain", "marginal_std"]].to_dict(
            orient="records"
        ),
        "audience_overlap": overlap.to_dict(orient="records"),
        "selection_cost": optimisation.to_dict(orient="records"),
        "proxy_spearman_with_single_seed_reach": agreement,
        "degree_spearman_with_single_seed_reach": tie_in_degree,
        "greedy_seeds_budget20": (
            [int(s) for s in greedy.seeds[: int(cfg["intervention.budget"])]]
            if greedy_at_default.shape[0]
            else [int(s) for s in greedy.seeds[:20]]
        ),
        "notes": [
            "Every strategy is scored with the same simulator, item and affinity vector.",
            "Greedy budgets are nested prefixes of one run up to the largest budget.",
            "Random is repeated; its spread is selection variance, not simulation noise.",
            "Final evaluation uses more simulations than selection to avoid reporting "
            "selection noise.",
        ],
    }
    write_json(ctx.summary_dir / "influence.json", payload)
    print(
        f"[rq5] proxy vs measured reach rho={agreement:.3f}; "
        f"CELF speedup={naive_seconds / max(celf_seconds, 1e-9):.2f}x"
    )
    return payload


if __name__ == "__main__":
    run(Config.load())
