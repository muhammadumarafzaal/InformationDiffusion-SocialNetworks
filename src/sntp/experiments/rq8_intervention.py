"""Requirement 8 -- limit the spread, measure the cost, and prove one optimisation.

Interventions
-------------
Two levers act at different places in the pipeline:

``exposure_cap``   the platform refuses to deliver the same item to a user more
                   than *c* times (limits reinforcement, i.e. the SIR advantage);
``affinity_gate``  only users whose predicted affinity clears a threshold can be
                   converted at all (targets the campaign instead of the network).

Both are applied inside the simulator, so their effect is measured on the same
seed set, item and random stream. The report distinguishes *reach reduction*
(the goal) from *collateral damage* (adoptions lost among the high-affinity
users the platform actually wanted to reach).

Cost and scaling
----------------
* recommender fitting cost against the number of users (real sub-samples);
* serving cost against catalogue size (score-matrix throughput);
* cascade cost against graph size (synthetic graphs, same simulator);
* two measured optimisations: the vectorised ragged neighbour gather against a
  per-node Python loop, and the ``reduceat`` group-sum against ``np.add.at``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import Config
from ..data import TRAIN, Dataset
from ..diffusion import CampaignSimulator, DiffusionSettings
from ..recommenders.bias_mf import BiasMF, _group_sum
from ..seeds import preference_selection
from ..utils import current_rss_mb, seeded_rng, write_json
from .common import (
    build_context,
    campaign_affinity,
    eligible_users,
    select_campaign_item,
    snapshot_config,
)


def _campaign_stats(
    simulator: CampaignSimulator,
    seeds: np.ndarray,
    n_simulations: int,
    seed: int,
    affinity: np.ndarray,
) -> dict:
    rng = seeded_rng(seed, stream=811)
    adoption = np.zeros(simulator.n_users, dtype=float)
    reach, exposures = [], []
    started = time.perf_counter()
    for _ in range(n_simulations):
        result = simulator.run(seeds, rng)
        adoption += result.adopted
        reach.append(result.reach)
        exposures.append(float(result.exposures.mean()))
    seconds = time.perf_counter() - started
    adoption /= n_simulations
    high = affinity >= 0.6
    return {
        "reach_mean": float(np.mean(reach)),
        "reach_std": float(np.std(reach, ddof=1)),
        "exposures_mean": float(np.mean(exposures)),
        "reach_high_affinity": float(adoption[high].sum()),
        "high_affinity_users": int(high.sum()),
        "reach_low_affinity": float(adoption[~high].sum()),
        "seconds_per_simulation": seconds / max(n_simulations, 1),
    }


def _subsample(ds: Dataset, n_users: int, seed: int) -> Dataset:
    """Keep all items but only ``n_users`` randomly chosen active users."""
    counts = np.bincount(ds.rating_user[ds.fold == TRAIN], minlength=ds.n_users)
    pool = np.flatnonzero(counts >= int(ds.metadata["filters"]["min_user_ratings"]))
    rng = seeded_rng(seed, stream=701)
    users = np.sort(rng.choice(pool, size=min(n_users, pool.size), replace=False))
    mapping = -np.ones(ds.n_users, dtype=np.int64)
    mapping[users] = np.arange(users.size)
    keep_rating = mapping[ds.rating_user] >= 0
    keep_trust = (mapping[ds.trust_src] >= 0) & (mapping[ds.trust_dst] >= 0)
    return Dataset(
        name=f"{ds.name}_sub{users.size}",
        user_ids=ds.user_ids[users],
        item_ids=ds.item_ids,
        rating_user=mapping[ds.rating_user[keep_rating]],
        rating_item=ds.rating_item[keep_rating],
        ratings=ds.ratings[keep_rating],
        fold=ds.fold[keep_rating],
        trust_src=mapping[ds.trust_src[keep_trust]],
        trust_dst=mapping[ds.trust_dst[keep_trust]],
        trust_weight=ds.trust_weight[keep_trust],
        metadata={},
    )


def _synthetic_graph(n_users: int, mean_degree: int, seed: int) -> sparse.csr_matrix:
    rng = seeded_rng(seed, stream=733)
    degree = np.clip(rng.poisson(mean_degree, size=n_users), 1, None)
    sources = np.repeat(np.arange(n_users), degree)
    targets = rng.integers(0, n_users, size=sources.size)
    keep = sources != targets
    return sparse.csr_matrix(
        (np.ones(keep.sum(), dtype=np.float32), (sources[keep], targets[keep])),
        shape=(n_users, n_users),
    )


def _naive_gather(indptr: np.ndarray, indices: np.ndarray, sources: np.ndarray) -> np.ndarray:
    chunks = [indices[indptr[int(s)] : indptr[int(s) + 1]] for s in sources]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    seed = int(cfg["seed"])
    train_seed = int(cfg["recommendation.training_seeds"][0])
    model = ctx.cached_model("bias_mf", train_seed)

    campaign_item = select_campaign_item(ds, model, pool=200, seed=seed)
    affinity = campaign_affinity(model, campaign_item)
    graph = ds.influence_graph()
    eligible = eligible_users(ds, min_interactions=int(cfg["dataset.min_user_ratings"]))
    budget = int(cfg["intervention.budget"])

    try:
        seeds = np.array(ctx.load_json("influence")["greedy_seeds_budget20"], dtype=np.int64)
        seed_source = "greedy_mc (selected in Requirement 5)"
    except (FileNotFoundError, KeyError):
        seeds = preference_selection(affinity, eligible, budget)
        seed_source = "preference fallback (run `sntp run influence` for greedy seeds)"
    n_simulations = int(cfg["intervention.n_simulations"])

    # --------------------------------------------------------- interventions
    rows = []
    baseline = None
    for cap in [int(c) for c in cfg["intervention.exposure_caps"]]:
        for gate in [float(g) for g in cfg["intervention.affinity_gates"]]:
            settings = DiffusionSettings.from_config(
                cfg, model="ic", exposure_cap=cap, affinity_gate=gate
            )
            simulator = CampaignSimulator(graph, affinity, settings)
            stats = _campaign_stats(simulator, seeds, n_simulations, seed, affinity)
            if baseline is None:
                baseline = stats
            row = {"exposure_cap": cap, "affinity_gate": gate, **stats}
            row["reach_reduction"] = 1.0 - stats["reach_mean"] / baseline["reach_mean"]
            row["high_affinity_loss"] = (
                1.0 - stats["reach_high_affinity"] / baseline["reach_high_affinity"]
                if baseline["reach_high_affinity"]
                else 0.0
            )
            rows.append(row)
            print(
                f"[rq8] cap={cap:<5} gate={gate:<5} reach={stats['reach_mean']:7.1f} "
                f"(-{row['reach_reduction'] * 100:5.1f}%) high-affinity loss="
                f"{row['high_affinity_loss'] * 100:5.1f}%"
            )
    intervention = pd.DataFrame(rows)
    ctx.save_raw(intervention, "intervention")
    ctx.save_table(
        intervention[
            [
                "exposure_cap",
                "affinity_gate",
                "reach_mean",
                "reach_std",
                "reach_reduction",
                "high_affinity_loss",
                "reach_high_affinity",
                "exposures_mean",
            ]
        ],
        "table_26_intervention",
        f"Table 26: limiting diffusion (campaign item {campaign_item}, budget {budget})",
    )

    # ------------------------------------------------------- scaling in users
    user_rows = []
    for n_users in [int(n) for n in cfg["cost.user_sizes"]]:
        small = _subsample(ds, n_users, seed)
        rss0 = current_rss_mb()
        started = time.perf_counter()
        fitted = BiasMF(
            n_factors=int(cfg["recommendation.n_factors"]),
            learning_rate=float(cfg["recommendation.learning_rate"]),
            reg=float(cfg["recommendation.reg"]),
            epochs=int(cfg["recommendation.epochs"]),
            batch_size=int(cfg["recommendation.batch_size"]),
            init_std=float(cfg["recommendation.init_std"]),
            seed=train_seed,
        ).fit(small)
        fit_seconds = time.perf_counter() - started
        probe = np.arange(min(512, small.n_users))
        started = time.perf_counter()
        for _ in range(5):
            fitted.predict_matrix(probe)
        serve_ms = (time.perf_counter() - started) / 5 * 1000
        user_rows.append(
            {
                "users": small.n_users,
                "ratings": small.n_ratings,
                "trust_edges": int(small.trust_src.size),
                "fit_seconds": round(fit_seconds, 2),
                "best_epoch": int(fitted.best_epoch),
                "rss_delta_mb": round(current_rss_mb() - rss0, 1),
                "serve_ms_per_512_users": round(serve_ms, 2),
                "ratings_per_second": round(small.n_ratings * int(fitted.best_epoch) / fit_seconds),
            }
        )
        print(
            f"[rq8] users={small.n_users:<6} ratings={small.n_ratings:<7} "
            f"fit={fit_seconds:5.1f}s serve={serve_ms:6.1f}ms/512"
        )
    user_scaling = pd.DataFrame(user_rows)
    ctx.save_table(
        user_scaling,
        "table_27_scaling_users",
        "Table 27: recommender cost against the number of users (real sub-samples)",
    )

    # ---------------------------------------------------- scaling in catalogue
    item_rows = []
    probes = np.arange(512)
    for n_items in [int(n) for n in cfg["cost.item_sizes"]]:
        n_items = min(n_items, ds.n_items)
        started = time.perf_counter()
        for _ in range(5):
            _ = (
                model.mu
                + model.bu[probes][:, None]
                + model.bi[:n_items][None, :]
                + model.P[probes] @ model.Q[:n_items].T
            )
        serve_ms = (time.perf_counter() - started) / 5 * 1000
        item_rows.append(
            {
                "items": n_items,
                "users_per_batch": int(probes.size),
                "score_ms": round(serve_ms, 2),
                "scores_per_second": round(probes.size * n_items / (serve_ms / 1000)),
                "dense_matrix_mb": round(probes.size * n_items * 4 / (1024**2), 2),
                "intrinsic_dim": int(model.Q.shape[1]),
            }
        )
    item_scaling = pd.DataFrame(item_rows)
    ctx.save_table(
        item_scaling,
        "table_28_scaling_catalogue",
        "Table 28: serving cost against catalogue size (one batch of 512 users)",
    )

    # ------------------------------------------------------ scaling in graph size
    graph_rows = []
    for n_users in [int(n) for n in cfg["cost.synthetic_users"]]:
        synthetic = _synthetic_graph(n_users, int(cfg["cost.synthetic_mean_degree"]), seed)
        settings = DiffusionSettings.from_config(cfg, model="ic")
        simulator = CampaignSimulator(synthetic, affinity[:1].repeat(n_users), settings)
        launch = seeded_rng(seed, stream=99).choice(
            n_users, size=max(10, n_users // 50), replace=False
        )
        simulator.run(launch, seeded_rng(seed, stream=1))
        started = time.perf_counter()
        repeats = 5
        rng = seeded_rng(seed, stream=17)
        for _ in range(repeats):
            simulator.run(launch, rng)
        per_sim = (time.perf_counter() - started) / repeats
        graph_rows.append(
            {
                "users": n_users,
                "edges": int(synthetic.nnz),
                "mean_degree": float(np.diff(synthetic.indptr).mean()),
                "seeds": int(launch.size),
                "ms_per_simulation": round(per_sim * 1000, 2),
                "ms_per_million_edges": round(per_sim * 1000 / max(synthetic.nnz / 1e6, 1e-9), 2),
            }
        )
        print(
            f"[rq8] synthetic users={n_users:<6} edges={synthetic.nnz:<7} "
            f"{per_sim * 1000:.1f} ms/sim"
        )
    graph_scaling = pd.DataFrame(graph_rows)
    ctx.save_table(
        graph_scaling,
        "table_29_scaling_diffusion",
        "Table 29: cascade cost against graph size (synthetic graphs, mean degree ~20)",
    )

    # ------------------------------------------------------ measured optimisations
    rng = seeded_rng(seed, stream=4242)
    sources = rng.choice(ds.n_users, size=2000, replace=False)
    indptr = graph.indptr.astype(np.int64)
    indices = graph.indices.astype(np.int64)
    starts = indptr[sources]
    counts = indptr[sources + 1] - starts
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    positions = np.arange(int(counts.sum()), dtype=np.int64) + offsets

    started = time.perf_counter()
    for _ in range(20):
        _ = indices[positions]
    vector_seconds = (time.perf_counter() - started) / 20

    started = time.perf_counter()
    for _ in range(20):
        _ = _naive_gather(indptr, indices, sources)
    naive_seconds = (time.perf_counter() - started) / 20

    batch = rng.integers(0, 4000, size=4096)
    values = rng.normal(size=(4096, 32))
    accumulated = np.zeros((4000, 32))
    started = time.perf_counter()
    for _ in range(10):
        np.add.at(accumulated, batch, values)
    add_at_seconds = (time.perf_counter() - started) / 10
    started = time.perf_counter()
    for _ in range(10):
        _group_sum(batch, values, 4000)
    reduceat_seconds = (time.perf_counter() - started) / 10

    optimisations = pd.DataFrame(
        [
            {
                "operation": "gather 2000 neighbour lists (IC step)",
                "baseline": "per-node Python loop",
                "baseline_ms": round(naive_seconds * 1000, 2),
                "optimised": "vectorised ragged gather",
                "optimised_ms": round(vector_seconds * 1000, 2),
                "speedup": round(naive_seconds / max(vector_seconds, 1e-9), 1),
            },
            {
                "operation": "accumulate 4096x32 gradients into 4000 rows",
                "baseline": "np.add.at",
                "baseline_ms": round(add_at_seconds * 1000, 2),
                "optimised": "sort + np.add.reduceat",
                "optimised_ms": round(reduceat_seconds * 1000, 2),
                "speedup": round(add_at_seconds / max(reduceat_seconds, 1e-9), 1),
            },
        ]
    )
    ctx.save_table(
        optimisations,
        "table_30_optimisations",
        "Table 30: measured effect of the two hot-path optimisations",
    )

    payload = {
        "campaign_item": int(campaign_item),
        "seeds": [int(s) for s in seeds],
        "seed_source": seed_source,
        "baseline_reach": float(baseline["reach_mean"]),
        "interventions": intervention.to_dict(orient="records"),
        "most_effective": intervention.sort_values("reach_reduction", ascending=False)
        .head(3)[["exposure_cap", "affinity_gate", "reach_reduction", "high_affinity_loss"]]
        .to_dict(orient="records"),
        "user_scaling": user_scaling.to_dict(orient="records"),
        "item_scaling": item_scaling.to_dict(orient="records"),
        "graph_scaling": graph_scaling.to_dict(orient="records"),
        "optimisations": optimisations.to_dict(orient="records"),
        "hardware": {
            "rss_mb": round(current_rss_mb(), 1),
            "graph_edges": int(graph.nnz),
        },
    }
    write_json(ctx.summary_dir / "intervention.json", payload)
    return payload


if __name__ == "__main__":
    run(Config.load())
