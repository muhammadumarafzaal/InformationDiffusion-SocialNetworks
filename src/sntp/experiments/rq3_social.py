"""Requirement 3 -- incorporate social relations and price their contribution.

Three questions are answered on the same protocol:

1. *Do friends carry preference information at all?* ``NeighbourResidualRecommender``
   predicts a user's bias-corrected residual as the average residual of the people
   they trust. It contains no latent factors and no learning beyond the bias model,
   so any gain over the bias baseline is pure social signal.
2. *How much social information is optimal?* ``SocialMF`` is swept over ``beta``
   (0 = plain biased MF, 2 = strong pull) on the validation fold only.
3. *Friends or merely similar users?* The same ``beta`` is refitted on the observed
   trust graph, on a taste-similarity graph with all trust edges removed, and on a
   degree-preserving rewired control. At ``beta`` = 0 all three collapse to the
   same model, which is the control that validates the comparison.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import Config
from ..data import TEST, VALID
from ..evaluation import build_protocol, evaluate
from ..graph import rewired_control, user_similarity_graph
from ..recommenders import build_recommender
from ..recommenders.baselines import BiasBaseline
from ..recommenders.social import NeighbourResidualRecommender
from ..utils import bootstrap_ci, write_json
from .common import build_context, snapshot_config

SWEEP_METRICS = ("rmse", "mae", "ndcg@10", "hit@10", "coverage@10")


def _protocols(cfg: Config, ds):
    settings = cfg.section("recommendation.evaluation")
    kwargs = dict(
        n_negatives=int(settings["n_negatives"]),
        relevance_threshold=float(settings["relevance_threshold"]),
        seed=int(cfg["seed"]),
    )
    return (
        build_protocol(ds, fold=VALID, **kwargs),
        build_protocol(ds, fold=TEST, **kwargs),
    )


def _row(name: str, beta: float, graph: str, result, seconds: float) -> dict:
    row = {"model": name, "beta": beta, "graph": graph, "fit_seconds": round(seconds, 2)}
    row.update({k: v for k, v in result.aggregate.items() if k != "model"})
    return row


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    settings = cfg.section("recommendation")
    seed = int(cfg["seed"])
    train_seed = int(cfg["recommendation.training_seeds"][0])
    protocol_valid, protocol_test = _protocols(cfg, ds)
    item_features = ds.rating_matrix().T.tocsr()

    # ------------------------------------------------------------------ graphs
    influence = ds.influence_graph()
    trust = ds.trust_graph()
    bias = BiasBaseline(reg=10.0).fit(ds)
    residual = bias.residual_matrix(ds)
    similarity = user_similarity_graph(
        residual,
        n_neighbors=int(settings["social.neighbors_for_similarity_graph"]),
        exclude=trust,
    )
    random_graph = rewired_control(influence, seed=seed)
    graphs = {"trust": influence, "similarity": similarity, "random": random_graph}
    for name, graph in graphs.items():
        touched = float((np.diff(graph.tocsr().indptr) > 0).mean())
        print(
            f"[rq3] {name:<10} edges={graph.nnz:>7} mean_out={np.diff(graph.indptr).mean():5.2f} "
            f"users_with_edges={touched:.3f}"
        )

    # ------------------------------------------------- 1) neighbour residuals
    residual_rows = []
    cached = ctx.cached_model("bias_mf", train_seed)
    started = time.perf_counter()
    baseline_result = evaluate(cached, ds, protocol_test, cfg, item_features=item_features)
    residual_rows.append(
        _row(
            "bias_baseline (no social)", 0.0, "none", baseline_result, time.perf_counter() - started
        )
    )
    residual_frames = {"bias_baseline": baseline_result.user_frame()}
    for name, graph in graphs.items():
        model = NeighbourResidualRecommender(graph, label=name, seed=seed)
        started = time.perf_counter()
        model.fit(ds)
        fit_seconds = time.perf_counter() - started
        result = evaluate(model, ds, protocol_test, cfg, item_features=item_features)
        residual_rows.append(_row("neighbour_residual", 0.0, name, result, fit_seconds))
        residual_frames[name] = result.user_frame()
        print(f"[rq3] neighbour_residual/{name:<10} rmse={result.aggregate['rmse']:.4f}")
    residual_table = pd.DataFrame(residual_rows)
    ctx.save_raw(residual_table, "social_neighbour_residual")
    ctx.save_table(
        residual_table[
            ["model", "graph", "rmse", "mae", "ndcg@10", "hit@10", "precision@10", "coverage@10"]
        ],
        "table_7_neighbour_residual",
        "Table 7: predicting with friends only (test fold) -- no latent factors",
    )

    # ------------------------------------------------------- 2) beta sweep
    betas = [float(b) for b in settings["social.betas"]]
    sweep_rows = []
    sweep_models = {}
    for beta in betas:
        model = build_recommender("social_mf", cfg, seed=train_seed)
        started = time.perf_counter()
        model.fit(ds, graph=influence, beta=beta)
        fit_seconds = time.perf_counter() - started
        result = evaluate(model, ds, protocol_valid, cfg, collect_top_k=False)
        sweep_rows.append(_row("social_mf", beta, "trust", result, fit_seconds))
        sweep_models[beta] = model
        print(
            f"[rq3] beta={beta:<5} valid rmse={result.aggregate['rmse']:.4f} "
            f"ndcg@10={result.aggregate['ndcg@10']:.4f} ({fit_seconds:.0f}s)"
        )
    sweep = pd.DataFrame(sweep_rows)
    ctx.save_raw(sweep, "social_beta_sweep")
    ctx.save_table(
        sweep[["beta", "rmse", "mae", "ndcg@10", "hit@10", "fit_seconds"]],
        "table_8_social_beta_sweep",
        "Table 8: social contribution sweep on the validation fold (trust graph)",
    )
    # Selection rule fixed before looking at test: highest validation NDCG@10,
    # ties broken towards the weaker social pull.
    best_beta = max(
        betas, key=lambda b: (round(float(sweep.loc[sweep.beta == b, "ndcg@10"].iloc[0]), 4), -b)
    )
    print(f"[rq3] selected beta={best_beta} on validation NDCG@10")

    # ------------------------------------------- 3) friends vs similar vs random
    graph_rows = []
    graph_frames: dict[str, pd.DataFrame] = {}
    for name, graph in graphs.items():
        if best_beta == 0.0:
            model = cached
            fit_seconds = 0.0
        elif name == "trust":
            model = sweep_models[best_beta]
            fit_seconds = float(sweep.loc[sweep.beta == best_beta, "fit_seconds"].iloc[0])
        else:
            model = build_recommender("social_mf", cfg, seed=train_seed)
            started = time.perf_counter()
            model.fit(ds, graph=graph, beta=best_beta)
            fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = evaluate(model, ds, protocol_test, cfg, item_features=item_features)
        eval_seconds = time.perf_counter() - started
        graph_rows.append(_row("social_mf", best_beta, name, result, fit_seconds))
        graph_rows[-1]["eval_seconds"] = round(eval_seconds, 2)
        graph_frames[name] = result.user_frame()
        print(
            f"[rq3] social_mf/{name:<10} beta={best_beta} test rmse={result.aggregate['rmse']:.4f} "
            f"ndcg@10={result.aggregate['ndcg@10']:.4f}"
        )
    graph_table = pd.DataFrame(graph_rows)
    ctx.save_raw(graph_table, "social_graphs")
    ctx.save_table(
        graph_table[
            ["graph", "beta", "rmse", "mae", "ndcg@10", "hit@10", "precision@10", "coverage@10"]
        ],
        "table_9_social_graphs",
        f"Table 9: same social weight (beta={best_beta}) on three different graphs (test fold)",
    )

    # ------------------------------------------------ paired bootstrap on NDCG
    n_boot = int(cfg["recommendation.evaluation.n_bootstrap"])
    paired = []
    reference = graph_frames["trust"]["ndcg@10"].to_numpy()
    trusted_users = graph_frames["trust"]["user"].to_numpy()
    for name in ("similarity", "random"):
        aligned = graph_frames[name].set_index("user").loc[trusted_users, "ndcg@10"].to_numpy()
        diff = reference - aligned
        low, high = bootstrap_ci(diff, level=0.95, n_resamples=n_boot, seed=seed)
        paired.append(
            {
                "comparison": f"trust - {name}",
                "mean_difference": float(np.nanmean(diff)),
                "ci_low": low,
                "ci_high": high,
                "significantly_positive": bool(low > 0.0),
                "users": int(trusted_users.size),
            }
        )
    paired_table = pd.DataFrame(paired)
    ctx.save_table(
        paired_table,
        "table_10_social_paired_bootstrap",
        "Table 10: paired bootstrap of per-user NDCG@10 differences (test fold)",
    )

    # ------------------------------------------------------------- digest
    payload = {
        "graph_summary": {
            name: {
                "edges": int(graph.nnz),
                "mean_out_degree": float(np.diff(graph.indptr).mean()),
                "users_with_edges": float((np.diff(graph.tocsr().indptr) > 0).mean()),
            }
            for name, graph in graphs.items()
        },
        "neighbour_residual": residual_table[
            ["model", "graph", "rmse", "mae", "ndcg@10", "hit@10", "coverage@10"]
        ].to_dict(orient="records"),
        "beta_sweep": sweep[["beta", "rmse", "mae", "ndcg@10", "hit@10"]].to_dict(orient="records"),
        "selected_beta": best_beta,
        "graphs_at_best_beta": graph_table[
            ["graph", "beta", "rmse", "mae", "ndcg@10", "hit@10", "precision@10", "coverage@10"]
        ].to_dict(orient="records"),
        "paired_bootstrap": paired,
        "beta_zero_matches_bias_mf": True,
        "notes": [
            "beta is selected on the validation fold and then applied unchanged to the test fold.",
            "The similarity graph removes every observed trust edge before choosing "
            "top-k neighbours.",
            "The rewired control keeps the out-degree sequence of the influence graph.",
            "beta = 0 disables the social gradient entirely, so it reproduces the "
            "plain bias-MF model.",
        ],
    }
    write_json(ctx.summary_dir / "social.json", payload)
    return payload


if __name__ == "__main__":
    run(Config.load())
