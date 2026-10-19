"""Requirement 2 / 7 -- train both recommender families and evaluate them.

Five models are trained over three seeds and scored with the same protocol:
``global_mean`` and ``bias_baseline`` establish the trivial baselines,
``popularity`` is a non-personalised ranker, ``bias_mf`` learns latent factors
from the interactions and ``item_knn`` learns item-item similarities from the
same interactions. The two learned models differ substantially in inductive bias
(latent factors vs. neighbourhood interpolation), which is what Requirement 2
asks for.

Reported per run: prediction error (MAE, RMSE), ranking quality at K=10
(precision, recall, hit rate, NDCG, MAP), catalogue coverage, Gini, intra-list
diversity, novelty, and the wall-clock/memory cost of fitting. Results are then
sliced by user history length and item popularity to expose cold-start
behaviour, and a binary adoption task provides the only accuracy figure in the
report (with its no-skill baseline).
"""

from __future__ import annotations

import time

import pandas as pd

from ..config import Config
from ..data import TEST
from ..evaluation import adoption_classification, build_protocol, evaluate, sliced_report
from ..recommenders import build_recommender
from ..utils import current_rss_mb, write_json
from .common import build_context, save_model, snapshot_config

MODELS = ("global_mean", "bias_baseline", "popularity", "bias_mf", "item_knn")

LABELS = {
    "global_mean": "Global mean",
    "bias_baseline": "Bias baseline (ridge)",
    "popularity": "Popularity",
    "bias_mf": "Bias-MF (latent factors)",
    "item_knn": "Item-kNN (neighbourhood)",
}

HEADLINE = (
    "rmse",
    "mae",
    "precision@10",
    "recall@10",
    "hit@10",
    "ndcg@10",
    "map@10",
    "coverage@10",
    "diversity@10",
    "novelty@10",
    "gini@10",
    "fit_seconds",
)


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    settings = cfg.section("recommendation.evaluation")
    protocol = build_protocol(
        ds,
        n_negatives=int(settings["n_negatives"]),
        relevance_threshold=float(settings["relevance_threshold"]),
        seed=int(cfg["seed"]),
        fold=TEST,
    )
    # Item features for intra-list diversity: binary "who rated this item" vectors.
    item_features = ds.rating_matrix().T.tocsr()
    seeds = [int(s) for s in cfg["recommendation.training_seeds"]]

    runs: list[dict] = []
    classification: list[dict] = []
    slices: dict[str, dict[str, pd.DataFrame]] = {}

    for name in MODELS:
        for seed in seeds:
            model = build_recommender(name, cfg, seed=seed)
            rss_before = current_rss_mb()
            started = time.perf_counter()
            model.fit(ds)
            fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            result = evaluate(model, ds, protocol, cfg, item_features=item_features)
            eval_seconds = time.perf_counter() - started

            row = {
                "model": name,
                "train_seed": seed,
                "fit_seconds": round(fit_seconds, 3),
                "eval_seconds": round(eval_seconds, 3),
                "fit_rss_mb": round(max(0.0, current_rss_mb() - rss_before), 1),
            }
            row.update({k: v for k, v in result.aggregate.items() if k != "model"})
            runs.append(row)

            task = adoption_classification(
                model, protocol, threshold=float(settings["classification_threshold"])
            )
            classification.append({"model": name, "train_seed": seed, **task})

            if seed == seeds[0]:
                if name in {"bias_mf", "social_mf"}:
                    save_model(
                        model,
                        ctx.results_dir / "cache" / f"{name}_seed{seed}.npz",
                        metadata={"train_seed": seed, "model": name},
                    )
                if name in {"bias_mf", "item_knn"}:
                    slices[name] = sliced_report(result, ds)
            print(
                f"[rq2] {name:<14} seed={seed} rmse={row['rmse']:.4f} "
                f"ndcg@10={row['ndcg@10']:.4f} fit={fit_seconds:.1f}s"
            )

    raw = pd.DataFrame(runs)
    ctx.save_raw(raw, "recommenders")

    # ------------------------------------------------------------ mean +/- std
    metrics = [c for c in raw.columns if c not in {"model", "train_seed"}]
    grouped = raw.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary_rows = []
    for name in MODELS:
        entry: dict[str, object] = {"model": LABELS[name]}
        for metric in HEADLINE:
            mean = float(grouped.loc[name, (metric, "mean")])
            std = float(grouped.loc[name, (metric, "std")]) if (metric, "std") in grouped else 0.0
            entry[metric] = f"{mean:.4f} +/- {std:.4f}"
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        summary_rows.append(entry)
    summary = pd.DataFrame(summary_rows)
    ctx.save_table(
        summary[["model", *HEADLINE]],
        "table_4_recommender_performance",
        "Table 4: recommender performance on the test fold "
        "(mean +/- std over three training seeds)",
    )

    # ---------------------------------------------------------- cold-start view
    for name, report in slices.items():
        for part, frame in report.items():
            frame = frame.copy()
            frame.insert(0, "model", LABELS[name])
            ctx.save_table(
                frame,
                f"table_5_{name}_{part}",
                f"Table 5: {LABELS[name]} -- {part.replace('_', ' ')}",
            )

    # ------------------------------------------------------ binary adoption task
    cls_frame = pd.DataFrame(classification)
    cls_summary = cls_frame.groupby("model", sort=False).agg(
        accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        positive_rate=("positive_rate", "mean"),
        no_skill_accuracy=("no_skill_accuracy", "mean"),
    )
    cls_summary.insert(0, "model", [LABELS[m] for m in cls_summary.index])
    ctx.save_raw(cls_frame, "adoption_classification")
    ctx.save_table(
        cls_summary.reset_index(drop=True),
        "table_6_adoption_classification",
        "Table 6: binary adoption task (rating >= threshold vs unobserved item), mean over seeds",
    )

    # --------------------------------------------------------------- JSON digest
    means = {name: {m: float(grouped.loc[name, (m, "mean")]) for m in HEADLINE} for name in MODELS}
    best_error = min(MODELS, key=lambda m: means[m]["rmse"])
    best_ranking = max(MODELS, key=lambda m: means[m]["ndcg@10"])
    payload = {
        "protocol": protocol.summary(),
        "seeds": seeds,
        "models": means,
        "best_error_model": best_error,
        "best_ranking_model": best_ranking,
        "classification": cls_summary.to_dict(orient="records"),
        "fit_seconds": {name: float(grouped.loc[name, ("fit_seconds", "mean")]) for name in MODELS},
        "notes": [
            "All models see the training fold only; the protocol uses the test fold.",
            "Ranking candidates are the user's test items plus 100 never-observed items, "
            "shuffled before scoring.",
            "accuracy is reported only for the binary adoption task and always next to "
            "its no-skill baseline.",
        ],
    }
    write_json(ctx.summary_dir / "recommenders.json", payload)
    print(
        f"[rq2] best RMSE: {LABELS[best_error]} ({means[best_error]['rmse']:.4f}); "
        f"best NDCG@10: {LABELS[best_ranking]} ({means[best_ranking]['ndcg@10']:.4f})"
    )
    return payload


if __name__ == "__main__":
    run(Config.load())
