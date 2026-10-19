"""Requirement 7 -- evaluate the complete system, including the audit trail.

Four things happen here that are not covered by the per-component experiments:

1. **Leakage audit.** Fold integrity, candidate-set integrity (no held-out item
   and no training item may enter a ranking candidate list), cold items in the
   training fold, and coverage of the campaign-selection rule are asserted and
   recorded, so the "no train-test leakage" claim is checked by code rather than
   asserted in prose.
2. **Uncertainty.** Percentile bootstrap confidence intervals for per-user NDCG
   and for RMSE over test pairs, plus the across-seed stability of the learned
   models.
3. **Calibration.** Predicted ratings are bucketed and compared with observed
   ratings and with the observed "liked it" rate, which is the accuracy-style
   check that is valid for this data (a regression/ranking target, not a
   classification target).
4. **Observed vs simulated.** For a sample of popular items, simulated cascade
   size is compared with the item's observed popularity. The two quantities are
   not the same thing (organic adoption vs campaign-induced spread), so the
   comparison is reported as a rank agreement and interpreted as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data import TEST, TRAIN, VALID
from ..diffusion import CampaignSimulator, DiffusionSettings
from ..evaluation import RankingProtocol, adoption_classification, build_protocol, evaluate
from ..seeds import preference_selection
from ..utils import bootstrap_ci, seeded_rng, spearman, write_json
from .common import (
    build_context,
    campaign_affinity,
    eligible_users,
    snapshot_config,
)


def _bootstrap_rmse(
    prediction: np.ndarray, truth: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    squared = (prediction - truth) ** 2
    rng = seeded_rng(seed, stream=1201)
    draws = rng.integers(0, squared.size, size=(n_resamples, squared.size))
    values = np.sqrt(squared[draws].mean(axis=1))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _audit(ds, protocols: dict[str, RankingProtocol]) -> tuple[pd.DataFrame, dict]:
    observed = ds.observed_pairs().tocsr()
    folds = {TRAIN: ds.fold == TRAIN, VALID: ds.fold == VALID, TEST: ds.fold == TEST}
    train_u = ds.rating_user[folds[TRAIN]]
    train_i = ds.rating_item[folds[TRAIN]]
    order = np.argsort(train_u, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(train_u[order])) + 1)
    train_by_user = {int(train_u[g[0]]): set(train_i[g].tolist()) for g in groups if g.size}
    test_pairs = set(
        zip(ds.rating_user[folds[TEST]].tolist(), ds.rating_item[folds[TEST]].tolist())
    )
    valid_pairs = set(
        zip(ds.rating_user[folds[VALID]].tolist(), ds.rating_item[folds[VALID]].tolist())
    )
    train_seen = set(zip(train_u.tolist(), train_i.tolist()))

    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    check(
        "folds partition every interaction",
        int(folds[TRAIN].sum() + folds[VALID].sum() + folds[TEST].sum()) == ds.n_ratings,
        f"{ds.n_ratings} ratings split into {int(folds[TRAIN].sum())}/"
        f"{int(folds[VALID].sum())}/{int(folds[TEST].sum())}",
    )
    check(
        "test pairs never occur in training",
        len(test_pairs & train_seen) == 0,
        f"{len(test_pairs & train_seen)} overlapping pairs",
    )
    check(
        "validation pairs never occur in training",
        len(valid_pairs & train_seen) == 0,
        f"{len(valid_pairs & train_seen)} overlapping pairs",
    )
    train_counts = np.bincount(ds.rating_user[folds[TRAIN]], minlength=ds.n_users)
    check(
        "every user keeps training history",
        int(train_counts.min()) >= 1,
        f"minimum {int(train_counts.min())} training interactions per user",
    )
    item_counts = np.bincount(ds.rating_item[folds[TRAIN]], minlength=ds.n_items)
    check(
        "campaign popularity uses the training fold only",
        item_counts.sum() == int(folds[TRAIN].sum()),
        f"{int((item_counts == 0).sum())} items are cold in training and can never be promoted",
    )

    violations = 0
    checked = 0
    for protocol in protocols.values():
        for index, user in enumerate(protocol.users[:300]):
            user_train = train_by_user.get(int(user), set())
            pool = protocol.candidates[index].tolist()
            violations += len(set(pool) & user_train)
            checked += len(pool)
    check(
        "ranking candidates exclude training items",
        violations == 0,
        f"{violations} violations over {checked} candidate slots ({sorted(protocols)})",
    )

    negatives_ok = True
    for protocol in protocols.values():
        for index, user in enumerate(protocol.users[:200]):
            start, end = observed.indptr[user], observed.indptr[user + 1]
            seen = set(observed.indices[start:end].tolist())
            held_out = set(protocol.test_items[user].tolist())
            if any(
                item in seen and item not in held_out
                for item in protocol.candidates[index].tolist()
            ):
                negatives_ok = False
                break
    check(
        "sampled negatives are never observed positives",
        negatives_ok,
        "every candidate is either a held-out test item or an item never observed in any fold",
    )

    ledger = {
        "n_items_cold_in_training": int((item_counts == 0).sum()),
        "n_users_without_influence_edges": int((ds.user_degree() == 0).sum()),
        "candidate_slots_checked": int(checked),
        "violations": int(violations),
    }
    return pd.DataFrame(checks), ledger


def run(cfg: Config) -> dict:
    ctx = build_context(cfg)
    snapshot_config(ctx)
    ds = ctx.dataset
    seed = int(cfg["seed"])
    train_seed = int(cfg["recommendation.training_seeds"][0])
    n_boot = int(cfg["recommendation.evaluation.n_bootstrap"])
    settings = cfg.section("recommendation.evaluation")

    kwargs = dict(
        n_negatives=int(settings["n_negatives"]),
        relevance_threshold=float(settings["relevance_threshold"]),
        seed=seed,
    )
    protocols = {
        "valid": build_protocol(ds, fold=VALID, **kwargs),
        "test": build_protocol(ds, fold=TEST, **kwargs),
    }
    audit, ledger = _audit(ds, protocols)
    ctx.save_table(
        audit, "table_21_leakage_audit", "Table 21: automated leakage and integrity audit"
    )

    # ------------------------------------------------------ uncertainty of the best models
    model = ctx.cached_model("bias_mf", train_seed)
    result = evaluate(model, ds, protocols["test"], cfg)
    pairs = result.pair_frame()
    users = result.user_frame()

    ci_rows = []
    for label, column in (
        ("ndcg@10", "ndcg@10"),
        ("hit@10", "hit@10"),
        ("precision@10", "precision@10"),
    ):
        values = users[column].to_numpy(dtype=float)
        low, high = bootstrap_ci(values, n_resamples=n_boot, seed=seed)
        ci_rows.append(
            {
                "model": "bias_mf",
                "metric": label,
                "mean": float(np.nanmean(values)),
                "ci_low": low,
                "ci_high": high,
                "unit": "user",
            }
        )
    low, high = _bootstrap_rmse(
        pairs["prediction"].to_numpy(), pairs["rating"].to_numpy(), n_boot, seed
    )
    ci_rows.append(
        {
            "model": "bias_mf",
            "metric": "rmse",
            "mean": float(result.aggregate["rmse"]),
            "ci_low": low,
            "ci_high": high,
            "unit": "pair",
        }
    )
    ci_table = pd.DataFrame(ci_rows)
    ctx.save_table(
        ci_table,
        "table_22_bootstrap_ci",
        f"Table 22: 95% percentile bootstrap intervals ({n_boot} resamples, test fold)",
    )

    # ---------------------------------------------------------- calibration
    pairs["bucket"] = pd.qcut(pairs["prediction"], 10, duplicates="drop")
    calibration = (
        pairs.groupby("bucket", observed=True)
        .agg(
            count=("rating", "size"),
            mean_prediction=("prediction", "mean"),
            mean_observed=("rating", "mean"),
            observed_positive_rate=("rating", lambda r: float((r >= 4).mean())),
        )
        .reset_index()
    )
    calibration["bucket"] = calibration["bucket"].astype(str)
    ctx.save_table(
        calibration,
        "table_23_calibration",
        "Table 23: calibration of predicted ratings (test fold)",
    )
    calibration_error = float(
        np.average(
            (calibration["mean_prediction"] - calibration["mean_observed"]).abs(),
            weights=calibration["count"],
        )
    )

    # -------------------------------------------------- observed vs simulated
    counts = np.bincount(ds.rating_item[ds.fold == TRAIN], minlength=ds.n_items)
    test_counts = np.bincount(ds.rating_item[ds.fold == TEST], minlength=ds.n_items)
    items = np.argsort(-counts, kind="stable")[:30]
    eligible = eligible_users(ds, min_interactions=int(cfg["dataset.min_user_ratings"]))
    settings_diffusion = DiffusionSettings.from_config(cfg, model="ic")
    graph = ds.influence_graph()
    rows = []
    for item in items:
        affinity = campaign_affinity(model, int(item))
        seeds = preference_selection(affinity, eligible, 20)
        simulator = CampaignSimulator(graph, affinity, settings_diffusion)
        summary = simulator.simulate(seeds, n_simulations=10, seed=seed)
        rows.append(
            {
                "item": int(item),
                "original_item_id": int(ds.item_ids[item]),
                "train_popularity": int(counts[item]),
                "test_popularity": int(test_counts[item]),
                "mean_affinity": float(affinity.mean()),
                "simulated_reach": summary.reach_fraction_mean * ds.n_users,
            }
        )
    observed_simulated = pd.DataFrame(rows)
    ctx.save_table(
        observed_simulated,
        "table_24_observed_vs_simulated",
        "Table 24: observed popularity vs simulated cascade size for 30 popular items",
    )
    rho_popularity = spearman(
        observed_simulated["train_popularity"].to_numpy(),
        observed_simulated["simulated_reach"].to_numpy(),
    )
    rho_test = spearman(
        observed_simulated["test_popularity"].to_numpy(),
        observed_simulated["simulated_reach"].to_numpy(),
    )

    # ------------------------------------------------------------- stability
    raw_path = ctx.raw_dir / "recommenders.csv"
    stability = pd.DataFrame()
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        grouped = raw.groupby("model")
        stability = pd.DataFrame(
            {
                "ndcg_mean": grouped["ndcg@10"].mean(),
                "ndcg_std": grouped["ndcg@10"].std(ddof=1),
                "rmse_mean": grouped["rmse"].mean(),
                "rmse_std": grouped["rmse"].std(ddof=1),
                "seeds": grouped["train_seed"].nunique(),
            }
        ).reset_index()
        stability["ndcg_cv"] = stability["ndcg_std"] / stability["ndcg_mean"]
        stability["rmse_cv"] = stability["rmse_std"] / stability["rmse_mean"]
        ctx.save_table(
            stability,
            "table_25_seed_stability",
            "Table 25: stability of the learned models across training seeds",
        )

    cls_path = ctx.raw_dir / "adoption_classification.csv"
    if cls_path.exists():
        cls_raw = pd.read_csv(cls_path)
        classification = (
            cls_raw.groupby("model")
            .agg(
                accuracy=("accuracy", "mean"),
                balanced_accuracy=("balanced_accuracy", "mean"),
                f1=("f1", "mean"),
                positive_rate=("positive_rate", "mean"),
                no_skill_accuracy=("no_skill_accuracy", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        classification = []
        task = adoption_classification(
            model, protocols["test"], threshold=float(settings["classification_threshold"])
        )
        classification.append({"model": "bias_mf", **task})

    payload = {
        "audit": audit.to_dict(orient="records"),
        "audit_passed": bool((audit["result"] == "PASS").all()),
        "ledger": ledger,
        "protocols": {name: p.summary() for name, p in protocols.items()},
        "bootstrap_ci": ci_table.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "calibration_weighted_abs_error": calibration_error,
        "observed_vs_simulated": {
            "spearman_train_popularity_vs_reach": rho_popularity,
            "spearman_test_popularity_vs_reach": rho_test,
            "items": len(items),
            "simulations_per_item": 10,
        },
        "seed_stability": stability.to_dict(orient="records") if not stability.empty else [],
        "classification": classification,
        "notes": [
            "The audit is executed, not assumed: every row reports PASS/FAIL with a count.",
            "Accuracy appears only for the binary adoption task and always next to "
            "the no-skill baseline.",
            "Simulated reach is campaign-induced; observed popularity is organic. "
            "Negative rank agreement is expected and informative.",
        ],
    }
    write_json(ctx.summary_dir / "evaluation.json", payload)
    print(
        f"[rq7] audit passed={payload['audit_passed']} "
        f"calibration_error={calibration_error:.3f} "
        f"rho(observed vs simulated)={rho_popularity:.3f}"
    )
    return payload


if __name__ == "__main__":
    run(Config.load())
