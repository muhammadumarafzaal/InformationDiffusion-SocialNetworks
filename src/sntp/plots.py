"""Result figures.

Every figure is produced from a file the experiments wrote (``results/raw`` or
``results/tables``); nothing is recomputed here, so figures and tables can never
disagree. A missing input only skips its figure with a warning, which keeps
partial runs usable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .utils import ensure_dir

COLOURS = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#4f772d"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 170,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
        }
    )


class Figures:
    def __init__(self, results: Path) -> None:
        self.results = results
        self.raw = results / "raw"
        self.tables = results / "tables"
        self.summary = results / "summary"
        self.out = ensure_dir(results / "figures")
        self.produced: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _read(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            print(f"[figures] skip missing {path.name}")
            return None
        return pd.read_csv(path)

    def _json(self, name: str) -> dict:
        path = self.summary / f"{name}.json"
        if not path.exists():
            print(f"[figures] skip missing {path.name}")
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, fig, name: str) -> None:
        target = self.out / f"{name}.png"
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        self.produced.append(target.name)
        print(f"[figures] wrote {target.name}")

    # ------------------------------------------------------------------ figures
    def rating_distribution(self) -> None:
        payload = self._json("dataset")
        histogram = payload.get("rating_histogram") or {}
        if not histogram:
            return
        values = sorted(histogram.items(), key=lambda kv: float(kv[0]))
        labels = [str(int(float(k))) for k, _ in values]
        counts = [v for _, v in values]
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        bars = ax.bar(labels, counts, color=COLOURS[0], width=0.62)
        ax.set_xlabel("rating")
        ax.set_ylabel("interactions")
        ax.set_title("Rating distribution of the analysed core")
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count / sum(counts):.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        self._save(fig, "fig_1_rating_distribution")

    def recommender_quality(self) -> None:
        frame = self._read(self.raw / "recommenders.csv")
        if frame is None:
            return
        order = ["global_mean", "bias_baseline", "popularity", "bias_mf", "item_knn"]
        grouped = frame.groupby("model").agg(
            ndcg=("ndcg@10", "mean"),
            ndcg_std=("ndcg@10", "std"),
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        grouped = grouped.reindex([m for m in order if m in grouped.index])
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
        positions = np.arange(len(grouped))
        axes[0].bar(
            positions,
            grouped["ndcg"],
            yerr=grouped["ndcg_std"].fillna(0),
            color=COLOURS[: len(grouped)],
            width=0.6,
            capsize=2,
        )
        axes[0].set_xticks(positions, [m.replace("_", "\n") for m in grouped.index], fontsize=7)
        axes[0].set_ylabel("NDCG@10")
        axes[0].set_title("Ranking quality (mean of 3 seeds)")
        axes[1].bar(
            positions,
            grouped["rmse"],
            yerr=grouped["rmse_std"].fillna(0),
            color=COLOURS[: len(grouped)],
            width=0.6,
            capsize=2,
        )
        axes[1].set_xticks(positions, [m.replace("_", "\n") for m in grouped.index], fontsize=7)
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Prediction error (mean of 3 seeds)")
        self._save(fig, "fig_2_recommender_quality")

    def cold_start(self) -> None:
        frames = {}
        for name in ("bias_mf", "item_knn"):
            frame = self._read(self.tables / f"table_5_{name}_by_history_ranking.csv")
            if frame is not None:
                frames[name] = frame
        if not frames:
            return
        order = ["<20", "20-29", "30-49", "50-99", ">=100"]
        fig, ax = plt.subplots(figsize=(5.2, 2.7))
        width = 0.8 / len(frames)
        for index, (name, frame) in enumerate(frames.items()):
            frame = frame.set_index("history_bucket").reindex(
                [o for o in order if o in set(frame.history_bucket)]
            )
            positions = np.arange(len(frame)) + index * width
            ax.bar(positions, frame["ndcg"], width=width, label=name, color=COLOURS[index])
        labels = [o for o in order if o in set(next(iter(frames.values())).history_bucket)]
        ax.set_xticks(np.arange(len(labels)) + width * (len(frames) - 1) / 2, labels)
        ax.set_xlabel("training interactions per user")
        ax.set_ylabel("NDCG@10")
        ax.set_title("Cold-start behaviour: ranking quality by user history")
        ax.legend(title="model")
        self._save(fig, "fig_3_cold_start")

    def social_beta(self) -> None:
        frame = self._read(self.raw / "social_beta_sweep.csv")
        if frame is None:
            return
        fig, ax = plt.subplots(figsize=(4.6, 2.7))
        ax.plot(frame["beta"], frame["ndcg@10"], marker="o", color=COLOURS[0], label="NDCG@10")
        ax.set_xlabel("social weight beta")
        ax.set_ylabel("NDCG@10 (validation)", color=COLOURS[0])
        twin = ax.twinx()
        twin.plot(frame["beta"], frame["rmse"], marker="s", color=COLOURS[2], label="RMSE")
        twin.set_ylabel("RMSE (validation)", color=COLOURS[2])
        twin.grid(False)
        selected = self._json("social").get("selected_beta")
        if isinstance(selected, (int, float)):
            ax.axvline(selected, color=COLOURS[3], linestyle="--", linewidth=0.9)
            ax.set_title(f"Tuning the social contribution (selected beta = {selected:g})")
        else:
            ax.set_title("Tuning the social contribution")
        self._save(fig, "fig_4_social_beta")

    def social_graphs(self) -> None:
        frame = self._read(self.raw / "social_neighbour_residual.csv")
        graphs = self._read(self.raw / "social_graphs.csv")
        if frame is None and graphs is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
        if frame is not None:
            positions = np.arange(frame.shape[0])
            axes[0].bar(positions, frame["ndcg@10"], color=COLOURS[: frame.shape[0]], width=0.6)
            axes[0].set_xticks(positions, frame["graph"], fontsize=7)
            axes[0].set_ylabel("NDCG@10")
            axes[0].set_title("Friends only (no latent factors)")
        if graphs is not None:
            positions = np.arange(graphs.shape[0])
            axes[1].bar(positions, graphs["ndcg@10"], color=COLOURS[: graphs.shape[0]], width=0.6)
            axes[1].set_xticks(positions, graphs["graph"], fontsize=7)
            axes[1].set_ylabel("NDCG@10")
            axes[1].set_title("Same beta, different graph")
        self._save(fig, "fig_5_social_graphs")

    def diffusion_curves(self) -> None:
        frame = self._read(self.raw / "diffusion_curves.csv")
        if frame is None:
            return
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        styles = {"ic": "-", "sir": "--"}
        p0_values = sorted(frame["p0"].unique())
        for index, p0 in enumerate(p0_values):
            for model in ("ic", "sir"):
                block = frame[(frame["p0"] == p0) & (frame["model"] == model)]
                if block.empty:
                    continue
                ax.plot(
                    block["step"],
                    block["cumulative"],
                    styles[model],
                    color=COLOURS[index % len(COLOURS)],
                    label=f"{model.upper()} p0={p0:g}",
                    linewidth=1.4,
                )
        ax.set_xlabel("step")
        ax.set_ylabel("cumulative adoptions")
        ax.set_title("Average cascade shape (100 simulations per curve)")
        ax.legend(ncol=2, fontsize=6.5)
        self._save(fig, "fig_6_diffusion_curves")

    def diffusion_sensitivity(self) -> None:
        grid = self._read(self.raw / "diffusion_grid.csv")
        weights = self._read(self.raw / "diffusion_influence_weight.csv")
        if grid is None and weights is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
        if grid is not None:
            for index, model in enumerate(("ic", "sir")):
                block = grid[grid["model"] == model].sort_values("p0")
                if block.empty:
                    continue
                axes[0].errorbar(
                    block["p0"],
                    block["reach_mean"],
                    yerr=block["reach_std"],
                    marker="o",
                    capsize=2,
                    color=COLOURS[index],
                    label=model.upper(),
                )
            axes[0].set_xlabel("base transmission probability p0")
            axes[0].set_ylabel("mean reach (users)")
            axes[0].set_title("Reach vs transmission rate")
            axes[0].legend()
        if weights is not None:
            axes[1].errorbar(
                weights["influence_weight"],
                weights["reach_mean"],
                yerr=weights["reach_std"],
                marker="s",
                capsize=2,
                color=COLOURS[3],
            )
            axes[1].set_xlabel("affinity weight")
            axes[1].set_ylabel("mean reach (users)")
            axes[1].set_title("Homogeneous (0) vs preference-driven (1)")
        self._save(fig, "fig_7_diffusion_sensitivity")

    def seed_selection(self) -> None:
        frame = self._read(self.raw / "influence_runs.csv")
        marginal = self._read(self.raw / "marginal_gains.csv")
        if frame is None and marginal is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
        if frame is not None:
            for index, (strategy, block) in enumerate(frame.groupby("strategy")):
                block = block.sort_values("budget")
                axes[0].errorbar(
                    block["budget"],
                    block["reach_mean"],
                    yerr=block["reach_std"],
                    marker="o",
                    capsize=2,
                    color=COLOURS[index % len(COLOURS)],
                    label=strategy,
                )
            axes[0].set_xlabel("budget (seeds)")
            axes[0].set_ylabel("realised reach (users)")
            axes[0].set_title("Budget vs reach")
            axes[0].legend()
        if marginal is not None:
            axes[1].errorbar(
                marginal["step"],
                marginal["marginal_gain"],
                yerr=marginal["marginal_std"],
                marker="o",
                capsize=2,
                color=COLOURS[2],
            )
            axes[1].set_xlabel("greedy step")
            axes[1].set_ylabel("marginal reach of the added seed")
            axes[1].set_title("Diminishing marginal gain")
        self._save(fig, "fig_8_seed_selection")

    def groups(self) -> None:
        frame = self._read(self.tables / "table_18_group_strategies.csv")
        if frame is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
        kinds = frame["group_kind"].unique()
        width = 0.8 / max(len(kinds), 1)
        strategies = list(frame["strategy"].unique())
        for index, kind in enumerate(kinds):
            block = frame[frame.group_kind == kind].set_index("strategy").reindex(strategies)
            positions = np.arange(len(strategies)) + index * width
            axes[0].bar(
                positions, block["mean_satisfaction"], width=width, label=kind, color=COLOURS[index]
            )
            axes[1].bar(positions, block["reach"], width=width, label=kind, color=COLOURS[index])
        for ax, title, label in (
            (axes[0], "Predicted satisfaction", "mean predicted rating"),
            (axes[1], "Realised spread", "reach (users)"),
        ):
            ax.set_xticks(
                np.arange(len(strategies)) + width * (len(kinds) - 1) / 2,
                [s.replace("_", "\n") for s in strategies],
                fontsize=7,
            )
            ax.set_title(title)
            ax.set_ylabel(label)
            ax.legend(title="group")
        self._save(fig, "fig_9_groups")

    def intervention(self) -> None:
        frame = self._read(self.raw / "intervention.csv")
        if frame is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
        caps = sorted(frame["exposure_cap"].unique())
        gates = sorted(frame["affinity_gate"].unique())
        width = 0.8 / len(gates)
        for index, gate in enumerate(gates):
            block = frame[frame.affinity_gate == gate].set_index("exposure_cap").reindex(caps)
            positions = np.arange(len(caps)) + index * width
            axes[0].bar(
                positions,
                block["reach_reduction"],
                width=width,
                label=f"gate {gate:g}",
                color=COLOURS[index],
            )
            axes[1].bar(
                positions,
                block["high_affinity_loss"],
                width=width,
                label=f"gate {gate:g}",
                color=COLOURS[index],
            )
        for ax, title, label in (
            (axes[0], "Effectiveness", "reach reduction"),
            (axes[1], "Collateral damage", "reach lost among high-affinity users"),
        ):
            ax.set_xticks(
                np.arange(len(caps)) + width * (len(gates) - 1) / 2, [str(c) for c in caps]
            )
            ax.set_xlabel("exposure cap per user")
            ax.set_title(title)
            ax.set_ylabel(label)
            ax.legend(title="affinity gate")
        self._save(fig, "fig_10_intervention")

    def scaling(self) -> None:
        users = self._read(self.tables / "table_27_scaling_users.csv")
        graphs = self._read(self.tables / "table_29_scaling_diffusion.csv")
        if users is None and graphs is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
        if users is not None:
            axes[0].plot(users["ratings"], users["fit_seconds"], marker="o", color=COLOURS[0])
            axes[0].set_xlabel("training interactions")
            axes[0].set_ylabel("fit wall-clock (s)")
            axes[0].set_title("Recommender training cost")
        if graphs is not None:
            axes[1].plot(graphs["edges"], graphs["ms_per_simulation"], marker="s", color=COLOURS[3])
            axes[1].set_xlabel("graph edges")
            axes[1].set_ylabel("ms per cascade")
            axes[1].set_title("Cascade simulation cost")
        self._save(fig, "fig_11_scaling")

    def calibration(self) -> None:
        frame = self._read(self.tables / "table_23_calibration.csv")
        if frame is None:
            return
        fig, ax = plt.subplots(figsize=(3.9, 3.0))
        low = float(min(frame["mean_prediction"].min(), frame["mean_observed"].min()))
        high = float(max(frame["mean_prediction"].max(), frame["mean_observed"].max()))
        ax.plot([low, high], [low, high], linestyle=":", color="grey", label="perfect calibration")
        ax.plot(frame["mean_prediction"], frame["mean_observed"], marker="o", color=COLOURS[0])
        ax.set_xlabel("mean predicted rating")
        ax.set_ylabel("mean observed rating")
        ax.set_title("Calibration on the test fold")
        ax.legend()
        self._save(fig, "fig_12_calibration")

    def run_all(self) -> list[str]:
        _style()
        for step in (
            self.rating_distribution,
            self.recommender_quality,
            self.cold_start,
            self.social_beta,
            self.social_graphs,
            self.diffusion_curves,
            self.diffusion_sensitivity,
            self.seed_selection,
            self.groups,
            self.intervention,
            self.scaling,
            self.calibration,
        ):
            try:
                step()
            except Exception as error:  # a single bad figure must not stop the rest
                print(f"[figures] {step.__name__} failed: {error}")
        return self.produced


def run(cfg: Config) -> dict:
    results = cfg.path("paths.results_dir")
    produced = Figures(results).run_all()
    print(f"[figures] {len(produced)} figures in {results / 'figures'}")
    return {"figures": produced}


if __name__ == "__main__":
    run(Config.load())
