"""Budgeted seed selection for a promotion campaign (Requirement 5).

Strategies compared
-------------------
``greedy_mc``      Monte-Carlo marginal-gain greedy with CELF lazy evaluation.
                   A candidate is scored by the *additional* expected reach it
                   creates on top of the already selected seeds, never by its own
                   popularity, which is what the assignment requires.
``random``         uniform random users from the eligible pool.
``degree``         highest influence out-degree (a simple network baseline).
``preference``     users with the strongest individual affinity for the item
                   (the "individual popularity" baseline).
``proxy``          a cheap expected-influence heuristic (affinity x 1-hop audience)
                   used both as a prefilter and as an ablation of the greedy search.

Uncertainty is handled explicitly: every marginal gain is a mean over
``n_mc`` independent cascades and is reported with its standard deviation, and
final seed sets are re-evaluated with a larger simulation budget.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from .diffusion import CampaignSimulator
from .utils import seeded_rng, spearman


@dataclass
class SeedSelection:
    strategy: str
    budget: int
    seeds: np.ndarray
    gains: np.ndarray = field(default_factory=lambda: np.empty(0))
    gain_std: np.ndarray = field(default_factory=lambda: np.empty(0))
    simulations: int = 0
    seconds: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget": self.budget,
            "seeds": " ".join(str(int(s)) for s in self.seeds),
            "mean_marginal_gain": float(self.gains.mean()) if self.gains.size else float("nan"),
            "last_marginal_gain": float(self.gains[-1]) if self.gains.size else float("nan"),
            "last_marginal_std": float(self.gain_std[-1]) if self.gain_std.size else float("nan"),
            "simulations": int(self.simulations),
            "seconds": round(self.seconds, 3),
        }


class SeedSelector:
    """Monte-Carlo greedy selection with optional CELF lazy evaluation."""

    def __init__(self, simulator: CampaignSimulator, seed: int = 0, paired: bool = True) -> None:
        self.simulator = simulator
        self.seed = int(seed)
        self.paired = paired
        self.simulations = 0

    # ------------------------------------------------------------------ scoring
    def reach(self, seeds: np.ndarray, n_mc: int, stream: int) -> tuple[float, float]:
        rng = seeded_rng(self.seed, stream=stream)
        values = np.empty(n_mc)
        for index in range(n_mc):
            values[index] = self.simulator.run(seeds, rng).reach
        self.simulations += n_mc
        if values.size == 1:
            return float(values[0]), 0.0
        return float(values.mean()), float(values.std(ddof=1))

    def marginal_gain(
        self,
        selected: np.ndarray,
        candidate: int,
        n_mc: int,
        stream: int,
    ) -> tuple[float, float]:
        """E[reach(S + c)] - E[reach(S)] using common random numbers per sample."""
        if selected.size == 0:
            return self.reach(np.array([candidate]), n_mc, stream)
        rng = seeded_rng(self.seed, stream=stream)
        differences = np.empty(n_mc)
        trial = np.empty(selected.size + 1, dtype=np.int64)
        for index in range(n_mc):
            trial[: selected.size] = selected
            trial[-1] = candidate
            with_candidate = self.simulator.run(trial, rng).reach
            without_candidate = self.simulator.run(selected, rng).reach
            differences[index] = with_candidate - without_candidate
        self.simulations += 2 * n_mc
        if differences.size == 1:
            return float(differences[0]), 0.0
        return float(differences.mean()), float(differences.std(ddof=1))

    # ------------------------------------------------------------------- greedy
    def greedy(
        self,
        candidates: np.ndarray,
        budget: int,
        n_mc: int,
        celf: bool = True,
        stream: int = 900,
    ) -> SeedSelection:
        started = time.perf_counter()
        candidates = np.asarray(candidates, dtype=np.int64)
        selected: list[int] = []
        gains: list[float] = []
        gain_std: list[float] = []

        if not celf:
            for step in range(budget):
                best = (-np.inf, -1, 0.0)
                for index, candidate in enumerate(candidates):
                    gain, std = self.marginal_gain(
                        np.asarray(selected, dtype=np.int64),
                        int(candidate),
                        n_mc,
                        stream + 101 * step + index,
                    )
                    if gain > best[0]:
                        best = (gain, int(candidate), std)
                selected.append(best[1])
                gains.append(best[0])
                gain_std.append(best[2])
            return SeedSelection(
                strategy="greedy_mc",
                budget=budget,
                seeds=np.asarray(selected, dtype=np.int64),
                gains=np.asarray(gains),
                gain_std=np.asarray(gain_std),
                simulations=self.simulations,
                seconds=time.perf_counter() - started,
            )

        heap: list[tuple[float, int, int, float]] = []
        for index, candidate in enumerate(candidates):
            gain, std = self.marginal_gain(
                np.empty(0, dtype=np.int64), int(candidate), n_mc, stream + index
            )
            heap.append((-gain, int(candidate), 0, std))
        heapq.heapify(heap)

        for step in range(budget):
            while True:
                negative_gain, candidate, computed_at, std = heapq.heappop(heap)
                if computed_at == step:
                    selected.append(candidate)
                    gains.append(-negative_gain)
                    gain_std.append(std)
                    break
                gain, new_std = self.marginal_gain(
                    np.asarray(selected, dtype=np.int64),
                    candidate,
                    n_mc,
                    stream + 7717 * (step + 1) + candidate,
                )
                heapq.heappush(heap, (-gain, candidate, step, new_std))
            if not heap:
                break

        return SeedSelection(
            strategy="greedy_mc",
            budget=budget,
            seeds=np.asarray(selected, dtype=np.int64),
            gains=np.asarray(gains),
            gain_std=np.asarray(gain_std),
            simulations=self.simulations,
            seconds=time.perf_counter() - started,
        )


# --------------------------------------------------------------------- baselines
def random_selection(
    candidates: np.ndarray, budget: int, seed: int, repetitions: int = 20
) -> np.ndarray:
    """One random draw; the experiment repeats it to report variability."""
    rng = seeded_rng(seed, stream=1234)
    return rng.choice(candidates, size=min(budget, candidates.size), replace=False)


def degree_selection(graph: sparse.csr_matrix, candidates: np.ndarray, budget: int) -> np.ndarray:
    degree = np.diff(graph.tocsr().indptr)
    order = np.argsort(-degree[candidates], kind="stable")
    return np.asarray(candidates, dtype=np.int64)[order[:budget]]


def preference_selection(scores: np.ndarray, candidates: np.ndarray, budget: int) -> np.ndarray:
    order = np.argsort(-scores[candidates], kind="stable")
    return np.asarray(candidates, dtype=np.int64)[order[:budget]]


# ------------------------------------------------------------------ heuristics
def expected_influence_proxy(
    graph: sparse.csr_matrix, affinity: np.ndarray, damping: float = 0.5
) -> np.ndarray:
    """Cheap two-hop influence estimate: affinity x (own + damped audience affinity).

    This is the signal behind the ``proxy`` baseline and the greedy prefilter. It
    ignores overlap between candidates, which is exactly the weakness the greedy
    marginal-gain strategy is designed to fix.
    """
    graph = graph.tocsr()
    hop_one = graph @ affinity
    scale = max(float(np.mean(hop_one[hop_one > 0])), 1e-9) if np.any(hop_one > 0) else 1.0
    return affinity * (1.0 + damping * hop_one / scale)


def prefilter_candidates(
    proxy: np.ndarray,
    eligible: np.ndarray,
    pool_size: int,
    seed: int,
    strategy: str = "proxy",
) -> np.ndarray:
    """Restrict the greedy candidate pool to keep the search affordable."""
    eligible = np.asarray(eligible, dtype=np.int64)
    if pool_size <= 0 or pool_size >= eligible.size:
        return eligible
    if strategy == "random":
        rng = seeded_rng(seed, stream=555)
        return np.sort(rng.choice(eligible, size=pool_size, replace=False))
    order = np.argsort(-proxy[eligible], kind="stable")
    return eligible[order[:pool_size]]


# --------------------------------------------------------------------- overlap
def audience_overlap(
    seeds: np.ndarray, graph: sparse.csr_matrix, n_pairs: int = 200, seed: int = 0
) -> dict[str, float]:
    """Overlap between the one-hop audiences of the selected seeds."""
    graph = graph.tocsr()
    indptr, indices = graph.indptr, graph.indices
    audiences = [set(indices[indptr[int(s)] : indptr[int(s) + 1]].tolist()) for s in seeds]
    total = sum(len(a) for a in audiences)
    union = set().union(*audiences) if audiences else set()
    rng = seeded_rng(seed, stream=88)
    jaccards: list[float] = []
    if len(audiences) > 1:
        pairs = min(n_pairs, len(audiences) * (len(audiences) - 1) // 2)
        for _ in range(pairs):
            i, j = rng.choice(len(audiences), size=2, replace=False)
            inter = len(audiences[i] & audiences[j])
            denominator = len(audiences[i] | audiences[j])
            jaccards.append(inter / denominator if denominator else 0.0)
    return {
        "sum_audience": float(total),
        "unique_audience": float(len(union)),
        "audience_overlap": float(1.0 - len(union) / total) if total else 0.0,
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "mean_degree": float(total / max(len(audiences), 1)),
    }


def rank_agreement(marginal_gains: np.ndarray, alternative: np.ndarray) -> float:
    """Spearman correlation between true marginal gain and a cheap heuristic."""
    return spearman(np.asarray(marginal_gains), np.asarray(alternative))
