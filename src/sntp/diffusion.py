"""Information-diffusion simulation for a single item campaign.

Platform semantics
------------------
Three different events are kept apart, as Requirement 4 demands:

``recommended``  a user received a direct promotion from the platform (seed);
``exposed``      a user received the item from an adopted neighbour;
``adopted``      a user actually consumed/endorsed the item.

Adoption happens only after an exposure succeeds. A successful exposure makes the
user infectious, which is what turns a recommendation campaign into a cascade.

Two models are implemented:

* **Independent Cascade (IC)** -- every adopter gets exactly one attempt per
  outgoing influence edge, in the step after adopting. Attempts are independent
  and never repeated, so a user who ignores a friend cannot be re-convinced
  through the same edge.
* **SIR** -- a discrete-time epidemic. Every infectious user attempts to infect
  each susceptible neighbour *once per step* while still infectious, and recovers
  with probability ``recovery_prob`` per step. "Infected" means actively sharing
  the item; "recovered" means the user adopted but stopped spreading it.

The two models therefore differ in two structural ways: IC gives a single
opportunity per edge (no reinforcement), SIR allows repeated exposure but stops
propagation after a short infectious period. SIR consequently produces faster but
shallower cascades; IC produces slower cascades with a longer tail.

Exposure probabilities depend on the item-user affinity produced by the
recommender, so a neighbour's taste influences whether the message is accepted.
``influence_weight`` = 0 reduces the process to the classical homogeneous models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from .utils import seeded_rng


@dataclass
class DiffusionSettings:
    model: str = "ic"
    p0: float = 0.04
    influence_weight: float = 0.70
    affinity_floor: float = 0.15
    recovery_prob: float = 0.35
    max_steps: int = 30
    # --- interventions (Requirement 8) -------------------------------------
    exposure_cap: int | None = None
    affinity_gate: float = 0.0
    # --- platform behaviour -----------------------------------------------
    seed_acceptance: bool = True
    # Exposure bookkeeping costs one scatter-add per delivered message; it is only
    # needed when an exposure cap is active or when exposure statistics are wanted.
    track_exposures: bool = False

    @classmethod
    def from_config(cls, cfg, **overrides: Any) -> DiffusionSettings:
        section = cfg.section("diffusion")
        values = dict(
            model=str(section.get("model", "ic")),
            p0=float(section["p0"]),
            influence_weight=float(section["influence_weight"]),
            affinity_floor=float(section["affinity_floor"]),
            recovery_prob=float(section.get("recovery_prob", 0.35)),
            max_steps=int(section.get("max_steps", 30)),
        )
        values.update(overrides)
        return cls(**values)


@dataclass
class CascadeResult:
    adopted: np.ndarray
    adoption_time: np.ndarray
    exposures: np.ndarray
    n_attempts: int
    steps: int
    recommended: np.ndarray
    frontier_sizes: list[int] = field(default_factory=list)

    @property
    def reach(self) -> int:
        return int(self.adopted.sum())

    @property
    def reach_fraction(self) -> float:
        return float(self.adopted.mean())

    @property
    def seed_adoptions(self) -> int:
        return int((self.adoption_time == 0).sum())

    @property
    def mean_adoption_step(self) -> float:
        times = self.adoption_time[self.adoption_time >= 0]
        return float(times.mean()) if times.size else float("nan")

    @property
    def time_to_90pct(self) -> int:
        """First step at which 90% of the final reach has been adopted."""
        if self.reach == 0 or not self.frontier_sizes:
            return 0
        cumulative = np.cumsum(self.frontier_sizes)
        target = 0.9 * cumulative[-1]
        return int(np.searchsorted(cumulative, target))

    @property
    def peak_size(self) -> int:
        return int(max(self.frontier_sizes)) if self.frontier_sizes else 0

    def summary(self) -> dict[str, float]:
        return {
            "reach": float(self.reach),
            "reach_fraction": self.reach_fraction,
            "steps": float(self.steps),
            "time_to_90pct": float(self.time_to_90pct),
            "peak_size": float(self.peak_size),
            "mean_adoption_step": self.mean_adoption_step,
            "attempts": float(self.n_attempts),
            "seed_adoptions": float(self.seed_adoptions),
            "mean_exposures": float(self.exposures.mean()),
            "zero_exposure_users": float((self.exposures == 0).mean()),
        }


@dataclass
class SimulationSummary:
    runs: list[dict[str, float]]
    adoption_curve: np.ndarray
    seeds: np.ndarray
    model: str

    @property
    def reach_fraction_mean(self) -> float:
        return float(np.mean([run["reach_fraction"] for run in self.runs]))

    @property
    def reach_fraction_std(self) -> float:
        return (
            float(np.std([run["reach_fraction"] for run in self.runs], ddof=1))
            if len(self.runs) > 1
            else 0.0
        )

    @property
    def mean_adoption_step(self) -> float:
        values = [
            run["mean_adoption_step"] for run in self.runs if np.isfinite(run["mean_adoption_step"])
        ]
        return float(np.mean(values)) if values else float("nan")

    def aggregate(self) -> dict[str, float]:
        keys = self.runs[0].keys() if self.runs else []
        out: dict[str, float] = {}
        for key in keys:
            values = np.array([run[key] for run in self.runs], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                out[f"{key}_mean"] = float(finite.mean())
                out[f"{key}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        out["model"] = self.model  # type: ignore[assignment]
        out["n_simulations"] = float(len(self.runs))
        return out


class CampaignSimulator:
    """Runs IC/SIR cascades for one item over a fixed influence graph."""

    def __init__(
        self,
        graph: sparse.csr_matrix,
        affinity: np.ndarray,
        settings: DiffusionSettings,
    ) -> None:
        self.graph = graph.tocsr()
        self.affinity = np.clip(np.asarray(affinity, dtype=np.float64), 0.0, 1.0)
        self.settings = settings
        self.n_users = self.graph.shape[0]
        self.indptr = self.graph.indptr.astype(np.int64, copy=False)
        self.indices = self.graph.indices.astype(np.int64, copy=False)
        self.weights = self.graph.data.astype(np.float64, copy=False)
        self._base_transmission = self._compute_base()

    # --------------------------------------------------------------- internals
    def _compute_base(self) -> np.ndarray:
        """Per-target difficulty factor shared by every incoming edge."""
        s = self.settings
        affinity = self.affinity
        if s.affinity_gate > 0:
            affinity = np.where(affinity >= s.affinity_gate, affinity, 0.0)
        shaped = (s.affinity_floor + (1.0 - s.affinity_floor) * affinity) ** s.influence_weight
        return shaped

    def _transmission(self, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
        probabilities = self.settings.p0 * self._base_transmission[targets] * weights
        return np.clip(probabilities, 0.0, 0.95)

    def _neighbours(self, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ragged gather of all outgoing neighbours of ``sources`` (vectorised)."""
        starts = self.indptr[sources]
        ends = self.indptr[sources + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float64)
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        positions = np.arange(total, dtype=np.int64) + offsets
        return self.indices[positions], self.weights[positions]

    def _acceptance(self, users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.settings.seed_acceptance:
            return np.ones(users.size, dtype=bool)
        probability = np.clip(self.affinity[users], 0.05, 0.95)
        return rng.random(users.size) < probability

    # ------------------------------------------------------------------ models
    def run(self, seeds: np.ndarray, rng: np.random.Generator) -> CascadeResult:
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        if self.settings.model == "sir":
            return self._run_sir(seeds, rng)
        if self.settings.model == "ic":
            return self._run_ic(seeds, rng)
        raise ValueError(f"unknown diffusion model '{self.settings.model}'")

    def _initial_state(
        self, seeds: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        adopted = np.zeros(self.n_users, dtype=bool)
        time = np.full(self.n_users, -1, dtype=np.int32)
        accepted = seeds[self._acceptance(seeds, rng)]
        adopted[accepted] = True
        time[accepted] = 0
        return adopted, time, np.zeros(self.n_users, dtype=np.int32), accepted

    def _run_ic(self, seeds: np.ndarray, rng: np.random.Generator) -> CascadeResult:
        adopted, time, exposures, frontier = self._initial_state(seeds, rng)
        recommended = np.zeros(self.n_users, dtype=bool)
        recommended[seeds] = True
        n_attempts = 0
        steps = 0
        frontier_sizes: list[int] = [int(frontier.size)]
        cap = self.settings.exposure_cap
        track = cap is not None or self.settings.track_exposures
        saturated = np.zeros(self.n_users, dtype=bool)

        while frontier.size and steps < self.settings.max_steps:
            steps += 1
            targets, weights = self._neighbours(frontier)
            n_attempts += int(targets.size)
            if targets.size:
                if track:
                    np.add.at(exposures, targets, 1)
                if cap is not None:
                    saturated |= exposures > cap
                successes = targets[rng.random(targets.size) < self._transmission(targets, weights)]
                successes = successes[~adopted[successes]]
                if cap is not None:
                    successes = successes[~saturated[successes]]
                frontier = np.unique(successes)
                if frontier.size:
                    adopted[frontier] = True
                    time[frontier] = steps
            else:
                frontier = np.empty(0, dtype=np.int64)
            frontier_sizes.append(int(frontier.size))

        return CascadeResult(
            adopted=adopted,
            adoption_time=time,
            exposures=exposures,
            n_attempts=n_attempts,
            steps=steps,
            recommended=recommended,
            frontier_sizes=frontier_sizes,
        )

    def _run_sir(self, seeds: np.ndarray, rng: np.random.Generator) -> CascadeResult:
        adopted, time, exposures, initial = self._initial_state(seeds, rng)
        recommended = np.zeros(self.n_users, dtype=bool)
        recommended[seeds] = True
        state = np.zeros(self.n_users, dtype=np.int8)  # 0 S, 1 I, 2 R
        state[initial] = 1
        infectious = initial
        n_attempts = 0
        steps = 0
        frontier_sizes: list[int] = [int(infectious.size)]
        cap = self.settings.exposure_cap
        track = cap is not None or self.settings.track_exposures
        saturated = np.zeros(self.n_users, dtype=bool)

        while infectious.size and steps < self.settings.max_steps:
            steps += 1
            targets, weights = self._neighbours(infectious)
            n_attempts += int(targets.size)
            if targets.size:
                if track:
                    np.add.at(exposures, targets, 1)
                if cap is not None:
                    saturated |= exposures > cap
                susceptible = targets[state[targets] == 0]
                susceptible_weights = weights[state[targets] == 0]
                if susceptible.size:
                    hit = susceptible[
                        rng.random(susceptible.size)
                        < self._transmission(susceptible, susceptible_weights)
                    ]
                    hit = hit[~adopted[hit]]
                    if cap is not None:
                        hit = hit[~saturated[hit]]
                    new_infections = np.unique(hit)
                else:
                    new_infections = np.empty(0, dtype=np.int64)
            else:
                new_infections = np.empty(0, dtype=np.int64)

            # recovery of the currently infectious cohort; survivors keep spreading
            recovering = infectious[rng.random(infectious.size) < self.settings.recovery_prob]
            state[recovering] = 2
            survivors = infectious[state[infectious] == 1]

            if new_infections.size:
                state[new_infections] = 1
                adopted[new_infections] = True
                time[new_infections] = steps
            frontier_sizes.append(int(new_infections.size))
            infectious = np.concatenate([survivors, new_infections])

        return CascadeResult(
            adopted=adopted,
            adoption_time=time,
            exposures=exposures,
            n_attempts=n_attempts,
            steps=steps,
            recommended=recommended,
            frontier_sizes=frontier_sizes,
        )

    # --------------------------------------------------------------- campaigns
    def simulate(
        self, seeds: np.ndarray, n_simulations: int = 100, seed: int = 0
    ) -> SimulationSummary:
        rng = seeded_rng(seed, stream=4111)
        runs: list[dict[str, float]] = []
        curves: list[np.ndarray] = []
        max_steps = 0
        for _ in range(n_simulations):
            result = self.run(seeds, rng)
            runs.append(result.summary())
            curves.append(np.asarray(result.frontier_sizes, dtype=float))
            max_steps = max(max_steps, curves[-1].size)
        padded = np.zeros((len(curves), max_steps))
        for index, curve in enumerate(curves):
            padded[index, : curve.size] = curve
        return SimulationSummary(
            runs=runs,
            adoption_curve=padded.mean(axis=0),
            seeds=np.asarray(seeds),
            model=self.settings.model,
        )


def affinity_from_scores(scores: np.ndarray, low: float = 1.0, high: float = 5.0) -> np.ndarray:
    """Map predicted ratings to the [0, 1] affinity used by the simulator."""
    return np.clip((np.asarray(scores, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
