# Recommendation & Information Diffusion in a Social Network

**SNTP Assignment 2 — Muhammad Umar Afzaal (23F-3106)** · umarafzaal182@gmail.com
Repository: <https://github.com/muhammadumarafzaal/InformationDiffusion-SocialNetworks>

One system that *recommends* an item, *explains itself socially*, *spreads* the item through a
trust network, *chooses whom to seed*, *serves groups*, and can be *evaluated* and *controlled*.
Everything is implemented from scratch on the public Epinions trust + rating data, driven by a
single YAML configuration, and reproducible with one command.

| | |
|---|---|
| Language / runtime | Python 3.11, [`uv`](https://docs.astral.sh/uv/) project (`uv.lock` pins every package) |
| Data | Epinions ratings (1–5) and trust edges, k-core filtered to 10,706 users / 8,945 items |
| Code | 8 experiment drivers, 8 library modules, 81 unit tests, 0 Ruff findings |
| Outputs | 30 result tables, 12 figures, a searchable PDF report |

---

## 1. Submission contents

| Path | What it is |
|---|---|
| `report/report.pdf` | **The report.** Searchable PDF, generated from the result files by `sntp run report`. |
| `README.md` | This file: environment, commands, requirement map, limitations. |
| `src/sntp/` | The package. Library modules plus one driver per requirement. |
| `tests/` | 81 unit tests covering metrics, data splitting, models, diffusion, seeding, groups, evaluation and configuration. |
| `configs/default.yaml` | Every hyper-parameter, path and seed used for the reported numbers. |
| `data/raw/epinions/` | The two public input files (`ratings_data.txt`, `trust_data.txt`). |
| `results/tables/` | 30 tables (`table_1_…` … `table_30_…`), as `.csv` and `.md`. |
| `results/raw/` | Per-run measurements the tables are aggregated from (per-user metrics, per-cascade results). |
| `results/summary/` | Machine-readable summary of each experiment (`dataset.json`, `recommenders.json`, …). |
| `results/figures/` | 12 publication-ready PNGs, all generated from the tables. |
| `results/config_snapshot.yaml` | Effective configuration of the run that produced `results/`. |
| `pyproject.toml`, `uv.lock` | Dependency list with exact resolved versions. |
| `Assignment#02.pdf` | The assignment brief (input, unchanged). |

The submission archive `23F-3106_SNTP_Assignment.zip` contains exactly this folder.

---

## 2. Environment

Tested on Windows 11 22H2, 8 logical cores, 15.7 GB RAM, Python 3.11, `uv` 0.5+.

```powershell
# 1. install uv once (any platform)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"     # macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the environment from the lock file
uv sync --frozen

# 3. smoke test: 81 tests, ~10 s
uv run pytest
```

`uv sync --frozen` reproduces the exact environment recorded in `uv.lock`; `uv sync` would
re-resolve and may pick newer patch releases.

The project installs a console script, so every experiment is reachable through `uv run sntp`.

---

## 3. Quick start (≈ 3 minutes)

A demo run on a *smaller* configuration, useful to check the installation without the full
sweep:

```powershell
uv run sntp run dataset
uv run sntp run recommenders --set recommendation.epochs=8 --set recommendation.training_seeds=[11]
uv run sntp run figures
uv run sntp run report
```

Produces `results/` and `report/report.pdf`. The numbers will differ from the submitted report
because the model is deliberately under-trained; the pipeline itself is identical.

---

## 4. Full reproduction

```powershell
uv sync --frozen
uv run sntp prepare --force        # rebuild data/processed/epinions from data/raw
uv run sntp run all                # every experiment, then figures, then the PDF
```

`run all` executes, in order:

| Step | Driver | Typical wall-clock |
|---|---|---|
| `dataset` | `experiments/rq1_dataset.py` | 5 s |
| `recommenders` | `experiments/rq2_recommenders.py` | 2.5 min |
| `social` | `experiments/rq3_social.py` | 8 min |
| `diffusion` | `experiments/rq4_diffusion.py` | 1.5 min |
| `influence` | `experiments/rq5_influence.py` | 6 min |
| `groups` | `experiments/rq6_groups.py` | 3 min |
| `evaluation` | `experiments/rq7_evaluation.py` | 2 min |
| `intervention` | `experiments/rq8_intervention.py` | 5 min |
| `figures` | `plots.py` | 10 s |
| `report` | `report.py` | 3 s |

Total ≈ 28 minutes on the reference machine, peak RSS ≈ 1.4 GB (the social sweep holds two
10,706 × 10,706 similarity matrices plus the factor matrices). Individual steps are resumable:
each experiment only needs `data/processed` and writes its own files under `results/`.

Any value can be overridden without editing the YAML:

```powershell
uv run sntp run diffusion --set diffusion.p0=0.08 --set diffusion.n_simulations=200
uv run sntp run social --config configs/default.yaml --set recommendation.social.beta=0.5
uv run sntp list
```

Every run writes `results/config_snapshot.yaml`, so a number in the report can always be traced
back to the configuration that produced it.

---

## 5. Requirement → implementation map

| # | Requirement | Where it is implemented | Where the evidence is |
|---|---|---|---|
| 1 | Social environment; observed vs created data | `src/sntp/data.py` (ingestion, iterative k-core, per-user folds, directed influence graph), `src/sntp/graph.py` (statistics, reciprocity, component size), `experiments/rq1_dataset.py` | Tables 1–3, Figure 1; report §1 |
| 2 | Two substantially different recommenders, one learning from interactions; cold start; hand-written core | `recommenders/bias_mf.py` (biased MF + Adam), `recommenders/item_knn.py` (blocked top-K cosine + shrinkage + clip), `recommenders/baselines.py`; no recommender library is used | Tables 4–6, Figures 2–3; report §2 |
| 3 | Tunable social contribution; friends vs similar-but-unconnected | `recommenders/social.py` (`SocialMF` with β, neighbour-residual model), `graph.py::user_similarity_graph`, `experiments/rq3_social.py` | Tables 7–10, Figures 4–5; report §3 |
| 4 | Independent Cascade + one epidemic model; recommendation vs exposure vs adoption; repeated simulations | `src/sntp/diffusion.py` (`CampaignSimulator`, IC and SIR, explicit `recommended` / `exposures` / `adopted`, affinity-shaped transmission) | Tables 11–13, Figures 6–7; report §4 |
| 5 | Budgeted seeding by marginal spread vs random and a network baseline; overlap and uncertainty | `src/sntp/seeds.py` (Monte-Carlo greedy with CELF, degree/preference/proxy baselines, audience overlap), `experiments/rq5_influence.py` | Tables 14–17, Figure 8; report §5 |
| 6 | Group recommendation: average / least misery / most pleasure vs realised spread | `src/sntp/groups.py`, `experiments/rq6_groups.py` | Tables 18–20, Figure 9; report §6 |
| 7 | Whole-system evaluation: error, precision, recall, ranking, coverage, diversity, diffusion, no leakage, repeated trials, a well-defined accuracy task | `src/sntp/metrics.py`, `src/sntp/evaluation.py` (candidate protocol, bootstrap, slices, adoption task), `experiments/rq7_evaluation.py` | Tables 21–25, Figure 12; report §7 |
| 8 | Intervention limiting diffusion; runtime/memory scaling; a measured optimisation | `diffusion.py` (exposure cap, affinity gate), `experiments/rq8_intervention.py` (sub-sample and synthetic scaling, two micro-benchmarks) | Tables 26–30, Figure 11; report §8 |

The report itself (`src/sntp/report.py`) is also part of the deliverable: it never hard-codes a
number, it only formats the files in `results/`.

---

## 6. How the analysis avoids fooling itself

* **Leakage.** Folds are assigned once, per user, before any model exists. Models only receive
  fold 0. Ranking candidates are the held-out items plus 100 items the user never rated *in any
  fold*, so a sampled negative cannot be a hidden positive. `experiments/rq7_evaluation.py`
  re-verifies all of this and writes the audit to `results/tables/table_21_leakage_audit.csv`.
* **No model selection on test.** β, the tuned hyper-parameters and early stopping use the
  validation fold; the test fold is touched once, at the end.
* **Same scores for ranking and error.** One score matrix produces both; no metric has a private
  path.
* **Repeated trials.** Three training seeds per recommender, 100 cascades per diffusion cell,
  20 Monte-Carlo samples per greedy marginal gain, 2,000 bootstrap resamples for intervals.
* **Accuracy only where it is defined.** Accuracy is reported for one explicit binary task
  (will this user adopt an unobserved item?) next to its no-skill baseline, never for ranking.
* **Figures from tables.** `plots.py` only reads the CSVs, so a figure cannot contradict a table.

### Diffusion semantics

| Event | Meaning |
|---|---|
| *recommended* | the platform pushed the item directly (seed) |
| *exposed* | a trusted neighbour who adopted passed the item on |
| *adopted* | the user actually consumed the item |

Transmission is `p = p0 · (floor + (1 − floor)·affinity)^γ · w`, where `affinity` is the
recommender's normalised predicted rating, `γ` (`influence_weight`) tunes how much preference
shapes the cascade, and `w` is the edge weight. `γ = 0` recovers the classical homogeneous
models. IC gives each adopter one attempt per edge; SIR retries while infectious and recovers
with probability 0.35 per step.

---

## 7. Repository layout

```
.
├── Assignment#02.pdf
├── configs/default.yaml          # every hyper-parameter, path and seed
├── data/
│   ├── raw/epinions/             # ratings_data.txt, trust_data.txt (input)
│   └── processed/epinions/       # cached core + folds (regenerable, git-ignored)
├── report/report.pdf             # generated report
├── results/
│   ├── cache/                    # fitted-model caches reused across experiments
│   ├── figures/  raw/  summary/  tables/
│   └── config_snapshot.yaml
├── src/sntp/
│   ├── cli.py                    # `sntp run <experiment|all>`, `prepare`, `list`
│   ├── config.py  data.py  graph.py  utils.py
│   ├── metrics.py  evaluation.py # metrics + leakage-safe protocol
│   ├── diffusion.py  seeds.py  groups.py
│   ├── recommenders/
│   │   ├── base.py  baselines.py  bias_mf.py  item_knn.py  social.py
│   ├── experiments/
│   │   ├── common.py  rq1_dataset.py … rq8_intervention.py
│   ├── plots.py                  # 12 figures from results/
│   └── report.py                 # searchable PDF from results/
├── tests/                        # pytest suite
├── pyproject.toml  uv.lock
└── README.md
```

---

## 8. Tests

```powershell
uv run pytest              # 81 tests, ~10 s
uv run ruff check src tests    # lint
uv run ruff format --check src tests
```

The suite does not touch the Epinions files: it builds an 80-user, 40-item synthetic network
with latent taste clusters and verifies the properties the report relies on — metric formulas by
hand, fold partitions, leakage-free candidate sets, "bias model beats global mean", "β = 0 is an
exact control", "zero transmission leaves only the seeds", "an exposure cap cannot increase
reach", "least misery protects the worst-off member".

---

## 9. Configuration

`configs/default.yaml` is the single source of truth. The entries that matter most:

| Key | Value | Why |
|---|---|---|
| `dataset.min_user_ratings` / `min_item_ratings` | 10 / 10 | iterative k-core; a tighter core (20/20) leaves 2,677 users, too small for diffusion |
| `dataset.split.{val,test}_fraction` | 0.10 / 0.10 | per-user hold-out (Epinions has no timestamps) |
| `recommendation.n_factors`, `learning_rate`, `reg`, `epochs` | 32, 0.002, 0.2, 40 | selected on the validation fold |
| `recommendation.knn_neighbors`, `knn_shrinkage`, `knn_max_correction` | 20, 10.0, 0.5 | idem |
| `recommendation.training_seeds` | 11, 23, 37 | repeated trials |
| `recommendation.social.beta`, `betas`, `graphs` | 0.25, 0…2, trust/similarity/random | Requirement 3 sweep |
| `diffusion.p0`, `influence_weight`, `recovery_prob`, `n_simulations` | 0.04, 0.70, 0.35, 100 | default campaign |
| `influence.budgets`, `mc_samples_selection`, `mc_samples_final` | 5/10/20/40, 20, 200 | Requirement 5 budget sweep |
| `groups.size`, `n_groups`, `n_simulations` | 5, 60, 50 | Requirement 6 |
| `intervention.caps`, `gates` | 1/2/3/∞, 0/0.35/0.5 | Requirement 8 |

Seeds: `seed: 20261019` drives the folds; training seeds are 11/23/37; diffusion streams are
derived from the same base seed through `utils.seeded_rng`, so every experiment is deterministic
and independent.

---

## 10. Limitations

* **No timestamps.** Epinions as distributed here has no review dates, so the hold-out is random
  per user. A temporal split would be harder and more realistic.
* **Binary trust.** Every published trust record has weight +1; distrust is not observable, so
  the influence graph cannot be signed.
* **Small core.** The k-core filter keeps 10,706 of the published users. That is large enough for
  stable recommender metrics but small for cascade statistics: a reach of a few hundred users is
  a meaningful sample, not a population estimate.
* **Simulation, not experiment.** IC and SIR are models. They are calibrated to the observed
  graph and to recommender affinity, not fitted to a real campaign, so absolute reach is not a
  prediction of Epinions behaviour — the *comparisons* between strategies are the result.
* **Single machine.** Timings characterise this machine; the scaling shapes should transfer, the
  constants will not.
* **One dataset.** No cross-dataset validation was attempted.

---

## 11. Data

The two input files under `data/raw/epinions/` are the public Epinions release used in the
assignment brief: `ratings_data.txt` (`user item rating`, 1–5) and `trust_data.txt`
(`source target +1`, meaning *source trusts target*). They are included so the pipeline runs
without network access. The processed core is regenerable with `uv run sntp prepare --force`
and is not committed.

Influence is assumed to flow **against** the stored trust direction: if *u* trusts *v*, then *v*
can influence *u*, so the influence graph contains the edge `v → u`.

---

## 12. Academic integrity

All code in `src/` and `tests/` was written for this assignment. NumPy, SciPy, pandas,
Matplotlib and ReportLab are used for linear algebra, sparse storage, tabulation, plotting and
PDF rendering; **no recommender-system or network-diffusion library is used** — the models,
metrics, protocol and simulators are implemented here.
