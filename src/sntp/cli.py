"""Command line entry point.

Examples
--------
    uv run sntp prepare                 # build data/processed/epinions
    uv run sntp run dataset             # Requirement 1
    uv run sntp run recommenders        # Requirements 2 and 7 (models)
    uv run sntp run all                 # every experiment, figures and the PDF
    uv run sntp run diffusion --set diffusion.p0=0.08
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from .config import Config, parse_overrides

EXPERIMENTS = (
    "dataset",
    "recommenders",
    "social",
    "diffusion",
    "influence",
    "groups",
    "evaluation",
    "intervention",
    "figures",
    "report",
)
PIPELINE = (
    "dataset",
    "recommenders",
    "social",
    "diffusion",
    "influence",
    "groups",
    "evaluation",
    "intervention",
    "figures",
    "report",
)


def _dispatch(name: str):
    """Import lazily so a single experiment never pulls the whole package."""
    if name == "figures":
        from .plots import run as figures

        return figures
    if name == "report":
        from .report import run as report

        return report
    if name == "dataset":
        from .experiments.rq1_dataset import run

        return run
    if name == "recommenders":
        from .experiments.rq2_recommenders import run

        return run
    if name == "social":
        from .experiments.rq3_social import run

        return run
    if name == "diffusion":
        from .experiments.rq4_diffusion import run

        return run
    if name == "influence":
        from .experiments.rq5_influence import run

        return run
    if name == "groups":
        from .experiments.rq6_groups import run

        return run
    if name == "evaluation":
        from .experiments.rq7_evaluation import run

        return run
    if name == "intervention":
        from .experiments.rq8_intervention import run

        return run
    raise KeyError(name)


def _run_one(name: str, cfg: Config) -> None:
    started = time.perf_counter()
    print(f"\n=== {name} " + "=" * (60 - len(name)))
    _dispatch(name)(cfg)
    print(f"=== {name} finished in {time.perf_counter() - started:.1f}s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sntp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one experiment or the whole pipeline")
    run_parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    run_parser.add_argument("--config", default=None, help="alternative YAML config")
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override, e.g. --set diffusion.n_simulations=20",
    )

    prepare = sub.add_parser("prepare", help="build the processed dataset")
    prepare.add_argument("--config", default=None)
    prepare.add_argument("--force", action="store_true", help="rebuild from the raw files")

    sub.add_parser("list", help="list the available experiments")

    args = parser.parse_args(argv)

    if args.command == "list":
        for name in EXPERIMENTS:
            print(name)
        return 0

    cfg = Config.load(args.config, parse_overrides(getattr(args, "set", [])))

    if args.command == "prepare":
        from .data import prepare_dataset

        dataset = prepare_dataset(cfg, force=args.force)
        core = dataset.metadata["core"]
        print(
            f"prepared {dataset.name}: {core['n_users']} users, {core['n_items']} items, "
            f"{core['n_ratings']} ratings, {core['trust_edges']} trust edges"
        )
        return 0

    names = PIPELINE if args.experiment == "all" else (args.experiment,)
    for name in names:
        _run_one(name, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
