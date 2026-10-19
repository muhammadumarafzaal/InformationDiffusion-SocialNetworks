"""Small shared helpers: seeding, timing, JSON I/O, memory accounting."""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seeded_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Independent, reproducible generator per (seed, stream) pair."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


@dataclass
class Timer:
    """Context manager measuring wall-clock seconds and peak RSS delta."""

    label: str = ""
    seconds: float = field(default=0.0, init=False)
    peak_rss_mb: float = field(default=0.0, init=False)

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        self._rss0 = current_rss_mb()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0
        self.peak_rss_mb = max(0.0, current_rss_mb() - self._rss0)

    def as_dict(self) -> dict[str, Any]:
        return {"seconds": round(self.seconds, 4), "peak_rss_mb": round(self.peak_rss_mb, 2)}


def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def hardware_summary() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(vm.total / (1024**3), 1),
    }


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def bootstrap_ci(
    values: np.ndarray, level: float = 0.95, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = seeded_rng(seed, stream=991)
    draws = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without external dependencies."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")

    def ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size, dtype=float)
        r[order] = np.arange(1, a.size + 1, dtype=float)
        # average ties
        _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inverse, r)
        return (sums / counts)[inverse]

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


@contextmanager
def timer(label: str = ""):
    started = time.perf_counter()
    box: dict[str, float] = {}
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - started
        box["label"] = label  # type: ignore[assignment]
