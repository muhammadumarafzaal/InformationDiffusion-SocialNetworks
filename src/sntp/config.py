"""Configuration loading with dotted access and reproducible snapshots."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class Config(Mapping[str, Any]):
    """Read-only nested configuration with ``cfg["a.b.c"]`` access."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = copy.deepcopy(data)

    # ------------------------------------------------------------------ mapping
    def __getitem__(self, key: str) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(key)
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        try:
            self[str(key)]
        except KeyError:
            return False
        return True

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, indent=2)[:400]})"

    # ---------------------------------------------------------------- accessors
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        try:
            return self[key]
        except KeyError:
            return default

    def section(self, name: str) -> Config:
        value = self[name]
        if not isinstance(value, Config):
            raise TypeError(f"section '{name}' is not a mapping")
        return value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        value = Path(str(self[key]))
        return value if value.is_absolute() else PROJECT_ROOT / value

    def save(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self._data, sort_keys=False), encoding="utf-8")

    # ------------------------------------------------------------------- loader
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Config:
        source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if overrides:
            for key, value in overrides.items():
                _set_dotted(data, key, value)
        return cls(data)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=3", "c=text"]`` into typed values where possible."""
    result: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override '{pair}' must look like key=value")
        key, raw = pair.split("=", 1)
        try:
            value: Any = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        result[key.strip()] = value
    return result
