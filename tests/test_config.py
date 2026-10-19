"""Configuration loading, dotted access and CLI overrides."""

from __future__ import annotations

import pytest

from sntp.config import Config, parse_overrides


def test_dotted_lookup_and_missing_keys():
    config = Config({"a": {"b": {"c": 3}}, "top": 1})
    assert config["a.b.c"] == 3
    assert config["top"] == 1
    assert config.get("a.b.missing") is None
    assert "a.b" in config
    with pytest.raises(KeyError):
        _ = config["nope"]


def test_section_requires_a_mapping():
    config = Config({"a": {"b": 1}, "c": 2})
    assert isinstance(config.section("a"), Config)
    with pytest.raises(TypeError):
        config.section("c")


def test_overrides_are_applied_without_touching_the_defaults():
    base = Config.load()
    changed = Config.load(None, overrides={"recommendation.evaluation.top_k": 5})
    assert base["recommendation.evaluation.top_k"] == 10
    assert changed["recommendation.evaluation.top_k"] == 5
    assert changed["dataset.min_user_ratings"] == base["dataset.min_user_ratings"]


def test_parse_overrides_infers_types():
    parsed = parse_overrides(["a.b=3", "c=text", "d=true", "e=1.5"])
    assert parsed == {"a.b": 3, "c": "text", "d": True, "e": 1.5}


def test_parse_overrides_rejects_malformed_pairs():
    with pytest.raises(ValueError):
        parse_overrides(["missing_equals_sign"])


def test_paths_are_resolved_against_the_repository_root():
    config = Config.load()
    results = config.path("paths.results_dir")
    processed = config.path("paths.processed_dir")
    assert results.is_absolute() and processed.is_absolute()
    # results/ and data/processed/ both live directly under the repository root
    assert results.parent == processed.parent.parent


def test_config_is_deep_copied():
    source = {"a": {"b": 1}}
    config = Config(source)
    source["a"]["b"] = 99
    assert config["a.b"] == 1


def test_required_sections_exist():
    config = Config.load()
    for section in (
        "dataset",
        "recommendation",
        "diffusion",
        "influence",
        "groups",
        "intervention",
        "cost",
        "report",
    ):
        assert isinstance(config.section(section), Config)
