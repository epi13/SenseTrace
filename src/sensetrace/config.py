"""Validated YAML configuration and reproducibility hashes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .hashing import sha256_json

DEFAULT_GROUP_KEYS = ["session_id", "device_id", "row_id", "cell_or_offset_id"]
DEFAULT_IDENTITY_FIELDS = [
    "host_id",
    "device_id",
    "bank_id",
    "row_id",
    "cell_or_offset_id",
    "session_id",
    "trial_index",
    "sample_id",
]


def _get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ConfigError("experiment.name is required")
    samples = _get(config, "data.samples")
    if samples is not None and (not isinstance(samples, int) or samples <= 0):
        raise ConfigError("data.samples must be a positive integer")
    fractions = [
        _get(config, "splits.primary.train_fraction", 0.7),
        _get(config, "splits.primary.validation_fraction", 0.15),
        _get(config, "splits.primary.test_fraction", 0.15),
    ]
    if any(not isinstance(item, (int, float)) or item <= 0 for item in fractions):
        raise ConfigError("split fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ConfigError("split fractions must sum to 1.0")
    balance = _get(config, "data.target_balance", _get(config, "acquisition.target_balance", 0.5))
    if balance != 0.5:
        raise ConfigError("Phase 0 requires exact balanced binary labels (target_balance=0.5)")
    trace_length = _get(config, "data.trace_length", 128)
    if not isinstance(trace_length, int) or trace_length < 8:
        raise ConfigError("data.trace_length must be an integer >= 8")
    group_keys = _get(config, "splits.primary.group_keys", DEFAULT_GROUP_KEYS)
    if not isinstance(group_keys, list) or not all(isinstance(key, str) for key in group_keys):
        raise ConfigError("splits.primary.group_keys must be a list of strings")
    shard_mb = _get(config, "acquisition.shard_target_mb", 512)
    if not isinstance(shard_mb, (int, float)) or shard_mb <= 0:
        raise ConfigError("acquisition.shard_target_mb must be positive")
    seeds = _get(config, "training.seeds", [11, 23, 37])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ConfigError("training.seeds must be a non-empty list of integers")
    backend = _get(config, "acquisition.backend", "synthetic")
    if backend not in {"synthetic", "commodity"}:
        raise ConfigError("acquisition.backend must be synthetic or commodity")
    ci_unit = _get(config, "reporting.ci_unit", "session_id")
    if not isinstance(ci_unit, str) or (ci_unit != "sample" and not ci_unit):
        raise ConfigError("reporting.ci_unit must be sample or a metadata grouping field")
    if backend == "commodity":
        pattern = _get(config, "phase1a.pattern", "single_bit")
        if pattern not in {"all_zero_one", "single_bit", "random_word"}:
            raise ConfigError("phase1a.pattern is not a supported safe memory pattern")
        cache_control = _get(config, "phase1a.cache_control", "eviction_buffer")
        if cache_control not in {"none", "eviction_buffer"}:
            raise ConfigError("phase1a.cache_control must be none or eviction_buffer")
        operation = _get(config, "phase1a.operation", "memory_read")
        if operation not in {"memory_read", "idle"}:
            raise ConfigError("phase1a.operation must be memory_read or idle")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    return validate_config(config)


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy safe for stable hashing and artifact serialization."""

    value = deepcopy(config)
    value.setdefault("splits", {}).setdefault("primary", {}).setdefault(
        "group_keys", DEFAULT_GROUP_KEYS
    )
    value.setdefault("feature_policy", {}).setdefault(
        "grouping_only_fields", DEFAULT_IDENTITY_FIELDS
    )
    value.setdefault("feature_policy", {}).setdefault("prohibit_identity_features", True)
    return value


def config_fingerprint(config: dict[str, Any]) -> str:
    return sha256_json(normalized_config(config))
