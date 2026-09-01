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
    "buffer_offset_id",
    "virtual_location_id",
    "session_id",
    "acquisition_session_id",
    "acquisition_block",
    "pair_id",
    "trial_pair_id",
    "pair_order",
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
    if samples is not None and (not isinstance(samples, int) or samples <= 0 or samples % 2):
        raise ConfigError("data.samples must be a positive even integer")
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
    calibration = config.get("calibration", {})
    if not isinstance(calibration, dict):
        raise ConfigError("calibration must be a mapping")
    alpha = calibration.get("alpha", 0.05)
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ConfigError("calibration.alpha must be between 0 and 1")
    protocol_version = calibration.get("protocol_version", "phase0-protocol-v1")
    if protocol_version not in {"phase0-protocol-v1", "phase0-protocol-v2"}:
        raise ConfigError("calibration.protocol_version must be phase0-protocol-v1 or v2")
    for name in [
        "samples",
        "trace_length",
        "null_replicates",
        "shuffled_replicates",
        "injected_replicates",
        "gate_validation_replicates",
        "permutation_repetitions",
    ]:
        value = calibration.get(name)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ConfigError(f"calibration.{name} must be a positive integer")
    modes = calibration.get("balance_modes", ["global_balance_only", "group_stratified_balance"])
    if (
        not isinstance(modes, list)
        or not modes
        or any(mode not in {"global_balance_only", "group_stratified_balance"} for mode in modes)
    ):
        raise ConfigError("calibration.balance_modes must name supported balance variants")
    permutation_strata = calibration.get("permutation_strata", ["synthetic_location_id"])
    if (
        not isinstance(permutation_strata, list)
        or not permutation_strata
        or not all(isinstance(key, str) for key in permutation_strata)
    ):
        raise ConfigError("calibration.permutation_strata must be a non-empty list of strings")
    by_mode = calibration.get("permutation_strata_by_balance_mode", {})
    if not isinstance(by_mode, dict) or any(
        mode not in {"global_balance_only", "group_stratified_balance"}
        or not isinstance(fields, list)
        or not fields
        or not all(isinstance(field, str) for field in fields)
        for mode, fields in by_mode.items()
    ):
        raise ConfigError("calibration.permutation_strata_by_balance_mode is invalid")
    injected_levels = calibration.get("injected_levels")
    if injected_levels is not None and (
        not isinstance(injected_levels, list)
        or not injected_levels
        or any(not isinstance(level, (int, float)) or level < 0 for level in injected_levels)
    ):
        raise ConfigError(
            "calibration.injected_levels must be a non-empty list of non-negative numbers"
        )
    power_study = calibration.get("power_study", {})
    if not isinstance(power_study, dict):
        raise ConfigError("calibration.power_study must be a mapping")
    power_counts = power_study.get("sample_counts")
    if power_counts is not None and (
        not isinstance(power_counts, list)
        or not power_counts
        or any(not isinstance(value, int) or value < 2 or value % 2 for value in power_counts)
    ):
        raise ConfigError("calibration.power_study.sample_counts must be positive even integers")
    power_replicates = power_study.get("replicates")
    if power_replicates is not None and (
        not isinstance(power_replicates, int) or power_replicates < 2
    ):
        raise ConfigError("calibration.power_study.replicates must be at least two")
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
        if cache_control not in {"none", "eviction_buffer", "clflush"}:
            raise ConfigError("phase1a.cache_control must be none, eviction_buffer, or clflush")
        operation = _get(config, "phase1a.operation", "memory_read")
        if operation not in {"memory_read", "idle"}:
            raise ConfigError("phase1a.operation must be memory_read or idle")
        location_count = _get(config, "phase1a.location_count")
        trials_per_location = _get(config, "phase1a.trials_per_location", 64)
        if location_count is not None and (
            not isinstance(location_count, int) or location_count < 1
        ):
            raise ConfigError("phase1a.location_count must be a positive integer")
        if (
            not isinstance(trials_per_location, int)
            or trials_per_location < 4
            or trials_per_location % 4
        ):
            raise ConfigError(
                "phase1a.trials_per_location must be a positive multiple of four "
                "for exact pair-order counterbalancing"
            )
        labels_per_location = _get(config, "phase1a.labels_per_location", trials_per_location // 2)
        if labels_per_location != trials_per_location // 2:
            raise ConfigError("phase1a.labels_per_location must equal half of trials_per_location")
        session_count = _get(config, "phase1a.session_count", 1)
        if not isinstance(session_count, int) or session_count < 1:
            raise ConfigError("phase1a.session_count must be a positive integer")
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
