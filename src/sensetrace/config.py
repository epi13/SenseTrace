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
    "allocation_id",
    "physical_allocation_id",
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
    if backend not in {"synthetic", "commodity", "controlled_mock", "controlled_hardware"}:
        raise ConfigError(
            "acquisition.backend must be synthetic, commodity, controlled_mock, or "
            "controlled_hardware"
        )
    ci_unit = _get(config, "reporting.ci_unit", "session_id")
    if not isinstance(ci_unit, str) or (ci_unit != "sample" and not ci_unit):
        raise ConfigError("reporting.ci_unit must be sample or a metadata grouping field")
    if backend == "commodity":
        campaign_intent = _get(config, "phase1a.campaign_intent", "historical_reproduction")
        if campaign_intent not in {
            "historical_reproduction",
            "measurement_characterization",
            "current_scaling",
        }:
            raise ConfigError(
                "phase1a.campaign_intent must be historical_reproduction, "
                "measurement_characterization, or current_scaling"
            )
        if campaign_intent == "current_scaling":
            raise ConfigError(
                "current commodity scaling is closed by the frozen C_primitive_unsuitable gate; "
                "use controlled_mock for Phase 2 progression"
            )
        protocol_version = _get(config, "phase1a.protocol_version", "phase1a-commodity-baseline-v1")
        if protocol_version != "phase1a-commodity-baseline-v1":
            raise ConfigError("phase1a.protocol_version must be phase1a-commodity-baseline-v1")
        primitive = _get(config, "phase1a.measurement_primitive", "commodity-clflush-timed-load")
        if primitive != "commodity-clflush-timed-load":
            raise ConfigError("phase1a.measurement_primitive must be commodity-clflush-timed-load")
        pattern = _get(config, "phase1a.pattern", "single_bit")
        if pattern not in {"all_zero_one", "single_bit", "random_word"}:
            raise ConfigError("phase1a.pattern is not a supported safe memory pattern")
        cache_control = _get(config, "phase1a.cache_control", "eviction_buffer")
        if cache_control not in {"none", "eviction_buffer", "clflush"}:
            raise ConfigError("phase1a.cache_control must be none, eviction_buffer, or clflush")
        operation = _get(config, "phase1a.operation", "memory_read")
        if operation not in {"memory_read", "idle"}:
            raise ConfigError("phase1a.operation must be memory_read or idle")
        perturbation_cycles = _get(config, "phase1a.timing_perturbation_cycles", 0)
        if not isinstance(perturbation_cycles, int) or perturbation_cycles < 0:
            raise ConfigError("phase1a.timing_perturbation_cycles must be a non-negative integer")
        if perturbation_cycles != 0:
            raise ConfigError(
                "phase1a.timing_perturbation_cycles is calibration-only and must be zero in "
                "physical configurations"
            )
        perturbation_label = _get(config, "phase1a.timing_perturbation_label", 1)
        if perturbation_label != 1:
            raise ConfigError(
                "phase1a.timing_perturbation_label must remain the default in physical Phase 1A"
            )
        if "calibration_namespace" in config.get("phase1a", {}):
            raise ConfigError(
                "phase1a.calibration_namespace is calibration-only and cannot appear in physical "
                "Phase 1A"
            )
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
    if backend == "controlled_mock":
        phase2 = config.get("phase2", {})
        if not isinstance(phase2, dict):
            raise ConfigError("phase2 must be a mapping for controlled_mock acquisition")
        mock = phase2.get("controlled_mock", {})
        if not isinstance(mock, dict):
            raise ConfigError("phase2.controlled_mock must be a mapping")
        protocol = mock.get("protocol_version", "controlled-memory-interface-mock-v1")
        if protocol != "controlled-memory-interface-mock-v1":
            raise ConfigError(
                "phase2.controlled_mock.protocol_version must be "
                "controlled-memory-interface-mock-v1"
            )
        for name in ("count", "trace_length"):
            value = mock.get(name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ConfigError(f"phase2.controlled_mock.{name} must be a positive integer")
        if mock.get("count", samples or 64) % 2:
            raise ConfigError("phase2.controlled_mock.count must be even")
        for name in ("target_id", "firmware_id"):
            value = mock.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"phase2.controlled_mock.{name} must be a non-empty string")
        if "topology" in mock and mock["topology"] != "unavailable":
            raise ConfigError("controlled_mock topology is always unavailable")
    if backend == "controlled_hardware":
        boundary = config.get("phase2", {}).get("controlled_hardware", {})
        if not isinstance(boundary, dict):
            raise ConfigError("phase2.controlled_hardware must be a mapping")
        adapter = boundary.get("adapter")
        if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
            raise ConfigError("phase2.controlled_hardware.adapter must be a non-empty string")
    native_sensitivity = config.get("native_sensitivity", {})
    if native_sensitivity and not isinstance(native_sensitivity, dict):
        raise ConfigError("native_sensitivity must be a mapping")
    for name in [
        "development_replicates",
        "development_null_replicates",
        "development_shuffled_replicates",
        "validation_replicates",
        "minimum_recommended_null_replicates",
    ]:
        value = native_sensitivity.get(name) if isinstance(native_sensitivity, dict) else None
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ConfigError(f"native_sensitivity.{name} must be a positive integer")
    characterization = config.get("characterization", {})
    if characterization and not isinstance(characterization, dict):
        raise ConfigError("characterization must be a mapping")
    if isinstance(characterization, dict):
        replicates = characterization.get("replicates", 3)
        if not isinstance(replicates, int) or replicates < 2:
            raise ConfigError("characterization.replicates must be an integer >= 2")
        location_count = characterization.get("location_count", 4)
        trials = characterization.get("trials_per_location", 16)
        if not isinstance(location_count, int) or location_count < 1:
            raise ConfigError("characterization.location_count must be positive")
        if not isinstance(trials, int) or trials < 4 or trials % 4:
            raise ConfigError(
                "characterization.trials_per_location must be a positive multiple of four"
            )
        weak_levels = characterization.get("weak_positive_control_cycles", [0, 32, 64, 128])
        if (
            not isinstance(weak_levels, list)
            or not weak_levels
            or any(not isinstance(value, int) or value < 0 for value in weak_levels)
            or 0 not in weak_levels
        ):
            raise ConfigError(
                "characterization.weak_positive_control_cycles must include non-negative zero"
            )
        null_stability = characterization.get("null_stability", {})
        if not isinstance(null_stability, dict):
            raise ConfigError("characterization.null_stability must be a mapping")
        for name, default in {
            "max_relative_deviation": 0.25,
            "max_relative_mad": 0.10,
        }.items():
            value = null_stability.get(name, default)
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"characterization.null_stability.{name} must be non-negative")
        minimum_replicates = null_stability.get("minimum_complete_replicates", 3)
        if not isinstance(minimum_replicates, int) or minimum_replicates < 3:
            raise ConfigError(
                "characterization.null_stability.minimum_complete_replicates must be >= 3"
            )
        warmup = characterization.get("allocation_warmup", None)
        if warmup is not None:
            if not isinstance(warmup, dict):
                raise ConfigError("characterization.allocation_warmup must be a mapping")
            unknown = set(warmup) - {"enabled", "touch_pages", "dummy_loads"}
            if unknown:
                raise ConfigError(
                    f"characterization.allocation_warmup has unknown fields {sorted(unknown)}"
                )
            dummy = warmup.get("dummy_loads", 0)
            if not isinstance(dummy, int) or dummy < 0 or dummy > 10_000:
                raise ConfigError(
                    "characterization.allocation_warmup.dummy_loads must be in [0, 10000]"
                )
            if warmup.get("enabled", False) and (
                not warmup.get("touch_pages", True) and dummy == 0
            ):
                raise ConfigError(
                    "characterization.allocation_warmup.enabled requires touch_pages or dummy_loads"
                )
    witness = config.get("witness", None)
    if witness is not None:
        if not isinstance(witness, dict):
            raise ConfigError("witness must be a mapping")
        if witness.get("requirement", "optional") not in {"disabled", "optional", "required"}:
            raise ConfigError("witness.requirement must be disabled, optional, or required")
        hooks = witness.get("hooks", [])
        if not isinstance(hooks, list) or not all(isinstance(item, str) for item in hooks):
            raise ConfigError("witness.hooks must be a list of strings")
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
