"""Versioned experimental protocols and stable protocol fingerprints."""

from __future__ import annotations

from typing import Any

from .config import normalized_config
from .hashing import sha256_json

PHASE0_PROTOCOL_VERSION = "phase0-protocol-v1"


def phase0_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen Phase 0 decisions represented by a configuration."""

    normalized = normalized_config(config)
    calibration = normalized.get("calibration", {})
    data = normalized.get("data", {})
    acquisition = normalized.get("acquisition", {})
    controls = normalized.get("controls", {})
    splits = normalized.get("splits", {}).get("primary", {})
    training = normalized.get("training", {})
    models = normalized.get("models", {})
    return {
        "version": PHASE0_PROTOCOL_VERSION,
        "label_construction": {
            "target_balance": data.get("target_balance", 0.5),
            "balance_mode": acquisition.get("synthetic_balance_mode", "global_balance_only"),
            "observations_per_location": acquisition.get("observations_per_location", 4),
            "label_seed_is_independent": True,
        },
        "trace_generation": {
            "backend": acquisition.get("backend", "synthetic"),
            "trace_length": data.get("trace_length", 128),
            "injection": controls.get("injected_weak_signal", {}),
            "calibration_injected_levels": calibration.get(
                "injected_levels", controls.get("injected_weak_signal", {}).get("levels", [])
            ),
            "trace_seed_is_independent": True,
        },
        "grouping": {
            "group_keys": splits.get(
                "group_keys",
                ["synthetic_dataset_id", "synthetic_session_id", "synthetic_location_id"],
            ),
            "split_policy": splits.get("strategy", "grouped"),
            "split_seed_is_independent": True,
        },
        "models": {
            "enabled": [name for name, value in models.items() if value.get("enabled", False)],
            "training_seeds": training.get("seeds", [11, 23, 37]),
            "model_seed_is_independent": True,
        },
        "statistics": {
            "metrics": ["balanced_accuracy", "auroc"],
            "chance": 0.5,
            "alpha": calibration.get("alpha", 0.05),
            "multiple_comparison": "empirical maximum statistic across enabled model/metric runs",
            "permutation_scheme": calibration.get(
                "permutation_strata_by_balance_mode",
                {
                    "global_balance_only": ["synthetic_dataset_id"],
                    "group_stratified_balance": ["synthetic_location_id"],
                },
            ),
        },
    }


def phase0_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(phase0_protocol(config))
