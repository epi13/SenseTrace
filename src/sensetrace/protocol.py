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
        "feature_extraction": {
            "implementation": "sensetrace.datasets.build_feature_matrix",
            "identity_metadata_excluded": True,
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
        "calibration_rule": {
            "fresh_validation_required": True,
            "minimum_injected_detection_rate": calibration.get(
                "minimum_injected_detection_rate", 0.8
            ),
            "false_positive_tolerance": "rate <= alpha + max(0.05, alpha) and Wilson lower bound <= alpha",
            "null_and_shuffled_decision": "no family-wise corrected departure from the empirical null",
            "injected_decision": "predeclared signal strength detected at the minimum configured rate",
        },
    }


def phase0_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(phase0_protocol(config))
