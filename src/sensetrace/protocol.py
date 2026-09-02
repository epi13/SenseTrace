"""Versioned experimental protocols and stable protocol fingerprints."""

from __future__ import annotations

from typing import Any

from .acquisition.capabilities import (
    commodity_timing_capabilities,
    commodity_timing_oracle,
    primitive_contract,
)
from .config import normalized_config
from .hashing import sha256_json

PHASE0_PROTOCOL_VERSION = "phase0-protocol-v1"
PHASE0_PROTOCOL_V2_VERSION = "phase0-protocol-v2"
PHASE1A_COMMODITY_BASELINE_VERSION = "phase1a-commodity-baseline-v1"


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
    version = str(calibration.get("protocol_version", PHASE0_PROTOCOL_VERSION))
    if version not in {PHASE0_PROTOCOL_VERSION, PHASE0_PROTOCOL_V2_VERSION}:
        raise ValueError(f"unsupported Phase 0 protocol version: {version}")
    protocol: dict[str, Any] = {
        "version": version,
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
    if version == PHASE0_PROTOCOL_V2_VERSION:
        power_study = calibration.get("power_study", {})
        protocol["power_design"] = {
            "development_ensemble_is_separate": True,
            "sample_count_grid": power_study.get("sample_counts", []),
            "candidate_sample_count": calibration.get("samples", "unavailable"),
            "candidate_replicates": power_study.get("replicates", "unavailable"),
            "selection_rule": (
                "choose the smallest predeclared sample count whose independent development "
                "ensemble reaches the minimum injected detection rate; if none qualifies, "
                "freeze the largest candidate and keep the gate closed"
            ),
            "target_injection_strength": calibration.get(
                "target_injection_strength",
                controls.get("injected_weak_signal", {}).get("amplitude_sigma", 0.1),
            ),
        }
        protocol["calibration_rule"]["final_gate_is_single_fresh_campaign"] = True
        protocol["calibration_rule"]["power_study_is_not_gate_evidence"] = True
    return protocol


def phase0_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(phase0_protocol(config))


def phase1a_commodity_baseline_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen identity of the commodity Phase 1A comparison.

    This is intentionally an explicit contract rather than a hash of the
    entire YAML file.  It captures the decisions that determine what the
    historical observable means while excluding run IDs, timestamps, and
    other bookkeeping that should not create a new scientific protocol.
    """

    normalized = normalized_config(config)
    physical = normalized.get("phase1a", {})
    reporting = normalized.get("reporting", {})
    splits = normalized.get("splits", {}).get("primary", {})
    feature_policy = normalized.get("feature_policy", {})
    training = normalized.get("training", {})
    model_config = normalized.get("models", {})
    operation = str(physical.get("operation", "memory_read"))
    cache_control = str(physical.get("cache_control", "eviction_buffer"))
    capabilities = commodity_timing_capabilities(
        operation=operation, cache_control=cache_control
    )
    oracle = commodity_timing_oracle(operation=operation, cache_control=cache_control)
    version = str(physical.get("protocol_version", PHASE1A_COMMODITY_BASELINE_VERSION))
    if version != PHASE1A_COMMODITY_BASELINE_VERSION:
        raise ValueError(f"unsupported commodity Phase 1A protocol version: {version}")
    return {
        "version": version,
        "status": "frozen_comparison_baseline",
        "target_state_preparation": {
            "patterns": [str(physical.get("pattern", "single_bit"))],
            "target_bit": physical.get("target_bit", 0),
            "paired_base_word": "one random base word per pair; target bit forced to 0/1",
            "label_balance": "exactly one label-0 and one label-1 trial per pair",
            "pair_order": "exactly counterbalanced per virtual location and pair types randomized",
            "ordinary_read": "audit-only digital verification; excluded from model features",
            "label_source": "balanced cryptographically seeded software labels; provenance audit only",
        },
        "operation_under_test": {
            "backend": "commodity",
            "operation": operation,
            "cache_control": cache_control,
            "word_count": physical.get("word_count", 1024),
            "eviction_bytes": physical.get("eviction_bytes", 4 * 1024 * 1024),
            "trace_length": physical.get("trace_length", normalized.get("data", {}).get("trace_length", 32)),
            "native_kernel_required": bool(physical.get("require_native_kernel", True)),
            "native_timing": "LFENCE/RDTSC start and RDTSCP/LFENCE end; raw TSC cycles",
        },
        "measurement_primitive": primitive_contract(
            primitive="commodity-clflush-timed-load",
            capabilities=capabilities,
            oracle=oracle,
        ),
        "allocation_and_location": {
            "allocation": "fresh page-aligned anonymous mmap per independently started session",
            "memory_lock": {
                "requested": physical.get("lock_memory", True),
                "actual_result_recorded": True,
            },
            "location_count": physical.get("location_count", 1),
            "trials_per_location": physical.get("trials_per_location", 64),
            "labels_per_location": physical.get("labels_per_location", 32),
            "virtual_location_semantics": "controlled virtual buffer offset only; physical topology unknown",
            "physical_address": "unsupported",
            "row_bank_channel_topology": "unsupported",
        },
        "session_and_boot_boundaries": {
            "session_definition": "one backend instance, allocation, label stream, journal, and host snapshot",
            "session_count": physical.get("session_count", 1),
            "boot_id": "genuine OS boot_id recorded; not manufactured from session IDs",
            "unseen_boot_requirement": "at least three independent genuine boot groups",
            "recovery": "interrupted allocation is immutable; replacement gets new allocation and session identity",
        },
        "split_hierarchy": {
            "levels": [
                "A_repeated_trial_holdout",
                "B_unseen_location",
                "C_unseen_acquisition_block",
                "D_unseen_acquisition_session",
                "E_unseen_boot",
            ],
            "group_keys": splits.get("group_keys", [
                "virtual_location_id",
                "acquisition_block",
                "acquisition_session_id",
                "boot_id",
            ]),
            "level_grouping": {
                "A_repeated_trial_holdout": ["virtual_location_id", "trial_pair_id"],
                "B_unseen_location": ["virtual_location_id"],
                "C_unseen_acquisition_block": ["acquisition_block"],
                "D_unseen_acquisition_session": ["acquisition_session_id"],
                "E_unseen_boot": ["boot_id"],
            },
            "all_available_levels_evaluated": True,
            "e_requires_three_boot_groups": True,
        },
        "feature_policy": {
            "model_inputs": "trace-derived features only by default",
            "audit_only": sorted(
                set(
                    feature_policy.get("grouping_only_fields", [])
                    + [
                        "oracle identity and result",
                        "address/allocation/session/boot IDs",
                        "ordinary digital read value",
                    ]
                )
            ),
            "identity_features_prohibited": feature_policy.get(
                "prohibit_identity_features", True
            ),
        },
        "controls": {
            "conditions": [
                "paired_single_bit",
                "cache_hit_control",
                "idle_no_memory_operation",
                "all_zero_vs_all_one",
                "random_word_null",
                "paired_single_bit_shuffled",
            ],
            "same_observation_shuffled_labels": True,
            "pair_order_counterbalanced": True,
            "raw_traces_retained": True,
        },
        "statistics": {
            "metrics": ["balanced_accuracy", "auroc"],
            "paired_statistic": "sample median timing delta, label_1 minus label_0",
            "paired_sign_flip": True,
            "confidence_interval_unit": reporting.get(
                "ci_unit", physical.get("ci_unit", "sample")
            ),
            "model_set": (
                ["logistic_regression", "boosted_trees"]
                if not model_config
                else [
                    name
                    for name in [
                        "logistic_regression",
                        "boosted_trees",
                        "tiny_mlp",
                        "tiny_cnn",
                    ]
                    if model_config.get(name, {}).get("enabled", False)
                ]
            ),
            "training_seeds": training.get("seeds", [11, 23]),
        },
        "claim_boundaries": {
            "baseline_observable": "commodity host timing/cache-path observation",
            "not_established": [
                "physical DRAM access",
                "physical address, row, bank, channel, subarray, chip, or DIMM identity",
                "device-independent generalization",
            ],
            "scientific_principle": "increasing N under the same uncertain observable is not automatically progress",
        },
    }


def phase1a_commodity_baseline_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(phase1a_commodity_baseline_protocol(config))
