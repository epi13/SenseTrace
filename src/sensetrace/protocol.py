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
WORKER03_FRAGMENTED_EXACT_HOST_VERSION = "worker03-fragmented-exact-host-v1"


def worker03_fragmented_exact_host_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Return the preregistered worker-03 fragmented-evidence contract.

    The contract is deliberately explicit even when a caller accepts the
    defaults.  Requested excitation is recorded here; executed excitation is
    an acquisition result and can never be inferred from this record.
    """

    normalized = normalized_config(config)
    experiment = normalized.get("worker03_experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("worker03_experiment must be a mapping")
    version = str(experiment.get("protocol_version", WORKER03_FRAGMENTED_EXACT_HOST_VERSION))
    if version != WORKER03_FRAGMENTED_EXACT_HOST_VERSION:
        raise ValueError(f"unsupported worker-03 fragmented protocol: {version}")
    data = normalized.get("data", {})
    receiver = experiment.get("receiver", {})
    probes = experiment.get(
        "probes",
        [
            {"probe_type": "cached_control", "probe_version": "native-v4"},
            {"probe_type": "dependency_chain", "probe_version": "native-v4"},
            {"probe_type": "repeated_load", "probe_version": "native-v4"},
            {"probe_type": "paired_cached_differential", "probe_version": "native-v4"},
        ],
    )
    if not isinstance(probes, list) or not probes:
        raise ValueError("worker03_experiment.probes must be a non-empty list")
    protocol: dict[str, Any] = {
        "version": version,
        "status": "preregistered_frozen_protocol",
        "dataset_purpose": "worker03_fragmented_exact_host_native",
        "target": {
            "hardware_identity": experiment.get("target_hardware_id", "worker03-hardware-v1"),
            "inventory_match_required": experiment.get("inventory_match", "matched"),
            "host_scope": "exact worker-03 host only; no portability claim",
            "native_evidence_only": True,
            "controlled_hardware_evidence": False,
        },
        "implementation": {
            "code_commit": experiment.get("code_commit", "resolved_at_freeze_time"),
            "native_kernel_version": experiment.get("native_kernel_version", "native-v4"),
            "probe_versions": probes,
            "timing_method": "native v4 TSC cycles; LFENCE/RDTSC and RDTSCP/LFENCE where available",
            "warmup": experiment.get(
                "warmup",
                {"enabled": True, "touch_pages": True, "dummy_loads": 0, "label_independent": True},
            ),
        },
        "packet_composition": {
            "fragment_order": experiment.get(
                "fragment_order", [str(item.get("probe_type")) for item in probes]
            ),
            "target_reference_relationship": experiment.get(
                "target_reference_relationship", "same target packet with fixed reference roles"
            ),
            "preserve_raw_fragments": True,
            "missing_fragment_behavior": "retain explicit status and mask; never impute as observed zero",
            "failure_states": ["unavailable", "failed", "partial", "corrupted"],
            "model_arrays": ["values", "observed_mask", "fragment_mask", "excitation", "quality"],
        },
        "target_and_label_generation": {
            "requested_schedule": experiment.get(
                "requested_schedule", {"family": "active_quiet", "length": 32, "seed": 1337}
            ),
            "label_generation": experiment.get(
                "label_generation", "balanced deterministic seeded labels; audit only"
            ),
            "reference_baseline_labels": "none; reference corpus is unlabeled",
            "digital_read": "audit-only and excluded from model arrays",
        },
        "core_roles": experiment.get(
            "core_roles",
            {
                "measurement": [0],
                "reference": [1],
                "excitation": [2, 3, 4, 5, 6],
                "orchestrator": [7],
            },
        ),
        "execution": {
            "cpu_affinity_expectation": experiment.get(
                "cpu_affinity_expectation", "role cores pinned and scheduler status recorded"
            ),
            "requested_vs_executed_separate": True,
            "noncompliant_execution": "retain as noncompliant control or fail the requested evidence gate",
            "interruptions_recorded": True,
            "timing_uncertainty_recorded": True,
        },
        "acquisition": {
            "repetition_counts": experiment.get(
                "repetition_counts", {"per_probe": 8, "packets": int(data.get("samples", 64))}
            ),
            "discard_rules": experiment.get(
                "discard_rules",
                ["nonfinite payload", "missing required target fragment", "noncompliant execution"],
            ),
            "session_definition": "one boot, host inventory, fresh allocation, label stream, and append-only packet source",
            "interruption": "finalize immutable prefix; replacement session gets new identity and parent reference",
            "streaming_memory_bound": True,
        },
        "reference_baseline": {
            "required": True,
            "unlabeled": True,
            "repetition_counts": experiment.get(
                "reference_repetition_counts",
                {"packets": int(experiment.get("reference_packets", 1024))},
            ),
            "drift_capture": "long reference acquisition across its declared duration and sessions",
            "residualizer": {
                "fit_corpus": "reference dataset only",
                "validation_test_excluded": True,
                "source_fingerprint_recorded": True,
            },
        },
        "splits": {
            "policy": "materialized immutable packet-id split; groups never cross partitions",
            "claim_levels": [
                "level_1_exact_host_calibrated",
                "level_2_exact_host_unseen_location",
                "level_3_exact_host_unseen_session",
                "level_4_exact_host_unseen_boot",
                "level_5_unseen_dimm",
                "level_6_unseen_host",
            ],
            "level_4_minimum_boot_groups": 3,
            "level_5_minimum_dimm_groups": 2,
            "level_6_minimum_host_groups": 2,
            "leakage_fields": [
                "packet_id",
                "run_id",
                "acquisition_id",
                "acquisition_session_id",
                "session_id",
                "boot_id",
                "target_reference",
                "address",
                "location_id",
                "virtual_location_id",
                "fragment_id",
                "sequence_position",
                "schedule_index",
                "excitation_phase",
            ],
        },
        "receivers": {
            "candidates": receiver.get(
                "candidates",
                [
                    "logistic_regression",
                    "boosted_trees",
                    "tiny_cnn_tcn",
                    "weak_evidence_aggregator",
                    "jepa_linear_probe",
                    "jepa_tiny_mlp",
                    "predictive_coding",
                    "jepa_predictive_coding_hybrid",
                ],
            ),
            "configuration_space": receiver.get(
                "configuration_space", "small frozen bounded space"
            ),
            "selection": "fit train; select on validation; evaluate selected configuration on test once",
            "test_requery": "forbidden",
        },
        "metrics": experiment.get(
            "metrics",
            ["balanced_accuracy", "auroc", "sample_count", "class_balance", "confidence_interval"],
        ),
        "controls": {
            "positive_control": "artificial injected contrast; control-only label",
            "null": "no-signal condition",
            "shuffled_labels": "frozen seed and procedure",
            "relation_ablation": "destroy cross-fragment relation while preserving marginal fragments where possible",
            "metadata_firewall": "audit metadata cannot change model arrays",
            "single_fragment_baseline": "strongest individual fragment compared with aggregate",
            "session_boot_generalization": "reported only when independent groups exist",
        },
        "state_machine": [
            "planned",
            "protocol_frozen",
            "inventory_verified",
            "reference_acquisition",
            "controlled_acquisition",
            "evidence_finalized",
            "split_frozen",
            "training",
            "validation_selection",
            "test_evaluation",
            "decision",
        ],
        "stop_rules": experiment.get(
            "stop_rules",
            [
                "stop on protocol divergence requiring a new experiment identity",
                "stop on invalid or noncompliant execution rather than silently relabeling it",
                "stop on provenance, duplicate, or leakage audit failure",
                "stop claim level at the strongest level supported by independent evidence",
            ],
        ),
        "claim_boundary": "native exact-host fragmented observations only; no controlled-memory, FPGA, DRAM-origin, hidden-state, or generalization claim",
        "historical_gate": "commodity Phase 1A remains C: primitive unsuitable and is not reopened by this protocol",
    }
    return protocol


def worker03_fragmented_exact_host_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(worker03_fragmented_exact_host_protocol(config))


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
    capabilities = commodity_timing_capabilities(operation=operation, cache_control=cache_control)
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
            "trace_length": physical.get(
                "trace_length", normalized.get("data", {}).get("trace_length", 32)
            ),
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
            "group_keys": splits.get(
                "group_keys",
                [
                    "virtual_location_id",
                    "acquisition_block",
                    "acquisition_session_id",
                    "boot_id",
                ],
            ),
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
            "identity_features_prohibited": feature_policy.get("prohibit_identity_features", True),
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
        "artificial_timing_perturbation": {
            "allowed": False,
            "timing_perturbation_cycles": 0,
            "label_correlated": False,
            "calibration_namespace": "forbidden",
            "enforcement": (
                "physical Phase 1A rejects nonzero artificial timing cycles, non-default "
                "perturbation labels, and calibration namespaces before acquisition"
            ),
        },
        "statistics": {
            "metrics": ["balanced_accuracy", "auroc"],
            "paired_statistic": "sample median timing delta, label_1 minus label_0",
            "paired_sign_flip": True,
            "confidence_interval_unit": reporting.get("ci_unit", physical.get("ci_unit", "sample")),
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
