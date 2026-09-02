"""Characterization experiments for measurement primitives.

These experiments deliberately stop before hidden-bit inference.  They test
whether a primitive is stable, responds to a deliberately large controlled
difference, and has independently auditable access-state provenance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition.base import AcquisitionBackend
from .acquisition.native import NativeMeasurementKernel, summarize_measurements
from .acquisition.primitive import (
    CharacterizationControl,
    MeasurementPrimitive,
    create_measurement_primitive,
)
from .config import config_fingerprint
from .hashing import sha256_json
from .inventory import collect_inventory
from .journal import Journal
from .runner import _git_commit, new_run_id

CHARACTERIZATION_PROTOCOL_VERSION = "measurement-primitive-characterization-v2"

DEFAULT_NULL_STABILITY_RULE: dict[str, Any] = {
    "statistic": "median_of_replicate_sample_medians",
    "spread_statistic": "median_absolute_deviation_of_replicate_sample_medians",
    "relative_scale": "absolute_median_center",
    "max_relative_deviation": 0.25,
    "max_relative_mad": 0.10,
    "minimum_complete_replicates": 3,
    "insufficient_evidence_status": "insufficient_evidence",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _primitive_from_config(
    config: dict[str, Any], kernel: NativeMeasurementKernel | None
) -> MeasurementPrimitive:
    physical = config.get("phase1a", {})
    return create_measurement_primitive(
        str(physical.get("measurement_primitive", "commodity-clflush-timed-load")),
        kernel,
        operation=str(physical.get("operation", "memory_read")),
        cache_control=str(physical.get("cache_control", "eviction_buffer")),
        eviction=bytearray(int(physical.get("eviction_bytes", 4 * 1024 * 1024))),
    )


def characterization_protocol(
    config: dict[str, Any], primitive: MeasurementPrimitive | None = None
) -> dict[str, Any]:
    """Return the primitive-declared characterization design and claim boundary."""

    characterization = config.get("characterization", {})
    primitive = primitive or _primitive_from_config(config, None)
    contract = primitive.characterization_contract(config)
    weak_levels = [
        int(value)
        for value in characterization.get("weak_positive_control_cycles", [0, 32, 64, 128])
    ]
    return {
        "version": CHARACTERIZATION_PROTOCOL_VERSION,
        "primitive": primitive.describe(),
        "primitive_characterization_contract": contract,
        "controls": {
            control["name"]: control["description"] for control in contract.get("controls", [])
        },
        "sample_design": {
            "replicates": int(characterization.get("replicates", 3)),
            "location_count": int(characterization.get("location_count", 4)),
            "trials_per_location": int(characterization.get("trials_per_location", 16)),
            "weak_positive_control_cycles": weak_levels,
            "scoped_perf_event": characterization.get("scoped_perf_event", "not_configured"),
        },
        "analysis": {
            "no_model_training": True,
            "summary": "raw observations and primitive-declared control contrasts",
            "uncertainty": "report replicate count and descriptive spread; no hidden-bit threshold tuning",
            "null_stability": dict(
                contract.get("null_stability", DEFAULT_NULL_STABILITY_RULE)
            ),
        },
        "claim_boundary": contract["claim_boundary"],
    }


def _operation_scoped_perf_summary(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain every per-operation counter read while factoring common provenance."""

    observations: list[dict[str, Any]] = []
    for item in metadata:
        raw_value = item.get("operation_scoped_perf_observation")
        if raw_value is None:
            observations.append({"status": "not_configured"})
            continue
        try:
            parsed = json.loads(str(raw_value))
        except (TypeError, json.JSONDecodeError):
            parsed = {"status": "malformed"}
        observations.append(parsed if isinstance(parsed, dict) else {"status": "malformed"})
    configured = [item for item in observations if item.get("status") != "not_configured"]
    if not configured:
        return {"status": "not_configured", "raw_observations_retained": True}
    first = configured[0]
    readings: list[dict[str, Any]] = []
    for item in configured:
        reading = item.get("reading")
        if isinstance(reading, dict):
            readings.append(reading)
    raw_counts = [int(item["raw_count"]) for item in readings if item.get("raw_count") is not None]
    scaled_counts = [
        float(item["scaled_count"])
        for item in readings
        if item.get("scaled_count") is not None
    ]
    time_enabled = [
        int(item["time_enabled"]) for item in readings if item.get("time_enabled") is not None
    ]
    time_running = [
        int(item["time_running"]) for item in readings if item.get("time_running") is not None
    ]
    multiplexed = [bool(item["multiplexed"]) for item in readings if "multiplexed" in item]
    return {
        "status": "complete"
        if len(configured) == len(metadata)
        and len(readings) == len(configured)
        and all(item.get("status") == "complete" for item in readings)
        else "incomplete",
        "event": first.get("event"),
        "perf_event_attr": first.get("perf_event_attr"),
        "scope": first.get("scope"),
        "read_format": first.get("read_format"),
        "read_format_fields": first.get("read_format_fields"),
        "multiplexing": first.get("multiplexing"),
        "errno": first.get("errno", 0),
        "errno_name": first.get("errno_name", "none"),
        "observation_count": len(observations),
        "complete_reading_count": len(readings),
        "first_reading": readings[0] if readings else None,
        "reading": readings[0] if readings else None,
        "raw_counts": raw_counts,
        "scaled_counts": scaled_counts,
        "time_enabled": time_enabled,
        "time_running": time_running,
        "multiplexed": multiplexed,
        "raw_count_summary": summarize_measurements(np.asarray(raw_counts, dtype=np.float64))
        if raw_counts
        else None,
        "raw_readings_retained": True,
    }


def _collect_backend(backend: AcquisitionBackend) -> dict[str, Any]:
    try:
        samples = list(backend.samples())
        traces = np.asarray([sample.trace for sample in samples], dtype=np.float64)
        labels = np.asarray([sample.label for sample in samples], dtype=np.uint8)
        metadata = [sample.metadata for sample in samples]
        medians = np.median(traces, axis=1)
        order = np.arange(len(medians), dtype=np.float64)
        correlation = float("nan")
        if len(medians) > 2 and np.std(medians) > 0:
            correlation = float(np.corrcoef(order, medians)[0, 1])
        return {
            "status": "complete",
            "sample_count": int(len(samples)),
            "trace_summary": summarize_measurements(traces.reshape(-1)),
            "sample_median_summary": summarize_measurements(medians),
            "label_median_summary": {
                str(label): summarize_measurements(medians[labels == label]) for label in [0, 1]
            },
            "label_median_delta": float(
                np.median(medians[labels == 1]) - np.median(medians[labels == 0])
            ),
            "order_median_correlation": correlation,
            "session_id": str(metadata[0]["acquisition_session_id"]),
            "boot_id": str(metadata[0]["boot_id"]),
            "allocation_id": str(metadata[0]["allocation_id"]),
            "primitive": str(metadata[0]["measurement_primitive"]),
            "capabilities": json.loads(str(metadata[0]["measurement_primitive_capabilities"])),
            "access_state_oracle": json.loads(str(metadata[0]["access_state_oracle_provenance"])),
            "operation_scoped_perf": _operation_scoped_perf_summary(metadata),
            "configuration_hash": str(metadata[0].get("configuration_hash", "unavailable")),
            "code_commit": str(metadata[0].get("code_commit", "unavailable")),
            "protocol_hash": str(metadata[0].get("protocol_hash", "unavailable")),
        }
    finally:
        backend.close()


def _record_replicate_id(record: dict[str, Any]) -> str:
    """Return the explicit replicate identity used for characterization joins."""

    if record.get("replicate_id") not in {None, ""}:
        return str(record["replicate_id"])
    if record.get("replicate") not in {None, ""}:
        return f"replicate-{int(record['replicate']):04d}"
    return ""


def _complete_finite_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        replicate_id = _record_replicate_id(record)
        try:
            median = float(record["sample_median_summary"]["median"])
        except (KeyError, TypeError, ValueError):
            continue
        if replicate_id and np.isfinite(median):
            result[replicate_id] = record
    return result


def _contrast(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    expected_replicate_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Join contrast observations by explicit replicate identity."""

    left_by_id = _complete_finite_records(left)
    right_by_id = _complete_finite_records(right)
    expected = {
        str(value)
        for value in (expected_replicate_ids or set(left_by_id) | set(right_by_id))
    }
    matched_ids = sorted(set(left_by_id) & set(right_by_id) & expected)
    missing_left_ids = sorted(expected - set(left_by_id))
    missing_right_ids = sorted(expected - set(right_by_id))
    paired_differences = [
        {
            "replicate_id": replicate_id,
            "left_median": float(left_by_id[replicate_id]["sample_median_summary"]["median"]),
            "right_median": float(right_by_id[replicate_id]["sample_median_summary"]["median"]),
            "difference": float(
                right_by_id[replicate_id]["sample_median_summary"]["median"]
                - left_by_id[replicate_id]["sample_median_summary"]["median"]
            ),
        }
        for replicate_id in matched_ids
    ]
    differences = np.asarray(
        [item["difference"] for item in paired_differences], dtype=np.float64
    )
    return {
        "left_replicates": int(len(left_by_id)),
        "right_replicates": int(len(right_by_id)),
        "expected_replicates": int(len(expected)),
        "paired_replicates": int(len(matched_ids)),
        "matched_replicate_ids": matched_ids,
        "missing_left_replicate_ids": missing_left_ids,
        "missing_right_replicate_ids": missing_right_ids,
        "required_replicates_present": not missing_left_ids and not missing_right_ids,
        "paired_differences": paired_differences,
        "left_label": "left",
        "right_label": "right",
        "difference_definition": "right median sample latency minus left median sample latency",
        "difference_summary": summarize_measurements(differences),
        "positive_direction_observed": bool(len(differences) and np.median(differences) > 0),
    }


def oracle_agreement_report(expected: Iterable[str], observed: Iterable[str]) -> dict[str, Any]:
    """Return an explicit confusion structure for a categorical oracle."""

    expected_values = [str(value) for value in expected]
    observed_values = [str(value) for value in observed]
    if len(expected_values) != len(observed_values) or not expected_values:
        return {
            "status": "unavailable",
            "reason": "expected and observed oracle states are absent or have different lengths",
            "sample_count": 0,
            "confusion_matrix": {},
        }
    matrix: dict[str, dict[str, int]] = {}
    for actual, predicted in zip(expected_values, observed_values, strict=True):
        row = matrix.setdefault(actual, {})
        row[predicted] = row.get(predicted, 0) + 1
    agreements = sum(
        count
        for actual, row in matrix.items()
        for predicted, count in row.items()
        if actual == predicted
    )
    return {
        "status": "pass" if agreements == len(expected_values) else "reported",
        "sample_count": len(expected_values),
        "agreement_count": agreements,
        "agreement_rate": float(agreements / len(expected_values)),
        "confusion_matrix": matrix,
    }


def decide_characterization(evidence: dict[str, Any]) -> dict[str, Any]:
    """Make the A/B/C decision from declared evidence, never timing alone."""

    observable = bool(evidence.get("observable_response", False))
    controls_pass = bool(evidence.get("controls_pass", False))
    null_stable = bool(evidence.get("null_stable", False))
    provenance_complete = bool(evidence.get("provenance_complete", False))
    scope_acceptable = bool(evidence.get("scope_acceptable", False))
    oracle_available = bool(evidence.get("oracle_available", False))
    oracle_independent = bool(evidence.get("oracle_independent", False))
    oracle_agreement_pass = bool(evidence.get("oracle_agreement_pass", False))
    oracle_stability_pass = bool(evidence.get("oracle_stability_pass", False))
    if not observable or not controls_pass or not null_stable or not provenance_complete:
        outcome = "C_primitive_unsuitable"
        reason = "required observable, control, null-stability, or provenance gate failed"
    elif not scope_acceptable:
        outcome = "C_primitive_unsuitable"
        reason = "the primitive requires an unacceptable measurement scope or permission boundary"
    elif not oracle_available:
        outcome = "B_observable_available_but_oracle_weak"
        reason = "the observable responds to controls but no usable independent oracle is present"
    elif not oracle_independent or not oracle_agreement_pass or not oracle_stability_pass:
        outcome = "B_observable_available_but_oracle_weak"
        reason = "the oracle is present but independence, agreement, or stability is insufficient"
    else:
        outcome = "A_usable_auditable_primitive"
        reason = "observable, controls, provenance, and independent oracle gates passed"
    return {
        "outcome": outcome,
        "reason": reason,
        "observable_response": observable,
        "controls_pass": controls_pass,
        "null_stable": null_stable,
        "provenance_complete": provenance_complete,
        "scope_acceptable": scope_acceptable,
        "oracle_available": oracle_available,
        "oracle_independent": oracle_independent,
        "oracle_agreement_pass": oracle_agreement_pass,
        "oracle_stability_pass": oracle_stability_pass,
    }


def _control_from_contract(value: dict[str, Any]) -> CharacterizationControl:
    return CharacterizationControl(
        name=str(value["name"]),
        role=str(value["role"]),
        description=str(value["description"]),
        parameters=dict(value.get("parameters", {})),
    )


def _control_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [item for item in records if item.get("status") == "complete"]
    medians = np.asarray(
        [item["sample_median_summary"]["median"] for item in complete], dtype=np.float64
    )
    return {
        "records": records,
        "replicates": len(records),
        "complete_replicates": len(complete),
        "replicate_ids": [_record_replicate_id(item) for item in records],
        "replicate_medians": [
            float(item["sample_median_summary"]["median"])
            for item in complete
            if np.isfinite(float(item["sample_median_summary"]["median"]))
        ],
        "median_of_replicate_medians": float(np.median(medians)) if len(medians) else float("nan"),
        "replicate_median_std": float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0,
        "raw_values_retained": True,
    }


def _null_stability_analysis(
    records: list[dict[str, Any]],
    *,
    expected_replicates: int,
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a predeclared robust null-stability rule without filtering values."""

    complete_records = [item for item in records if item.get("status") == "complete"]
    complete_medians: list[float] = []
    for record in complete_records:
        try:
            complete_medians.append(float(record["sample_median_summary"]["median"]))
        except (KeyError, TypeError, ValueError):
            complete_medians.append(float("nan"))
    values = np.asarray(complete_medians, dtype=np.float64)
    completeness = {
        "status": "pass"
        if len(records) == expected_replicates and len(complete_records) == expected_replicates
        else "fail",
        "expected_replicates": expected_replicates,
        "observed_records": len(records),
        "complete_records": len(complete_records),
    }
    finite_validity = {
        "status": "pass"
        if len(values) == expected_replicates and bool(np.all(np.isfinite(values)))
        else "fail",
        "finite_replicate_medians": int(np.sum(np.isfinite(values))),
        "nonfinite_replicate_medians": int(np.sum(~np.isfinite(values))),
        "raw_replicate_medians": complete_medians,
    }
    minimum_replicates = int(rule.get("minimum_complete_replicates", 3))
    if completeness["status"] != "pass" or finite_validity["status"] != "pass":
        return {
            "status": "fail",
            "completeness": completeness,
            "finite_value_validity": finite_validity,
            "stability": {
                "status": "not_evaluable",
                "reason": "complete finite null replicates are required before stability can be evaluated",
            },
            "rule": rule,
        }
    if len(values) < minimum_replicates:
        return {
            "status": "insufficient_evidence",
            "completeness": completeness,
            "finite_value_validity": finite_validity,
            "stability": {
                "status": "insufficient_evidence",
                "reason": "too few complete finite null replicates for a stability statement",
                "minimum_complete_replicates": minimum_replicates,
            },
            "rule": rule,
        }

    center = float(np.median(values))
    absolute_deviations = np.abs(values - center)
    mad = float(np.median(absolute_deviations))
    denominator = max(abs(center), np.finfo(np.float64).tiny)
    relative_deviations = absolute_deviations / denominator
    relative_mad = mad / denominator
    max_relative_deviation = float(np.max(relative_deviations))
    stability_pass = bool(
        max_relative_deviation <= float(rule["max_relative_deviation"])
        and relative_mad <= float(rule["max_relative_mad"])
    )
    return {
        "status": "pass" if stability_pass else "fail",
        "completeness": completeness,
        "finite_value_validity": finite_validity,
        "stability": {
            "status": "pass" if stability_pass else "fail",
            "center": center,
            "median_absolute_deviation": mad,
            "relative_mad": relative_mad,
            "relative_deviations": [float(value) for value in relative_deviations],
            "max_relative_deviation": max_relative_deviation,
        },
        "rule": rule,
    }


def _operation_scoped_perf_median(record: dict[str, Any]) -> float | None:
    """Return one replicate's robust PMU summary when the read is complete."""

    operation = record.get("operation_scoped_perf")
    if not isinstance(operation, dict) or operation.get("status") != "complete":
        return None
    summary = operation.get("raw_count_summary")
    if not isinstance(summary, dict):
        return None
    try:
        value = float(summary["median"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _operation_scoped_perf_oracle_analysis(
    records_by_control: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
    replicates: int,
) -> dict[str, Any]:
    """Assess directional PMU agreement and null stability without hidden-bit labels."""

    required = contract.get("required_contrasts", [])
    null_name = next(
        (
            str(item["name"])
            for item in contract.get("controls", [])
            if item.get("role") == "null"
        ),
        "",
    )
    null_records = records_by_control.get(null_name, [])
    null_values = [
        value
        for record in null_records
        if (value := _operation_scoped_perf_median(record)) is not None
    ]
    null_stability_records = [
        {"status": "complete", "sample_median_summary": {"median": value}}
        for value in null_values
    ]
    null_stability = _null_stability_analysis(
        null_stability_records,
        expected_replicates=replicates,
        rule=dict(contract.get("null_stability", DEFAULT_NULL_STABILITY_RULE)),
    )
    null_stability["value_definition"] = (
        "median of all per-operation raw cache-miss counts within each null replicate"
    )
    if not required:
        return {
            "status": "unavailable",
            "reason": "no required contrast defines an expected access-state relationship",
            "null_stability": null_stability,
        }
    contrast = required[0]
    left_name = str(contrast["left_control"])
    right_name = str(contrast["right_control"])
    left_by_id = {
        _record_replicate_id(record): value
        for record in records_by_control.get(left_name, [])
        if (value := _operation_scoped_perf_median(record)) is not None
        and _record_replicate_id(record)
    }
    right_by_id = {
        _record_replicate_id(record): value
        for record in records_by_control.get(right_name, [])
        if (value := _operation_scoped_perf_median(record)) is not None
        and _record_replicate_id(record)
    }
    expected_ids = {f"replicate-{index:04d}" for index in range(replicates)}
    matched_ids = sorted(set(left_by_id) & set(right_by_id) & expected_ids)
    paired = [
        {
            "replicate_id": replicate_id,
            "left_median": left_by_id[replicate_id],
            "right_median": right_by_id[replicate_id],
            "difference": right_by_id[replicate_id] - left_by_id[replicate_id],
            "observed_relationship": (
                "right_above_left"
                if right_by_id[replicate_id] > left_by_id[replicate_id]
                else "right_not_above_left"
            ),
        }
        for replicate_id in matched_ids
    ]
    agreement_count = sum(item["observed_relationship"] == "right_above_left" for item in paired)
    agreement = {
        "status": "pass"
        if len(matched_ids) == len(expected_ids) and agreement_count == len(expected_ids)
        else "fail",
        "expected_relationship": "requested_clflush PMU median above cached PMU median",
        "matched_replicate_ids": matched_ids,
        "missing_left_replicate_ids": sorted(expected_ids - set(left_by_id)),
        "missing_right_replicate_ids": sorted(expected_ids - set(right_by_id)),
        "paired_differences": paired,
        "agreement_count": agreement_count,
        "sample_count": len(paired),
        "confusion_matrix": {
            "expected_right_above_left": {
                "observed_right_above_left": agreement_count,
                "observed_right_not_above_left": len(paired) - agreement_count,
            }
        },
    }
    return {
        "status": "complete" if paired else "unavailable",
        "left_control": left_name,
        "right_control": right_name,
        "agreement": agreement,
        "null_stability": null_stability,
        "stability_pass": null_stability["status"] == "pass",
        "raw_replicate_medians_retained": True,
    }


def _decision_evidence(
    contract: dict[str, Any],
    records_by_control: dict[str, list[dict[str, Any]]],
    contrasts: dict[str, dict[str, Any]],
    replicates: int,
) -> dict[str, Any]:
    required = contract.get("required_contrasts", [])
    contrast_ok = [
        contrasts.get(str(item["name"]), {}).get("positive_direction_observed", False)
        for item in required
    ]
    controls_pass = bool(required) and all(
        contrasts.get(str(item["name"]), {}).get("required_replicates_present", False)
        and contrasts.get(str(item["name"]), {}).get("paired_replicates", 0) == replicates
        for item in required
    )
    observable_response = bool(contrast_ok) and all(contrast_ok)
    null_controls = [
        name
        for name, records in records_by_control.items()
        if any(record.get("role") == "null" for record in records)
    ]
    null_name = (
        null_controls[0]
        if null_controls
        else next(
            (
                str(item["name"])
                for item in contract.get("controls", [])
                if item.get("role") == "null"
            ),
            "",
        )
    )
    null_records = records_by_control.get(null_name, [])
    null_stability = _null_stability_analysis(
        null_records,
        expected_replicates=replicates,
        rule=dict(contract.get("null_stability", DEFAULT_NULL_STABILITY_RULE)),
    )
    null_stable = null_stability["status"] == "pass"
    all_records = [record for records in records_by_control.values() for record in records]
    required_provenance = (
        "session_id",
        "boot_id",
        "allocation_id",
        "primitive",
        "configuration_hash",
        "code_commit",
        "protocol_hash",
    )
    provenance_complete = bool(all_records) and all(
        record.get("status") == "complete"
        and all(record.get(field) not in {None, "", "unavailable"} for field in required_provenance)
        for record in all_records
    )
    oracle_records = [
        record
        for item in required
        for record in records_by_control.get(str(item["right_control"]), [])
    ]
    oracle_values = [record.get("access_state_oracle", {}) for record in oracle_records]
    oracle_available = bool(oracle_values) and all(
        value.get("status") != "unavailable" for value in oracle_values
    )
    oracle_independent = oracle_available and all(
        bool(value.get("independent_of_latency", False)) for value in oracle_values
    )
    agreement_values = [record.get("oracle_agreement") for record in oracle_records]
    oracle_agreement_pass = (
        oracle_available
        and bool(agreement_values)
        and all(
            isinstance(value, dict) and value.get("status") == "pass" for value in agreement_values
        )
    )
    oracle_stability_pass = (
        oracle_available
        and bool(oracle_values)
        and all(bool(value.get("stable", False)) for value in oracle_values)
    )
    return {
        "observable_response": observable_response,
        "controls_pass": controls_pass,
        "null_stable": null_stable,
        "null_stability": null_stability,
        "provenance_complete": provenance_complete,
        "scope_acceptable": bool(contract.get("scope_acceptable", False)),
        "oracle_available": oracle_available,
        "oracle_independent": oracle_independent,
        "oracle_agreement_pass": oracle_agreement_pass,
        "oracle_stability_pass": oracle_stability_pass,
    }


def run_measurement_primitive_characterization(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run a small non-inference characterization campaign from a primitive contract."""

    kernel = NativeMeasurementKernel.load()
    primitive = _primitive_from_config(config, kernel)
    runtime = primitive.characterization_runtime()
    if runtime.get("status") != "available":
        return {
            "schema": "sensetrace.measurement-primitive-characterization-report.v1",
            "status": "unavailable",
            "primitive": primitive.name,
            "runtime": runtime,
            "reason": runtime.get("reason", "primitive runtime is unavailable"),
            "claim_boundary": primitive.characterization_contract(config)["claim_boundary"],
        }
    assert kernel is not None
    contract = primitive.characterization_contract(config)
    protocol = characterization_protocol(config, primitive=primitive)
    protocol_hash = sha256_json(protocol)
    design = protocol["sample_design"]
    replicates = int(design["replicates"])
    if replicates < 2:
        raise ValueError("characterization.replicates must be at least two")
    root = Path(output_root)
    run_id = run_id or new_run_id("primitive-characterization")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "run.json",
        {
            "schema": "sensetrace.run.v2",
            "run_id": run_id,
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(config),
            "protocol_hash": protocol_hash,
            "claim_scope": protocol["claim_boundary"],
        },
    )
    _write_json(run_dir / "host.json", collect_inventory())
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "protocol.json", {**protocol, "protocol_hash": protocol_hash})
    journal = Journal(run_dir / "events.jsonl")
    journal.append("characterization_started", protocol_hash=protocol_hash)

    controls = [_control_from_contract(item) for item in contract.get("controls", [])]
    records_by_control: dict[str, list[dict[str, Any]]] = {control.name: [] for control in controls}
    base_seed = int(config.get("experiment", {}).get("seed", 1337))
    for replicate in range(replicates):
        seed = base_seed + 104729 * (replicate + 1)
        replicate_id = f"replicate-{replicate:04d}"
        boot_ids: list[str] = []
        order = [int(index) for index in np.random.default_rng(seed).permutation(len(controls))]
        matched_backends: dict[str, Any] | None = None
        try:
            matched_backends = primitive.build_characterization_backends(
                config, controls, seed=seed
            )
            for acquisition_order_index, control_index in enumerate(order):
                control = controls[control_index]
                try:
                    backend = (
                        matched_backends[control.name]
                        if matched_backends is not None
                        else primitive.build_characterization_backend(config, control, seed=seed)
                    )
                    record = _collect_backend(backend)
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    record = {
                        "status": "unavailable",
                        "reason": f"control acquisition failed: {exc}",
                        "control": control.name,
                    }
                record["control"] = control.name
                record["role"] = control.role
                record["replicate"] = replicate
                record["replicate_id"] = replicate_id
                record["acquisition_order_index"] = acquisition_order_index
                record["protocol_hash"] = protocol_hash
                records_by_control[control.name].append(record)
                if record.get("boot_id"):
                    boot_ids.append(str(record["boot_id"]))
        finally:
            if matched_backends is not None:
                primitive.release_characterization_backends(matched_backends, seed=seed)
        journal.append(
            "replicate_completed",
            replicate=replicate,
            boot_ids=sorted(set(boot_ids)),
            control_order=[controls[index].name for index in order],
        )

    contrast_results: dict[str, dict[str, Any]] = {}
    for item in contract.get("required_contrasts", []):
        name = str(item["name"])
        contrast_results[name] = _contrast(
            records_by_control.get(str(item["left_control"]), []),
            records_by_control.get(str(item["right_control"]), []),
            expected_replicate_ids=[f"replicate-{index:04d}" for index in range(replicates)],
        )
    operation_scoped_perf_oracle = _operation_scoped_perf_oracle_analysis(
        records_by_control, contract, replicates
    )
    oracle_agreement_by_replicate = {
        item["replicate_id"]: {
            "status": "pass"
            if item["observed_relationship"] == "right_above_left"
            else "fail",
            "expected_relationship": "requested_clflush PMU median above cached PMU median",
            "observed_relationship": item["observed_relationship"],
            "difference": item["difference"],
        }
        for item in operation_scoped_perf_oracle.get("agreement", {}).get(
            "paired_differences", []
        )
    }
    oracle_stable = operation_scoped_perf_oracle.get("stability_pass", False)
    if operation_scoped_perf_oracle.get("status") == "complete":
        primary_contrast = contract.get("required_contrasts", [])[0]
        for record in records_by_control.get(
            str(primary_contrast["right_control"]),
            [],
        ):
            replicate_id = _record_replicate_id(record)
            record["oracle_agreement"] = oracle_agreement_by_replicate.get(
                replicate_id, {"status": "fail", "reason": "replicate not matched"}
            )
            access_oracle = record.get("access_state_oracle")
            if isinstance(access_oracle, dict):
                access_oracle["stable"] = bool(oracle_stable)
    evidence = _decision_evidence(contract, records_by_control, contrast_results, replicates)
    decision_gate = decide_characterization(evidence)
    primary_right = next(
        (
            str(item["right_control"])
            for item in contract.get("required_contrasts", [])
            if str(item["right_control"]) in records_by_control
        ),
        controls[0].name if controls else "",
    )
    primary_records = records_by_control.get(primary_right, [])
    oracle = (
        primary_records[0].get("access_state_oracle", {"status": "unavailable"})
        if primary_records
        else {"status": "unavailable"}
    )
    weak_summary: dict[str, Any] = {}
    for control in controls:
        if control.role != "weak_positive_calibration":
            continue
        cycles = int(control.parameters.get("timing_perturbation_cycles", 0))
        records = records_by_control[control.name]
        weak_summary[str(cycles)] = {
            "requested_cycles": cycles,
            "records": records,
            "median_label_delta": float(
                np.median(
                    [
                        item["label_median_delta"]
                        for item in records
                        if item.get("status") == "complete"
                    ]
                )
            )
            if any(item.get("status") == "complete" for item in records)
            else float("nan"),
            "replicates": len(records),
            "interpretation": "artificial timing control; not physical memory evidence",
        }
    session_records = records_by_control.get(primary_right, [])
    unique_sessions = sorted(
        {item["session_id"] for item in primary_records if item.get("session_id")}
    )
    unique_allocations = sorted(
        {item["allocation_id"] for item in primary_records if item.get("allocation_id")}
    )
    unique_boots = sorted({item["boot_id"] for item in primary_records if item.get("boot_id")})
    allocation_ids_by_replicate = {
        str(replicate): sorted(
            {
                record["allocation_id"]
                for records in records_by_control.values()
                for record in records
                if record.get("replicate") == replicate and record.get("allocation_id")
            }
        )
        for replicate in range(replicates)
    }
    first_required_contrast = (
        contract.get("required_contrasts", [])[0] if contract.get("required_contrasts") else None
    )
    strong_positive_control = (
        {
            "left_control": str(first_required_contrast["left_control"]),
            "right_control": str(first_required_contrast["right_control"]),
            "contrast": contrast_results[str(first_required_contrast["name"])],
        }
        if first_required_contrast is not None
        else None
    )
    oracle_agreement = {
        "status": oracle.get("status", "unavailable"),
        "strength": oracle.get("strength", "unavailable"),
        "independent_of_latency": oracle.get("independent_of_latency", False),
        "records": [record.get("oracle_agreement", {}) for record in session_records],
        "reason": (
            "no independent oracle agreement/confusion evidence was emitted"
            if not any(record.get("oracle_agreement") for record in session_records)
            else "primitive-provided oracle agreement evidence"
        ),
    }
    report = {
        "schema": "sensetrace.measurement-primitive-characterization-report.v1",
        "status": "complete",
        "run_id": run_id,
        "protocol": {**protocol, "protocol_hash": protocol_hash},
        "kernel": kernel.provenance(),
        "capabilities": primary_records[0].get("capabilities", {}) if primary_records else {},
        "controls": {
            "by_control": {
                name: _control_summary(records) for name, records in records_by_control.items()
            },
            "contrasts": contrast_results,
            **(
                {"strong_positive_control": strong_positive_control}
                if strong_positive_control is not None
                else {}
            ),
            "weak_positive_control_curve": weak_summary,
            "session_dependence": {
                "status": "descriptive_replicate_ledger",
                "unique_sessions": unique_sessions,
                "session_count": len(unique_sessions),
            },
            "allocation_dependence": {
                "status": "matched_within_replicate",
                "unique_allocations": unique_allocations,
                "allocation_count": len(unique_allocations),
                "allocation_ids_by_replicate": allocation_ids_by_replicate,
                "interpretation": "one fresh allocation is shared by controls within each replicate; allocations are fresh across replicates",
            },
            "boot_dependence": {
                "status": "measured" if len(unique_boots) >= 3 else "unavailable",
                "unique_boots": unique_boots,
                "boot_count": len(unique_boots),
                "reason": (
                    "at least three genuine OS boot groups were observed"
                    if len(unique_boots) >= 3
                    else "fewer than three genuine OS boot groups; no boot dependence claim"
                ),
            },
            "order_and_drift": {
                "per_replicate_order_correlations": [
                    item["order_median_correlation"] for item in primary_records
                ],
                "raw_values_retained": True,
            },
            "oracle_agreement": oracle_agreement,
            "operation_scoped_perf_oracle": operation_scoped_perf_oracle,
        },
        "decision_gate": {
            **decision_gate,
            "null_stability": evidence["null_stability"],
            "required_contrasts": [
                str(item["name"]) for item in contract.get("required_contrasts", [])
            ],
            "next_action": {
                "A_usable_auditable_primitive": "freeze the new protocol and run the smallest fresh validation",
                "B_observable_available_but_oracle_weak": "improve access-state instrumentation before hidden-bit inference or larger N",
                "C_primitive_unsuitable": "stop the commodity physical line and design controlled-memory-interface experimentation",
            }[decision_gate["outcome"]],
        },
        "characterization_engine": {
            "operates_on_measurement_primitive_contract": True,
            "model_training": "forbidden in this stage",
            "model_inputs": "none; oracle and provenance metadata are audit-only",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(run_dir / "metrics.json", report)
    journal.append(
        "characterization_completed",
        outcome=report["decision_gate"]["outcome"],
        oracle_strength=oracle.get("strength", "unavailable"),
    )
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_record.update({"status": "completed", "completed_at": datetime.now(UTC).isoformat()})
    _write_json(run_dir / "run.json", run_record)
    return report
