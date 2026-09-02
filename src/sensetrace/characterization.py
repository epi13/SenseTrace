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

CHARACTERIZATION_PROTOCOL_VERSION = "measurement-primitive-characterization-v1"


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
        },
        "analysis": {
            "no_model_training": True,
            "summary": "raw observations and primitive-declared control contrasts",
            "uncertainty": "report replicate count and descriptive spread; no hidden-bit threshold tuning",
        },
        "claim_boundary": contract["claim_boundary"],
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
            "configuration_hash": str(metadata[0].get("configuration_hash", "unavailable")),
            "code_commit": str(metadata[0].get("code_commit", "unavailable")),
            "protocol_hash": str(metadata[0].get("protocol_hash", "unavailable")),
        }
    finally:
        backend.close()


def _contrast(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_values = np.asarray(
        [item["sample_median_summary"]["median"] for item in left if item["status"] == "complete"],
        dtype=np.float64,
    )
    right_values = np.asarray(
        [item["sample_median_summary"]["median"] for item in right if item["status"] == "complete"],
        dtype=np.float64,
    )
    paired_count = min(len(left_values), len(right_values))
    differences = right_values[:paired_count] - left_values[:paired_count]
    return {
        "left_replicates": int(len(left_values)),
        "right_replicates": int(len(right_values)),
        "paired_replicates": int(paired_count),
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
        "median_of_replicate_medians": float(np.median(medians)) if len(medians) else float("nan"),
        "replicate_median_std": float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0,
        "raw_values_retained": True,
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
        contrasts.get(str(item["name"]), {}).get("paired_replicates", 0) == replicates
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
    null_medians = [
        item["sample_median_summary"]["median"]
        for item in null_records
        if item.get("status") == "complete"
    ]
    null_stable = (
        len(null_records) == replicates
        and len(null_medians) == replicates
        and all(np.isfinite(null_medians))
    )
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
        )
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
        },
        "decision_gate": {
            **decision_gate,
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
