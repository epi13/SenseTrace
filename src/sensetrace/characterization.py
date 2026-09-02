"""Characterization experiments for measurement primitives.

These experiments deliberately stop before hidden-bit inference.  They test
whether a primitive is stable, responds to a deliberately large controlled
difference, and has independently auditable access-state provenance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition.commodity import CommodityDramBackend
from .acquisition.native import NativeMeasurementKernel, summarize_measurements
from .config import config_fingerprint
from .hashing import sha256_json
from .inventory import collect_inventory
from .journal import Journal
from .protocol import phase1a_commodity_baseline_protocol
from .runner import _git_commit, new_run_id

CHARACTERIZATION_PROTOCOL_VERSION = "measurement-primitive-characterization-v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def characterization_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Return the predeclared characterization design and its claim boundary."""

    characterization = config.get("characterization", {})
    weak_levels = [
        int(value)
        for value in characterization.get("weak_positive_control_cycles", [0, 32, 64, 128])
    ]
    return {
        "version": CHARACTERIZATION_PROTOCOL_VERSION,
        "primitive_protocol": phase1a_commodity_baseline_protocol(config),
        "controls": {
            "null": "repeated identical controlled-state acquisition with no artificial delay",
            "strong_positive": "cached control versus requested CLFLUSH control",
            "weak_positive": "predeclared artificial post-load TSC delay; instrumentation control only",
            "session_dependence": "independent backend/allocation per replicate",
            "boot_dependence": "record genuine boot_id; no synthetic boot groups",
            "order_drift": "audit correlation of trace summaries with acquisition order",
            "allocation_dependence": "record fresh allocation identity per backend",
            "oracle_agreement": "compare only when an independent oracle exists; unavailable is a valid result",
        },
        "sample_design": {
            "replicates": int(characterization.get("replicates", 3)),
            "location_count": int(characterization.get("location_count", 4)),
            "trials_per_location": int(characterization.get("trials_per_location", 16)),
            "weak_positive_control_cycles": weak_levels,
        },
        "analysis": {
            "no_model_training": True,
            "summary": "raw timing distributions and label-independent control contrasts",
            "uncertainty": "report replicate count and descriptive spread; no hidden-bit threshold tuning",
        },
        "claim_boundary": (
            "primitive characterization only; a cache-path contrast or artificial delay does not "
            "establish physical DRAM access, row activation, or hidden-bit information"
        ),
    }


def _backend(
    config: dict[str, Any],
    *,
    cache_control: str,
    seed: int,
    timing_perturbation_cycles: int = 0,
) -> CommodityDramBackend:
    physical = config.get("phase1a", {})
    characterization = config.get("characterization", {})
    location_count = int(characterization.get("location_count", 4))
    trials = int(characterization.get("trials_per_location", 16))
    return CommodityDramBackend(
        count=location_count * trials,
        trace_length=int(characterization.get("trace_length", physical.get("trace_length", 32))),
        seed=seed,
        pattern=str(physical.get("pattern", "single_bit")),
        target_bit=int(physical.get("target_bit", 0)),
        word_count=int(physical.get("word_count", 1024)),
        lock_memory=bool(physical.get("lock_memory", False)),
        cache_control=cache_control,
        operation="memory_read",
        measurement_primitive=str(
            physical.get("measurement_primitive", "commodity-clflush-timed-load")
        ),
        eviction_bytes=int(physical.get("eviction_bytes", 4 * 1024 * 1024)),
        location_count=location_count,
        trials_per_location=trials,
        labels_per_location=trials // 2,
        use_native_kernel=True,
        timing_perturbation_cycles=timing_perturbation_cycles,
        acquisition_session_id=f"characterization-session-{seed:010d}",
        session_index=0,
        campaign_id="measurement-primitive-characterization",
        host_inventory_snapshot=collect_inventory(),
        code_commit=_git_commit(),
        configuration_hash=config_fingerprint(config),
    )


def _collect_backend(backend: CommodityDramBackend) -> dict[str, Any]:
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
                str(label): summarize_measurements(medians[labels == label])
                for label in [0, 1]
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
            "access_state_oracle": json.loads(
                str(metadata[0]["access_state_oracle_provenance"])
            ),
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
    differences = right_values - left_values
    return {
        "left_replicates": int(len(left_values)),
        "right_replicates": int(len(right_values)),
        "paired_replicates": int(min(len(left_values), len(right_values))),
        "left_label": "left",
        "right_label": "right",
        "difference_definition": "right median sample latency minus left median sample latency",
        "difference_summary": summarize_measurements(differences),
        "positive_direction_observed": bool(len(differences) and np.median(differences) > 0),
    }


def run_measurement_primitive_characterization(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run a small non-inference characterization campaign."""

    kernel = NativeMeasurementKernel.load()
    if kernel is None or not kernel.supports_clflush:
        return {
            "schema": "sensetrace.measurement-primitive-characterization-report.v1",
            "status": "unavailable",
            "reason": "native x86 CLFLUSH path is unavailable; strong control cannot be run",
            "claim_boundary": "no primitive characterization evidence collected",
        }
    protocol = characterization_protocol(config)
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

    null: list[dict[str, Any]] = []
    cached: list[dict[str, Any]] = []
    flushed: list[dict[str, Any]] = []
    weak: dict[str, list[dict[str, Any]]] = {
        str(cycles): [] for cycles in design["weak_positive_control_cycles"]
    }
    base_seed = int(config.get("experiment", {}).get("seed", 1337))
    for replicate in range(replicates):
        seed = base_seed + 104729 * (replicate + 1)
        null.append(_collect_backend(_backend(config, cache_control="none", seed=seed)))
        cached.append(_collect_backend(_backend(config, cache_control="none", seed=seed + 17)))
        flushed.append(_collect_backend(_backend(config, cache_control="clflush", seed=seed + 31)))
        for cycles in design["weak_positive_control_cycles"]:
            weak[str(cycles)].append(
                _collect_backend(
                    _backend(
                        config,
                        cache_control="clflush",
                        seed=seed + 1000 + int(cycles),
                        timing_perturbation_cycles=int(cycles),
                    )
                )
            )
        journal.append("replicate_completed", replicate=replicate, boot_id=null[-1]["boot_id"])

    oracle = flushed[0]["access_state_oracle"] if flushed else {"status": "unavailable"}
    strong_contrast = _contrast(cached, flushed)
    weak_summary = {
        cycles: {
            "requested_cycles": int(cycles),
            "records": records,
            "median_label_delta": float(
                np.median([item["label_median_delta"] for item in records])
            ),
            "replicates": len(records),
            "interpretation": "artificial timing control; not physical memory evidence",
        }
        for cycles, records in weak.items()
    }
    report = {
        "schema": "sensetrace.measurement-primitive-characterization-report.v1",
        "status": "complete",
        "run_id": run_id,
        "protocol": {**protocol, "protocol_hash": protocol_hash},
        "kernel": kernel.provenance(),
        "capabilities": flushed[0]["capabilities"] if flushed else {},
        "controls": {
            "null_behavior": {
                "records": null,
                "replicates": len(null),
                "median_of_replicate_medians": float(
                    np.median([item["sample_median_summary"]["median"] for item in null])
                ),
                "replicate_median_std": float(
                    np.std([item["sample_median_summary"]["median"] for item in null], ddof=1)
                )
                if len(null) > 1
                else 0.0,
            },
            "strong_positive_control": {
                "cached": cached,
                "requested_clflush": flushed,
                "contrast": strong_contrast,
            },
            "weak_positive_control_curve": weak_summary,
            "session_dependence": {
                "unique_sessions": sorted({item["session_id"] for item in flushed}),
                "unique_allocations": sorted({item["allocation_id"] for item in flushed}),
                "unique_boots": sorted({item["boot_id"] for item in flushed}),
            },
            "order_and_drift": {
                "per_replicate_order_correlations": [
                    item["order_median_correlation"] for item in flushed
                ],
                "raw_values_retained": True,
            },
            "oracle_agreement": {
                "status": oracle.get("status", "unavailable"),
                "strength": oracle.get("strength", "unavailable"),
                "independent_of_latency": oracle.get("independent_of_latency", False),
                "reason": "commodity primitive has no independent cache-level or DRAM-event oracle",
            },
        },
        "decision_gate": {
            "strong_control_observed": bool(strong_contrast["positive_direction_observed"]),
            "oracle_is_meaningfully_independent": False,
            "outcome": "B_observable_available_but_oracle_weak",
            "next_action": "improve access-state instrumentation before hidden-bit inference or larger N",
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
