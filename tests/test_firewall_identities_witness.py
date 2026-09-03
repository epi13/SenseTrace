"""Firewall hardening: witness runtime, controlled identities, pilot scheduler."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from sensetrace.acquisition.controlled import (
    ControlledAcquisitionProvenance,
    ControlledTraceAcquisition,
    ControlledTraceChannel,
)
from sensetrace.config import validate_config
from sensetrace.witness.observer import BpftraceWitnessObserver
from sensetrace.witness.program import (
    observed_tids_from_program,
    render_bpftrace_program,
    render_bpftrace_program_for_targets,
)


def _characterization_config(witness_requirement: str) -> dict:
    return validate_config(
        {
            "experiment": {"name": "witness-runtime", "seed": 7},
            "data": {"target_balance": 0.5, "samples": 8, "trace_length": 8},
            "acquisition": {"backend": "commodity"},
            "phase1a": {
                "protocol_version": "phase1a-commodity-baseline-v1",
                "measurement_primitive": "commodity-clflush-timed-load",
                "location_count": 1,
                "trials_per_location": 4,
                "labels_per_location": 2,
                "word_count": 8,
                "lock_memory": False,
                "cache_control": "clflush",
                "eviction_bytes": 1024,
                "use_native_kernel": True,
                "require_native_kernel": True,
            },
            "characterization": {
                "replicates": 2,
                "location_count": 1,
                "trials_per_location": 4,
                "trace_length": 8,
                "weak_positive_control_cycles": [0],
            },
            "witness": {"requirement": witness_requirement, "hooks": []},
        }
    )


def test_witness_required_is_rejected_at_characterization_runtime(tmp_path):
    import sensetrace.characterization as characterization

    config = _characterization_config("required")
    with pytest.raises(RuntimeError, match="witness.requirement='required'"):
        characterization.run_measurement_primitive_characterization(
            config, tmp_path, run_id="witness-required"
        )
    assert not (tmp_path / "witness-required").exists()


def test_witness_disabled_and_optional_do_not_collect_a_session(tmp_path, monkeypatch):
    import sensetrace.characterization as characterization

    class FakeKernel:
        supports_clflush = True

        def measure_cached(self, address, repetitions, extra_delay_cycles=0):
            return np.zeros(repetitions, dtype=np.float64)

        def measure_flushed(self, address, repetitions, extra_delay_cycles=0):
            return np.full(repetitions, 200.0, dtype=np.float64)

        def provenance(self):
            return {
                "implementation": "fake",
                "version": "fake-v1",
                "library": "fake",
                "library_sha256": "f" * 64,
                "timer_source": "fake",
                "clflush_supported": True,
                "raw_units": "fake",
                "guarantees": [],
                "limitations": ["fake kernel"],
            }

    monkeypatch.setattr(
        characterization.NativeMeasurementKernel, "load", classmethod(lambda cls: FakeKernel())
    )
    for requirement in ("disabled", "optional"):
        report = characterization.run_measurement_primitive_characterization(
            _characterization_config(requirement), tmp_path, run_id=f"witness-{requirement}"
        )
        assert report["status"] == "complete"
        assert report["witness_evidence"]["requirement"] == requirement
        assert report["witness_evidence"]["collection"] == "not_collected"
        assert report["witness_evidence"]["session"] is None


def _valid_channel(**overrides) -> ControlledTraceChannel:
    fields = {
        "channel_id": "adc-0",
        "channel_kind": "analog",
        "units": "volts",
        "sampling_clock_id": "clock-1",
        "calibration_id": "cal-1",
    }
    fields.update(overrides)
    return ControlledTraceChannel(**fields)  # type: ignore[arg-type]


def _valid_acquisition(**overrides) -> ControlledTraceAcquisition:
    fields: dict = {
        "acquisition_id": "acq-1",
        "trigger_id": "trigger-1",
        "trigger_hardware_ticks": 1,
        "hardware_clock_id": "clock-1",
        "timing_uncertainty_ticks": 1,
        "channels": (_valid_channel(),),
        "refresh_relationship": "controlled schedule r1",
        "command_sequence_id": "sequence-1",
    }
    fields.update(overrides)
    return ControlledTraceAcquisition(**fields)


@pytest.mark.parametrize(
    "field,value",
    [
        ("calibration_id", "unavailable"),
        ("calibration_id", "unknown"),
        ("calibration_id", ""),
        ("sampling_clock_id", "unavailable"),
        ("sampling_clock_id", "unknown"),
        ("sampling_clock_id", ""),
        ("channel_id", "unavailable"),
        ("channel_id", "unknown"),
        ("units", "unavailable"),
    ],
)
def test_controlled_channel_rejects_placeholder_identities(field, value):
    with pytest.raises(ValueError, match="missing"):
        _valid_channel(**{field: value}).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("trigger_id", "unavailable"),
        ("trigger_id", "unknown"),
        ("trigger_id", ""),
        ("hardware_clock_id", "unavailable"),
        ("hardware_clock_id", "unknown"),
        ("acquisition_id", "unavailable"),
        ("acquisition_id", "unknown"),
        ("command_sequence_id", "unavailable"),
        ("command_sequence_id", "unknown"),
        ("refresh_relationship", "unavailable"),
        ("refresh_relationship", "unknown"),
    ],
)
def test_controlled_acquisition_rejects_placeholder_identities(field, value):
    with pytest.raises(ValueError, match="missing"):
        _valid_acquisition(**{field: value}).validate()


def test_controlled_valid_identities_still_validate():
    _valid_channel().validate()
    _valid_acquisition().validate()
    provenance = ControlledAcquisitionProvenance(
        experiment_target_id="target-1",
        controller_firmware_id="firmware-1",
        controller_config_hash="config-hash-1",
        device_identity="device-1",
        dimm_identity="dimm-1",
        calibration_state="calibration-1",
        hardware_clock_id="clock-1",
        acquisition_trigger="command trigger",
        acquisition_configuration_hash="acquisition-hash-1",
        trigger_identity="trigger-1",
        timing_provenance="controller clock clock-1",
        refresh_relationship="after refresh 42",
        command_provenance="command log 1",
    )
    provenance.validate()


def test_multi_tid_program_covers_scheduler_and_memory_hooks():
    source = render_bpftrace_program_for_targets(
        "session-1", ("context_switch", "cpu_migration", "page_fault"), target_tids=(111, 222)
    )
    assert "args->prev_pid == 111" in source
    assert "args->next_pid == 222" in source
    assert "args->pid == 111" in source
    assert "args->pid == 222" in source
    assert "tid == 111" in source
    assert "tid == 222" in source
    assert observed_tids_from_program(source) == (111, 222)
    legacy = render_bpftrace_program("session-1", ("context_switch",))
    assert "$2" in legacy


def test_observer_records_multiple_watched_tids(tmp_path):
    observer = BpftraceWitnessObserver(
        session_id="observer-1",
        experiment_id="experiment-1",
        target_pid=1000,
        target_tid=111,
        target_tids=(111, 222),
        output_dir=tmp_path,
        requested_hooks=("context_switch",),
    )
    assert observer.target_tids == (111, 222)
    with pytest.raises(ValueError, match="positive thread IDs"):
        BpftraceWitnessObserver(
            session_id="observer-1",
            experiment_id="experiment-1",
            target_pid=1000,
            target_tid=111,
            target_tids=(),
            output_dir=tmp_path,
        )


def test_positive_control_runs_on_watched_worker_tid():
    import sensetrace.witness.pilot as pilot

    stop = threading.Event()
    tid_ready = threading.Event()
    churn_start = threading.Event()
    tid_box: list[int] = []
    worker = threading.Thread(
        target=pilot._positive_control, args=(stop, tid_ready, churn_start, tid_box), daemon=True
    )
    worker.start()
    try:
        assert tid_ready.wait(timeout=5)
        worker_tid = int(tid_box[0])
        assert worker_tid > 0
        assert worker_tid != threading.get_native_id()
        # Baseline phase: worker is idle until churn starts.
        assert not churn_start.is_set()
        source = render_bpftrace_program_for_targets(
            "session-test",
            ("context_switch", "page_fault"),
            target_tids=(threading.get_native_id(), worker_tid),
        )
        covered = observed_tids_from_program(source)
        assert worker_tid in covered
        assert threading.get_native_id() in covered
    finally:
        churn_start.set()
        stop.set()
        worker.join(timeout=5)
