"""Adversarial regression tests for PMU hardening, characterization gates,
multi-boot combination, and the Phase 2 boundary."""

from __future__ import annotations

import json
import struct

import pytest

from sensetrace.acquisition.controlled import (
    ControlledAcquisitionProvenance,
    ControlledCommand,
    ControlledCommandResult,
    ControlledMemoryTopology,
    ControlledTraceAcquisition,
    ControlledTraceChannel,
    SyntheticMockControlledBackend,
)
from sensetrace.acquisition.perf import (
    OperationScopedPerfEvent,
    PerfEventEncoding,
    _encode_sysfs_config,
    build_perf_event_attr,
    classify_pmu_scope,
    discover_counter_capabilities,
    expected_read_size,
    select_sysfs_event_encoding,
)
from sensetrace.characterization import (
    _null_stability_analysis,
    _operation_scoped_perf_oracle_analysis,
    _operation_scoped_perf_summary,
)
from sensetrace.multiboot import (
    MULTIBOOT_CANDIDATE_EVENT,
    combine_multiboot_reports,
    multiboot_protocol,
)


def _observation(
    raw_count: int,
    *,
    status: str = "complete",
    multiplexed: bool = False,
    qualified: str = "cpu/cache-misses/",
    attr_config: int = 0x412E,
) -> str:
    reading_status = (
        "complete"
        if status == "complete" and not multiplexed
        else ("complete" if status == "complete" else status)
    )
    time_running = 10 if status == "complete" else 0
    return json.dumps(
        {
            "status": status,
            "event": {"qualified_name": qualified},
            "perf_event_attr": {"type": 4, "config": attr_config},
            "scope": {"kind": "calling_thread", "cpu_argument": -1},
            "reading": {
                "status": reading_status,
                "raw_count": raw_count,
                "scaled_count": float(raw_count),
                "time_enabled": 10,
                "time_running": time_running,
                "multiplexed": multiplexed,
            },
        }
    )


def test_expected_read_size_follows_read_format_bits():
    assert expected_read_size(0) == 8
    assert expected_read_size(1) == 16
    assert expected_read_size(2) == 16
    assert expected_read_size(3) == 24


def test_unsupported_perf_read_format_fails_closed():
    with pytest.raises(ValueError, match="unsupported perf read_format"):
        expected_read_size(1 << 2)  # PERF_FORMAT_ID
    with pytest.raises(ValueError, match="unsupported perf read_format"):
        build_perf_event_attr("cache-misses", read_format=1 << 3)  # PERF_FORMAT_GROUP


def test_read_honors_reduced_read_format(monkeypatch):
    import sensetrace.acquisition.perf as perf

    monkeypatch.setattr(perf, "_perf_event_open", lambda *a, **k: 101)
    monkeypatch.setattr(perf.fcntl, "ioctl", lambda *a, **k: 0)
    monkeypatch.setattr(perf.fcntl, "fcntl", lambda *a, **k: 0)
    monkeypatch.setattr(perf.os, "read", lambda fd, size: struct.pack("<Q", 9)[:size])
    closed: list[int] = []
    monkeypatch.setattr(perf.os, "close", lambda fd: closed.append(fd))
    reader = OperationScopedPerfEvent("cache-misses", syscall_number=298, read_format=0)
    reader.open()
    reading = reader.read()
    assert reading.raw_count == 9
    assert (reading.time_enabled, reading.time_running) == (0, 0)
    reader.close()
    assert closed == [101]


def test_close_clears_identity_when_os_close_fails(monkeypatch):
    import sensetrace.acquisition.perf as perf

    monkeypatch.setattr(perf, "_perf_event_open", lambda *a, **k: 102)
    monkeypatch.setattr(perf.fcntl, "ioctl", lambda *a, **k: 0)
    monkeypatch.setattr(perf.fcntl, "fcntl", lambda *a, **k: 0)

    def boom(fd):
        raise OSError("close failed")

    monkeypatch.setattr(perf.os, "close", boom)
    reader = OperationScopedPerfEvent("cache-misses", syscall_number=298)
    reader.open()
    reader.close()
    assert reader._fd is None
    assert reader._enabled is False


def test_uncore_encoding_cannot_back_thread_scoped_reader():
    encoding = PerfEventEncoding(
        device="uncore_imc",
        alias="cas_count_read",
        source_type=14,
        config=0x04,
        config_fields={"event": "0x04"},
        format_fields={"event": "config:0-7"},
        raw_spec="event=0x04",
    )
    with pytest.raises(ValueError, match="cannot back a calling-thread scoped reader"):
        OperationScopedPerfEvent(encoding, syscall_number=298)


@pytest.mark.parametrize("device", ["cpu", "cpu_core", "cpu_atom", "armv8_pmuv3_0"])
def test_architecture_specific_core_pmu_scope_is_accepted(device):
    classification = classify_pmu_scope(device)
    assert classification["status"] == "accepted"
    encoding = PerfEventEncoding(
        device=device,
        alias="cache-misses",
        source_type=4,
        config=0x2E,
        config_fields={"event": "0x2e"},
        format_fields={"event": "config:0-7"},
        raw_spec="event=0x2e",
    )
    assert (
        OperationScopedPerfEvent(encoding, syscall_number=298).provenance["scope"][
            "pmu_classification"
        ]["scope"]
        == "thread_scoped_core_pmu"
    )


def test_unknown_pmu_scope_is_rejected():
    assert classify_pmu_scope("mystery_pmu")["status"] == "rejected"


def test_single_bit_format_field_encodes():
    assert _encode_sysfs_config({"event": "0x1"}, {"event": "config:0"}) == 1
    assert _encode_sysfs_config({"event": "0x0"}, {"event": "config:0"}) == 0


def test_format_fields_are_not_shared_between_encodings(tmp_path):
    events = tmp_path / "bus" / "event_source" / "devices" / "cpu" / "events"
    events.mkdir(parents=True)
    (events.parent / "type").write_text("4\n")
    format_dir = events.parent / "format"
    format_dir.mkdir()
    (format_dir / "event").write_text("config:0-7\n")
    (events / "a").write_text("event=0x01\n")
    (events / "b").write_text("event=0x02\n")
    first = select_sysfs_event_encoding(tmp_path, "cpu/a/")
    second = select_sysfs_event_encoding(tmp_path, "cpu/b/")
    assert first.format_fields is not second.format_fields
    first.format_fields["mutated"] = "yes"
    assert "mutated" not in second.format_fields


def test_unreadable_event_file_does_not_drop_sibling(tmp_path, monkeypatch):
    import sensetrace.acquisition.perf as perf

    events = tmp_path / "bus" / "event_source" / "devices" / "cpu" / "events"
    events.mkdir(parents=True)
    (events.parent / "type").write_text("4\n")
    format_dir = events.parent / "format"
    format_dir.mkdir()
    (format_dir / "event").write_text("config:0-7\n")
    (events / "good").write_text("event=0x01\n")
    (events / "bad").write_text("event=0x02\n")
    real_read_text = perf.Path.read_text

    def flaky(self, *args, **kwargs):
        if self.name == "bad":
            raise OSError("unreadable")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(perf.Path, "read_text", flaky)
    selected = select_sysfs_event_encoding(tmp_path, "cpu/good/")
    assert selected.config == 0x01


def test_bare_unknown_event_rejected_at_construction():
    from sensetrace.acquisition.primitive import CommodityTimingPrimitive

    primitive = CommodityTimingPrimitive(
        None, operation="memory_read", cache_control="none", eviction=bytearray(64)
    )
    config = {
        "phase1a": {"cpu_affinity": None},
        "characterization": {
            "location_count": 1,
            "trials_per_location": 4,
            "scoped_perf_event": "mem_load_retired.llc_miss",
        },
    }
    from sensetrace.acquisition.primitive import CharacterizationControl

    control = CharacterizationControl(name="x", role="null", description="x")
    with pytest.raises(ValueError, match="no generic encoding"):
        primitive._build_characterization_backend(config, control, seed=1)


def test_cpu_affinity_forwarded_to_characterization_backend(monkeypatch):
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    applied: list = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, mask: applied.append(set(mask)))
    from sensetrace.acquisition.primitive import (
        CharacterizationControl,
        CommodityTimingPrimitive,
    )

    primitive = CommodityTimingPrimitive(
        None, operation="memory_read", cache_control="none", eviction=bytearray(64)
    )
    config = {
        "phase1a": {"cpu_affinity": [0]},
        "characterization": {"location_count": 1, "trials_per_location": 4},
    }
    backend = primitive._build_characterization_backend(
        config, CharacterizationControl(name="x", role="null", description="x"), seed=1
    )
    try:
        assert backend.cpu_affinity == [0]
    finally:
        backend.close()


def test_heterogeneous_provenance_is_explicit():
    summary = _operation_scoped_perf_summary(
        [
            {"operation_scoped_perf_observation": _observation(3)},
            {"operation_scoped_perf_observation": _observation(7, attr_config=0x4F2E)},
        ]
    )
    assert summary["status"] == "heterogeneous_provenance"


def test_not_running_excluded_from_complete_summary():
    summary = _operation_scoped_perf_summary(
        [
            {"operation_scoped_perf_observation": _observation(3)},
            {"operation_scoped_perf_observation": _observation(99, status="not_running")},
        ]
    )
    assert summary["status"] == "incomplete"
    assert summary["not_running_count"] == 1
    assert summary["raw_count_summary"]["median"] == 3.0


def test_missing_middle_pmu_reading_preserves_operation_alignment():
    summary = _operation_scoped_perf_summary(
        [
            {"operation_scoped_perf_observation": _observation(3)},
            {},
            {"operation_scoped_perf_observation": _observation(7)},
        ]
    )
    assert summary["raw_counts"] == [3, 7]
    assert [
        reading["raw_count"] if reading is not None else None
        for reading in summary["per_operation_readings"]
    ] == [3, None, 7]


def _oracle_record(replicate_id: str, median: float, *, multiplexed_fraction=0.0):
    return {
        "replicate_id": replicate_id,
        "operation_scoped_perf": {
            "status": "complete",
            "raw_count_summary": {"median": median},
            "scaled_count_summary": {"median": median},
            "multiplexed_fraction": multiplexed_fraction,
        },
    }


def _oracle_contract():
    return {
        "controls": [{"name": "null_control", "role": "null"}],
        "required_contrasts": [
            {"left_control": "cached_control", "right_control": "requested_clflush_control"}
        ],
        "null_stability": {
            "max_relative_deviation": 0.25,
            "max_relative_mad": 0.10,
            "minimum_complete_replicates": 3,
        },
    }


def test_multiplex_veto_fails_agreement_and_stability():
    records = {
        "null_control": [_oracle_record(f"replicate-{i:04d}", 4.0) for i in range(3)],
        "cached_control": [
            _oracle_record(f"replicate-{i:04d}", 4.0, multiplexed_fraction=0.5) for i in range(3)
        ],
        "requested_clflush_control": [
            _oracle_record(f"replicate-{i:04d}", 30.0, multiplexed_fraction=0.5) for i in range(3)
        ],
    }
    analysis = _operation_scoped_perf_oracle_analysis(records, _oracle_contract(), 3)
    assert analysis["agreement"]["status"] == "fail"
    assert analysis["agreement"]["multiplex_veto"] is True
    assert analysis["stability_pass"] is False


def test_frozen_analysis_rejects_missing_multiplex_telemetry():
    def legacy_record(replicate_id, median):
        return {
            "replicate_id": replicate_id,
            "operation_scoped_perf": {
                "status": "complete",
                "raw_count_summary": {"median": median},
                "scaled_count_summary": {"median": median},
            },
        }

    records = {
        "null_control": [legacy_record(f"replicate-{i:04d}", 4.0) for i in range(3)],
        "cached_control": [legacy_record(f"replicate-{i:04d}", 4.0) for i in range(3)],
        "requested_clflush_control": [legacy_record(f"replicate-{i:04d}", 30.0) for i in range(3)],
    }
    analysis = _operation_scoped_perf_oracle_analysis(
        records, _oracle_contract(), 3, require_explicit_multiplex_telemetry=True
    )
    assert analysis["agreement"]["status"] == "fail"
    assert analysis["agreement"]["multiplex_veto"] is True


def test_frozen_analysis_rejects_null_or_partial_multiplex_telemetry():
    records = {
        "null_control": [
            _oracle_record(f"replicate-{i:04d}", 4.0, multiplexed_fraction=None) for i in range(3)
        ],
        "cached_control": [
            _oracle_record(f"replicate-{i:04d}", 4.0, multiplexed_fraction=0.0) for i in range(3)
        ],
        "requested_clflush_control": [
            _oracle_record(f"replicate-{i:04d}", 30.0, multiplexed_fraction=0.0) for i in range(3)
        ],
    }
    analysis = _operation_scoped_perf_oracle_analysis(
        records, _oracle_contract(), 3, require_explicit_multiplex_telemetry=True
    )
    assert analysis["agreement"]["multiplex_telemetry_present"] is True
    assert analysis["agreement"]["multiplex_telemetry_complete"] is False
    assert analysis["agreement"]["multiplex_veto"] is True


def test_scaled_disagreement_fails_agreement():
    records = {
        "null_control": [_oracle_record(f"replicate-{i:04d}", 4.0) for i in range(3)],
        "cached_control": [_oracle_record(f"replicate-{i:04d}", 4.0) for i in range(3)],
        "requested_clflush_control": [
            {
                "replicate_id": f"replicate-{i:04d}",
                "operation_scoped_perf": {
                    "status": "complete",
                    "raw_count_summary": {"median": 30.0},
                    "scaled_count_summary": {"median": 2.0},
                    "multiplexed_fraction": 0.0,
                },
            }
            for i in range(3)
        ],
    }
    analysis = _operation_scoped_perf_oracle_analysis(records, _oracle_contract(), 3)
    assert analysis["agreement"]["status"] == "fail"


def test_zero_center_null_fails_with_explicit_reason():
    result = _null_stability_analysis(
        [
            {"status": "complete", "sample_median_summary": {"median": 0.0}},
            {"status": "complete", "sample_median_summary": {"median": 0.0}},
            {"status": "complete", "sample_median_summary": {"median": 1.0}},
        ],
        expected_replicates=3,
        rule={
            "max_relative_deviation": 0.25,
            "max_relative_mad": 0.10,
            "minimum_complete_replicates": 3,
        },
    )
    assert result["status"] == "fail"
    assert "zero or near-zero" in result["stability"]["reason"]


def _multiboot_report(
    boot_id: str,
    null_medians,
    left,
    right,
    protocol_hash="ph",
    *,
    multiplex_veto=False,
):
    def rec(rid, median):
        return {
            "replicate_id": rid,
            "operation_scoped_perf": {
                "status": "complete",
                "raw_count_summary": {"median": median},
                "scaled_count_summary": {"median": median},
                "multiplexed_fraction": 0.0,
            },
        }

    def summary(medians):
        import numpy as np

        values = np.asarray(medians, dtype=np.float64)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return {
            "status": "pass",
            "completeness": {"status": "pass"},
            "finite_value_validity": {
                "status": "pass",
                "raw_replicate_medians": list(medians),
            },
            "stability": {"status": "pass", "center": center, "median_absolute_deviation": mad},
        }

    n = len(null_medians)
    pairs = [
        {
            "replicate_id": f"replicate-{i:04d}",
            "left_median": left[i],
            "right_median": right[i],
            "difference": right[i] - left[i],
            "observed_relationship": "right_above_left",
        }
        for i in range(n)
    ]
    return {
        "protocol": {
            "protocol_hash": protocol_hash,
            "sample_design": {
                "replicates": n,
                "scoped_perf_event": MULTIBOOT_CANDIDATE_EVENT,
            },
        },
        "controls": {
            "boot_dependence": {"unique_boots": [boot_id]},
            "operation_scoped_perf_oracle": {
                "agreement": {
                    "status": "pass",
                    "agreement_count": n,
                    "sample_count": n,
                    "multiplex_veto": multiplex_veto,
                    "paired_differences": pairs,
                },
                "null_stability": summary(null_medians),
                "stability_pass": True,
            },
        },
        "decision_gate": {"outcome": "B_observable_available_but_oracle_weak"},
    }


def test_frozen_candidate_substitution_is_rejected():
    with pytest.raises(ValueError, match="frozen multiboot candidate"):
        multiboot_protocol(
            {
                "characterization": {
                    "scoped_perf_event": "cpu/cache-references/",
                    "replicates": 3,
                    "multiboot_boots": 3,
                }
            }
        )


def test_non_three_replicates_are_derived_from_frozen_reports():
    reports = [
        _multiboot_report(f"boot-{boot}", [10.0] * 4, [4.0] * 4, [30.0] * 4) for boot in range(3)
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["replicates_per_boot"] == 4
    assert combined["cross_boot_null_stability"]["completeness"]["expected_replicates"] == 12


def test_boot_count_mismatch_fails_provenance_gate():
    reports = [
        _multiboot_report(f"boot-{boot}", [10.0] * 3, [4.0] * 3, [30.0] * 3) for boot in range(3)
    ]
    combined = combine_multiboot_reports(reports, expected_boots=4)
    assert combined["boots_distinct_and_genuine"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"


def test_missing_multiboot_multiplex_telemetry_cannot_pass_cleanliness():
    reports = [
        _multiboot_report(f"boot-{boot}", [10.0] * 3, [4.0] * 3, [30.0] * 3, multiplex_veto=None)
        for boot in range(3)
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["outcome"] != "A_usable_auditable_primitive"


def test_multiboot_combine_reaches_a_when_stable():
    reports = [
        _multiboot_report(f"boot-{i}", [10.0, 10.5, 9.5], [4.0] * 3, [30.0] * 3) for i in range(3)
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["boots_distinct_and_genuine"] is True
    assert combined["directional_agreement_every_boot"] is True
    assert combined["outcome"] == "A_usable_auditable_primitive"


def test_multiboot_combine_is_c_on_reused_boot():
    reports = [
        _multiboot_report("same-boot", [10.0, 10.5, 9.5], [4.0] * 3, [30.0] * 3),
        _multiboot_report("same-boot", [10.0, 10.5, 9.5], [4.0] * 3, [30.0] * 3),
        _multiboot_report("other-boot", [10.0, 10.5, 9.5], [4.0] * 3, [30.0] * 3),
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["boots_distinct_and_genuine"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"


def test_multiboot_combine_is_b_when_cross_boot_null_unstable():
    reports = [
        _multiboot_report("boot-0", [4.5, 4.6, 4.4], [8.0] * 3, [33.0] * 3),
        _multiboot_report("boot-1", [10.0, 10.1, 9.9], [3.5] * 3, [34.0] * 3),
        _multiboot_report("boot-2", [3.0, 3.1, 2.9], [2.0] * 3, [33.5] * 3),
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["directional_agreement_every_boot"] is True
    assert combined["cross_boot_null_stable"] is False
    assert combined["outcome"] == "B_observable_available_but_oracle_weak"


def test_multiboot_combine_is_c_on_protocol_mismatch():
    reports = [
        _multiboot_report("boot-0", [10.0] * 3, [4.0] * 3, [30.0] * 3, protocol_hash="a"),
        _multiboot_report("boot-1", [10.0] * 3, [4.0] * 3, [30.0] * 3, protocol_hash="b"),
        _multiboot_report("boot-2", [10.0] * 3, [4.0] * 3, [30.0] * 3, protocol_hash="a"),
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["protocol_agreement"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"


def test_controlled_topology_rejects_virtual_derivation():
    with pytest.raises(ValueError, match="cannot be derived from a virtual address"):
        SyntheticMockControlledBackend(
            count=4, trace_length=8
        ).derive_topology_from_virtual_address(0x7FFF0000)


def test_controlled_topology_requires_hardware_source():
    with pytest.raises(ValueError, match="require source='controlled_hardware'"):
        ControlledMemoryTopology(source="unavailable", row="1234").validate()
    hardware = ControlledMemoryTopology(source="controlled_hardware", row="1234")
    hardware.validate()
    assert hardware.as_dict()["row"] == "1234"


def test_mock_controlled_backend_never_claims_topology():
    backend = SyntheticMockControlledBackend(count=4, trace_length=8, seed=5)
    try:
        samples = list(backend.samples())
        assert len(samples) == 4
        for sample in samples:
            assert sample.metadata["controlled_topology_source"] == "unavailable"
            assert "row_id" in sample.metadata
            assert sample.metadata["row_id"] == "row-unknown"
    finally:
        backend.close()


def test_controlled_trace_contract_requires_identified_channels():
    with pytest.raises(ValueError, match="at least one identified channel"):
        ControlledTraceAcquisition(
            acquisition_id="acq-1",
            trigger_id="trigger-1",
            trigger_hardware_ticks=1,
            hardware_clock_id="clock-1",
            timing_uncertainty_ticks=0,
            channels=(),
            refresh_relationship="controlled schedule r1",
            command_sequence_id="sequence-1",
        ).validate()
    acquisition = ControlledTraceAcquisition(
        acquisition_id="acq-1",
        trigger_id="trigger-1",
        trigger_hardware_ticks=1,
        hardware_clock_id="clock-1",
        timing_uncertainty_ticks=1,
        channels=(
            ControlledTraceChannel(
                channel_id="adc-0",
                channel_kind="analog",
                units="volts",
                sampling_clock_id="clock-1",
                calibration_id="cal-1",
            ),
        ),
        refresh_relationship="controlled schedule r1",
        command_sequence_id="sequence-1",
    )
    assert acquisition.as_dict()["channels"][0]["channel_id"] == "adc-0"


def test_controlled_contract_rejects_placeholder_command_provenance():
    with pytest.raises(ValueError, match="command provenance is missing"):
        ControlledCommand(
            command_id="command-1",
            kind="read",
            address_token="controller-address-1",
            issued_at_hardware_ticks=10,
        ).validate()


def test_controlled_command_result_binds_command_trace_and_provenance():
    command = ControlledCommand(
        command_id="command-1",
        kind="read",
        address_token="controller-address-1",
        issued_at_hardware_ticks=10,
        command_sequence_id="sequence-1",
        refresh_relationship="after refresh 42",
        timing_provenance="controller clock clock-1",
    )
    acquisition = ControlledTraceAcquisition(
        acquisition_id="acq-1",
        trigger_id="trigger-1",
        trigger_hardware_ticks=11,
        hardware_clock_id="clock-1",
        timing_uncertainty_ticks=1,
        channels=(
            ControlledTraceChannel(
                channel_id="adc-0",
                channel_kind="analog",
                units="volts",
                sampling_clock_id="clock-1",
                calibration_id="cal-1",
            ),
        ),
        refresh_relationship="after refresh 42",
        command_sequence_id="sequence-1",
    )
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
    result = ControlledCommandResult(
        command=command,
        status="complete",
        topology=ControlledMemoryTopology(source="controlled_hardware", row="row-1"),
        provenance=provenance,
        completed_at_hardware_ticks=12,
        acquisition=acquisition,
    )
    assert result.as_dict()["acquisition"]["command_sequence_id"] == "sequence-1"

    mismatched = ControlledCommandResult(
        command=command,
        status="complete",
        topology=ControlledMemoryTopology(source="controlled_hardware", row="row-1"),
        provenance=provenance,
        completed_at_hardware_ticks=12,
        acquisition=ControlledTraceAcquisition(
            **{**acquisition.__dict__, "command_sequence_id": "different-sequence"}
        ),
    )
    with pytest.raises(ValueError, match="does not match command"):
        mismatched.validate()


def test_mock_controlled_reconstruction_matches_uninterrupted_sequence():
    uninterrupted_backend = SyntheticMockControlledBackend(count=8, trace_length=8, seed=29)
    prefix_backend = SyntheticMockControlledBackend(count=8, trace_length=8, seed=29)
    resumed_backend = SyntheticMockControlledBackend(count=8, trace_length=8, seed=29)
    try:
        uninterrupted = list(uninterrupted_backend.samples())
        prefix = list(prefix_backend.samples())[:3]
        resumed = list(resumed_backend.samples(start_index=3))
        reconstructed = prefix + resumed
        assert [sample.label for sample in reconstructed] == [
            sample.label for sample in uninterrupted
        ]
        for expected, actual in zip(uninterrupted, reconstructed, strict=True):
            import numpy as np

            np.testing.assert_array_equal(expected.trace, actual.trace)
            assert expected.metadata["sample_id"] == actual.metadata["sample_id"]
            assert "synthetic" in str(actual.metadata["physical_observation_semantics"])
    finally:
        uninterrupted_backend.close()
        prefix_backend.close()
        resumed_backend.close()


def test_cpu_id_falls_back_to_libc_when_os_helper_missing(monkeypatch):
    import os

    import sensetrace.acquisition.commodity as commodity

    monkeypatch.delattr(os, "sched_getcpu", raising=False)
    assert isinstance(commodity.CommodityDramBackend._cpu_id(), int)


def test_discovery_lists_but_never_selects_ambiguous_alias(tmp_path):
    for device in ["cpu", "alternate"]:
        events = tmp_path / "bus" / "event_source" / "devices" / device / "events"
        events.mkdir(parents=True)
        (events.parent / "type").write_text("4\n")
        format_dir = events.parent / "format"
        format_dir.mkdir()
        (format_dir / "event").write_text("config:0-7\n")
        (events / "cache-misses").write_text("event=0x2e\n")
    result = discover_counter_capabilities(sysfs_root=tmp_path, perf_output="")
    misses = next(item for item in result["events"] if item["logical_name"] == "cache_misses")
    assert misses["selection_status"] == "ambiguous_unqualified_alias"
    with pytest.raises(ValueError, match="not uniquely qualified"):
        select_sysfs_event_encoding(tmp_path, "cache-misses")
