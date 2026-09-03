"""Adversarial regression tests for PMU hardening, characterization gates,
multi-boot combination, and the Phase 2 boundary."""

from __future__ import annotations

import json
import struct

import pytest

from sensetrace.acquisition.controlled import (
    ControlledMemoryTopology,
    SyntheticMockControlledBackend,
)
from sensetrace.acquisition.perf import (
    OperationScopedPerfEvent,
    PerfEventEncoding,
    _encode_sysfs_config,
    discover_counter_capabilities,
    expected_read_size,
    select_sysfs_event_encoding,
)
from sensetrace.characterization import (
    _null_stability_analysis,
    _operation_scoped_perf_oracle_analysis,
    _operation_scoped_perf_summary,
)
from sensetrace.multiboot import combine_multiboot_reports


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


def _multiboot_report(boot_id: str, null_medians, left, right, protocol_hash="ph"):
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
        "protocol": {"protocol_hash": protocol_hash},
        "controls": {
            "boot_dependence": {"unique_boots": [boot_id]},
            "operation_scoped_perf_oracle": {
                "agreement": {
                    "status": "pass",
                    "agreement_count": n,
                    "sample_count": n,
                    "multiplex_veto": False,
                    "paired_differences": pairs,
                },
                "null_stability": summary(null_medians),
                "stability_pass": True,
            },
        },
        "decision_gate": {"outcome": "B_observable_available_but_oracle_weak"},
    }


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
