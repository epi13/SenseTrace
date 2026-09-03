from __future__ import annotations

import struct

import pytest

from sensetrace.acquisition.capabilities import commodity_timing_oracle
from sensetrace.acquisition.perf import (
    PERF_ATTR_DISABLED,
    PERF_ATTR_EXCLUDE_HV,
    PERF_ATTR_EXCLUDE_KERNEL,
    PERF_EVENT_IOC_DISABLE,
    PERF_EVENT_IOC_ENABLE,
    PERF_EVENT_IOC_RESET,
    OperationScopedPerfEvent,
    build_perf_event_attr,
    discover_counter_capabilities,
    select_sysfs_event_encoding,
)
from sensetrace.acquisition.primitive import (
    CommodityTimingPrimitive,
    PrimitiveCapabilities,
    available_measurement_primitives,
)
from sensetrace.characterization import (
    _contrast,
    _decision_evidence,
    _operation_scoped_perf_oracle_analysis,
    _operation_scoped_perf_summary,
    characterization_protocol,
    decide_characterization,
)
from sensetrace.config import validate_config


def _config() -> dict:
    return validate_config(
        {
            "experiment": {"name": "primitive-contract-test", "seed": 7},
            "data": {"target_balance": 0.5, "samples": 8, "trace_length": 32},
            "acquisition": {"backend": "commodity"},
            "phase1a": {
                "protocol_version": "phase1a-commodity-baseline-v1",
                "measurement_primitive": "commodity-clflush-timed-load",
                "location_count": 1,
                "trials_per_location": 8,
                "labels_per_location": 4,
                "cache_control": "none",
                "use_native_kernel": False,
                "require_native_kernel": False,
            },
        }
    )


def test_commodity_primitive_contract_uses_explicit_unknown_values():
    capabilities = PrimitiveCapabilities(
        operation_issues_memory_access="known",
        independent_access_state_oracle="unsupported",
        cache_residency_control="known",
        translation_state="unknown",
        depends_on_virtual_addresses="known",
        physical_address_information="unsupported",
        row_bank_channel_topology="unsupported",
        privileged_counters="unsupported",
        kernel_support="unknown",
        external_hardware="unsupported",
        destructive_or_state_changing="known",
        replay_across_sessions_boots_devices="unknown",
    )
    assert capabilities.as_dict()["translation_state"] == "unknown"
    oracle = commodity_timing_oracle(operation="memory_read", cache_control="clflush")
    assert oracle.strength == "unavailable"
    assert oracle.independent_of_latency is False
    assert available_measurement_primitives() == ("commodity-clflush-timed-load",)
    assert (
        CommodityTimingPrimitive(
            None, operation="memory_read", cache_control="none", eviction=bytearray(64)
        )
        .describe()["model_eligible_features"]
        .startswith("raw trace")
    )


def test_perf_discovery_is_cpu_vocabulary_aware(tmp_path):
    events = tmp_path / "bus" / "event_source" / "devices" / "cpu" / "events"
    events.mkdir(parents=True)
    (events / "cache-misses").write_text("event=0x2e,umask=0x41\n")
    uncore = tmp_path / "bus" / "event_source" / "devices" / "uncore_imc_0" / "events"
    uncore.mkdir(parents=True)
    (uncore / "cas_count_read").write_text("event=0x04\n")
    result = discover_counter_capabilities(
        sysfs_root=tmp_path,
        perf_path="/usr/bin/perf",
        perf_output="cache-references\ncache-misses\n",
        command_runner=lambda command: "Model name: Test CPU" if command == ["lscpu"] else None,
    )
    assert result["cpu_model"] == "Test CPU"
    by_name = {event["logical_name"]: event for event in result["events"]}
    assert by_name["cache_misses"]["status"] == "available"
    assert by_name["uncore_memory_reads"]["status"] == "available"
    assert result["collection_policy"].startswith("No system-wide")
    assert result["process_thread_scope"]["system_wide"] is False
    assert result["event_multiplexing"]["status"] == "not_tested"


def test_characterization_protocol_is_not_hidden_bit_inference():
    protocol = characterization_protocol(_config())
    assert protocol["version"] == "measurement-primitive-characterization-v3"
    assert protocol["analysis"]["no_model_training"] is True
    assert "physical DRAM access" in protocol["claim_boundary"]
    assert protocol["primitive"]["access_state_oracle"]["model_feature_eligible"] is False
    assert protocol["analysis"]["no_model_training"] is True
    # v3 freezes the first-touch control and the witness policy explicitly.
    assert protocol["sample_design"]["allocation_warmup"] == {
        "enabled": False,
        "touch_pages": True,
        "dummy_loads": 0,
    }
    assert protocol["witness"]["requirement"] in {"disabled", "optional", "required"}
    assert protocol["witness"]["automatic_sample_veto"] is False


def test_perf_event_attr_is_disabled_and_excludes_kernel_and_hypervisor():
    attribute, record = build_perf_event_attr("cache-misses")
    flags = int.from_bytes(attribute[40:48], byteorder="little")
    assert flags & PERF_ATTR_DISABLED
    assert flags & PERF_ATTR_EXCLUDE_KERNEL
    assert flags & PERF_ATTR_EXCLUDE_HV
    assert record["flags_by_name"]["inherit"] is False
    assert record["flags_by_name"]["exclude_user"] is False
    assert record["config_width_bits"] == 64
    assert record["read_format_fields"]["total_time_running"] is True


def test_perf_raw_config_accepts_full_64_bit_width():
    _attribute, record = build_perf_event_attr("raw", raw_event=0x1_0000_0000)
    assert record["config"] == 0x1_0000_0000
    assert record["config_width_bits"] == 64


def test_sysfs_event_identity_and_format_are_preserved_without_alias_collision(tmp_path):
    for device in ["cpu", "alternate"]:
        events = tmp_path / "bus" / "event_source" / "devices" / device / "events"
        events.mkdir(parents=True)
        (events.parent / "type").write_text("4\n")
        format_dir = events.parent / "format"
        format_dir.mkdir()
        (format_dir / "event").write_text("config:0-7\n")
        (format_dir / "umask").write_text("config:8-15\n")
        (events / "cache-misses").write_text("event=0x2e,umask=0x41\n")

    result = discover_counter_capabilities(sysfs_root=tmp_path, perf_output="")
    cache_misses = next(item for item in result["events"] if item["logical_name"] == "cache_misses")
    assert cache_misses["status"] == "ambiguous"
    assert cache_misses["selection_status"] == "ambiguous_unqualified_alias"
    assert {item["device"] for item in cache_misses["encodings"]} == {"alternate", "cpu"}
    assert all(item["event_source_type"]["value"] == 4 for item in cache_misses["encodings"])
    assert all(item["config"] == 0x412E for item in cache_misses["encodings"])
    assert all("event" in item["sysfs_format_fields"] for item in cache_misses["encodings"])
    with pytest.raises(ValueError, match="not uniquely qualified"):
        select_sysfs_event_encoding(tmp_path, "cache-misses")
    selected = select_sysfs_event_encoding(tmp_path, "cpu/cache-misses/")
    assert selected.device == "cpu"


def test_operation_scoped_reader_enables_only_around_callback_and_closes_fd(monkeypatch):
    ioctls = []
    closed = []

    def fake_open(attribute, *, syscall_number, thread_id, cpu):
        assert syscall_number == 298
        assert thread_id > 0
        assert cpu == -1
        assert int.from_bytes(bytes(attribute)[40:48], "little") & PERF_ATTR_DISABLED
        return 73

    monkeypatch.setattr("sensetrace.acquisition.perf._perf_event_open", fake_open)
    monkeypatch.setattr(
        "sensetrace.acquisition.perf.fcntl.ioctl",
        lambda fd, request, value: ioctls.append((fd, request, value)),
    )
    monkeypatch.setattr(
        "sensetrace.acquisition.perf.os.read",
        lambda fd, size: struct.pack("<QQQ", 7, 10, 10),
    )
    monkeypatch.setattr("sensetrace.acquisition.perf.os.close", lambda fd: closed.append(fd))

    reader = OperationScopedPerfEvent("cache-misses", syscall_number=298)
    result, reading = reader.measure(lambda: "controlled-result")
    assert result == "controlled-result"
    assert reading.raw_count == 7
    assert reading.multiplexed is False
    assert [request for _fd, request, _value in ioctls] == [
        PERF_EVENT_IOC_RESET,
        PERF_EVENT_IOC_ENABLE,
        PERF_EVENT_IOC_DISABLE,
    ]
    assert closed == [73]
    assert reader.provenance["scope"]["cpu_argument"] == -1
    assert reader.provenance["scope"]["system_wide"] is False
    assert reader.provenance["scope"]["inherit"] is False


def test_characterization_retains_all_operation_scoped_perf_readings():
    def observation(raw_count: int) -> str:
        return (
            '{"status":"complete","event":{"qualified_name":"cpu/cache-misses/"},'
            f'"reading":{{"status":"complete","raw_count":{raw_count},'
            '"scaled_count":' + str(float(raw_count)) + ',"time_enabled":10,'
            '"time_running":10,"multiplexed":false}}'
        )

    summary = _operation_scoped_perf_summary(
        [
            {"operation_scoped_perf_observation": observation(3)},
            {"operation_scoped_perf_observation": observation(7)},
        ]
    )
    assert summary["status"] == "complete"
    assert summary["observation_count"] == 2
    assert summary["complete_reading_count"] == 2
    assert summary["raw_counts"] == [3, 7]
    assert summary["time_enabled"] == [10, 10]
    assert summary["time_running"] == [10, 10]
    assert summary["multiplexed"] == [False, False]
    assert summary["raw_readings_retained"] is True


def test_permission_denied_perf_probe_is_machine_readable(monkeypatch):
    import ctypes
    import errno

    def denied(*_args, **_kwargs):
        ctypes.set_errno(errno.EACCES)
        return -1

    monkeypatch.setattr("sensetrace.acquisition.perf._perf_event_open", denied)
    result = discover_counter_capabilities(
        sysfs_root="/nonexistent",
        perf_output="cache-misses\n",
        probe_hardware_events=True,
    )
    cache_misses = next(item for item in result["events"] if item["logical_name"] == "cache_misses")
    assert cache_misses["probe_status"] == "permission_denied"
    assert cache_misses["probe"]["errno_name"] == "EACCES"
    assert cache_misses["probe"]["scope"]["system_wide"] is False
    assert cache_misses["probe"]["perf_event_attr"]["flags_by_name"]["disabled"] is True


def _complete_characterization_evidence(**overrides):
    evidence = {
        "observable_response": True,
        "controls_pass": True,
        "null_stable": True,
        "provenance_complete": True,
        "scope_acceptable": True,
        "oracle_available": True,
        "oracle_independent": True,
        "oracle_agreement_pass": True,
        "oracle_stability_pass": True,
    }
    evidence.update(overrides)
    return evidence


def test_characterization_decision_can_reach_a_with_a_fake_independent_oracle():
    assert decide_characterization(_complete_characterization_evidence())["outcome"] == (
        "A_usable_auditable_primitive"
    )


def test_characterization_decision_returns_b_for_an_unavailable_oracle():
    result = decide_characterization(
        _complete_characterization_evidence(
            oracle_available=False,
            oracle_independent=False,
            oracle_agreement_pass=False,
            oracle_stability_pass=False,
        )
    )
    assert result["outcome"] == "B_observable_available_but_oracle_weak"


def test_characterization_decision_returns_c_for_failed_controls():
    result = decide_characterization(_complete_characterization_evidence(controls_pass=False))
    assert result["outcome"] == "C_primitive_unsuitable"


def _replicate_record(replicate_id: str, median: float, *, status: str = "complete") -> dict:
    return {
        "replicate_id": replicate_id,
        "status": status,
        "sample_median_summary": {"median": median},
    }


def test_null_stability_rejects_extremely_drifting_finite_replicates():
    contract = {
        "controls": [{"name": "null", "role": "null"}],
        "required_contrasts": [],
        "null_stability": {
            "max_relative_deviation": 0.25,
            "max_relative_mad": 0.10,
            "minimum_complete_replicates": 3,
        },
    }
    evidence = _decision_evidence(
        contract,
        {
            "null": [
                _replicate_record("replicate-0000", 10.0),
                _replicate_record("replicate-0001", 10.0),
                _replicate_record("replicate-0002", 100.0),
            ]
        },
        {},
        3,
    )
    assert evidence["null_stable"] is False
    assert evidence["null_stability"]["completeness"]["status"] == "pass"
    assert evidence["null_stability"]["finite_value_validity"]["status"] == "pass"
    assert evidence["null_stability"]["stability"]["status"] == "fail"


def test_contrast_pairs_only_matching_replicate_ids_and_reports_missing_side():
    left = [
        _replicate_record("replicate-0000", 10.0),
        _replicate_record("replicate-0001", 20.0),
        _replicate_record("replicate-0002", 30.0),
    ]
    right = [
        _replicate_record("replicate-0000", 15.0),
        _replicate_record("replicate-0001", 25.0, status="unavailable"),
        _replicate_record("replicate-0002", 45.0),
    ]
    result = _contrast(
        left,
        right,
        expected_replicate_ids=["replicate-0000", "replicate-0001", "replicate-0002"],
    )
    assert result["matched_replicate_ids"] == ["replicate-0000", "replicate-0002"]
    assert result["missing_right_replicate_ids"] == ["replicate-0001"]
    assert result["paired_differences"] == [
        {
            "replicate_id": "replicate-0000",
            "left_median": 10.0,
            "right_median": 15.0,
            "difference": 5.0,
        },
        {
            "replicate_id": "replicate-0002",
            "left_median": 30.0,
            "right_median": 45.0,
            "difference": 15.0,
        },
    ]
    assert result["required_replicates_present"] is False


def test_scoped_perf_oracle_reports_directional_agreement_and_null_stability():
    def record(replicate_id: str, median: float) -> dict:
        return {
            "replicate_id": replicate_id,
            "operation_scoped_perf": {
                "status": "complete",
                "raw_count_summary": {"median": median},
            },
        }

    contract = {
        "controls": [{"name": "null", "role": "null"}],
        "required_contrasts": [
            {
                "left_control": "cached",
                "right_control": "flushed",
            }
        ],
        "null_stability": {
            "max_relative_deviation": 0.25,
            "max_relative_mad": 0.10,
            "minimum_complete_replicates": 3,
        },
    }
    analysis = _operation_scoped_perf_oracle_analysis(
        {
            "null": [record(f"replicate-{index:04d}", 1.0) for index in range(3)],
            "cached": [record(f"replicate-{index:04d}", 4.0) for index in range(3)],
            "flushed": [record(f"replicate-{index:04d}", 32.0) for index in range(3)],
        },
        contract,
        3,
    )
    assert analysis["agreement"]["status"] == "pass"
    assert analysis["agreement"]["confusion_matrix"] == {
        "expected_right_above_left": {
            "observed_right_above_left": 3,
            "observed_right_not_above_left": 0,
        }
    }
    assert analysis["stability_pass"] is True
