from __future__ import annotations

from sensetrace.acquisition.capabilities import commodity_timing_oracle
from sensetrace.acquisition.perf import (
    PERF_ATTR_DISABLED,
    PERF_ATTR_EXCLUDE_HV,
    PERF_ATTR_EXCLUDE_KERNEL,
    build_perf_event_attr,
    discover_counter_capabilities,
)
from sensetrace.acquisition.primitive import (
    CommodityTimingPrimitive,
    PrimitiveCapabilities,
    available_measurement_primitives,
)
from sensetrace.characterization import characterization_protocol, decide_characterization
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
    assert protocol["version"] == "measurement-primitive-characterization-v1"
    assert protocol["analysis"]["no_model_training"] is True
    assert "physical DRAM access" in protocol["claim_boundary"]
    assert protocol["primitive"]["access_state_oracle"]["model_feature_eligible"] is False
    assert protocol["analysis"]["no_model_training"] is True


def test_perf_event_attr_is_disabled_and_excludes_kernel_and_hypervisor():
    attribute, record = build_perf_event_attr("cache-misses")
    flags = int.from_bytes(attribute[40:48], byteorder="little")
    assert flags & PERF_ATTR_DISABLED
    assert flags & PERF_ATTR_EXCLUDE_KERNEL
    assert flags & PERF_ATTR_EXCLUDE_HV
    assert record["flags_by_name"]["inherit"] is False
    assert record["flags_by_name"]["exclude_user"] is False


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
