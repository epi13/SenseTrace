from __future__ import annotations

from sensetrace.acquisition.capabilities import commodity_timing_oracle
from sensetrace.acquisition.perf import discover_counter_capabilities
from sensetrace.acquisition.primitive import (
    CommodityTimingPrimitive,
    PrimitiveCapabilities,
    available_measurement_primitives,
)
from sensetrace.characterization import characterization_protocol
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
    assert CommodityTimingPrimitive(None, operation="memory_read", cache_control="none", eviction=bytearray(64)).describe()["model_eligible_features"].startswith("raw trace")


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


def test_characterization_protocol_is_not_hidden_bit_inference():
    protocol = characterization_protocol(_config())
    assert protocol["version"] == "measurement-primitive-characterization-v1"
    assert protocol["analysis"]["no_model_training"] is True
    assert "physical DRAM access" in protocol["claim_boundary"]
