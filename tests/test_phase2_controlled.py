from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sensetrace.acquisition.controlled import (
    ControlledAcquisitionProvenance,
    ControlledCommand,
    ControlledCommandResult,
    ControlledMemoryTopology,
    ControlledTraceAcquisition,
    ControlledTraceChannel,
    SyntheticMockControlledBackend,
    SyntheticMockControlledInterface,
)
from sensetrace.config import validate_config
from sensetrace.datasets import load_dataset, validate_physical_evidence_dataset
from sensetrace.errors import ConfigError, IntegrityError
from sensetrace.runner import AcquisitionRunner


@pytest.mark.parametrize(
    "source,field,value",
    [
        ("unavailable", "row", " unknown "),
        ("unavailable", "row", "   "),
        ("unavailable", "row", 12),
        ("unavailable", "row", "row-1"),
        ("not-a-source", "row", "unavailable"),
        ("controlled_hardware", "row", "unavailable"),
    ],
)
def test_topology_validation_rejects_malformed_or_unsupported_claims(source, field, value):
    topology = ControlledMemoryTopology(source=source, **{field: value})
    with pytest.raises(ValueError):
        topology.validate()


def test_topology_accepts_only_explicit_hardware_sourced_concrete_values():
    topology = ControlledMemoryTopology(source="controlled_hardware", row="row-1", bank="bank-2")
    topology.validate()
    assert topology.as_dict()["row"] == "row-1"


@pytest.mark.parametrize("kind", ["analog", "unsupported", "", [], None])
def test_channel_kind_is_runtime_checked(kind):
    channel = ControlledTraceChannel(
        channel_id="channel-1",
        channel_kind=kind,
        units="volts",
        sampling_clock_id="sample-clock",
        calibration_id="calibration-1",
    )
    if kind == "analog":
        channel.validate()
    else:
        with pytest.raises(ValueError):
            channel.validate()


def _command(**changes) -> ControlledCommand:
    value = ControlledCommand(
        command_id="command-1",
        kind="read",
        address_token="opaque-token-1",
        issued_at_hardware_ticks=10,
        command_sequence_id="sequence-1",
        refresh_relationship="refresh-1",
        timing_provenance="command-clock ticks",
        command_clock_id="command-clock",
    )
    return replace(value, **changes)


def _provenance(**changes) -> ControlledAcquisitionProvenance:
    value = ControlledAcquisitionProvenance(
        experiment_target_id="target-1",
        controller_firmware_id="firmware-1",
        controller_config_hash="config-hash-1",
        device_identity="device-1",
        dimm_identity="dimm-1",
        calibration_state="calibration-1",
        hardware_clock_id="command-clock",
        acquisition_trigger="trigger-domain-1",
        acquisition_configuration_hash="acquisition-hash-1",
        trigger_identity="trigger-1",
        timing_provenance="command-clock ticks",
        refresh_relationship="refresh-1",
        command_provenance="command-log-1",
        sampling_clock_id="sample-clock",
        command_sequence_id="sequence-1",
    )
    return replace(value, **changes)


def _acquisition(**changes) -> ControlledTraceAcquisition:
    value = ControlledTraceAcquisition(
        acquisition_id="acquisition-1",
        trigger_id="trigger-1",
        trigger_hardware_ticks=11,
        hardware_clock_id="sample-clock",
        timing_uncertainty_ticks=0,
        channels=(
            ControlledTraceChannel(
                channel_id="channel-1",
                channel_kind="analog",
                units="volts",
                sampling_clock_id="sample-clock",
                calibration_id="calibration-1",
            ),
        ),
        refresh_relationship="refresh-1",
        command_sequence_id="sequence-1",
    )
    return replace(value, **changes)


def _result(**changes) -> ControlledCommandResult:
    value = ControlledCommandResult(
        command=_command(),
        status="complete",
        topology=ControlledMemoryTopology(source="controlled_hardware", row="row-1"),
        provenance=_provenance(),
        completed_at_hardware_ticks=12,
        acquisition=_acquisition(),
    )
    return replace(value, **changes)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"provenance": _provenance(trigger_identity="different-trigger")}, "trigger identity"),
        ({"provenance": _provenance(refresh_relationship="different-refresh")}, "refresh"),
        ({"provenance": _provenance(command_sequence_id="different-sequence")}, "command sequence"),
        ({"provenance": _provenance(sampling_clock_id="different-clock")}, "sampling clock"),
        ({"command": _command(command_clock_id="different-clock")}, "command clock"),
        ({"command": _command(timing_provenance="different-timing")}, "timing provenance"),
    ],
)
def test_controlled_result_rejects_cross_object_provenance_mismatch(changes, match):
    with pytest.raises(ValueError, match=match):
        _result(**changes).validate()


def test_controlled_result_rejects_invalid_status_and_timing_types():
    with pytest.raises(ValueError, match="status"):
        _result(status=[]).validate()
    with pytest.raises(ValueError, match="completion timing"):
        _result(completed_at_hardware_ticks=True).validate()
    with pytest.raises(ValueError, match="trigger timing"):
        _result(acquisition=_acquisition(trigger_hardware_ticks=-1)).validate()


def test_interface_lifecycle_is_exercised_and_clocks_are_distinct():
    interface = SyntheticMockControlledInterface(
        count=4,
        trace_length=8,
        seed=11,
        target_id="mock-target",
        firmware_id="mock-firmware",
        controller_config_hash="a" * 64,
    )
    command = replace(
        _command(),
        parameters={"sample_index": 0},
        address_token="opaque-token-0",
        refresh_relationship="synthetic/no-refresh-schedule",
        timing_provenance="synthetic command clock ticks",
        command_clock_id="mock-command-clock",
    )
    result = interface.issue(command)
    result.validate()
    assert result.provenance.hardware_clock_id == "mock-command-clock"
    assert result.acquisition is not None
    assert result.acquisition.hardware_clock_id == "mock-sampling-clock"
    trace = interface.read_trace(result.acquisition)
    assert trace.shape == (8,)


def test_mock_backend_replays_identical_controller_sequence():
    first = SyntheticMockControlledBackend(
        count=8, trace_length=8, seed=21, controller_config_hash="h"
    )
    second = SyntheticMockControlledBackend(
        count=8, trace_length=8, seed=21, controller_config_hash="h"
    )
    try:
        full = list(first.samples())
        reconstructed = list(second.samples(start_index=3))
        assert [item.metadata["controlled_command_sequence_id"] for item in full[3:]] == [
            item.metadata["controlled_command_sequence_id"] for item in reconstructed
        ]
        for expected, actual in zip(full[3:], reconstructed, strict=True):
            np.testing.assert_array_equal(expected.trace, actual.trace)
            assert (
                expected.metadata["controlled_trace_acquisition"]
                == actual.metadata["controlled_trace_acquisition"]
            )
    finally:
        first.close()
        second.close()


def _mock_config() -> dict:
    return validate_config(
        {
            "experiment": {"name": "phase2-dry-run", "seed": 17},
            "data": {"samples": 8, "trace_length": 8, "target_balance": 0.5},
            "acquisition": {
                "backend": "controlled_mock",
                "shard_target_mb": 1,
                "max_samples_per_shard": 2,
            },
            "phase2": {
                "controlled_mock": {
                    "protocol_version": "controlled-memory-interface-mock-v1",
                    "count": 8,
                    "trace_length": 8,
                    "seed": 17,
                    "target_id": "mock-target",
                    "firmware_id": "mock-firmware",
                    "topology": "unavailable",
                }
            },
        }
    )


def test_phase2_mock_vertical_run_persists_auditable_nonphysical_evidence(tmp_path):
    config = _mock_config()
    runner = AcquisitionRunner(config, tmp_path / "run", run_id="phase2-run")
    first = runner.run(stop_after=3)
    second = AcquisitionRunner(config, tmp_path / "run", run_id="phase2-run").run()
    assert first["status"] == "interrupted"
    assert second["status"] == "completed"
    traces, labels, metadata, _shards, manifest = load_dataset(tmp_path / "run")
    assert traces.shape == (8, 8)
    assert labels.sum() == 4
    assert manifest["dataset_purpose"] == "phase2_mock_controlled"
    assert manifest["provenance"]["interface_name"] == "controlled-memory-interface-mock-v1"
    assert len(set(metadata["controlled_command_sequence_id"])) == 8
    assert set(metadata["controlled_topology_source"]) == {"unavailable"}
    assert all("virtual" not in str(value) for value in metadata["controlled_topology"])
    journal = (tmp_path / "run" / "events.jsonl").read_text()
    assert "dataset_manifest_written" in journal
    with pytest.raises(IntegrityError, match="physical controlled-hardware"):
        validate_physical_evidence_dataset(tmp_path / "run")


def test_mock_manifest_tampering_is_rejected(tmp_path):
    AcquisitionRunner(_mock_config(), tmp_path / "run", run_id="phase2-run").run()
    manifest_path = tmp_path / "run" / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["topology"]["source"] = "controlled_hardware"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(IntegrityError, match="topology"):
        load_dataset(tmp_path / "run")


def test_config_distinguishes_mock_from_future_real_boundary():
    config = _mock_config()
    assert config["acquisition"]["backend"] == "controlled_mock"
    future = dict(config)
    future["acquisition"] = {"backend": "controlled_hardware"}
    future["phase2"] = {"controlled_hardware": {"adapter": "future-adapter"}}
    assert validate_config(future)["acquisition"]["backend"] == "controlled_hardware"
    with pytest.raises(ConfigError, match="controlled_mock topology"):
        bad = _mock_config()
        bad["phase2"]["controlled_mock"]["topology"] = "row-1"
        validate_config(bad)


def test_frozen_commodity_gate_rejects_new_scaling_intent():
    config = {
        "experiment": {"name": "closed-commodity", "seed": 1},
        "data": {"samples": 8, "trace_length": 8, "target_balance": 0.5},
        "acquisition": {"backend": "commodity"},
        "phase1a": {"campaign_intent": "current_scaling"},
    }
    with pytest.raises(ConfigError, match="C_primitive_unsuitable"):
        validate_config(config)
