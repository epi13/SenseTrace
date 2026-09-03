from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sensetrace.acquisition.controlled import (
    CONTROLLED_CONFORMANCE_FAULTS,
    ControlledAcquisitionProvenance,
    ControlledCommand,
    ControlledCommandResult,
    ControlledInterfaceAcquisitionBackend,
    ControlledMemoryTopology,
    ControlledTraceAcquisition,
    ControlledTraceChannel,
    FaultInjectingControlledInterface,
    RecoveryPolicy,
    SyntheticMockControlledBackend,
    SyntheticMockControlledInterface,
)
from sensetrace.config import validate_config
from sensetrace.datasets import (
    load_dataset,
    validate_physical_evidence_dataset,
    write_dataset_manifest,
)
from sensetrace.errors import ConfigError, IntegrityError
from sensetrace.runner import AcquisitionRunner
from sensetrace.storage import ShardWriter, validate_all_shards


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


@pytest.mark.parametrize(
    "change",
    [
        lambda config: config["experiment"].update(seed=18),
        lambda config: config["phase2"]["controlled_mock"].update(firmware_id="firmware-2"),
        lambda config: config["phase2"]["controlled_mock"].update(target_id="target-2"),
        lambda config: config["phase2"]["controlled_mock"].update(trace_length=16),
        lambda config: config["experiment"].update(name="different-experiment"),
    ],
    ids=["seed", "firmware", "target", "trace-design", "experiment-config"],
)
def test_resume_rejects_any_changed_immutable_config_without_touching_evidence(tmp_path, change):
    root = tmp_path / "run"
    config = _mock_config()
    AcquisitionRunner(config, root, run_id="phase2-run").run(stop_after=3)
    before_config = (root / "config.json").read_bytes()
    before_shards = {
        path.name: (path.read_bytes(), path.with_suffix(".json").read_bytes())
        for path in root.glob("shard-*.npz")
    }
    changed = json.loads(json.dumps(config))
    change(changed)
    changed = validate_config(changed)
    with pytest.raises(RuntimeError, match="immutable run identity mismatch"):
        AcquisitionRunner(changed, root, run_id="phase2-run").run()
    assert (root / "config.json").read_bytes() == before_config
    assert {
        path.name: (path.read_bytes(), path.with_suffix(".json").read_bytes())
        for path in root.glob("shard-*.npz")
    } == before_shards


def test_identical_mock_resume_is_allowed_and_generic_controlled_resume_is_not():
    interface = SyntheticMockControlledInterface(
        count=4,
        trace_length=8,
        seed=9,
        target_id="mock-target",
        firmware_id="mock-firmware",
        controller_config_hash="config-hash",
    )
    backend = ControlledInterfaceAcquisitionBackend(
        interface,
        count=4,
        trace_length=8,
        labels=np.asarray([0, 1, 0, 1], dtype=np.uint8),
        session_id="session-1",
        allocation_id="allocation-1",
        label_stream_fingerprint="labels",
    )
    try:
        assert backend.recovery_policy.allow_resume is False
        assert backend.recovery_policy.deterministic_replay is False
        backend.recovery_policy = RecoveryPolicy(True, True, "accidentally permissive")
        assert (
            backend.validate_resume(
                persisted_run={},
                persisted_config={},
                current_config={},
                resume_index=1,
            ).allowed
            is False
        )
    finally:
        backend.close()


@pytest.mark.parametrize("fault", CONTROLLED_CONFORMANCE_FAULTS)
def test_controller_conformance_faults_fail_closed_at_backend_boundary(fault):
    delegate = SyntheticMockControlledInterface(
        count=4,
        trace_length=8,
        seed=7,
        target_id="mock-target",
        firmware_id="mock-firmware",
        controller_config_hash="config-hash",
    )
    interface = FaultInjectingControlledInterface(delegate, fault)
    backend = ControlledInterfaceAcquisitionBackend(
        interface,
        count=4,
        trace_length=8,
        labels=np.asarray([0, 1, 0, 1], dtype=np.uint8),
        session_id="session-1",
        allocation_id="allocation-1",
        label_stream_fingerprint="labels",
    )
    try:
        with pytest.raises((RuntimeError, ValueError)):
            list(backend.samples())
    finally:
        backend.close()


def test_command_parameters_reject_nan_and_non_json_values():
    with pytest.raises(ValueError, match="JSON-compatible"):
        _command(parameters={"nan": float("nan")}).validate()
    with pytest.raises(ValueError, match="JSON-compatible"):
        _command(parameters={"object": object()}).validate()


class _PhysicalContractInterface(SyntheticMockControlledInterface):
    interface_name = "controlled-memory-interface-v1"
    evidence_plane = "controlled_memory_interface_hardware"

    def __post_init__(self):
        super().__post_init__()
        self.session_id = "physical-controlled-session-1"
        self.protocol_hash = "physical-protocol-hash-1"

    def provenance(self):
        return replace(
            super().provenance(),
            experiment_target_id="physical-target-1",
            controller_firmware_id="physical-firmware-1",
            device_identity="physical-device-1",
            dimm_identity="physical-dimm-1",
            calibration_state="calibration-state-1",
            hardware_clock_id="physical-command-clock-1",
            acquisition_trigger="physical-trigger-domain-1",
            trigger_identity="physical-trigger-1",
            timing_provenance="hardware command clock ticks",
            refresh_relationship="controller refresh schedule-1",
            command_provenance="hardware command log-1",
            sampling_clock_id="physical-sampling-clock-1",
        )

    def topology_for(self, address_token):
        return ControlledMemoryTopology(
            source="controlled_hardware",
            channel="channel-1",
            rank="rank-1",
            bank="bank-1",
            row="row-1",
            dimm="physical-dimm-1",
        )

    def acquire_trace(self, command, channels):
        provenance = self.provenance()
        result = ControlledTraceAcquisition(
            acquisition_id=f"{self.session_id}:acquisition-{command.parameters['sample_index']:012d}",
            trigger_id=provenance.trigger_identity,
            trigger_hardware_ticks=command.issued_at_hardware_ticks,
            hardware_clock_id=provenance.sampling_clock_id,
            timing_uncertainty_ticks=0,
            channels=tuple(
                ControlledTraceChannel(
                    channel_id=channel,
                    channel_kind="analog",
                    units="volts",
                    sampling_clock_id=provenance.sampling_clock_id,
                    calibration_id="calibration-id-1",
                )
                for channel in channels
            ),
            refresh_relationship=command.refresh_relationship,
            command_sequence_id=command.command_sequence_id,
        )
        result.validate()
        return result


def _physical_fixture(tmp_path):
    config = json.loads(json.dumps(_mock_config()))
    config["acquisition"]["backend"] = "controlled_hardware"
    config["phase2"] = {"controlled_hardware": {"adapter": "test-fixture"}}
    config = validate_config(config)
    interface = _PhysicalContractInterface(
        count=4,
        trace_length=8,
        seed=5,
        target_id="fixture-target",
        firmware_id="fixture-firmware",
        controller_config_hash="fixture-config-hash",
    )
    backend = ControlledInterfaceAcquisitionBackend(
        interface,
        count=4,
        trace_length=8,
        labels=np.asarray([0, 1, 0, 1], dtype=np.uint8),
        session_id=interface.session_id,
        allocation_id="physical-allocation-1",
        label_stream_fingerprint="labels",
    )
    root = tmp_path / "physical"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config, sort_keys=True))
    writer = ShardWriter(root, max_samples_per_shard=2)
    for sample in backend.samples():
        writer.add(sample.trace, sample.label, sample.metadata)
    writer.finalize()
    manifest = write_dataset_manifest(
        root,
        config=config,
        condition="null",
        shard_infos=validate_all_shards(root),
        label_stream_fingerprint="labels",
        provenance=backend.manifest_provenance(condition="null"),
        acquisition_sessions=[backend.session_provenance()],
        dataset_purpose="physical_controlled_hardware",
        protocol_identity=interface.interface_name,
        protocol_hash=interface.protocol_hash,
    )
    return root, manifest


def test_physical_contract_fixture_is_accepted_only_when_every_chain_is_consistent(tmp_path):
    root, _manifest = _physical_fixture(tmp_path)
    accepted = validate_physical_evidence_dataset(root)
    assert accepted["dataset_purpose"] == "physical_controlled_hardware"


@pytest.mark.parametrize(
    "field",
    [
        "controlled_command_id",
        "controlled_command_sequence_id",
        "controlled_address_token",
        "controlled_acquisition_id",
        "controlled_topology_source",
    ],
)
def test_physical_contract_rejects_flattened_provenance_tampering(tmp_path, field):
    root, _manifest = _physical_fixture(tmp_path)
    payload = next(root.glob("shard-*.npz"))
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[field] = arrays[field].copy()
    arrays[field][0] = "tampered-identity"
    np.savez_compressed(payload, **arrays)
    sidecar = payload.with_suffix(".json")
    sidecar_record = json.loads(sidecar.read_text())
    import hashlib

    sidecar_record["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(sidecar_record))
    with pytest.raises(IntegrityError):
        validate_physical_evidence_dataset(root)
