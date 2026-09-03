from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sensetrace.acquisition.base import Sample
from sensetrace.attestation import (
    ControlledAdapterAttestation,
    evidence_source_record,
    require_adapter_attestation,
)
from sensetrace.errors import IntegrityError, SchemaError
from sensetrace.excitation import CodedExcitationSchedule
from sensetrace.experiment import (
    ExperimentProtocolError,
    ExperimentStateMachine,
    execute_coded_excitation,
    packet_from_sample,
    write_fragmented_packet_dataset,
)
from sensetrace.fragmented import (
    build_fragmented_split,
    fingerprint_fragmented_split,
    validate_fragmented_split,
)
from sensetrace.packets import EvidencePacket, ProbeFragment, iter_packets
from sensetrace.protocol import (
    worker03_fragmented_exact_host_protocol,
    worker03_fragmented_exact_host_protocol_hash,
)
from sensetrace.receiver import NoiseResidualizer, ReceiverTournament, execute_receiver_tournament


def _attestation() -> ControlledAdapterAttestation:
    return ControlledAdapterAttestation(
        adapter_identity="adapter-1",
        adapter_source_module="controller.adapter",
        adapter_driver_identity="driver-1",
        controller_identity="controller-1",
        controller_firmware_identity="firmware-1",
        controller_configuration_fingerprint="config-1",
        transport_type="pcie",
        target_identity="target-1",
        acquisition_session_identity="session-1",
        observed_hardware_capability_record={"channels": ["adc-0"]},
        calibration_identity="calibration-1",
        clock_identities={"command": "clock-1", "sampling": "clock-2"},
        topology_source_authority="controlled_hardware",
        adapter_binary_source_fingerprint="binary-1",
        code_commit="commit-1",
        host_inventory_fingerprint="host-1",
        created_at="2026-09-03T00:00:00Z",
        trust_assumptions=("adapter reports the attached target truthfully",),
    )


def _packet(index: int, *, payload_offset: float = 0.0) -> EvidencePacket:
    return EvidencePacket(
        packet_id=f"packet-{index:03d}",
        target_reference="target-1",
        acquisition_id="acquisition-1",
        protocol_id="worker03-fragmented-exact-host-v1",
        fragments=(
            ProbeFragment(
                fragment_id=f"fragment-{index:03d}-a",
                probe_type="cached_control",
                probe_version="native-v4",
                sequence_position=0,
                target_role="target",
                payload=np.asarray([float(index) + payload_offset], dtype=np.float32),
            ),
            ProbeFragment(
                fragment_id=f"fragment-{index:03d}-b",
                probe_type="dependency_chain",
                probe_version="native-v4",
                sequence_position=1,
                target_role="reference",
                payload=np.asarray([float(index) + 1.0 + payload_offset], dtype=np.float32),
            ),
        ),
        provenance={
            "session_id": f"session-{index // 2}",
            "acquisition_session_id": f"session-{index // 2}",
            "virtual_location_id": f"location-{index // 3}",
            "boot_id": "boot-1",
            "host_id": "host-1",
            "dimm_id": "dimm-1",
        },
        label=index % 2,
    )


def test_adapter_attestation_is_bound_and_missing_attestation_fails_closed():
    attestation = _attestation()
    source = evidence_source_record(
        internally_consistent=True,
        tier="adapter_attested_physical",
        adapter_attestation=attestation,
    )
    record = {"evidence_source": source}
    assert (
        require_adapter_attestation(
            record,
            adapter_identity="adapter-1",
            controller_identity="controller-1",
            firmware_identity="firmware-1",
            configuration_fingerprint="config-1",
            target_identity="target-1",
            acquisition_session_identity="session-1",
            host_inventory_fingerprint="host-1",
        )
        == attestation
    )
    with pytest.raises(ValueError, match="requires adapter attestation"):
        require_adapter_attestation(
            {"evidence_source": {}},
            controller_identity="controller-1",
            firmware_identity="firmware-1",
            configuration_fingerprint="config-1",
            target_identity="target-1",
            acquisition_session_identity="session-1",
            host_inventory_fingerprint="host-1",
        )
    with pytest.raises(ValueError, match="does not bind"):
        require_adapter_attestation(
            record,
            adapter_identity="incompatible-adapter",
            controller_identity="changed-controller",
            firmware_identity="firmware-1",
            configuration_fingerprint="config-1",
            target_identity="target-1",
            acquisition_session_identity="session-1",
            host_inventory_fingerprint="host-1",
        )


def test_native_evidence_cannot_be_upgraded_with_adapter_attestation():
    with pytest.raises(ValueError, match="native or synthetic"):
        evidence_source_record(
            internally_consistent=True,
            tier="native_exact_host",
            adapter_attestation=_attestation(),
        )


def test_synthetic_identity_cannot_satisfy_physical_attestation():
    attestation = _attestation()
    with pytest.raises(ValueError, match="synthetic identity"):
        replace(attestation, controller_identity="mock-controller").validate(require_physical=True)


def test_protocol_is_frozen_and_changes_change_fingerprint():
    config = {"experiment": {"name": "test", "seed": 1}, "data": {"samples": 8}}
    protocol = worker03_fragmented_exact_host_protocol(config)
    assert protocol["version"] == "worker03-fragmented-exact-host-v1"
    assert protocol["target"]["controlled_hardware_evidence"] is False
    assert (
        protocol["historical_gate"]
        == "commodity Phase 1A remains C: primitive unsuitable and is not reopened by this protocol"
    )
    changed = json.loads(json.dumps(config))
    changed["worker03_experiment"] = {"native_kernel_version": "native-v5"}
    assert worker03_fragmented_exact_host_protocol_hash(
        changed
    ) != worker03_fragmented_exact_host_protocol_hash(config)


def test_state_machine_rejects_out_of_order_and_repeat_test(tmp_path):
    machine = ExperimentStateMachine(tmp_path, "experiment-1", protocol_hash="unfrozen")
    with pytest.raises(ExperimentProtocolError, match="expected"):
        machine.transition("training")
    machine.freeze_protocol({"version": "worker03-fragmented-exact-host-v1"})
    machine.transition("inventory_verified")
    machine.transition("reference_acquisition")
    machine.transition("controlled_acquisition")
    machine.transition("evidence_finalized")
    machine.transition("split_frozen")
    machine.transition("training")
    machine.transition("validation_selection")
    machine.mark_test_evaluated()
    with pytest.raises(ExperimentProtocolError, match="one-time"):
        machine.mark_test_evaluated()


def test_packetization_keeps_multiple_fragments_and_audit_metadata_outside_arrays():
    packet = packet_from_sample(
        Sample(np.arange(8, dtype=np.float32), 1, {"sample_id": "sample-1", "session_id": "s1"}),
        packet_id="packet-1",
        protocol_id="worker03-fragmented-exact-host-v1",
        acquisition_id="acquisition-1",
    )
    assert len(packet.fragments) == 4
    values, observed, fragment_mask, excitation, quality = packet.model_arrays(
        max_fragments=4, max_payload_length=2
    )
    assert values.shape == (4, 2)
    assert observed.all()
    assert fragment_mask.all()
    assert excitation.shape == (4, 0)
    assert quality.shape == (4,)
    assert "source_sample_id" not in json.dumps(
        {"values": values.tolist(), "observed": observed.tolist()}
    )


def test_fragmented_packet_dataset_round_trip_is_immutable(tmp_path):
    config = {"experiment": {"name": "round-trip", "seed": 3}, "data": {"samples": 8}}
    protocol = worker03_fragmented_exact_host_protocol(config)
    packet = packet_from_sample(
        Sample(np.arange(8, dtype=np.float32), 0, {"sample_id": "sample-1"}),
        packet_id="packet-1",
        protocol_id=protocol["version"],
        acquisition_id="acquisition-1",
        include_label=False,
    )
    manifest = write_fragmented_packet_dataset(
        [packet], tmp_path, config=config, protocol=protocol, max_packets_per_shard=1
    )
    assert manifest["packet_count"] == 1
    loaded = list(iter_packets(tmp_path))
    assert loaded[0].label is None
    assert loaded[0].fragments[0].probe_version == "native-v4"
    with pytest.raises(IntegrityError, match="immutable"):
        write_fragmented_packet_dataset(
            [packet],
            tmp_path,
            config=config,
            protocol=protocol,
            max_packets_per_shard=1,
            purpose="changed",
        )


def test_fragmented_split_rejects_cross_partition_copied_packets():
    packets = [_packet(index) for index in range(12)]
    split = build_fragmented_split(
        packets,
        dataset_fingerprint="dataset-1",
        claim_level="level_1_exact_host_calibrated",
        seed=7,
    )
    assert (
        validate_fragmented_split(packets, split, dataset_fingerprint="dataset-1")["status"]
        == "pass"
    )
    source_id = packets[0].packet_id
    source_partition = next(
        partition
        for partition in ("train", "validation", "test")
        if source_id in split[f"{partition}_packet_ids"]
    )
    destination_partition = next(
        partition
        for partition in ("train", "validation", "test")
        if partition != source_partition and split[f"{partition}_packet_ids"]
    )
    copied_id = "packet-copied"
    mutated_split = json.loads(json.dumps(split))
    destination_id = mutated_split[f"{destination_partition}_packet_ids"].pop()
    mutated_split[f"{destination_partition}_packet_ids"].append(copied_id)
    mutated_split["split_fingerprint"] = fingerprint_fragmented_split(mutated_split)
    copied = [
        replace(packets[0], packet_id=copied_id) if packet.packet_id == destination_id else packet
        for packet in packets
    ]
    with pytest.raises(SchemaError, match="copied or duplicated"):
        validate_fragmented_split(copied, mutated_split, dataset_fingerprint="dataset-1")


def test_claim_level_does_not_infer_unseen_boot_from_one_boot():
    from sensetrace.fragmented import authorize_claim_level, claim_level_availability

    availability = claim_level_availability(_packet(index) for index in range(12))
    assert availability["levels"]["level_4_exact_host_unseen_boot"]["status"] == "unavailable"
    with pytest.raises(SchemaError, match="unavailable"):
        authorize_claim_level(
            (_packet(index) for index in range(12)), "level_4_exact_host_unseen_boot"
        )


def test_reference_residualizer_requires_unlabeled_packets():
    with pytest.raises(ValueError, match="unlabeled"):
        NoiseResidualizer().fit_reference([_packet(0)], dataset_fingerprint="reference-1")


def test_coded_execution_records_interruption_and_never_calls_it_compliant():
    schedule = CodedExcitationSchedule("schedule-1", "prbs", 4, 2)
    execution = execute_coded_excitation(schedule, step_runner=lambda index, _code, _op: index < 2)
    assert execution.compliance == "partial"
    assert execution.interrupted_positions == (2,)
    assert len(execution.executed_code) == 2


def test_receiver_tournament_selects_on_validation_and_streams_test_once():
    packets = [_packet(index) for index in range(24)]
    split = build_fragmented_split(
        packets,
        dataset_fingerprint="dataset-1",
        claim_level="level_1_exact_host_calibrated",
        seed=11,
    )
    tournament = ReceiverTournament(
        "dataset-1",
        split["split_fingerprint"],
        candidates=("logistic_regression", "weak_evidence_aggregator"),
    )
    report = execute_receiver_tournament(
        tournament,
        packet_factory=lambda: iter(packets),
        split=split,
        max_fragments=2,
        max_payload_length=1,
        maximum_training_packets=16,
        seed=11,
    )
    assert set(report["candidates"]) == set(tournament.candidates)
    assert report["selection"]["test_evaluation_count"] == 1
    assert report["selected"]["test"]["prediction_retention"].startswith("none;")
    assert report["controls"]["shuffled_labels"]["status"] == "evaluated"
