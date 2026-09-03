from __future__ import annotations

import numpy as np
import pytest

from sensetrace.excitation import CodedExcitationSchedule, ExcitationExecution
from sensetrace.packets import (
    EvidencePacket,
    PacketShardWriter,
    ProbeFragment,
    iter_packet_batches,
    iter_packets,
    validate_packet_stream,
)
from sensetrace.receiver import (
    NoiseResidualizer,
    ReceiverConfig,
    ReceiverTournament,
    evaluate_hybrid,
    packet_summary_features,
    train_hybrid,
)
from sensetrace.storage import _string_array
from sensetrace.synthetic_receiver import (
    evaluate_broken_packet_control,
    generate_broken_packet_control,
)
from sensetrace.worker03 import collect_worker03_inventory


def _packet(index: int, label: int = 0) -> EvidencePacket:
    return EvidencePacket(
        packet_id=f"packet-{index:04d}",
        target_reference="owned-target",
        acquisition_id="acquisition-0001",
        protocol_id="packet-test-v1",
        fragments=(
            ProbeFragment(
                fragment_id=f"packet-{index:04d}-observed",
                probe_type="scalar",
                probe_version="v1",
                sequence_position=0,
                target_role="target",
                payload=np.asarray([0.0], dtype=np.float32),
                quality=1.0,
            ),
            ProbeFragment(
                fragment_id=f"packet-{index:04d}-missing",
                probe_type="scalar",
                probe_version="v1",
                sequence_position=1,
                target_role="reference",
                payload=None,
                status="unavailable",
                model_eligible=False,
            ),
        ),
        controls={"code": "test"},
        provenance={"session": "audit-only"},
        label=label,
    )


def test_string_metadata_width_is_lossless_without_fixed_64k_reservation():
    values = _string_array(["short", "a" * 137])
    assert values.dtype.itemsize == 137 * 4
    assert values.dtype.itemsize < 65536 * 4


def test_packet_round_trip_is_streaming_and_preserves_zero_missing_distinction(tmp_path):
    writer = PacketShardWriter(tmp_path, max_packets_per_shard=2)
    writer.add(_packet(0, 0))
    writer.add(_packet(1, 1))
    writer.add(_packet(2, 0))
    writer.finalize()
    infos = validate_packet_stream(tmp_path)
    assert [info.packets for info in infos] == [2, 1]
    packets = list(iter_packets(tmp_path))
    assert packets[0].fragments[0].payload is not None
    assert float(packets[0].fragments[0].payload[0]) == 0.0
    assert packets[0].fragments[1].payload is None
    batch = next(
        iter_packet_batches(
            iter_packets(tmp_path),
            batch_size=2,
            max_fragments=2,
            max_payload_length=1,
        )
    )
    assert bool(batch.observed_mask[0, 0, 0])
    assert not batch.observed_mask[0, 1, 0]
    assert batch.labels is not None
    assert packet_summary_features(batch).dtype == np.float32


def test_noise_residualizer_is_unlabeled_and_records_source_fingerprint():
    packets = [_packet(0), _packet(1)]
    residualizer = NoiseResidualizer().fit(packets, dataset_fingerprint="immutable-noise-v1")
    transformed = residualizer.transform(packets[0])
    assert residualizer.state_record()["labels_used"] is False
    assert residualizer.state_record()["source_dataset_fingerprint"] == "immutable-noise-v1"
    payload = transformed.fragments[0].payload
    assert payload is not None
    assert float(payload[0]) == 0.0


def test_coded_excitation_keeps_requested_and_executed_code_distinct():
    schedule = CodedExcitationSchedule("worker03-test", "walsh", length=8, seed=4)
    request = schedule.request_record()
    assert request["execution_claim"].startswith("request only")
    execution = ExcitationExecution(
        schedule_fingerprint=schedule.fingerprint(),
        executed_code=schedule.requested_code(),
        start_clock=10,
        end_clock=20,
        core_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    )
    execution.validate(schedule)
    assert execution.as_dict()["executed_code"] == request["requested_code"]


def test_broken_packet_control_reports_positive_null_shuffle_and_ablation():
    control = generate_broken_packet_control(packet_count=32, fragment_count=8, seed=17)
    report = evaluate_broken_packet_control(control)
    assert (
        report["positive_control_balanced_accuracy"] > report["relation_ablated_balanced_accuracy"]
    )
    assert report["claim_boundary"].startswith("synthetic receiver validation")


def test_tournament_requires_immutable_dataset_and_split_fingerprints():
    record = ReceiverTournament("dataset-v1", "split-v1").as_dict()
    assert record["selection_policy"].startswith("validation split selects")
    with pytest.raises(ValueError, match="fingerprints"):
        ReceiverTournament("", "split-v1").validate()


def test_worker03_inventory_does_not_infer_missing_hardware_values(tmp_path):
    def command_runner(command, **_kwargs):
        if command == ["lscpu"]:
            return type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "Model name: Test CPU\nCPU(s): 2\nCore(s) per socket: 2\n",
                },
            )()
        return type("Result", (), {"returncode": 0, "stdout": ""})()

    inventory = collect_worker03_inventory(
        sysfs_root=tmp_path / "sys",
        proc_root=tmp_path / "proc",
        command_runner=command_runner,
    )
    assert inventory["target_match"] == "not_verified"
    assert inventory["observed"]["cache_line_size"] == "unavailable"
    assert inventory["observed"]["system_vendor"] == "unavailable"


def test_tiny_hybrid_trains_from_reopenable_batches_without_corpus_materialization():
    pytest.importorskip("torch")
    control = generate_broken_packet_control(packet_count=16, fragment_count=4, seed=9)
    packets = control.positive_packets

    def batches():
        return iter_packet_batches(
            packets,
            batch_size=4,
            max_fragments=4,
            max_payload_length=1,
            include_labels=True,
        )

    model, report = train_hybrid(
        batches,
        config=ReceiverConfig(latent_width=16, hidden_width=16, refinement_steps=2),
        excitation_width=0,
        epochs=1,
        seed=3,
    )
    assert report["parameter_count"] > 0
    assert report["memory_policy"].startswith("re-openable batch factory")
    assert evaluate_hybrid(model, batches())["sample_count"] == len(packets)
