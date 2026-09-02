from __future__ import annotations

import json

import pytest

from sensetrace.acquisition.commodity import CommodityDramBackend
from sensetrace.acquisition.primitive import TimingPerturbationCalibration
from sensetrace.acquisition.synthetic import SyntheticBackend
from sensetrace.datasets import build_feature_matrix, load_dataset, write_dataset_manifest
from sensetrace.errors import ForbiddenFeatureError, IntegrityError, SchemaError
from sensetrace.protocol import phase1a_commodity_baseline_protocol_hash
from sensetrace.schema import FeaturePolicy
from sensetrace.storage import ShardWriter, validate_shard


def _write_fixture(root, count=12):
    writer = ShardWriter(root, shard_target_mb=1, max_samples_per_shard=4)
    backend = SyntheticBackend(count=count, trace_length=32, seed=9)
    for sample in backend.samples():
        writer.add(sample.trace, sample.label, sample.metadata)
    writer.finalize()
    config = {"experiment": {"name": "fixture", "seed": 9}}
    manifest = write_dataset_manifest(
        root,
        config=config,
        condition="null",
        shard_infos=[validate_shard(path) for path in sorted(root.glob("shard-*.npz"))],
        label_stream_fingerprint=backend.label_stream_fingerprint,
    )
    return manifest


def test_round_trip_and_exact_balance(tmp_path):
    manifest = _write_fixture(tmp_path)
    traces, labels, metadata, shards, loaded_manifest = load_dataset(tmp_path)
    assert manifest["dataset_fingerprint"] == loaded_manifest["dataset_fingerprint"]
    assert traces.shape == (12, 32)
    assert labels.sum() == 6
    assert len(shards) == 3
    assert set(metadata) >= {"sample_id", "row_id", "trial_index"}


def test_corrupt_payload_is_rejected(tmp_path):
    _write_fixture(tmp_path)
    payload = sorted(tmp_path.glob("shard-*.npz"))[0]
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        validate_shard(payload)


def test_manifest_change_is_rejected(tmp_path):
    _write_fixture(tmp_path)
    path = tmp_path / "dataset.json"
    manifest = json.loads(path.read_text())
    manifest["rows"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(IntegrityError, match="row count"):
        load_dataset(tmp_path)


def test_identity_feature_policy_fails_loudly(tmp_path):
    _write_fixture(tmp_path)
    traces, _labels, metadata, _shards, _manifest = load_dataset(tmp_path)
    with pytest.raises(ForbiddenFeatureError, match="row_id"):
        build_feature_matrix(traces, metadata, feature_fields=["row_id"])
    with pytest.raises(SchemaError, match="unsupported"):
        build_feature_matrix(traces, metadata, feature_fields=["filename"])
    encoded = build_feature_matrix(traces, metadata, feature_fields=["row_id"], allow_identity=True)
    assert encoded.shape[0] == traces.shape[0]
    FeaturePolicy().validate([])


def test_overlapping_finalized_ranges_are_rejected(tmp_path):
    _write_fixture(tmp_path)
    first_payload = sorted(tmp_path.glob("shard-*.npz"))[0]
    first_sidecar = first_payload.with_suffix(".json")
    duplicate_payload = tmp_path / "shard-999999.npz"
    duplicate_sidecar = tmp_path / "shard-999999.json"
    duplicate_payload.write_bytes(first_payload.read_bytes())
    duplicate_sidecar.write_bytes(first_sidecar.read_bytes())
    from sensetrace.storage import validate_all_shards

    with pytest.raises(IntegrityError, match="overlapping"):
        validate_all_shards(tmp_path)


def test_baseline_loader_rejects_calibration_contamination(tmp_path):
    backend = CommodityDramBackend(
        count=4,
        location_count=1,
        trials_per_location=4,
        trace_length=8,
        word_count=4,
        lock_memory=False,
        cache_control="none",
        use_native_kernel=False,
        calibration_context=TimingPerturbationCalibration("contaminated", 32),
    )
    writer = ShardWriter(tmp_path, shard_target_mb=1, max_samples_per_shard=4)
    try:
        for sample in backend.samples():
            writer.add(sample.trace, sample.label, sample.metadata)
    finally:
        backend.close()
    writer.finalize()
    write_dataset_manifest(
        tmp_path,
        config={"experiment": {"name": "contaminated", "seed": 1}},
        condition="paired_single_bit",
        shard_infos=[validate_shard(path) for path in sorted(tmp_path.glob("shard-*.npz"))],
        label_stream_fingerprint="fixture",
        provenance={"protocol_identity": "phase1a-commodity-baseline-v1"},
    )
    with pytest.raises(IntegrityError, match="calibration contamination"):
        load_dataset(tmp_path)


def _write_physical_fixture(
    root,
    *,
    dataset_purpose="physical_phase1a",
    protocol_identity="phase1a-commodity-baseline-v1",
    mutate_metadata=None,
    mutate_provenance=None,
):
    config = {
        "experiment": {"name": "physical-fixture", "seed": 1},
        "data": {"trace_length": 32},
        "phase1a": {"cache_control": "none", "operation": "memory_read"},
    }
    protocol_hash = phase1a_commodity_baseline_protocol_hash(config)
    backend = CommodityDramBackend(
        count=4,
        location_count=1,
        trials_per_location=4,
        trace_length=8,
        word_count=4,
        lock_memory=False,
        cache_control="none",
        use_native_kernel=False,
        acquisition_session_id="physical-session",
        protocol_identity=protocol_identity,
        protocol_hash=protocol_hash,
    )
    writer = ShardWriter(root, shard_target_mb=1, max_samples_per_shard=4)
    session = backend.session_provenance()
    session.update(
        {
            "protocol_identity": protocol_identity,
            "protocol_hash": protocol_hash,
            "acquisition_scope": "physical Phase 1A commodity baseline",
        }
    )
    try:
        for sample in backend.samples():
            if mutate_metadata is not None:
                mutate_metadata(sample.metadata)
            writer.add(sample.trace, sample.label, sample.metadata)
    finally:
        backend.close()
    writer.finalize()
    provenance = {
        "backend": "CommodityDramBackend",
        "protocol_identity": protocol_identity,
        "protocol_hash": protocol_hash,
        "artificial_timing_perturbation": {
            "allowed": False,
            "timing_perturbation_cycles": 0,
            "timing_perturbation_label": 1,
            "label_correlated": False,
            "applied": False,
            "calibration_namespace": "forbidden",
            "physical_phase1a_forbidden": True,
        },
    }
    if mutate_provenance is not None:
        mutate_provenance(provenance)
    return write_dataset_manifest(
        root,
        config=config,
        condition="paired_single_bit",
        shard_infos=[validate_shard(path) for path in sorted(root.glob("shard-*.npz"))],
        label_stream_fingerprint="fixture",
        class_balance={"0": 2, "1": 2},
        provenance=provenance,
        acquisition_sessions=[session],
        dataset_purpose=dataset_purpose,
        protocol_identity=protocol_identity,
        protocol_hash=protocol_hash,
    )


def test_physical_loader_requires_explicit_identity_and_purpose(tmp_path):
    _write_physical_fixture(tmp_path, dataset_purpose="generic")
    with pytest.raises(IntegrityError, match="not explicitly marked"):
        load_dataset(tmp_path, expected_purpose="physical_phase1a")


def test_physical_loader_rejects_wrong_protocol_identity(tmp_path):
    _write_physical_fixture(tmp_path, protocol_identity="other-protocol")
    with pytest.raises(IntegrityError, match="requires protocol identity"):
        load_dataset(tmp_path, expected_purpose="physical_phase1a")


def test_physical_loader_rejects_manifest_shard_protocol_disagreement(tmp_path):
    _write_physical_fixture(
        tmp_path,
        mutate_metadata=lambda metadata: metadata.update(
            {"protocol_hash": "different-shard-protocol-hash"}
        ),
    )
    with pytest.raises(IntegrityError, match="protocol_hash.*not uniformly"):
        load_dataset(tmp_path, expected_purpose="physical_phase1a")


def test_physical_loader_rejects_inserted_calibration_metadata(tmp_path):
    _write_physical_fixture(
        tmp_path,
        mutate_metadata=lambda metadata: metadata.update(
            {"calibration_namespace": "measurement-calibration"}
        ),
    )
    with pytest.raises(IntegrityError, match="calibration_namespace"):
        load_dataset(tmp_path, expected_purpose="physical_phase1a")
