from __future__ import annotations

import json

import pytest

from sensetrace.acquisition.synthetic import SyntheticBackend
from sensetrace.datasets import build_feature_matrix, load_dataset, write_dataset_manifest
from sensetrace.errors import ForbiddenFeatureError, IntegrityError, SchemaError
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
    assert FeaturePolicy().validate([]) is None


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
