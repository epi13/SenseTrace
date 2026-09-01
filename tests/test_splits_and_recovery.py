from __future__ import annotations

import numpy as np
import pytest

from sensetrace.acquisition.commodity import CommodityDramBackend
from sensetrace.acquisition.synthetic import SyntheticBackend
from sensetrace.datasets import build_feature_matrix
from sensetrace.errors import JournalCorruptionError
from sensetrace.journal import Journal
from sensetrace.recovery import recovery_test
from sensetrace.splits import grouped_split, partition_indices, read_split, write_split


def test_grouped_split_keeps_groups_together(tmp_path):
    metadata: dict[str, np.ndarray] = {
        "sample_id": np.asarray([f"sample-{i:012d}" for i in range(40)]),
        "session_id": np.asarray([f"session-{i // 8}" for i in range(40)]),
        "device_id": np.asarray([f"device-{i % 2}" for i in range(40)]),
        "row_id": np.asarray([f"row-{i // 4}" for i in range(40)]),
        "cell_or_offset_id": np.asarray([f"cell-{i // 2}" for i in range(40)]),
    }
    split = grouped_split(
        metadata,
        dataset_fingerprint="dataset-hash",
        group_keys=["session_id", "row_id", "cell_or_offset_id"],
        seed=12,
    )
    path = tmp_path / "split.json"
    write_split(path, split)
    loaded = read_split(path, expected_dataset_fingerprint="dataset-hash")
    partitions = partition_indices(metadata, loaded)
    membership = {}
    for part, indices in partitions.items():
        for index in indices:
            group = (
                metadata["session_id"][index],
                metadata["row_id"][index],
                metadata["cell_or_offset_id"][index],
            )
            assert group not in membership or membership[group] == part
            membership[group] = part
    assert set(np.concatenate(list(partitions.values()))) == set(range(40))


def test_backend_resume_replays_identical_trace_stream():
    full = list(
        SyntheticBackend(20, 32, 19, condition="injected", start_index=8, width=4).samples()
    )
    resumed = list(
        SyntheticBackend(20, 32, 19, condition="injected", start_index=8, width=4).samples(
            start_index=7
        )
    )
    assert np.array_equal(full[7].trace, resumed[0].trace)
    assert full[7].label == resumed[0].label


def test_recovery_quarantines_temp_and_avoids_duplicate_ranges():
    result = recovery_test()
    assert result["passed"]
    assert result["second_run"]["quarantined_temporary_shards"]
    assert len(result["finalized_next_sample_indices"]) == len(
        set(result["finalized_next_sample_indices"])
    )


def test_journal_rejects_non_trailing_corruption(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event":"ok"}\nnot-json\n')
    with pytest.raises(JournalCorruptionError):
        Journal(path).read()


def test_journal_recovers_trailing_partial_and_remains_appendable(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = Journal(path)
    journal.append("first", value=1)
    valid_prefix = path.read_bytes()
    path.write_bytes(valid_prefix + b'{"event":"crashed","value":')

    recovered = journal.recover()
    assert [event["event"] for event in recovered.events] == ["first", "journal_recovery"]
    recovery = recovered.events[-1]
    assert recovery["discarded_byte_count"] == len(b'{"event":"crashed","value":')
    assert recovery["previous_file_size"] == len(valid_prefix) + recovery["discarded_byte_count"]
    assert recovery["recovered_file_size"] == len(valid_prefix)
    journal.append("after_recovery", value=2)
    assert [event["event"] for event in journal.read().events] == [
        "first",
        "journal_recovery",
        "after_recovery",
    ]


def test_journal_recovery_handles_invalid_utf8_tail_and_is_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = Journal(path)
    journal.append("first")
    prefix = path.read_bytes()
    tail = b'{"event":"bad","text":"\xff'
    path.write_bytes(prefix + tail)
    recovered = journal.recover()
    assert recovered.events[-1]["discarded_bytes_sha256"]
    size_after_first_recovery = path.stat().st_size
    recovered_again = journal.recover()
    assert len(recovered_again.events) == len(recovered.events)
    assert path.stat().st_size == size_after_first_recovery


def test_shuffled_backend_preserves_observations_and_changes_only_labels():
    injected = list(
        SyntheticBackend(64, 32, 19, condition="injected", amplitude_sigma=0.5).samples()
    )
    shuffled = list(
        SyntheticBackend(
            64, 32, 19, condition="shuffled", amplitude_sigma=0.5, permute_seed=100
        ).samples()
    )
    pairs = zip(injected, shuffled, strict=True)
    assert all(np.array_equal(left.trace, right.trace) for left, right in pairs)
    pairs = zip(injected, shuffled, strict=True)
    assert all(left.metadata == right.metadata for left, right in pairs)
    pairs = zip(injected, shuffled, strict=True)
    assert any(left.label != right.label for left, right in pairs)


def test_commodity_backend_is_safe_and_keeps_digital_value_out_of_features():
    backend = CommodityDramBackend(
        count=4,
        trace_length=32,
        word_count=8,
        lock_memory=False,
        cache_control="none",
    )
    try:
        samples = list(backend.samples())
    finally:
        backend.close()
    assert [sample.label for sample in samples].count(0) == 2
    assert all(sample.metadata["digital_verification_passed"] is True for sample in samples)
    traces = np.stack([sample.trace for sample in samples])
    metadata = {
        key: np.asarray([sample.metadata[key] for sample in samples]) for key in samples[0].metadata
    }
    features = build_feature_matrix(traces, metadata)
    assert features.shape == (4, 69)
