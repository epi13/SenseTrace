from __future__ import annotations

import numpy as np
import pytest

from sensetrace.acquisition.synthetic import SyntheticBackend
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
