"""Synthetic null, injected-signal, and label-permutation acquisition backends."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .base import AcquisitionBackend, RecoveryPolicy, Sample


def balanced_labels(count: int, seed: int) -> np.ndarray:
    if count < 2 or count % 2:
        raise ValueError("an exactly balanced binary dataset requires a positive even sample count")
    labels = np.concatenate(
        [np.zeros(count // 2, dtype=np.uint8), np.ones(count - count // 2, dtype=np.uint8)]
    )
    np.random.default_rng(seed).shuffle(labels)
    return labels


@dataclass
class SyntheticBackend(AcquisitionBackend):
    count: int
    trace_length: int
    seed: int
    condition: str = "null"
    amplitude_sigma: float = 0.1
    start_index: int = -1
    width: int = 8
    session_count: int = 4
    device_count: int = 2
    cells_per_device: int | None = None
    permute_seed: int | None = None
    acquisition_seed: int | None = None
    label_seed: int | None = None
    trace_seed: int | None = None
    dataset_id: str | None = None
    balance_mode: str = "global_balance_only"
    observations_per_location: int = 4

    name = "synthetic"
    recovery_policy = RecoveryPolicy(
        allow_resume=True,
        deterministic_replay=True,
        continuity_requirement="deterministic synthetic seed and dataset identity",
    )

    def __post_init__(self) -> None:
        if self.start_index < 0:
            self.start_index = self.trace_length // 3
        if self.condition not in {"null", "injected", "shuffled"}:
            raise ValueError(f"unknown synthetic condition {self.condition!r}")
        if self.start_index < 0 or self.start_index + self.width > self.trace_length:
            raise ValueError("injected signal region must fit within trace")
        if self.session_count < 1 or self.device_count < 1:
            raise ValueError("session_count and device_count must be positive")
        if self.balance_mode not in {"global_balance_only", "group_stratified_balance"}:
            raise ValueError(f"unknown synthetic balance mode {self.balance_mode!r}")
        if self.observations_per_location < 2 or self.observations_per_location % 2:
            raise ValueError("observations_per_location must be a positive even number")
        self.acquisition_seed = (
            self.seed if self.acquisition_seed is None else self.acquisition_seed
        )
        self.label_seed = self.seed if self.label_seed is None else self.label_seed
        self.trace_seed = self.seed + 1 if self.trace_seed is None else self.trace_seed
        assert self.acquisition_seed is not None
        assert self.label_seed is not None
        assert self.trace_seed is not None
        acquisition_seed = self.acquisition_seed
        trace_seed = self.trace_seed
        self.synthetic_dataset_id = self.dataset_id or f"synthetic-dataset-{acquisition_seed:010d}"
        self._source_labels = self._make_labels()
        self._labels = self._source_labels.copy()
        self.permutation_seed = (
            self.permute_seed if self.permute_seed is not None else self.seed + 7919
        )
        self.permutation_strata = (
            "synthetic_location_id"
            if self.balance_mode == "group_stratified_balance"
            else "synthetic_dataset_id"
        )
        self.permutation_fingerprint: str | None = None
        if self.condition == "shuffled":
            permutation = np.arange(self.count, dtype=np.int64)
            permutation_rng = np.random.default_rng(self.permutation_seed)
            stratum_size = (
                self.observations_per_location
                if self.permutation_strata == "synthetic_location_id"
                else self.count
            )
            for start in range(0, self.count, stratum_size):
                stop = min(start + stratum_size, self.count)
                indexes = np.arange(start, stop, dtype=np.int64)
                permutation[indexes] = permutation_rng.permutation(indexes)
            self._labels = self._source_labels[permutation]
            self.permutation_fingerprint = hashlib.sha256(
                np.asarray(permutation, dtype=np.int64).tobytes()
            ).hexdigest()
        self.original_label_stream_fingerprint = hashlib.sha256(
            self._source_labels.tobytes()
        ).hexdigest()
        self.label_stream_fingerprint = hashlib.sha256(self._labels.tobytes()).hexdigest()
        self._trace_rng = np.random.default_rng(trace_seed)
        self._acquisition_rng = np.random.default_rng(acquisition_seed)
        self._cells = self.cells_per_device or max(
            1, (self.count + self.device_count - 1) // self.device_count
        )

    def _make_labels(self) -> np.ndarray:
        assert self.label_seed is not None
        label_seed = self.label_seed
        if self.balance_mode == "global_balance_only":
            return balanced_labels(self.count, int(label_seed))
        labels = np.empty(self.count, dtype=np.uint8)
        rng = np.random.default_rng(label_seed)
        for start in range(0, self.count, self.observations_per_location):
            stop = min(start + self.observations_per_location, self.count)
            size = stop - start
            if size != self.observations_per_location:
                labels[start:stop] = balanced_labels(size, int(rng.integers(0, 2**32)))
                continue
            local = np.concatenate(
                [
                    np.zeros(size // 2, dtype=np.uint8),
                    np.ones(size // 2, dtype=np.uint8),
                ]
            )
            rng.shuffle(local)
            labels[start:stop] = local
        return labels

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        if start_index < 0 or start_index > self.count:
            raise ValueError("start_index outside synthetic dataset")
        # Replay the deterministic prefix on recovery so the remaining trace stream is identical.
        for index in range(self.count):
            label = int(self._labels[index])
            trace = self._trace_rng.normal(0.0, 1.0, self.trace_length).astype(np.float32)
            if index < start_index:
                continue
            if self.condition in {"injected", "shuffled"}:
                # Shuffled is a post-acquisition label permutation.  Its trace
                # is generated from the same hidden source label as injected,
                # while the persisted label is deliberately decoupled.
                source_label = int(self._source_labels[index])
                direction = 1.0 if source_label else -1.0
                trace[self.start_index : self.start_index + self.width] += (
                    direction * self.amplitude_sigma
                )
            location = index // self.observations_per_location
            device = location % self.device_count
            row = location // self.device_count
            cell = location
            session = location // max(
                1, (self._cells + self.session_count - 1) // self.session_count
            )
            block = location // 16
            yield Sample(
                trace=trace,
                label=label,
                metadata={
                    "sample_id": f"sample-{index:012d}",
                    "session_id": f"session-{session:04d}",
                    "synthetic_dataset_id": self.synthetic_dataset_id,
                    "synthetic_session_id": f"{self.synthetic_dataset_id}-session-{session:04d}",
                    "synthetic_location_id": f"{self.synthetic_dataset_id}-location-{location:08d}",
                    "synthetic_block_id": f"{self.synthetic_dataset_id}-block-{block:08d}",
                    "acquisition_seed": int(self.acquisition_seed or self.seed),
                    "label_seed": int(self.label_seed or self.seed),
                    "trace_seed": int(self.trace_seed or self.seed + 1),
                    "device_id": f"device-{device:02d}",
                    "bank_id": f"bank-{(index // 16) % 4:02d}",
                    "row_id": f"row-{row:08d}",
                    "cell_or_offset_id": f"cell-{device:02d}-{cell:08d}",
                    "trial_index": index,
                    "temperature_c": 35.0 + (index % 7) * 0.01,
                    "vdd_v": 1.2,
                    "refresh_age_ns": 1000 + (index % 13),
                    "wait_ns": 500,
                    "channel_mask": "trace0",
                    "seed_id": (
                        f"synthetic:acquisition={self.acquisition_seed}:"
                        f"label={self.label_seed}:trace={self.trace_seed}"
                    ),
                },
            )
