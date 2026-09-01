"""Synthetic null, injected-signal, and label-permutation acquisition backends."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .base import AcquisitionBackend, Sample


def balanced_labels(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("at least two samples are required for a balanced binary dataset")
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

    name = "synthetic"

    def __post_init__(self) -> None:
        if self.start_index < 0:
            self.start_index = self.trace_length // 3
        if self.condition not in {"null", "injected", "shuffled"}:
            raise ValueError(f"unknown synthetic condition {self.condition!r}")
        if self.start_index < 0 or self.start_index + self.width > self.trace_length:
            raise ValueError("injected signal region must fit within trace")
        if self.session_count < 1 or self.device_count < 1:
            raise ValueError("session_count and device_count must be positive")
        self._labels = balanced_labels(self.count, self.seed)
        if self.condition == "shuffled":
            shuffle_rng = np.random.default_rng(
                self.permute_seed if self.permute_seed is not None else self.seed + 7919
            )
            shuffle_rng.shuffle(self._labels)
        self.label_stream_fingerprint = hashlib.sha256(self._labels.tobytes()).hexdigest()
        self._trace_rng = np.random.default_rng(self.seed + 1)
        self._cells = self.cells_per_device or max(
            1, (self.count + self.device_count - 1) // self.device_count
        )

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        if start_index < 0 or start_index > self.count:
            raise ValueError("start_index outside synthetic dataset")
        # Replay the deterministic prefix on recovery so the remaining trace stream is identical.
        for index in range(self.count):
            label = int(self._labels[index])
            trace = self._trace_rng.normal(0.0, 1.0, self.trace_length).astype(np.float32)
            if index < start_index:
                continue
            if self.condition == "injected":
                direction = 1.0 if label else -1.0
                trace[self.start_index : self.start_index + self.width] += (
                    direction * self.amplitude_sigma
                )
            group = index // 4
            device = group % self.device_count
            row = group // self.device_count
            cell = group
            yield Sample(
                trace=trace,
                label=label,
                metadata={
                    "sample_id": f"sample-{index:012d}",
                    "session_id": f"session-{group // 64 % self.session_count:04d}",
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
                    "seed_id": f"synthetic:{self.seed}",
                },
            )
