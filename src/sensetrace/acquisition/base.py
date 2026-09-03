"""Stable interface between acquisition and persistence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sample:
    trace: np.ndarray
    label: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class RecoveryPolicy:
    """Backend recovery contract used by the runner instead of backend names."""

    allow_resume: bool
    deterministic_replay: bool
    continuity_requirement: str


class AcquisitionBackend:
    name = "abstract"
    recovery_policy = RecoveryPolicy(
        allow_resume=False,
        deterministic_replay=False,
        continuity_requirement="backend has not declared resumable identity continuity",
    )

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources; backends without resources are no-ops."""

        return None
