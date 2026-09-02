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


class AcquisitionBackend:
    name = "abstract"

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources; backends without resources are no-ops."""

        return None
