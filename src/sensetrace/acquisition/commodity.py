"""Safe placeholder for future commodity-DRAM telemetry acquisition.

This milestone deliberately does not expose raw physical addresses, refresh
disabling, voltage changes, or disturbance loops. A future backend must prove
its isolation and measurement contract before it is enabled.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import AcquisitionBackend, Sample


class CommodityDramBackend(AcquisitionBackend):
    name = "commodity-dram"

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        raise RuntimeError(
            "commodity DRAM acquisition is intentionally unavailable in the initial safe milestone; "
            "use SyntheticBackend for Phase 0"
        )
