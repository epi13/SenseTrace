"""Phase 2 controlled-memory-interface boundary.

No FPGA/controller hardware is connected. This module defines the
software/research contract a future controlled memory controller must satisfy
before its topology claims can enter SenseTrace evidence, plus a
synthetic/mock backend so the data contract and recovery path can be tested
now.

Critical invariant: physical topology (row/bank/channel/rank/device identity)
may only exist when genuinely supplied by the controlled hardware interface.
It must never be synthesized from virtual addresses.
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from .base import AcquisitionBackend, Sample

TopologySource = Literal["controlled_hardware", "unavailable"]


@dataclass(frozen=True)
class ControlledMemoryTopology:
    """Physical topology supplied exclusively by controlled hardware."""

    source: TopologySource
    channel: str = "unavailable"
    rank: str = "unavailable"
    bank_group: str = "unavailable"
    bank: str = "unavailable"
    row: str = "unavailable"
    column: str = "unavailable"
    device: str = "unavailable"
    dimm: str = "unavailable"

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def validate(self) -> None:
        """Fail closed: any concrete topology field requires hardware sourcing."""
        fields = self.as_dict()
        concrete = {
            key: value
            for key, value in fields.items()
            if key != "source" and value not in {"unavailable", "unknown"}
        }
        if self.source != "controlled_hardware" and concrete:
            raise ValueError(
                "controlled-memory topology fields require source='controlled_hardware'; "
                f"got {sorted(concrete)} from {self.source!r}"
            )
        if self.source == "controlled_hardware" and not concrete:
            raise ValueError(
                "source='controlled_hardware' must supply at least one concrete topology field"
            )


@dataclass(frozen=True)
class ControlledCommand:
    """One externally controlled memory command."""

    command_id: str
    kind: str  # e.g. "activate", "read", "write", "precharge", "refresh"
    address_token: str  # opaque hardware token; never a virtual address
    issued_at_hardware_ticks: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledAcquisitionProvenance:
    """Acquisition provenance for the controlled-memory interface."""

    experiment_target_id: str
    controller_firmware_id: str
    controller_config_hash: str
    device_identity: str
    dimm_identity: str
    calibration_state: str
    hardware_clock_id: str
    acquisition_trigger: str

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


class ControlledMemoryInterface(ABC):
    """Abstract boundary for a future controlled memory controller."""

    interface_name = "controlled-memory-interface-v1"

    @abstractmethod
    def provenance(self) -> ControlledAcquisitionProvenance:
        raise NotImplementedError

    @abstractmethod
    def topology_for(self, address_token: str) -> ControlledMemoryTopology:
        """Return hardware-supplied topology for an opaque hardware token."""
        raise NotImplementedError

    @abstractmethod
    def issue(self, command: ControlledCommand) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class SyntheticMockControlledBackend(AcquisitionBackend):
    """Test-only controlled-interface backend with honest unavailable topology.

    Uses generated random traces and labels so the Phase 2 data contract,
    grouping, and recovery path can be exercised before hardware arrives.
    Topology is always unavailable; any attempt to derive it from the virtual
    buffer raises instead of synthesizing row/bank identities.
    """

    count: int = 64
    trace_length: int = 32
    seed: int = 1337
    target_id: str = "mock-controlled-target-0000"
    firmware_id: str = "mock-controller-firmware-v0"
    controller_config_hash: str = "mock-config-unavailable"
    name: str = "mock-controlled-memory"

    def __post_init__(self) -> None:
        if self.count < 2 or self.trace_length < 1:
            raise ValueError("count must be >= 2 and trace_length must be positive")
        self._rng = np.random.default_rng(self.seed)
        self._labels = np.asarray(
            [0] * (self.count // 2) + [1] * (self.count - self.count // 2),
            dtype=np.uint8,
        )
        order = np.random.default_rng(self.seed + 1).permutation(self.count)
        self._labels = self._labels[order]
        self._session_id = f"mock-controlled-session-{uuid.uuid4().hex}"
        self._allocation_id = f"mock-controlled-allocation-{uuid.uuid4().hex}"
        self._started_at = datetime.now(UTC).isoformat()
        self._closed = False

    def topology_for_virtual_offset(self, _offset: int) -> ControlledMemoryTopology:
        """Virtual offsets carry no physical topology. Always unavailable."""
        return ControlledMemoryTopology(source="unavailable")

    def derive_topology_from_virtual_address(self, _address: int) -> ControlledMemoryTopology:
        """Forbidden constructor: virtual addresses must never yield topology."""
        raise ValueError(
            "physical topology cannot be derived from a virtual address; "
            "it may only be supplied by the controlled hardware interface"
        )

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        if start_index < 0 or start_index > self.count:
            raise ValueError("start_index outside mock controlled dataset")
        for index in range(start_index, self.count):
            label = int(self._labels[index])
            trace = self._rng.normal(0.0, 1.0, self.trace_length).astype(np.float32)
            topology = self.topology_for_virtual_offset(index)
            topology.validate()
            provenance = ControlledAcquisitionProvenance(
                experiment_target_id=self.target_id,
                controller_firmware_id=self.firmware_id,
                controller_config_hash=self.controller_config_hash,
                device_identity="mock-device-unknown",
                dimm_identity="mock-dimm-unknown",
                calibration_state="mock-uncalibrated",
                hardware_clock_id="mock-clock",
                acquisition_trigger="mock-software-trigger",
            )
            yield Sample(
                trace=trace,
                label=label,
                metadata={
                    "sample_id": f"{self._session_id}:sample-{index:012d}",
                    "session_id": self._session_id,
                    "acquisition_session_id": self._session_id,
                    "allocation_id": self._allocation_id,
                    "virtual_location_id": f"{self._session_id}:virtual-{index:08d}",
                    "trial_index": index,
                    "device_id": "device-unknown",
                    "bank_id": "bank-unknown",
                    "row_id": "row-unknown",
                    "cell_or_offset_id": f"{self._session_id}:offset-{index:08d}",
                    "controlled_topology": str(topology.as_dict()),
                    "controlled_topology_source": topology.source,
                    "controlled_provenance": str(provenance.as_dict()),
                    "controller_firmware_id": self.firmware_id,
                    "controller_config_hash": self.controller_config_hash,
                    "label_stream_fingerprint": hashlib.sha256(self._labels.tobytes()).hexdigest(),
                    "measurement_primitive": "controlled-memory-interface-mock",
                    "physical_observation_semantics": (
                        "synthetic mock trace; no physical memory claim"
                    ),
                    "session_started_at": self._started_at,
                },
            )

    def close(self) -> None:
        self._closed = True
