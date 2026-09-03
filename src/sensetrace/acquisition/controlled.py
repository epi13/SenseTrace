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
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from .base import AcquisitionBackend, Sample

TopologySource = Literal["controlled_hardware", "unavailable"]


def _is_missing_identity(value: str) -> bool:
    """Return whether a required identity is an explicit placeholder."""
    return value.strip().lower() in {"", "unavailable", "unknown"}


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
    command_sequence_id: str = "unavailable"
    refresh_relationship: str = "unavailable"
    timing_provenance: str = "unavailable"
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "command_id": self.command_id,
            "kind": self.kind,
            "address_token": self.address_token,
            "command_sequence_id": self.command_sequence_id,
            "refresh_relationship": self.refresh_relationship,
            "timing_provenance": self.timing_provenance,
        }
        missing = sorted(key for key, value in required.items() if _is_missing_identity(value))
        if missing:
            raise ValueError(f"controlled command provenance is missing {missing}")
        if self.issued_at_hardware_ticks < 0:
            raise ValueError("controlled command timing must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
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
    acquisition_configuration_hash: str = "unavailable"
    trigger_identity: str = "unavailable"
    timing_provenance: str = "unavailable"
    refresh_relationship: str = "unavailable"
    command_provenance: str = "unavailable"

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def validate(self) -> None:
        required = {
            "experiment_target_id": self.experiment_target_id,
            "controller_firmware_id": self.controller_firmware_id,
            "controller_config_hash": self.controller_config_hash,
            "device_identity": self.device_identity,
            "dimm_identity": self.dimm_identity,
            "calibration_state": self.calibration_state,
            "hardware_clock_id": self.hardware_clock_id,
            "acquisition_trigger": self.acquisition_trigger,
            "acquisition_configuration_hash": self.acquisition_configuration_hash,
            "trigger_identity": self.trigger_identity,
            "timing_provenance": self.timing_provenance,
            "refresh_relationship": self.refresh_relationship,
            "command_provenance": self.command_provenance,
        }
        missing = sorted(key for key, value in required.items() if _is_missing_identity(value))
        if missing:
            raise ValueError(f"controlled acquisition provenance is missing {missing}")


@dataclass(frozen=True)
class ControlledTraceChannel:
    """One controller-declared analog or digital acquisition channel."""

    channel_id: str
    channel_kind: Literal["analog", "digital"]
    units: str
    sampling_clock_id: str
    calibration_id: str

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.channel_id,
                self.units,
                self.sampling_clock_id,
                self.calibration_id,
            )
        ):
            raise ValueError("controlled trace channel identity is incomplete")


@dataclass(frozen=True)
class ControlledTraceAcquisition:
    """Trace/trigger contract a real controlled interface must produce."""

    acquisition_id: str
    trigger_id: str
    trigger_hardware_ticks: int
    hardware_clock_id: str
    timing_uncertainty_ticks: int
    channels: tuple[ControlledTraceChannel, ...]
    refresh_relationship: str
    command_sequence_id: str

    def validate(self) -> None:
        if not self.acquisition_id or not self.trigger_id or not self.hardware_clock_id:
            raise ValueError("controlled trace acquisition identity is incomplete")
        if self.trigger_hardware_ticks < 0 or self.timing_uncertainty_ticks < 0:
            raise ValueError("controlled trace timing values must be non-negative")
        if not self.channels:
            raise ValueError(
                "controlled trace acquisition requires at least one identified channel"
            )
        trace_provenance = {
            "refresh_relationship": self.refresh_relationship,
            "command_sequence_id": self.command_sequence_id,
        }
        missing = sorted(
            key for key, value in trace_provenance.items() if _is_missing_identity(value)
        )
        if missing:
            raise ValueError(f"controlled trace provenance is missing {missing}")
        for channel in self.channels:
            channel.validate()

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ControlledCommandResult:
    """Typed result binding a command to topology, trace, and completion state."""

    command: ControlledCommand
    status: Literal["complete", "unsupported", "failed"]
    topology: ControlledMemoryTopology
    provenance: ControlledAcquisitionProvenance
    completed_at_hardware_ticks: int
    acquisition: ControlledTraceAcquisition | None = None
    failure: str | None = None

    def validate(self) -> None:
        self.command.validate()
        self.topology.validate()
        self.provenance.validate()
        if self.completed_at_hardware_ticks < self.command.issued_at_hardware_ticks:
            raise ValueError("controlled command completion precedes command issue")
        if self.status == "complete":
            if self.acquisition is None:
                raise ValueError("complete controlled command requires trace acquisition")
            if self.failure is not None:
                raise ValueError("complete controlled command cannot include failure")
            self.acquisition.validate()
            if self.acquisition.command_sequence_id != self.command.command_sequence_id:
                raise ValueError("trace acquisition command sequence does not match command")
        elif not self.failure:
            raise ValueError("non-complete controlled command requires explicit failure")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


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
    def issue(self, command: ControlledCommand) -> ControlledCommandResult:
        """Issue one command and return its typed, provenance-bound result."""
        raise NotImplementedError

    @abstractmethod
    def acquire_trace(
        self, command: ControlledCommand, channels: tuple[str, ...]
    ) -> ControlledTraceAcquisition:
        """Acquire identified channels with controller timing and trigger provenance."""
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
        self._labels = np.asarray(
            [0] * (self.count // 2) + [1] * (self.count - self.count // 2),
            dtype=np.uint8,
        )
        order = np.random.default_rng(self.seed + 1).permutation(self.count)
        self._labels = self._labels[order]
        identity_material = json.dumps(
            {
                "seed": self.seed,
                "count": self.count,
                "trace_length": self.trace_length,
                "target_id": self.target_id,
                "firmware_id": self.firmware_id,
                "controller_config_hash": self.controller_config_hash,
                "operation": "synthetic-controlled-trace-v1",
            },
            sort_keys=True,
        ).encode("utf-8")
        identity = hashlib.sha256(identity_material).hexdigest()
        self._session_id = f"mock-controlled-logical-session-{identity[:24]}"
        self._allocation_id = f"mock-controlled-logical-allocation-{identity[24:48]}"
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
            trace_seed_material = f"{self.seed}:synthetic-controlled-trace-v1:{index}".encode()
            trace_seed = int.from_bytes(hashlib.sha256(trace_seed_material).digest()[:16], "big")
            trace = (
                np.random.default_rng(trace_seed)
                .normal(0.0, 1.0, self.trace_length)
                .astype(np.float32)
            )
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
                acquisition_configuration_hash=self.controller_config_hash,
                trigger_identity=f"mock-trigger-{index:012d}",
                timing_provenance="synthetic index domain; no hardware timing",
                refresh_relationship="synthetic/unavailable",
                command_provenance="synthetic operation identity; no DRAM command issued",
            )
            provenance.validate()
            channel = ControlledTraceChannel(
                channel_id="mock-synthetic-channel-0",
                channel_kind="digital",
                units="synthetic arbitrary units",
                sampling_clock_id="mock-index-clock",
                calibration_id="mock-uncalibrated",
            )
            acquisition = ControlledTraceAcquisition(
                acquisition_id=f"{self._session_id}:acquisition-{index:012d}",
                trigger_id=f"mock-trigger-{index:012d}",
                trigger_hardware_ticks=index,
                hardware_clock_id="mock-index-clock",
                timing_uncertainty_ticks=0,
                channels=(channel,),
                refresh_relationship="synthetic/unavailable",
                command_sequence_id="synthetic-no-command",
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
                    "controlled_trace_acquisition": json.dumps(
                        acquisition.as_dict(), sort_keys=True
                    ),
                    "controlled_trace_channel_ids": json.dumps([channel.channel_id]),
                    "controlled_trigger_identity": acquisition.trigger_id,
                    "controlled_timing_provenance": provenance.timing_provenance,
                    "controlled_refresh_relationship": provenance.refresh_relationship,
                    "controlled_command_provenance": provenance.command_provenance,
                    "controller_firmware_id": self.firmware_id,
                    "controller_config_hash": self.controller_config_hash,
                    "label_stream_fingerprint": hashlib.sha256(self._labels.tobytes()).hexdigest(),
                    "measurement_primitive": "controlled-memory-interface-mock",
                    "physical_observation_semantics": (
                        "synthetic mock trace; no physical memory claim"
                    ),
                    "session_started_at": self._started_at,
                    "synthetic_recovery_identity": (
                        "logical session/sample identity and trace are deterministic functions of "
                        "seed, sample index, and operation identity; wall-clock start is provenance only"
                    ),
                },
            )

    def close(self) -> None:
        self._closed = True
