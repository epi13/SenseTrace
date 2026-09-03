"""Phase 2 controlled-memory-interface boundary and synthetic contract emulator.

The classes in this module are the controller-facing evidence contract. They
validate at runtime because annotations do not protect records arriving from a
device, a wire format, or a future adapter. The synthetic implementation is
deliberately a software emulator: it exercises the same lifecycle as a real
controller while remaining unable to claim physical topology or physical
measurement evidence.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from .base import AcquisitionBackend, RecoveryPolicy, Sample

TopologySource = Literal["controlled_hardware", "unavailable"]
CommandStatus = Literal["complete", "unsupported", "failed"]
ChannelKind = Literal["analog", "digital"]

_PLACEHOLDERS = frozenset({"", "unavailable", "unknown", "none", "null", "n/a"})
_COMMAND_KINDS = frozenset({"activate", "read", "write", "precharge", "refresh", "idle"})


def _identity(value: object, field_name: str, *, allow_placeholder: bool = False) -> str:
    """Validate a serialized identity without coercing malformed input."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    normalized = value.lower()
    if not value or (not allow_placeholder and normalized in _PLACEHOLDERS):
        raise ValueError(f"{field_name} is missing or is an invalid placeholder")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} contains control characters")
    return value


def _is_missing_identity(value: object) -> bool:
    """Return whether a value is an explicit placeholder, safely."""

    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS


def _ticks(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _parameters(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("controlled command parameters must be a mapping")
    for key in value:
        _identity(key, "controlled command parameter name")
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("controlled command parameters must be JSON-compatible") from exc
    return value


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

    def validate(self) -> None:
        if not isinstance(self.source, str) or self.source not in {
            "controlled_hardware",
            "unavailable",
        }:
            raise ValueError("controlled-memory topology has an unsupported source")
        fields = {key: value for key, value in asdict(self).items() if key != "source"}
        concrete: dict[str, str] = {}
        for key, value in fields.items():
            _identity(value, f"controlled topology {key}", allow_placeholder=True)
            if value.lower() not in _PLACEHOLDERS:
                concrete[key] = value
        if self.source != "controlled_hardware" and concrete:
            raise ValueError(
                "controlled-memory topology fields require source='controlled_hardware'; "
                f"got {sorted(concrete)} from {self.source!r}"
            )
        if self.source == "controlled_hardware" and not concrete:
            raise ValueError(
                "source='controlled_hardware' must supply at least one concrete topology field"
            )

    def as_dict(self) -> dict[str, str]:
        self.validate()
        return {key: value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ControlledCommand:
    """One externally controlled memory command."""

    command_id: str
    kind: str
    address_token: str  # opaque hardware token; never a virtual address
    issued_at_hardware_ticks: int
    command_sequence_id: str = "unavailable"
    refresh_relationship: str = "unavailable"
    timing_provenance: str = "unavailable"
    parameters: dict[str, Any] = field(default_factory=dict)
    command_clock_id: str = "unavailable"

    def validate(self) -> None:
        required = {
            "command_id": self.command_id,
            "address_token": self.address_token,
            "command_sequence_id": self.command_sequence_id,
            "refresh_relationship": self.refresh_relationship,
            "timing_provenance": self.timing_provenance,
        }
        missing = sorted(
            key
            for key, value in required.items()
            if not isinstance(value, str) or _is_missing_identity(value)
        )
        if missing:
            raise ValueError(f"controlled command provenance is missing {missing}")
        for key, value in required.items():
            _identity(value, f"controlled command {key}")
        if not isinstance(self.kind, str) or self.kind not in _COMMAND_KINDS:
            raise ValueError(f"controlled command kind {self.kind!r} is unsupported")
        _ticks(self.issued_at_hardware_ticks, "controlled command issue timing")
        _parameters(self.parameters)
        _identity(self.command_clock_id, "controlled command clock", allow_placeholder=True)

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ControlledAcquisitionProvenance:
    """Acquisition provenance for the controlled-memory interface.

    ``hardware_clock_id`` identifies the command/controller clock. Trace
    sampling is represented separately by ``sampling_clock_id`` so a real
    controller is not forced to pretend those clocks are the same.
    """

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
    sampling_clock_id: str = "unavailable"
    command_sequence_id: str = "unavailable"

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
        for key, value in required.items():
            _identity(value, f"controlled acquisition provenance {key}")
        _identity(
            self.sampling_clock_id, "controlled acquisition sampling clock", allow_placeholder=True
        )
        _identity(
            self.command_sequence_id,
            "controlled acquisition command sequence",
            allow_placeholder=True,
        )

    def as_dict(self) -> dict[str, str]:
        self.validate()
        return {key: value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ControlledTraceChannel:
    """One controller-declared analog or digital acquisition channel."""

    channel_id: str
    channel_kind: ChannelKind
    units: str
    sampling_clock_id: str
    calibration_id: str

    def validate(self) -> None:
        if not isinstance(self.channel_kind, str) or self.channel_kind not in {
            "analog",
            "digital",
        }:
            raise ValueError(f"controlled trace channel kind {self.channel_kind!r} is unsupported")
        for key, value in {
            "channel_id": self.channel_id,
            "units": self.units,
            "sampling_clock_id": self.sampling_clock_id,
            "calibration_id": self.calibration_id,
        }.items():
            _identity(value, f"controlled trace channel {key}")


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
        for key, value in {
            "acquisition_id": self.acquisition_id,
            "trigger_id": self.trigger_id,
            "hardware_clock_id": self.hardware_clock_id,
            "refresh_relationship": self.refresh_relationship,
            "command_sequence_id": self.command_sequence_id,
        }.items():
            _identity(value, f"controlled trace acquisition {key}")
        _ticks(self.trigger_hardware_ticks, "controlled trace trigger timing")
        _ticks(self.timing_uncertainty_ticks, "controlled trace timing uncertainty")
        if not isinstance(self.channels, tuple) or not self.channels:
            raise ValueError(
                "controlled trace acquisition requires at least one identified channel"
            )
        channel_ids: set[str] = set()
        for channel in self.channels:
            if not isinstance(channel, ControlledTraceChannel):
                raise ValueError("controlled trace acquisition contains an invalid channel object")
            channel.validate()
            if channel.channel_id in channel_ids:
                raise ValueError("controlled trace acquisition contains duplicate channel IDs")
            channel_ids.add(channel.channel_id)
        if any(channel.sampling_clock_id != self.hardware_clock_id for channel in self.channels):
            raise ValueError("trace channel sampling clocks disagree with acquisition clock")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ControlledCommandResult:
    """Typed result binding command, topology, trace, and completion state."""

    command: ControlledCommand
    status: CommandStatus
    topology: ControlledMemoryTopology
    provenance: ControlledAcquisitionProvenance
    completed_at_hardware_ticks: int
    acquisition: ControlledTraceAcquisition | None = None
    failure: str | None = None

    def validate(self) -> None:
        if not isinstance(self.status, str) or self.status not in {
            "complete",
            "unsupported",
            "failed",
        }:
            raise ValueError(f"controlled command result status {self.status!r} is unsupported")
        if not isinstance(self.command, ControlledCommand):
            raise ValueError("controlled command result has an invalid command object")
        if not isinstance(self.topology, ControlledMemoryTopology):
            raise ValueError("controlled command result has an invalid topology object")
        if not isinstance(self.provenance, ControlledAcquisitionProvenance):
            raise ValueError("controlled command result has an invalid provenance object")
        self.command.validate()
        self.topology.validate()
        self.provenance.validate()
        completed = _ticks(self.completed_at_hardware_ticks, "controlled command completion timing")
        if completed < self.command.issued_at_hardware_ticks:
            raise ValueError("controlled command completion precedes command issue")
        if self.command.command_clock_id != "unavailable" and (
            self.command.command_clock_id != self.provenance.hardware_clock_id
        ):
            raise ValueError("command clock does not match acquisition provenance command clock")
        if self.command.timing_provenance != self.provenance.timing_provenance:
            raise ValueError("command and acquisition timing provenance disagree")
        if self.command.refresh_relationship != self.provenance.refresh_relationship:
            raise ValueError("command and acquisition refresh relationships disagree")
        if self.provenance.command_sequence_id != "unavailable" and (
            self.provenance.command_sequence_id != self.command.command_sequence_id
        ):
            raise ValueError("provenance command sequence does not match command")
        if self.status == "complete":
            if self.acquisition is None:
                raise ValueError("complete controlled command requires trace acquisition")
            if self.failure is not None:
                raise ValueError("complete controlled command cannot include failure")
            self.acquisition.validate()
            if self.acquisition.trigger_hardware_ticks < self.command.issued_at_hardware_ticks:
                raise ValueError("trace trigger precedes command issue")
            if self.acquisition.trigger_hardware_ticks > completed:
                raise ValueError("trace trigger follows command completion")
            if self.acquisition.command_sequence_id != self.command.command_sequence_id:
                raise ValueError("trace acquisition command sequence does not match command")
            if self.acquisition.trigger_id != self.provenance.trigger_identity:
                raise ValueError("trace trigger identity does not match provenance")
            if self.acquisition.refresh_relationship != self.provenance.refresh_relationship:
                raise ValueError("trace and provenance refresh relationships disagree")
            if self.provenance.sampling_clock_id != "unavailable" and any(
                channel.sampling_clock_id != self.provenance.sampling_clock_id
                for channel in self.acquisition.channels
            ):
                raise ValueError("trace sampling clock does not match provenance")
        elif self.acquisition is not None:
            raise ValueError("non-complete controlled command cannot include trace acquisition")
        elif not isinstance(self.failure, str) or not self.failure.strip():
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

    @abstractmethod
    def read_trace(self, acquisition: ControlledTraceAcquisition) -> np.ndarray:
        """Return the trace payload associated with a validated acquisition."""
        raise NotImplementedError

    def close(self) -> None:
        return None


class ControlledInterfaceAcquisitionBackend(AcquisitionBackend):
    """Adapt a validated controller lifecycle to SenseTrace ``Sample`` rows."""

    name = "controlled"
    recovery_policy = RecoveryPolicy(
        allow_resume=True,
        deterministic_replay=True,
        continuity_requirement="deterministic logical interface session identity",
    )

    def __init__(
        self,
        interface: ControlledMemoryInterface,
        *,
        count: int,
        trace_length: int,
        labels: np.ndarray,
        session_id: str,
        allocation_id: str,
        label_stream_fingerprint: str,
    ) -> None:
        self.interface = interface
        self.count = count
        self.trace_length = trace_length
        self._labels = labels
        self.acquisition_session_id = session_id
        self._allocation_id = allocation_id
        self.label_stream_fingerprint = label_stream_fingerprint

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        if start_index < 0 or start_index > self.count:
            raise ValueError("start_index outside controlled dataset")
        for index in range(start_index, self.count):
            interface_provenance = self.interface.provenance()
            interface_provenance.validate()
            sequence_id = f"{self.acquisition_session_id}:sequence-{index:012d}"
            command = ControlledCommand(
                command_id=f"{self.acquisition_session_id}:command-{index:012d}",
                kind="read",
                address_token=f"{self.acquisition_session_id}:opaque-token-{index:012d}",
                issued_at_hardware_ticks=index * 10,
                command_sequence_id=sequence_id,
                refresh_relationship=interface_provenance.refresh_relationship,
                timing_provenance=interface_provenance.timing_provenance,
                parameters={"sample_index": index},
                command_clock_id=interface_provenance.hardware_clock_id,
            )
            result = self.interface.issue(command)
            result.validate()
            if result.status != "complete" or result.acquisition is None:
                raise RuntimeError(f"controlled interface could not complete sample {index}")
            trace = np.asarray(self.interface.read_trace(result.acquisition), dtype=np.float32)
            if trace.ndim != 1 or len(trace) != self.trace_length:
                raise ValueError("controlled interface returned a trace with the wrong shape")
            provenance = result.provenance
            topology = result.topology
            acquisition = result.acquisition
            device_id = provenance.device_identity
            row_id = topology.row if topology.source == "controlled_hardware" else "row-unknown"
            bank_id = topology.bank if topology.source == "controlled_hardware" else "bank-unknown"
            cell_id = (
                topology.column
                if topology.source == "controlled_hardware"
                else f"{self.acquisition_session_id}:offset-{index:08d}"
            )
            yield Sample(
                trace=trace,
                label=int(self._labels[index]),
                metadata={
                    "sample_id": f"{self.acquisition_session_id}:sample-{index:012d}",
                    "session_id": self.acquisition_session_id,
                    "acquisition_session_id": self.acquisition_session_id,
                    "allocation_id": self._allocation_id,
                    "virtual_location_id": f"{self.acquisition_session_id}:virtual-{index:08d}",
                    "trial_index": index,
                    "device_id": device_id,
                    "bank_id": bank_id,
                    "row_id": row_id,
                    "cell_or_offset_id": cell_id,
                    "controlled_interface_name": self.interface.interface_name,
                    "controlled_command_id": command.command_id,
                    "controlled_command_sequence_id": command.command_sequence_id,
                    "controlled_address_token": command.address_token,
                    "controlled_command_clock_id": provenance.hardware_clock_id,
                    "controlled_sampling_clock_id": acquisition.hardware_clock_id,
                    "controlled_trigger_identity": acquisition.trigger_id,
                    "controlled_trace_channel_ids": json.dumps(
                        [channel.channel_id for channel in acquisition.channels]
                    ),
                    "controlled_topology": json.dumps(topology.as_dict(), sort_keys=True),
                    "controlled_topology_source": topology.source,
                    "controlled_provenance": json.dumps(provenance.as_dict(), sort_keys=True),
                    "controlled_trace_acquisition": json.dumps(
                        acquisition.as_dict(), sort_keys=True
                    ),
                    "controlled_timing_provenance": provenance.timing_provenance,
                    "controlled_refresh_relationship": provenance.refresh_relationship,
                    "controlled_command_provenance": provenance.command_provenance,
                    "controller_firmware_id": provenance.controller_firmware_id,
                    "controller_config_hash": provenance.controller_config_hash,
                    "label_stream_fingerprint": self.label_stream_fingerprint,
                    "measurement_primitive": (
                        "controlled-memory-interface-mock"
                        if "mock" in self.interface.interface_name
                        else self.interface.interface_name
                    ),
                    "physical_observation_semantics": (
                        "synthetic mock trace; no physical memory claim"
                        if "mock" in self.interface.interface_name
                        else "controlled-interface trace; claims limited to supplied hardware contract"
                    ),
                    "synthetic_recovery_identity": (
                        "logical session, command, acquisition, and trace identities are deterministic "
                        "functions of the versioned mock configuration and sample index"
                    ),
                },
            )

    def close(self) -> None:
        self.interface.close()


@dataclass
class SyntheticMockControlledInterface(ControlledMemoryInterface):
    """Deterministic software emulator of the future controller lifecycle."""

    count: int
    trace_length: int
    seed: int
    target_id: str
    firmware_id: str
    controller_config_hash: str
    interface_name = "controlled-memory-interface-mock-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 2
            or self.count % 2
            or isinstance(self.trace_length, bool)
            or not isinstance(self.trace_length, int)
            or self.trace_length < 1
        ):
            raise ValueError("count must be an even integer >= 2 and trace_length must be positive")
        _identity(self.target_id, "mock target identity")
        _identity(self.firmware_id, "mock firmware identity")
        _identity(self.controller_config_hash, "mock controller configuration hash")
        material = json.dumps(
            {
                "count": self.count,
                "trace_length": self.trace_length,
                "seed": self.seed,
                "target_id": self.target_id,
                "firmware_id": self.firmware_id,
                "controller_config_hash": self.controller_config_hash,
                "interface": self.interface_name,
            },
            sort_keys=True,
        ).encode()
        identity = hashlib.sha256(material).hexdigest()
        self.session_id = f"mock-controlled-logical-session-{identity[:24]}"
        self.allocation_id = f"mock-controlled-logical-allocation-{identity[24:48]}"
        self._traces: dict[str, np.ndarray] = {}
        self._closed = False

    def provenance(self) -> ControlledAcquisitionProvenance:
        return ControlledAcquisitionProvenance(
            experiment_target_id=self.target_id,
            controller_firmware_id=self.firmware_id,
            controller_config_hash=self.controller_config_hash,
            device_identity="mock-device-unknown",
            dimm_identity="mock-dimm-unknown",
            calibration_state="mock-uncalibrated",
            hardware_clock_id="mock-command-clock",
            acquisition_trigger="mock-software-trigger",
            acquisition_configuration_hash=self.controller_config_hash,
            trigger_identity="mock-trigger",
            timing_provenance="synthetic command clock ticks",
            refresh_relationship="synthetic/no-refresh-schedule",
            command_provenance="synthetic operation identity; no DRAM command issued",
            sampling_clock_id="mock-sampling-clock",
        )

    def topology_for(self, address_token: str) -> ControlledMemoryTopology:
        _identity(address_token, "controlled topology address token")
        return ControlledMemoryTopology(source="unavailable")

    def acquire_trace(
        self, command: ControlledCommand, channels: tuple[str, ...]
    ) -> ControlledTraceAcquisition:
        command.validate()
        if not isinstance(channels, tuple) or not channels:
            raise ValueError("controlled trace acquisition requires channel IDs")
        if "sample_index" not in command.parameters:
            raise ValueError("mock controlled commands require an integer sample_index")
        channel_records = tuple(
            ControlledTraceChannel(
                channel_id=channel_id,
                channel_kind="digital",
                units="synthetic arbitrary units",
                sampling_clock_id="mock-sampling-clock",
                calibration_id="mock-uncalibrated",
            )
            for channel_id in channels
        )
        acquisition = ControlledTraceAcquisition(
            acquisition_id=f"{self.session_id}:acquisition-{command.parameters['sample_index']:012d}",
            trigger_id="mock-trigger",
            trigger_hardware_ticks=command.issued_at_hardware_ticks,
            hardware_clock_id="mock-sampling-clock",
            timing_uncertainty_ticks=0,
            channels=channel_records,
            refresh_relationship=command.refresh_relationship,
            command_sequence_id=command.command_sequence_id,
        )
        acquisition.validate()
        return acquisition

    def issue(self, command: ControlledCommand) -> ControlledCommandResult:
        command.validate()
        sample_index = command.parameters.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError("mock controlled commands require an integer sample_index")
        if sample_index < 0 or sample_index >= self.count:
            raise ValueError("mock controlled command sample_index is outside the dataset")
        acquisition = self.acquire_trace(command, ("mock-synthetic-channel-0",))
        trace_seed_material = (
            f"{self.seed}:{self.interface_name}:{self.controller_config_hash}:{sample_index}"
        ).encode()
        trace_seed = int.from_bytes(hashlib.sha256(trace_seed_material).digest()[:16], "big")
        self._traces[acquisition.acquisition_id] = (
            np.random.default_rng(trace_seed).normal(0.0, 1.0, self.trace_length).astype(np.float32)
        )
        provenance = replace(self.provenance(), command_sequence_id=command.command_sequence_id)
        result = ControlledCommandResult(
            command=command,
            status="complete",
            topology=self.topology_for(command.address_token),
            provenance=provenance,
            completed_at_hardware_ticks=command.issued_at_hardware_ticks,
            acquisition=acquisition,
        )
        result.validate()
        return result

    def read_trace(self, acquisition: ControlledTraceAcquisition) -> np.ndarray:
        acquisition.validate()
        try:
            return self._traces[acquisition.acquisition_id].copy()
        except KeyError as exc:
            raise ValueError("mock trace payload is unavailable for acquisition identity") from exc

    def close(self) -> None:
        self._closed = True


@dataclass
class SyntheticMockControlledBackend(ControlledInterfaceAcquisitionBackend):
    """Phase 2 mock backend routed through ``ControlledMemoryInterface``."""

    count: int = 64
    trace_length: int = 32
    seed: int = 1337
    target_id: str = "mock-controlled-target-0000"
    firmware_id: str = "mock-controller-firmware-v0"
    controller_config_hash: str = "mock-config-unavailable"
    name: str = "controlled_mock"

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 2
            or self.count % 2
            or isinstance(self.trace_length, bool)
            or not isinstance(self.trace_length, int)
            or self.trace_length < 1
        ):
            raise ValueError("count must be an even integer >= 2 and trace_length must be positive")
        if self.controller_config_hash == "mock-config-unavailable":
            self.controller_config_hash = hashlib.sha256(
                json.dumps(
                    {
                        "count": self.count,
                        "trace_length": self.trace_length,
                        "seed": self.seed,
                        "target_id": self.target_id,
                        "firmware_id": self.firmware_id,
                        "interface": "controlled-memory-interface-mock-v1",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        labels = np.asarray(
            [0] * (self.count // 2) + [1] * (self.count - self.count // 2), dtype=np.uint8
        )
        labels = labels[np.random.default_rng(self.seed + 1).permutation(self.count)]
        label_fingerprint = hashlib.sha256(labels.tobytes()).hexdigest()
        interface = SyntheticMockControlledInterface(
            count=self.count,
            trace_length=self.trace_length,
            seed=self.seed,
            target_id=self.target_id,
            firmware_id=self.firmware_id,
            controller_config_hash=self.controller_config_hash,
        )
        ControlledInterfaceAcquisitionBackend.__init__(
            self,
            interface,
            count=self.count,
            trace_length=self.trace_length,
            labels=labels,
            session_id=interface.session_id,
            allocation_id=interface.allocation_id,
            label_stream_fingerprint=label_fingerprint,
        )
        self.interface = interface
        self._labels = labels
        self._started_at = datetime.now(UTC).isoformat()

    def topology_for_virtual_offset(self, _offset: int) -> ControlledMemoryTopology:
        """Virtual offsets carry no physical topology. Always unavailable."""
        return self.interface.topology_for(f"{self.acquisition_session_id}:virtual-token")

    def derive_topology_from_virtual_address(self, _address: int) -> ControlledMemoryTopology:
        raise ValueError(
            "physical topology cannot be derived from a virtual address; "
            "it may only be supplied by the controlled memory interface"
        )

    def session_provenance(self, *, status: str = "completed") -> dict[str, Any]:
        provenance = self.interface.provenance()
        provenance.validate()
        return {
            "acquisition_session_id": self.acquisition_session_id,
            "allocation_id": self._allocation_id,
            "status": status,
            "interface_name": self.interface.interface_name,
            "controller_firmware_id": provenance.controller_firmware_id,
            "controller_config_hash": provenance.controller_config_hash,
            "device_identity": provenance.device_identity,
            "dimm_identity": provenance.dimm_identity,
            "topology_source": "unavailable",
            "physical_topology": "unavailable; mock has no physical topology source",
            "started_at": self._started_at,
            "recovery_policy": {
                "allow_resume": self.recovery_policy.allow_resume,
                "deterministic_replay": self.recovery_policy.deterministic_replay,
                "continuity_requirement": self.recovery_policy.continuity_requirement,
            },
        }

    def manifest_provenance(self, *, condition: str) -> dict[str, Any]:
        provenance = self.interface.provenance().as_dict()
        return {
            "backend": self.name,
            "interface_name": self.interface.interface_name,
            "protocol_identity": self.interface.interface_name,
            "protocol_hash": self.controller_config_hash,
            "dataset_purpose": "phase2_mock_controlled",
            "evidence_plane": "controlled_memory_interface_mock",
            "claim_boundary": "software/evidence contract only; no physical DRAM claim",
            "topology": {
                "source": "unavailable",
                "virtual_addresses_promoted_to_physical": False,
            },
            "controller_configuration_hash": self.controller_config_hash,
            "condition": condition,
            "provenance_contract": provenance,
            "recovery": {
                "policy": {
                    "allow_resume": self.recovery_policy.allow_resume,
                    "deterministic_replay": self.recovery_policy.deterministic_replay,
                    "continuity_requirement": self.recovery_policy.continuity_requirement,
                },
                "session_identity": self.acquisition_session_id,
                "deterministic": True,
            },
        }
