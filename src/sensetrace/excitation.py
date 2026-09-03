"""Deterministic coded multi-core excitation contracts.

This module describes an active acquisition schedule; it does not claim that a
requested schedule executed on hardware.  The executed code and compliance
record must be supplied by the acquisition adapter and are kept distinct from
the request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ExcitationFamily = Literal["prbs", "walsh", "read_pressure", "write_pressure", "active_quiet"]
CoreRoleName = Literal["measurement", "reference", "excitation", "orchestrator"]


@dataclass(frozen=True)
class CoreRoleAssignment:
    """Pinned role-to-core mapping for the exact worker-03 target."""

    target_id: str
    roles: dict[str, tuple[int, ...]]

    @classmethod
    def worker03_default(cls) -> CoreRoleAssignment:
        return cls(
            target_id="worker03-hardware-v1",
            roles={
                "measurement": (0,),
                "reference": (1,),
                "excitation": (2, 3, 4, 5, 6),
                "orchestrator": (7,),
            },
        )

    def validate(self, *, physical_core_count: int = 8) -> None:
        if self.target_id != "worker03-hardware-v1":
            raise ValueError("coded excitation currently targets worker03-hardware-v1 only")
        if physical_core_count < 1:
            raise ValueError("physical_core_count must be positive")
        required_roles = {"measurement", "reference", "excitation", "orchestrator"}
        if set(self.roles) != required_roles:
            raise ValueError("core role assignment must define exactly the four worker-03 roles")
        seen: set[int] = set()
        for role, cores in self.roles.items():
            if role not in {"measurement", "reference", "excitation", "orchestrator"}:
                raise ValueError(f"unsupported core role {role!r}")
            if not cores:
                raise ValueError(f"core role {role!r} has no assigned cores")
            for core in cores:
                if (
                    isinstance(core, bool)
                    or not isinstance(core, int)
                    or not 0 <= core < physical_core_count
                ):
                    raise ValueError(f"core {core!r} is outside the physical CPU layout")
                if core in seen:
                    raise ValueError(f"physical core {core} is assigned to multiple roles")
                seen.add(core)

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "target_id": self.target_id,
            "roles": {key: list(value) for key, value in self.roles.items()},
        }


@dataclass(frozen=True)
class ExcitationStep:
    sequence_position: int
    active: tuple[bool, ...]
    operation: Literal["read", "write", "idle"]
    phase_ticks: int

    def validate(self, width: int) -> None:
        if isinstance(self.sequence_position, bool) or self.sequence_position < 0:
            raise ValueError("excitation sequence position must be non-negative")
        if len(self.active) != width:
            raise ValueError("excitation step width does not match role assignment")
        if self.operation not in {"read", "write", "idle"}:
            raise ValueError("unsupported excitation operation")
        if isinstance(self.phase_ticks, bool) or self.phase_ticks < 0:
            raise ValueError("excitation phase_ticks must be non-negative")


@dataclass(frozen=True)
class CodedExcitationSchedule:
    """A reproducible codebook and its intended core roles."""

    schedule_id: str
    family: ExcitationFamily
    length: int
    seed: int
    role_assignment: CoreRoleAssignment = field(default_factory=CoreRoleAssignment.worker03_default)
    operation: Literal["read", "write", "idle"] = "read"
    phase_ticks: int = 1

    def validate(self) -> None:
        if not self.schedule_id or self.schedule_id != self.schedule_id.strip():
            raise ValueError("schedule_id must be a non-empty stable identifier")
        if self.family not in {"prbs", "walsh", "read_pressure", "write_pressure", "active_quiet"}:
            raise ValueError(f"unsupported excitation family {self.family!r}")
        if isinstance(self.length, bool) or self.length < 2:
            raise ValueError("excitation length must be at least two")
        if isinstance(self.seed, bool):
            raise ValueError("excitation seed must be an integer")
        if self.operation not in {"read", "write", "idle"}:
            raise ValueError("unsupported excitation operation")
        if self.family == "write_pressure" and self.operation != "write":
            raise ValueError("write_pressure requires operation='write'")
        if self.family in {"read_pressure", "prbs", "walsh"} and self.operation == "write":
            raise ValueError("read-oriented excitation family cannot issue write operations")
        if isinstance(self.phase_ticks, bool) or self.phase_ticks < 1:
            raise ValueError("phase_ticks must be positive")
        self.role_assignment.validate()

    @property
    def excitation_width(self) -> int:
        return len(self.role_assignment.roles["excitation"])

    def _bits(self) -> np.ndarray:
        self.validate()
        width = self.excitation_width
        if self.family == "walsh":
            size = 1
            while size < max(self.length, width + 1):
                size <<= 1
            hadamard = np.asarray([[1]], dtype=np.int8)
            while hadamard.shape[0] < size:
                hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
            return (hadamard[: self.length, :width] > 0).astype(np.int8)
        if self.family == "active_quiet":
            return np.asarray(
                [
                    [
                        1 if (position % 2 == 0 and core == position // 2 % width) else 0
                        for core in range(width)
                    ]
                    for position in range(self.length)
                ],
                dtype=np.int8,
            )
        rng = np.random.default_rng(self.seed)
        if self.family == "write_pressure":
            return rng.integers(0, 2, size=(self.length, width), dtype=np.int8)
        if self.family == "read_pressure":
            return np.ones((self.length, width), dtype=np.int8)
        return rng.integers(0, 2, size=(self.length, width), dtype=np.int8)

    def steps(self) -> tuple[ExcitationStep, ...]:
        bits = self._bits()
        steps = tuple(
            ExcitationStep(
                index, tuple(bool(value) for value in bits[index]), self.operation, self.phase_ticks
            )
            for index in range(self.length)
        )
        for step in steps:
            step.validate(self.excitation_width)
        return steps

    def requested_code(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in step.active) for step in self.steps())

    def fingerprint(self) -> str:
        material = {
            "schedule_id": self.schedule_id,
            "family": self.family,
            "length": self.length,
            "seed": self.seed,
            "role_assignment": self.role_assignment.as_dict(),
            "operation": self.operation,
            "phase_ticks": self.phase_ticks,
            "requested_code": self.requested_code(),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def request_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "sensetrace.coded-excitation-request.v1",
            "schedule_id": self.schedule_id,
            "family": self.family,
            "schedule_fingerprint": self.fingerprint(),
            "role_assignment": self.role_assignment.as_dict(),
            "operation": self.operation,
            "phase_ticks": self.phase_ticks,
            "requested_code": [list(row) for row in self.requested_code()],
            "execution_claim": "request only; adapter must record executed code separately",
        }


@dataclass(frozen=True)
class ExcitationExecution:
    """Observed schedule execution, distinct from the requested codebook."""

    schedule_fingerprint: str
    executed_code: tuple[tuple[int, ...], ...]
    start_clock: int
    end_clock: int
    core_ids: tuple[int, ...]
    interrupted_positions: tuple[int, ...] = ()
    compliance: Literal["compliant", "partial", "failed"] = "compliant"
    witness: dict[str, Any] = field(default_factory=dict)

    def validate(self, schedule: CodedExcitationSchedule) -> None:
        schedule.validate()
        if self.schedule_fingerprint != schedule.fingerprint():
            raise ValueError("executed excitation refers to a different schedule")
        if self.start_clock < 0 or self.end_clock < self.start_clock:
            raise ValueError("excitation execution clock window is invalid")
        if (
            not self.core_ids
            or any(
                isinstance(core, bool) or not isinstance(core, int) or core < 0
                for core in self.core_ids
            )
            or len(set(self.core_ids)) != len(self.core_ids)
        ):
            raise ValueError("excitation execution must record unique non-negative core IDs")
        if len(self.executed_code) > schedule.length:
            raise ValueError("executed excitation has more steps than requested")
        if any(
            len(row) != schedule.excitation_width or any(value not in (0, 1) for value in row)
            for row in self.executed_code
        ):
            raise ValueError("executed excitation contains malformed code values")
        if self.compliance not in {"compliant", "partial", "failed"}:
            raise ValueError("unsupported excitation compliance state")
        if self.compliance == "compliant" and len(self.executed_code) != schedule.length:
            raise ValueError("compliant excitation must contain every requested step")
        if any(
            position < 0 or position >= schedule.length for position in self.interrupted_positions
        ):
            raise ValueError("interrupted excitation position is outside the schedule")
        if len(set(self.interrupted_positions)) != len(self.interrupted_positions):
            raise ValueError("interrupted excitation positions must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "sensetrace.coded-excitation-execution.v1",
            "schedule_fingerprint": self.schedule_fingerprint,
            "executed_code": [list(row) for row in self.executed_code],
            "start_clock": self.start_clock,
            "end_clock": self.end_clock,
            "core_ids": list(self.core_ids),
            "interrupted_positions": list(self.interrupted_positions),
            "compliance": self.compliance,
            "witness": self.witness,
        }
