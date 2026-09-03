"""Evidence models for the optional eBPF observation/witness plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Availability = Literal[
    "supported", "unsupported", "unavailable", "permission_denied", "not_collected"
]


@dataclass(frozen=True)
class WitnessHookCapability:
    name: str
    tracepoint: str
    status: Availability
    reason: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WitnessEvent:
    session_id: str
    event_type: str
    timestamp_ns: int
    clock_domain: str
    cpu: int | None
    pid: int | None
    tid: int | None
    fields: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.session_id or not self.event_type:
            raise ValueError("witness event session and type are required")
        if self.timestamp_ns < 0:
            raise ValueError("witness timestamp must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ClockAlignment:
    event_clock_domain: str
    sample_clock_domain: str
    offset_ns: int | None
    uncertainty_ns: int | None
    status: Literal["bounded", "uncertain", "unavailable"]
    method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WitnessSession:
    session_id: str
    experiment_id: str
    status: Literal["operational", "incomplete", "unavailable", "failed"]
    target_pid: int
    target_tid: int | None
    requested_hooks: tuple[str, ...]
    attached_hooks: tuple[str, ...]
    unavailable_hooks: tuple[WitnessHookCapability, ...]
    events: tuple[WitnessEvent, ...]
    provenance: dict[str, Any]
    alignment: ClockAlignment
    started_at: str
    ended_at: str
    failure: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "schema": "sensetrace.ebpf-witness-session.v1",
            "claim_boundary": (
                "kernel-side contextual evidence about observable host confounders only; "
                "not direct DRAM command, topology, cell, or analog evidence"
            ),
            "sample_veto_policy": "none; frozen protocols decide how witness states are used",
        }
