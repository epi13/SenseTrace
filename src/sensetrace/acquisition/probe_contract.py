"""Backend-independent contract between Python and measurement probe planes.

The contract describes what a probe executed and observed.  It does not grant
the backend any stronger physical-memory semantics than its access boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProbeStatus = Literal["complete", "unsupported", "failed"]


@dataclass(frozen=True)
class ProbeImplementation:
    """Immutable identity and compatibility description for one backend."""

    implementation_id: str
    implementation_version: str
    backend_kind: str
    artifact_sha256: str
    architecture: str
    kernel_release: str
    compatibility_status: str
    timing_source: str
    result_units: str
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def validate(self) -> None:
        required = (
            self.implementation_id,
            self.implementation_version,
            self.backend_kind,
            self.artifact_sha256,
            self.architecture,
            self.kernel_release,
            self.compatibility_status,
            self.timing_source,
            self.result_units,
        )
        if any(not value for value in required):
            raise ValueError("probe implementation identity and compatibility fields are required")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ProbeRequest:
    """One high-level request; implementation-specific instructions stay opaque."""

    session_id: str
    sample_index: int
    operation: str
    parameters: dict[str, Any]
    correlation_id: str | None = None

    def validate(self) -> None:
        if not self.session_id:
            raise ValueError("probe session_id is required")
        if self.sample_index < 0:
            raise ValueError("probe sample_index must be non-negative")
        if not self.operation:
            raise ValueError("probe operation is required")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ProbeFailure:
    kind: str
    message: str
    capability: str | None = None
    errno: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeSampleRecord:
    """Canonical sample result emitted by native or future probe backends."""

    implementation: ProbeImplementation
    request: ProbeRequest
    status: ProbeStatus
    monotonic_start_ns: int
    monotonic_end_ns: int
    clock_domain: str
    raw_result: list[int | float] | None
    result_units: str
    cpu_before: int | None
    cpu_after: int | None
    requested_affinity: tuple[int, ...] | None
    effective_affinity: tuple[int, ...] | None
    failure: ProbeFailure | None = None
    witness_correlation_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.implementation.validate()
        self.request.validate()
        if self.monotonic_start_ns < 0 or self.monotonic_end_ns < self.monotonic_start_ns:
            raise ValueError("probe monotonic timing window is invalid")
        if self.status == "complete" and (self.raw_result is None or self.failure is not None):
            raise ValueError("complete probe samples require a raw result and no failure")
        if self.status != "complete" and self.failure is None:
            raise ValueError(
                "unsupported or failed probe samples require explicit failure evidence"
            )
        if self.result_units != self.implementation.result_units:
            raise ValueError("sample result units disagree with implementation contract")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["schema"] = "sensetrace.probe-sample.v1"
        value["claim_boundary"] = (
            "CPU-side measurement evidence only; native execution does not establish direct "
            "DRAM commands, topology, physical-cell identity, or cache/MMU bypass"
        )
        return value
