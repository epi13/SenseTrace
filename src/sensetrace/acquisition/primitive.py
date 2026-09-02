"""Measurement-primitive contracts and access-state provenance.

The acquisition engine owns labels, allocation, journaling, and persistence.
This module owns the narrower question of *what operation was performed and
what can be said about the resulting observation*.  Keeping those concerns
separate makes it possible to add a better primitive without changing the
split, model, or recovery machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

CapabilityState = Literal["known", "unknown", "unsupported"]
OracleStrength = Literal["exact", "probabilistic", "partial", "unavailable"]


@dataclass(frozen=True)
class PrimitiveCapabilities:
    """Machine-readable capability declarations.

    Values describe the evidence available to the protocol, not what the
    operator hopes the machine did.  ``unknown`` is intentionally distinct
    from ``unsupported``: the former means an interface was not able to
    establish the fact, while the latter means the primitive does not expose
    it at all.
    """

    operation_issues_memory_access: CapabilityState
    independent_access_state_oracle: CapabilityState
    cache_residency_control: CapabilityState
    translation_state: CapabilityState
    depends_on_virtual_addresses: CapabilityState
    physical_address_information: CapabilityState
    row_bank_channel_topology: CapabilityState
    privileged_counters: CapabilityState
    kernel_support: CapabilityState
    external_hardware: CapabilityState
    destructive_or_state_changing: CapabilityState
    replay_across_sessions_boots_devices: CapabilityState

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class AccessStateOracle:
    """Description of an independent access-state oracle.

    An oracle describes evidence about the access path; it is not itself a
    model feature.  ``observation`` is optional because a capability can be
    documented before a concrete per-sample reading is available.
    """

    name: str
    strength: OracleStrength
    status: str
    observation: str
    source: str
    independent_of_latency: bool
    model_feature_eligible: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveObservation:
    """One primitive result with separated audit and model channels."""

    trace: np.ndarray
    access_state: AccessStateOracle
    physical_observation: str
    audit_metadata: dict[str, Any]
    model_eligible_features: dict[str, float]


class MeasurementPrimitive(ABC):
    """Interface consumed by a measurement backend.

    Target-state preparation remains outside this interface because the same
    target generator must be reusable across primitives.  A primitive receives
    the prepared address and a callable operation, performs the operation
    under test, and returns a physical observation plus provenance.
    """

    name = "abstract"

    @property
    @abstractmethod
    def capabilities(self) -> PrimitiveCapabilities:
        raise NotImplementedError

    @property
    @abstractmethod
    def oracle(self) -> AccessStateOracle:
        raise NotImplementedError

    @abstractmethod
    def measure(
        self,
        address: int,
        operation: str,
        read_operation: Any,
        trace_length: int,
        *,
        perturbation_cycles: int = 0,
        perturbation_label_applied: bool = False,
    ) -> PrimitiveObservation:
        raise NotImplementedError

    def cache_provenance(self) -> dict[str, Any]:
        return {
            "method": "unspecified",
            "primitive": self.name,
            "guarantee": "unspecified",
            "limitations": ["primitive did not declare cache-control semantics"],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": self.capabilities.as_dict(),
            "access_state_oracle": self.oracle.as_dict(),
            "physical_observation": "primitive-specific; preserve raw trace",
            "audit_only_metadata": [
                "addresses and allocation identifiers",
                "session/boot identifiers",
                "oracle identity and result",
            ],
            "model_eligible_features": "raw trace-derived features only unless protocol explicitly opts in",
        }


class CommodityTimingPrimitive(MeasurementPrimitive):
    """The frozen CLFLUSH/timed-load commodity primitive.

    This class intentionally exposes the conservative observable that the
    historical Phase 1A engine used.  In particular, a flushed load is not
    relabeled as a DRAM observation merely because its latency is higher.
    """

    name = "commodity-clflush-timed-load"

    def __init__(
        self,
        kernel: Any | None,
        *,
        operation: str,
        cache_control: str,
        eviction: bytearray,
    ) -> None:
        from .capabilities import commodity_timing_capabilities, commodity_timing_oracle

        self.kernel = kernel
        self.operation = operation
        self.cache_control = cache_control
        self.eviction = eviction
        self._capabilities = commodity_timing_capabilities(
            operation=operation, cache_control=cache_control
        )
        self._oracle = commodity_timing_oracle(
            operation=operation, cache_control=cache_control
        )

    @property
    def capabilities(self) -> PrimitiveCapabilities:
        return self._capabilities

    @property
    def oracle(self) -> AccessStateOracle:
        return self._oracle

    def cache_provenance(self) -> dict[str, Any]:
        if self.cache_control == "clflush":
            return {
                "method": "clflush",
                "primitive": "_mm_clflush(address) followed by _mm_mfence() before each timed load",
                "fences": [
                    "LFENCE before RDTSC",
                    "LFENCE immediately after the volatile load before delay-clock start",
                    "RDTSCP then LFENCE at timed-region end",
                    "MFENCE after CLFLUSH",
                ],
                "timing_entry_point": "st_measure_flushed_control",
                "guarantee": (
                    "CLFLUSH is supported by the native x86 kernel and requests invalidation "
                    "of the addressed cache line before the timed load"
                ),
                "limitations": [
                    "does not prove that the load reached DRAM",
                    "does not reveal a physical address, row, bank, subarray, chip, or DIMM",
                    "does not guarantee absence of all coherence or prefetch effects",
                    "valid only on the native kernel path when CPU support is reported",
                ],
                "eviction_bytes": 0,
            }
        if self.cache_control == "eviction_buffer":
            return {
                "method": "eviction_buffer",
                "primitive": "best-effort sweep of a user-space eviction buffer",
                "fences": [],
                "timing_entry_point": "st_measure_cached_control",
                "guarantee": "best-effort cache eviction; does not prove DRAM access",
                "limitations": ["cache hierarchy and replacement behavior are not controlled"],
                "eviction_bytes": len(self.eviction),
            }
        return {
            "method": "none",
            "primitive": "no cache eviction before the timed load",
            "fences": [],
            "timing_entry_point": "st_measure_cached_control",
            "guarantee": "cache-hit control; no cache eviction is requested",
            "limitations": ["the load may be satisfied by any level of the cache hierarchy"],
            "eviction_bytes": 0,
        }

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "operation": self.operation,
            "cache_control_provenance": self.cache_provenance(),
        }

    def _evict_cache(self) -> int:
        value = 0
        for index in range(0, len(self.eviction), 64):
            value ^= self.eviction[index]
        return value

    def measure(
        self,
        address: int,
        operation: str,
        read_operation: Any,
        trace_length: int,
        *,
        perturbation_cycles: int = 0,
        perturbation_label_applied: bool = False,
    ) -> PrimitiveObservation:
        if trace_length < 1:
            raise ValueError("trace_length must be positive")
        observed: list[float] = []
        for _ in range(trace_length):
            if operation == "memory_read" and self.cache_control == "eviction_buffer":
                self._evict_cache()
            if self.kernel is not None:
                if operation == "idle":
                    values = self.kernel.idle_calibration(1)
                elif self.cache_control == "clflush":
                    values = self.kernel.measure_flushed(
                        address,
                        1,
                        extra_delay_cycles=(
                            perturbation_cycles if perturbation_label_applied else 0
                        ),
                    )
                else:
                    values = self.kernel.measure_cached(
                        address,
                        1,
                        extra_delay_cycles=(
                            perturbation_cycles if perturbation_label_applied else 0
                        ),
                    )
                observed.append(float(values[0]))
            else:
                import time

                started = time.perf_counter_ns()
                if operation == "memory_read":
                    read_operation()
                observed.append(float(time.perf_counter_ns() - started))
        audit = {
            "primitive": self.name,
            "operation": operation,
            "cache_control": self.cache_control,
            "requested_access_state": self.oracle.observation,
            "oracle_status": self.oracle.status,
            "oracle_strength": self.oracle.strength,
            "oracle_independent_of_latency": self.oracle.independent_of_latency,
            "perturbation_cycles": int(perturbation_cycles),
            "perturbation_applied": bool(perturbation_label_applied and perturbation_cycles > 0),
            "timing_entry_point": (
                "st_measure_flushed_control"
                if self.cache_control == "clflush" and self.kernel is not None
                else "st_measure_cached_control"
                if self.kernel is not None
                else "python_perf_counter_control"
            ),
        }
        return PrimitiveObservation(
            trace=np.asarray(observed, dtype=np.float32),
            access_state=self.oracle,
            physical_observation=(
                "raw TSC-cycle timed-load trace"
                if self.kernel is not None
                else "raw perf_counter_ns fallback timing trace"
            ),
            audit_metadata=audit,
            model_eligible_features={},
        )


def available_measurement_primitives() -> tuple[str, ...]:
    """Names understood by the acquisition factory."""

    return ("commodity-clflush-timed-load",)


def create_measurement_primitive(
    name: str,
    kernel: Any | None,
    *,
    operation: str,
    cache_control: str,
    eviction: bytearray,
) -> MeasurementPrimitive:
    """Create a named primitive without changing the experiment engine."""

    if name != "commodity-clflush-timed-load":
        raise ValueError(
            f"unsupported measurement primitive {name!r}; "
            f"available: {', '.join(available_measurement_primitives())}"
        )
    return CommodityTimingPrimitive(
        kernel,
        operation=operation,
        cache_control=cache_control,
        eviction=eviction,
    )
