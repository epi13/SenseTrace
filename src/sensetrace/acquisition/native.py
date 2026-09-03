"""Optional narrow ctypes wrapper around the native timing kernel."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .probe_contract import (
    ProbeFailure,
    ProbeImplementation,
    ProbeRequest,
    ProbeSampleRecord,
)

_CALIBRATION_VALUE: ctypes.c_uint64 | None = None


class NativeMeasurementKernel:
    def __init__(self, library: ctypes.CDLL, path: Path):
        self.library = library
        self.path = path
        self.library.st_kernel_version.restype = ctypes.c_char_p
        self.library.st_cpu_supports_clflush.restype = ctypes.c_int
        for name in [
            "st_measure_cached",
            "st_measure_flushed",
            "st_measure_cached_delayed",
            "st_measure_flushed_delayed",
            "st_measure_cached_control",
            "st_measure_flushed_control",
            "st_timer_calibration",
            "st_idle_calibration",
        ]:
            function = getattr(self.library, name)
            if name.endswith("_delayed") or name.endswith("_control"):
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
            elif name.startswith("st_measure"):
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
            else:
                function.argtypes = [
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
            function.restype = ctypes.c_int
        self.supports_clflush = bool(self.library.st_cpu_supports_clflush())

    @classmethod
    def load(cls) -> NativeMeasurementKernel | None:
        candidates = []
        configured = os.environ.get("SENSETRACE_NATIVE_LIB")
        if configured:
            candidates.append(Path(configured))
        candidates.append(
            Path(__file__).resolve().parents[3] / "native" / "libsensetrace_measurement.so"
        )
        for candidate in candidates:
            if candidate.exists():
                try:
                    return cls(ctypes.CDLL(str(candidate)), candidate)
                except (OSError, AttributeError):
                    continue
        return None

    def _measure(
        self,
        function_name: str,
        address: int,
        repetitions: int,
        extra_delay_cycles: int = 0,
    ) -> np.ndarray:
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        output = (ctypes.c_uint64 * repetitions)()
        function = getattr(self.library, function_name)
        if function_name.endswith("_delayed") or function_name.endswith("_control"):
            result = function(
                ctypes.c_void_p(address),
                repetitions,
                ctypes.c_uint64(extra_delay_cycles),
                output,
            )
        else:
            result = function(ctypes.c_void_p(address), repetitions, output)
        if result != 0:
            raise OSError(-result, f"native {function_name} failed")
        return np.ctypeslib.as_array(output).astype(np.float64, copy=True)

    def measure_cached(
        self, address: int, repetitions: int, *, extra_delay_cycles: int = 0
    ) -> np.ndarray:
        return self._measure(
            "st_measure_cached_control",
            address,
            repetitions,
            extra_delay_cycles,
        )

    def measure_flushed(
        self, address: int, repetitions: int, *, extra_delay_cycles: int = 0
    ) -> np.ndarray:
        return self._measure(
            "st_measure_flushed_control",
            address,
            repetitions,
            extra_delay_cycles,
        )

    def flush_calibration(self, address: int, repetitions: int) -> np.ndarray:
        """Return raw cycle counts for the CLFLUSH control path."""

        return self.measure_flushed(address, repetitions)

    def _calibrate(self, function_name: str, repetitions: int) -> np.ndarray:
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        output = (ctypes.c_uint64 * repetitions)()
        result = getattr(self.library, function_name)(repetitions, output)
        if result != 0:
            raise OSError(-result, f"native {function_name} failed")
        return np.ctypeslib.as_array(output).astype(np.float64, copy=True)

    def timer_calibration(self, repetitions: int) -> np.ndarray:
        return self._calibrate("st_timer_calibration", repetitions)

    def idle_calibration(self, repetitions: int) -> np.ndarray:
        return self._calibrate("st_idle_calibration", repetitions)

    @staticmethod
    def calibration_address() -> int:
        global _CALIBRATION_VALUE
        value = ctypes.c_uint64(0xA5A5A5A5A5A5A5A5)
        _CALIBRATION_VALUE = value
        return ctypes.addressof(value)

    def provenance(self) -> dict[str, Any]:
        return {
            "implementation": "native/measurement_kernel.c",
            "version": self.library.st_kernel_version().decode("ascii"),
            "library": str(self.path),
            "library_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "timer_source": (
                "explicit compiler barrier; LFENCE; RDTSC start; RDTSCP end; "
                "LFENCE; explicit compiler barrier"
            ),
            "cached_measurement_primitive": (
                "st_measure_cached_control(address, repetitions, delay_cycles, output); "
                "one delayed-capable exported primitive for zero and nonzero delay"
            ),
            "clflush_measurement_primitive": (
                "st_measure_flushed_control(address, repetitions, delay_cycles, output); "
                "_mm_clflush(address), _mm_mfence(), then one delayed-capable timed-load "
                "primitive for zero and nonzero delay"
            ),
            "cache_control": "CLFLUSH plus MFENCE for the flushed control path",
            "exported_measurement_entry_points": {
                "cached_zero_and_nonzero_delay": "st_measure_cached_control",
                "flushed_zero_and_nonzero_delay": "st_measure_flushed_control",
                "legacy_zero_delay_aliases": ["st_measure_cached", "st_measure_flushed"],
            },
            "compiler_barriers": (
                "explicit GCC/Clang memory barriers surround timing fences, the volatile load, "
                "and the delay-clock boundary"
            ),
            "delay_semantics": {
                "requested_units": "TSC cycles",
                "delay_starts": "after the volatile load and an LFENCE load-ordering fence",
                "delay_deadline": "read with RDTSC inside the timed region",
                "load_serialization": "LFENCE immediately after the volatile load on x86",
                "delay_loop": "RDTSC deadline with PAUSE; zero and nonzero delay use the same branch structure",
                "added_effect_includes": [
                    "deadline RDTSC read",
                    "conditional branch",
                    "PAUSE loop when delay_cycles is nonzero",
                    "normal timed-region end sequence",
                ],
                "observed_latency_warning": (
                    "requested cycles are not asserted to equal added measured latency; "
                    "report paired observed latency distributions"
                ),
            },
            "clflush_supported": self.supports_clflush,
            "raw_units": "TSC cycles",
            "guarantees": [
                "the native kernel reports CPU support before exposing the CLFLUSH path",
                "the measured load follows the CLFLUSH and MFENCE sequence on that path",
                "zero and nonzero artificial delays use the same exported delayed-capable primitive",
                "the artificial delay begins only after the load-ordering fence",
            ],
            "limitations": [
                "CLFLUSH does not prove that the load reached DRAM",
                "no physical address, row, bank, subarray, chip, or DIMM identity is exposed",
                "cache coherence, prefetch, replacement, and memory-controller behavior remain uncontrolled",
                "the delayed control is an artificial instrumentation calibration, not a physical memory effect",
            ],
        }

    def implementation_contract(self) -> ProbeImplementation:
        """Describe this loaded artifact without implying hardware capability."""

        provenance = self.provenance()
        return ProbeImplementation(
            implementation_id="sensetrace.native.measurement-kernel",
            implementation_version=str(provenance["version"]),
            backend_kind="native_shared_library",
            artifact_sha256=str(provenance["library_sha256"]),
            architecture=platform.machine() or "unavailable",
            kernel_release=platform.release() or "unavailable",
            compatibility_status=(
                "available" if platform.machine() in {"x86_64", "i386", "i686"} else "unsupported"
            ),
            timing_source=str(provenance["timer_source"]),
            result_units="TSC cycles",
            provenance=provenance,
            limitations=tuple(str(item) for item in provenance["limitations"]),
        )

    @staticmethod
    def _cpu_id() -> int | None:
        helper = getattr(os, "sched_getcpu", None)
        if helper is not None:
            try:
                return int(helper())
            except OSError:
                pass
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.sched_getcpu.argtypes = []
            libc.sched_getcpu.restype = ctypes.c_int
            value = int(libc.sched_getcpu())
            return value if value >= 0 else None
        except (AttributeError, OSError):
            return None

    def execute(self, request: ProbeRequest, *, address: int | None = None) -> ProbeSampleRecord:
        """Execute one request through the stable probe contract.

        Unsupported operations are returned as explicit evidence.  Runtime
        failures are retained in the record instead of being mistaken for a
        zero measurement.
        """

        request.validate()
        implementation = self.implementation_contract()
        requested_affinity_value = request.parameters.get("cpu_affinity")
        requested_affinity = (
            tuple(int(value) for value in requested_affinity_value)
            if isinstance(requested_affinity_value, (list, tuple))
            else None
        )
        try:
            effective_affinity = tuple(sorted(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            effective_affinity = None
        started = time.monotonic_ns()
        cpu_before = self._cpu_id()
        failure: ProbeFailure | None = None
        raw_result: list[int | float] | None = None
        status: str = "complete"
        repetitions = int(request.parameters.get("repetitions", 1))
        delay = int(request.parameters.get("extra_delay_cycles", 0))
        try:
            if request.operation == "cached_load":
                if address is None:
                    raise ValueError("cached_load requires an address")
                raw_result = [
                    int(value)
                    for value in self.measure_cached(address, repetitions, extra_delay_cycles=delay)
                ]
            elif request.operation == "flushed_load":
                if address is None:
                    raise ValueError("flushed_load requires an address")
                if not self.supports_clflush:
                    status = "unsupported"
                    failure = ProbeFailure(
                        kind="unsupported_capability",
                        capability="clflush",
                        message="native library or CPU does not report CLFLUSH support",
                    )
                else:
                    raw_result = [
                        int(value)
                        for value in self.measure_flushed(
                            address, repetitions, extra_delay_cycles=delay
                        )
                    ]
            elif request.operation == "timer_calibration":
                raw_result = [int(value) for value in self.timer_calibration(repetitions)]
            elif request.operation == "idle_calibration":
                raw_result = [int(value) for value in self.idle_calibration(repetitions)]
            else:
                status = "unsupported"
                failure = ProbeFailure(
                    kind="unsupported_operation",
                    capability=request.operation,
                    message=f"native measurement kernel does not implement {request.operation!r}",
                )
        except (OSError, ValueError) as exc:
            status = "failed"
            failure = ProbeFailure(
                kind="execution_failure",
                message=str(exc),
                errno=exc.errno if isinstance(exc, OSError) else None,
            )
        finished = time.monotonic_ns()
        return ProbeSampleRecord(
            implementation=implementation,
            request=request,
            status=status,  # type: ignore[arg-type]
            monotonic_start_ns=started,
            monotonic_end_ns=finished,
            clock_domain="userspace CLOCK_MONOTONIC nanoseconds",
            raw_result=raw_result,
            result_units=implementation.result_units,
            cpu_before=cpu_before,
            cpu_after=self._cpu_id(),
            requested_affinity=requested_affinity,
            effective_affinity=effective_affinity,
            failure=failure,
            witness_correlation_ids=(request.correlation_id,) if request.correlation_id else (),
            provenance={
                "address_semantics": (
                    "process virtual address passed to a CPU load primitive"
                    if address is not None
                    else "no address used"
                ),
                "physical_address": "unavailable",
                "dram_topology": "unavailable",
            },
        )


def summarize_measurements(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": 0, "median": float("nan"), "mean": float("nan")}
    percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    q1, q3 = np.percentile(finite, [25, 75])
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "percentiles": {
            str(level): float(value)
            for level, value in zip([1, 5, 25, 50, 75, 95, 99], percentiles, strict=True)
        },
        "outlier_fraction_iqr": float(
            np.mean((finite < q1 - 1.5 * (q3 - q1)) | (finite > q3 + 1.5 * (q3 - q1)))
        ),
        "lag_1_autocorrelation": (
            float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
            if len(finite) > 2 and np.std(finite[:-1]) > 0 and np.std(finite[1:]) > 0
            else float("nan")
        ),
        "raw_samples_retained": True,
        "outlier_filtering": "none; quantile and IQR values are audit summaries only",
    }
