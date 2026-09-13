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

TRAJECTORY_CACHED_PRELOADED = 0
TRAJECTORY_CLFLUSH = 1
TRAJECTORY_EVICTION = 2
TRAJECTORY_TIMER_ONLY = 3
TRAJECTORY_CONDITIONS = {
    "cached_preloaded": TRAJECTORY_CACHED_PRELOADED,
    "clflush": TRAJECTORY_CLFLUSH,
    "eviction": TRAJECTORY_EVICTION,
    "timer_only": TRAJECTORY_TIMER_ONLY,
}


class NativeMeasurementKernel:
    def __init__(self, library: ctypes.CDLL, path: Path):
        self.library = library
        self.path = path
        self.library.st_kernel_version.restype = ctypes.c_char_p
        self.library.st_trajectory_kernel_version.restype = ctypes.c_char_p
        self.library.st_cpu_supports_clflush.restype = ctypes.c_int
        self.library.st_cpu_supports_rdtscp.restype = ctypes.c_int
        self.library.st_cpu_supports_avx2.restype = ctypes.c_int
        for name in [
            "st_measure_cached",
            "st_measure_flushed",
            "st_measure_cached_delayed",
            "st_measure_flushed_delayed",
            "st_measure_cached_control",
            "st_measure_flushed_control",
            "st_measure_dependency_chain",
            "st_measure_repeated_load",
            "st_measure_paired_cached",
            "st_timer_calibration",
            "st_idle_calibration",
            "st_read_tsc_aux",
            "st_measure_trajectory",
            "st_run_memory_pressure",
        ]:
            function = getattr(self.library, name)
            if name == "st_measure_trajectory":
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_size_t,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint8),
                ]
            elif name == "st_run_memory_pressure":
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_uint64,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                ]
            elif name == "st_read_tsc_aux":
                function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
            elif name == "st_measure_paired_cached":
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_int64),
                ]
            elif name.endswith("_delayed") or name.endswith("_control"):
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
        self.supports_rdtscp = bool(self.library.st_cpu_supports_rdtscp())
        self.supports_avx2 = bool(self.library.st_cpu_supports_avx2())

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

    def measure_dependency_chain(self, address: int, repetitions: int) -> np.ndarray:
        return self._measure("st_measure_dependency_chain", address, repetitions)

    def measure_repeated_load(self, address: int, repetitions: int) -> np.ndarray:
        return self._measure("st_measure_repeated_load", address, repetitions)

    def measure_paired_cached(self, address_a: int, address_b: int, repetitions: int) -> np.ndarray:
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        output = (ctypes.c_int64 * repetitions)()
        result = self.library.st_measure_paired_cached(
            ctypes.c_void_p(address_a), ctypes.c_void_p(address_b), repetitions, output
        )
        if result != 0:
            raise OSError(-result, "native st_measure_paired_cached failed")
        return np.ctypeslib.as_array(output).astype(np.float64, copy=True)

    def read_tsc_aux(self) -> tuple[int, int]:
        tsc = ctypes.c_uint64()
        aux = ctypes.c_uint32()
        result = self.library.st_read_tsc_aux(ctypes.byref(tsc), ctypes.byref(aux))
        if result != 0:
            raise OSError(-result, "native st_read_tsc_aux failed")
        return int(tsc.value), int(aux.value)

    def measure_trajectory(
        self,
        target_address: int,
        reference_address: int | None,
        conditions: np.ndarray,
        orders: np.ndarray | None,
        *,
        eviction_address: int | None = None,
        eviction_bytes: int = 0,
    ) -> dict[str, np.ndarray]:
        """Run the versioned v5 trajectory loop without Python callbacks.

        The returned first/second arrays retain acquisition order.  ``orders``
        says whether that order was target/reference (0) or reference/target
        (1); callers may map them back to channel identity without losing the
        perturbing sequential order.
        """

        conditions = np.asarray(conditions, dtype=np.uint8)
        if conditions.ndim != 1 or len(conditions) < 1:
            raise ValueError("trajectory conditions must be a non-empty vector")
        if np.any(conditions > TRAJECTORY_TIMER_ONLY):
            raise ValueError("trajectory condition code is outside the v5 ABI")
        repetitions = len(conditions)
        if reference_address is not None:
            if orders is None:
                raise ValueError("paired trajectory measurement requires acquisition orders")
            orders = np.asarray(orders, dtype=np.uint8)
            if orders.shape != conditions.shape:
                raise ValueError("trajectory orders must align with conditions")
        else:
            orders = np.zeros(repetitions, dtype=np.uint8)
        def zeros64() -> Any:
            return (ctypes.c_uint64 * repetitions)()

        def zeros32() -> Any:
            return (ctypes.c_uint32 * repetitions)()
        first_start, first_end = zeros64(), zeros64()
        first_aux_start, first_aux_end = zeros32(), zeros32()
        first_quality = (ctypes.c_uint8 * repetitions)()
        second_start = second_end = None
        second_aux_start = second_aux_end = None
        second_quality = None
        if reference_address is not None:
            second_start, second_end = zeros64(), zeros64()
            second_aux_start, second_aux_end = zeros32(), zeros32()
            second_quality = (ctypes.c_uint8 * repetitions)()
        condition_buffer = (ctypes.c_uint8 * repetitions).from_buffer_copy(conditions)
        order_buffer = (ctypes.c_uint8 * repetitions).from_buffer_copy(orders)
        result = self.library.st_measure_trajectory(
            ctypes.c_void_p(target_address),
            ctypes.c_void_p(reference_address) if reference_address is not None else None,
            condition_buffer,
            order_buffer,
            repetitions,
            ctypes.c_void_p(eviction_address) if eviction_address is not None else None,
            int(eviction_bytes),
            first_start,
            first_end,
            first_aux_start,
            first_aux_end,
            first_quality,
            second_start,
            second_end,
            second_aux_start,
            second_aux_end,
            second_quality,
        )
        if result != 0:
            raise OSError(-result, "native st_measure_trajectory failed")
        result_arrays: dict[str, np.ndarray] = {
            "first_start_tsc": np.ctypeslib.as_array(first_start).copy(),
            "first_end_tsc": np.ctypeslib.as_array(first_end).copy(),
            "first_start_aux": np.ctypeslib.as_array(first_aux_start).copy(),
            "first_end_aux": np.ctypeslib.as_array(first_aux_end).copy(),
            "first_quality": np.ctypeslib.as_array(first_quality).copy(),
        }
        if reference_address is not None:
            assert second_start is not None
            assert second_end is not None
            assert second_aux_start is not None
            assert second_aux_end is not None
            assert second_quality is not None
            result_arrays.update(
                {
                    "second_start_tsc": np.ctypeslib.as_array(second_start).copy(),
                    "second_end_tsc": np.ctypeslib.as_array(second_end).copy(),
                    "second_start_aux": np.ctypeslib.as_array(second_aux_start).copy(),
                    "second_end_aux": np.ctypeslib.as_array(second_aux_end).copy(),
                    "second_quality": np.ctypeslib.as_array(second_quality).copy(),
                }
            )
        return result_arrays

    def run_memory_pressure(
        self,
        buffer_address: int,
        word_count: int,
        duration_cycles: int,
        *,
        operation: str = "read",
        requested_cpu: int = -1,
        started_out: ctypes.c_uint32 | None = None,
    ) -> dict[str, int]:
        if operation not in {"read", "write"}:
            raise ValueError("memory pressure operation must be read or write")
        start_tsc = ctypes.c_uint64()
        end_tsc = ctypes.c_uint64()
        start_aux = ctypes.c_uint32()
        end_aux = ctypes.c_uint32()
        iterations = ctypes.c_uint64()
        status = ctypes.c_uint32()
        started = started_out if started_out is not None else ctypes.c_uint32()
        result = self.library.st_run_memory_pressure(
            ctypes.c_void_p(buffer_address),
            int(word_count),
            int(duration_cycles),
            0 if operation == "read" else 1,
            int(requested_cpu),
            ctypes.byref(start_tsc),
            ctypes.byref(end_tsc),
            ctypes.byref(start_aux),
            ctypes.byref(end_aux),
            ctypes.byref(iterations),
            ctypes.byref(status),
            ctypes.byref(started),
        )
        if result != 0:
            raise OSError(-result, "native st_run_memory_pressure failed")
        return {
            "start_tsc": int(start_tsc.value),
            "end_tsc": int(end_tsc.value),
            "start_aux": int(start_aux.value),
            "end_aux": int(end_aux.value),
            "iterations": int(iterations.value),
            "status": int(status.value),
            "started": int(started.value),
            "requested_cpu": int(requested_cpu),
        }

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
            "trajectory_version": self.library.st_trajectory_kernel_version().decode("ascii"),
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
                "dependency_chain": "st_measure_dependency_chain",
                "repeated_load_response": "st_measure_repeated_load",
                "paired_cached_differential": "st_measure_paired_cached",
                "trajectory_v1": "st_measure_trajectory",
                "memory_pressure_v1": "st_run_memory_pressure",
                "tsc_aux_read": "st_read_tsc_aux",
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
            "rdtscp_supported": self.supports_rdtscp,
            "avx2_supported": self.supports_avx2,
            "raw_units": "TSC cycles",
            "guarantees": [
                "the native kernel reports CPU support before exposing the CLFLUSH path",
                "the measured load follows the CLFLUSH and MFENCE sequence on that path",
                "zero and nonzero artificial delays use the same exported delayed-capable primitive",
                "the artificial delay begins only after the load-ordering fence",
                "dependency_chain keeps the loaded value live through explicit data-dependent operations",
                "paired_cached_differential emits the second-minus-first observed cycle count",
                "trajectory_v1 performs bounded condition/order loops in native code and retains raw endpoint timestamps/AUX values",
                "trajectory_v1 eviction sweeps before the timed load without a target preload between sweep and load",
                "historical entry points retain the v4 version identity and semantics",
            ],
            "limitations": [
                "CLFLUSH does not prove that the load reached DRAM",
                "no physical address, row, bank, subarray, chip, or DIMM identity is exposed",
                "cache coherence, prefetch, replacement, and memory-controller behavior remain uncontrolled",
                "the delayed control is an artificial instrumentation calibration, not a physical memory effect",
                "paired differences compare two CPU-side load timings; they do not isolate a DRAM-only response",
                "TSC_AUX equality is endpoint evidence only and cannot prove no migration occurred between endpoints",
                "trajectory paired probes are sequential and the first probe can perturb the second",
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
            elif request.operation == "dependency_chain":
                if address is None:
                    raise ValueError("dependency_chain requires an address")
                raw_result = [
                    int(value) for value in self.measure_dependency_chain(address, repetitions)
                ]
            elif request.operation == "repeated_load":
                if address is None:
                    raise ValueError("repeated_load requires an address")
                raw_result = [
                    int(value) for value in self.measure_repeated_load(address, repetitions)
                ]
            elif request.operation == "paired_cached":
                if address is None:
                    raise ValueError("paired_cached requires an address")
                second_address = request.parameters.get("address_b")
                if isinstance(second_address, bool) or not isinstance(second_address, int):
                    raise ValueError("paired_cached requires integer parameters.address_b")
                raw_result = [
                    float(value)
                    for value in self.measure_paired_cached(address, second_address, repetitions)
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
