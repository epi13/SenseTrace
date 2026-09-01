"""Optional narrow ctypes wrapper around the native timing kernel."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

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
            "st_timer_calibration",
            "st_idle_calibration",
        ]:
            function = getattr(self.library, name)
            function.argtypes = (
                [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                if name.startswith("st_measure")
                else [
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
            )
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
                except OSError:
                    continue
        return None

    def _measure(self, function_name: str, address: int, repetitions: int) -> np.ndarray:
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        output = (ctypes.c_uint64 * repetitions)()
        function = getattr(self.library, function_name)
        result = function(ctypes.c_void_p(address), repetitions, output)
        if result != 0:
            raise OSError(-result, f"native {function_name} failed")
        return np.ctypeslib.as_array(output).astype(np.float64, copy=True)

    def measure_cached(self, address: int, repetitions: int) -> np.ndarray:
        return self._measure("st_measure_cached", address, repetitions)

    def measure_flushed(self, address: int, repetitions: int) -> np.ndarray:
        return self._measure("st_measure_flushed", address, repetitions)

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
            "timer_source": "RDTSC start with LFENCE; RDTSCP end with LFENCE",
            "cache_control": "CLFLUSH plus MFENCE",
            "clflush_supported": self.supports_clflush,
            "raw_units": "TSC cycles",
        }


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
    }
