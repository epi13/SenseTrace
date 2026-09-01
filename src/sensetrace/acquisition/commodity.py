"""Safe, topology-agnostic commodity-memory observables.

This backend deliberately performs ordinary user-space memory writes and reads
only. It does not expose physical addresses, disable refresh, change voltage,
or run disturbance loops. The timing trace is an observation of the host
access path; it is not evidence that a particular DRAM row was reached.
"""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import platform
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import AcquisitionBackend, Sample
from .synthetic import balanced_labels


def _cpu_frequency_regime() -> dict[str, object]:
    governors: dict[str, str] = {}
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")):
        try:
            governors[path.parent.parent.name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    driver = "unavailable"
    try:
        driver = (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        pass
    return {
        "governors": governors or "unavailable",
        "cpu_frequency_driver": driver,
        "turbo_state": "unavailable: inspect actual CPU/kernel interface",
        "provenance": "local cpufreq sysfs",
    }


class ControlledMemoryBuffer:
    """Page-aligned anonymous memory with an explicit best-effort mlock result."""

    def __init__(self, word_count: int, *, lock_memory: bool = True):
        if word_count < 1:
            raise ValueError("word_count must be positive")
        self.word_count = word_count
        self.byte_count = word_count * ctypes.sizeof(ctypes.c_uint64)
        self._mapping = mmap.mmap(-1, self.byte_count, access=mmap.ACCESS_WRITE)
        self._words = (ctypes.c_uint64 * word_count).from_buffer(self._mapping)
        self._address = ctypes.addressof(self._words)
        self.lock_requested = lock_memory
        self.locked = False
        self.lock_error: str | None = None
        if lock_memory:
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                result = libc.mlock(
                    ctypes.c_void_p(self._address), ctypes.c_size_t(self.byte_count)
                )
                if result == 0:
                    self.locked = True
                else:
                    self.lock_error = os.strerror(ctypes.get_errno())
            except (AttributeError, OSError) as exc:
                self.lock_error = str(exc)

    def write(self, index: int, value: int) -> None:
        self._words[index % self.word_count] = ctypes.c_uint64(value).value

    def read(self, index: int) -> int:
        return int(self._words[index % self.word_count])

    def close(self) -> None:
        if self.locked:
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                libc.munlock(ctypes.c_void_p(self._address), ctypes.c_size_t(self.byte_count))
            except (AttributeError, OSError):
                pass
            self.locked = False
        # ctypes' exported view must be released before mmap can close.
        del self._words
        self._mapping.close()


@dataclass
class CommodityDramBackend(AcquisitionBackend):
    """Acquire timing/scalar observations from a controlled ordinary memory buffer."""

    count: int = 128
    trace_length: int = 32
    seed: int = 1337
    pattern: str = "single_bit"
    target_bit: int = 0
    word_count: int = 1024
    lock_memory: bool = True
    cache_control: str = "eviction_buffer"
    operation: str = "memory_read"
    eviction_bytes: int = 4 * 1024 * 1024
    cpu_affinity: list[int] | None = None
    _buffer: ControlledMemoryBuffer = field(init=False, repr=False)

    name = "commodity-dram"

    def __post_init__(self) -> None:
        if self.count < 2 or self.trace_length < 1:
            raise ValueError("count must be >= 2 and trace_length must be positive")
        if self.pattern not in {"all_zero_one", "single_bit", "random_word"}:
            raise ValueError(f"unsupported safe memory pattern: {self.pattern}")
        if self.cache_control not in {"none", "eviction_buffer"}:
            raise ValueError(f"unsupported cache-control method: {self.cache_control}")
        if self.operation not in {"memory_read", "idle"}:
            raise ValueError(f"unsupported safe memory operation: {self.operation}")
        if not 0 <= self.target_bit < 64:
            raise ValueError("target_bit must be in [0, 63]")
        if self.eviction_bytes < 64:
            raise ValueError("eviction_bytes must be at least one cache line")
        self._buffer = ControlledMemoryBuffer(self.word_count, lock_memory=self.lock_memory)
        self._labels = balanced_labels(self.count, self.seed)
        self._word_rng = np.random.default_rng(self.seed + 1)
        self._eviction = bytearray(self.eviction_bytes)
        self._affinity_before: list[int] | None = None
        self._affinity_applied: list[int] | None = None
        if self.cpu_affinity is not None:
            requested = {int(cpu) for cpu in self.cpu_affinity}
            available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
            if not requested or (available and not requested.issubset(available)):
                raise ValueError("cpu_affinity must be a non-empty subset of available CPUs")
            if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
                self._affinity_before = sorted(os.sched_getaffinity(0))
                os.sched_setaffinity(0, requested)
                self._affinity_applied = sorted(requested)
        self._timer_overhead_ns = self._calibrate_timer()
        self._cache_provenance = {
            "method": self.cache_control,
            "eviction_bytes": self.eviction_bytes if self.cache_control == "eviction_buffer" else 0,
            "guarantee": (
                "best-effort cache eviction; does not prove DRAM access"
                if self.cache_control == "eviction_buffer"
                else "no cache eviction; cache-hit control"
            ),
        }
        self._frequency_regime = _cpu_frequency_regime()

    @staticmethod
    def _calibrate_timer() -> float:
        samples = []
        for _ in range(101):
            started = time.perf_counter_ns()
            time.perf_counter_ns()
            samples.append(time.perf_counter_ns() - started)
        return float(np.median(samples))

    def _evict_cache(self) -> int:
        value = 0
        for index in range(0, len(self._eviction), 64):
            value ^= self._eviction[index]
        return value

    def _word_for(self, index: int, label: int) -> int:
        if self.pattern == "all_zero_one":
            return 0 if label == 0 else 0xFFFFFFFFFFFFFFFF
        if self.pattern == "single_bit":
            surrounding = int(self._word_rng.integers(0, 2**64, dtype=np.uint64))
            mask = 1 << self.target_bit
            return (surrounding & ~mask) | (label * mask)
        return int(self._word_rng.integers(0, 2**64, dtype=np.uint64))

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        if start_index < 0 or start_index > self.count:
            raise ValueError("start_index outside commodity dataset")
        for index in range(self.count):
            label = int(self._labels[index])
            buffer_index = index % self.word_count
            observed: list[float] = []
            word: int | None = None
            digital_value: int | None = None
            if self.operation == "memory_read":
                word = self._word_for(index, label)
                self._buffer.write(buffer_index, word)
                digital_value = self._buffer.read(buffer_index)
            for _ in range(self.trace_length):
                if self.operation == "memory_read" and self.cache_control == "eviction_buffer":
                    self._evict_cache()
                started = time.perf_counter_ns()
                if self.operation == "memory_read":
                    self._buffer.read(buffer_index)
                observed.append(float(time.perf_counter_ns() - started))
            if index < start_index:
                continue
            yield Sample(
                trace=np.asarray(observed, dtype=np.float32),
                label=label,
                metadata={
                    "sample_id": f"sample-{index:012d}",
                    "session_id": "session-0000",
                    "acquisition_block": f"block-{index // 64:04d}",
                    "device_id": "device-unknown",
                    "bank_id": "bank-unknown",
                    "row_id": "row-unknown",
                    "cell_or_offset_id": f"buffer-word-{buffer_index:08d}",
                    "trial_index": index,
                    "physical_operation": (
                        "ordinary_user_space_write_then_read"
                        if self.operation == "memory_read"
                        else "idle_timer_control_without_memory_operation"
                    ),
                    "pattern": self.pattern,
                    "target_bit": self.target_bit,
                    "cache_control_method": self._cache_provenance["method"],
                    "cache_control_provenance": str(self._cache_provenance),
                    "timer_source": "time.perf_counter_ns",
                    "timer_overhead_ns": self._timer_overhead_ns,
                    "digital_verification_value": (
                        str(digital_value) if digital_value is not None else "not_applicable"
                    ),
                    "digital_verification_passed": (
                        digital_value == word if digital_value is not None else "not_applicable"
                    ),
                    "memory_lock_requested": self._buffer.lock_requested,
                    "memory_lock_actual": self._buffer.locked,
                    "memory_lock_error": self._buffer.lock_error or "none",
                    "buffer_guarantee": "virtual anonymous allocation; physical topology unknown",
                    "cpu_affinity_actual": str(self._affinity_applied or "unchanged"),
                    "cpu_frequency_regime": str(self._frequency_regime),
                    "numa_topology": platform.node(),
                    "seed_id": f"commodity:{self.seed}",
                    "label_stream_fingerprint": hashlib.sha256(self._labels.tobytes()).hexdigest(),
                },
            )

    def close(self) -> None:
        self._buffer.close()
        if self._affinity_before is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(self._affinity_before))
