"""Safe, topology-agnostic commodity-memory observables.

This backend deliberately performs ordinary user-space memory writes and reads
only. It does not expose physical addresses, disable refresh, change voltage,
or run disturbance loops. The timing trace is an observation of the host
access path; it is not evidence that a particular DRAM row was reached.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import platform
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .base import AcquisitionBackend, Sample
from .native import NativeMeasurementKernel
from .primitive import (
    OperationScopedPerfOracle,
    PrimitiveObservation,
    TimingPerturbationCalibration,
    create_measurement_primitive,
)


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


def _numa_topology() -> str:
    try:
        nodes = sorted(path.name for path in Path("/sys/devices/system/node").glob("node[0-9]*"))
    except OSError:
        nodes = []
    return ",".join(nodes) if nodes else "unavailable"


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

    @property
    def address(self) -> int:
        return self._address


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
    measurement_primitive: str = "commodity-clflush-timed-load"
    eviction_bytes: int = 4 * 1024 * 1024
    cpu_affinity: list[int] | None = None
    location_count: int | None = None
    trials_per_location: int = 64
    labels_per_location: int | None = None
    session_count: int = 1
    use_native_kernel: bool = True
    timing_perturbation_cycles: int = 0
    timing_perturbation_label: int = 1
    calibration_namespace: str | None = None
    calibration_context: TimingPerturbationCalibration | None = None
    acquisition_session_id: str | None = None
    # Kept as an input alias so older callers can supply session_id while the
    # emitted contract uses acquisition_session_id explicitly.
    session_id: str | None = None
    session_index: int = 0
    campaign_id: str | None = None
    session_started_at: str | None = None
    host_inventory_snapshot: dict[str, Any] | None = None
    code_commit: str | None = None
    configuration_hash: str | None = None
    protocol_identity: str | None = None
    protocol_hash: str | None = None
    scoped_perf_oracle: OperationScopedPerfOracle | None = field(default=None, repr=False)
    shared_buffer: ControlledMemoryBuffer | None = field(default=None, repr=False)
    shared_allocation_id: str | None = None
    _buffer: ControlledMemoryBuffer = field(init=False, repr=False)
    _owns_buffer: bool = field(init=False, repr=False)
    _closed: bool = field(init=False, repr=False, default=False)

    name = "commodity-dram"

    def __post_init__(self) -> None:
        if self.count < 2 or self.trace_length < 1:
            raise ValueError("count must be >= 2 and trace_length must be positive")
        if self.pattern not in {"all_zero_one", "single_bit", "random_word"}:
            raise ValueError(f"unsupported safe memory pattern: {self.pattern}")
        if self.cache_control not in {"none", "eviction_buffer", "clflush"}:
            raise ValueError(f"unsupported cache-control method: {self.cache_control}")
        if self.operation not in {"memory_read", "idle"}:
            raise ValueError(f"unsupported safe memory operation: {self.operation}")
        if self.timing_perturbation_cycles < 0:
            raise ValueError("timing_perturbation_cycles must be non-negative")
        if self.timing_perturbation_label not in {0, 1}:
            raise ValueError("timing_perturbation_label must be 0 or 1")
        if self.calibration_context is None:
            if self.timing_perturbation_cycles != 0 or self.timing_perturbation_label != 1:
                raise ValueError(
                    "artificial timing perturbation is calibration-only; provide an explicit "
                    "TimingPerturbationCalibration context"
                )
            if self.calibration_namespace is not None:
                raise ValueError(
                    "calibration_namespace is calibration-only; provide an explicit "
                    "TimingPerturbationCalibration context"
                )
        else:
            if (
                self.timing_perturbation_cycles != 0
                or self.timing_perturbation_label != 1
                or self.calibration_namespace is not None
            ):
                raise ValueError(
                    "legacy timing perturbation fields cannot be combined with an explicit "
                    "calibration context"
                )
            self.timing_perturbation_cycles = self.calibration_context.cycles
            self.timing_perturbation_label = self.calibration_context.label
            self.calibration_namespace = self.calibration_context.namespace
        if not 0 <= self.target_bit < 64:
            raise ValueError("target_bit must be in [0, 63]")
        if self.eviction_bytes < 64:
            raise ValueError("eviction_bytes must be at least one cache line")
        if self.location_count is None:
            self.location_count = 1
            self.trials_per_location = self.count
        if self.location_count < 1 or self.trials_per_location < 4:
            raise ValueError("location_count and trials_per_location must be positive")
        if self.trials_per_location % 4:
            raise ValueError(
                "trials_per_location must be a multiple of four for exact pair-order balance"
            )
        if (
            self.labels_per_location is not None
            and self.labels_per_location != self.trials_per_location // 2
        ):
            raise ValueError("labels_per_location must equal half of trials_per_location")
        expected_count = self.location_count * self.trials_per_location
        if self.location_count != 1 and self.count != expected_count:
            raise ValueError("count must equal location_count * trials_per_location")
        self.count = expected_count
        self.labels_per_location = self.trials_per_location // 2
        if self.session_count < 1:
            raise ValueError("session_count must be positive")
        if self.session_index < 0:
            raise ValueError("session_index must be non-negative")
        self.acquisition_session_id = (
            self.acquisition_session_id or self.session_id or f"session-{uuid.uuid4().hex}"
        )
        self.session_id = self.acquisition_session_id
        self.session_started_at = self.session_started_at or datetime.now(UTC).isoformat()
        self._boot_id_value = self._boot_id()
        self._host_inventory_snapshot = dict(self.host_inventory_snapshot or {})
        self._owns_buffer = self.shared_buffer is None
        self._buffer = self.shared_buffer or ControlledMemoryBuffer(
            self.word_count, lock_memory=self.lock_memory
        )
        self._allocation_id = self.shared_allocation_id or f"buffer-{uuid.uuid4().hex}"
        self._labels, self._pair_order = self._make_labels()
        self._word_rng = np.random.default_rng(self.seed + 1)
        self._eviction = bytearray(self.eviction_bytes)
        self._native_kernel = NativeMeasurementKernel.load() if self.use_native_kernel else None
        self._native_provenance: dict[str, Any] = (
            self._native_kernel.provenance() if self._native_kernel is not None else None
        ) or {}
        if self.cache_control == "clflush" and (
            self._native_kernel is None or not self._native_kernel.supports_clflush
        ):
            if self._owns_buffer:
                self._buffer.close()
            raise RuntimeError(
                "cache_control=clflush requires a native x86 kernel with CLFLUSH support; "
                "the fallback timing path cannot claim or perform CLFLUSH"
            )
        self._measurement_primitive = create_measurement_primitive(
            self.measurement_primitive,
            self._native_kernel,
            operation=self.operation,
            cache_control=self.cache_control,
            eviction=self._eviction,
        )
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
        self._cache_provenance = self._measurement_primitive.cache_provenance()
        self._frequency_regime = _cpu_frequency_regime()

    def _make_labels(self) -> tuple[np.ndarray, np.ndarray]:
        labels = np.empty(self.count, dtype=np.uint8)
        pair_order = np.empty(self.count // 2, dtype=np.uint8)
        rng = np.random.default_rng(self.seed)
        for location in range(int(self.location_count or 1)):
            start = location * self.trials_per_location
            pair_count = self.trials_per_location // 2
            # Exactly half of the pairs use each temporal order.  Only the
            # order of these pair types is randomized, so pair position cannot
            # become an accidental proxy for the hidden label.
            local_order = np.concatenate(
                [
                    np.zeros(pair_count // 2, dtype=np.uint8),
                    np.ones(pair_count // 2, dtype=np.uint8),
                ]
            )
            rng.shuffle(local_order)
            for pair in range(pair_count):
                order = int(local_order[pair])
                pair_order[(start // 2) + pair] = order
                labels[start + pair * 2 : start + pair * 2 + 2] = (
                    np.asarray([0, 1], dtype=np.uint8)
                    if order == 0
                    else np.asarray([1, 0], dtype=np.uint8)
                )
        return labels, pair_order

    def _base_word(self, location: int, pair_index: int) -> int:
        rng = np.random.default_rng(
            self.seed + 0x10000 + location * (self.trials_per_location // 2) + pair_index
        )
        return int(rng.integers(0, 2**64, dtype=np.uint64)) & ~(1 << self.target_bit)

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
            location = index // self.trials_per_location
            pair_index = (index % self.trials_per_location) // 2
            surrounding = self._base_word(location, pair_index)
            mask = 1 << self.target_bit
            return (surrounding & ~mask) | (label * mask)
        # random_word is an explicit physical/random-word null control: labels
        # are balanced, but the generated word is independent of the label.
        return int(self._word_rng.integers(0, 2**64, dtype=np.uint64))

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        if start_index < 0 or start_index > self.count:
            raise ValueError("start_index outside commodity dataset")
        for index in range(start_index, self.count):
            label = int(self._labels[index])
            location = index // self.trials_per_location
            within_location = index % self.trials_per_location
            pair_index = within_location // 2
            pair_id = f"pair-{location:08d}-{pair_index:08d}"
            trial_pair_id = f"trial-pair-{location:08d}-{pair_index:08d}"
            buffer_index = location % self.word_count
            observed: list[float] = []
            word: int | None = None
            digital_value: int | None = None
            if self.operation == "memory_read":
                word = self._word_for(index, label)
                self._buffer.write(buffer_index, word)
                digital_value = self._buffer.read(buffer_index)
            address = self._buffer.address + buffer_index * ctypes.sizeof(ctypes.c_uint64)

            def controlled_measurement(
                address: int = address,
                buffer_index: int = buffer_index,
                label: int = label,
            ) -> PrimitiveObservation:
                return self._measurement_primitive.measure(
                    address,
                    self.operation,
                    lambda controlled_index=buffer_index: self._buffer.read(controlled_index),
                    self.trace_length,
                    perturbation_cycles=self.timing_perturbation_cycles,
                    perturbation_label_applied=label == self.timing_perturbation_label,
                )

            scoped_perf_provenance: dict[str, Any] | None = None
            if self.scoped_perf_oracle is not None:
                primitive_observation, access_oracle, scoped_perf_provenance = (
                    self.scoped_perf_oracle.observe(controlled_measurement)
                )
                if not isinstance(primitive_observation, PrimitiveObservation):
                    raise RuntimeError(
                        "scoped perf oracle callback did not return a primitive observation"
                    )
                primitive_observation = PrimitiveObservation(
                    trace=primitive_observation.trace,
                    access_state=access_oracle,
                    physical_observation=primitive_observation.physical_observation,
                    audit_metadata={
                        **primitive_observation.audit_metadata,
                        "operation_scoped_perf": scoped_perf_provenance,
                    },
                    model_eligible_features=primitive_observation.model_eligible_features,
                )
            else:
                primitive_observation = controlled_measurement()
            observed = primitive_observation.trace.tolist()
            if index < start_index:
                continue
            yield Sample(
                trace=np.asarray(observed, dtype=np.float32),
                label=label,
                metadata={
                    "sample_id": (
                        f"session-{self.session_index:06d}-{self.acquisition_session_id}:sample-{index:012d}"
                    ),
                    "session_id": self.acquisition_session_id,
                    "acquisition_session_id": self.acquisition_session_id,
                    "boot_id": self._boot_id_value,
                    "allocation_id": self._allocation_id,
                    "physical_allocation_id": self._allocation_id,
                    "acquisition_block": f"{self.acquisition_session_id}:block-{location // 16:04d}",
                    "location_id": (
                        f"{self.acquisition_session_id}:{self._allocation_id}:location-{location:08d}"
                    ),
                    "virtual_location_id": (
                        f"{self.acquisition_session_id}:{self._allocation_id}:"
                        f"virtual-location-{location:08d}"
                    ),
                    "pair_id": f"{self.acquisition_session_id}:{self._allocation_id}:{pair_id}",
                    "trial_pair_id": (
                        f"{self.acquisition_session_id}:{self._allocation_id}:{trial_pair_id}"
                    ),
                    "pair_order": (
                        "label_0_first"
                        if self._pair_order[(location * self.trials_per_location // 2) + pair_index]
                        == 0
                        else "label_1_first"
                    ),
                    "pair_position": within_location % 2,
                    "trial_within_location": within_location,
                    "device_id": "device-unknown",
                    "bank_id": "bank-unknown",
                    "row_id": "row-unknown",
                    "cell_or_offset_id": (
                        f"{self.acquisition_session_id}:{self._allocation_id}:"
                        f"buffer-word-{buffer_index:08d}"
                    ),
                    "buffer_offset_id": (
                        f"{self.acquisition_session_id}:{self._allocation_id}:"
                        f"buffer-offset-{buffer_index:08d}"
                    ),
                    "trial_index": index,
                    "physical_operation": (
                        "ordinary_user_space_write_then_read"
                        if self.operation == "memory_read"
                        else "idle_timer_control_without_memory_operation"
                    ),
                    "pattern": self.pattern,
                    "label_semantics": (
                        "target bit equals label"
                        if self.pattern == "single_bit"
                        else "all-zero versus all-one word"
                        if self.pattern == "all_zero_one"
                        else "balanced labels independent of random word"
                    ),
                    "target_bit": self.target_bit,
                    "cache_control_method": self._cache_provenance["method"],
                    "cache_control_provenance": json.dumps(self._cache_provenance, sort_keys=True),
                    "timer_source": (
                        self._native_provenance["timer_source"]
                        if self._native_kernel is not None
                        else "time.perf_counter_ns"
                    ),
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
                    "numa_topology": _numa_topology(),
                    "cpu_id": self._cpu_id(),
                    "measurement_kernel_version": (
                        self._native_provenance["version"]
                        if self._native_kernel is not None
                        else "python-control-v1"
                    ),
                    "measurement_kernel_hash": (
                        self._native_provenance["library_sha256"]
                        if self._native_kernel is not None
                        else "unavailable"
                    ),
                    "cache_control_primitive": self._cache_provenance["primitive"],
                    "measurement_primitive": self._measurement_primitive.name,
                    "measurement_primitive_capabilities": json.dumps(
                        self._measurement_primitive.capabilities.as_dict(), sort_keys=True
                    ),
                    "access_state_oracle_provenance": json.dumps(
                        primitive_observation.access_state.as_dict(), sort_keys=True
                    ),
                    "physical_observation_semantics": primitive_observation.physical_observation,
                    "model_eligible_feature_policy": "trace-derived features only; primitive/audit metadata excluded",
                    "clflush_supported": (
                        self._native_kernel.supports_clflush
                        if self._native_kernel is not None
                        else False
                    ),
                    "timing_perturbation_cycles": self.timing_perturbation_cycles,
                    "timing_perturbation_label": self.timing_perturbation_label,
                    "timing_perturbation_applied": bool(
                        self.timing_perturbation_cycles > 0
                        and label == self.timing_perturbation_label
                    ),
                    "calibration_namespace": self.calibration_namespace or "not_calibration",
                    "artificial_timing_perturbation_allowed": self.calibration_context is not None,
                    "configuration_hash": self.configuration_hash or "unavailable",
                    "code_commit": self.code_commit or "unavailable",
                    "seed_id": f"commodity:{self.seed}",
                    "label_stream_fingerprint": hashlib.sha256(self._labels.tobytes()).hexdigest(),
                    "session_manifest_ref": f"sessions/{self.acquisition_session_id}/session.json",
                    "protocol_identity": self.protocol_identity or "unavailable",
                    "protocol_hash": self.protocol_hash or "unavailable",
                    "operation_scoped_perf_observation": json.dumps(
                        scoped_perf_provenance or {"status": "not_configured"}, sort_keys=True
                    ),
                },
            )

    def session_provenance(self) -> dict[str, Any]:
        """Return the immutable acquisition-session ledger for this backend."""

        return {
            "schema": "sensetrace.acquisition-session.v1",
            "acquisition_session_id": self.acquisition_session_id,
            "session_id": self.acquisition_session_id,
            "campaign_id": self.campaign_id or "unavailable",
            "session_index": self.session_index,
            "started_at": self.session_started_at,
            "boot_id": self._boot_id_value,
            "host_inventory_snapshot": self._host_inventory_snapshot or {"value": "unavailable"},
            "controlled_memory_region": {
                "allocation": "fresh page-aligned anonymous mmap allocated for this session",
                "allocation_id": self._allocation_id,
                "word_count": self.word_count,
                "memory_lock_requested": self._buffer.lock_requested,
                "memory_lock_actual": self._buffer.locked,
                "memory_lock_error": self._buffer.lock_error or "none",
                "physical_topology": "unknown; virtual buffer location only",
            },
            "label_stream_fingerprint": hashlib.sha256(self._labels.tobytes()).hexdigest(),
            "pair_order_balance": self.pair_order_balance(),
            "measurement_kernel_provenance": self._native_provenance
            or {
                "implementation": "python fallback timing control",
                "limitations": "not a native timing kernel",
            },
            "cache_control_provenance": self._cache_provenance,
            "measurement_primitive": self._measurement_primitive.describe(),
            "operation_scoped_perf_oracle": (
                self.scoped_perf_oracle.description.as_dict()
                if self.scoped_perf_oracle is not None
                else {"status": "not_configured"}
            ),
            "cpu_frequency_regime": self._frequency_regime,
            "configuration_hash": self.configuration_hash or "unavailable",
            "code_commit": self.code_commit or "unavailable",
            "protocol_identity": self.protocol_identity or "unavailable",
            "protocol_hash": self.protocol_hash or "unavailable",
            "timing_perturbation": {
                "cycles": self.timing_perturbation_cycles,
                "label": self.timing_perturbation_label,
                "mechanism": (
                    "st_measure_flushed_control/st_measure_cached_control uses one delayed-capable "
                    "timed-load path; delay_cycles is zero for the control"
                ),
                "namespace": self.calibration_namespace or "not_calibration",
                "allowed": self.calibration_context is not None,
                "physical_phase1a_forbidden": True,
                "scope": (
                    "explicit TimingPerturbationCalibration context"
                    if self.calibration_context is not None
                    else "physical zero only"
                ),
            },
        }

    def pair_order_balance(self) -> dict[str, Any]:
        values = self._pair_order
        zero_first = int(np.sum(values == 0))
        one_first = int(np.sum(values == 1))
        per_location = []
        pair_count = self.trials_per_location // 2
        for location in range(int(self.location_count or 1)):
            local = values[location * pair_count : (location + 1) * pair_count]
            per_location.append(
                {
                    "virtual_location_index": location,
                    "label_0_first": int(np.sum(local == 0)),
                    "label_1_first": int(np.sum(local == 1)),
                    "exact": bool(np.sum(local == 0) == np.sum(local == 1)),
                }
            )
        return {
            "pair_count": int(len(values)),
            "label_0_first": zero_first,
            "label_1_first": one_first,
            "exact": zero_first == one_first and all(item["exact"] for item in per_location),
            "per_location": per_location,
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_buffer:
            self._buffer.close()
        self._closed = True
        if self._affinity_before is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(self._affinity_before))

    @staticmethod
    def _boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            return "unavailable"

    @staticmethod
    def _cpu_id() -> int | str:
        get_cpu = getattr(os, "sched_getcpu", None)
        if get_cpu is not None:
            try:
                return int(get_cpu())
            except OSError:
                pass
        # Some builds (observed on worker-03) lack os.sched_getcpu; fall back
        # to libc so per-operation CPU telemetry is not silently blind.
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            cpu = libc.sched_getcpu()
            if int(cpu) >= 0:
                return int(cpu)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        return platform.processor() or "unavailable"
