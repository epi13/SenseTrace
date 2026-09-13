"""Controlled-excitation, quiet-interval predictive-state campaign.

This module is intentionally a small extension beside the historical passive
``real_horizon`` path.  It owns a versioned acquisition record, a causal
streaming interface, and a bounded continuous model ladder.  The data source
is still an ordinary user-space virtual allocation: this code never promotes
virtual spacing or cache timing to DRAM topology or hidden-state evidence.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .acquisition.commodity import ControlledMemoryBuffer
from .acquisition.native import (
    TRAJECTORY_CONDITIONS,
    NativeMeasurementKernel,
)
from .config import config_fingerprint
from .errors import IntegrityError, SchemaError
from .excitation import CodedExcitationSchedule, ExcitationExecution
from .hashing import sha256_file
from .runner import _git_commit
from .worker03 import collect_worker03_inventory

CONTROLLED_FORECAST_PROTOCOL_VERSION = "controlled-predictive-state-v1"
ACQUISITION_SCHEMA = "sensetrace.controlled-forecast-acquisition.v1"
ANALYSIS_SCHEMA = "sensetrace.controlled-forecast-analysis.v1"
QUALITY_AUX_PRESENT = 1
QUALITY_AUX_MISMATCH = 2
_MODEL_NAMES = (
    "training_mean",
    "persistence",
    "current_observation",
    "workload_only",
    "workload_history",
    "arx_history",
    "delay_dmd_control",
)
_BASELINES = (
    "training_mean",
    "persistence",
    "current_observation",
    "workload_only",
    "workload_history",
)
_CANDIDATES = ("arx_history", "delay_dmd_control")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cache_line_size() -> tuple[int | None, str]:
    candidates = [
        Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"),
    ]
    for path in candidates:
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if value > 0:
            return value, "Linux cache sysfs"
    try:
        value = int(os.sysconf("SC_LEVEL1_DCACHE_LINESIZE"))
    except (AttributeError, ValueError, OSError):
        return None, "unavailable"
    return (value, "POSIX sysconf") if value > 0 else (None, "unavailable")


def _tsc_calibration(kernel: NativeMeasurementKernel, seconds: float = 0.025) -> dict[str, Any]:
    """Calibrate TSC ticks against monotonic nanoseconds in a separate control."""

    start_ns = time.monotonic_ns()
    start_tsc, start_aux = kernel.read_tsc_aux()
    time.sleep(seconds)
    end_tsc, end_aux = kernel.read_tsc_aux()
    end_ns = time.monotonic_ns()
    elapsed_ns = max(end_ns - start_ns, 1)
    elapsed_tsc = max(end_tsc - start_tsc, 1)
    return {
        "method": "paired monotonic_ns and RDTSCP endpoint reads outside acquisition",
        "sample_sleep_seconds": seconds,
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": end_ns,
        "start_tsc": start_tsc,
        "end_tsc": end_tsc,
        "start_aux": start_aux,
        "end_aux": end_aux,
        "tsc_ticks_per_nanosecond": elapsed_tsc / elapsed_ns,
        "nanoseconds_per_tsc_tick": elapsed_ns / elapsed_tsc,
        "endpoint_aux_equal": start_aux == end_aux,
        "interpretation": (
            "conversion for elapsed-time reporting; TSC ticks remain the raw measurement unit; "
            "AUX equality is endpoint evidence only"
        ),
    }


def _requested_cpu_affinity(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise SchemaError("controlled_forecast.cpu_affinity must be a non-empty list when set")
    return [int(item) for item in value]


def validate_controlled_forecast_config(config: Mapping[str, Any]) -> dict[str, Any]:
    campaign = config.get("controlled_forecast")
    if not isinstance(campaign, Mapping):
        raise SchemaError("controlled_forecast configuration is required")
    if campaign.get("protocol_version", CONTROLLED_FORECAST_PROTOCOL_VERSION) != (
        CONTROLLED_FORECAST_PROTOCOL_VERSION
    ):
        raise SchemaError("unsupported controlled_forecast.protocol_version")
    stage = str(campaign.get("stage", "development"))
    if stage not in {"development", "confirmation"}:
        raise SchemaError("controlled_forecast.stage must be development or confirmation")
    for name, minimum in {
        "sessions_per_cell": 1,
        "trajectories_per_session": 1,
        "baseline_repetitions": 2,
        "excitation_length": 2,
        "repetitions_per_phase": 2,
        "quiet_future_repetitions": 4,
        "history_length": 2,
    }.items():
        value = campaign.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise SchemaError(f"controlled_forecast.{name} must be an integer >= {minimum}")
    if int(campaign["repetitions_per_phase"]) % 2:
        raise SchemaError("controlled_forecast.repetitions_per_phase must be even for paired order balance")
    families = campaign.get("excitation_families", ["read_pressure", "active_quiet", "sham", "passive"])
    allowed_families = {"read_pressure", "active_quiet", "sham", "passive"}
    if (
        not isinstance(families, list)
        or not families
        or any(str(value) not in allowed_families for value in families)
        or len(set(str(value) for value in families)) != len(families)
    ):
        raise SchemaError("controlled_forecast.excitation_families is invalid")
    conditions = campaign.get(
        "measurement_conditions",
        ["cached_preloaded", "clflush", "eviction", "timer_only"],
    )
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(str(value) not in TRAJECTORY_CONDITIONS for value in conditions)
        or len(set(str(value) for value in conditions)) != len(conditions)
    ):
        raise SchemaError("controlled_forecast.measurement_conditions is invalid")
    horizons = campaign.get("horizons", [1, 4, 8, 16])
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise SchemaError("controlled_forecast.horizons must be unique positive integers")
    block_length = campaign.get("future_block_length", 8)
    if not isinstance(block_length, int) or block_length < 1:
        raise SchemaError("controlled_forecast.future_block_length must be positive")
    if int(campaign["quiet_future_repetitions"]) < max(horizons) + block_length:
        raise SchemaError("quiet_future_repetitions must cover the largest target block")
    primary = campaign.get("primary", {})
    if not isinstance(primary, Mapping):
        raise SchemaError("controlled_forecast.primary must be a mapping")
    if str(primary.get("excitation_family", "read_pressure")) not in set(str(x) for x in families):
        raise SchemaError("primary excitation family is not acquired")
    if str(primary.get("measurement_condition", "cached_preloaded")) not in set(
        str(x) for x in conditions
    ):
        raise SchemaError("primary measurement condition is not acquired")
    if int(primary.get("horizon", max(horizons))) not in horizons:
        raise SchemaError("primary horizon must be one of controlled_forecast.horizons")
    if str(primary.get("target", "future_block_mean")) not in {
        "future_timing_level",
        "future_block_mean",
    }:
        raise SchemaError("primary target is unsupported")
    for name in ("phase_duration_us", "eviction_bytes", "word_count"):
        value = campaign.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SchemaError(f"controlled_forecast.{name} must be a positive integer")
    rank = campaign.get("state_rank", 3)
    if not isinstance(rank, int) or rank < 1:
        raise SchemaError("controlled_forecast.state_rank must be positive")
    alpha = campaign.get("ridge_alpha", 1.0)
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise SchemaError("controlled_forecast.ridge_alpha must be positive")
    return dict(config)


@dataclass
class ForecastTrajectory:
    trajectory_id: str
    session_id: str
    family: str
    measurement_condition: str
    target_ticks: np.ndarray
    reference_ticks: np.ndarray
    target_start_tsc: np.ndarray
    target_end_tsc: np.ndarray
    reference_start_tsc: np.ndarray
    reference_end_tsc: np.ndarray
    target_start_aux: np.ndarray
    target_end_aux: np.ndarray
    reference_start_aux: np.ndarray
    reference_end_aux: np.ndarray
    target_quality: np.ndarray
    reference_quality: np.ndarray
    acquisition_order: np.ndarray
    workload_history: np.ndarray
    origin_index: int
    metadata: dict[str, Any]

    def validate(self) -> None:
        arrays = [
            self.target_ticks,
            self.reference_ticks,
            self.target_start_tsc,
            self.target_end_tsc,
            self.target_start_aux,
            self.target_end_aux,
            self.target_quality,
            self.acquisition_order,
            self.workload_history,
        ]
        if any(np.asarray(item).ndim != 1 for item in arrays):
            raise SchemaError(f"trajectory {self.trajectory_id} contains a non-vector channel")
        lengths = {len(np.asarray(item)) for item in arrays}
        if len(lengths) != 1:
            raise SchemaError(f"trajectory {self.trajectory_id} channel lengths disagree")
        if self.reference_ticks.shape != self.target_ticks.shape:
            raise SchemaError("reference timing channel is not aligned")
        if len(self.target_ticks) < self.origin_index + 2:
            raise SchemaError("trajectory does not contain a disjoint post-origin observation")
        if not 0 <= self.origin_index < len(self.target_ticks):
            raise SchemaError("trajectory origin is outside the observation interval")
        if not np.isfinite(np.asarray(self.workload_history, dtype=np.float64)).all():
            raise SchemaError("workload history is not finite")
        if np.any((self.acquisition_order < 0) | (self.acquisition_order > 1)):
            raise SchemaError("acquisition order contains an invalid value")
        for value in (self.family, self.measurement_condition, self.session_id, self.trajectory_id):
            if not isinstance(value, str) or not value.strip():
                raise SchemaError("trajectory identity fields must be non-empty strings")

    def as_journal_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "family": self.family,
            "measurement_condition": self.measurement_condition,
            "origin_index": self.origin_index,
            "target_ticks": np.asarray(self.target_ticks, dtype=np.uint64).tolist(),
            "reference_ticks": np.asarray(self.reference_ticks, dtype=np.int64).tolist(),
            "target_start_tsc": np.asarray(self.target_start_tsc, dtype=np.uint64).tolist(),
            "target_end_tsc": np.asarray(self.target_end_tsc, dtype=np.uint64).tolist(),
            "reference_start_tsc": np.asarray(self.reference_start_tsc, dtype=np.int64).tolist(),
            "reference_end_tsc": np.asarray(self.reference_end_tsc, dtype=np.int64).tolist(),
            "target_start_aux": np.asarray(self.target_start_aux, dtype=np.int64).tolist(),
            "target_end_aux": np.asarray(self.target_end_aux, dtype=np.int64).tolist(),
            "reference_start_aux": np.asarray(self.reference_start_aux, dtype=np.int64).tolist(),
            "reference_end_aux": np.asarray(self.reference_end_aux, dtype=np.int64).tolist(),
            "target_quality": np.asarray(self.target_quality, dtype=np.uint8).tolist(),
            "reference_quality": np.asarray(self.reference_quality, dtype=np.uint8).tolist(),
            "acquisition_order": np.asarray(self.acquisition_order, dtype=np.uint8).tolist(),
            "workload_history": np.asarray(self.workload_history, dtype=np.float32).tolist(),
            "metadata": self.metadata,
        }


def _record_from_journal(value: Mapping[str, Any]) -> ForecastTrajectory:
    record = ForecastTrajectory(
        trajectory_id=str(value["trajectory_id"]),
        session_id=str(value["session_id"]),
        family=str(value["family"]),
        measurement_condition=str(value["measurement_condition"]),
        target_ticks=np.asarray(value["target_ticks"], dtype=np.uint64),
        reference_ticks=np.asarray(value["reference_ticks"], dtype=np.int64),
        target_start_tsc=np.asarray(value["target_start_tsc"], dtype=np.uint64),
        target_end_tsc=np.asarray(value["target_end_tsc"], dtype=np.uint64),
        reference_start_tsc=np.asarray(value["reference_start_tsc"], dtype=np.int64),
        reference_end_tsc=np.asarray(value["reference_end_tsc"], dtype=np.int64),
        target_start_aux=np.asarray(value["target_start_aux"], dtype=np.int64),
        target_end_aux=np.asarray(value["target_end_aux"], dtype=np.int64),
        reference_start_aux=np.asarray(value["reference_start_aux"], dtype=np.int64),
        reference_end_aux=np.asarray(value["reference_end_aux"], dtype=np.int64),
        target_quality=np.asarray(value["target_quality"], dtype=np.uint8),
        reference_quality=np.asarray(value["reference_quality"], dtype=np.uint8),
        acquisition_order=np.asarray(value["acquisition_order"], dtype=np.uint8),
        workload_history=np.asarray(value["workload_history"], dtype=np.float32),
        origin_index=int(value["origin_index"]),
        metadata=dict(value.get("metadata", {})),
    )
    record.validate()
    return record


def _map_native_channels(
    raw: Mapping[str, np.ndarray], orders: np.ndarray, *, paired: bool
) -> dict[str, np.ndarray]:
    first_duration = np.asarray(raw["first_end_tsc"] - raw["first_start_tsc"], dtype=np.uint64)
    result: dict[str, np.ndarray] = {
        "first_ticks": first_duration,
        "first_start_tsc": np.asarray(raw["first_start_tsc"], dtype=np.uint64),
        "first_end_tsc": np.asarray(raw["first_end_tsc"], dtype=np.uint64),
        "first_start_aux": np.asarray(raw["first_start_aux"], dtype=np.int64),
        "first_end_aux": np.asarray(raw["first_end_aux"], dtype=np.int64),
        "first_quality": np.asarray(raw["first_quality"], dtype=np.uint8),
    }
    length = len(first_duration)
    if not paired:
        result.update(
            {
                "target_ticks": first_duration,
                "reference_ticks": np.full(length, -1, dtype=np.int64),
                "target_start_tsc": result["first_start_tsc"],
                "target_end_tsc": result["first_end_tsc"],
                "reference_start_tsc": np.full(length, -1, dtype=np.int64),
                "reference_end_tsc": np.full(length, -1, dtype=np.int64),
                "target_start_aux": result["first_start_aux"],
                "target_end_aux": result["first_end_aux"],
                "reference_start_aux": np.full(length, -1, dtype=np.int64),
                "reference_end_aux": np.full(length, -1, dtype=np.int64),
                "target_quality": result["first_quality"],
                "reference_quality": np.zeros(length, dtype=np.uint8),
            }
        )
        return result
    second_start = np.asarray(raw["second_start_tsc"], dtype=np.uint64)
    second_end = np.asarray(raw["second_end_tsc"], dtype=np.uint64)
    second_duration = second_end - second_start
    first_is_target = orders == 0
    def select(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.where(first_is_target, first, second)
    def select_signed(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.where(first_is_target, first, second)
    result.update(
        {
            "target_ticks": select(first_duration, second_duration),
            "reference_ticks": select_signed(second_duration.astype(np.int64), first_duration.astype(np.int64)),
            "target_start_tsc": select(result["first_start_tsc"], second_start),
            "target_end_tsc": select(result["first_end_tsc"], second_end),
            "reference_start_tsc": select_signed(second_start.astype(np.int64), result["first_start_tsc"].astype(np.int64)),
            "reference_end_tsc": select_signed(second_end.astype(np.int64), result["first_end_tsc"].astype(np.int64)),
            "target_start_aux": select(result["first_start_aux"], np.asarray(raw["second_start_aux"], dtype=np.int64)),
            "target_end_aux": select(result["first_end_aux"], np.asarray(raw["second_end_aux"], dtype=np.int64)),
            "reference_start_aux": select_signed(np.asarray(raw["second_start_aux"], dtype=np.int64), result["first_start_aux"]),
            "reference_end_aux": select_signed(np.asarray(raw["second_end_aux"], dtype=np.int64), result["first_end_aux"]),
            "target_quality": select(result["first_quality"], np.asarray(raw["second_quality"], dtype=np.uint8)),
            "reference_quality": select_signed(np.asarray(raw["second_quality"], dtype=np.uint8), result["first_quality"]),
        }
    )
    return result


class _ExcitationPhase:
    def __init__(self, requested: tuple[int, ...], actual: dict[str, Any], value: float):
        self.requested = requested
        self.actual = actual
        self.value = value


def _counterbalanced_orders(rng: np.random.Generator, count: int) -> np.ndarray:
    orders = np.concatenate(
        [np.zeros(count // 2, dtype=np.uint8), np.ones(count // 2, dtype=np.uint8)]
    )
    if count % 2:
        # The origin is intentionally one observation, so exact balance is
        # impossible there; keep the extra order explicitly randomized.
        orders = np.concatenate([orders, rng.integers(0, 2, size=1, dtype=np.uint8)])
    rng.shuffle(orders)
    return orders


def _measure_block(
    kernel: NativeMeasurementKernel,
    buffer: ControlledMemoryBuffer,
    target_index: int,
    reference_index: int,
    condition: str,
    count: int,
    orders: np.ndarray,
    eviction: bytearray,
    *,
    paired: bool,
) -> dict[str, np.ndarray]:
    target_address = buffer.address + target_index * 8
    reference_address = buffer.address + reference_index * 8 if paired else None
    code = np.full(count, TRAJECTORY_CONDITIONS[condition], dtype=np.uint8)
    return kernel.measure_trajectory(
        target_address,
        reference_address,
        code,
        orders if paired else np.zeros(count, dtype=np.uint8),
        eviction_address=ctypes_address(eviction),
        eviction_bytes=len(eviction),
    )


def ctypes_address(value: Any) -> int:
    """Return the address of a writable bytearray without retaining a ctypes view."""

    import ctypes

    view = (ctypes.c_uint8 * len(value)).from_buffer(value)
    return ctypes.addressof(view)


def _run_pressure_phase(
    kernel: NativeMeasurementKernel,
    buffer: ControlledMemoryBuffer,
    pressure_index: int,
    pressure_words: int,
    cores: Sequence[int],
    duration_cycles: int,
    requested_active: tuple[int, ...],
    operation: str,
    measure: callable,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    active_cores = [cpu for cpu, active in zip(cores, requested_active, strict=True) if active]
    barrier = threading.Barrier(len(active_cores) + 1)
    results: dict[int, dict[str, int] | str] = {}
    native_started = {cpu: ctypes.c_uint32(0) for cpu in active_cores}

    def worker(cpu: int) -> None:
        try:
            barrier.wait()
            result = kernel.run_memory_pressure(
                buffer.address + pressure_index * 8,
                pressure_words,
                duration_cycles,
                operation=operation,
                requested_cpu=cpu,
                started_out=native_started[cpu],
            )
            results[cpu] = result
        except Exception as exc:  # retained in execution evidence below
            results[cpu] = f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max(len(active_cores), 1)) as pool:
        futures = [pool.submit(worker, cpu) for cpu in active_cores]
        barrier.wait()
        # The native ABI raises a per-worker flag immediately after its RDTSCP
        # start boundary.  Wait for all flags before launching the measurement
        # block so an active phase is not merely a requested overlap.
        deadline = time.monotonic() + max(1.0, duration_cycles / 1e9)
        while active_cores and not all(native_started[cpu].value for cpu in active_cores):
            if time.monotonic() >= deadline:
                raise RuntimeError("memory-pressure workers did not reach their native start boundary")
            time.sleep(0.00001)
        measured = measure()
        for future in futures:
            future.result()
    successful = [value for value in results.values() if isinstance(value, dict)]
    failed = [value for value in results.values() if isinstance(value, str)]
    if successful:
        start_clock = min(int(value["start_tsc"]) for value in successful)
        end_clock = max(int(value["end_tsc"]) for value in successful)
    else:
        start_clock = end_clock = 0
    actual = {
        "requested_active_cores": list(active_cores),
        "workers": {str(cpu): value for cpu, value in sorted(results.items())},
        "native_start_flags": {str(cpu): int(value.value) for cpu, value in sorted(native_started.items())},
        "start_clock": start_clock,
        "end_clock": end_clock,
        "compliance": "compliant" if not failed and len(successful) == len(active_cores) else "failed",
        "synchronization": "thread barrier before native worker and measurement call",
        "measurement_overlap_claim": "measurement call was launched after barrier; exact per-probe overlap is not asserted",
    }
    if not active_cores:
        actual.update(
            {
                "start_clock": int(kernel.read_tsc_aux()[0]),
                "end_clock": int(kernel.read_tsc_aux()[0]),
                "compliance": "compliant",
            }
        )
    return measured, actual, float(len(active_cores)) / max(len(cores), 1)


def _schedule_for(family: str, length: int, seed: int) -> CodedExcitationSchedule:
    if family in {"sham", "passive"}:
        return CodedExcitationSchedule(
            f"{family}-{seed}", "sham", length, seed, operation="idle", phase_ticks=1
        )
    operation = "read"
    return CodedExcitationSchedule(f"{family}-{seed}", cast(Any, family), length, seed, operation=operation, phase_ticks=1)


def _acquire_trajectory(
    *,
    kernel: NativeMeasurementKernel,
    campaign: Mapping[str, Any],
    family: str,
    condition: str,
    session_id: str,
    trajectory_id: str,
    seed: int,
    session_provenance: Mapping[str, Any],
) -> ForecastTrajectory:
    line_size, line_source = _cache_line_size()
    if line_size is None:
        raise SchemaError("worker-03 trajectory acquisition requires observed cache-line size")
    line_words = max(1, (line_size + 7) // 8)
    word_count = int(campaign["word_count"])
    reference_index = line_words
    pressure_index = reference_index + line_words
    pressure_words = word_count - pressure_index
    if pressure_words < 1:
        raise SchemaError("word_count is too small for target/reference/pressure spacing")
    buffer = ControlledMemoryBuffer(word_count, lock_memory=bool(campaign.get("lock_memory", True)))
    eviction = bytearray(int(campaign["eviction_bytes"]))
    rng = np.random.default_rng(seed)
    try:
        warmup = buffer.warmup_touch()
        buffer.write(0, int(rng.integers(0, 2**64, dtype=np.uint64)))
        buffer.write(reference_index, int(rng.integers(0, 2**64, dtype=np.uint64)))
        paired = condition != "timer_only"
        baseline_count = int(campaign["baseline_repetitions"])
        phase_count = int(campaign["repetitions_per_phase"])
        future_count = int(campaign["quiet_future_repetitions"])
        all_channels: list[dict[str, np.ndarray]] = []
        all_workload: list[float] = []
        all_orders: list[int] = []
        phase_records: list[dict[str, Any]] = []
        executed_steps: list[tuple[int, ...]] = []
        interrupted_positions: list[int] = []

        def block(count: int) -> dict[str, np.ndarray]:
            orders = _counterbalanced_orders(rng, count) if paired else np.zeros(count, dtype=np.uint8)
            measured = _measure_block(
                kernel, buffer, 0, reference_index, condition, count, orders, eviction, paired=paired
            )
            measured["orders"] = orders
            return measured

        baseline = block(baseline_count)
        all_channels.append(baseline)
        all_workload.extend([0.0] * baseline_count)
        all_orders.extend(baseline["orders"].astype(int).tolist())

        schedule = _schedule_for(family, int(campaign["excitation_length"]), seed)
        schedule.validate()
        cores = tuple(int(cpu) for cpu in campaign.get("excitation_cores", [2, 3, 4, 5, 6]))
        duration_ticks = max(
            1,
            int(
                float(campaign["phase_duration_us"])
                * 1000.0
                * float(session_provenance["timing_conversion"]["tsc_ticks_per_nanosecond"])
            ),
        )
        for step in schedule.steps():
            orders = _counterbalanced_orders(rng, phase_count) if paired else np.zeros(phase_count, dtype=np.uint8)
            measured, actual, active_fraction = _run_pressure_phase(
                kernel,
                buffer,
                pressure_index,
                pressure_words,
                cores,
                duration_ticks,
                tuple(int(value) for value in step.active),
                "read",
                lambda count=phase_count, orders=orders: _measure_block(
                    kernel, buffer, 0, reference_index, condition, count, orders, eviction, paired=paired
                ),
            )
            measured["orders"] = orders
            all_channels.append(measured)
            all_workload.extend([active_fraction] * phase_count)
            all_orders.extend(orders.astype(int).tolist())
            phase_records.append(
                {
                    "sequence_position": step.sequence_position,
                    "requested_active": list(step.active),
                    "actual_active_fraction": active_fraction,
                    "execution": actual,
                }
            )
            if actual["compliance"] == "compliant":
                executed_steps.append(tuple(int(value) for value in step.active))
            else:
                interrupted_positions.append(step.sequence_position)

        stop_confirmed_tsc = max(
            [int(item["execution"]["end_clock"]) for item in phase_records if item["execution"]["end_clock"]]
            or [int(kernel.read_tsc_aux()[0])]
        )
        excitation_compliance = (
            "compliant"
            if len(executed_steps) == len(schedule.steps())
            else "partial"
            if executed_steps
            else "failed"
        )
        excitation_execution = ExcitationExecution(
            schedule_fingerprint=schedule.fingerprint(),
            executed_code=tuple(executed_steps),
            start_clock=min(
                [int(item["execution"]["start_clock"]) for item in phase_records if item["execution"]["start_clock"]]
                or [stop_confirmed_tsc]
            ),
            end_clock=stop_confirmed_tsc,
            core_ids=tuple(cores) if cores else (7,),
            interrupted_positions=tuple(interrupted_positions),
            compliance=cast(Any, excitation_compliance),
            witness={"status": "disabled"},
        )
        excitation_execution.validate(schedule)
        origin = block(1)
        origin_index = sum(len(item["first_start_tsc"]) for item in all_channels)
        all_channels.append(origin)
        all_workload.append(0.0)
        all_orders.extend(origin["orders"].astype(int).tolist())
        quiet = block(future_count)
        all_channels.append(quiet)
        all_workload.extend([0.0] * future_count)
        all_orders.extend(quiet["orders"].astype(int).tolist())

        merged: dict[str, np.ndarray] = {}
        for key in (
            "target_ticks",
            "reference_ticks",
            "target_start_tsc",
            "target_end_tsc",
            "reference_start_tsc",
            "reference_end_tsc",
            "target_start_aux",
            "target_end_aux",
            "reference_start_aux",
            "reference_end_aux",
            "target_quality",
            "reference_quality",
        ):
            values = []
            for item in all_channels:
                mapped = _map_native_channels(item, item["orders"], paired=paired)
                values.append(mapped[key])
            merged[key] = np.concatenate(values)
        record = ForecastTrajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            family=family,
            measurement_condition=condition,
            target_ticks=merged["target_ticks"],
            reference_ticks=merged["reference_ticks"],
            target_start_tsc=merged["target_start_tsc"],
            target_end_tsc=merged["target_end_tsc"],
            reference_start_tsc=merged["reference_start_tsc"],
            reference_end_tsc=merged["reference_end_tsc"],
            target_start_aux=merged["target_start_aux"],
            target_end_aux=merged["target_end_aux"],
            reference_start_aux=merged["reference_start_aux"],
            reference_end_aux=merged["reference_end_aux"],
            target_quality=merged["target_quality"],
            reference_quality=merged["reference_quality"],
            acquisition_order=np.asarray(all_orders, dtype=np.uint8),
            workload_history=np.asarray(all_workload, dtype=np.float32),
            origin_index=origin_index,
            metadata={
                "schema": "sensetrace.controlled-forecast-trajectory.v1",
                "session_provenance": dict(session_provenance),
                "allocation_id": f"buffer-{uuid.uuid4().hex}",
                "allocation_warmup": warmup,
                "cache_line_size_bytes": line_size,
                "cache_line_size_source": line_source,
                "target_word_index": 0,
                "reference_word_index": reference_index,
                "reference_spacing_bytes": reference_index * 8,
                "spacing_claim": "distinct observed cache lines only; no bank/row/channel claim",
                "paired_measurement": paired,
                "paired_order_policy": "exact per-block target-first/reference-first balance; randomized block order",
                "raw_duration_units": "TSC ticks",
                "derived_difference": "reference_ticks - target_ticks when reference is available; optional model feature",
                "sequential_probe_perturbation": "the first paired probe can alter the second probe's cache state",
                "schedule_request": schedule.request_record(),
                "excitation_execution": excitation_execution.as_dict(),
                "phase_execution": phase_records,
                "stop_confirmed_tsc": stop_confirmed_tsc,
                "forecast_origin_policy": "first quiet observation after all excitation workers joined; targets start strictly after origin",
                "quiet_policy": "no excitation workers are launched after stop_confirmed_tsc",
                "quality_policy": "quality bit 1 means AUX endpoints present; bit 2 means endpoint AUX mismatch; equality does not prove no migration between endpoints",
                "measurement_condition_code": TRAJECTORY_CONDITIONS[condition],
                "model_feature_policy": "raw channels, workload active fraction through origin, and origin position only; no IDs, seeds, future schedule, or future witness state",
            },
        )
        record.validate()
        return record
    finally:
        buffer.close()


def _make_design(config: Mapping[str, Any], output: Path, stage: str) -> dict[str, Any]:
    campaign = config["controlled_forecast"]
    path = output / "design.json"
    if path.exists():
        try:
            design = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("cannot read controlled-forecast design") from exc
        if design.get("config_hash") != config_fingerprint(dict(config)) or design.get("stage") != stage:
            raise IntegrityError("existing controlled-forecast design does not match config/stage")
        return design
    families = [str(value) for value in campaign.get("excitation_families", ["read_pressure", "active_quiet", "sham", "passive"])]
    conditions = [str(value) for value in campaign.get("measurement_conditions", list(TRAJECTORY_CONDITIONS))]
    sessions_per_cell = int(campaign["sessions_per_cell"])
    trajectories_per_session = int(campaign["trajectories_per_session"])
    base_seed = int(config.get("experiment", {}).get("seed", 1337))
    # Confirmation is a fresh realization, not a replay of development code
    # words.  The stage offset is part of the immutable design record.
    if stage == "confirmation":
        base_seed += 10_000_000
    sessions: list[dict[str, Any]] = []
    cell_index = 0
    for family in families:
        for condition in conditions:
            for repeat in range(sessions_per_cell):
                session_id = f"{stage}-session-{cell_index:03d}-{uuid.uuid4().hex}"
                sessions.append(
                    {
                        "session_id": session_id,
                        "family": family,
                        "measurement_condition": condition,
                        "repeat": repeat,
                        "seed": base_seed + cell_index * 100003 + repeat * 7919,
                        "trajectory_ids": [
                            f"{session_id}-trajectory-{index:03d}"
                            for index in range(trajectories_per_session)
                        ],
                    }
                )
                cell_index += 1
    design = {
        "schema": "sensetrace.controlled-forecast-design.v1",
        "protocol_version": CONTROLLED_FORECAST_PROTOCOL_VERSION,
        "stage": stage,
        "config_hash": config_fingerprint(dict(config)),
        "created_at": datetime.now(UTC).isoformat(),
        "sessions": sessions,
        "fresh_sequence_policy": "session design IDs are generated once and retained; confirmation uses a distinct output/design and therefore fresh schedule seeds",
    }
    _atomic_json(path, design)
    return design


def _session_provenance(
    kernel: NativeMeasurementKernel,
    config: Mapping[str, Any],
    timing_conversion: Mapping[str, Any],
    *,
    session_id: str,
    requested_affinity: list[int] | None,
) -> dict[str, Any]:
    observed_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "schema": "sensetrace.controlled-forecast-session.v1",
        "acquisition_session_id": session_id,
        "session_started_at": datetime.now(UTC).isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        if Path("/proc/sys/kernel/random/boot_id").exists()
        else "unavailable",
        "host": platform.node() or "unavailable",
        "code_commit": _git_commit(),
        "configuration_hash": config_fingerprint(dict(config)),
        "native": kernel.provenance(),
        "requested_cpu_affinity": requested_affinity or "unchanged",
        "observed_process_affinity": observed_affinity or "unavailable",
        "cpu_identity_interpretation": "process affinity is a requested/observed mask; per-probe AUX endpoints cannot prove no migration in the interval",
        "timing_conversion": dict(timing_conversion),
        "witness": {"status": "disabled", "synchronization_quality": "native barrier plus worker completion; no eBPF witness attached"},
        "configuration_identities": {
            "session_id": session_id,
            "allocation_identity": "fresh allocation per trajectory; physical address unavailable",
        },
    }


def _load_journal(path: Path) -> dict[str, ForecastTrajectory]:
    records: dict[str, ForecastTrajectory] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            record = _record_from_journal(value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, SchemaError) as exc:
            raise IntegrityError(f"invalid controlled-forecast journal line {line_number}") from exc
        if record.trajectory_id in records:
            raise IntegrityError(f"duplicate controlled-forecast journal trajectory {record.trajectory_id}")
        records[record.trajectory_id] = record
    return records


def _append_journal(path: Path, record: ForecastTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_journal_record(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_raw_artifact(root: Path, records: Sequence[ForecastTrajectory]) -> dict[str, Any]:
    if not records:
        raise IntegrityError("cannot write an empty controlled-forecast acquisition")
    for record in records:
        record.validate()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "raw_trajectories.npz"
    temporary = root / "raw_trajectories.npz.tmp"
    arrays: dict[str, Any] = {
        "target_ticks": np.stack([item.target_ticks for item in records]),
        "reference_ticks": np.stack([item.reference_ticks for item in records]),
        "target_start_tsc": np.stack([item.target_start_tsc for item in records]),
        "target_end_tsc": np.stack([item.target_end_tsc for item in records]),
        "reference_start_tsc": np.stack([item.reference_start_tsc for item in records]),
        "reference_end_tsc": np.stack([item.reference_end_tsc for item in records]),
        "target_start_aux": np.stack([item.target_start_aux for item in records]),
        "target_end_aux": np.stack([item.target_end_aux for item in records]),
        "reference_start_aux": np.stack([item.reference_start_aux for item in records]),
        "reference_end_aux": np.stack([item.reference_end_aux for item in records]),
        "target_quality": np.stack([item.target_quality for item in records]),
        "reference_quality": np.stack([item.reference_quality for item in records]),
        "acquisition_order": np.stack([item.acquisition_order for item in records]),
        "workload_history": np.stack([item.workload_history for item in records]),
        "origin_index": np.asarray([item.origin_index for item in records], dtype=np.int64),
        "trajectory_ids": np.asarray([item.trajectory_id for item in records]),
        "session_ids": np.asarray([item.session_id for item in records]),
        "families": np.asarray([item.family for item in records]),
        "measurement_conditions": np.asarray([item.measurement_condition for item in records]),
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    metadata = [item.metadata for item in records]
    _atomic_json(root / "trajectory_metadata.json", metadata)
    raw_hash = sha256_file(path)
    return {
        "schema": "sensetrace.controlled-forecast-raw.v1",
        "path": path.name,
        "sha256": raw_hash,
        "trajectory_count": len(records),
        "repetitions_per_trajectory": int(records[0].target_ticks.size),
        "raw_integer_channels": [
            "target_ticks",
            "reference_ticks",
            "target_start_tsc",
            "target_end_tsc",
            "reference_start_tsc",
            "reference_end_tsc",
            "target_start_aux",
            "target_end_aux",
            "reference_start_aux",
            "reference_end_aux",
            "target_quality",
            "reference_quality",
            "acquisition_order",
        ],
        "unavailable_reference_sentinel": -1,
        "trajectory_ids": [item.trajectory_id for item in records],
        "sensitive_data": False,
    }


def run_predictive_calibration(config: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Run a separate real instrument calibration, including a synthetic delay control."""

    campaign = validate_controlled_forecast_config(config)["controlled_forecast"]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        raise RuntimeError("controlled predictive calibration requires the native kernel")
    line_size, line_source = _cache_line_size()
    if line_size is None:
        raise SchemaError("calibration requires an observed cache-line size")
    line_words = max(1, (line_size + 7) // 8)
    buffer = ControlledMemoryBuffer(int(campaign["word_count"]), lock_memory=False)
    eviction = bytearray(int(campaign["eviction_bytes"]))
    try:
        buffer.warmup_touch()
        buffer.write(0, 0x1234567890ABCDEF)
        buffer.write(line_words, 0x0FEDCBA098765432)
        timing_conversion = _tsc_calibration(kernel)
        rows: dict[str, Any] = {}
        raw: dict[str, np.ndarray] = {}
        for condition in ("cached_preloaded", "clflush", "eviction", "timer_only"):
            paired = condition != "timer_only"
            count = int(campaign.get("calibration_repetitions", 128))
            orders = _counterbalanced_orders(np.random.default_rng(17), count) if paired else np.zeros(count, dtype=np.uint8)
            result = _measure_block(
                kernel, buffer, 0, line_words, condition, count, orders, eviction, paired=paired
            )
            mapped = _map_native_channels(result, orders, paired=paired)
            values = np.asarray(mapped["target_ticks"], dtype=np.float64)
            raw[condition] = mapped["target_ticks"]
            rows[condition] = {
                "sample_count": count,
                "median_ticks": float(np.median(values)),
                "quantiles_ticks": [float(value) for value in np.quantile(values, [0.05, 0.5, 0.95])],
                "quality_counts": {str(value): int(np.sum(mapped["target_quality"] == value)) for value in np.unique(mapped["target_quality"])},
                "paired": paired,
            }
        synthetic = kernel.measure_cached(
            buffer.address,
            int(campaign.get("synthetic_calibration_repetitions", 64)),
            extra_delay_cycles=int(campaign.get("synthetic_delay_cycles", 256)),
        )
        raw["synthetic_delay_control"] = synthetic.astype(np.uint64)
        np.savez_compressed(root / "calibration_raw.npz", **raw)
        result = {
            "schema": "sensetrace.controlled-forecast-calibration.v1",
            "dataset_role": "instrument_calibration_only; not a model reference corpus",
            "protocol_version": CONTROLLED_FORECAST_PROTOCOL_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(dict(config)),
            "native": kernel.provenance(),
            "line_size_bytes": line_size,
            "line_size_source": line_source,
            "timing_conversion": timing_conversion,
            "controls": rows,
            "synthetic_control": {
                "delay_cycles": int(campaign.get("synthetic_delay_cycles", 256)),
                "scope": "labeled calibration only; never used in physical campaign features",
                "median_ticks": float(np.median(synthetic)),
            },
            "raw_sha256": sha256_file(root / "calibration_raw.npz"),
            "claim_boundary": "native timer/cache-path calibration; no physical DRAM or hidden-state claim",
        }
        _atomic_json(root / "calibration.json", result)
        return result
    finally:
        buffer.close()


def run_controlled_forecast_acquisition(
    config: Mapping[str, Any], output: str | Path, *, stage: str | None = None
) -> dict[str, Any]:
    """Acquire fresh worker-03 trajectories and checkpoint after every trajectory."""

    config = validate_controlled_forecast_config(config)
    campaign = config["controlled_forecast"]
    effective_stage = stage or str(campaign.get("stage", "development"))
    if effective_stage not in {"development", "confirmation"}:
        raise SchemaError("acquisition stage must be development or confirmation")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    design = _make_design(config, root, effective_stage)
    journal_path = root / "trajectory_journal.jsonl"
    completed = _load_journal(journal_path)
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        raise RuntimeError("controlled predictive acquisition requires the native kernel")
    requested_affinity = _requested_cpu_affinity(campaign.get("measurement_cpu_affinity"))
    old_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if requested_affinity is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(requested_affinity))
    try:
        timing_conversion = _tsc_calibration(kernel)
        inventory = collect_worker03_inventory(
            native_library=Path(__file__).resolve().parents[2] / "native" / "libsensetrace_measurement.so"
        )
        for session in design["sessions"]:
            session_id = str(session["session_id"])
            provenance = _session_provenance(
                kernel,
                config,
                timing_conversion,
                session_id=session_id,
                requested_affinity=requested_affinity,
            )
            provenance["host_inventory_snapshot"] = inventory
            for index, trajectory_id in enumerate(session["trajectory_ids"]):
                if trajectory_id in completed:
                    continue
                print(
                    f"controlled-forecast {effective_stage}: {len(completed) + 1}/{sum(len(item['trajectory_ids']) for item in design['sessions'])} {trajectory_id}",
                    flush=True,
                )
                record = _acquire_trajectory(
                    kernel=kernel,
                    campaign=campaign,
                    family=str(session["family"]),
                    condition=str(session["measurement_condition"]),
                    session_id=session_id,
                    trajectory_id=str(trajectory_id),
                    seed=int(session["seed"]) + index * 101,
                    session_provenance=provenance,
                )
                _append_journal(journal_path, record)
                completed[trajectory_id] = record
                _atomic_json(
                    root / "progress.json",
                    {
                        "schema": "sensetrace.controlled-forecast-progress.v1",
                        "stage": effective_stage,
                        "completed_trajectory_count": len(completed),
                        "planned_trajectory_count": sum(len(item["trajectory_ids"]) for item in design["sessions"]),
                        "last_completed_trajectory_id": trajectory_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
        records = [completed[item] for session in design["sessions"] for item in session["trajectory_ids"] if item in completed]
        planned_count = sum(len(item["trajectory_ids"]) for item in design["sessions"])
        if len(records) != planned_count:
            raise IntegrityError("controlled-forecast acquisition ended before all trajectories completed")
        raw_manifest = _write_raw_artifact(root, records)
        acquisition = {
            "schema": ACQUISITION_SCHEMA,
            "protocol_version": CONTROLLED_FORECAST_PROTOCOL_VERSION,
            "stage": effective_stage,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "execution_host": platform.node() or "unavailable",
            "requested_node": config.get("run_metadata", {}).get("node", "worker-03"),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(dict(config)),
            "design": design,
            "raw_artifact": raw_manifest,
            "trajectory_journal": {"path": journal_path.name, "sha256": sha256_file(journal_path)},
            "session_count": len(design["sessions"]),
            "trajectory_count": len(records),
            "session_ids": [str(item["session_id"]) for item in design["sessions"]],
            "boot_ids": sorted({str(item.metadata["session_provenance"]["boot_id"]) for item in records}),
            "timing_units": {"raw": "TSC ticks", "elapsed": "converted nanoseconds from retained TSC endpoints"},
            "feature_firewall": {
                "allowed_through_origin": ["raw target/reference durations", "derived paired difference", "actual workload active fraction", "origin position", "quality masks"],
                "forbidden": ["future observations", "future witness events", "future scheduler state", "session/boot/allocation IDs", "seed", "full excitation schedule", "confirmation values", "whole-trajectory normalization"],
            },
            "excitation_contract": {
                "requested_vs_actual_separate": True,
                "stop_then_origin": True,
                "quiet_policy": "no workers after stop confirmation; first quiet observation is origin; targets are strictly later",
                "families": ["read_pressure", "active_quiet", "sham", "passive"],
                "passive_condition": "same measurement/origin contract with no excitation workers; diagnostic passive result kept separate from controlled-response result",
            },
            "witness": {"status": "disabled", "synchronization_quality": "barriers and native worker records retained"},
            "claim_boundary": "worker-03 ordinary user-space timing trajectory prediction; no hidden DRAM state, DRAM-origin, or unique mechanism claim",
        }
        _atomic_json(root / "acquisition.json", acquisition)
        _atomic_json(
            root / "experiment.json",
            {
                "schema": "sensetrace.controlled-forecast-experiment.v1",
                "stage": effective_stage,
                "acquisition": acquisition,
                "analysis_status": "not_run; confirmation data are untouched until the separate analysis command",
                "claim_boundary": acquisition["claim_boundary"],
            },
        )
        return acquisition
    finally:
        if old_affinity is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(old_affinity))


def load_controlled_forecast_acquisition(path: str | Path) -> list[ForecastTrajectory]:
    root = Path(path)
    try:
        manifest = json.loads((root / "acquisition.json").read_text(encoding="utf-8"))
        raw = manifest["raw_artifact"]
        raw_path = root / str(raw["path"])
        if raw["sha256"] != sha256_file(raw_path):
            raise IntegrityError("controlled-forecast raw artifact hash mismatch")
        with np.load(raw_path, allow_pickle=False) as archive:
            metadata = json.loads((root / "trajectory_metadata.json").read_text(encoding="utf-8"))
            ids = archive["trajectory_ids"].astype(str)
            records: list[ForecastTrajectory] = []
            for index, item in enumerate(metadata):
                record = ForecastTrajectory(
                    trajectory_id=str(ids[index]),
                    session_id=str(archive["session_ids"][index]),
                    family=str(archive["families"][index]),
                    measurement_condition=str(archive["measurement_conditions"][index]),
                    target_ticks=archive["target_ticks"][index].copy(),
                    reference_ticks=archive["reference_ticks"][index].copy(),
                    target_start_tsc=archive["target_start_tsc"][index].copy(),
                    target_end_tsc=archive["target_end_tsc"][index].copy(),
                    reference_start_tsc=archive["reference_start_tsc"][index].copy(),
                    reference_end_tsc=archive["reference_end_tsc"][index].copy(),
                    target_start_aux=archive["target_start_aux"][index].copy(),
                    target_end_aux=archive["target_end_aux"][index].copy(),
                    reference_start_aux=archive["reference_start_aux"][index].copy(),
                    reference_end_aux=archive["reference_end_aux"][index].copy(),
                    target_quality=archive["target_quality"][index].copy(),
                    reference_quality=archive["reference_quality"][index].copy(),
                    acquisition_order=archive["acquisition_order"][index].copy(),
                    workload_history=archive["workload_history"][index].copy(),
                    origin_index=int(archive["origin_index"][index]),
                    metadata=dict(item),
                )
                record.validate()
                records.append(record)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot load controlled-forecast acquisition under {root}") from exc
    if not records:
        raise IntegrityError("controlled-forecast acquisition contains no trajectories")
    if len(records) != int(manifest.get("trajectory_count", -1)):
        raise IntegrityError("controlled-forecast manifest trajectory count mismatch")
    return records


@dataclass
class FeatureBatch:
    current: np.ndarray
    workload: np.ndarray
    observation_history: np.ndarray
    arx: np.ndarray
    positions: np.ndarray


def _history(values: np.ndarray, origin: int, length: int) -> np.ndarray:
    values = np.asarray(values)
    start = max(0, origin - length + 1)
    selected = values[start : origin + 1]
    if len(selected) < length:
        selected = np.concatenate([np.zeros((length - len(selected), values.shape[1])), selected])
    return selected


def _feature_batch(records: Sequence[ForecastTrajectory], history_length: int) -> FeatureBatch:
    current: list[np.ndarray] = []
    workload: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    arx: list[np.ndarray] = []
    positions: list[float] = []
    for record in records:
        target = record.target_ticks.astype(np.float64)
        reference_available = (record.reference_ticks >= 0).astype(np.float64)
        reference = np.where(record.reference_ticks >= 0, record.reference_ticks, 0).astype(np.float64)
        difference = reference - target
        obs = np.column_stack([target, reference, difference, reference_available])
        obs_hist = _history(obs, record.origin_index, history_length)
        u_hist = _history(record.workload_history[:, None].astype(np.float64), record.origin_index, history_length)[:, 0]
        position = float(record.origin_index) / max(len(record.target_ticks) - 1, 1)
        current_row = np.asarray([*obs[record.origin_index], position], dtype=np.float64)
        current.append(current_row)
        workload.append(u_hist)
        observations.append(obs_hist.reshape(-1))
        arx.append(np.concatenate([obs_hist.reshape(-1), u_hist, [position]]))
        positions.append(position)
    return FeatureBatch(
        current=np.asarray(current, dtype=np.float64),
        workload=np.asarray(workload, dtype=np.float64),
        observation_history=np.asarray(observations, dtype=np.float64),
        arx=np.asarray(arx, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
    )


def _target_values(
    records: Sequence[ForecastTrajectory], target: str, horizon: int, block_length: int
) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    valid: list[bool] = []
    for record in records:
        start = record.origin_index + horizon
        end = start + (block_length if target == "future_block_mean" else 1)
        if end > len(record.target_ticks):
            values.append(float("nan"))
            valid.append(False)
            continue
        quality = record.target_quality[start:end]
        is_valid = bool(np.all((quality & QUALITY_AUX_PRESENT) != 0))
        values.append(float(np.mean(record.target_ticks[start:end], dtype=np.float64)))
        valid.append(is_valid)
    return np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)


def _split_sessions(records: Sequence[ForecastTrajectory], seed: int, fractions: Sequence[float]) -> dict[str, list[int]]:
    sessions = sorted({record.session_id for record in records})
    if len(sessions) < 3:
        raise SchemaError("controlled forecast requires at least three sessions for grouped evaluation")
    rng = np.random.default_rng(seed)
    shuffled = [sessions[index] for index in rng.permutation(len(sessions))]
    n = len(shuffled)
    train_count = max(1, int(round(n * fractions[0])))
    validation_count = max(1, int(round(n * fractions[1])))
    if train_count + validation_count >= n:
        validation_count = 1
        train_count = n - 2
    partitions = {
        "train": set(shuffled[:train_count]),
        "validation": set(shuffled[train_count : train_count + validation_count]),
        "test": set(shuffled[train_count + validation_count :]),
    }
    return {
        name: [index for index, record in enumerate(records) if record.session_id in values]
        for name, values in partitions.items()
    } | {"session_ids": shuffled}  # type: ignore[dict-item]


def _model_features(batch: FeatureBatch, model: str) -> np.ndarray:
    if model == "current_observation":
        return batch.current
    if model == "workload_history":
        # Strong workload baseline: current observed state plus the permitted
        # workload history through the origin.  This is the fair comparison
        # for asking whether observation history adds anything incremental.
        return np.hstack([batch.current, batch.workload])
    if model == "workload_only":
        return batch.workload
    if model == "arx_history":
        return batch.arx
    if model == "delay_dmd_control":
        return batch.observation_history
    raise SchemaError(f"model {model!r} does not use a fitted feature matrix")


@dataclass
class FittedModel:
    name: str
    target: str
    horizon: int
    history_length: int
    constant: float | None = None
    scaler: StandardScaler | None = None
    estimator: Ridge | None = None
    pca: PCA | None = None
    workload_width: int = 0

    def predict_batch(self, batch: FeatureBatch) -> np.ndarray:
        if self.name == "training_mean":
            assert self.constant is not None
            return np.full(len(batch.current), self.constant, dtype=np.float64)
        if self.name == "persistence":
            return batch.current[:, 0].astype(np.float64)
        if self.name == "delay_dmd_control":
            assert self.pca is not None and self.estimator is not None and self.scaler is not None
            state = self.pca.transform(batch.observation_history)
            values = np.hstack([state, batch.workload, batch.positions[:, None]])
        else:
            assert self.estimator is not None and self.scaler is not None
            values = _model_features(batch, self.name)
        return np.asarray(self.estimator.predict(self.scaler.transform(values)), dtype=np.float64)


def _fit_model(
    name: str,
    target: str,
    horizon: int,
    records: Sequence[ForecastTrajectory],
    train: Sequence[int],
    config: Mapping[str, Any],
) -> FittedModel:
    campaign = config["controlled_forecast"]
    history_length = int(campaign["history_length"])
    batch = _feature_batch(records, history_length)
    y, valid = _target_values(records, target, horizon, int(campaign["future_block_length"]))
    selected_train = np.asarray(train, dtype=np.int64)[valid[np.asarray(train, dtype=np.int64)]]
    if len(selected_train) < 2:
        raise SchemaError("not enough valid training targets after quality filtering")
    mean = float(np.mean(y[selected_train]))
    if name == "training_mean":
        return FittedModel(name, target, horizon, history_length, constant=mean)
    if name == "persistence":
        return FittedModel(name, target, horizon, history_length)
    if name == "delay_dmd_control":
        pca = PCA(n_components=min(int(campaign.get("state_rank", 3)), batch.observation_history.shape[1], len(selected_train)))
        state = pca.fit_transform(batch.observation_history[selected_train])
        x_train = np.hstack([state, batch.workload[selected_train], batch.positions[selected_train, None]])
    else:
        x_train = _model_features(batch, name)[selected_train]
    scaler = StandardScaler().fit(x_train)
    estimator = Ridge(alpha=float(campaign.get("ridge_alpha", 1.0)))
    estimator.fit(scaler.transform(x_train), y[selected_train])
    return FittedModel(name, target, horizon, history_length, scaler=scaler, estimator=estimator, pca=pca if name == "delay_dmd_control" else None, workload_width=batch.workload.shape[1])


class CausalForecastInterface:
    """Streaming forecast interface used for replay and live-shadow checks."""

    def __init__(self, models: Mapping[int, FittedModel], *, history_length: int):
        self.models = dict(models)
        self.history_length = history_length
        self._observations: list[np.ndarray] = []
        self._workload: list[float] = []

    def update(self, observation: Mapping[str, Any], available_inputs: Mapping[str, Any]) -> None:
        forbidden = {"future", "future_observation", "session_id", "boot_id", "seed", "schedule", "full_excitation_schedule", "witness_future"}
        if forbidden.intersection(observation) or forbidden.intersection(available_inputs):
            raise SchemaError("causal update received a forbidden future/identity input")
        if "target_ticks" not in observation:
            raise SchemaError("causal update requires target_ticks")
        target = float(observation["target_ticks"])
        reference = float(observation.get("reference_ticks", 0.0))
        reference_available = float(bool(observation.get("reference_available", "reference_ticks" in observation)))
        difference = reference - target if reference_available else 0.0
        self._observations.append(np.asarray([target, reference, difference, reference_available], dtype=np.float64))
        self._workload.append(float(available_inputs.get("workload_active", 0.0)))

    def forecast(self, horizons: Sequence[int], declared_future_policy: str) -> dict[int, float]:
        if declared_future_policy != "quiet":
            raise SchemaError("this interface only forecasts under the declared quiet policy")
        if not self._observations:
            raise SchemaError("cannot forecast before a causal update")
        obs = np.asarray(self._observations, dtype=np.float64)
        u = np.asarray(self._workload, dtype=np.float64)
        origin = len(obs) - 1
        dummy = ForecastTrajectory(
            "stream", "stream", "stream", "stream",
            target_ticks=obs[:, 0].astype(np.uint64),
            reference_ticks=np.where(obs[:, 3] > 0, obs[:, 1], -1).astype(np.int64),
            target_start_tsc=np.arange(len(obs), dtype=np.uint64), target_end_tsc=np.arange(len(obs), dtype=np.uint64),
            reference_start_tsc=np.full(len(obs), -1, dtype=np.int64), reference_end_tsc=np.full(len(obs), -1, dtype=np.int64),
            target_start_aux=np.zeros(len(obs), dtype=np.int64), target_end_aux=np.zeros(len(obs), dtype=np.int64),
            reference_start_aux=np.full(len(obs), -1, dtype=np.int64), reference_end_aux=np.full(len(obs), -1, dtype=np.int64),
            target_quality=np.ones(len(obs), dtype=np.uint8), reference_quality=np.zeros(len(obs), dtype=np.uint8),
            acquisition_order=np.zeros(len(obs), dtype=np.uint8), workload_history=u.astype(np.float32), origin_index=origin, metadata={"stream": True},
        )
        batch = _feature_batch([dummy], self.history_length)
        available: dict[int, float] = {}
        for horizon in horizons:
            if int(horizon) not in self.models:
                raise SchemaError(f"stream model has no declared horizon {horizon}")
            available[int(horizon)] = float(self.models[int(horizon)].predict_batch(batch)[0])
        return available


def _replay_predictions(
    models: Mapping[int, FittedModel], records: Sequence[ForecastTrajectory], indices: Sequence[int], horizons: Sequence[int], history_length: int
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    predictions = {int(horizon): [] for horizon in horizons}
    latencies: list[int] = []
    offline_batch = _feature_batch([records[index] for index in indices], history_length)
    offline = {int(horizon): models[int(horizon)].predict_batch(offline_batch) for horizon in horizons}
    agreements: list[bool] = []
    for row, index in enumerate(indices):
        record = records[index]
        stream = CausalForecastInterface(models, history_length=history_length)
        started = time.perf_counter_ns()
        for position in range(record.origin_index + 1):
            reference_available = record.reference_ticks[position] >= 0
            stream.update(
                {"target_ticks": int(record.target_ticks[position]), **({"reference_ticks": int(record.reference_ticks[position]), "reference_available": True} if reference_available else {})},
                {"workload_active": float(record.workload_history[position])},
            )
        result = stream.forecast(horizons, "quiet")
        latencies.append(time.perf_counter_ns() - started)
        for horizon in horizons:
            predictions[int(horizon)].append(result[int(horizon)])
            agreements.append(bool(np.isclose(result[int(horizon)], offline[int(horizon)][row], rtol=0.0, atol=1e-9)))
    lead_times: dict[str, Any] = {}
    for horizon in horizons:
        elapsed: list[float] = []
        for index in indices:
            record = records[index]
            target_index = record.origin_index + int(horizon)
            if target_index >= len(record.target_start_tsc):
                continue
            conversion = float(
                record.metadata.get("session_provenance", {})
                .get("timing_conversion", {})
                .get("nanoseconds_per_tsc_tick", float("nan"))
            )
            if np.isfinite(conversion):
                elapsed.append(
                    max(0.0, float(record.target_start_tsc[target_index] - record.target_end_tsc[record.origin_index]))
                    * conversion
                )
        p95_latency = float(np.quantile(latencies, 0.95)) if latencies else float("nan")
        lead_times[str(horizon)] = {
            "median_future_elapsed_ns": float(np.median(elapsed)) if elapsed else float("nan"),
            "p05_future_elapsed_ns": float(np.quantile(elapsed, 0.05)) if elapsed else float("nan"),
            "inference_p95_ns": p95_latency,
            "p05_lead_exceeds_inference_p95": bool(elapsed and np.quantile(elapsed, 0.05) > p95_latency),
            "interpretation": "future elapsed time is measured from origin endpoint to first target endpoint; comparison is actionable only when the lower tail exceeds inference cost",
        }
    return {horizon: np.asarray(values, dtype=np.float64) for horizon, values in predictions.items()}, {
        "offline_stream_agreement": bool(all(agreements)),
        "agreement_count": int(sum(agreements)),
        "comparison_count": len(agreements),
        "inference_latency_ns": {
            "median": float(np.median(latencies)) if latencies else float("nan"),
            "p95": float(np.quantile(latencies, 0.95)) if latencies else float("nan"),
            "sample_count": len(latencies),
            "includes": "causal warmup updates through origin plus all declared quiet-policy forecasts",
            "clean_acquisition_separate": True,
        },
        "lead_time": lead_times,
    }


def _bootstrap_loss_difference(
    targets: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int, repetitions: int = 500
) -> list[float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        values.append(float(np.mean((targets[indices] - baseline[indices]) ** 2) - np.mean((targets[indices] - candidate[indices]) ** 2)))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [float("nan"), float("nan")]


def _evaluate_model(
    model: FittedModel, records: Sequence[ForecastTrajectory], indices: Sequence[int], target: str, horizon: int, block_length: int, history_length: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    selected = [records[index] for index in indices]
    values, valid = _target_values(selected, target, horizon, block_length)
    valid_indices = np.flatnonzero(valid)
    batch = _feature_batch(selected, history_length)
    predictions = model.predict_batch(batch)
    values = values[valid_indices]
    predictions = predictions[valid_indices]
    if not len(values):
        raise SchemaError("no valid held-out target values")
    mse = float(np.mean((values - predictions) ** 2))
    constant = float(np.mean(values))
    baseline_mse = float(np.mean((values - constant) ** 2))
    return {
        "status": "evaluated",
        "sample_count": len(values),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(values - predictions))),
        "skill_over_split_mean": float(1.0 - mse / baseline_mse) if baseline_mse > 0 else float("nan"),
        "target_mean": constant,
        "quality_excluded_count": int(np.sum(~valid)),
    }, values, predictions


def _analyze_cell(
    records: Sequence[ForecastTrajectory], config: Mapping[str, Any], *, stage: str, family: str, condition: str, confirmation_records: Sequence[ForecastTrajectory] | None = None
) -> dict[str, Any]:
    campaign = config["controlled_forecast"]
    selected = [item for item in records if item.family == family and item.measurement_condition == condition]
    if len(selected) < 3:
        return {"status": "unavailable", "reason": "fewer than three trajectories in cell", "family": family, "measurement_condition": condition}
    horizons = [int(value) for value in campaign["horizons"]]
    history_length = int(campaign["history_length"])
    split = _split_sessions(selected, int(config.get("experiment", {}).get("seed", 1337)), [0.7, 0.15, 0.15])
    dev_report: dict[str, Any] = {}
    confirmation_report: dict[str, Any] = {}
    for target in ("future_timing_level", "future_block_mean"):
        target_report: dict[str, Any] = {}
        selected_streaming: dict[str, Any] = {}
        for horizon in horizons:
            fits = {name: _fit_model(name, target, horizon, selected, split["train"], config) for name in _MODEL_NAMES}
            validation: dict[str, float] = {}
            for name, model in fits.items():
                metrics, _, _ = _evaluate_model(model, selected, split["validation"], target, horizon, int(campaign["future_block_length"]), history_length)
                validation[name] = float(metrics["mse"])
            baseline = min(_BASELINES, key=lambda name: validation[name])
            candidate = min(_CANDIDATES, key=lambda name: validation[name])
            model_results: dict[str, Any] = {}
            all_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name, model in fits.items():
                metrics, y, p = _evaluate_model(model, selected, split["test"], target, horizon, int(campaign["future_block_length"]), history_length)
                model_results[name] = metrics
                all_predictions[name] = (y, p)
            y_base, p_base = all_predictions[baseline]
            y_candidate, p_candidate = all_predictions[candidate]
            if not np.array_equal(y_base, y_candidate):
                raise IntegrityError("paired baseline/candidate targets are not aligned")
            groups = np.asarray([selected[index].session_id for index in split["test"]], dtype=str)
            difference = float(np.mean((y_base - p_base) ** 2) - np.mean((y_candidate - p_candidate) ** 2))
            model_results["paired_primary_contrast"] = {
                "strongest_eligible_baseline_selected_on": "development validation MSE",
                "baseline_model": baseline,
                "candidate_model": candidate,
                "loss": "squared_error",
                "loss_improvement_baseline_minus_candidate": difference,
                "scale_normalized_skill": difference / max(float(np.mean((y_base - np.mean(y_base)) ** 2)), 1e-12),
                "confidence_interval_95": _bootstrap_loss_difference(y_base, p_base, p_candidate, groups, seed=7301 + horizon),
                "confidence_interval_unit": "session_id; one forecast origin per trajectory",
            }
            target_report[str(horizon)] = {
                "validation_mse": validation,
                "selection": {"baseline": baseline, "candidate": candidate},
                "test_models": model_results,
            }
        for model_name in _MODEL_NAMES:
            model_by_horizon = {
                horizon: _fit_model(model_name, target, horizon, selected, split["train"], config)
                for horizon in horizons
            }
            _predictions, replay = _replay_predictions(
                model_by_horizon,
                selected,
                split["test"],
                horizons,
                history_length,
            )
            selected_streaming[model_name] = replay
        target_report["streaming_replay"] = selected_streaming
        dev_report[target] = target_report

        if confirmation_records is not None:
            confirm_selected = [item for item in confirmation_records if item.family == family and item.measurement_condition == condition]
            if confirm_selected:
                train_indices = split["train"] + split["validation"]
                confirm_target: dict[str, Any] = {}
                for horizon in horizons:
                    # Fit only on development data. Confirmation records are
                    # passed to prediction/evaluation but never to fit or scale.
                    fits = {name: _fit_model(name, target, horizon, selected, train_indices, config) for name in _MODEL_NAMES}
                    baseline = dev_report[target][str(horizon)]["selection"]["baseline"]
                    candidate = dev_report[target][str(horizon)]["selection"]["candidate"]
                    predictions: dict[str, np.ndarray] = {}
                    values: np.ndarray | None = None
                    for name in _MODEL_NAMES:
                        metrics, y, p = _evaluate_model(fits[name], confirm_selected, list(range(len(confirm_selected))), target, horizon, int(campaign["future_block_length"]), history_length)
                        predictions[name] = p
                        values = y if values is None else values
                        confirm_target.setdefault("models", {})[name] = metrics
                    assert values is not None
                    paired = float(np.mean((values - predictions[baseline]) ** 2) - np.mean((values - predictions[candidate]) ** 2))
                    groups = np.asarray([item.session_id for item in confirm_selected], dtype=str)
                    confirm_target["paired_primary_contrast"] = {
                        "baseline_model": baseline,
                        "candidate_model": candidate,
                        "loss_improvement_baseline_minus_candidate": paired,
                        "scale_normalized_skill": paired / max(float(np.mean((values - np.mean(values)) ** 2)), 1e-12),
                        "confidence_interval_95": _bootstrap_loss_difference(values, predictions[baseline], predictions[candidate], groups, seed=19001 + horizon),
                        "confidence_interval_unit": "session_id; confirmation sessions are not used for model selection or preprocessing",
                    }
                    confirm_target[str(horizon)] = confirm_target.pop("paired_primary_contrast")
                    # Keep per-horizon model records under a stable key.
                    confirm_target.setdefault("horizon_models", {})[str(horizon)] = confirm_target.pop("models")
                selected_streaming: dict[str, Any] = {}
                for model_name in _MODEL_NAMES:
                    model_by_horizon = {
                        horizon: _fit_model(model_name, target, horizon, selected, train_indices, config)
                        for horizon in horizons
                    }
                    _predictions, replay = _replay_predictions(
                        model_by_horizon,
                        confirm_selected,
                        list(range(len(confirm_selected))),
                        horizons,
                        history_length,
                    )
                    selected_streaming[model_name] = replay
                confirm_target["streaming_replay"] = selected_streaming
                confirmation_report[target] = confirm_target
    primary = campaign.get("primary", {})
    primary_target = str(primary.get("target", "future_block_mean"))
    primary_horizon = str(int(primary.get("horizon", max(horizons))))
    primary_dev = dev_report[primary_target][primary_horizon]
    primary_confirmation = confirmation_report.get(primary_target, {})
    return {
        "status": "evaluated",
        "family": family,
        "measurement_condition": condition,
        "trajectory_count": len(selected),
        "session_count": len({item.session_id for item in selected}),
        "split": {key: value for key, value in split.items() if key != "session_ids"},
        "session_split_order": split["session_ids"],
        "targets": dev_report,
        "confirmation": primary_confirmation,
        "primary": {
            "target": primary_target,
            "horizon": int(primary_horizon),
            "development_selection": primary_dev["selection"],
            "development_paired_contrast": primary_dev["test_models"]["paired_primary_contrast"],
            "confirmation": primary_confirmation.get(primary_horizon, "unavailable"),
        },
        "workload_history_interpretation": {
            "definition": "compare workload_history with arx_history on the same held-out rows; no claim that workload is an unknown future input because quiet policy is declared",
            "reported": True,
        },
        "streaming_contract": {"latest_required_observation_time": "origin inclusive for every feature", "target_minimum_time": "origin + horizon; block targets extend later", "future_policy": "quiet", "teacher_forcing": False},
    }


def analyze_controlled_forecast(
    development: str | Path, config: Mapping[str, Any], output: str | Path, *, confirmation: str | Path | None = None
) -> dict[str, Any]:
    """Fit/select on development data and evaluate untouched confirmation data."""

    config = validate_controlled_forecast_config(config)
    dev_records = load_controlled_forecast_acquisition(development)
    confirm_records = load_controlled_forecast_acquisition(confirmation) if confirmation else None
    campaign = config["controlled_forecast"]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cells: dict[str, Any] = {}
    for family in [str(value) for value in campaign.get("excitation_families", [])]:
        for condition in [str(value) for value in campaign.get("measurement_conditions", [])]:
            cells[f"{family}:{condition}"] = _analyze_cell(
                dev_records,
                config,
                stage="development",
                family=family,
                condition=condition,
                confirmation_records=confirm_records,
            )
    primary = campaign.get("primary", {})
    primary_key = f"{primary.get('excitation_family', 'read_pressure')}:{primary.get('measurement_condition', 'cached_preloaded')}"
    report = {
        "schema": ANALYSIS_SCHEMA,
        "protocol_version": CONTROLLED_FORECAST_PROTOCOL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "configuration_hash": config_fingerprint(dict(config)),
        "development_source": str(development),
        "confirmation_source": str(confirmation) if confirmation else None,
        "development_trajectory_count": len(dev_records),
        "confirmation_trajectory_count": len(confirm_records) if confirm_records else 0,
        "confirmation_policy": "confirmation is never used for target/model/feature selection, scaling, residualization, thresholds, or state representation fitting",
        "cells": cells,
        "primary_cell": primary_key,
        "controlled_response": {key: value for key, value in cells.items() if not key.startswith("passive:")},
        "passive_forecasting": {key: value for key, value in cells.items() if key.startswith("passive:")},
        "claim_boundary": "ordinary worker-03 measurement trajectory forecasting; learned state is predictive representation only; no hidden DRAM or unique mechanism claim",
    }
    _atomic_json(root / "analysis.json", report)
    return report


def _synthetic_record(condition: str, index: int, rng: np.random.Generator) -> ForecastTrajectory:
    length = 48
    origin = 15
    workload = np.zeros(length, dtype=np.float32)
    workload[4:origin] = rng.integers(0, 2, size=origin - 4)
    if condition == "iid_quantized":
        target = rng.integers(80, 120, size=length).astype(np.float64)
    elif condition == "current_state_markov":
        target = np.zeros(length)
        target[0] = rng.normal()
        for pos in range(1, length):
            target[pos] = 0.9 * target[pos - 1] + rng.normal(scale=0.5)
        target = 100 + 5 * target
    elif condition == "partially_observed":
        latent = np.zeros(length)
        latent[0] = rng.normal()
        for pos in range(1, length):
            latent[pos] = 0.95 * latent[pos - 1] + rng.normal(scale=0.2)
        target = 100 + 8 * latent + rng.normal(scale=2.0, size=length)
    elif condition == "workload_only":
        target = 100 + np.convolve(workload, [1.0, 3.0, 1.0], mode="same") + rng.normal(scale=1.0, size=length)
    else:
        target = 80 + np.arange(length) * 0.7 + rng.normal(scale=1.5, size=length)
    target = np.maximum(target, 1).astype(np.uint64)
    reference = (target.astype(np.int64) + rng.integers(-2, 3, size=length)).astype(np.int64)
    quality = np.ones(length, dtype=np.uint8)
    return ForecastTrajectory(
        f"synthetic-{condition}-{index}", f"synthetic-session-{index:03d}", condition, "cached_preloaded",
        target, reference, np.arange(length, dtype=np.uint64), np.arange(1, length + 1, dtype=np.uint64),
        np.arange(length, dtype=np.int64), np.arange(length, dtype=np.int64), np.zeros(length, dtype=np.int64), np.zeros(length, dtype=np.int64),
        np.zeros(length, dtype=np.int64), np.zeros(length, dtype=np.int64), quality, quality, np.zeros(length, dtype=np.uint8), workload, origin,
        {"synthetic": True},
    )


def run_predictive_synthetic_validation(config: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Check false-positive and positive-control behavior on independent simulations."""

    config = validate_controlled_forecast_config(config)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cases = ["iid_quantized", "current_state_markov", "partially_observed", "workload_only", "drift_position"]
    results: dict[str, Any] = {}
    for case_index, case in enumerate(cases):
        records = [_synthetic_record(case, index, np.random.default_rng(1000 + case_index * 97 + index)) for index in range(24)]
        # Use the production evaluator with a fixed single primary cell.  The
        # case labels are retained only in this calibration artifact.
        local_config = json.loads(json.dumps(config))
        local_config["controlled_forecast"]["excitation_families"] = [case]
        local_config["controlled_forecast"]["measurement_conditions"] = ["cached_preloaded"]
        local_config["controlled_forecast"]["primary"] = {"excitation_family": case, "measurement_condition": "cached_preloaded", "target": "future_block_mean", "horizon": 4}
        # _analyze_cell accepts arbitrary family names in supplied records.
        result = _analyze_cell(records, local_config, stage="synthetic", family=case, condition="cached_preloaded")
        results[case] = {"expected": "history_helpful" if case == "partially_observed" else "history_not_required", "observed": result["primary"] if result.get("status") == "evaluated" else result}
    report = {
        "schema": "sensetrace.controlled-forecast-synthetic-validation.v1",
        "protocol_version": CONTROLLED_FORECAST_PROTOCOL_VERSION,
        "cases": results,
        "null_calibration": {"iid_quantized": "tests quantization/ties", "current_state_markov": "extra history should add little beyond current state", "workload_only": "workload history should explain response without observation-history increment", "drift_position": "position/current nuisance should explain trend", "partially_observed": "history should help when current observation is noisy"},
        "independent_simulation_seeds": "one deterministic seed per trajectory; no worker data used",
    }
    _atomic_json(root / "synthetic_validation.json", report)
    return report
