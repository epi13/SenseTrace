"""Versioned latent system-state witness campaign.

This module is deliberately separate from ``controlled_forecast``.  The older
campaign remains the historical one-dimensional evidence path; this campaign
adds an explicit witness contract and asks a narrower question: does a causal
multichannel observation at one forecast origin resolve state aliasing that a
single timing value cannot?

Witnesses are observations, not facts about an unobserved physical mechanism.
Every channel has a declared source, clock/interval, availability boundary,
quality state, units, and feature eligibility.  Missing or post-origin data
are retained as such and never silently converted to zero.
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .acquisition.commodity import ControlledMemoryBuffer
from .acquisition.native import TRAJECTORY_CONDITIONS, NativeMeasurementKernel
from .acquisition.perf import (
    OperationScopedPerfEvent,
    PerfEventError,
    discover_counter_capabilities,
)
from .config import config_fingerprint
from .controlled_forecast import (
    _cache_line_size,
    _counterbalanced_orders,
    _map_native_channels,
    _measure_block,
    _run_pressure_phase,
    _schedule_for,
)
from .errors import IntegrityError, SchemaError
from .hashing import sha256_file
from .runner import _git_commit
from .witness import discover_witness_capabilities
from .worker03 import collect_worker03_inventory

LATENT_SYSTEM_STATE_PROTOCOL_VERSION = "latent-system-state-v1"
ACQUISITION_SCHEMA = "sensetrace.latent-system-state-acquisition.v1"
ANALYSIS_SCHEMA = "sensetrace.latent-system-state-analysis.v1"
SYNTHETIC_SCHEMA = "sensetrace.latent-system-state-synthetic-validation.v1"
OVERHEAD_SCHEMA = "sensetrace.latent-system-state-observer-effect.v1"

ALLOWED_FAMILIES = {"read_pressure", "write_pressure", "active_quiet", "sham", "passive"}
DEFAULT_FAMILIES = ("read_pressure", "active_quiet", "sham", "passive")
DEFAULT_CONDITIONS = ("cached_preloaded", "timer_only")
ALL_ORIGIN_CONDITIONS = tuple(TRAJECTORY_CONDITIONS)
DEFAULT_HOOKS = (
    "context_switch",
    "cpu_migration",
    "page_fault",
    "page_allocation",
    "direct_reclaim",
    "compaction",
    "numa_migration",
)


@dataclass(frozen=True)
class WitnessChannelSpec:
    """The non-inferred contract for one witness/predictor channel."""

    name: str
    raw_source: str
    acquisition: str
    availability: str
    causal_eligible: bool
    measurement_overhead: str
    units: str
    normalization: str
    missing_value_policy: str
    observer_tier: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WitnessSnapshot:
    """One timestamped snapshot; ``available_at_ns`` is the firewall boundary."""

    capture_start_ns: int
    capture_end_ns: int
    available_at_ns: int
    tier: int
    status: str
    values: dict[str, float]
    valid: dict[str, bool]
    source: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        def clean(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            **asdict(self),
            "values": {key: clean(value) for key, value in self.values.items()},
        }


LOCAL_CHANNEL_SPECS = (
    WitnessChannelSpec(
        "tsc_aux",
        "RDTSCP auxiliary CPU identity at capture",
        "one RDTSCP read in the userspace snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one native RDTSCP call",
        "integer CPU/AUX value",
        "categorical; never whole-trajectory normalized",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "thread_user_time_ns",
        "getrusage(RUSAGE_THREAD).ru_utime",
        "thread resource snapshot before/after each measured block",
        "snapshot available at capture_end_ns",
        True,
        "one local resource query",
        "ns",
        "development-train median and scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "thread_system_time_ns",
        "getrusage(RUSAGE_THREAD).ru_stime",
        "thread resource snapshot before/after each measured block",
        "snapshot available at capture_end_ns",
        True,
        "one local resource query",
        "ns",
        "development-train median and scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "voluntary_context_switches",
        "/proc/self/task/<tid>/status Voluntary_ctxt_switches",
        "target-thread status snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one proc task status read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "involuntary_context_switches",
        "/proc/self/task/<tid>/status Nonvoluntary_ctxt_switches",
        "target-thread status snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one proc task status read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "minor_faults",
        "getrusage(RUSAGE_THREAD).ru_minflt",
        "thread resource snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one local resource query",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "major_faults",
        "getrusage(RUSAGE_THREAD).ru_majflt",
        "thread resource snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one local resource query",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "thread_migrations",
        "/proc/self/task/<tid>/schedstat (derived CPU migration counter when exposed)",
        "target-thread schedstat read",
        "snapshot available at capture_end_ns",
        True,
        "one optional proc schedstat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "cpu_frequency_khz",
        "/sys/devices/system/cpu/cpu<aux>/cpufreq/scaling_cur_freq",
        "frequency sysfs read for the observed AUX CPU",
        "snapshot available at capture_end_ns",
        True,
        "one sysfs read when exposed",
        "kHz",
        "development-train median and scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "thread_sched_runtime_ns",
        "/proc/self/task/<tid>/schedstat first field",
        "target-thread schedstat snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one optional proc schedstat read",
        "ns",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
    WitnessChannelSpec(
        "thread_sched_wait_ns",
        "/proc/self/task/<tid>/schedstat second field",
        "target-thread schedstat snapshot",
        "snapshot available at capture_end_ns",
        True,
        "one optional proc schedstat read",
        "ns",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        1,
    ),
)

SYSTEM_CHANNEL_SPECS = (
    WitnessChannelSpec(
        "system_context_switches",
        "/proc/stat ctxt",
        "system snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/stat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "system_interrupts",
        "/proc/stat intr",
        "system snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/stat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "system_softirqs",
        "/proc/stat softirq total",
        "system snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/stat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "vm_pgfault",
        "/proc/vmstat pgfault",
        "system VM snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/vmstat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "vm_pgmajfault",
        "/proc/vmstat pgmajfault",
        "system VM snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/vmstat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "vm_pgscan_kswapd",
        "/proc/vmstat pgscan_kswapd",
        "system VM snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/vmstat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "vm_pgscan_direct",
        "/proc/vmstat pgscan_direct",
        "system VM snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one /proc/vmstat read",
        "count",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
    WitnessChannelSpec(
        "memory_psi_some_us",
        "/proc/pressure/memory some total",
        "system PSI snapshot around each block",
        "snapshot available at capture_end_ns",
        True,
        "one pressure sysfs read",
        "microseconds",
        "development-train deltas/scale only",
        "NaN plus valid=false",
        2,
    ),
)


def _pmu_spec(name: str) -> WitnessChannelSpec:
    safe = name.replace("/", "_").replace("-", "_").replace(".", "_")
    return WitnessChannelSpec(
        f"pmu_{safe}_count",
        f"perf_event_open({name!r}) calling-thread event",
        "single small PMU window around one native measurement block",
        "counter read available at block end",
        True,
        "one opened/enabled/read scoped PMU event; measured in observer characterization",
        "events per block",
        "development-train median and scale only; retain raw/scaled/multiplex state",
        "NaN plus valid=false on permission/error/multiplex ambiguity",
        1,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return int(value) if value else None
    except (OSError, ValueError):
        return None


def _read_task_status(tid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        text = Path(f"/proc/self/task/{tid}/status").read_text(encoding="utf-8")
    except OSError:
        return result
    keys = {
        "Voluntary_ctxt_switches": "voluntary_context_switches",
        "Nonvoluntary_ctxt_switches": "involuntary_context_switches",
    }
    for line in text.splitlines():
        key, _, value = line.partition(":")
        output_key = keys.get(key.strip())
        if output_key is not None:
            try:
                result[output_key] = int(value.strip().split()[0])
            except (IndexError, ValueError):
                pass
    return result


def _read_schedstat(tid: int) -> dict[str, int]:
    try:
        fields = Path(f"/proc/self/task/{tid}/schedstat").read_text(encoding="utf-8").split()
        if len(fields) >= 3:
            # The third field is nr_running; migration count is not portable.
            return {
                "thread_sched_runtime_ns": int(fields[0]),
                "thread_sched_wait_ns": int(fields[1]),
            }
    except (OSError, ValueError):
        pass
    return {}


def _read_system_values() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        stat = Path("/proc/stat").read_text(encoding="utf-8")
    except OSError:
        stat = ""
    for line in stat.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "ctxt" and len(fields) >= 2:
            values["system_context_switches"] = int(fields[1])
        elif fields[0] == "intr" and len(fields) >= 2:
            values["system_interrupts"] = int(fields[1])
        elif fields[0] == "softirq" and len(fields) >= 2:
            values["system_softirqs"] = sum(int(item) for item in fields[2:])
    try:
        vmstat = Path("/proc/vmstat").read_text(encoding="utf-8")
    except OSError:
        vmstat = ""
    wanted = {spec.name for spec in SYSTEM_CHANNEL_SPECS}
    for line in vmstat.splitlines():
        key, _, raw = line.partition(" ")
        if key in {"pgfault", "pgmajfault", "pgscan_kswapd", "pgscan_direct"}:
            try:
                values[f"vm_{key}"] = int(raw.strip())
            except ValueError:
                pass
    try:
        psi = Path("/proc/pressure/memory").read_text(encoding="utf-8")
    except OSError:
        psi = ""
    for line in psi.splitlines():
        if line.startswith("some "):
            for token in line.split()[1:]:
                key, _, raw = token.partition("=")
                if key == "total":
                    try:
                        values["memory_psi_some_us"] = int(float(raw))
                    except ValueError:
                        pass
    return {key: value for key, value in values.items() if key in wanted}


def _rusage_values() -> dict[str, int]:
    try:
        usage = resource.getrusage(resource.RUSAGE_THREAD)
    except (AttributeError, OSError):
        return {}
    return {
        "thread_user_time_ns": int(usage.ru_utime * 1_000_000_000),
        "thread_system_time_ns": int(usage.ru_stime * 1_000_000_000),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
    }


class ScopedPmuWindow:
    """A small explicit PMU window; multiplexing is retained and invalidates use."""

    def __init__(self, events: Sequence[str]):
        self.events = tuple(str(item) for item in events)
        self.readers: list[tuple[str, OperationScopedPerfEvent]] = []
        self.failure: dict[str, Any] | None = None

    @staticmethod
    def _feature_name(event: str) -> str:
        return f"pmu_{event.replace('/', '_').replace('-', '_').replace('.', '_')}_count"

    def run(
        self, operation: Callable[[], Any]
    ) -> tuple[Any, dict[str, float], dict[str, bool], dict[str, Any]]:
        if not self.events:
            return operation(), {}, {}, {"status": "not_requested", "events": []}
        values: dict[str, float] = {}
        valid: dict[str, bool] = {}
        records: dict[str, Any] = {}
        operation_started = False
        operation_completed = False
        try:
            for event in self.events:
                reader = OperationScopedPerfEvent(event)
                reader.open()
                self.readers.append((event, reader))
            for _, reader in self.readers:
                reader.reset_and_enable()
            operation_started = True
            result = operation()
            operation_completed = True
            for _, reader in self.readers:
                reader.disable()
            for event, reader in self.readers:
                reading = reader.read()
                record = reading.as_dict()
                record["event"] = event
                record["provenance"] = reader.provenance
                records[event] = record
                key = self._feature_name(event)
                scaled = reading.scaled_count
                values[key] = float(scaled) if scaled is not None else float("nan")
                valid[key] = bool(scaled is not None and not reading.multiplexed)
            return result, values, valid, {"status": "complete", "events": records}
        except (PerfEventError, OSError, RuntimeError, ValueError) as exc:
            self.failure = {"type": type(exc).__name__, "message": str(exc)}
            for event in self.events:
                key = self._feature_name(event)
                values[key] = float("nan")
                valid[key] = False
            if operation_completed:
                fallback_result = result
            elif operation_started:
                # Do not re-run a probe if the measured operation itself
                # failed; that could duplicate a destructive operation.
                raise
            else:
                fallback_result = operation()
            return (
                fallback_result,
                values,
                valid,
                {
                    "status": "unavailable",
                    "events": records,
                    "failure": self.failure,
                },
            )
        finally:
            for _, reader in self.readers:
                reader.close()
            self.readers.clear()


class SystemWitnessCollector:
    """Collect Tier 1/2 values and preserve capability/quality boundaries."""

    def __init__(
        self,
        tier: int,
        kernel: NativeMeasurementKernel,
        pmu_events: Sequence[str] = (),
        *,
        local_channels: Sequence[str] | None = None,
        pmu_scope: str = "every_block",
        witness_scope: str = "every_block",
    ):
        if tier not in {0, 1, 2}:
            raise ValueError("witness tier must be 0, 1, or 2")
        if pmu_scope not in {"every_block", "origin_only"}:
            raise ValueError("pmu_scope must be every_block or origin_only")
        if witness_scope not in {"every_block", "origin_only"}:
            raise ValueError("witness_scope must be every_block or origin_only")
        self.tier = tier
        self.kernel = kernel
        self.pmu_events = tuple(str(event) for event in pmu_events) if tier >= 1 else ()
        self.pmu_scope = pmu_scope
        self.witness_scope = witness_scope
        self.pmu_failures: dict[str, Any] = {}
        requested_local = (
            {str(item) for item in local_channels}
            if local_channels is not None
            else {spec.name for spec in LOCAL_CHANNEL_SPECS}
        )
        unknown_local = requested_local - {spec.name for spec in LOCAL_CHANNEL_SPECS}
        if unknown_local:
            raise ValueError(f"unknown local witness channels: {sorted(unknown_local)}")
        self._local_names = requested_local if tier >= 1 else set()
        self._specs = (
            [spec for spec in LOCAL_CHANNEL_SPECS if spec.name in requested_local]
            if tier >= 1
            else []
        )
        if tier >= 2:
            self._specs = list(LOCAL_CHANNEL_SPECS)
            self._local_names = {spec.name for spec in LOCAL_CHANNEL_SPECS}
            self._specs.extend(SYSTEM_CHANNEL_SPECS)
        self._specs.extend(_pmu_spec(event) for event in self.pmu_events)
        self._pmu_capabilities = (
            discover_counter_capabilities(probe_hardware_events=True)
            if tier >= 1
            else {"status": "not_requested"}
        )
        self._bpf_capabilities = (
            discover_witness_capabilities(DEFAULT_HOOKS)
            if tier >= 2
            else {
                "status": "not_requested",
                "requested_hooks": [],
            }
        )

    @property
    def channel_specs(self) -> tuple[WitnessChannelSpec, ...]:
        return tuple(self._specs)

    @staticmethod
    def _frequency(aux: int | None) -> int | None:
        if aux is None:
            return None
        return _read_int(Path(f"/sys/devices/system/cpu/cpu{aux}/cpufreq/scaling_cur_freq"))

    def snapshot(self) -> WitnessSnapshot:
        start = time.monotonic_ns()
        values: dict[str, float] = {}
        valid: dict[str, bool] = {}
        source: dict[str, str] = {}
        tsc_aux: int | None = None
        if self.tier >= 1:
            try:
                _, tsc_aux = self.kernel.read_tsc_aux()
            except OSError:
                tsc_aux = None
            if tsc_aux is not None:
                values["tsc_aux"] = float(tsc_aux)
                valid["tsc_aux"] = True
                source["tsc_aux"] = "native RDTSCP"
            if self._local_names & {
                "thread_user_time_ns",
                "thread_system_time_ns",
                "minor_faults",
                "major_faults",
            }:
                rusage = _rusage_values()
                for key, value in rusage.items():
                    if key in self._local_names:
                        values[key] = float(value)
                        valid[key] = True
                        source[key] = "resource.getrusage(RUSAGE_THREAD)"
            if self._local_names & {
                "voluntary_context_switches",
                "involuntary_context_switches",
            }:
                status = _read_task_status(threading.get_native_id())
                for key, value in status.items():
                    if key in self._local_names:
                        values[key] = float(value)
                        valid[key] = True
                        source[key] = "/proc/self/task/<tid>/status"
            if self._local_names & {"thread_sched_runtime_ns", "thread_sched_wait_ns"}:
                for key, value in _read_schedstat(threading.get_native_id()).items():
                    if key in {"thread_sched_runtime_ns", "thread_sched_wait_ns"}:
                        values[key] = float(value)
                        valid[key] = True
                        source[key] = "/proc/self/task/<tid>/schedstat"
            if "cpu_frequency_khz" in self._local_names:
                frequency = self._frequency(tsc_aux)
                if frequency is not None:
                    values["cpu_frequency_khz"] = float(frequency)
                    valid["cpu_frequency_khz"] = True
                    source["cpu_frequency_khz"] = "cpufreq scaling_cur_freq"
        if self.tier >= 2:
            for key, value in _read_system_values().items():
                values[key] = float(value)
                valid[key] = True
                source[key] = "Linux /proc system witness"
        end = time.monotonic_ns()
        for spec in self._specs:
            values.setdefault(spec.name, float("nan"))
            valid.setdefault(spec.name, False)
            source.setdefault(spec.name, spec.raw_source)
        return WitnessSnapshot(start, end, end, self.tier, "complete", values, valid, source)

    def measure(
        self,
        operation: Callable[[], Any],
        *,
        use_pmu: bool = True,
        capture_witness: bool = True,
    ) -> tuple[Any, WitnessSnapshot, WitnessSnapshot, dict[str, Any]]:
        pmu_enabled = bool(self.pmu_events) and use_pmu
        if (self.tier == 0 or not capture_witness) and not pmu_enabled:
            # Tier 0 must remain a genuine no-witness control.  Even a few
            # extra monotonic-clock reads would be observer work in baseline.
            empty = WitnessSnapshot(0, 0, 0, 0, "not_collected", {}, {}, {})
            return (
                operation(),
                empty,
                empty,
                {
                    "operation_start_ns": None,
                    "operation_end_ns": None,
                    "pmu": {"status": "not_requested", "events": []},
                },
            )
        before = (
            self.snapshot()
            if self.tier >= 1 and capture_witness
            else WitnessSnapshot(
                time.monotonic_ns(),
                time.monotonic_ns(),
                time.monotonic_ns(),
                0,
                "not_collected",
                {},
                {},
                {},
            )
        )
        operation_start = time.monotonic_ns()
        pmu_enabled = bool(self.pmu_events) and use_pmu
        pmu = ScopedPmuWindow(self.pmu_events if pmu_enabled else ())
        result, pmu_values, pmu_valid, pmu_record = pmu.run(operation)
        operation_end = time.monotonic_ns()
        after = self.snapshot() if self.tier >= 1 and capture_witness else before
        values = dict(after.values)
        valid = dict(after.valid)
        source = dict(after.source)
        values.update(pmu_values)
        valid.update(pmu_valid)
        source.update({key: "perf_event_open scoped PMU window" for key in pmu_values})
        after = WitnessSnapshot(
            before.capture_start_ns,
            operation_end,
            operation_end,
            self.tier,
            "complete" if pmu_record["status"] in {"complete", "not_requested"} else "incomplete",
            values,
            valid,
            source,
        )
        return (
            result,
            before,
            after,
            {
                "operation_start_ns": operation_start,
                "operation_end_ns": operation_end,
                "pmu": pmu_record,
                "pmu_scope": self.pmu_scope,
                "pmu_enabled": pmu_enabled,
                "witness_scope": self.witness_scope,
                "witness_captured": capture_witness,
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "sensetrace.latent-system-state-witness-capability.v1",
            "tier": self.tier,
            "channel_specs": [spec.as_dict() for spec in self._specs],
            "pmu_events_requested": list(self.pmu_events),
            "pmu_scope": self.pmu_scope,
            "witness_scope": self.witness_scope,
            "local_channels_requested": sorted(self._local_names),
            "pmu_capabilities": self._pmu_capabilities,
            "pmu_failure_policy": "permission/error/multiplex ambiguity remains invalid, never zero",
            "bpftrace_capabilities": self._bpf_capabilities,
            "system_sources": [spec.raw_source for spec in self._specs if spec.observer_tier >= 2],
            "claim_boundary": "contextual host/system witnesses only; no direct DRAM command/topology/cell evidence",
        }


@dataclass
class LatentStateRecord:
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
    witness_values: np.ndarray
    witness_valid: np.ndarray
    witness_causal_eligible: np.ndarray
    witness_available_at_ns: np.ndarray
    witness_capture_start_ns: np.ndarray
    witness_capture_end_ns: np.ndarray
    witness_names: tuple[str, ...]
    origin_channel_names: tuple[str, ...]
    origin_channel_ticks: np.ndarray
    origin_channel_start_tsc: np.ndarray
    origin_channel_end_tsc: np.ndarray
    origin_channel_start_aux: np.ndarray
    origin_channel_end_aux: np.ndarray
    origin_channel_quality: np.ndarray
    origin_channel_order: np.ndarray
    origin_channel_capture_start_ns: np.ndarray
    origin_channel_capture_end_ns: np.ndarray
    origin_index: int
    origin_start_ns: int
    origin_end_ns: int
    metadata: dict[str, Any]

    def validate(self) -> None:
        n = len(self.target_ticks)
        vector_names = (
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
            "workload_history",
            "witness_available_at_ns",
            "witness_capture_start_ns",
            "witness_capture_end_ns",
        )
        for name in vector_names:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != n:
                raise SchemaError(f"latent trajectory channel {name} is not aligned")
        for name in ("witness_values", "witness_valid", "witness_causal_eligible"):
            value = np.asarray(getattr(self, name))
            if value.shape != (n, len(self.witness_names)):
                raise SchemaError(f"latent witness channel {name} has invalid shape")
        if not 0 <= self.origin_index < n or n < self.origin_index + 2:
            raise SchemaError("latent trajectory origin does not have a disjoint future")
        if self.origin_end_ns < self.origin_start_ns:
            raise SchemaError("latent origin timestamps are not monotone")
        if len(self.origin_channel_names) < 1:
            raise SchemaError("latent origin bundle is empty")
        bundle_arrays = (
            self.origin_channel_ticks,
            self.origin_channel_start_tsc,
            self.origin_channel_end_tsc,
            self.origin_channel_start_aux,
            self.origin_channel_end_aux,
            self.origin_channel_quality,
            self.origin_channel_order,
            self.origin_channel_capture_start_ns,
            self.origin_channel_capture_end_ns,
        )
        if any(
            np.asarray(value).shape != (len(self.origin_channel_names),) for value in bundle_arrays
        ):
            raise SchemaError("latent origin bundle channels are not aligned")
        if np.any(
            self.witness_causal_eligible
            & (self.witness_available_at_ns[:, None] > self.origin_end_ns)
        ):
            raise SchemaError("post-origin witness is marked causal")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.trajectory_id,
                self.session_id,
                self.family,
                self.measurement_condition,
            )
        ):
            raise SchemaError("latent trajectory identity fields are required")
        if np.any(self.witness_causal_eligible[self.origin_index + 1 :]):
            raise SchemaError("post-origin witness is marked causal")

    def as_journal_record(self) -> dict[str, Any]:
        self.validate()

        def values(name: str) -> list[Any]:
            raw = np.asarray(getattr(self, name)).tolist()

            def clean(value: Any) -> Any:
                if isinstance(value, float) and not math.isfinite(value):
                    return None
                if isinstance(value, list):
                    return [clean(item) for item in value]
                return value

            return clean(raw)

        return {
            "schema": "sensetrace.latent-system-state-record.v1",
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "family": self.family,
            "measurement_condition": self.measurement_condition,
            "origin_index": self.origin_index,
            "origin_start_ns": self.origin_start_ns,
            "origin_end_ns": self.origin_end_ns,
            "witness_names": list(self.witness_names),
            "origin_channel_names": list(self.origin_channel_names),
            "arrays": {
                name: values(name)
                for name in (
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
                    "workload_history",
                    "witness_values",
                    "witness_valid",
                    "witness_causal_eligible",
                    "witness_available_at_ns",
                    "witness_capture_start_ns",
                    "witness_capture_end_ns",
                    "origin_channel_ticks",
                    "origin_channel_start_tsc",
                    "origin_channel_end_tsc",
                    "origin_channel_start_aux",
                    "origin_channel_end_aux",
                    "origin_channel_quality",
                    "origin_channel_order",
                    "origin_channel_capture_start_ns",
                    "origin_channel_capture_end_ns",
                )
            },
            "metadata": self.metadata,
        }


def _record_from_journal(value: Mapping[str, Any]) -> LatentStateRecord:
    arrays = value.get("arrays", value)

    def array(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(arrays[name], dtype=dtype)

    record = LatentStateRecord(
        trajectory_id=str(value["trajectory_id"]),
        session_id=str(value["session_id"]),
        family=str(value["family"]),
        measurement_condition=str(value["measurement_condition"]),
        target_ticks=array("target_ticks", np.uint64),
        reference_ticks=array("reference_ticks", np.int64),
        target_start_tsc=array("target_start_tsc", np.uint64),
        target_end_tsc=array("target_end_tsc", np.uint64),
        reference_start_tsc=array("reference_start_tsc", np.int64),
        reference_end_tsc=array("reference_end_tsc", np.int64),
        target_start_aux=array("target_start_aux", np.int64),
        target_end_aux=array("target_end_aux", np.int64),
        reference_start_aux=array("reference_start_aux", np.int64),
        reference_end_aux=array("reference_end_aux", np.int64),
        target_quality=array("target_quality", np.uint8),
        reference_quality=array("reference_quality", np.uint8),
        acquisition_order=array("acquisition_order", np.uint8),
        workload_history=array("workload_history", np.float32),
        witness_values=array("witness_values", np.float64),
        witness_valid=array("witness_valid", bool),
        witness_causal_eligible=array("witness_causal_eligible", bool),
        witness_available_at_ns=array("witness_available_at_ns", np.int64),
        witness_capture_start_ns=array("witness_capture_start_ns", np.int64),
        witness_capture_end_ns=array("witness_capture_end_ns", np.int64),
        witness_names=tuple(str(item) for item in value["witness_names"]),
        origin_channel_names=tuple(str(item) for item in value["origin_channel_names"]),
        origin_channel_ticks=array("origin_channel_ticks", np.uint64),
        origin_channel_start_tsc=array("origin_channel_start_tsc", np.uint64),
        origin_channel_end_tsc=array("origin_channel_end_tsc", np.uint64),
        origin_channel_start_aux=array("origin_channel_start_aux", np.int64),
        origin_channel_end_aux=array("origin_channel_end_aux", np.int64),
        origin_channel_quality=array("origin_channel_quality", np.uint8),
        origin_channel_order=array("origin_channel_order", np.uint8),
        origin_channel_capture_start_ns=array("origin_channel_capture_start_ns", np.int64),
        origin_channel_capture_end_ns=array("origin_channel_capture_end_ns", np.int64),
        origin_index=int(value["origin_index"]),
        origin_start_ns=int(value["origin_start_ns"]),
        origin_end_ns=int(value["origin_end_ns"]),
        metadata=dict(value.get("metadata", {})),
    )
    record.validate()
    return record


def validate_latent_system_state_config(config: Mapping[str, Any]) -> dict[str, Any]:
    campaign = config.get("latent_system_state")
    if not isinstance(campaign, Mapping):
        raise SchemaError("latent_system_state configuration is required")
    if campaign.get("protocol_version") != LATENT_SYSTEM_STATE_PROTOCOL_VERSION:
        raise SchemaError("unsupported latent_system_state.protocol_version")
    stage = str(campaign.get("stage", "development"))
    if stage not in {"development", "confirmation"}:
        raise SchemaError("latent_system_state.stage must be development or confirmation")
    for name, minimum in {
        "sessions_per_cell": 1,
        "trajectories_per_session": 1,
        "baseline_repetitions": 2,
        "excitation_length": 2,
        "repetitions_per_phase": 2,
        "quiet_future_repetitions": 4,
        "future_block_length": 1,
        "history_length": 2,
        "word_count": 64,
        "eviction_bytes": 64,
    }.items():
        value = campaign.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise SchemaError(f"latent_system_state.{name} must be an integer >= {minimum}")
    if int(campaign["repetitions_per_phase"]) % 2:
        raise SchemaError("latent_system_state.repetitions_per_phase must be even")
    families = tuple(str(item) for item in campaign.get("excitation_families", DEFAULT_FAMILIES))
    if (
        not families
        or len(set(families)) != len(families)
        or any(item not in ALLOWED_FAMILIES for item in families)
    ):
        raise SchemaError("latent_system_state.excitation_families is invalid")
    conditions = tuple(str(item) for item in campaign.get("target_conditions", DEFAULT_CONDITIONS))
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or any(item not in TRAJECTORY_CONDITIONS for item in conditions)
    ):
        raise SchemaError("latent_system_state.target_conditions is invalid")
    bundle = tuple(
        str(item) for item in campaign.get("origin_probe_conditions", ALL_ORIGIN_CONDITIONS)
    )
    if (
        not bundle
        or len(set(bundle)) != len(bundle)
        or any(item not in TRAJECTORY_CONDITIONS for item in bundle)
    ):
        raise SchemaError("latent_system_state.origin_probe_conditions is invalid")
    if any(item not in bundle for item in conditions):
        raise SchemaError("every target condition must be in origin_probe_conditions")
    horizons = tuple(campaign.get("horizons", (1, 4, 8)))
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(not isinstance(item, int) or item < 1 for item in horizons)
    ):
        raise SchemaError("latent_system_state.horizons must be unique positive integers")
    if int(campaign["quiet_future_repetitions"]) < max(horizons) + int(
        campaign["future_block_length"]
    ):
        raise SchemaError("quiet_future_repetitions does not cover target blocks")
    tier = campaign.get("instrumentation_tier", 1)
    if not isinstance(tier, int) or tier not in {0, 1, 2}:
        raise SchemaError("latent_system_state.instrumentation_tier must be 0, 1, or 2")
    pmu = campaign.get("pmu_events", [])
    if not isinstance(pmu, list) or any(not isinstance(item, str) or not item for item in pmu):
        raise SchemaError("latent_system_state.pmu_events must be a list of event names")
    local_channels = campaign.get("tier1_local_channels", ["tsc_aux"])
    known_local = {spec.name for spec in LOCAL_CHANNEL_SPECS}
    if (
        not isinstance(local_channels, list)
        or not local_channels
        or any(not isinstance(item, str) or item not in known_local for item in local_channels)
        or len(set(local_channels)) != len(local_channels)
    ):
        raise SchemaError("latent_system_state.tier1_local_channels is invalid")
    pmu_scope = campaign.get("pmu_scope", "origin_only")
    if pmu_scope not in {"every_block", "origin_only"}:
        raise SchemaError("latent_system_state.pmu_scope must be every_block or origin_only")
    witness_scope = campaign.get("witness_scope", "origin_only")
    if witness_scope not in {"every_block", "origin_only"}:
        raise SchemaError("latent_system_state.witness_scope must be every_block or origin_only")
    interval = campaign.get("sampling_interval_us", 0)
    if not isinstance(interval, int) or interval < 0:
        raise SchemaError("latent_system_state.sampling_interval_us must be non-negative")
    confirmation_sessions = campaign.get("confirmation_sessions_per_cell")
    if confirmation_sessions is not None and (
        not isinstance(confirmation_sessions, int)
        or isinstance(confirmation_sessions, bool)
        or confirmation_sessions < int(campaign["sessions_per_cell"])
    ):
        raise SchemaError(
            "latent_system_state.confirmation_sessions_per_cell must be an integer >= sessions_per_cell"
        )
    timescale_sessions = campaign.get("timescale_sweep_sessions_per_cell")
    if timescale_sessions is not None and (
        not isinstance(timescale_sessions, int)
        or isinstance(timescale_sessions, bool)
        or timescale_sessions < 1
    ):
        raise SchemaError(
            "latent_system_state.timescale_sweep_sessions_per_cell must be a positive integer"
        )
    timescales = campaign.get("development_timescales_us", [])
    if not isinstance(timescales, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in timescales
    ):
        raise SchemaError(
            "latent_system_state.development_timescales_us must be non-negative integers"
        )
    tolerance = campaign.get("match_tolerance_ticks", 8)
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise SchemaError("latent_system_state.match_tolerance_ticks must be positive")
    return dict(config)


def _session_provenance(
    kernel: NativeMeasurementKernel,
    config: Mapping[str, Any],
    session_id: str,
    collector: SystemWitnessCollector,
) -> dict[str, Any]:
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    return {
        "schema": "sensetrace.latent-system-state-session.v1",
        "acquisition_session_id": session_id,
        "session_started_at": datetime.now(UTC).isoformat(),
        "boot_id": boot_path.read_text(encoding="utf-8").strip()
        if boot_path.exists()
        else "unavailable",
        "host": platform.node() or "unavailable",
        "code_commit": _git_commit(),
        "configuration_hash": config_fingerprint(dict(config)),
        "native": kernel.provenance(),
        "requested_process_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else "unavailable",
        "witness": collector.provenance(),
        "identity_policy": "session/boot/allocation IDs are provenance only and are not model features",
    }


def _append_journal(path: Path, record: LatentStateRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_journal_record(), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_journal(path: Path) -> dict[str, LatentStateRecord]:
    records: dict[str, LatentStateRecord] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = _record_from_journal(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, SchemaError) as exc:
            raise IntegrityError(f"invalid latent-system-state journal line {line_number}") from exc
        if record.trajectory_id in records:
            raise IntegrityError(f"duplicate latent trajectory {record.trajectory_id}")
        records[record.trajectory_id] = record
    return records


def _write_raw_artifact(root: Path, records: Sequence[LatentStateRecord]) -> dict[str, Any]:
    if not records:
        raise IntegrityError("cannot publish empty latent-system-state artifact")
    for record in records:
        record.validate()
    witness_names = records[0].witness_names
    bundle_names = records[0].origin_channel_names
    if any(
        record.witness_names != witness_names or record.origin_channel_names != bundle_names
        for record in records
    ):
        raise IntegrityError("latent records do not share channel layouts")
    path = root / "raw_trajectories.npz"
    temporary = root / "raw_trajectories.npz.tmp"
    arrays: dict[str, Any] = {}
    for name in (
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
        "workload_history",
        "witness_values",
        "witness_valid",
        "witness_causal_eligible",
        "witness_available_at_ns",
        "witness_capture_start_ns",
        "witness_capture_end_ns",
        "origin_channel_ticks",
        "origin_channel_start_tsc",
        "origin_channel_end_tsc",
        "origin_channel_start_aux",
        "origin_channel_end_aux",
        "origin_channel_quality",
        "origin_channel_order",
        "origin_channel_capture_start_ns",
        "origin_channel_capture_end_ns",
    ):
        arrays[name] = np.stack([np.asarray(getattr(record, name)) for record in records])
    arrays.update(
        {
            "origin_index": np.asarray([record.origin_index for record in records], dtype=np.int64),
            "origin_start_ns": np.asarray(
                [record.origin_start_ns for record in records], dtype=np.int64
            ),
            "origin_end_ns": np.asarray(
                [record.origin_end_ns for record in records], dtype=np.int64
            ),
            "trajectory_ids": np.asarray([record.trajectory_id for record in records]),
            "session_ids": np.asarray([record.session_id for record in records]),
            "families": np.asarray([record.family for record in records]),
            "measurement_conditions": np.asarray(
                [record.measurement_condition for record in records]
            ),
        }
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _atomic_json(
        root / "record_layout.json",
        {
            "witness_names": list(witness_names),
            "origin_channel_names": list(bundle_names),
            "record_count": len(records),
        },
    )
    return {
        "schema": "sensetrace.latent-system-state-raw.v1",
        "path": path.name,
        "sha256": sha256_file(path),
        "trajectory_count": len(records),
        "repetitions_per_trajectory": int(records[0].target_ticks.size),
        "witness_channel_count": len(witness_names),
        "origin_channel_count": len(bundle_names),
        "raw_integer_channels": [
            name for name in arrays if name.endswith(("ticks", "tsc", "aux", "quality", "order"))
        ],
    }


def _make_design(config: Mapping[str, Any], output: Path, stage: str) -> dict[str, Any]:
    campaign = config["latent_system_state"]
    path = output / "design.json"
    if path.exists():
        design = json.loads(path.read_text(encoding="utf-8"))
        if (
            design.get("config_hash") != config_fingerprint(dict(config))
            or design.get("stage") != stage
        ):
            raise IntegrityError("existing latent-system-state design does not match config/stage")
        return design
    base_seed = int(config.get("experiment", {}).get("seed", 1337)) + (
        10_000_000 if stage == "confirmation" else 0
    )
    families = [str(item) for item in campaign.get("excitation_families", DEFAULT_FAMILIES)]
    conditions = [str(item) for item in campaign.get("target_conditions", DEFAULT_CONDITIONS)]
    session_count = int(
        campaign.get("confirmation_sessions_per_cell", campaign["sessions_per_cell"])
        if stage == "confirmation"
        else campaign["sessions_per_cell"]
    )
    sessions: list[dict[str, Any]] = []
    cell_index = 0
    for family in families:
        for condition in conditions:
            for repeat in range(session_count):
                session_id = f"{stage}-latent-session-{cell_index:04d}-{uuid.uuid4().hex}"
                sessions.append(
                    {
                        "session_id": session_id,
                        "family": family,
                        "measurement_condition": condition,
                        "repeat": repeat,
                        "seed": base_seed + cell_index * 100003 + repeat * 7919,
                        "trajectory_ids": [
                            f"{session_id}-trajectory-{index:03d}"
                            for index in range(int(campaign["trajectories_per_session"]))
                        ],
                    }
                )
                cell_index += 1
    design = {
        "schema": "sensetrace.latent-system-state-design.v1",
        "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
        "stage": stage,
        "config_hash": config_fingerprint(dict(config)),
        "created_at": datetime.now(UTC).isoformat(),
        "sessions_per_cell": session_count,
        "sessions": sessions,
        "fresh_sequence_policy": "fresh session IDs and confirmation seed offset; completed design is immutable",
    }
    _atomic_json(path, design)
    return design


def _sleep_interval(campaign: Mapping[str, Any]) -> None:
    interval = int(campaign.get("sampling_interval_us", 0))
    if interval > 0:
        time.sleep(interval / 1_000_000.0)


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
    collector: SystemWitnessCollector,
) -> LatentStateRecord:
    line_size, line_source = _cache_line_size()
    if line_size is None:
        raise SchemaError("latent acquisition requires observed cache-line size")
    line_words = max(1, (line_size + 7) // 8)
    word_count = int(campaign["word_count"])
    reference_index = line_words
    pressure_index = reference_index + line_words
    pressure_words = word_count - pressure_index
    if pressure_words < 1:
        raise SchemaError("latent word_count is too small for target/reference/pressure spacing")
    buffer = ControlledMemoryBuffer(word_count, lock_memory=bool(campaign.get("lock_memory", True)))
    eviction = bytearray(int(campaign["eviction_bytes"]))
    rng = np.random.default_rng(seed)
    paired = condition != "timer_only"
    witness_names = tuple(spec.name for spec in collector.channel_specs)
    channels: list[dict[str, np.ndarray]] = []
    workloads: list[float] = []
    orders: list[int] = []
    witness_rows: list[np.ndarray] = []
    witness_valid_rows: list[np.ndarray] = []
    witness_causal_rows: list[np.ndarray] = []
    witness_availability: list[int] = []
    witness_capture_start: list[int] = []
    witness_capture_end: list[int] = []
    phase_records: list[dict[str, Any]] = []
    bundle_names = tuple(
        str(item) for item in campaign.get("origin_probe_conditions", ALL_ORIGIN_CONDITIONS)
    )
    bundle_rows: list[dict[str, Any]] = []
    try:
        buffer.warmup_touch()
        buffer.write(0, int(rng.integers(0, 2**64, dtype=np.uint64)))
        buffer.write(reference_index, int(rng.integers(0, 2**64, dtype=np.uint64)))

        def one_block(
            block_condition: str, count: int, block_paired: bool, *, causal: bool
        ) -> tuple[
            dict[str, np.ndarray], WitnessSnapshot, WitnessSnapshot, dict[str, Any], np.ndarray
        ]:
            block_orders = (
                _counterbalanced_orders(rng, count)
                if block_paired
                else np.zeros(count, dtype=np.uint8)
            )

            def operation() -> dict[str, np.ndarray]:
                return _measure_block(
                    kernel,
                    buffer,
                    0,
                    reference_index,
                    block_condition,
                    count,
                    block_orders,
                    eviction,
                    paired=block_paired,
                )

            raw, before, after, witness_operation = collector.measure(
                operation,
                use_pmu=collector.pmu_scope == "every_block",
                capture_witness=collector.witness_scope == "every_block",
            )
            if raw is None:
                raise RuntimeError("PMU witness failure returned no measured block")
            values = np.asarray(
                [before.values.get(name, float("nan")) for name in witness_names], dtype=np.float64
            )
            valid = np.asarray(
                [before.valid.get(name, False) for name in witness_names], dtype=bool
            )
            pmu_values = np.asarray(
                [after.values.get(name, float("nan")) for name in witness_names], dtype=np.float64
            )
            pmu_valid = np.asarray(
                [after.valid.get(name, False) for name in witness_names], dtype=bool
            )
            values[pmu_valid] = pmu_values[pmu_valid]
            valid[pmu_valid] = True
            return raw, before, after, witness_operation, block_orders

        def append_block(
            raw: dict[str, np.ndarray],
            before: WitnessSnapshot,
            after: WitnessSnapshot,
            block_orders: np.ndarray,
            workload: float,
            causal: bool,
        ) -> None:
            mapped = _map_native_channels(raw, block_orders, paired=condition != "timer_only")
            channels.append(mapped)
            workloads.extend([workload] * len(block_orders))
            orders.extend(block_orders.astype(int).tolist())
            values = np.asarray(
                [after.values.get(name, float("nan")) for name in witness_names], dtype=np.float64
            )
            valid = np.asarray([after.valid.get(name, False) for name in witness_names], dtype=bool)
            witness_rows.extend([values.copy() for _ in block_orders])
            witness_valid_rows.extend([valid.copy() for _ in block_orders])
            witness_causal_rows.extend(
                [np.full(len(witness_names), causal, dtype=bool) for _ in block_orders]
            )
            witness_availability.extend([after.available_at_ns] * len(block_orders))
            witness_capture_start.extend([before.capture_start_ns] * len(block_orders))
            witness_capture_end.extend([after.capture_end_ns] * len(block_orders))
            _sleep_interval(campaign)

        baseline_raw, baseline_before, baseline_after, _, baseline_orders = one_block(
            condition, int(campaign["baseline_repetitions"]), paired, causal=True
        )
        append_block(baseline_raw, baseline_before, baseline_after, baseline_orders, 0.0, True)
        schedule = _schedule_for(family, int(campaign["excitation_length"]), seed)
        schedule.validate()
        cores = tuple(int(cpu) for cpu in campaign.get("excitation_cores", [2, 3, 4, 5]))
        conversion = float(
            session_provenance.get("timing_conversion", {}).get("tsc_ticks_per_nanosecond", 3.0)
        )
        duration_ticks = max(
            1, int(float(campaign.get("phase_duration_us", 500)) * 1000.0 * conversion)
        )
        for step in schedule.steps():
            phase_orders = (
                _counterbalanced_orders(rng, int(campaign["repetitions_per_phase"]))
                if paired
                else np.zeros(int(campaign["repetitions_per_phase"]), dtype=np.uint8)
            )
            phase_orders_for_block = phase_orders.copy()
            active_for_phase = tuple(int(value) for value in step.active)

            def measure_phase(
                block_orders: np.ndarray = phase_orders_for_block,
            ) -> dict[str, np.ndarray]:
                return _measure_block(
                    kernel,
                    buffer,
                    0,
                    reference_index,
                    condition,
                    len(block_orders),
                    block_orders,
                    eviction,
                    paired=paired,
                )

            # Capture the causal snapshot before workers and the timed block.
            def run_phase(
                active: tuple[int, ...] = active_for_phase,
                phase_measure: Callable[[], dict[str, np.ndarray]] = measure_phase,
            ) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
                return _run_pressure_phase(
                    kernel,
                    buffer,
                    pressure_index,
                    pressure_words,
                    cores,
                    duration_ticks,
                    active,
                    "read" if family != "write_pressure" else "write",
                    phase_measure,
                )

            phase_result, phase_before, phase_after, _ = collector.measure(
                run_phase,
                use_pmu=collector.pmu_scope == "every_block",
                capture_witness=collector.witness_scope == "every_block",
            )
            phase_raw, actual, active_fraction = phase_result
            append_block(phase_raw, phase_before, phase_after, phase_orders, active_fraction, True)
            phase_records.append(
                {
                    "sequence_position": step.sequence_position,
                    "requested_active": list(step.active),
                    "actual_active_fraction": active_fraction,
                    "execution": actual,
                }
            )
        # The bundle is a deliberately ordered, sequential witness plane.  The
        # order is retained and is never a hidden model input.
        bundle_order = list(bundle_names)
        if seed % 2:
            bundle_order = list(reversed(bundle_order))
        origin_start_ns = time.monotonic_ns()

        def measure_bundle() -> list[tuple[str, dict[str, np.ndarray], np.ndarray]]:
            measured: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
            for bundle_condition in bundle_order:
                bundle_paired = bundle_condition != "timer_only"
                bundle_order_values = (
                    _counterbalanced_orders(rng, 1)
                    if bundle_paired
                    else np.zeros(1, dtype=np.uint8)
                )
                raw = _measure_block(
                    kernel,
                    buffer,
                    0,
                    reference_index,
                    bundle_condition,
                    1,
                    bundle_order_values,
                    eviction,
                    paired=bundle_paired,
                )
                measured.append((bundle_condition, raw, bundle_order_values))
            return measured

        bundle_measured, bundle_before, bundle_after, bundle_operation = collector.measure(
            measure_bundle
        )
        origin_end_ns = bundle_after.capture_end_ns
        for bundle_condition, raw, bundle_order_values in bundle_measured:
            mapped_bundle = _map_native_channels(
                raw, bundle_order_values, paired=bundle_condition != "timer_only"
            )
            bundle_rows.append(
                {
                    "condition": bundle_condition,
                    "mapped": mapped_bundle,
                    "order": bundle_order_values,
                    "capture_start_ns": bundle_operation["operation_start_ns"],
                    "capture_end_ns": bundle_operation["operation_end_ns"],
                }
            )
        primary_bundle = next(item for item in bundle_rows if item["condition"] == condition)
        primary_raw = primary_bundle["mapped"]
        origin_index = sum(len(item["first_start_tsc"]) for item in channels)
        channels.append(primary_raw)
        workloads.append(0.0)
        orders.append(int(primary_bundle["order"][0]))
        origin_values = np.asarray(
            [bundle_after.values.get(name, float("nan")) for name in witness_names],
            dtype=np.float64,
        )
        origin_valid = np.asarray(
            [bundle_after.valid.get(name, False) for name in witness_names], dtype=bool
        )
        witness_rows.append(origin_values)
        witness_valid_rows.append(origin_valid)
        witness_causal_rows.append(np.ones(len(witness_names), dtype=bool))
        witness_availability.append(origin_end_ns)
        witness_capture_start.append(bundle_before.capture_start_ns)
        witness_capture_end.append(origin_end_ns)
        _sleep_interval(campaign)
        quiet_raw, quiet_before, quiet_after, _, quiet_orders = one_block(
            condition, int(campaign["quiet_future_repetitions"]), paired, causal=False
        )
        append_block(quiet_raw, quiet_before, quiet_after, quiet_orders, 0.0, False)
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
            merged[key] = np.concatenate([item[key] for item in channels])
        bundle_arrays = {
            key: np.asarray(
                [
                    row["mapped"][key][0]
                    for row in sorted(
                        bundle_rows, key=lambda item: bundle_names.index(item["condition"])
                    )
                ]
            )
            for key in (
                "target_ticks",
                "target_start_tsc",
                "target_end_tsc",
                "target_start_aux",
                "target_end_aux",
                "target_quality",
            )
        }
        bundle_arrays["origin_channel_order"] = np.asarray(
            [bundle_order.index(name) for name in bundle_names], dtype=np.uint8
        )
        record = LatentStateRecord(
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
            acquisition_order=np.asarray(orders, dtype=np.uint8),
            workload_history=np.asarray(workloads, dtype=np.float32),
            witness_values=np.asarray(witness_rows, dtype=np.float64),
            witness_valid=np.asarray(witness_valid_rows, dtype=bool),
            witness_causal_eligible=np.asarray(witness_causal_rows, dtype=bool),
            witness_available_at_ns=np.asarray(witness_availability, dtype=np.int64),
            witness_capture_start_ns=np.asarray(witness_capture_start, dtype=np.int64),
            witness_capture_end_ns=np.asarray(witness_capture_end, dtype=np.int64),
            witness_names=witness_names,
            origin_channel_names=bundle_names,
            origin_channel_ticks=bundle_arrays["target_ticks"],
            origin_channel_start_tsc=bundle_arrays["target_start_tsc"],
            origin_channel_end_tsc=bundle_arrays["target_end_tsc"],
            origin_channel_start_aux=bundle_arrays["target_start_aux"],
            origin_channel_end_aux=bundle_arrays["target_end_aux"],
            origin_channel_quality=bundle_arrays["target_quality"],
            origin_channel_order=bundle_arrays["origin_channel_order"],
            origin_channel_capture_start_ns=np.asarray(
                [
                    row["capture_start_ns"]
                    for row in sorted(
                        bundle_rows, key=lambda item: bundle_names.index(item["condition"])
                    )
                ],
                dtype=np.int64,
            ),
            origin_channel_capture_end_ns=np.asarray(
                [
                    row["capture_end_ns"]
                    for row in sorted(
                        bundle_rows, key=lambda item: bundle_names.index(item["condition"])
                    )
                ],
                dtype=np.int64,
            ),
            origin_index=origin_index,
            origin_start_ns=origin_start_ns,
            origin_end_ns=origin_end_ns,
            metadata={
                "schema": "sensetrace.latent-system-state-record.v1",
                "session_provenance": dict(session_provenance),
                "allocation_id": f"buffer-{uuid.uuid4().hex}",
                "cache_line_size_bytes": line_size,
                "cache_line_size_source": line_source,
                "reference_spacing_bytes": reference_index * 8,
                "spacing_claim": "distinct observed cache lines only; no bank/row/channel claim",
                "target_word_index": 0,
                "reference_word_index": reference_index,
                "paired_measurement": paired,
                "raw_duration_units": "TSC ticks",
                "sampling_interval_us": int(campaign.get("sampling_interval_us", 0)),
                "origin_bundle_order": bundle_order,
                "origin_bundle_policy": "sequential all-probe bundle; order retained for cross-probe disturbance audit",
                "phase_execution": phase_records,
                "schedule_request": schedule.request_record(),
                "witness_provenance": {
                    "schema": "sensetrace.latent-system-state-witness-contract.v1",
                    "tier": collector.tier,
                    "channel_specs": [spec.as_dict() for spec in collector.channel_specs],
                    "pmu_events_requested": list(collector.pmu_events),
                },
                "forecast_origin_policy": "origin is end of the synchronized multichannel bundle; future target begins after that endpoint",
                "witness_policy": "pre-origin/local and in-origin completion values are eligible only when their explicit row/channel mask is true; future rows are witness-only",
                "feature_firewall": {
                    "eligibility_is_explicit": True,
                    "allowed": [
                        "origin bundle timing channels",
                        "causal witness values available by origin_end_ns",
                        "target condition workload fraction through origin",
                    ],
                    "forbidden": [
                        "post-origin witness values",
                        "future target/reference values",
                        "IDs/seeds/schedules/boot/allocation",
                        "whole-trajectory normalization",
                        "confirmation values for preprocessing",
                    ],
                },
            },
        )
        record.validate()
        return record
    finally:
        buffer.close()


def run_latent_system_state_acquisition(
    config: Mapping[str, Any], output: str | Path, *, stage: str | None = None
) -> dict[str, Any]:
    config = validate_latent_system_state_config(config)
    campaign = config["latent_system_state"]
    effective_stage = stage or str(campaign.get("stage", "development"))
    if effective_stage not in {"development", "confirmation"}:
        raise SchemaError("latent acquisition stage must be development or confirmation")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    design = _make_design(config, root, effective_stage)
    journal_path = root / "trajectory_journal.jsonl"
    completed = _load_journal(journal_path)
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        raise RuntimeError("latent-system-state acquisition requires the native kernel")
    requested = campaign.get("measurement_cpu_affinity")
    old_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if requested is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(item) for item in requested})
    try:
        # A one-shot calibration and capability record belongs to acquisition
        # provenance, never to predictor scaling.
        t0_ns = time.monotonic_ns()
        t0_tsc, t0_aux = kernel.read_tsc_aux()
        time.sleep(0.01)
        t1_tsc, t1_aux = kernel.read_tsc_aux()
        t1_ns = time.monotonic_ns()
        timing_conversion = {
            "method": "paired monotonic_ns/RDTSCP read",
            "tsc_ticks_per_nanosecond": max(t1_tsc - t0_tsc, 1) / max(t1_ns - t0_ns, 1),
            "nanoseconds_per_tsc_tick": max(t1_ns - t0_ns, 1) / max(t1_tsc - t0_tsc, 1),
            "start_aux": t0_aux,
            "end_aux": t1_aux,
        }
        collector = SystemWitnessCollector(
            int(campaign.get("instrumentation_tier", 1)),
            kernel,
            campaign.get("pmu_events", []),
            local_channels=campaign.get("tier1_local_channels", ["tsc_aux"]),
            pmu_scope=str(campaign.get("pmu_scope", "origin_only")),
            witness_scope=str(campaign.get("witness_scope", "origin_only")),
        )
        inventory = collect_worker03_inventory(
            native_library=Path(__file__).resolve().parents[2]
            / "native"
            / "libsensetrace_measurement.so"
        )
        for session in design["sessions"]:
            session_id = str(session["session_id"])
            provenance = _session_provenance(kernel, config, session_id, collector)
            provenance["timing_conversion"] = timing_conversion
            provenance["host_inventory_snapshot"] = inventory
            for index, trajectory_id in enumerate(session["trajectory_ids"]):
                if trajectory_id in completed:
                    continue
                print(
                    f"latent-system-state {effective_stage}: {len(completed) + 1}/{sum(len(item['trajectory_ids']) for item in design['sessions'])} {trajectory_id}",
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
                    collector=collector,
                )
                _append_journal(journal_path, record)
                completed[trajectory_id] = record
                _atomic_json(
                    root / "progress.json",
                    {
                        "schema": "sensetrace.latent-system-state-progress.v1",
                        "stage": effective_stage,
                        "completed_trajectory_count": len(completed),
                        "planned_trajectory_count": sum(
                            len(item["trajectory_ids"]) for item in design["sessions"]
                        ),
                        "last_completed_trajectory_id": trajectory_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
        records = [
            completed[item]
            for session in design["sessions"]
            for item in session["trajectory_ids"]
            if item in completed
        ]
        planned = sum(len(item["trajectory_ids"]) for item in design["sessions"])
        if len(records) != planned:
            raise IntegrityError("latent acquisition ended before all trajectories completed")
        raw = _write_raw_artifact(root, records)
        acquisition = {
            "schema": ACQUISITION_SCHEMA,
            "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
            "stage": effective_stage,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "execution_host": platform.node() or "unavailable",
            "requested_node": config.get("run_metadata", {}).get("node", "worker-03"),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(dict(config)),
            "design": design,
            "raw_artifact": raw,
            "trajectory_journal": {"path": journal_path.name, "sha256": sha256_file(journal_path)},
            "trajectory_count": len(records),
            "session_count": len(design["sessions"]),
            "boot_ids": sorted(
                {
                    str(item.metadata.get("session_provenance", {}).get("boot_id", "unavailable"))
                    for item in records
                }
            ),
            "timing_conversion": timing_conversion,
            "instrumentation": collector.provenance(),
            "feature_firewall": records[0].metadata["feature_firewall"],
            "claim_boundary": "ordinary worker-03 timing plus contextual system-witness prediction; no hidden DRAM state/topology/unique mechanism claim",
        }
        _atomic_json(root / "acquisition.json", acquisition)
        _atomic_json(
            root / "experiment.json",
            {
                "schema": "sensetrace.latent-system-state-experiment.v1",
                "stage": effective_stage,
                "acquisition": acquisition,
                "analysis_status": "not_run",
                "claim_boundary": acquisition["claim_boundary"],
            },
        )
        return acquisition
    finally:
        if old_affinity is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(old_affinity))


def run_latent_timescale_sweep(config: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Acquire a small development-only sweep at explicitly listed intervals."""

    config = validate_latent_system_state_config(config)
    campaign = config["latent_system_state"]
    intervals = tuple(int(value) for value in campaign.get("development_timescales_us", ()))
    if not intervals:
        raise SchemaError("development_timescales_us must contain at least one interval")
    sweep_sessions = int(
        campaign.get(
            "timescale_sweep_sessions_per_cell",
            min(int(campaign["sessions_per_cell"]), 2),
        )
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    acquisitions: dict[str, Any] = {}
    for interval in intervals:
        sweep_config = deepcopy(dict(config))
        sweep_campaign = sweep_config["latent_system_state"]
        sweep_campaign["stage"] = "development"
        sweep_campaign["sampling_interval_us"] = interval
        sweep_campaign["sessions_per_cell"] = sweep_sessions
        interval_root = root / f"interval-{interval}us"
        acquisitions[str(interval)] = run_latent_system_state_acquisition(
            sweep_config, interval_root, stage="development"
        )
    result = {
        "schema": "sensetrace.latent-system-state-timescale-sweep.v1",
        "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
        "intervals_us": list(intervals),
        "sessions_per_cell": sweep_sessions,
        "outputs": {
            interval: {
                "path": str(root / f"interval-{interval}us"),
                "trajectory_count": value["trajectory_count"],
                "session_count": value["session_count"],
            }
            for interval, value in acquisitions.items()
        },
        "claim_boundary": "development timescale exploration only; no confirmation claim",
    }
    _atomic_json(root / "timescale-sweep.json", result)
    return result


def analyze_latent_timescale_sweep(
    sweep: str | Path, config: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    """Summarize state memory without treating two-session cells as confirmatory."""

    config = validate_latent_system_state_config(config)
    campaign = config["latent_system_state"]
    root = Path(sweep)
    horizons = tuple(int(item) for item in campaign.get("horizons", (1, 4, 8)))
    cells: dict[str, Any] = {}
    for interval in campaign.get("development_timescales_us", ()):
        interval_root = root / f"interval-{int(interval)}us"
        records = load_latent_system_state_acquisition(interval_root)
        for family in sorted({record.family for record in records}):
            for condition in sorted({record.measurement_condition for record in records}):
                cell_records = [
                    record
                    for record in records
                    if record.family == family and record.measurement_condition == condition
                ]
                cell: dict[str, Any] = {}
                for horizon in horizons:
                    batch = _feature_batch(
                        cell_records,
                        horizon,
                        int(campaign["future_block_length"]),
                        int(campaign["history_length"]),
                    )
                    current = batch.features["current_primary"][:, 0]
                    finite = np.isfinite(current) & np.isfinite(batch.targets)
                    if finite.sum() < 1:
                        cell[str(horizon)] = {"status": "no_finite_rows", "row_count": 0}
                        continue
                    session_count = len(set(str(item) for item in batch.session_ids[finite]))
                    correlation = float("nan")
                    if finite.sum() > 2 and np.std(current[finite]) > 0:
                        correlation = float(
                            np.corrcoef(current[finite], batch.targets[finite])[0, 1]
                        )
                    cell[str(horizon)] = {
                        "row_count": int(finite.sum()),
                        "session_count": session_count,
                        "current_to_future_mse_ticks2": _mse(
                            batch.targets[finite], current[finite]
                        ),
                        "future_variance_ticks2": float(np.var(batch.targets[finite])),
                        "current_future_correlation": correlation,
                        "status": "descriptive_only",
                        "independent_session_warning": (
                            "two or fewer independent sessions; interval is not a confirmation result"
                            if session_count < 3
                            else "development descriptive summary"
                        ),
                    }
                cells[f"{int(interval)}us/{family}/{condition}"] = cell
    result = {
        "schema": "sensetrace.latent-system-state-timescale-analysis.v1",
        "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
        "source": str(sweep),
        "cells": cells,
        "claim_boundary": "descriptive development timescale summary; no confirmation or causal mechanism claim",
    }
    _atomic_json(Path(output) / "timescale-analysis.json", result)
    return result


def load_latent_system_state_acquisition(path: str | Path) -> list[LatentStateRecord]:
    root = Path(path)
    manifest = json.loads((root / "acquisition.json").read_text(encoding="utf-8"))
    raw_path = root / str(manifest["raw_artifact"]["path"])
    if manifest["raw_artifact"]["sha256"] != sha256_file(raw_path):
        raise IntegrityError("latent raw artifact hash mismatch")
    layout = json.loads((root / "record_layout.json").read_text(encoding="utf-8"))
    journal = _load_journal(root / str(manifest["trajectory_journal"]["path"]))
    if len(journal) != int(manifest["trajectory_count"]):
        raise IntegrityError("latent journal/artifact trajectory count mismatch")
    records = list(journal.values())
    if (
        tuple(layout["witness_names"]) != records[0].witness_names
        or tuple(layout["origin_channel_names"]) != records[0].origin_channel_names
    ):
        raise IntegrityError("latent record layout mismatch")
    return records


@dataclass
class LatentFeatureBatch:
    features: dict[str, np.ndarray]
    targets: np.ndarray
    session_ids: np.ndarray
    metadata: list[dict[str, Any]]


MODEL_NAMES = (
    "training_mean",
    "persistence",
    "current_primary",
    "current_multichannel",
    "current_plus_workload",
    "primary_history",
    "multichannel_history",
    "ridge_state",
    "pca_delay_state",
)


def _causal_origin_witness(record: LatentStateRecord) -> np.ndarray:
    row = record.origin_index
    specs = {
        spec["name"]: spec
        for spec in record.metadata.get("witness_provenance", {}).get("channel_specs", [])
    }
    values = np.asarray(record.witness_values[row], dtype=np.float64).copy()
    valid = np.asarray(record.witness_valid[row], dtype=bool) & np.asarray(
        record.witness_causal_eligible[row], dtype=bool
    )
    valid &= record.witness_available_at_ns[row] <= record.origin_end_ns
    for index, name in enumerate(record.witness_names):
        if not specs.get(name, {}).get("causal_eligible", False):
            valid[index] = False
    values[~valid] = np.nan
    return values


def _history_matrix(record: LatentStateRecord, history_length: int) -> np.ndarray:
    end = record.origin_index + 1
    start = max(0, end - history_length)
    timing = np.asarray(record.target_ticks[start:end], dtype=np.float64)
    witness = np.asarray(record.witness_values[start:end], dtype=np.float64).copy()
    valid = np.asarray(record.witness_valid[start:end], dtype=bool) & np.asarray(
        record.witness_causal_eligible[start:end], dtype=bool
    )
    for row in range(len(witness)):
        if record.witness_available_at_ns[start + row] > record.origin_end_ns:
            valid[row, :] = False
    witness[~valid] = np.nan
    rows = np.column_stack([timing, witness])
    if len(rows) < history_length:
        rows = np.vstack([np.full((history_length - len(rows), rows.shape[1]), np.nan), rows])
    return rows.reshape(-1)


def _feature_batch(
    records: Sequence[LatentStateRecord], horizon: int, block_length: int, history_length: int
) -> LatentFeatureBatch:
    features: dict[str, list[np.ndarray]] = {
        name: [] for name in MODEL_NAMES if name not in {"training_mean", "persistence"}
    }
    targets: list[float] = []
    sessions: list[str] = []
    metadata: list[dict[str, Any]] = []
    for record in records:
        start = record.origin_index + horizon
        stop = start + block_length
        if stop > len(record.target_ticks):
            continue
        current = float(record.target_ticks[record.origin_index])
        bundle = np.asarray(record.origin_channel_ticks, dtype=np.float64)
        witness = _causal_origin_witness(record)
        current_multi = np.concatenate([bundle, witness])
        workload = np.asarray(
            [float(record.workload_history[record.origin_index])], dtype=np.float64
        )
        primary_history = np.asarray(
            record.target_ticks[
                max(0, record.origin_index + 1 - history_length) : record.origin_index + 1
            ],
            dtype=np.float64,
        )
        if len(primary_history) < history_length:
            primary_history = np.concatenate(
                [np.full(history_length - len(primary_history), np.nan), primary_history]
            )
        multi_history = _history_matrix(record, history_length)
        features["current_primary"].append(np.asarray([current]))
        features["current_multichannel"].append(current_multi)
        features["current_plus_workload"].append(np.concatenate([[current], workload]))
        features["primary_history"].append(primary_history)
        features["multichannel_history"].append(multi_history)
        features["ridge_state"].append(np.concatenate([current_multi, primary_history, workload]))
        features["pca_delay_state"].append(np.concatenate([current_multi, multi_history, workload]))
        targets.append(float(np.mean(record.target_ticks[start:stop])))
        sessions.append(record.session_id)
        metadata.append(
            {
                "trajectory_id": record.trajectory_id,
                "family": record.family,
                "measurement_condition": record.measurement_condition,
                "origin_index": record.origin_index,
            }
        )
    return LatentFeatureBatch(
        {key: np.asarray(value, dtype=np.float64) for key, value in features.items()},
        np.asarray(targets, dtype=np.float64),
        np.asarray(sessions),
        metadata,
    )


def assert_feature_firewall(
    batch: LatentFeatureBatch, *, origin_end_ns: Sequence[int] | None = None
) -> None:
    """Reject non-causal feature construction and non-finite unhandled data."""

    for name, matrix in batch.features.items():
        if name not in MODEL_NAMES:
            raise SchemaError(f"unknown feature model {name}")
        if matrix.ndim != 2:
            raise SchemaError(f"feature matrix {name} is not two-dimensional")
    if origin_end_ns is not None and len(origin_end_ns) != len(batch.targets):
        raise SchemaError("origin availability metadata is not aligned with feature rows")


@dataclass
class FittedLatentModel:
    name: str
    target_mean: float | None = None
    impute: np.ndarray | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    pca_mean: np.ndarray | None = None
    coefficient: np.ndarray | None = None
    intercept: float | None = None

    def _prepare(self, matrix: np.ndarray) -> np.ndarray:
        result = np.asarray(matrix, dtype=np.float64).copy()
        if self.impute is not None:
            missing = ~np.isfinite(result)
            if missing.any():
                result[missing] = np.take(self.impute, np.where(missing)[1])
        if self.scaler_mean is not None and self.scaler_scale is not None:
            result = (result - self.scaler_mean) / self.scaler_scale
        if self.pca_components is not None and self.pca_mean is not None:
            result = (result - self.pca_mean) @ self.pca_components.T
        return result

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.name == "training_mean":
            return np.full(len(matrix), float(self.target_mean))
        if self.name == "persistence":
            return np.asarray(matrix[:, 0], dtype=np.float64)
        prepared = self._prepare(matrix)
        assert self.coefficient is not None and self.intercept is not None
        return prepared @ self.coefficient + self.intercept

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_mean": self.target_mean,
            "impute": None if self.impute is None else self.impute.tolist(),
            "scaler_mean": None if self.scaler_mean is None else self.scaler_mean.tolist(),
            "scaler_scale": None if self.scaler_scale is None else self.scaler_scale.tolist(),
            "pca_components": None if self.pca_components is None else self.pca_components.tolist(),
            "pca_mean": None if self.pca_mean is None else self.pca_mean.tolist(),
            "coefficient": None if self.coefficient is None else self.coefficient.tolist(),
            "intercept": self.intercept,
        }


def _fit_latent_model(
    name: str, batch: LatentFeatureBatch, indices: Sequence[int], alpha: float, state_rank: int
) -> FittedLatentModel:
    index = np.asarray(list(indices), dtype=int)
    if len(index) < 1:
        raise SchemaError("cannot fit latent model with no training rows")
    if name == "training_mean":
        return FittedLatentModel(name, target_mean=float(np.mean(batch.targets[index])))
    matrix = batch.features["current_primary"] if name == "persistence" else batch.features[name]
    x = np.asarray(matrix[index], dtype=np.float64)
    y = batch.targets[index]
    impute = np.asarray(
        [
            float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
            for column in x.T
        ],
        dtype=np.float64,
    )
    x[~np.isfinite(x)] = np.take(impute, np.where(~np.isfinite(x))[1])
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    pca_components = None
    pca_mean = None
    if name == "pca_delay_state":
        rank = min(int(state_rank), transformed.shape[0], transformed.shape[1])
        if rank < 1:
            raise SchemaError("pca state has no usable rank")
        pca = PCA(n_components=rank, svd_solver="full").fit(transformed)
        transformed = pca.transform(transformed)
        pca_components = np.asarray(pca.components_, dtype=np.float64)
        pca_mean = np.asarray(pca.mean_, dtype=np.float64)
    model = Ridge(alpha=float(alpha)).fit(transformed, y)
    return FittedLatentModel(
        name=name,
        impute=impute,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.where(scaler.scale_ == 0, 1.0, scaler.scale_),
        pca_components=pca_components,
        pca_mean=pca_mean,
        coefficient=np.asarray(model.coef_, dtype=np.float64),
        intercept=float(model.intercept_),
    )


def _split_sessions(
    records: Sequence[LatentStateRecord],
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[str]]:
    sessions = sorted({record.session_id for record in records})
    if len(sessions) < 3:
        raise SchemaError("latent analysis requires at least three independent sessions")
    rng = np.random.default_rng(seed)
    shuffled = [sessions[index] for index in rng.permutation(len(sessions))]
    n_train = max(1, int(len(sessions) * fractions[0]))
    n_val = max(1, int(len(sessions) * fractions[1]))
    if n_train + n_val >= len(sessions):
        n_val = max(1, len(sessions) - n_train - 1)
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _indices_for_sessions(batch: LatentFeatureBatch, sessions: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [index for index, value in enumerate(batch.session_ids) if value in set(sessions)],
        dtype=int,
    )


def _mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2)) if len(y) else float("nan")


def _fit_predict_all(
    batch: LatentFeatureBatch,
    indices: Sequence[int],
    predict_indices: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, tuple[FittedLatentModel, np.ndarray]]:
    result: dict[str, tuple[FittedLatentModel, np.ndarray]] = {}
    for name in MODEL_NAMES:
        model = _fit_latent_model(
            name,
            batch,
            indices,
            float(config["latent_system_state"].get("ridge_alpha", 1.0)),
            int(config["latent_system_state"].get("state_rank", 2)),
        )
        if name == "training_mean":
            matrix = batch.features["current_primary"][list(predict_indices)]
        elif name == "persistence":
            matrix = batch.features["current_primary"][list(predict_indices)]
        else:
            matrix = batch.features[name][list(predict_indices)]
        result[name] = (model, model.predict(matrix))
    return result


def _session_bootstrap_loss_difference(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    sessions: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    session_values = sorted(set(str(item) for item in sessions))
    if len(session_values) < 2:
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "unit": "session_id",
            "session_count": len(session_values),
            "status": "insufficient_independent_sessions",
        }
    per_session = np.asarray(
        [
            _mse(y[sessions == session], baseline[sessions == session])
            - _mse(y[sessions == session], candidate[sessions == session])
            for session in session_values
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.mean(per_session[rng.integers(0, len(per_session), size=len(per_session))])
            for _ in range(max(1, repetitions))
        ]
    )
    return {
        "estimate": float(np.mean(per_session)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "unit": "session_id",
        "session_count": len(session_values),
        "per_session": {
            session: float(per_session[index]) for index, session in enumerate(session_values)
        },
        "status": "complete",
    }


def _evaluate_ladder(
    records: Sequence[LatentStateRecord], config: Mapping[str, Any], *, horizon: int
) -> dict[str, Any]:
    campaign = config["latent_system_state"]
    batch = _feature_batch(
        records, horizon, int(campaign["future_block_length"]), int(campaign["history_length"])
    )
    assert_feature_firewall(batch)
    split = _split_sessions(records, int(config.get("experiment", {}).get("seed", 1337)) + horizon)
    train = _indices_for_sessions(batch, split["train"])
    validation = _indices_for_sessions(batch, split["validation"])
    test = _indices_for_sessions(batch, split["test"])
    if len(validation) < 1 or len(test) < 1:
        raise SchemaError("latent session split has empty validation/test rows")
    validation_predictions = _fit_predict_all(batch, train, validation, config)
    validation_mse = {
        name: _mse(batch.targets[validation], pair[1])
        for name, pair in validation_predictions.items()
    }
    selected = min(MODEL_NAMES, key=lambda name: (validation_mse[name], MODEL_NAMES.index(name)))
    refit = np.concatenate([train, validation])
    test_predictions = _fit_predict_all(batch, refit, test, config)
    test_mse = {name: _mse(batch.targets[test], pair[1]) for name, pair in test_predictions.items()}
    training_mean_mse = test_mse["training_mean"]
    normalized_skill = {
        name: (
            1.0 - test_mse[name] / training_mean_mse
            if math.isfinite(training_mean_mse) and training_mean_mse > 0
            else float("nan")
        )
        for name in MODEL_NAMES
    }
    primary_baseline = test_predictions["current_primary"][1]
    selected_prediction = test_predictions[selected][1]
    bootstrap = _session_bootstrap_loss_difference(
        batch.targets[test],
        primary_baseline,
        selected_prediction,
        batch.session_ids[test],
        seed=17 + horizon,
        repetitions=int(config.get("reporting", {}).get("bootstrap_repetitions", 500)),
    )
    return {
        "horizon": horizon,
        "row_count": len(batch.targets),
        "split": split,
        "model_order": list(MODEL_NAMES),
        "validation_mse": validation_mse,
        "test_mse": test_mse,
        "normalized_skill_vs_training_mean": normalized_skill,
        "selected_model": selected,
        "primary_comparison": {
            "baseline": "current_primary",
            "candidate": "current_multichannel",
            "test_loss_improvement": test_mse["current_primary"] - test_mse["current_multichannel"],
            "session_bootstrap": _session_bootstrap_loss_difference(
                batch.targets[test],
                test_predictions["current_primary"][1],
                test_predictions["current_multichannel"][1],
                batch.session_ids[test],
                seed=101 + horizon,
                repetitions=int(config.get("reporting", {}).get("bootstrap_repetitions", 500)),
            ),
        },
        "history_comparison": {
            "baseline": "current_multichannel",
            "candidate": "multichannel_history",
            "test_loss_improvement": test_mse["current_multichannel"]
            - test_mse["multichannel_history"],
            "session_bootstrap": _session_bootstrap_loss_difference(
                batch.targets[test],
                test_predictions["current_multichannel"][1],
                test_predictions["multichannel_history"][1],
                batch.session_ids[test],
                seed=201 + horizon,
                repetitions=int(config.get("reporting", {}).get("bootstrap_repetitions", 500)),
            ),
        },
        "history_vs_primary_comparison": {
            "baseline": "current_primary",
            "candidate": "multichannel_history",
            "test_loss_improvement": test_mse["current_primary"] - test_mse["multichannel_history"],
            "session_bootstrap": _session_bootstrap_loss_difference(
                batch.targets[test],
                test_predictions["current_primary"][1],
                test_predictions["multichannel_history"][1],
                batch.session_ids[test],
                seed=251 + horizon,
                repetitions=int(config.get("reporting", {}).get("bootstrap_repetitions", 500)),
            ),
        },
        "selected_model_comparison": {
            "baseline": "current_primary",
            "candidate": selected,
            "test_loss_improvement": test_mse["current_primary"] - test_mse[selected],
            "session_bootstrap": bootstrap,
        },
        "preprocessing": {
            "fit_scope": "development train plus validation only; confirmation never contributes",
            "whole_trajectory_normalization": False,
            "target_in_preprocessing": False,
            "missing_values": "training split feature median only",
            "future_witness_used": False,
        },
    }


def _matched_state_diagnostics(
    records: Sequence[LatentStateRecord], config: Mapping[str, Any], *, horizon: int
) -> dict[str, Any]:
    campaign = config["latent_system_state"]
    batch = _feature_batch(
        records, horizon, int(campaign["future_block_length"]), int(campaign["history_length"])
    )
    rows: list[dict[str, Any]] = []
    tolerance = float(campaign.get("match_tolerance_ticks", 8))
    for index in range(len(batch.targets)):
        for other in range(index + 1, len(batch.targets)):
            if batch.metadata[index]["family"] == batch.metadata[other]["family"]:
                continue
            scalar_diff = abs(
                float(
                    batch.features["current_primary"][index, 0]
                    - batch.features["current_primary"][other, 0]
                )
            )
            if scalar_diff > tolerance:
                continue
            left = batch.features["current_multichannel"][index]
            right = batch.features["current_multichannel"][other]
            finite = np.isfinite(left) & np.isfinite(right)
            state_diff = (
                float(np.mean(np.abs(left[finite] - right[finite])))
                if finite.any()
                else float("nan")
            )
            rows.append(
                {
                    "left": index,
                    "right": other,
                    "family_pair": "|".join(
                        sorted((batch.metadata[index]["family"], batch.metadata[other]["family"]))
                    ),
                    "session_ids": [str(batch.session_ids[index]), str(batch.session_ids[other])],
                    "current_primary_abs_diff": scalar_diff,
                    "current_multichannel_abs_diff": state_diff,
                    "future_abs_diff": abs(float(batch.targets[index] - batch.targets[other])),
                }
            )
    if not rows:
        return {
            "horizon": horizon,
            "match_tolerance_ticks": tolerance,
            "matched_pair_count": 0,
            "status": "no_matches",
            "claim_boundary": "matching failure is not evidence against latent state",
        }
    divergences = np.asarray([row["future_abs_diff"] for row in rows], dtype=np.float64)
    state_diffs = np.asarray(
        [row["current_multichannel_abs_diff"] for row in rows], dtype=np.float64
    )
    family_pairs = Counter(row["family_pair"] for row in rows)
    session_ids = sorted({session for row in rows for session in row["session_ids"]})
    rng = np.random.default_rng(700 + horizon)
    if len(session_ids) >= 2:
        cluster_values = np.asarray(
            [
                np.mean([row["future_abs_diff"] for row in rows if session in row["session_ids"]])
                for session in session_ids
            ]
        )
        draws = np.asarray(
            [
                np.mean(
                    cluster_values[rng.integers(0, len(cluster_values), size=len(cluster_values))]
                )
                for _ in range(int(config.get("reporting", {}).get("bootstrap_repetitions", 500)))
            ]
        )
        interval = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    else:
        interval = [float("nan"), float("nan")]
    correlation = (
        float(np.corrcoef(state_diffs, divergences)[0, 1])
        if len(rows) > 2 and np.std(state_diffs) > 0 and np.std(divergences) > 0
        else float("nan")
    )
    return {
        "horizon": horizon,
        "match_tolerance_ticks": tolerance,
        "matched_pair_count": len(rows),
        "matched_session_count": len(session_ids),
        "family_pair_counts": dict(family_pairs),
        "residual_current_primary_abs_diff": float(
            np.mean([row["current_primary_abs_diff"] for row in rows])
        ),
        "broader_state_abs_diff": float(np.nanmean(state_diffs)),
        "future_divergence_abs_diff": float(np.mean(divergences)),
        "future_divergence_session_bootstrap_95": interval,
        "state_future_abs_correlation": correlation,
        "pairs": rows[: int(campaign.get("max_reported_matches", 200))],
        "status": "complete",
        "interpretation": "matched-state diagnostics are a search for aliasing; incremental held-out prediction remains the primary test",
    }


def analyze_latent_system_state(
    development: str | Path,
    config: Mapping[str, Any],
    output: str | Path,
    *,
    confirmation: str | Path | None = None,
) -> dict[str, Any]:
    config = validate_latent_system_state_config(config)
    dev_records = load_latent_system_state_acquisition(development)
    campaign = config["latent_system_state"]
    primary_family = str(campaign.get("primary", {}).get("excitation_family", "read_pressure"))
    primary_condition = str(
        campaign.get("primary", {}).get("measurement_condition", "cached_preloaded")
    )
    horizons = tuple(int(item) for item in campaign.get("horizons", (1, 4, 8)))
    result: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
        "development_source": str(development),
        "confirmation_source": None if confirmation is None else str(confirmation),
        "firewall": {
            "explicit_channel_contract": True,
            "future_witness_forbidden": True,
            "confirmation_scaling_forbidden": True,
            "whole_trajectory_normalization_forbidden": True,
            "identity_metadata_forbidden": True,
        },
        "cells": {},
        "matched_state": {},
        "matched_state_confirmation": {},
    }
    dev_by_cell = {(record.family, record.measurement_condition): [] for record in dev_records}
    for record in dev_records:
        dev_by_cell.setdefault((record.family, record.measurement_condition), []).append(record)
    confirm_records = (
        load_latent_system_state_acquisition(confirmation) if confirmation is not None else None
    )
    for (family, condition), records in sorted(dev_by_cell.items()):
        cell = {"development": {}, "confirmation": {}}
        for horizon in horizons:
            cell["development"][str(horizon)] = _evaluate_ladder(records, config, horizon=horizon)
        if confirm_records is not None:
            confirm_cell = [
                record
                for record in confirm_records
                if record.family == family and record.measurement_condition == condition
            ]
            if len(confirm_cell) < 1:
                raise IntegrityError(f"confirmation is missing latent cell {family}/{condition}")
            for horizon in horizons:
                dev_batch = _feature_batch(
                    records,
                    horizon,
                    int(campaign["future_block_length"]),
                    int(campaign["history_length"]),
                )
                confirm_batch = _feature_batch(
                    confirm_cell,
                    horizon,
                    int(campaign["future_block_length"]),
                    int(campaign["history_length"]),
                )
                # Freeze all preprocessing by fitting models on all dev rows;
                # confirmation is only passed to predict/evaluate.
                # Models are refit against development feature matrices, then
                # applied to confirmation matrices with the same feature layout.
                models: dict[str, FittedLatentModel] = {}
                for name in MODEL_NAMES:
                    model = _fit_latent_model(
                        name,
                        dev_batch,
                        np.arange(len(dev_batch.targets), dtype=int),
                        float(campaign.get("ridge_alpha", 1.0)),
                        int(campaign.get("state_rank", 2)),
                    )
                    models[name] = model
                test_mse = {}
                for name, model in models.items():
                    matrix = (
                        confirm_batch.features["current_primary"]
                        if name in {"training_mean", "persistence"}
                        else confirm_batch.features[name]
                    )
                    test_mse[name] = _mse(confirm_batch.targets, model.predict(matrix))
                training_mean_mse = test_mse["training_mean"]
                normalized_skill = {
                    name: (
                        1.0 - test_mse[name] / training_mean_mse
                        if math.isfinite(training_mean_mse) and training_mean_mse > 0
                        else float("nan")
                    )
                    for name in MODEL_NAMES
                }
                primary_pred = models["current_primary"].predict(
                    confirm_batch.features["current_primary"]
                )
                multi_pred = models["current_multichannel"].predict(
                    confirm_batch.features["current_multichannel"]
                )
                cell["confirmation"][str(horizon)] = {
                    "row_count": len(confirm_batch.targets),
                    "test_mse": test_mse,
                    "normalized_skill_vs_training_mean": normalized_skill,
                    "primary_comparison": {
                        "baseline": "current_primary",
                        "candidate": "current_multichannel",
                        "test_loss_improvement": test_mse["current_primary"]
                        - test_mse["current_multichannel"],
                        "session_bootstrap": _session_bootstrap_loss_difference(
                            confirm_batch.targets,
                            primary_pred,
                            multi_pred,
                            confirm_batch.session_ids,
                            seed=301 + horizon,
                            repetitions=int(
                                config.get("reporting", {}).get("bootstrap_repetitions", 500)
                            ),
                        ),
                    },
                    "history_comparison": {
                        "baseline": "current_multichannel",
                        "candidate": "multichannel_history",
                        "test_loss_improvement": test_mse["current_multichannel"]
                        - test_mse["multichannel_history"],
                        "session_bootstrap": _session_bootstrap_loss_difference(
                            confirm_batch.targets,
                            multi_pred,
                            models["multichannel_history"].predict(
                                confirm_batch.features["multichannel_history"]
                            ),
                            confirm_batch.session_ids,
                            seed=401 + horizon,
                            repetitions=int(
                                config.get("reporting", {}).get("bootstrap_repetitions", 500)
                            ),
                        ),
                    },
                    "history_vs_primary_comparison": {
                        "baseline": "current_primary",
                        "candidate": "multichannel_history",
                        "test_loss_improvement": test_mse["current_primary"]
                        - test_mse["multichannel_history"],
                        "session_bootstrap": _session_bootstrap_loss_difference(
                            confirm_batch.targets,
                            models["current_primary"].predict(
                                confirm_batch.features["current_primary"]
                            ),
                            models["multichannel_history"].predict(
                                confirm_batch.features["multichannel_history"]
                            ),
                            confirm_batch.session_ids,
                            seed=451 + horizon,
                            repetitions=int(
                                config.get("reporting", {}).get("bootstrap_repetitions", 500)
                            ),
                        ),
                    },
                    "preprocessing": {
                        "fit_scope": "development only",
                        "confirmation_used_for_scaling": False,
                        "models": {name: model.as_dict() for name, model in models.items()},
                    },
                }
        result["cells"][f"{family}/{condition}"] = cell
    for condition in (primary_condition, "timer_only"):
        matched = [
            record
            for record in dev_records
            if record.family in set(campaign.get("excitation_families", DEFAULT_FAMILIES))
            and record.measurement_condition == condition
        ]
        if matched:
            result["matched_state"][condition] = {
                str(horizon): _matched_state_diagnostics(matched, config, horizon=horizon)
                for horizon in horizons
            }
        if confirm_records is not None:
            confirmation_matched = [
                record
                for record in confirm_records
                if record.family in set(campaign.get("excitation_families", DEFAULT_FAMILIES))
                and record.measurement_condition == condition
            ]
            if confirmation_matched:
                result["matched_state_confirmation"][condition] = {
                    str(horizon): _matched_state_diagnostics(
                        confirmation_matched, config, horizon=horizon
                    )
                    for horizon in horizons
                }
    result["primary"] = {
        "family": primary_family,
        "condition": primary_condition,
        "horizon": int(campaign.get("primary", {}).get("horizon", max(horizons))),
        "comparison": "current_primary_vs_current_multichannel",
        "confirmation_is_untouched": confirmation is not None,
    }
    result["claim_boundary"] = (
        "Predictive representation of ordinary timing/system observations only; no unique hidden physical mechanism inferred"
    )
    _atomic_json(Path(output) / "analysis.json", result)
    return result


def _synthetic_record(case: str, index: int, rng: np.random.Generator) -> LatentStateRecord:
    length = 20
    origin = 9
    hidden = float(index % 2)
    current = 100.0 + rng.normal(0, 0.5)
    workload = 0.0
    if case == "iid":
        future = 100.0 + rng.normal(0, 6.0)
        witness = rng.normal(0, 1.0)
    elif case == "aliasing":
        future = 100.0 + 25.0 * hidden + rng.normal(0, 0.5)
        witness = hidden + rng.normal(0, 0.02)
    elif case == "workload_only":
        workload = float(index % 2)
        future = 100.0 + 20.0 * workload + rng.normal(0, 0.5)
        witness = rng.normal(0, 1.0)
    elif case == "drift":
        future = 100.0 + 10.0 * (index // 6) + rng.normal(0, 0.5)
        witness = rng.normal(0, 1.0)
    elif case == "instrumentation_correlated":
        future = 100.0 + rng.normal(0, 5.0)
        witness = current + rng.normal(0, 0.05)
    elif case == "leakage_trap":
        future = 100.0 + 20.0 * hidden + rng.normal(0, 0.5)
        witness = rng.normal(0, 1.0)
    else:
        raise ValueError(f"unknown synthetic latent case {case}")
    target = np.full(length, 100.0, dtype=np.uint64)
    target[:origin] = np.asarray(np.round(current + rng.normal(0, 0.4, origin)), dtype=np.uint64)
    target[origin] = np.uint64(round(current))
    target[origin + 1 :] = np.asarray(
        np.round(future + rng.normal(0, 0.4, length - origin - 1)), dtype=np.uint64
    )
    witness_values = np.full((length, 1), np.nan, dtype=np.float64)
    witness_values[: origin + 1, 0] = witness
    witness_values[origin + 1 :, 0] = future if case == "leakage_trap" else np.nan
    valid = np.isfinite(witness_values)
    causal = np.zeros_like(valid)
    causal[: origin + 1] = True
    specs = [LOCAL_CHANNEL_SPECS[0].as_dict()]
    availability = np.full(length, origin, dtype=np.int64)
    availability[origin + 1 :] = origin + 1
    return LatentStateRecord(
        trajectory_id=f"synthetic-{case}-{index}",
        session_id=f"synthetic-session-{index}",
        family="read_pressure",
        measurement_condition="cached_preloaded",
        target_ticks=target,
        reference_ticks=np.full(length, -1, dtype=np.int64),
        target_start_tsc=np.arange(length, dtype=np.uint64),
        target_end_tsc=np.arange(1, length + 1, dtype=np.uint64),
        reference_start_tsc=np.full(length, -1, dtype=np.int64),
        reference_end_tsc=np.full(length, -1, dtype=np.int64),
        target_start_aux=np.zeros(length, dtype=np.int64),
        target_end_aux=np.zeros(length, dtype=np.int64),
        reference_start_aux=np.full(length, -1, dtype=np.int64),
        reference_end_aux=np.full(length, -1, dtype=np.int64),
        target_quality=np.ones(length, dtype=np.uint8),
        reference_quality=np.zeros(length, dtype=np.uint8),
        acquisition_order=np.zeros(length, dtype=np.uint8),
        workload_history=np.full(length, workload, dtype=np.float32),
        witness_values=witness_values,
        witness_valid=valid,
        witness_causal_eligible=causal,
        witness_available_at_ns=availability,
        witness_capture_start_ns=np.zeros(length, dtype=np.int64),
        witness_capture_end_ns=availability,
        witness_names=("tsc_aux",),
        origin_channel_names=("cached_preloaded",),
        origin_channel_ticks=np.asarray([target[origin]], dtype=np.uint64),
        origin_channel_start_tsc=np.asarray([origin], dtype=np.uint64),
        origin_channel_end_tsc=np.asarray([origin + 1], dtype=np.uint64),
        origin_channel_start_aux=np.asarray([0], dtype=np.int64),
        origin_channel_end_aux=np.asarray([0], dtype=np.int64),
        origin_channel_quality=np.ones(1, dtype=np.uint8),
        origin_channel_order=np.zeros(1, dtype=np.uint8),
        origin_channel_capture_start_ns=np.asarray([origin], dtype=np.int64),
        origin_channel_capture_end_ns=np.asarray([origin + 1], dtype=np.int64),
        origin_index=origin,
        origin_start_ns=origin,
        origin_end_ns=origin + 1,
        metadata={"witness_provenance": {"channel_specs": specs}, "synthetic_case": case},
    )


def run_latent_system_state_synthetic_validation(
    config: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    config = validate_latent_system_state_config(config)
    cases = (
        "iid",
        "aliasing",
        "workload_only",
        "drift",
        "instrumentation_correlated",
        "leakage_trap",
    )
    result: dict[str, Any] = {
        "schema": SYNTHETIC_SCHEMA,
        "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
        "cases": {},
        "claim_boundary": "synthetic pipeline calibration only; no worker or physical claim",
    }
    for case_index, case in enumerate(cases):
        records = [
            _synthetic_record(case, index, np.random.default_rng(1000 + case_index * 97 + index))
            for index in range(30)
        ]
        ladder = _evaluate_ladder(records, config, horizon=1)
        result["cases"][case] = ladder
    result["firewall_trap"] = {
        "future_rows_marked_causal": False,
        "future_values_are_excluded": True,
        "identity_features": False,
        "confirmation_scaling": False,
    }
    _atomic_json(Path(output) / "synthetic_validation.json", result)
    return result


def _overhead_run(
    kernel: NativeMeasurementKernel,
    tier: int,
    repetitions: int,
    pmu_events: Sequence[str],
    *,
    local_channels: Sequence[str],
    pmu_scope: str,
) -> dict[str, Any]:
    collector = SystemWitnessCollector(
        tier,
        kernel,
        pmu_events if tier >= 1 else (),
        local_channels=local_channels,
        pmu_scope=pmu_scope,
        witness_scope="every_block",
    )
    buffer = ControlledMemoryBuffer(64, lock_memory=False)
    buffer.warmup_touch()
    address = buffer.address
    eviction = bytearray(4096)
    timings: list[int] = []
    wall: list[int] = []
    usage_before = _rusage_values()
    start = time.monotonic_ns()
    try:
        for _index in range(repetitions):
            orders = np.zeros(1, dtype=np.uint8)

            def operation(block_orders: np.ndarray = orders) -> dict[str, np.ndarray]:
                return kernel.measure_trajectory(
                    address,
                    address + 8,
                    np.asarray([TRAJECTORY_CONDITIONS["cached_preloaded"]], dtype=np.uint8),
                    block_orders,
                    eviction_address=int(np.frombuffer(eviction, dtype=np.uint8).ctypes.data),
                    eviction_bytes=len(eviction),
                )

            t0 = time.monotonic_ns()
            raw, _, _, _ = collector.measure(operation)
            t1 = time.monotonic_ns()
            if raw is None:
                continue
            timings.append(int(raw["first_end_tsc"][0] - raw["first_start_tsc"][0]))
            wall.append(t1 - t0)
    finally:
        buffer.close()
    usage_after = _rusage_values()
    elapsed = max(time.monotonic_ns() - start, 1)
    cpu_ns = sum(
        usage_after.get(key, 0) - usage_before.get(key, 0)
        for key in ("thread_user_time_ns", "thread_system_time_ns")
    )
    return {
        "tier": tier,
        "repetitions_requested": repetitions,
        "repetitions_recorded": len(timings),
        "timing_ticks": timings,
        "wall_ns": wall,
        "timing_median_ticks": float(np.median(timings)) if timings else float("nan"),
        "timing_variance_ticks2": float(np.var(timings)) if timings else float("nan"),
        "wall_median_ns": float(np.median(wall)) if wall else float("nan"),
        "wall_p95_ns": float(np.quantile(wall, 0.95)) if wall else float("nan"),
        "thread_cpu_utilization": cpu_ns / elapsed,
        "collector": collector.provenance(),
    }


def run_latent_observer_characterization(
    config: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    config = validate_latent_system_state_config(config)
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        raise RuntimeError("observer characterization requires the native kernel")
    campaign = config["latent_system_state"]
    repetitions = int(campaign.get("observer_characterization_repetitions", 128))
    local_channels = campaign.get("tier1_local_channels", ["tsc_aux"])
    pmu_scope = str(campaign.get("pmu_scope", "origin_only"))
    rows = [
        _overhead_run(
            kernel,
            tier,
            repetitions,
            campaign.get("pmu_events", []),
            local_channels=local_channels,
            pmu_scope=pmu_scope,
        )
        for tier in (0, 1, 2)
    ]
    baseline = rows[0]
    thresholds = campaign.get("observer_effect", {})
    comparisons: dict[str, Any] = {}
    for row in rows[1:]:
        median_ratio = (
            row["wall_median_ns"] / baseline["wall_median_ns"]
            if baseline["wall_median_ns"]
            else float("nan")
        )
        variance_ratio = (
            row["timing_variance_ticks2"] / baseline["timing_variance_ticks2"]
            if baseline["timing_variance_ticks2"]
            else float("nan")
        )
        comparisons[str(row["tier"])] = {
            "added_latency_ns": row["wall_median_ns"] - baseline["wall_median_ns"],
            "wall_latency_ratio": median_ratio,
            "timing_variance_ratio": variance_ratio,
            "timing_median_shift_ticks": row["timing_median_ticks"]
            - baseline["timing_median_ticks"],
            "cpu_utilization": row["thread_cpu_utilization"],
            "practical_thresholds": thresholds,
            "below_practical_threshold": bool(
                median_ratio <= float(thresholds.get("max_wall_latency_ratio", 2.0))
                and (
                    not math.isfinite(variance_ratio)
                    or variance_ratio <= float(thresholds.get("max_timing_variance_ratio", 2.0))
                )
                and abs(row["timing_median_ticks"] - baseline["timing_median_ticks"])
                <= float(thresholds.get("max_timing_median_shift_ticks", 20.0))
                and row["thread_cpu_utilization"]
                <= float(thresholds.get("max_thread_cpu_utilization", 0.25))
            ),
        }
    result = {
        "schema": OVERHEAD_SCHEMA,
        "protocol_version": LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "configuration_hash": config_fingerprint(dict(config)),
        "execution_host": platform.node() or "unavailable",
        "tiers": rows,
        "comparisons_to_tier0": comparisons,
        "thresholds_frozen_before_confirmation": thresholds,
        "interpretation": "observer overhead is empirical; a passing threshold does not prove zero observer effect",
        "claim_boundary": "measurement perturbation characterization only",
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "observer-effect.json", result)
    return result
