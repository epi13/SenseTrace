"""Safe, scoped discovery of Linux performance-counter capabilities.

Discovery is deliberately non-invasive.  It inspects the local PMU sysfs
description and, when available, ``perf list`` output; it does not attach to
unrelated processes or collect system-wide counters.  Actual event reads are
left to a future primitive that can scope them to a SenseTrace-owned process.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import shutil
import struct
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CounterEventCapability:
    logical_name: str
    candidate_events: tuple[str, ...]
    available_events: tuple[str, ...]
    status: str
    probe_status: str
    scope: str
    provenance: str
    encoding: dict[str, str] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EVENT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cache_references": ("cache-references",),
    "cache_misses": ("cache-misses",),
    "llc_load_misses": (
        "mem_load_retired.llc_miss",
        "longest_lat_cache.miss",
        "LLC-load-misses",
    ),
    "retired_memory_loads": ("mem_inst_retired.all_loads",),
    "memory_stall_cycles": ("cycle_activity.stalls_mem_any",),
    "data_tlb_load_misses": ("dtlb_load_misses.miss_causes_a_walk",),
    "uncore_memory_reads": ("uncore_imc_0/cas_count_read/",),
}

_GENERIC_HARDWARE_CODES = {
    "cache-references": 2,
    "cache-misses": 3,
}

PERF_TYPE_HARDWARE = 0
PERF_TYPE_RAW = 4
PERF_ATTR_SIZE = 128
PERF_ATTR_DISABLED = 1 << 0
PERF_ATTR_INHERIT = 1 << 1
PERF_ATTR_PINNED = 1 << 2
PERF_ATTR_EXCLUDE_USER = 1 << 4
PERF_ATTR_EXCLUDE_KERNEL = 1 << 5
PERF_ATTR_EXCLUDE_HV = 1 << 6
PERF_ATTR_EXCLUDE_IDLE = 1 << 7


def build_perf_event_attr(
    event: str,
    *,
    raw_event: int | None = None,
    size: int = PERF_ATTR_SIZE,
) -> tuple[bytes, dict[str, Any]]:
    """Build the ABI prefix used by a scoped ``perf_event_open`` probe.

    The bitfield is serialized explicitly because ctypes bitfield layout is
    compiler-dependent.  The returned bytes are suitable for the Linux ABI's
    little-endian x86_64/aarch64 layout through the flags word at offset 40.
    """

    if size < 48 or size > PERF_ATTR_SIZE:
        raise ValueError("perf_event_attr size must cover the flags word and fit the probe buffer")
    if raw_event is None:
        if event not in _GENERIC_HARDWARE_CODES:
            raise ValueError(f"no generic encoding is known for event {event!r}")
        event_type = PERF_TYPE_HARDWARE
        config = _GENERIC_HARDWARE_CODES[event]
        encoding = f"PERF_TYPE_HARDWARE:{config}"
    else:
        if not 0 <= raw_event <= 0xFFFFFFFF:
            raise ValueError("raw PMU event encoding must fit the perf config field")
        event_type = PERF_TYPE_RAW
        config = raw_event
        encoding = f"PERF_TYPE_RAW:0x{raw_event:x}"
    flags = PERF_ATTR_DISABLED | PERF_ATTR_EXCLUDE_KERNEL | PERF_ATTR_EXCLUDE_HV
    attribute = bytearray(PERF_ATTR_SIZE)
    struct.pack_into("<IIQ", attribute, 0, event_type, size, config)
    struct.pack_into("<Q", attribute, 40, flags)
    return bytes(attribute), {
        "type": event_type,
        "config": config,
        "size": size,
        "flags": flags,
        "flags_by_name": {
            "disabled": True,
            "inherit": False,
            "pinned": False,
            "exclude_user": False,
            "exclude_kernel": True,
            "exclude_hypervisor": True,
            "exclude_idle": False,
        },
        "encoding": encoding,
    }


def _perf_event_open(
    attribute: ctypes.Array[Any], *, syscall_number: int, thread_id: int, cpu: int
) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    return int(
        libc.syscall(
            ctypes.c_long(syscall_number),
            ctypes.byref(attribute),
            ctypes.c_int(thread_id),
            ctypes.c_int(cpu),
            ctypes.c_int(-1),
            ctypes.c_ulong(0),
        )
    )


def _probe_generic_event(event: str, *, thread_id: int | None = None) -> dict[str, Any]:
    """Open one generic event for this process and close it immediately.

    This is a capability probe, not a measurement.  The native calling-thread
    ID and ``cpu=-1`` scope it to the current SenseTrace thread; kernel and
    hypervisor counting are not requested.  The kernel ABI struct is passed as
    a zero-filled buffer so unknown future fields remain disabled.
    """

    code = _GENERIC_HARDWARE_CODES.get(event)
    if code is None:
        return {
            "status": "not_probed_non_generic_event",
            "event": event,
            "reason": "no generic encoding is available for this vocabulary entry",
        }
    syscall = getattr(os, "SYS_perf_event_open", None)
    if syscall is None:
        syscall = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
    if syscall is None:
        return {
            "status": "unsupported_architecture",
            "event": event,
            "architecture": platform.machine(),
        }
    attribute_bytes, attribute_record = build_perf_event_attr(event)
    attribute = (ctypes.c_ubyte * len(attribute_bytes)).from_buffer_copy(attribute_bytes)
    scoped_thread = int(thread_id if thread_id is not None else threading.get_native_id())
    fd = _perf_event_open(attribute, syscall_number=syscall, thread_id=scoped_thread, cpu=-1)
    if fd < 0:
        error = ctypes.get_errno()
        result = {
            "status": "permission_denied"
            if error in {errno.EACCES, errno.EPERM}
            else "unsupported_or_disabled"
            if error in {errno.EINVAL, errno.ENODEV, errno.ENOENT}
            else f"error:{errno.errorcode.get(error, error)}",
            "event": event,
            "errno": error,
            "errno_name": errno.errorcode.get(error, "unknown"),
            "scope": {
                "kind": "calling_thread",
                "pid_argument": scoped_thread,
                "cpu_argument": -1,
                "inherit": False,
                "system_wide": False,
            },
            "perf_event_attr": attribute_record,
        }
        return result
    os.close(fd)
    return {
        "status": "opened_scoped_probe",
        "event": event,
        "errno": 0,
        "errno_name": "none",
        "scope": {
            "kind": "calling_thread",
            "pid_argument": scoped_thread,
            "cpu_argument": -1,
            "inherit": False,
            "system_wide": False,
        },
        "perf_event_attr": attribute_record,
        "collection": "not_started; open-and-close capability probe only",
    }


def _run(command: list[str], *, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _sysfs_event_names(sysfs_root: Path) -> set[str]:
    names: set[str] = set()
    devices = sysfs_root / "bus/event_source/devices"
    try:
        device_paths = sorted(path for path in devices.iterdir() if path.is_dir())
    except OSError:
        return names
    for device in device_paths:
        events_dir = device / "events"
        try:
            for path in events_dir.iterdir():
                if path.is_file():
                    names.add(path.name)
                    names.add(f"{device.name}/{path.name}/")
        except OSError:
            continue
    return names


def _sysfs_event_encodings(sysfs_root: Path) -> dict[str, str]:
    encodings: dict[str, str] = {}
    devices = sysfs_root / "bus/event_source/devices"
    try:
        device_paths = sorted(path for path in devices.iterdir() if path.is_dir())
    except OSError:
        return encodings
    for device in device_paths:
        events_dir = device / "events"
        try:
            for path in events_dir.iterdir():
                if path.is_file():
                    value = path.read_text(encoding="utf-8").strip()
                    encodings[path.name] = value
                    encodings[f"{device.name}/{path.name}/"] = value
        except OSError:
            continue
    return encodings


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _permission_boundary(proc_root: Path) -> dict[str, Any]:
    paranoid_path = proc_root / "sys/kernel/perf_event_paranoid"
    paranoid_text = _read_text(paranoid_path)
    try:
        paranoid: int | str = int(paranoid_text) if paranoid_text is not None else "unavailable"
    except ValueError:
        paranoid = paranoid_text or "unavailable"
    status_text = _read_text(proc_root / "self/status") or ""
    cap_eff_text = next(
        (
            line.split(":", 1)[1].strip()
            for line in status_text.splitlines()
            if line.startswith("CapEff:")
        ),
        None,
    )
    try:
        cap_eff = int(cap_eff_text, 16) if cap_eff_text is not None else None
    except ValueError:
        cap_eff = None
    cap_perfmon = bool(cap_eff is not None and cap_eff & (1 << 38))
    cap_sys_admin = bool(cap_eff is not None and cap_eff & (1 << 21))
    return {
        "perf_event_paranoid": paranoid,
        "perf_event_paranoid_source": str(paranoid_path),
        "effective_capabilities_hex": cap_eff_text or "unavailable",
        "cap_perfmon": cap_perfmon,
        "cap_sys_admin_equivalent": cap_sys_admin,
        "capability_source": str(proc_root / "self/status"),
        "permission_interpretation": (
            "CAP_PERFMON present"
            if cap_perfmon
            else "CAP_SYS_ADMIN may provide legacy perf override"
            if cap_sys_admin
            else "no CAP_PERFMON or CAP_SYS_ADMIN bit observed"
        ),
    }


def _pmu_descriptors(sysfs_root: Path) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    devices = sysfs_root / "bus/event_source/devices"
    try:
        paths = sorted(path for path in devices.iterdir() if path.is_dir())
    except OSError:
        return descriptors
    for path in paths:
        descriptors.append(
            {
                "name": path.name,
                "type": _read_text(path / "type") or "unavailable",
                "cpumask": _read_text(path / "cpumask") or "unavailable",
                "cpus": _read_text(path / "cpus") or "unavailable",
            }
        )
    return descriptors


def _perf_event_names(perf_output: str | None) -> set[str]:
    if not perf_output:
        return set()
    names: set[str] = set()
    for line in perf_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("List of"):  # perf list header
            continue
        # perf list presents aliases before a description, often as
        # ``cache-misses OR cpu/cache-misses/``. Keep only safe tokens.
        for token in stripped.split():
            token = token.rstrip(":,")
            if token.startswith(("#", "[", "(", "-")):
                continue
            if any(char.isalpha() for char in token) and "/" not in token[:1]:
                names.add(token)
    return names


def discover_counter_capabilities(
    *,
    sysfs_root: str | Path = "/sys",
    perf_output: str | None = None,
    perf_path: str | None = None,
    command_runner: Callable[[list[str]], str | None] | None = None,
    probe_hardware_events: bool = False,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Discover event names without assuming one CPU's PMU vocabulary."""

    root = Path(sysfs_root)
    runner = command_runner or (lambda command: _run(command, timeout=8.0))
    selected_perf = perf_path or shutil.which("perf")
    if perf_output is None and selected_perf:
        perf_output = runner([selected_perf, "list", "--no-desc"])
    available = _sysfs_event_names(root) | _perf_event_names(perf_output)
    encodings = _sysfs_event_encodings(root)
    permission_boundary = _permission_boundary(Path(proc_root))
    probe_thread_id = threading.get_native_id()
    event_records: list[CounterEventCapability] = []
    for logical_name, candidates in EVENT_CANDIDATES.items():
        found = tuple(candidate for candidate in candidates if candidate in available)
        probe = (
            _probe_generic_event(found[0], thread_id=probe_thread_id)
            if probe_hardware_events and found
            else {}
        )
        event_records.append(
            CounterEventCapability(
                logical_name=logical_name,
                candidate_events=candidates,
                available_events=found,
                status="available" if found else "unavailable",
                probe_status=probe.get("status", "not_requested"),
                scope="SenseTrace-owned calling thread only; cpu=-1 follows that thread; no system-wide collection",
                provenance=(
                    "Linux PMU sysfs and perf list"
                    if available
                    else "no readable PMU event description"
                ),
                encoding={
                    candidate: encodings[candidate] for candidate in found if candidate in encodings
                },
                probe=probe,
            )
        )
    cpu_model = platform.processor() or "unavailable"
    lscpu = runner(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if line.lower().startswith(("model name:", "model:")):
                cpu_model = line.split(":", 1)[1].strip() or cpu_model
                break
    pmu_devices: list[str] = []
    try:
        pmu_devices = sorted(
            path.name for path in (root / "bus/event_source/devices").iterdir() if path.is_dir()
        )
    except OSError:
        pass
    return {
        "schema": "sensetrace.performance-counter-capabilities.v1",
        "cpu_model": cpu_model,
        "architecture": platform.machine(),
        "perf_path": selected_perf or "unavailable",
        "perf_list_status": "available" if perf_output else "unavailable",
        "pmu_devices": pmu_devices or ["unavailable"],
        "pmu_descriptors": _pmu_descriptors(root),
        "events": [item.as_dict() for item in event_records],
        "permission_boundary": permission_boundary,
        "process_thread_scope": {
            "kind": "calling_thread",
            "pid_argument": probe_thread_id,
            "cpu_argument": -1,
            "inherit": False,
            "system_wide": False,
            "unrelated_processes": "not attached",
        },
        "raw_core_event_support": {
            "status": "vocabulary_discovered",
            "sysfs_event_encodings": encodings,
            "raw_encoding_probe": "not_requested_for_non-generic_events",
        },
        "event_multiplexing": {
            "status": "not_tested",
            "reason": "capability discovery does not collect or multiplex counters",
        },
        "operation_scoped_read": {
            "status": "not_implemented",
            "reason": "no counter was read around a SenseTrace-owned operation",
        },
        "probe_policy": (
            "An opened_scoped_probe only opens and closes a generic event for the current "
            "SenseTrace-owned calling thread with disabled creation flags; it does not collect "
            "a sample or observe another process."
            if probe_hardware_events
            else "No event probe requested; status reflects vocabulary discovery only."
        ),
        "collection_policy": (
            "No system-wide or unrelated-process collection. A future reader must use a "
            "SenseTrace-owned process/thread scope and record permission failures explicitly."
        ),
        "privilege_policy": "Do not broaden kernel perf permissions or disable security controls for discovery.",
    }
