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
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
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


def _probe_generic_event(event: str) -> str:
    """Open one generic event for this process and close it immediately.

    This is a capability probe, not a measurement.  ``pid=0`` and
    ``cpu=-1`` scope it to the current SenseTrace process; kernel and
    hypervisor counting are not requested.  The kernel ABI struct is passed as
    a zero-filled buffer so unknown future fields remain disabled.
    """

    code = _GENERIC_HARDWARE_CODES.get(event)
    if code is None:
        return "not_probed_non_generic_event"
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = getattr(os, "SYS_perf_event_open", None)
    if syscall is None:
        syscall = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
    if syscall is None:
        return "unsupported_architecture"
    attribute = (ctypes.c_ubyte * 128)()
    ctypes.c_uint32.from_buffer(attribute, 0).value = 0  # PERF_TYPE_HARDWARE
    ctypes.c_uint32.from_buffer(attribute, 4).value = 128
    ctypes.c_uint64.from_buffer(attribute, 8).value = code
    fd = libc.syscall(
        ctypes.c_long(syscall),
        ctypes.byref(attribute),
        ctypes.c_int(0),  # current SenseTrace process
        ctypes.c_int(-1),  # any CPU, never system-wide
        ctypes.c_int(-1),
        ctypes.c_ulong(0),
    )
    if fd < 0:
        error = ctypes.get_errno()
        if error in {errno.EACCES, errno.EPERM}:
            return "permission_denied"
        if error in {errno.EINVAL, errno.ENODEV, errno.ENOENT}:
            return "unsupported_or_disabled"
        return f"error:{errno.errorcode.get(error, error)}"
    os.close(fd)
    return "readable_scoped_probe"


def _run(command: list[str], *, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
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
) -> dict[str, Any]:
    """Discover event names without assuming one CPU's PMU vocabulary."""

    root = Path(sysfs_root)
    runner = command_runner or (lambda command: _run(command, timeout=8.0))
    selected_perf = perf_path or shutil.which("perf")
    if perf_output is None and selected_perf:
        perf_output = runner([selected_perf, "list", "--no-desc"])
    available = _sysfs_event_names(root) | _perf_event_names(perf_output)
    event_records: list[CounterEventCapability] = []
    for logical_name, candidates in EVENT_CANDIDATES.items():
        found = tuple(candidate for candidate in candidates if candidate in available)
        event_records.append(
            CounterEventCapability(
                logical_name=logical_name,
                candidate_events=candidates,
                available_events=found,
                status="available" if found else "unavailable",
                probe_status=(
                    _probe_generic_event(found[0])
                    if probe_hardware_events and found
                    else "not_requested"
                ),
                scope="SenseTrace-owned process only; discovery performs no collection",
                provenance=(
                    "Linux PMU sysfs and perf list"
                    if available
                    else "no readable PMU event description"
                ),
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
            path.name
            for path in (root / "bus/event_source/devices").iterdir()
            if path.is_dir()
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
        "events": [item.as_dict() for item in event_records],
        "probe_policy": (
            "A readable_scoped_probe only opens and closes a generic event for the current "
            "SenseTrace process; it does not collect a sample or observe another process."
            if probe_hardware_events
            else "No event probe requested; status reflects vocabulary discovery only."
        ),
        "collection_policy": (
            "No system-wide or unrelated-process collection. A future reader must use a "
            "SenseTrace-owned process/thread scope and record permission failures explicitly."
        ),
        "privilege_policy": "Do not broaden kernel perf permissions or disable security controls for discovery.",
    }
