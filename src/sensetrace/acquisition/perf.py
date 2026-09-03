"""Safe, scoped discovery of Linux performance-counter capabilities.

Discovery is deliberately non-invasive. It inspects the local PMU sysfs
description and, when available, ``perf list`` output; it does not attach to
unrelated processes or collect system-wide counters. The reusable reader in
this module accepts only a SenseTrace-owned calling-thread operation scope.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
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
    selection_status: str
    encodings: tuple[dict[str, Any], ...] = ()
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

_GENERIC_HARDWARE_NAMES = {
    0: "PERF_TYPE_HARDWARE",
    4: "PERF_TYPE_RAW",
}

PERF_TYPE_HARDWARE = 0
PERF_TYPE_RAW = 4
PERF_ATTR_SIZE = 128
PERF_FLAG_FD_CLOEXEC = 1 << 3
# Thread-scoped core PMUs have architecture-specific names.  This allowlist is
# deliberately conservative: unknown devices are rejected until their scope is
# classified, while common hybrid Intel and ARM core PMUs remain usable.
THREAD_SCOPED_PMU_EXACT_NAMES = frozenset({"cpu", "cpu_core", "cpu_atom"})
THREAD_SCOPED_PMU_PREFIXES = ("armv8_pmuv3", "riscv_pmu")
SYSTEM_WIDE_PMU_PREFIXES = (
    "uncore_",
    "cstate_",
    "i915",
    "msr",
    "power",
    "intel_pt",
    "intel_bts",
    "arm_spe",
)
PERF_ATTR_DISABLED = 1 << 0
PERF_ATTR_INHERIT = 1 << 1
PERF_ATTR_PINNED = 1 << 2
PERF_ATTR_EXCLUDE_USER = 1 << 4
PERF_ATTR_EXCLUDE_KERNEL = 1 << 5
PERF_ATTR_EXCLUDE_HV = 1 << 6
PERF_ATTR_EXCLUDE_IDLE = 1 << 7
PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
SUPPORTED_READ_FORMAT_MASK = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403


class PerfEventError(RuntimeError):
    """A scoped perf-event operation failed with preserved provenance."""

    def __init__(self, message: str, *, provenance: dict[str, Any], errno_value: int | None = None):
        super().__init__(message)
        self.provenance = provenance
        self.errno_value = errno_value


@dataclass(frozen=True)
class PerfEventEncoding:
    """One fully qualified PMU event description."""

    device: str
    alias: str
    source_type: int | str
    config: int | None
    config_fields: dict[str, str]
    format_fields: dict[str, str]
    raw_spec: str

    @property
    def qualified_name(self) -> str:
        return f"{self.device}/{self.alias}/"

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "alias": self.alias,
            "qualified_name": self.qualified_name,
            "event_source_type": {
                "value": self.source_type,
                "name": _GENERIC_HARDWARE_NAMES.get(
                    self.source_type if isinstance(self.source_type, int) else -1,
                    "sysfs_pmu_type",
                ),
            },
            "config": self.config,
            "config_width_bits": 64,
            "config_fields": dict(self.config_fields),
            "sysfs_format_fields": dict(self.format_fields),
            "raw_spec": self.raw_spec,
            "selection": "qualified device plus alias; bare aliases are never used for selection",
        }


@dataclass(frozen=True)
class ScopedPerfEventReading:
    """A single operation-scoped counter read with multiplexing evidence."""

    raw_count: int
    time_enabled: int
    time_running: int
    read_format: int

    @property
    def multiplexed(self) -> bool:
        return self.time_running != self.time_enabled

    @property
    def scaled_count(self) -> float | None:
        if self.time_running <= 0:
            return None
        return float(self.raw_count * self.time_enabled / self.time_running)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "complete" if self.time_running > 0 else "not_running",
            "raw_count": self.raw_count,
            "scaled_count": self.scaled_count,
            "time_enabled": self.time_enabled,
            "time_running": self.time_running,
            "multiplexed": self.multiplexed,
            "read_format": self.read_format,
            "read_format_fields": {
                "total_time_enabled": bool(self.read_format & PERF_FORMAT_TOTAL_TIME_ENABLED),
                "total_time_running": bool(self.read_format & PERF_FORMAT_TOTAL_TIME_RUNNING),
            },
        }


def build_perf_event_attr(
    event: str,
    *,
    raw_event: int | None = None,
    event_type: int | None = None,
    device: str = "kernel-generic",
    read_format: int = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
    size: int = PERF_ATTR_SIZE,
) -> tuple[bytes, dict[str, Any]]:
    """Build the ABI prefix used by a scoped ``perf_event_open`` probe.

    The bitfield is serialized explicitly because ctypes bitfield layout is
    compiler-dependent.  The returned bytes are suitable for the Linux ABI's
    little-endian x86_64/aarch64 layout through the flags word at offset 40.
    """

    if size < 48 or size > PERF_ATTR_SIZE:
        raise ValueError("perf_event_attr size must cover the flags word and fit the probe buffer")
    if raw_event is None and event_type is not None:
        raise ValueError("event_type requires an explicit raw_event config")
    if raw_event is None:
        if event not in _GENERIC_HARDWARE_CODES:
            raise ValueError(f"no generic encoding is known for event {event!r}")
        event_type = PERF_TYPE_HARDWARE
        config = _GENERIC_HARDWARE_CODES[event]
        encoding = f"PERF_TYPE_HARDWARE:{config}"
    else:
        if not 0 <= raw_event <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("raw PMU event encoding must fit the 64-bit perf config field")
        event_type = PERF_TYPE_RAW if event_type is None else event_type
        config = raw_event
        encoding = f"type={event_type}:config=0x{raw_event:x}"
    validate_read_format(read_format)
    flags = PERF_ATTR_DISABLED | PERF_ATTR_EXCLUDE_KERNEL | PERF_ATTR_EXCLUDE_HV
    attribute = bytearray(PERF_ATTR_SIZE)
    struct.pack_into("<IIQ", attribute, 0, event_type, size, config)
    struct.pack_into("<Q", attribute, 32, read_format)
    struct.pack_into("<Q", attribute, 40, flags)
    return bytes(attribute), {
        "type": event_type,
        "event_source_type": {
            "value": event_type,
            "name": _GENERIC_HARDWARE_NAMES.get(event_type, "sysfs_pmu_type"),
            "device": device,
        },
        "config": config,
        "config_width_bits": 64,
        "size": size,
        "read_format": read_format,
        "read_format_fields": {
            "total_time_enabled": bool(read_format & PERF_FORMAT_TOTAL_TIME_ENABLED),
            "total_time_running": bool(read_format & PERF_FORMAT_TOTAL_TIME_RUNNING),
        },
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
        "device": device,
    }


def expected_read_size(read_format: int) -> int:
    """Return the exact byte count the kernel returns for a read_format."""
    validate_read_format(read_format)
    size = 8  # value field is always present
    if read_format & PERF_FORMAT_TOTAL_TIME_ENABLED:
        size += 8
    if read_format & PERF_FORMAT_TOTAL_TIME_RUNNING:
        size += 8
    return size


def validate_read_format(read_format: int) -> None:
    """Reject ABI layouts that this single-event parser does not decode."""

    if not isinstance(read_format, int) or not 0 <= read_format <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("perf read_format must fit the 64-bit ABI field")
    unsupported = read_format & ~SUPPORTED_READ_FORMAT_MASK
    if unsupported:
        raise ValueError(
            "unsupported perf read_format flags "
            f"{unsupported:#x}; only TOTAL_TIME_ENABLED and TOTAL_TIME_RUNNING are decoded"
        )


def classify_pmu_scope(device: str) -> dict[str, str]:
    """Classify a PMU device for the calling-thread reader, fail-closed.

    Linux does not expose one portable sysfs scope attribute.  The evidence
    therefore records the conservative name rule used rather than pretending
    the classification was discovered from hardware.
    """

    if device in THREAD_SCOPED_PMU_EXACT_NAMES:
        return {
            "scope": "thread_scoped_core_pmu",
            "status": "accepted",
            "basis": "conservative exact-name classification",
        }
    if any(device.startswith(prefix) for prefix in THREAD_SCOPED_PMU_PREFIXES):
        return {
            "scope": "thread_scoped_core_pmu",
            "status": "accepted",
            "basis": "conservative architecture-specific core-PMU prefix classification",
        }
    if any(device.startswith(prefix) for prefix in SYSTEM_WIDE_PMU_PREFIXES):
        return {
            "scope": "system_or_package_scoped_pmu",
            "status": "rejected",
            "basis": "known non-core PMU prefix classification",
        }
    return {
        "scope": "unknown",
        "status": "rejected",
        "basis": "unclassified PMU devices fail closed",
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
            ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
        )
    )


class OperationScopedPerfEvent:
    """Read one PMU event only while a caller-owned operation is executing.

    The object always binds to the current native thread, uses ``cpu=-1`` in
    the per-thread sense, creates the event disabled, and never accepts a
    caller-supplied PID.  It is intentionally a single-event reader: a future
    grouped or system-wide collector would need a separate, explicit contract.
    """

    def __init__(
        self,
        event: str | PerfEventEncoding,
        *,
        syscall_number: int | None = None,
        read_format: int = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
    ) -> None:
        if isinstance(event, PerfEventEncoding):
            if event.config is None:
                raise ValueError(
                    f"PMU event {event.qualified_name!r} has no directly usable config encoding"
                )
            event_type = event.source_type if isinstance(event.source_type, int) else None
            if event_type is None:
                raise ValueError("PMU event source type must be numeric for perf_event_open")
            self._pmu_scope = classify_pmu_scope(event.device)
            if self._pmu_scope["status"] != "accepted":
                raise ValueError(
                    f"PMU device {event.device!r} cannot back a calling-thread scoped "
                    f"reader (classified scope={self._pmu_scope['scope']!r}; "
                    f"basis={self._pmu_scope['basis']})"
                )
            self.event = event
            self._attribute_bytes, self._attribute_record = build_perf_event_attr(
                event.alias,
                raw_event=event.config,
                event_type=event_type,
                device=event.device,
                read_format=read_format,
            )
        else:
            self._pmu_scope = {
                "scope": "thread_scoped_kernel_generic_hardware_event",
                "status": "accepted",
                "basis": "PERF_TYPE_HARDWARE generic event opened for the calling thread",
            }
            self.event = PerfEventEncoding(
                device="kernel-generic",
                alias=event,
                source_type=PERF_TYPE_HARDWARE,
                config=_GENERIC_HARDWARE_CODES.get(event),
                config_fields={"generic_alias": event},
                format_fields={},
                raw_spec=event,
            )
            self._attribute_bytes, self._attribute_record = build_perf_event_attr(
                event, read_format=read_format
            )
        self._syscall_number = syscall_number or getattr(os, "SYS_perf_event_open", None)
        if self._syscall_number is None:
            self._syscall_number = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
        self._read_format = read_format
        self._fd: int | None = None
        self._enabled = False
        self._thread_id: int | None = None
        self._open_errno: int | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "event": self.event.as_dict(),
            "perf_event_attr": self._attribute_record,
            "scope": {
                "kind": "calling_thread",
                "pid_argument": self._thread_id,
                "cpu_argument": -1,
                "inherit": False,
                "system_wide": False,
                "unrelated_processes": "not attached",
                "pmu_classification": dict(self._pmu_scope),
            },
            "read_format": self._read_format,
            "read_format_fields": {
                "total_time_enabled": bool(self._read_format & PERF_FORMAT_TOTAL_TIME_ENABLED),
                "total_time_running": bool(self._read_format & PERF_FORMAT_TOTAL_TIME_RUNNING),
            },
            "multiplexing": "single event; read time_enabled and time_running explicitly",
            "errno": self._open_errno or 0,
            "errno_name": errno.errorcode.get(self._open_errno or 0, "none"),
        }

    def open(self) -> OperationScopedPerfEvent:
        if self._fd is not None:
            raise RuntimeError("scoped perf event is already open")
        if self._syscall_number is None:
            raise PerfEventError(
                "perf_event_open is unsupported on this architecture",
                provenance=self.provenance,
            )
        self._thread_id = threading.get_native_id()
        attribute = (ctypes.c_ubyte * len(self._attribute_bytes)).from_buffer_copy(
            self._attribute_bytes
        )
        fd = _perf_event_open(
            attribute,
            syscall_number=self._syscall_number,
            thread_id=self._thread_id,
            cpu=-1,
        )
        if fd < 0:
            self._open_errno = ctypes.get_errno()
            provenance = self.provenance
            raise PerfEventError(
                f"perf_event_open failed: {errno.errorcode.get(self._open_errno, self._open_errno)}",
                provenance=provenance,
                errno_value=self._open_errno,
            )
        # Defense in depth: the syscall already passes FD_CLOEXEC, but an
        # explicit flag guards against a kernel that ignores it.
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        except OSError:
            pass
        self._fd = fd
        return self

    def _ioctl(self, request: int) -> None:
        if self._fd is None:
            raise RuntimeError("scoped perf event is not open")
        try:
            fcntl.ioctl(self._fd, request, 0)
        except OSError as exc:
            raise PerfEventError(
                f"perf event ioctl {request:#x} failed: {exc}",
                provenance=self.provenance,
                errno_value=exc.errno,
            ) from exc

    def reset_and_enable(self) -> None:
        self._ioctl(PERF_EVENT_IOC_RESET)
        self._ioctl(PERF_EVENT_IOC_ENABLE)
        self._enabled = True

    def disable(self) -> None:
        if self._enabled:
            self._ioctl(PERF_EVENT_IOC_DISABLE)
            self._enabled = False

    def read(self) -> ScopedPerfEventReading:
        if self._fd is None:
            raise RuntimeError("scoped perf event is not open")
        expected_size = expected_read_size(self._read_format)
        try:
            payload = os.read(self._fd, expected_size)
        except OSError as exc:
            raise PerfEventError(
                f"reading scoped perf event failed: {exc}",
                provenance=self.provenance,
                errno_value=exc.errno,
            ) from exc
        if len(payload) != expected_size:
            raise PerfEventError(
                f"scoped perf event returned {len(payload)} bytes; expected {expected_size}",
                provenance=self.provenance,
            )
        if expected_size == 8:
            (raw_count,) = struct.unpack("<Q", payload)
            time_enabled, time_running = 0, 0
        elif expected_size == 16:
            raw_count, only = struct.unpack("<QQ", payload)
            if self._read_format & PERF_FORMAT_TOTAL_TIME_ENABLED:
                time_enabled, time_running = int(only), 0
            else:
                time_enabled, time_running = 0, int(only)
        else:
            raw_count, time_enabled, time_running = struct.unpack("<QQQ", payload)
        return ScopedPerfEventReading(
            raw_count=int(raw_count),
            time_enabled=int(time_enabled),
            time_running=int(time_running),
            read_format=self._read_format,
        )

    def measure(self, operation: Callable[[], Any]) -> tuple[Any, ScopedPerfEventReading]:
        """Run exactly one controlled callback between enable and disable."""

        self.open()
        try:
            self.reset_and_enable()
            result = operation()
            self.disable()
            return result, self.read()
        finally:
            self.close()

    def close(self) -> None:
        if self._fd is None:
            return
        # Clear identity first so a failed os.close cannot leave a stale
        # open-fd identity behind (previous code skipped the reset on error).
        fd, was_enabled = self._fd, self._enabled
        self._fd = None
        self._enabled = False
        if was_enabled:
            try:
                fcntl.ioctl(fd, PERF_EVENT_IOC_DISABLE, 0)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> OperationScopedPerfEvent:
        return self.open()

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


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
        if error in {errno.EACCES, errno.EPERM}:
            status = "permission_denied"
        elif error in {errno.ENODEV, errno.ENOENT, errno.EOPNOTSUPP}:
            status = "unsupported_or_disabled"
        elif error == errno.EINVAL:
            # EINVAL means the attr/encoding itself was rejected (bad size,
            # unknown config, or a non-per-thread device misused as per-thread),
            # not merely a locked-down paranoid level.
            status = "invalid_encoding_or_scope"
        elif error == errno.EBUSY:
            status = "event_busy_multiplexed_or_exclusive"
        elif error == errno.E2BIG:
            status = "attr_size_mismatch"
        else:
            status = f"error:{errno.errorcode.get(error, error)}"
        result = {
            "status": status,
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
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
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


def _parse_sysfs_event_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in value.strip().split(","):
        key, separator, field_value = token.partition("=")
        if separator and key:
            fields[key.strip()] = field_value.strip()
        elif token.strip():
            fields.setdefault("unparsed", token.strip())
    return fields


def _parse_numeric(value: str) -> int | None:
    try:
        return int(value, 0)
    except ValueError:
        return None


def _encode_sysfs_config(
    config_fields: dict[str, str], format_fields: dict[str, str]
) -> int | None:
    """Encode only fully understood ``config`` fields; preserve the rest verbatim."""

    config = 0
    for config_field, value_text in config_fields.items():
        if config_field == "unparsed":
            return None
        value = _parse_numeric(value_text)
        format_text = format_fields.get(config_field)
        if value is None or format_text is None or ":" not in format_text:
            return None
        register, _, ranges = format_text.partition(":")
        if register != "config":
            return None
        bit_positions: list[int] = []
        for interval in ranges.split(","):
            interval = interval.strip()
            if not interval:
                return None
            start_text, separator, end_text = interval.partition("-")
            try:
                if not separator:
                    # Single-bit fields such as "config:0" carry no dash.
                    start = end = int(start_text)
                else:
                    start, end = int(start_text), int(end_text)
            except ValueError:
                return None
            if start < 0 or end < start:
                return None
            bit_positions.extend(range(start, end + 1))
        if value >= 1 << len(bit_positions):
            return None
        for source_bit, destination_bit in enumerate(bit_positions):
            if value & (1 << source_bit):
                config |= 1 << destination_bit
    return config


def _sysfs_event_encodings(sysfs_root: Path) -> list[PerfEventEncoding]:
    encodings: list[PerfEventEncoding] = []
    devices = sysfs_root / "bus/event_source/devices"
    try:
        device_paths = sorted(path for path in devices.iterdir() if path.is_dir())
    except OSError:
        return encodings
    for device in device_paths:
        source_type_text = _read_text(device / "type") or "unavailable"
        try:
            source_type: int | str = int(source_type_text, 0)
        except ValueError:
            source_type = source_type_text
        format_fields: dict[str, str] = {}
        try:
            for path in sorted((device / "format").iterdir()):
                if path.is_file():
                    format_fields[path.name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        events_dir = device / "events"
        try:
            event_paths = sorted(path for path in events_dir.iterdir() if path.is_file())
        except OSError:
            continue
        for path in event_paths:
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            config_fields = _parse_sysfs_event_fields(value)
            config_value = _encode_sysfs_config(config_fields, format_fields)
            encodings.append(
                PerfEventEncoding(
                    device=device.name,
                    alias=path.name,
                    source_type=source_type,
                    config=config_value,
                    config_fields=dict(config_fields),
                    format_fields=dict(format_fields),
                    raw_spec=value,
                )
            )
    return encodings


def select_sysfs_event_encoding(sysfs_root: str | Path, event_name: str) -> PerfEventEncoding:
    """Resolve a PMU event only by qualified ``device/alias/`` identity."""

    encodings = _sysfs_event_encodings(Path(sysfs_root))
    if event_name.endswith("/"):
        matches = [item for item in encodings if item.qualified_name == event_name]
    else:
        matches = [item for item in encodings if item.alias == event_name]
    if len(matches) != 1:
        raise ValueError(
            f"PMU event {event_name!r} is not uniquely qualified; use device/alias/ "
            f"(matches={len(matches)})"
        )
    if matches[0].config is None or not isinstance(matches[0].source_type, int):
        raise ValueError(
            f"PMU event {matches[0].qualified_name!r} has no directly usable config encoding"
        )
    return matches[0]


def _sysfs_event_names(sysfs_root: Path) -> set[str]:
    names: set[str] = set()
    for encoding in _sysfs_event_encodings(sysfs_root):
        names.add(encoding.alias)
        names.add(encoding.qualified_name)
    return names


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
    sysfs_encodings = _sysfs_event_encodings(root)
    available = _sysfs_event_names(root) | _perf_event_names(perf_output)
    encodings_by_alias: dict[str, list[PerfEventEncoding]] = {}
    encodings_by_qualified: dict[str, PerfEventEncoding] = {}
    for encoding in sysfs_encodings:
        encodings_by_alias.setdefault(encoding.alias, []).append(encoding)
        encodings_by_qualified[encoding.qualified_name] = encoding
    permission_boundary = _permission_boundary(Path(proc_root))
    probe_thread_id = threading.get_native_id()
    event_records: list[CounterEventCapability] = []
    for logical_name, candidates in EVENT_CANDIDATES.items():
        found = tuple(candidate for candidate in candidates if candidate in available)
        selected_encodings: list[PerfEventEncoding] = []
        ambiguous_candidates: list[str] = []
        for candidate in found:
            if candidate.endswith("/"):
                resolved_encoding = encodings_by_qualified.get(candidate)
                if resolved_encoding is not None:
                    selected_encodings.append(resolved_encoding)
                continue
            matches = encodings_by_alias.get(candidate, [])
            if len(matches) == 1:
                selected_encodings.append(matches[0])
            elif len(matches) > 1:
                ambiguous_candidates.append(candidate)
                selected_encodings.extend(matches)
        if ambiguous_candidates:
            selection_status = "ambiguous_unqualified_alias"
        elif selected_encodings:
            selection_status = "qualified_encoding_available"
        else:
            selection_status = "generic_or_vocabulary_only"
        probe = (
            _probe_generic_event(found[0], thread_id=probe_thread_id)
            if probe_hardware_events and found and not ambiguous_candidates
            else {}
        )
        event_status = (
            "ambiguous" if ambiguous_candidates else "available" if found else "unavailable"
        )
        event_records.append(
            CounterEventCapability(
                logical_name=logical_name,
                candidate_events=candidates,
                available_events=found,
                status=event_status,
                probe_status=probe.get("status", "not_requested"),
                scope="SenseTrace-owned calling thread only; cpu=-1 follows that thread; no system-wide collection",
                provenance=(
                    "Linux PMU sysfs and perf list"
                    if available
                    else "no readable PMU event description"
                ),
                selection_status=selection_status,
                encodings=tuple(item.as_dict() for item in selected_encodings),
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
            "sysfs_event_encodings": [item.as_dict() for item in sysfs_encodings],
            "raw_encoding_probe": "not_requested_for_non-generic_events",
            "selection_policy": (
                "qualified PMU device/type/config/format records are preserved; ambiguous bare aliases "
                "are never selected"
            ),
        },
        "event_multiplexing": {
            "status": "not_tested",
            "reason": "capability discovery does not collect or multiplex counters",
        },
        "operation_scoped_read": {
            "status": "implemented_not_run_by_discovery",
            "scope": "calling thread only; cpu=-1; inherit=0; no system-wide or unrelated-process collection",
            "reason": "capability discovery does not run an operation-scoped measurement",
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
