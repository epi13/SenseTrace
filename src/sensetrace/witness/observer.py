"""Optional bounded bpftrace-backed kernel witness observer."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ClockAlignment, WitnessEvent, WitnessHookCapability, WitnessSession
from .program import (
    HOOK_TRACEPOINTS,
    program_manifest,
    render_bpftrace_program,
    render_bpftrace_program_for_targets,
)

KERNEL_CLOCK_DOMAIN = "kernel bpf_ktime_get_ns monotonic nanoseconds"
USER_CLOCK_DOMAIN = "userspace CLOCK_MONOTONIC nanoseconds"


def _tracefs_root() -> Path | None:
    for candidate in (Path("/sys/kernel/tracing/events"), Path("/sys/kernel/debug/tracing/events")):
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _effective_capabilities() -> dict[str, Any]:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return {
            "effective_hex": "unavailable",
            "cap_bpf": None,
            "cap_perfmon": None,
            "cap_sys_admin": None,
        }
    text = next(
        (
            line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.startswith("CapEff:")
        ),
        "",
    )
    try:
        value = int(text, 16)
    except ValueError:
        value = 0
    return {
        "effective_hex": text or "unavailable",
        "cap_bpf": bool(value & (1 << 39)),
        "cap_perfmon": bool(value & (1 << 38)),
        "cap_sys_admin": bool(value & (1 << 21)),
    }


def _tool_identity(name: str, arguments: tuple[str, ...]) -> dict[str, str]:
    path = shutil.which(name)
    if path is None:
        return {"path": "unavailable", "version": "unavailable"}
    try:
        result = subprocess.run(
            [path, *arguments], capture_output=True, text=True, timeout=5, check=False
        )
        version = (result.stdout or result.stderr).splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired, IndexError):
        version = "unavailable"
    return {"path": path, "version": version or "unavailable"}


def _host_identity() -> dict[str, str]:
    def read(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip() or "unavailable"
        except OSError:
            return "unavailable"

    machine_id = read("/etc/machine-id")
    return {
        "hostname": platform.node() or "unavailable",
        "boot_id": read("/proc/sys/kernel/random/boot_id"),
        "machine_id_sha256": (
            hashlib.sha256(machine_id.encode()).hexdigest()
            if machine_id != "unavailable"
            else "unavailable"
        ),
    }


def discover_witness_capabilities(
    requested_hooks: tuple[str, ...] | None = None,
    *,
    bpftrace_path: str | None = None,
    tracefs_root: str | Path | None = None,
    btf_path: str | Path = "/sys/kernel/btf/vmlinux",
) -> dict[str, Any]:
    """Discover tools and hooks without loading a BPF program."""

    requested = requested_hooks or tuple(HOOK_TRACEPOINTS)
    unknown = sorted(set(requested) - set(HOOK_TRACEPOINTS))
    if unknown:
        raise ValueError(f"unknown witness hooks: {unknown}")
    tool = bpftrace_path or shutil.which("bpftrace")
    root = Path(tracefs_root) if tracefs_root is not None else _tracefs_root()
    hooks: list[WitnessHookCapability] = []
    for name in requested:
        category, event = HOOK_TRACEPOINTS[name].split(":", 1)
        path = root / category / event if root is not None else Path("unavailable")
        if root is None:
            state, reason = "unavailable", "tracefs events directory is unavailable"
        elif _is_dir(path):
            state, reason = "supported", "tracepoint exists in tracefs"
        else:
            state, reason = "unsupported", "tracepoint is absent on this kernel"
        hooks.append(WitnessHookCapability(name, HOOK_TRACEPOINTS[name], state, reason, str(path)))  # type: ignore[arg-type]
    btf = Path(btf_path)
    if platform.system() != "Linux":
        status = "unsupported"
    elif tool is None or root is None:
        status = "unavailable"
    elif any(hook.status == "supported" for hook in hooks):
        status = "supported"
    else:
        status = "unavailable"
    version = "unavailable"
    if tool:
        try:
            result = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, timeout=5, check=False
            )
            version = (result.stdout or result.stderr).strip() or "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "schema": "sensetrace.ebpf-witness-capabilities.v1",
        "status": status,
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
        "host_identity": _host_identity(),
        "backend": "bpftrace",
        "backend_path": tool or "unavailable",
        "backend_version": version,
        "toolchain": {
            "bpftool": _tool_identity("bpftool", ("version",)),
            "clang": _tool_identity("clang", ("--version",)),
        },
        "tracefs_root": str(root) if root is not None else "unavailable",
        "btf": {
            "status": "available" if _is_file(btf) else "unavailable",
            "path": str(btf),
            "required_by_selected_backend": False,
        },
        "privilege": {"euid": os.geteuid(), **_effective_capabilities()},
        "requested_hooks": list(requested),
        "hooks": [hook.as_dict() for hook in hooks],
        "missing_semantics": "an absent hook is unsupported/unavailable, never interpreted as zero events",
        "collection": "not_collected",
    }


class BpftraceWitnessObserver:
    """Load a temporary, target-filtered observer and retain its provenance."""

    def __init__(
        self,
        *,
        session_id: str,
        experiment_id: str,
        target_pid: int,
        target_tid: int | None,
        output_dir: str | Path,
        requested_hooks: tuple[str, ...] | None = None,
        use_sudo: bool = False,
        bpftrace_path: str | None = None,
        target_tids: tuple[int, ...] | None = None,
    ) -> None:
        self.session_id = session_id
        self.experiment_id = experiment_id
        self.target_pid = int(target_pid)
        self.target_tid = int(target_tid) if target_tid is not None else self.target_pid
        if target_tids is not None:
            observed = tuple(sorted({int(tid) for tid in target_tids}))
            if not observed or any(tid <= 0 for tid in observed):
                raise ValueError("target_tids must be non-empty positive thread IDs")
            self.target_tids: tuple[int, ...] = observed
        else:
            self.target_tids = (self.target_tid,)
        self.output_dir = Path(output_dir)
        self.requested_hooks = requested_hooks or tuple(HOOK_TRACEPOINTS)
        self.use_sudo = use_sudo
        self.bpftrace_path = bpftrace_path
        self._process: subprocess.Popen[str] | None = None
        self._started_at = ""
        self._start_user_before = 0
        self._source = ""
        self._attached: tuple[str, ...] = ()
        self._capabilities: dict[str, Any] = {}
        self._failure: dict[str, Any] | None = None

    @property
    def event_path(self) -> Path:
        return self.output_dir / "events.jsonl"

    def start(self, *, timeout: float = 8.0) -> bool:
        self._started_at = datetime.now(UTC).isoformat()
        self._capabilities = discover_witness_capabilities(
            self.requested_hooks, bpftrace_path=self.bpftrace_path
        )
        tool = self._capabilities["backend_path"]
        if self._capabilities["status"] != "supported" or tool == "unavailable":
            self._failure = {
                "kind": "observer_unavailable",
                "message": "required backend or tracepoints unavailable",
            }
            return False
        by_name = {item["name"]: item for item in self._capabilities["hooks"]}
        self._attached = tuple(
            name for name in self.requested_hooks if by_name[name]["status"] == "supported"
        )
        if len(self.target_tids) == 1:
            self._source = render_bpftrace_program(self.session_id, self._attached)
            extra_args = [str(self.target_pid), str(self.target_tids[0])]
        else:
            self._source = render_bpftrace_program_for_targets(
                self.session_id, self._attached, target_tids=self.target_tids
            )
            extra_args = [str(self.target_pid)]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.output_dir / "observer.bt"
        source_path.write_text(self._source, encoding="utf-8")
        command = [
            str(tool),
            "-q",
            "-B",
            "line",
            "-o",
            str(self.event_path),
            str(source_path),
            *extra_args,
        ]
        if self.use_sudo:
            command = ["sudo", "-n", *command]
        self._start_user_before = time.monotonic_ns()
        try:
            self._process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
        except OSError as exc:
            self._failure = {"kind": "startup_failure", "message": str(exc)}
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                error = self._process.stderr.read() if self._process.stderr else ""
                kind = "permission_denied" if "permission" in error.lower() else "startup_failure"
                self._failure = {
                    "kind": kind,
                    "message": error.strip() or "observer exited during startup",
                    "returncode": self._process.returncode,
                }
                return False
            ready = self._read_ready()
            if ready is not None:
                self._ready_kernel_ns = ready
                self._ready_user_after = time.monotonic_ns()
                return True
            time.sleep(0.02)
        self._failure = {"kind": "startup_failure", "message": "observer readiness timed out"}
        self._terminate()
        return False

    def _read_ready(self) -> int | None:
        try:
            lines = self.event_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                value.get("schema") == "sensetrace.witness-ready.v1"
                and value.get("session_id") == self.session_id
            ):
                return int(value["timestamp_ns"])
        return None

    def _terminate(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)

    def stop(self) -> WitnessSession:
        self._terminate()
        ended_at = datetime.now(UTC).isoformat()
        events: list[WitnessEvent] = []
        malformed = 0
        try:
            lines = self.event_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if value.get("schema") != "sensetrace.witness-event.v1":
                continue
            fields = {
                key: item
                for key, item in value.items()
                if key
                not in {"schema", "session_id", "event_type", "timestamp_ns", "cpu", "pid", "tid"}
            }
            event = WitnessEvent(
                session_id=str(value.get("session_id", "")),
                event_type=str(value.get("event_type", "")),
                timestamp_ns=int(value.get("timestamp_ns", -1)),
                clock_domain=KERNEL_CLOCK_DOMAIN,
                cpu=int(value["cpu"]) if value.get("cpu") is not None else None,
                pid=int(value["pid"]) if value.get("pid") is not None else None,
                tid=int(value["tid"]) if value.get("tid") is not None else None,
                fields=fields,
            )
            try:
                event.validate()
            except ValueError:
                malformed += 1
                continue
            events.append(event)
        unavailable = tuple(
            WitnessHookCapability(**item)
            for item in self._capabilities.get("hooks", [])
            if item["status"] != "supported"
        )
        if hasattr(self, "_ready_kernel_ns"):
            handshake_latency = max(0, self._ready_user_after - self._ready_kernel_ns)
            alignment = ClockAlignment(
                KERNEL_CLOCK_DOMAIN,
                USER_CLOCK_DOMAIN,
                0,
                100_000,
                "bounded",
                "Linux bpf_ktime_get_ns and CLOCK_MONOTONIC semantic clock identity; "
                "zero offset with a conservative 100 us boundary uncertainty, not a claim "
                "of exact instruction-level synchronization",
            )
        else:
            handshake_latency = None
            alignment = ClockAlignment(
                KERNEL_CLOCK_DOMAIN,
                USER_CLOCK_DOMAIN,
                None,
                None,
                "unavailable",
                "observer readiness handshake unavailable",
            )
        if self._failure is not None:
            status = (
                "unavailable"
                if self._failure["kind"] in {"observer_unavailable", "permission_denied"}
                else "failed"
            )
        elif unavailable or malformed:
            status = "incomplete"
        else:
            status = "operational"
        source_hash = (
            hashlib.sha256(self._source.encode("utf-8")).hexdigest()
            if self._source
            else "not_collected"
        )
        provenance = {
            **self._capabilities,
            "program_sha256": source_hash,
            "source_sha256": source_hash,
            "program_manifest": {
                **program_manifest(self.session_id, self._attached),
                "target_tids": list(self.target_tids),
                "target_filtering": (
                    "single TID via $2"
                    if len(self.target_tids) == 1
                    else "multiple TIDs embedded literally; $1 remains target PID"
                ),
            },
            "target_tids": list(self.target_tids),
            "loaded_program_identifiers": "not_collected_by_bpftrace_backend",
            "observer_process_id": self._process.pid if self._process is not None else None,
            "attached_tracepoints": [HOOK_TRACEPOINTS[name] for name in self._attached],
            "malformed_event_lines": malformed,
            "event_count": len(events),
            "hook_event_counts": {
                name: sum(event.event_type == name for event in events) for name in self._attached
            },
            "clock_source": KERNEL_CLOCK_DOMAIN,
            "readiness_output_latency_ns": handshake_latency,
            "loader_privilege": {
                "mode": "sudo_noninteractive" if self.use_sudo else "current_process_credentials",
                "sudo_requested": self.use_sudo,
                "load_verified_by_readiness_event": hasattr(self, "_ready_kernel_ns"),
            },
            "historical_evidence_mutation": "none; this session is a new immutable artifact",
        }
        return WitnessSession(
            session_id=self.session_id,
            experiment_id=self.experiment_id,
            status=status,  # type: ignore[arg-type]
            target_pid=self.target_pid,
            target_tid=self.target_tid,
            requested_hooks=self.requested_hooks,
            attached_hooks=self._attached,
            unavailable_hooks=unavailable,
            events=tuple(events),
            provenance=provenance,
            alignment=alignment,
            started_at=self._started_at,
            ended_at=ended_at,
            failure=self._failure,
        )
