"""Best-effort host inventory with explicit unavailable values."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

from . import __version__


def _command(command: list[str], *, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _command_ok(command: list[str], *, timeout: float = 5.0) -> bool:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _read(path: str) -> str | None:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _value(value: str | None, reason: str) -> dict[str, str]:
    return {"value": value if value is not None else "unavailable", "provenance": reason}


def _rapl_domains() -> list[dict[str, str]]:
    domains: list[dict[str, str]] = []
    for path in sorted(Path("/sys/class/powercap").glob("intel-rapl:*")):
        domains.append(
            {
                "path": str(path),
                "name": _read(str(path / "name")) or "unavailable",
                "energy_uj": _read(str(path / "energy_uj")) or "unavailable",
                "max_energy_range_uj": _read(str(path / "max_energy_range_uj")) or "unavailable",
            }
        )
    return domains


def _watchdog_inventory() -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for path in sorted(Path("/sys/class/watchdog").glob("watchdog*")):
        devices.append(
            {
                "name": path.name,
                "identity": _read(str(path / "identity")) or "unavailable",
                "state": _read(str(path / "state")) or "unavailable",
                "timeout": _read(str(path / "timeout")) or "unavailable",
                "timeleft": _read(str(path / "timeleft")) or "unavailable",
                "driver": (
                    os.path.basename(os.path.realpath(path / "device/driver"))
                    if (path / "device/driver").exists()
                    else "unavailable"
                ),
            }
        )
    return devices


def collect_inventory() -> dict[str, Any]:
    machine_id = _read("/etc/machine-id")
    boot_id = _read("/proc/sys/kernel/random/boot_id")
    hostname = socket.gethostname()
    governor_paths = list(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
    governors = {}
    for path in governor_paths:
        governors[path.parent.parent.name] = _read(str(path)) or "unavailable"
    dmi = {}
    for key in [
        "sys_vendor",
        "product_name",
        "product_version",
        "board_vendor",
        "board_name",
        "board_version",
        "bios_vendor",
        "bios_version",
        "bios_date",
    ]:
        dmi[key] = _read(f"/sys/devices/virtual/dmi/id/{key}") or "unavailable"
    commands = {
        "os_release": ["cat", "/etc/os-release"],
        "cpu": ["lscpu"],
        "memory": ["free", "-h"],
        "storage": ["lsblk", "-o", "NAME,MODEL,SERIAL,SIZE,TYPE,FSTYPE,MOUNTPOINTS"],
        "filesystems": ["df", "-hT"],
        "network": ["ip", "-brief", "addr"],
        "gpu": ["sh", "-c", "command -v lspci >/dev/null && lspci -nn"],
        "sensors": ["sensors"],
        "dimm_spd": ["sh", "-c", "command -v dmidecode >/dev/null && dmidecode --type memory"],
    }
    command_output = {
        name: _value(_command(command), f"command: {' '.join(command)}")
        for name, command in commands.items()
    }
    systemd = {
        "default_target": _value(_command(["systemctl", "get-default"]), "systemctl get-default"),
        "sshd_active": _value(
            _command(["systemctl", "is-active", "sshd"]), "systemctl is-active sshd"
        ),
        "sshd_enabled": _value(
            _command(["systemctl", "is-enabled", "sshd"]), "systemctl is-enabled sshd"
        ),
        "display_manager": _value(
            _command(["systemctl", "is-enabled", "display-manager.service"]),
            "systemctl is-enabled display-manager.service",
        ),
        "running_services": _value(
            _command(
                [
                    "systemctl",
                    "list-units",
                    "--type=service",
                    "--state=running",
                    "--no-legend",
                    "--no-pager",
                ]
            ),
            "systemctl list-units --type=service --state=running",
        ),
        "sensetrace_system": {
            "load_state": _value(
                _command(["systemctl", "show", "sensetrace.service", "-p", "LoadState", "--value"]),
                "systemctl show sensetrace.service LoadState",
            ),
            "active": _value(
                _command(["systemctl", "is-active", "sensetrace.service"]),
                "systemctl is-active sensetrace.service",
            ),
            "enabled": _value(
                _command(["systemctl", "is-enabled", "sensetrace.service"]),
                "systemctl is-enabled sensetrace.service",
            ),
        },
        "sensetrace_user": {
            "active": _value(
                _command(["systemctl", "--user", "is-active", "sensetrace.service"]),
                "systemctl --user is-active sensetrace.service",
            ),
            "enabled": _value(
                _command(["systemctl", "--user", "is-enabled", "sensetrace.service"]),
                "systemctl --user is-enabled sensetrace.service",
            ),
        },
    }
    sysctl = {
        key: _value(_command(["sysctl", "-n", key]), f"sysctl {key}")
        for key in [
            "kernel.panic",
            "kernel.panic_on_oops",
            "kernel.softlockup_panic",
            "kernel.hardlockup_panic",
        ]
    }
    watchdog = {
        "devices": [str(path) for path in Path("/dev").glob("watchdog*")],
        "sysfs": _watchdog_inventory(),
        "systemd": _value(
            _command(
                [
                    "systemctl",
                    "show",
                    "systemd",
                    "--property=RuntimeWatchdogUSec,RebootWatchdogUSec",
                ]
            ),
            "systemctl show systemd watchdog properties",
        ),
        "decision": "not enabled: provider and safe timeout behavior require privileged validation",
    }
    packages = {}
    for name in ["python3", "rustc", "cargo", "git", "gcc", "make", "sensors", "wdctl", "dnf"]:
        packages[name] = shutil.which(name) or "unavailable"
    return {
        "schema": "sensetrace.host-inventory.v1",
        "sensetrace_version": __version__,
        "host_id": machine_id or f"hostname:{hostname}",
        "hostname": hostname,
        "boot_id": boot_id or "unavailable",
        "timestamp_utc": _command(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]) or "unavailable",
        "os": _value(_read("/etc/os-release"), "/etc/os-release"),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "dmi": dmi,
        "cpu": command_output["cpu"],
        "memory": command_output["memory"],
        "storage": command_output["storage"],
        "filesystems": command_output["filesystems"],
        "network": command_output["network"],
        "gpu": command_output["gpu"],
        "dimm_spd": command_output["dimm_spd"],
        "sensors": command_output["sensors"],
        "memory_frequency_and_timings": {
            "value": "unavailable",
            "provenance": "not exposed by current collector",
        },
        "cpu_governor": governors
        or {"value": "unavailable", "provenance": "cpufreq sysfs unavailable"},
        "systemd": systemd,
        "sysctl": sysctl,
        "watchdog": watchdog,
        "power_energy": {
            "rapl_domains": _rapl_domains(),
            "scope": "domain-specific; interpret only using the domain name",
            "provenance": "/sys/class/powercap/intel-rapl:*",
        },
        "performance_counters": {
            "perf_path": shutil.which("perf") or "unavailable",
            "provenance": "perf executable presence only; no event was interpreted as bit-state",
        },
        "packages": packages,
        "sudo_noninteractive": os.geteuid() == 0 or _command_ok(["sudo", "-n", "true"]),
        "reboot_history": _value(_command(["last", "-x", "-n", "20"]), "last -x -n 20"),
    }
