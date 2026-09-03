"""Exact-host target identity and non-destructive worker-03 inventory."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

WORKER03_HARDWARE_ID = "worker03-hardware-v1"
WORKER03_BASELINE = {
    "system_vendor": "Dell Inc.",
    "product_name": "Precision Tower 3431",
    "cpu_model": "Intel(R) Core(TM) i7-9700 CPU",
    "physical_cores": 8,
    "smt": "disabled",
    "ram_bytes": 32 * 1024**3,
    "target_scope": "exact-host experimental target; not a portability claim",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unavailable"
    except (OSError, UnicodeError):
        return "unavailable"


def _command(command: list[str], runner: Callable[..., Any] = subprocess.run) -> str:
    try:
        result = runner(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired, TypeError):
        return "unavailable"
    if getattr(result, "returncode", 1) != 0:
        return "unavailable"
    return str(getattr(result, "stdout", "")).strip() or "unavailable"


def _lscpu_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _integer(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_topology(sysfs_root: Path) -> list[dict[str, str]]:
    caches: list[dict[str, str]] = []
    for path in sorted((sysfs_root / "devices/system/cpu/cpu0/cache").glob("index*")):
        caches.append(
            {
                "index": path.name,
                "level": _read(path / "level"),
                "type": _read(path / "type"),
                "size": _read(path / "size"),
                "ways": _read(path / "ways_of_associativity"),
                "line_size": _read(path / "coherency_line_size"),
                "shared_cpu_list": _read(path / "shared_cpu_list"),
            }
        )
    return caches or [{"status": "unavailable", "provenance": "cache sysfs unavailable"}]


def collect_worker03_inventory(
    *,
    sysfs_root: str | Path = "/sys",
    proc_root: str | Path = "/proc",
    command_runner: Callable[..., Any] = subprocess.run,
    native_library: str | Path | None = None,
) -> dict[str, Any]:
    """Collect observable target properties without changing machine state.

    A target is ``matched`` only when the observed CPU, chassis, and core/SMT
    shape agree with the frozen baseline.  Missing observations remain
    ``unavailable`` and never get filled from the baseline.
    """

    sysfs = Path(sysfs_root)
    proc = Path(proc_root)
    lscpu = _lscpu_fields(_command(["lscpu"], command_runner))
    cpuinfo = _read(proc / "cpuinfo")
    flags_line = next(
        (line for line in cpuinfo.splitlines() if line.lower().startswith("flags")), ""
    )
    flags = flags_line.split(":", 1)[1].split() if ":" in flags_line else []
    dmi = {
        key: _read(sysfs / "devices/virtual/dmi/id" / key)
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
        ]
    }
    cache_topology = _cache_topology(sysfs)
    cache_line_sizes = sorted(
        {
            item["line_size"]
            for item in cache_topology
            if item.get("line_size") not in {None, "", "unavailable"}
        }
    )
    physical_cores = _integer(lscpu.get("Core(s) per socket", "unavailable"))
    sockets = _integer(lscpu.get("Socket(s)", "1")) or 1
    threads = _integer(lscpu.get("CPU(s)", "unavailable"))
    smt_active = _read(sysfs / "devices/system/cpu/smt/active")
    observed_smt = (
        "disabled"
        if smt_active == "0" or lscpu.get("Thread(s) per core") == "1"
        else "enabled"
        if smt_active == "1" or lscpu.get("Thread(s) per core") not in {"", "unavailable"}
        else "unavailable"
    )
    observed = {
        "system_vendor": dmi["sys_vendor"],
        "product_name": dmi["product_name"],
        "cpu_model": lscpu.get("Model name", "unavailable"),
        "physical_cores": None if physical_cores is None else physical_cores * sockets,
        "logical_cpus": threads,
        "smt": observed_smt,
        "architecture": platform.machine() or "unavailable",
        "kernel": platform.release() or "unavailable",
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "governor": {
            path.parent.parent.name: _read(path)
            for path in sorted(sysfs.glob("devices/system/cpu/cpu*/cpufreq/scaling_governor"))
        },
        "turbo_no_turbo": _read(sysfs / "devices/system/cpu/intel_pstate/no_turbo"),
        "cache_topology": cache_topology,
        "cache_line_size": cache_line_sizes[0]
        if len(cache_line_sizes) == 1
        else (cache_line_sizes if cache_line_sizes else "unavailable"),
        "cpuid_feature_flags": flags or ["unavailable"],
        "native_feature_indicators": {
            "rdtscp": "available" if "rdtscp" in flags else "unavailable",
            "clflush": "available" if "clflush" in flags else "unavailable",
            "avx2": "available" if "avx2" in flags else "unavailable",
        },
        "cpuid_feature_leaves": {
            "status": "unavailable",
            "provenance": "no privileged CPUID-leaf collector in this inventory path",
        },
        "microcode": next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.splitlines()
                if line.startswith("microcode:")
            ),
            "unavailable",
        ),
        "tsc": {key: key in flags for key in ["constant_tsc", "nonstop_tsc", "rdtscp"]},
        "numa": {
            "nodes": _command(
                [
                    "find",
                    str(sysfs / "devices/system/node"),
                    "-maxdepth",
                    "1",
                    "-name",
                    "node[0-9]*",
                ],
                command_runner,
            ),
            "online": _read(sysfs / "devices/system/node/online"),
        },
        "kernel_command_line": _read(proc / "cmdline"),
        "hugepages": {
            "nr_hugepages": _read(sysfs / "kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"),
            "transparent_hugepage": _read(sysfs / "kernel/mm/transparent_hugepage/enabled"),
        },
        "memory": _command(["free", "-b"], command_runner),
        "dimm_spd_or_dmi": _command(["dmidecode", "--type", "memory"], command_runner),
        "temperatures": _command(["sensors"], command_runner),
    }
    expected_cpu = str(WORKER03_BASELINE["cpu_model"])
    matches = (
        observed["system_vendor"] == WORKER03_BASELINE["system_vendor"]
        and observed["product_name"] == WORKER03_BASELINE["product_name"]
        and isinstance(observed["cpu_model"], str)
        and expected_cpu in observed["cpu_model"]
        and observed["physical_cores"] == WORKER03_BASELINE["physical_cores"]
        and observed["smt"] == WORKER03_BASELINE["smt"]
    )
    native: str | dict[str, Any] = "unavailable"
    if native_library is not None:
        path = Path(native_library)
        if path.exists():
            native = {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
    return {
        "schema": "sensetrace.worker03-inventory.v1",
        "hardware_target_id": WORKER03_HARDWARE_ID,
        "target_match": "matched" if matches else "not_verified",
        "target_match_rule": "all frozen vendor/product/CPU/core/SMT observations must agree",
        "baseline": WORKER03_BASELINE,
        "observed": observed,
        "native_probe_artifact": native,
        "collection": "non-destructive local observation only",
        "unsupported": [
            "exact CPUID leaves unless a future native collector is present",
            "DIMM SPD fields when dmidecode is unavailable or permission denied",
            "unobservable BIOS/turbo/topology fields are not inferred",
        ],
    }
