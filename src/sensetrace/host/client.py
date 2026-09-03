"""One Fabric-backed controller interface for worker operations."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from shlex import quote
from typing import Any

from fabric import Connection


class RemoteHost:
    def __init__(self, alias: str = "worker-03", *, project_root: str | Path | None = None):
        self.alias = alias
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3])
        self.connection = self._connection()

    def _ssh_config(self) -> dict[str, str]:
        try:
            result = subprocess.run(
                ["ssh", "-G", self.alias], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot resolve SSH alias {self.alias}: {exc}") from exc
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, _, value = line.partition(" ")
            if key and value:
                values[key] = value.strip()
        return values

    def _connection(self) -> Connection:
        config = self._ssh_config()
        kwargs: dict[str, Any] = {}
        identity = config.get("identityfile")
        if identity:
            kwargs["key_filename"] = os.path.expanduser(identity.split()[0])
        return Connection(
            config.get("hostname", self.alias),
            user=config.get("user"),
            port=int(config.get("port", "22")),
            connect_kwargs=kwargs,
        )

    def run(self, command: str, *, warn: bool = False, hide: bool = False) -> Any:
        return self.connection.run(command, warn=warn, hide=hide)

    def sudo_available(self) -> bool:
        result = self.run("sudo -n true", warn=True, hide=True)
        return result.ok

    def service_management(self) -> dict[str, Any]:
        """Return the only service scope the controller is allowed to manage."""

        system_load = self.run(
            "systemctl show sensetrace.service -p LoadState --value", warn=True, hide=True
        )
        system_enabled = self.run("systemctl is-enabled sensetrace.service", warn=True, hide=True)
        system_active = self.run("systemctl is-active sensetrace.service", warn=True, hide=True)
        user_enabled = self.run(
            "systemctl --user is-enabled sensetrace.service", warn=True, hide=True
        )
        user_active = self.run(
            "systemctl --user is-active sensetrace.service", warn=True, hide=True
        )
        system_present = (
            system_load.stdout.strip() == "loaded"
            or system_enabled.stdout.strip()
            in {
                "enabled",
                "static",
                "indirect",
            }
        )
        user_present = (
            user_enabled.stdout.strip() in {"enabled", "static", "indirect"}
            or user_active.stdout.strip() == "active"
        )
        if system_present and user_present:
            mode = "invalid-dual-service"
        elif system_present:
            mode = "system"
        elif user_present:
            mode = "user-fallback"
        else:
            mode = "not-installed"
        return {
            "mode": mode,
            "system": {
                "load_state": system_load.stdout.strip() or "not-found",
                "enabled": system_enabled.stdout.strip() or "not-found",
                "active": system_active.stdout.strip() or "inactive",
            },
            "user": {
                "enabled": user_enabled.stdout.strip() or "not-found",
                "active": user_active.stdout.strip() or "inactive",
            },
        }

    def _home(self) -> str:
        return self.run("printf '%s' \"$HOME\"", hide=True).stdout.strip()

    def inventory(self) -> dict[str, Any]:
        home = self._home()
        command = (
            f"{home}/.local/share/sensetrace/venv/bin/python -m sensetrace.cli inventory --json "
            f"2>/dev/null || python3 -m sensetrace.cli inventory --json"
        )
        result = self.run(command, warn=True, hide=True)
        return json.loads(result.stdout) if result.ok else {"error": result.stderr.strip()}

    def doctor(self) -> dict[str, Any]:
        checks = {
            "ssh": True,
            "hostname": self.run("hostname", hide=True).stdout.strip(),
            "user": self.run("id -un", hide=True).stdout.strip(),
            "python": self.run("python3 --version", warn=True, hide=True).stdout.strip(),
            "systemd": self.run("command -v systemctl", warn=True, hide=True).ok,
            "sudo_noninteractive": self.sudo_available(),
            "installation": self.run(
                "test -x ~/.local/share/sensetrace/venv/bin/sensetrace && "
                "printf '%s' ~/.local/share/sensetrace/venv/bin/sensetrace || "
                "command -v sensetrace || true",
                warn=True,
                hide=True,
            ).stdout.strip()
            or "not installed",
        }
        checks["service_management"] = self.service_management()
        checks["management_mode"] = checks["service_management"]["mode"]
        checks["warning"] = (
            "both system and user SenseTrace services are present/enabled; stop and disable the fallback"
            if checks["management_mode"] == "invalid-dual-service"
            else "privileged provisioning requires an interactive sudo authorization"
            if not checks["sudo_noninteractive"]
            else None
        )
        return checks

    def _archive_project(self) -> Path:
        temporary = Path(tempfile.mkdtemp(prefix="sensetrace-source-"))
        archive = temporary / "source.tar.gz"
        excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "data/raw", "runs"}
        with tarfile.open(archive, "w:gz") as handle:
            for path in self.project_root.rglob("*"):
                relative = path.relative_to(self.project_root)
                relative_name = str(relative)
                if any(
                    relative_name == item or relative_name.startswith(item + "/")
                    for item in excluded
                ):
                    continue
                if path.is_file() and path.suffix != ".so":
                    handle.add(path, arcname=str(relative))
        return archive

    def _local_commit(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    def bootstrap(self) -> dict[str, Any]:
        home = self._home()
        sudo = self.sudo_available()
        management = self.service_management()
        if management["mode"] == "invalid-dual-service":
            raise RuntimeError("refusing bootstrap in invalid-dual-service mode")
        if management["mode"] == "system" and not sudo:
            raise RuntimeError(
                "system service is authoritative but sudo authorization is unavailable"
            )
        mode = "system" if sudo else "user-fallback"
        self.run(
            f"mkdir -p {home}/.local/share/sensetrace/{{source,runs,state,logs}} {home}/.config/sensetrace",
            hide=True,
        )
        return {
            "mode": mode,
            "home": home,
            "sudo_noninteractive": sudo,
            "existing_service_management": management,
        }

    def deploy(self, *, reset_config: bool = False) -> dict[str, Any]:
        bootstrap = self.bootstrap()
        home = bootstrap["home"]
        existing = bootstrap["existing_service_management"]
        if existing["mode"] == "user-fallback":
            self.run("systemctl --user stop sensetrace.service 2>/dev/null || true", warn=True)
        elif existing["mode"] == "system":
            self.run("sudo -n systemctl stop sensetrace.service 2>/dev/null || true", warn=True)
        archive = self._archive_project()
        remote_archive = f"/tmp/sensetrace-source-{os.getpid()}.tar.gz"
        try:
            self.connection.put(str(archive), remote=remote_archive)
            source = f"{home}/.local/share/sensetrace/source"
            self.run(
                f"rm -rf {source} && mkdir -p {source} && tar -xzf {remote_archive} -C {source}"
            )
            self.run(f"printf '%s\\n' {self._local_commit()} > {source}/.sensetrace-commit")
            venv = f"{home}/.local/share/sensetrace/venv"
            self.run(
                f"python3 -m venv {venv} && {venv}/bin/python -m pip install --upgrade pip && {venv}/bin/pip install -e '{source}'",
                warn=False,
            )
            native_build = self.run(f"make -C {source}/native", warn=True, hide=True)
            if bootstrap["mode"] == "system":
                self.run(
                    "sudo -n mkdir -p /opt/sensetrace/source /etc/sensetrace "
                    "/var/lib/sensetrace/{data,runs,state}"
                )
                self.run(f"sudo -n cp -a {source}/. /opt/sensetrace/source/")
                self.run(
                    "sudo -n python3 -m venv /opt/sensetrace/venv && sudo -n /opt/sensetrace/venv/bin/pip install -e '/opt/sensetrace/source'"
                )
                # A system install is authoritative. Remove a user fallback
                # before enabling it, and make the installer preserve live
                # operator configuration unless reset_config was explicit.
                self.run("systemctl --user stop sensetrace.service 2>/dev/null || true", warn=True)
                self.run(
                    "systemctl --user disable sensetrace.service 2>/dev/null || true", warn=True
                )
                fallback_after = self.service_management()
                if (
                    fallback_after["user"]["active"] == "active"
                    or fallback_after["user"]["enabled"] == "enabled"
                ):
                    raise RuntimeError(
                        "user fallback remained active/enabled before system migration"
                    )
                live_config = "/etc/sensetrace/worker03.yaml"
                fallback_config = f"{home}/.config/sensetrace/worker03.yaml"
                config_migration = self._preserve_fallback_config(
                    fallback_config=fallback_config,
                    system_config=live_config,
                    reset_config=reset_config,
                )
                reset = "1" if reset_config else "0"
                self.run(
                    "sudo -n bash /opt/sensetrace/source/infra/worker03/install-root.sh "
                    f"/opt/sensetrace/source {reset}"
                )
                after = self._remote_hash(live_config)
                management = self.service_management()
                if management["mode"] == "invalid-dual-service":
                    raise RuntimeError("system deployment left an invalid dual-service state")
                process_count = self._runner_process_count()
                if management["mode"] != "system" or process_count != 1:
                    raise RuntimeError(
                        "system deployment did not establish exactly one authoritative runner"
                    )
                return {
                    "mode": "system",
                    "service": "sensetrace.service",
                    "source": "/opt/sensetrace/source",
                    "service_management": management,
                    "runner_process_count": process_count,
                    "config_path": live_config,
                    "config_sha256_before": config_migration["before"],
                    "config_sha256_after": after,
                    "config_preserved": (
                        config_migration["before"] != "missing"
                        and config_migration["before"] == after
                    ),
                    "config_source_before_migration": config_migration["source"],
                    "system_config_sha256_before": config_migration["system_before"],
                    "config_reset_requested": reset_config,
                    "native_kernel_build": native_build.ok,
                    "native_kernel_build_error": native_build.stderr.strip()
                    if not native_build.ok
                    else None,
                }
            user_unit = f"{home}/.config/systemd/user/sensetrace.service"
            live_config = f"{home}/.config/sensetrace/worker03.yaml"
            before = self._remote_hash(live_config)
            if before == "missing" or reset_config:
                self.run(f"cp {quote(source)}/configs/worker03.example.yaml {quote(live_config)}")
            after = self._remote_hash(live_config)
            self.run(
                f"mkdir -p {home}/.config/systemd/user && cp {source}/infra/worker03/sensetrace-user.service {user_unit}"
            )
            user_result = self.run(
                "systemctl --user daemon-reload && "
                "(systemctl --user restart sensetrace.service || systemctl --user start sensetrace.service)",
                warn=True,
            )
            return {
                "mode": "user-fallback",
                "service": "sensetrace.service (user)",
                "source": source,
                "service_started": user_result.ok,
                "service_error": user_result.stderr.strip() if not user_result.ok else None,
                "reboot_persistence": "not proven: user manager linger is disabled",
                "service_management": self.service_management(),
                "config_path": live_config,
                "config_sha256_before": before,
                "config_sha256_after": after,
                "config_preserved": before != "missing" and before == after,
                "config_reset_requested": reset_config,
                "native_kernel_build": native_build.ok,
                "native_kernel_build_error": native_build.stderr.strip()
                if not native_build.ok
                else None,
            }
        finally:
            self.run(f"rm -f {remote_archive}", warn=True, hide=True)

    def _remote_hash(self, path: str) -> str:
        result = self.run(f"sha256sum {quote(path)}", warn=True, hide=True)
        if not result.ok:
            return "missing"
        return result.stdout.split()[0]

    def _preserve_fallback_config(
        self,
        *,
        fallback_config: str,
        system_config: str,
        reset_config: bool,
    ) -> dict[str, str]:
        """Carry the live user config into a first system installation."""

        system_before = self._remote_hash(system_config)
        fallback_before = self._remote_hash(fallback_config)
        if not reset_config and system_before == "missing" and fallback_before != "missing":
            self.run(
                "sudo -n install -m 0640 -o worker-03 -g worker-03 "
                f"{quote(fallback_config)} {quote(system_config)}"
            )
            return {
                "before": fallback_before,
                "source": "user-fallback",
                "system_before": system_before,
            }
        return {
            "before": system_before,
            "source": "system" if system_before != "missing" else "missing",
            "system_before": system_before,
        }

    def _runner_process_count(self) -> int:
        result = self.run("pgrep -f '[s]ensetrace runner' | wc -l", warn=True, hide=True)
        try:
            return int(result.stdout.strip())
        except ValueError:
            return 0

    def _runner_pids(self) -> list[int]:
        result = self.run("pgrep -f '[s]ensetrace runner' || true", warn=True, hide=True)
        pids: list[int] = []
        for value in result.stdout.split():
            try:
                pids.append(int(value))
            except ValueError:
                continue
        return pids

    def status(self) -> dict[str, Any]:
        management = self.service_management()
        disk = self.run("df -h /", warn=True, hide=True)
        return {
            "host": self.alias,
            "management_mode": management["mode"],
            "service_management": management,
            "runner_process_count": self._runner_process_count(),
            "runner_pids": self._runner_pids(),
            "disk_root": disk.stdout.strip(),
        }

    def boot_profile(self) -> dict[str, Any]:
        default = self.run("systemctl get-default", warn=True, hide=True)
        active = self.run(
            "systemctl list-units --type=target --state=active --no-legend --no-pager",
            warn=True,
            hide=True,
        )
        candidate = self.run(
            "systemctl list-unit-files sensetrace.target --no-legend --no-pager",
            warn=True,
            hide=True,
        )
        display = self.run("systemctl is-active display-manager.service", warn=True, hide=True)
        return {
            "host": self.alias,
            "default_target": default.stdout.strip() or "unavailable",
            "active_targets": active.stdout.splitlines(),
            "sensetrace_target": candidate.stdout.strip() or "not-installed",
            "display_manager_active": display.stdout.strip() or "inactive",
            "reversible": True,
            "claim": "profile inventory only; no default-target change was made",
        }

    def watchdog_profile(self) -> dict[str, Any]:
        commands = {
            "devices": "ls -l /dev/watchdog* 2>/dev/null || true",
            "drivers": 'for p in /sys/class/watchdog/watchdog*/device/driver; do printf \'%s \' "$p"; readlink -f "$p" 2>/dev/null || true; done',
            "timeouts": 'for p in /sys/class/watchdog/watchdog*; do printf \'%s \' "$p"; cat "$p"/timeout 2>/dev/null || true; done',
            "nowayout": 'for p in /sys/class/watchdog/watchdog*; do printf \'%s \' "$p"; cat "$p"/nowayout 2>/dev/null || true; done',
            "systemd_watchdog": "systemctl show -p RuntimeWatchdogUSec -p RebootWatchdogUSec --value",
            "loaded_modules": "lsmod | grep -E '(^| )(intel_oc_wdt|iTCO_wdt)( |$)' || true",
        }
        return {
            "host": self.alias,
            "timestamp_epoch": time.time(),
            "measurements": {
                name: self.run(command, warn=True, hide=True).stdout
                for name, command in commands.items()
            },
            "decision": "inventory only; do not enable both watchdog drivers without hardware-specific validation",
        }

    def services(self) -> dict[str, Any]:
        running = self.run(
            "systemctl list-units --type=service --state=running --no-legend --no-pager",
            warn=True,
            hide=True,
        )
        return {
            "host": self.alias,
            "running_services": running.stdout.splitlines(),
            "running_service_count": len(running.stdout.splitlines()),
            "runner_process_count": self._runner_process_count(),
        }

    def noise_baseline(self) -> dict[str, Any]:
        commands = {
            "load_average": "cat /proc/loadavg",
            "memory": "free -b",
            "vmstat": "vmstat 1 2",
            "processes": "ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -40",
            "network": "ip -s link",
            "storage": "iostat -dx 1 2 2>/dev/null || true",
            "cpu_frequency": "grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || true",
            "temperatures": "sensors 2>/dev/null || true",
        }
        return {
            "host": self.alias,
            "timestamp_epoch": time.time(),
            "measurements": {
                name: self.run(command, warn=True, hide=True).stdout
                for name, command in commands.items()
            },
            "interpretation": "baseline evidence; fewer services alone do not establish lower measurement noise",
        }

    def cpu_profile(self) -> dict[str, Any]:
        commands = {
            "frequency_info": "cpupower frequency-info 2>/dev/null || true",
            "governors": "grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true",
            "current_frequency": "grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || true",
            "available_governors": "grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_available_governors 2>/dev/null || true",
            "turbo": "cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true",
            "turbostat_available": "command -v turbostat || true",
            "temperature": "sensors 2>/dev/null || true",
        }
        return {
            "host": self.alias,
            "timestamp_epoch": time.time(),
            "measurements": {
                name: self.run(command, warn=True, hide=True).stdout
                for name, command in commands.items()
            },
            "claim": "policy inventory; compare separate controlled runs before selecting a timing profile",
        }

    def verify_boot(
        self,
        *,
        expected_boot_id: str | None = None,
        require_appliance: bool = False,
    ) -> dict[str, Any]:
        boot_id = self.run("cat /proc/sys/kernel/random/boot_id", warn=True, hide=True)
        profile = self.boot_profile()
        service = self.status()
        doctor = self.doctor()
        services = self.services()
        home = self._home()
        recovery = self.run(
            f"grep 'journal_recovery' {quote(home)}/.local/share/sensetrace/runs/*/events.jsonl "
            "2>/dev/null | tail -1 || true",
            warn=True,
            hide=True,
        )
        result = {
            "host": self.alias,
            "ssh": True,
            "boot_id": boot_id.stdout.strip() or "unavailable",
            "expected_boot_id": expected_boot_id,
            "boot_id_changed": (
                None if expected_boot_id is None else boot_id.stdout.strip() != expected_boot_id
            ),
            "boot_profile": profile,
            "service": service,
            "doctor": doctor,
            "services": services,
            "journal_recovery_evidence": recovery.stdout.strip() or "not observed",
            "acceptance_profile": "dedicated-appliance" if require_appliance else "runner",
        }
        runner_checks = bool(
            result["ssh"]
            and result["boot_id"] != "unavailable"
            and service["management_mode"] in {"system", "user-fallback"}
            and service["runner_process_count"] == 1
            and doctor["management_mode"] != "invalid-dual-service"
        )
        appliance_checks = bool(
            profile["default_target"] == "sensetrace.target"
            and profile["display_manager_active"] != "active"
            and service["management_mode"] == "system"
            and any("sensetrace.target" in target for target in profile["active_targets"])
        )
        result["passed"] = runner_checks and (appliance_checks if require_appliance else True)
        result["appliance_checks"] = {
            "default_target": profile["default_target"],
            "display_manager_inactive": profile["display_manager_active"] != "active",
            "system_service_authoritative": service["management_mode"] == "system",
            "dedicated_target_active": any(
                "sensetrace.target" in target for target in profile["active_targets"]
            ),
        }
        return result

    def reboot_acceptance(
        self,
        *,
        cycles: int = 3,
        timeout_seconds: int = 180,
        require_appliance: bool = False,
    ) -> dict[str, Any]:
        """Perform repeated fresh-SSH reboot checks for the selected boot stage."""

        if cycles < 1:
            raise ValueError("cycles must be positive")
        records: list[dict[str, Any]] = []
        current = self
        for cycle in range(1, cycles + 1):
            old_boot_id = current.run(
                "cat /proc/sys/kernel/random/boot_id", hide=True
            ).stdout.strip()
            request_timestamp = time.time()
            request = current.reboot()
            current.connection.close()
            deadline = time.monotonic() + timeout_seconds
            fresh: RemoteHost | None = None
            verification: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                try:
                    candidate = RemoteHost(self.alias, project_root=self.project_root)
                    verification = candidate.verify_boot(
                        expected_boot_id=old_boot_id, require_appliance=require_appliance
                    )
                    if verification["passed"] and verification["boot_id_changed"]:
                        fresh = candidate
                        break
                    candidate.connection.close()
                except Exception:
                    pass
                time.sleep(5)
            records.append(
                {
                    "cycle": cycle,
                    "old_boot_id": old_boot_id,
                    "request_timestamp_epoch": request_timestamp,
                    "request": request,
                    "fresh_ssh_verification": verification,
                    "passed": fresh is not None,
                }
            )
            if fresh is None:
                break
            current = fresh
        self.connection = current.connection
        return {
            "host": self.alias,
            "cycles_requested": cycles,
            "cycles_completed": len(records),
            "require_appliance": require_appliance,
            "records": records,
            "passed": len(records) == cycles and all(record["passed"] for record in records),
        }

    def set_headless(self) -> dict[str, Any]:
        """Switch the default to multi-user.target after SSH is already verified."""

        if not self.sudo_available():
            raise RuntimeError("headless transition requires existing authorized sudo")
        before = self.boot_profile()
        fresh_connection = RemoteHost(self.alias, project_root=self.project_root)
        try:
            fresh_ssh = fresh_connection.run("true", hide=True).ok
        finally:
            fresh_connection.connection.close()
        if not fresh_ssh:
            raise RuntimeError("fresh SSH validation failed before headless transition")
        self.run("sudo -n systemctl set-default multi-user.target")
        # Fedora systems can have no display-manager alias at all once the
        # graphical stack is inactive.  Stopping/disabling that absent alias
        # is not a failed headless transition; verify the resulting profile.
        self.run("sudo -n systemctl disable display-manager.service", warn=True, hide=True)
        self.run("sudo -n systemctl stop display-manager.service", warn=True, hide=True)
        after = self.boot_profile()
        if (
            after["default_target"] != "multi-user.target"
            or after["display_manager_active"] == "active"
        ):
            raise RuntimeError("headless transition did not verify a stopped display manager")
        return {
            "before": before,
            "after": after,
            "fresh_ssh_before_transition": fresh_ssh,
            "reversible": "systemctl set-default graphical.target",
        }

    def isolate_sensetrace_target(self) -> dict[str, Any]:
        if not self.sudo_available():
            raise RuntimeError("target isolation requires existing authorized sudo")
        self.run("sudo -n systemctl isolate sensetrace.target")
        result = self.verify_boot()
        if not any(
            "sensetrace.target" in target for target in result["boot_profile"]["active_targets"]
        ):
            result["passed"] = False
            result["error"] = "sensetrace.target is not active after isolation"
        return result

    def set_sensetrace_default(self) -> dict[str, Any]:
        if not self.sudo_available():
            raise RuntimeError("dedicated target transition requires existing authorized sudo")
        before = self.boot_profile()
        if (
            before["default_target"] != "multi-user.target"
            or before["display_manager_active"] == "active"
        ):
            raise RuntimeError(
                "dedicated target requires a proven multi-user headless baseline first"
            )
        self.run("sudo -n systemctl set-default sensetrace.target")
        after = self.boot_profile()
        if after["default_target"] != "sensetrace.target":
            raise RuntimeError("dedicated target was not made default")
        return {
            "before": before,
            "after": after,
            "reversible": "systemctl set-default multi-user.target",
        }

    def authorize_sudo(self) -> dict[str, Any]:
        """Prompt through the controlling terminal without capturing credentials."""

        result = self.connection.run("sudo -v", pty=True, warn=True)
        return {
            "authorized": result.ok and self.sudo_available(),
            "error": result.stderr.strip() if not result.ok else None,
            "password_stored": False,
        }

    def service_action(self, action: str) -> dict[str, Any]:
        if action not in {"start", "stop", "restart"}:
            raise ValueError(f"unsupported service action: {action}")
        management = self.service_management()
        if management["mode"] == "invalid-dual-service":
            raise RuntimeError("refusing service action in invalid-dual-service mode")
        if management["mode"] == "not-installed":
            raise RuntimeError("SenseTrace runner service is not installed")
        if management["mode"] == "system":
            result = self.run(
                f"sudo -n systemctl {action} sensetrace.service", warn=True, hide=True
            )
            scope = "system"
        else:
            result = self.run(f"systemctl --user {action} sensetrace.service", warn=True, hide=True)
            scope = "user-fallback"
        return {
            "scope": scope,
            "action": action,
            "ok": result.ok,
            "error": result.stderr.strip() if not result.ok else None,
        }

    def logs(self, *, lines: int = 100) -> str:
        management = self.service_management()
        if management["mode"] == "invalid-dual-service":
            raise RuntimeError("refusing logs in invalid-dual-service mode")
        if management["mode"] == "system":
            command = f"journalctl -u sensetrace.service -n {int(lines)} --no-pager"
        elif management["mode"] == "user-fallback":
            command = (
                f"journalctl --user -u sensetrace.service -n {int(lines)} --no-pager 2>/dev/null; "
                f"tail -n {int(lines)} ~/.local/share/sensetrace/state/service-events.jsonl 2>/dev/null"
            )
        else:
            return "runner service is not installed\n"
        return self.run(command, warn=True).stdout

    def run_phase0(
        self,
        config: str | Path,
        *,
        output: str | None = None,
        include_curve: bool = False,
        samples: int | None = None,
        trace_length: int | None = None,
        seed: int | None = None,
        models: list[str] | None = None,
    ) -> str:
        home = self._home()
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        remote_config = f"{home}/.config/sensetrace/phase0.yaml"
        self.connection.put(str(config), remote=remote_config)
        destination = output or f"{home}/.local/share/sensetrace/runs"
        command = f"{venv} run phase0 --config {remote_config} --output {destination}"
        if include_curve:
            command += " --curve"
        if samples is not None:
            command += f" --samples {int(samples)}"
        if trace_length is not None:
            command += f" --trace-length {int(trace_length)}"
        if seed is not None:
            command += f" --seed {int(seed)}"
        if models:
            command += " --models " + " ".join(models)
        result = self.run(command, warn=True, hide=True)
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def run_phase0_calibration(
        self,
        config: str | Path,
        *,
        output: str | None = None,
        null_replicates: int | None = None,
        shuffled_replicates: int | None = None,
        injected_replicates: int | None = None,
        gate_validation_replicates: int | None = None,
    ) -> str:
        home = self._home()
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        remote_config = f"{home}/.config/sensetrace/phase0-calibration.yaml"
        self.connection.put(str(config), remote=remote_config)
        destination = output or f"{home}/.local/share/sensetrace/runs"
        command = (
            f"{venv} calibrate phase0 --config {quote(remote_config)} --output {quote(destination)}"
        )
        for flag, value in [
            ("--null-replicates", null_replicates),
            ("--shuffled-replicates", shuffled_replicates),
            ("--injected-replicates", injected_replicates),
            ("--gate-validation-replicates", gate_validation_replicates),
        ]:
            if value is not None:
                command += f" {flag} {int(value)}"
        result = self.run(command, warn=True, hide=True)
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def run_phase1a(
        self,
        config: str | Path,
        phase0_report: str | Path,
        *,
        output: str | None = None,
    ) -> str:
        home = self._home()
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        remote_config = f"{home}/.config/sensetrace/phase1a.yaml"
        remote_gate = f"{home}/.config/sensetrace/phase0-gate.json"
        self.connection.put(str(config), remote=remote_config)
        self.connection.put(str(phase0_report), remote=remote_gate)
        destination = output or f"{home}/.local/share/sensetrace/runs"
        result = self.run(
            f"{venv} run phase1a --config {quote(remote_config)} --phase0-report "
            f"{quote(remote_gate)} --output {quote(destination)}",
            warn=True,
            hide=True,
        )
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def run_native_sensitivity_calibration(
        self,
        config: str | Path,
        *,
        output: str | None = None,
        development_magnitudes: list[int] | None = None,
        development_replicates: int | None = None,
        validation_replicates: int | None = None,
    ) -> str:
        """Run the separately namespaced native-path calibration on the worker."""

        home = self._home()
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        remote_config = f"{home}/.config/sensetrace/native-sensitivity.yaml"
        self.connection.put(str(config), remote=remote_config)
        destination = output or f"{home}/.local/share/sensetrace/runs/native-sensitivity"
        command = (
            f"{venv} calibrate native-sensitivity --config {quote(remote_config)} "
            f"--output {quote(destination)}"
        )
        if development_magnitudes:
            command += " --development-magnitudes " + " ".join(
                str(int(value)) for value in development_magnitudes
            )
        if development_replicates is not None:
            command += f" --development-replicates {int(development_replicates)}"
        if validation_replicates is not None:
            command += f" --validation-replicates {int(validation_replicates)}"
        result = self.run(command, warn=True, hide=True)
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def characterize_primitive(
        self,
        config: str | Path,
        *,
        output: str | None = None,
    ) -> str:
        """Run the non-inference primitive characterization on the worker."""

        home = self._home()
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        remote_config = f"{home}/.config/sensetrace/primitive-characterization.yaml"
        self.connection.put(str(config), remote=remote_config)
        destination = output or f"{home}/.local/share/sensetrace/runs/primitive-characterization"
        result = self.run(
            f"{venv} characterize primitive --config {quote(remote_config)} "
            f"--output {quote(destination)}",
            warn=True,
            hide=True,
        )
        if not result.ok:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def characterize_multiboot(
        self,
        config: str | Path,
        *,
        output: str | Path,
        boots: int = 3,
        timeout_seconds: int = 300,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Repeat one frozen characterization across genuinely distinct boots.

        Each iteration verifies the actual ``/proc/.../boot_id``, runs the
        frozen scoped-PMU characterization, fetches the evidence, then reboots
        (except after the final boot). Reused boot IDs fail closed; the local
        manifest is written after every boot so an interruption can resume with
        ``resume=True`` instead of fabricating reboot evidence.
        """

        import hashlib

        from ..multiboot import combine_multiboot_reports, multiboot_protocol_hash

        if boots < 3:
            raise ValueError("multi-boot characterization requires at least three boots")
        local_root = Path(output)
        manifest_path = local_root / "multiboot-manifest.json"
        try:
            config_text = Path(config).read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot read multiboot config: {exc}") from exc
        config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        parsed_config = __import__("yaml").safe_load(config_text)
        frozen_boots = int(parsed_config.get("characterization", {}).get("multiboot_boots", 0))
        if boots != frozen_boots:
            raise ValueError(
                f"CLI boot count {boots} disagrees with frozen protocol boot count {frozen_boots}"
            )
        frozen = multiboot_protocol_hash(parsed_config)
        entries: list[dict[str, Any]] = []
        seen_boot_ids: set[str] = set()
        if manifest_path.exists():
            if not resume:
                raise RuntimeError(
                    f"multiboot manifest already exists at {manifest_path}; "
                    "pass resume=True to continue it or use a fresh output directory"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_sha256") != config_hash:
                raise RuntimeError("resume requested with a different config; refusing")
            for entry in manifest.get("boots", []):
                entries.append(entry)
                if entry.get("boot_id"):
                    seen_boot_ids.add(entry["boot_id"])
        home = self._home()
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        remote_config = f"{home}/.config/sensetrace/primitive-characterization.yaml"
        self.connection.put(str(config), remote=remote_config)
        current: RemoteHost = self
        start_index = len(entries)
        for boot_index in range(start_index, boots):
            boot_id = current.run("cat /proc/sys/kernel/random/boot_id", hide=True).stdout.strip()
            if not boot_id or boot_id == "unavailable":
                raise RuntimeError("worker boot_id is unavailable; refusing to label evidence")
            if boot_id in seen_boot_ids:
                raise RuntimeError(
                    f"worker is still on already-recorded boot {boot_id}; "
                    "a genuine reboot is required, not a relabeled run"
                )
            remote_output = f"{home}/.local/share/sensetrace/runs/multiboot-boot-{boot_index:02d}"
            result = current.run(
                f"{venv} characterize primitive --config {quote(remote_config)} "
                f"--output {quote(remote_output)}",
                warn=True,
                hide=True,
            )
            if not result.ok:
                raise RuntimeError(result.stderr or result.stdout)
            try:
                report = json.loads(result.stdout)
                run_id = str(report.get("run_id", ""))
            except (json.JSONDecodeError, AttributeError):
                run_id = ""
            if not run_id:
                located = current.run(
                    f"find {quote(remote_output)} -maxdepth 2 -name metrics.json -type f "
                    "-printf '%T@ %p\\n' | sort -n | tail -1",
                    warn=True,
                    hide=True,
                )
                if not located.stdout.strip():
                    raise RuntimeError("remote characterization produced no metrics.json")
                run_id = located.stdout.strip().split(maxsplit=1)[1].rsplit("/", 2)[-2]
            remote_run_dir = f"{remote_output}/{run_id}"
            local_boot_dir = local_root / f"boot-{boot_index:02d}"
            local_boot_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "run.json",
                "host.json",
                "config.json",
                "protocol.json",
                "metrics.json",
                "events.jsonl",
            ):
                remote = f"{remote_run_dir}/{name}"
                if current.run(f"test -f {remote}", warn=True, hide=True).ok:
                    current.connection.get(remote, local=str(local_boot_dir / name))
            hashes = {}
            for name in (
                "run.json",
                "host.json",
                "config.json",
                "protocol.json",
                "metrics.json",
                "events.jsonl",
            ):
                path = local_boot_dir / name
                if path.exists():
                    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if not (local_boot_dir / "metrics.json").exists():
                raise RuntimeError(f"boot {boot_index}: metrics.json fetch failed")
            fetched_boot = boot_id
            try:
                fetched_metrics = json.loads(
                    (local_boot_dir / "metrics.json").read_text(encoding="utf-8")
                )
                controls = fetched_metrics.get("controls", {})
                unique = controls.get("boot_dependence", {}).get("unique_boots", [])
                if len(unique) == 1 and unique[0] not in {"", "unavailable", "unknown"}:
                    if unique[0] != boot_id:
                        raise RuntimeError(
                            f"boot {boot_index}: SSH boot_id {boot_id} disagrees with "
                            f"metrics boot {unique[0]}"
                        )
                    fetched_boot = str(unique[0])
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"boot {boot_index}: metrics.json is malformed: {exc}") from exc
            entries.append(
                {
                    "boot_index": boot_index,
                    "boot_id": fetched_boot,
                    "run_id": run_id,
                    "remote_run_dir": remote_run_dir,
                    "local_dir": str(local_boot_dir),
                    "artifact_sha256": hashes,
                    "protocol_hash": fetched_metrics.get("protocol", {}).get(
                        "protocol_hash", "unavailable"
                    ),
                }
            )
            seen_boot_ids.add(fetched_boot)
            local_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "sensetrace.multiboot-manifest.v1",
                        "config": str(config),
                        "config_sha256": config_hash,
                        "multiboot_protocol_hint": frozen,
                        "boots_requested": boots,
                        "boots_completed": len(entries),
                        "boots": entries,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if boot_index < boots - 1:
                reboot_request = current.reboot()
                entries[-1]["reboot_request"] = reboot_request
                current.connection.close()
                deadline = time.monotonic() + timeout_seconds
                fresh: RemoteHost | None = None
                verification: dict[str, Any] | None = None
                while time.monotonic() < deadline:
                    try:
                        candidate = RemoteHost(self.alias, project_root=self.project_root)
                        verification = candidate.verify_boot(expected_boot_id=boot_id)
                        if verification["passed"] and verification["boot_id_changed"]:
                            fresh = candidate
                            break
                        candidate.connection.close()
                    except Exception:
                        pass
                    time.sleep(5)
                if fresh is None:
                    raise RuntimeError(
                        f"reboot after boot {boot_index} did not verify a new boot_id "
                        f"(last verification: {verification})"
                    )
                new_boot = fresh.run(
                    "cat /proc/sys/kernel/random/boot_id", hide=True
                ).stdout.strip()
                if new_boot in seen_boot_ids:
                    raise RuntimeError(
                        f"post-reboot boot_id {new_boot} repeats a recorded boot; refusing"
                    )
                current = fresh
        self.connection = current.connection
        reports = [
            json.loads((local_root / f"boot-{index:02d}" / "metrics.json").read_text())
            for index in range(boots)
        ]
        combined = combine_multiboot_reports(reports, expected_boots=boots)
        (local_root / "multiboot-report.json").write_text(
            json.dumps(combined, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return {
            "local_root": str(local_root),
            "boots_completed": len(entries),
            "boots": entries,
            "combined": combined,
        }

    def verify_recovery(self) -> dict[str, Any]:
        home = self._home()
        output = f"{home}/.local/share/sensetrace/runs/recovery-check"
        venv = f"{home}/.local/share/sensetrace/venv/bin/sensetrace"
        result = self.run(
            f"rm -rf {output} && {venv} selftest recovery --output {output}", warn=True
        )
        service = self.status()
        if not result.ok:
            return {"passed": False, "error": result.stderr or result.stdout, "service": service}
        process_restart = self.verify_process_restart()
        return {
            "selftest": json.loads(result.stdout),
            "process_restart": process_restart,
            "service": service,
            "passed": json.loads(result.stdout).get("passed", False) and process_restart["passed"],
        }

    def verify_process_restart(self) -> dict[str, Any]:
        """Kill only the authoritative runner process and prove its service restarts it."""
        management = self.service_management()
        if management["mode"] == "invalid-dual-service":
            return {"passed": False, "error": "invalid-dual-service"}
        if management["mode"] == "system":
            scope = "sudo -n systemctl"
        elif management["mode"] == "user-fallback":
            scope = "systemctl --user"
        else:
            return {"passed": False, "error": "runner service is not installed"}
        command = (
            "set -eu; "
            "unit=sensetrace.service; scope='" + scope + "'; "
            "state=$($scope is-active $unit 2>/dev/null || true); "
            '[ "$state" = active ]; '
            'old=$($scope show $unit -p MainPID --value); kill -KILL "$old"; '
            "new=''; for i in $(seq 1 20); do sleep 1; new=$($scope show $unit -p MainPID --value); "
            '[ "$new" != 0 ] && [ "$new" != "$old" ] && break; done; '
            "state=$($scope is-active $unit); restarts=$($scope show $unit -p NRestarts --value); "
            'printf \'{"old_pid":%s,"new_pid":%s,"state":"%s","restarts":%s}\n\' "$old" "$new" "$state" "$restarts"'
        )
        result = self.run(command, warn=True, hide=True)
        if not result.ok:
            return {"passed": False, "error": result.stderr.strip() or result.stdout.strip()}
        evidence = json.loads(result.stdout)
        evidence["passed"] = (
            evidence["state"] == "active" and evidence["new_pid"] != evidence["old_pid"]
        )
        return evidence

    def latest_result(self) -> dict[str, Any]:
        home = self._home()
        result = self.run(
            f"find {home}/.local/share/sensetrace/runs -name metrics.json -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1",
            warn=True,
            hide=True,
        )
        if not result.stdout.strip():
            return {"status": "no-results"}
        path = result.stdout.strip().split(maxsplit=1)[1]
        fetched = self.run(f"python3 -c 'import json; print(open({path!r}).read())'", warn=True)
        return (
            json.loads(fetched.stdout)
            if fetched.ok
            else {"path": path, "error": fetched.stderr.strip()}
        )

    def fetch_results(self, destination: str | Path, *, run_id: str | None = None) -> str:
        home = self._home()
        if run_id:
            remote_dir = f"{home}/.local/share/sensetrace/runs/{run_id}"
        else:
            latest = self.run(
                f"find {home}/.local/share/sensetrace/runs -maxdepth 2 -name metrics.json -type f -printf '%T@ %p\\n' | sort -n | tail -1",
                warn=True,
                hide=True,
            )
            if not latest.stdout.strip():
                raise RuntimeError("no result artifacts found")
            remote_dir = latest.stdout.strip().split(maxsplit=1)[1].rsplit("/", 1)[0]
        local_dir = Path(destination)
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "run.json",
            "host.json",
            "config.json",
            "config.yaml",
            "environment.json",
            "metrics.json",
            "events.jsonl",
        ]:
            remote = f"{remote_dir}/{name}"
            if self.run(f"test -f {remote}", warn=True, hide=True).ok:
                self.connection.get(remote, local=str(local_dir / name))
        return str(local_dir)

    def reboot(self) -> str:
        # A transient systemd timer detaches reboot scheduling from the SSH
        # channel.  Shell-backgrounding a one-second delay can still reset the
        # transport before Fabric receives the request result.
        result = self.run(
            "boot_id=$(cat /proc/sys/kernel/random/boot_id); "
            "sudo -n true && "
            "sudo -n systemd-run --quiet --collect --on-active=5s "
            "/usr/bin/systemctl reboot && "
            'printf \'{"old_boot_id":"%s","request_epoch":%.6f}\n\' '
            '"$boot_id" "$(date +%s.%N)"',
            warn=True,
        )
        if result.ok:
            return result.stdout.strip() or "reboot requested"
        raise RuntimeError(
            result.stderr.strip() or "remote reboot requires privileged sudo authorization"
        )
