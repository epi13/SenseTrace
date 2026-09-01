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
                if path.is_file():
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
                before = self._remote_hash(live_config)
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
                    "config_sha256_before": before,
                    "config_sha256_after": after,
                    "config_preserved": before != "missing" and before == after,
                    "config_reset_requested": reset_config,
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
            }
        finally:
            self.run(f"rm -f {remote_archive}", warn=True, hide=True)

    def _remote_hash(self, path: str) -> str:
        result = self.run(f"sha256sum {quote(path)}", warn=True, hide=True)
        if not result.ok:
            return "missing"
        return result.stdout.split()[0]

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

    def verify_boot(self, *, expected_boot_id: str | None = None) -> dict[str, Any]:
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
        }
        result["passed"] = bool(
            result["ssh"]
            and result["boot_id"] != "unavailable"
            and service["management_mode"] in {"system", "user-fallback"}
            and service["runner_process_count"] == 1
            and doctor["management_mode"] != "invalid-dual-service"
        )
        return result

    def reboot_acceptance(self, *, cycles: int = 3, timeout_seconds: int = 180) -> dict[str, Any]:
        """Perform repeated fresh-SSH reboot checks once sudo authorization exists."""

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
                    verification = candidate.verify_boot(expected_boot_id=old_boot_id)
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
            "records": records,
            "passed": len(records) == cycles and all(record["passed"] for record in records),
        }

    def set_headless(self) -> dict[str, Any]:
        """Switch the default to multi-user.target after SSH is already verified."""

        if not self.sudo_available():
            raise RuntimeError("headless transition requires existing authorized sudo")
        before = self.boot_profile()
        self.run("sudo -n systemctl set-default multi-user.target")
        display = self.run("sudo -n systemctl disable --now display-manager.service", warn=True)
        if not display.ok:
            raise RuntimeError(
                display.stderr.strip() or "could not disable display-manager.service"
            )
        after = self.boot_profile()
        if (
            after["default_target"] != "multi-user.target"
            or after["display_manager_active"] == "active"
        ):
            raise RuntimeError("headless transition did not verify a stopped display manager")
        return {
            "before": before,
            "after": after,
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
        self.run("sudo -n systemctl set-default sensetrace.target")
        after = self.boot_profile()
        if after["default_target"] != "sensetrace.target":
            raise RuntimeError("dedicated target was not made default")
        return {
            "before": before,
            "after": after,
            "reversible": "systemctl set-default multi-user.target",
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
        # Schedule the reboot after this SSH command returns so the controller
        # receives an unambiguous request record instead of a transport error.
        result = self.run(
            "boot_id=$(cat /proc/sys/kernel/random/boot_id); "
            "sudo -n true && "
            "(nohup sh -c 'sleep 1; exec sudo -n systemctl reboot' "
            ">/dev/null 2>&1 </dev/null &) && "
            'printf \'{"old_boot_id":"%s","request_epoch":%.6f}\n\' '
            '"$boot_id" "$(date +%s.%N)"',
            warn=True,
        )
        if result.ok:
            return result.stdout.strip() or "reboot requested"
        raise RuntimeError(
            result.stderr.strip() or "remote reboot requires privileged sudo authorization"
        )
