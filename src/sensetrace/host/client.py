"""One Fabric-backed controller interface for worker operations."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
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
        checks["management_mode"] = "system" if checks["sudo_noninteractive"] else "user-fallback"
        checks["warning"] = (
            None
            if checks["sudo_noninteractive"]
            else "privileged provisioning requires an interactive sudo authorization"
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
        mode = "system" if self.sudo_available() else "user-fallback"
        self.run(
            f"mkdir -p {home}/.local/share/sensetrace/{{source,runs,state,logs}} {home}/.config/sensetrace",
            hide=True,
        )
        return {"mode": mode, "home": home, "sudo_noninteractive": mode == "system"}

    def deploy(self) -> dict[str, Any]:
        bootstrap = self.bootstrap()
        home = bootstrap["home"]
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
            self.run(
                f"cp {source}/configs/worker03.example.yaml {home}/.config/sensetrace/worker03.yaml"
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
                self.run(
                    "sudo -n bash /opt/sensetrace/source/infra/worker03/install-root.sh "
                    "/opt/sensetrace/source"
                )
                return {
                    "mode": "system",
                    "service": "sensetrace.service",
                    "source": "/opt/sensetrace/source",
                }
            user_unit = f"{home}/.config/systemd/user/sensetrace.service"
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
            }
        finally:
            self.run(f"rm -f {remote_archive}", warn=True, hide=True)

    def status(self) -> dict[str, Any]:
        system = self.run("systemctl is-active sensetrace.service", warn=True, hide=True)
        user = self.run("systemctl --user is-active sensetrace.service", warn=True, hide=True)
        disk = self.run("df -h /", warn=True, hide=True)
        return {
            "host": self.alias,
            "system_service": system.stdout.strip()
            if system.ok
            else system.stderr.strip() or system.stdout.strip() or "inactive",
            "user_service": user.stdout.strip()
            if user.ok
            else user.stderr.strip() or user.stdout.strip() or "inactive",
            "disk_root": disk.stdout.strip(),
        }

    def service_action(self, action: str) -> dict[str, Any]:
        if action not in {"start", "stop", "restart"}:
            raise ValueError(f"unsupported service action: {action}")
        user = self.run(f"systemctl --user {action} sensetrace.service", warn=True, hide=True)
        if user.ok:
            return {"scope": "user", "action": action, "ok": True}
        if self.sudo_available():
            system = self.run(
                f"sudo -n systemctl {action} sensetrace.service", warn=True, hide=True
            )
            return {
                "scope": "system",
                "action": action,
                "ok": system.ok,
                "error": system.stderr.strip() if not system.ok else None,
            }
        return {
            "scope": "user",
            "action": action,
            "ok": False,
            "error": user.stderr.strip(),
        }

    def logs(self, *, lines: int = 100) -> str:
        command = (
            f"journalctl -u sensetrace.service -n {int(lines)} --no-pager 2>/dev/null; "
            f"journalctl --user -u sensetrace.service -n {int(lines)} --no-pager 2>/dev/null; "
            f"tail -n {int(lines)} ~/.local/share/sensetrace/state/service-events.jsonl 2>/dev/null"
        )
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
        """Kill only the runner process and prove the user unit restarts it."""
        command = (
            "set -eu; "
            "unit=sensetrace.service; scope='systemctl --user'; "
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
        result = self.run("sudo -n systemctl reboot", warn=True)
        if result.ok:
            return "reboot requested"
        raise RuntimeError(
            result.stderr.strip() or "remote reboot requires privileged sudo authorization"
        )
