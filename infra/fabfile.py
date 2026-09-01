"""Fabric collection; the CLI and these tasks share the same RemoteHost client."""

from __future__ import annotations

from fabric import Connection, task
from invoke import Collection

from sensetrace.host.client import RemoteHost


def _remote(_c: Connection) -> RemoteHost:
    return RemoteHost("worker-03")


@task
def inventory(c: Connection) -> None:
    print(_remote(c).inventory())


@task
def doctor(c: Connection) -> None:
    print(_remote(c).doctor())


@task
def bootstrap(c: Connection) -> None:
    print(_remote(c).bootstrap())


@task
def deploy(c: Connection) -> None:
    print(_remote(c).deploy())


@task
def status(c: Connection) -> None:
    print(_remote(c).status())


@task
def start(c: Connection) -> None:
    print(_remote(c).service_action("start"))


@task
def stop(c: Connection) -> None:
    print(_remote(c).service_action("stop"))


@task
def restart(c: Connection) -> None:
    print(_remote(c).service_action("restart"))


@task
def logs(c: Connection, lines: int = 100) -> None:
    print(_remote(c).logs(lines=lines), end="")


@task
def verify_recovery(c: Connection) -> None:
    print(_remote(c).verify_recovery())


@task
def reboot(c: Connection) -> None:
    print(_remote(c).reboot())


@task
def reboot_acceptance(c: Connection, cycles: int = 3, timeout: int = 180) -> None:
    print(_remote(c).reboot_acceptance(cycles=cycles, timeout_seconds=timeout))


@task
def boot_profile(c: Connection) -> None:
    print(_remote(c).boot_profile())


@task
def services(c: Connection) -> None:
    print(_remote(c).services())


@task
def noise_baseline(c: Connection) -> None:
    print(_remote(c).noise_baseline())


@task
def cpu_profile(c: Connection) -> None:
    print(_remote(c).cpu_profile())


@task
def verify_boot(c: Connection) -> None:
    print(_remote(c).verify_boot())


@task
def headless(c: Connection) -> None:
    print(_remote(c).set_headless())


@task
def isolate_target(c: Connection) -> None:
    print(_remote(c).isolate_sensetrace_target())


@task
def set_target(c: Connection) -> None:
    print(_remote(c).set_sensetrace_default())


worker03 = Collection("worker03")
for _name in [
    "inventory",
    "doctor",
    "bootstrap",
    "deploy",
    "status",
    "start",
    "stop",
    "restart",
    "logs",
    "verify_recovery",
    "reboot",
    "reboot_acceptance",
    "boot_profile",
    "services",
    "noise_baseline",
    "cpu_profile",
    "verify_boot",
    "headless",
    "isolate_target",
    "set_target",
]:
    worker03.add_task(globals()[_name], name=_name)
namespace = Collection(worker03)
