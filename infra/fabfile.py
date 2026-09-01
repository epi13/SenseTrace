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
]:
    worker03.add_task(globals()[_name], name=_name)
namespace = Collection(worker03)
