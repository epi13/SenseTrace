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
def deploy(c: Connection, reset_config: bool = False) -> None:
    print(_remote(c).deploy(reset_config=reset_config))


@task
def status(c: Connection) -> None:
    print(_remote(c).status())


@task
def run_phase0(
    c: Connection,
    config: str = "configs/phase0.example.yaml",
    output: str | None = None,
    curve: bool = False,
    models: str = "",
) -> None:
    print(
        _remote(c).run_phase0(
            config,
            output=output,
            include_curve=curve,
            models=models.split() or None,
        )
    )


@task
def run_phase1a(
    c: Connection,
    config: str = "configs/worker03.example.yaml",
    phase0_report: str = "evidence/phase0/metrics.json",
    output: str | None = None,
) -> None:
    print(_remote(c).run_phase1a(config, phase0_report, output=output))


@task
def calibrate_phase0(
    c: Connection,
    config: str = "configs/phase0.example.yaml",
    output: str | None = None,
    null_replicates: int | None = None,
    shuffled_replicates: int | None = None,
    injected_replicates: int | None = None,
    gate_validation_replicates: int | None = None,
) -> None:
    print(
        _remote(c).run_phase0_calibration(
            config,
            output=output,
            null_replicates=null_replicates,
            shuffled_replicates=shuffled_replicates,
            injected_replicates=injected_replicates,
            gate_validation_replicates=gate_validation_replicates,
        )
    )


@task
def calibrate_native_sensitivity(
    c: Connection,
    config: str = "configs/worker03.example.yaml",
    output: str | None = None,
    development_magnitudes: str = "",
    development_replicates: int | None = None,
    validation_replicates: int | None = None,
) -> None:
    magnitudes = [int(value) for value in development_magnitudes.split()]
    print(
        _remote(c).run_native_sensitivity_calibration(
            config,
            output=output,
            development_magnitudes=magnitudes or None,
            development_replicates=development_replicates,
            validation_replicates=validation_replicates,
        )
    )


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
    "run_phase0",
    "run_phase1a",
    "calibrate_phase0",
    "calibrate_native_sensitivity",
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
