"""Command-line entrypoint for local experiments and controller operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import (
    run_native_calibration,
    run_native_sensitivity_calibration,
    run_phase0_calibration,
    run_phase0_power_study,
)
from .characterization import run_measurement_primitive_characterization
from .config import load_config, validate_config
from .datasets import load_dataset
from .host.client import RemoteHost
from .inventory import collect_inventory
from .phase0 import run_phase0
from .phase1a import run_phase1a
from .recovery import recovery_test
from .runner import AcquisitionRunner, daemon


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _override_phase0_config(
    config: dict[str, object], args: argparse.Namespace
) -> dict[str, object]:
    value = json.loads(json.dumps(config))
    if args.samples is not None:
        value.setdefault("data", {})["samples"] = args.samples
    if args.trace_length is not None:
        value.setdefault("data", {})["trace_length"] = args.trace_length
        signal = value.setdefault("controls", {}).setdefault("injected_weak_signal", {})
        if int(signal.get("start_index", 0)) + int(signal.get("width", 1)) > args.trace_length:
            signal["start_index"] = args.trace_length // 3
            signal["width"] = max(4, args.trace_length // 16)
    if args.seed is not None:
        value.setdefault("experiment", {})["seed"] = args.seed
    if args.models:
        for model in ["logistic_regression", "boosted_trees", "tiny_mlp", "tiny_cnn"]:
            value.setdefault("models", {}).setdefault(model, {})["enabled"] = model in args.models
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sensetrace")
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="capture the local host inventory")
    inventory.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="execute an experiment")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    phase0 = run_sub.add_parser("phase0", help="run synthetic Phase 0 controls")
    phase0.add_argument("--config", default="configs/phase0.example.yaml")
    phase0.add_argument("--output")
    phase0.add_argument("--host")
    phase0.add_argument("--curve", action="store_true")
    phase0.add_argument("--conditions", nargs="*")
    phase0.add_argument("--samples", type=int)
    phase0.add_argument("--trace-length", type=int)
    phase0.add_argument("--seed", type=int)
    phase0.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "boosted_trees", "tiny_mlp", "tiny_cnn"],
    )
    phase1a = run_sub.add_parser("phase1a", help="run gated safe commodity-memory observables")
    phase1a.add_argument("--config", default="configs/worker03.example.yaml")
    phase1a.add_argument("--phase0-report", required=True)
    phase1a.add_argument("--output")
    phase1a.add_argument("--host")
    phase2_mock = run_sub.add_parser(
        "phase2-mock", help="run the Phase 2 controlled-memory software contract emulator"
    )
    phase2_mock.add_argument("--config", default="configs/phase2-controlled-mock.example.yaml")
    phase2_mock.add_argument("--output", default="runs/phase2-controlled-mock")
    phase2_mock.add_argument("--condition", default="null")
    phase2_mock.add_argument("--stop-after", type=int)

    calibrate = sub.add_parser("calibrate", help="calibrate an analysis pipeline")
    calibrate_sub = calibrate.add_subparsers(dest="calibrate_command", required=True)
    phase0_calibration = calibrate_sub.add_parser(
        "phase0", help="materialize independent null, shuffled, and injected replicates"
    )
    phase0_calibration.add_argument("--config", default="configs/phase0.example.yaml")
    phase0_calibration.add_argument("--output", default="runs")
    phase0_calibration.add_argument("--null-replicates", type=int)
    phase0_calibration.add_argument("--shuffled-replicates", type=int)
    phase0_calibration.add_argument("--injected-replicates", type=int)
    phase0_calibration.add_argument("--gate-validation-replicates", type=int)
    phase0_power = calibrate_sub.add_parser(
        "phase0-power", help="run a development-only Phase 0 v2 sample-count power study"
    )
    phase0_power.add_argument("--config", default="configs/phase0.example.yaml")
    phase0_power.add_argument("--output", default="runs")
    phase0_power.add_argument("--sample-counts", nargs="+", type=int)
    phase0_power.add_argument("--replicates", type=int)
    native_calibration = calibrate_sub.add_parser(
        "native", help="calibrate native timer and cache-control distributions"
    )
    native_calibration.add_argument("--output", default="runs/native-calibration")
    native_calibration.add_argument("--repetitions", type=int, default=200)
    native_sensitivity = calibrate_sub.add_parser(
        "native-sensitivity",
        help="measure the native acquisition pipeline's artificial timing detection floor",
    )
    native_sensitivity.add_argument("--config", default="configs/worker03.example.yaml")
    native_sensitivity.add_argument("--output", default="runs/native-sensitivity")
    native_sensitivity.add_argument("--development-magnitudes", nargs="+", type=int)
    native_sensitivity.add_argument("--development-replicates", type=int)
    native_sensitivity.add_argument("--validation-replicates", type=int)

    characterize = sub.add_parser("characterize", help="characterize a measurement primitive")
    characterize_sub = characterize.add_subparsers(dest="characterize_command", required=True)
    primitive = characterize_sub.add_parser(
        "primitive", help="run null and positive-control primitive characterization"
    )
    primitive.add_argument("--config", default="configs/worker03.example.yaml")
    primitive.add_argument("--output", default="runs/primitive-characterization")
    multiboot = characterize_sub.add_parser(
        "multiboot", help="combine one characterization report per genuine boot"
    )
    multiboot.add_argument("--inputs", nargs="+", required=True)
    multiboot.add_argument("--output", required=True)
    multiboot.add_argument(
        "--config",
        required=True,
        help="frozen multiboot configuration; authoritative for expected boot count",
    )

    witness = sub.add_parser("witness", help="inspect or run the optional eBPF witness plane")
    witness_sub = witness.add_subparsers(dest="witness_command", required=True)
    witness_capabilities = witness_sub.add_parser("capabilities")
    witness_capabilities.add_argument("--hooks", nargs="*")
    witness_pilot = witness_sub.add_parser("pilot")
    witness_pilot.add_argument("--output", required=True)
    witness_pilot.add_argument("--sudo", action="store_true")
    witness_pilot.add_argument("--repetitions", type=int, default=20_000)

    validate = sub.add_parser("validate", help="validate a dataset run directory")
    validate.add_argument("dataset")
    status = sub.add_parser("status", help="show local run status or remote host status")
    status.add_argument("--run-dir")
    status.add_argument("--host")

    runner = sub.add_parser("runner", help="run the unattended systemd process")
    runner.add_argument("--config", required=True)

    selftest = sub.add_parser("selftest", help="run local integration self-tests")
    selftest_sub = selftest.add_subparsers(dest="selftest_command", required=True)
    recovery = selftest_sub.add_parser("recovery")
    recovery.add_argument("--output")

    host = sub.add_parser("host", help="controller-side Fabric/SSH operations")
    host_sub = host.add_subparsers(dest="host_command", required=True)
    for name in [
        "inventory",
        "doctor",
        "bootstrap",
        "authorize-sudo",
        "deploy",
        "status",
        "start",
        "stop",
        "restart",
        "verify-recovery",
        "reboot",
        "reboot-acceptance",
        "boot-profile",
        "services",
        "noise-baseline",
        "cpu-profile",
        "verify-boot",
        "headless",
        "isolate-target",
        "set-target",
        "watchdog-profile",
    ]:
        command = host_sub.add_parser(name)
        command.add_argument("host", nargs="?", default="worker-03")
        if name == "deploy":
            command.add_argument(
                "--reset-config",
                action="store_true",
                help="deliberately replace the live operator config with the example template",
            )
        if name == "reboot-acceptance":
            command.add_argument("--cycles", type=int, default=3)
            command.add_argument("--timeout", type=int, default=180)
            command.add_argument(
                "--require-appliance",
                action="store_true",
                help="require sensetrace.target as the active default target",
            )
    logs = host_sub.add_parser("logs")
    logs.add_argument("host", nargs="?", default="worker-03")
    logs.add_argument("--lines", type=int, default=100)
    remote_run = host_sub.add_parser("run-phase0")
    remote_run.add_argument("host", nargs="?", default="worker-03")
    remote_run.add_argument("--config", default="configs/phase0.example.yaml")
    remote_run.add_argument("--output")
    remote_run.add_argument("--curve", action="store_true")
    remote_run.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "boosted_trees", "tiny_mlp", "tiny_cnn"],
    )
    remote_phase1a = host_sub.add_parser("run-phase1a")
    remote_phase1a.add_argument("host", nargs="?", default="worker-03")
    remote_phase1a.add_argument("--config", default="configs/worker03.example.yaml")
    remote_phase1a.add_argument("--phase0-report", required=True)
    remote_phase1a.add_argument("--output")
    remote_calibration = host_sub.add_parser("calibrate-phase0")
    remote_calibration.add_argument("host", nargs="?", default="worker-03")
    remote_calibration.add_argument("--config", default="configs/phase0.example.yaml")
    remote_calibration.add_argument("--output")
    remote_calibration.add_argument("--null-replicates", type=int)
    remote_calibration.add_argument("--shuffled-replicates", type=int)
    remote_calibration.add_argument("--injected-replicates", type=int)
    remote_calibration.add_argument("--gate-validation-replicates", type=int)
    remote_sensitivity = host_sub.add_parser("calibrate-native-sensitivity")
    remote_sensitivity.add_argument("host", nargs="?", default="worker-03")
    remote_sensitivity.add_argument("--config", default="configs/worker03.example.yaml")
    remote_sensitivity.add_argument("--output")
    remote_sensitivity.add_argument("--development-magnitudes", nargs="+", type=int)
    remote_sensitivity.add_argument("--development-replicates", type=int)
    remote_sensitivity.add_argument("--validation-replicates", type=int)
    remote_characterize = host_sub.add_parser("characterize-primitive")
    remote_characterize.add_argument("host", nargs="?", default="worker-03")
    remote_characterize.add_argument("--config", default="configs/worker03.example.yaml")
    remote_characterize.add_argument("--output")
    remote_multiboot = host_sub.add_parser("characterize-multiboot")
    remote_multiboot.add_argument("host", nargs="?", default="worker-03")
    remote_multiboot.add_argument("--config", default="configs/worker03.example.yaml")
    remote_multiboot.add_argument("--output", required=True)
    remote_multiboot.add_argument("--boots", type=int, default=3)
    remote_multiboot.add_argument("--timeout", type=int, default=300)
    remote_multiboot.add_argument("--resume", action="store_true")
    results = sub.add_parser("results", help="retrieve result artifacts")
    results_sub = results.add_subparsers(dest="results_command", required=True)
    latest = results_sub.add_parser("latest")
    latest.add_argument("--host", default="worker-03")
    fetch = results_sub.add_parser("fetch")
    fetch.add_argument("--host", default="worker-03")
    fetch.add_argument("--destination", default="evidence/latest")
    fetch.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        _json(collect_inventory())
        return 0
    if args.command == "run" and args.run_command == "phase0":
        config = validate_config(_override_phase0_config(load_config(args.config), args))
        if args.host:
            print(
                RemoteHost(args.host).run_phase0(
                    args.config,
                    output=args.output,
                    include_curve=args.curve,
                    samples=args.samples,
                    trace_length=args.trace_length,
                    seed=args.seed,
                    models=args.models,
                )
            )
        else:
            _json(
                run_phase0(
                    config,
                    args.output or "runs",
                    include_curve=args.curve,
                    conditions=args.conditions,
                )
            )
        return 0
    if args.command == "run" and args.run_command == "phase1a":
        config = load_config(args.config)
        if args.host:
            print(
                RemoteHost(args.host).run_phase1a(
                    args.config, args.phase0_report, output=args.output
                )
            )
        else:
            _json(
                run_phase1a(
                    config,
                    args.output or "runs",
                    phase0_report=args.phase0_report,
                )
            )
        return 0
    if args.command == "run" and args.run_command == "phase2-mock":
        config = load_config(args.config)
        if config.get("acquisition", {}).get("backend") != "controlled_mock":
            raise SystemExit("run phase2-mock requires acquisition.backend=controlled_mock")
        output = Path(args.output)
        _json(
            AcquisitionRunner(config, output, run_id=output.name).run(
                condition=args.condition, stop_after=args.stop_after
            )
        )
        return 0
    if args.command == "calibrate" and args.calibrate_command == "phase0":
        config = validate_config(load_config(args.config))
        _json(
            run_phase0_calibration(
                config,
                args.output,
                null_replicates=args.null_replicates,
                shuffled_replicates=args.shuffled_replicates,
                injected_replicates=args.injected_replicates,
                gate_validation_replicates=args.gate_validation_replicates,
            )
        )
        return 0
    if args.command == "calibrate" and args.calibrate_command == "phase0-power":
        config = validate_config(load_config(args.config))
        _json(
            run_phase0_power_study(
                config,
                args.output,
                sample_counts=args.sample_counts,
                candidate_replicates=args.replicates,
            )
        )
        return 0
    if args.command == "calibrate" and args.calibrate_command == "native":
        _json(run_native_calibration(args.output, repetitions=args.repetitions))
        return 0
    if args.command == "calibrate" and args.calibrate_command == "native-sensitivity":
        _json(
            run_native_sensitivity_calibration(
                validate_config(load_config(args.config)),
                args.output,
                development_magnitudes=args.development_magnitudes,
                development_replicates=args.development_replicates,
                validation_replicates=args.validation_replicates,
            )
        )
        return 0
    if args.command == "characterize" and args.characterize_command == "primitive":
        _json(
            run_measurement_primitive_characterization(
                validate_config(load_config(args.config)), args.output
            )
        )
        return 0
    if args.command == "characterize" and args.characterize_command == "multiboot":
        from .multiboot import write_combined_report

        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
        config = validate_config(load_config(args.config))
        expected_boots = int(config.get("characterization", {}).get("multiboot_boots", 0))
        if expected_boots < 3 or len(reports) != expected_boots:
            raise ValueError(
                f"frozen protocol requires {expected_boots} boot reports; got {len(reports)}"
            )
        _json(
            write_combined_report(
                reports, args.output, expected_boots=expected_boots, frozen_config=config
            )
        )
        return 0
    if args.command == "witness" and args.witness_command == "capabilities":
        from .witness import discover_witness_capabilities

        hooks = tuple(args.hooks) if args.hooks else None
        _json(discover_witness_capabilities(hooks))
        return 0
    if args.command == "witness" and args.witness_command == "pilot":
        from .witness.pilot import run_witness_pilot

        _json(run_witness_pilot(args.output, use_sudo=args.sudo, repetitions=args.repetitions))
        return 0
    if args.command == "validate":
        traces, labels, metadata, shards, manifest = load_dataset(args.dataset)
        _json(
            {
                "valid": True,
                "rows": len(labels),
                "trace_shape": list(traces.shape),
                "shards": [item.as_dict() for item in shards],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "metadata_fields": sorted(metadata),
            }
        )
        return 0
    if args.command == "status":
        if args.host:
            _json(RemoteHost(args.host).status())
        else:
            run_path = Path(args.run_dir or "runs")
            _json(
                {
                    "path": str(run_path),
                    "exists": run_path.exists(),
                    "run": json.loads((run_path / "run.json").read_text())
                    if (run_path / "run.json").exists()
                    else None,
                }
            )
        return 0
    if args.command == "runner":
        daemon(args.config)
        return 0
    if args.command == "selftest" and args.selftest_command == "recovery":
        result = recovery_test(args.output)
        _json(result)
        return 0 if result["passed"] else 1
    if args.command == "host":
        remote = RemoteHost(args.host)
        if args.host_command == "inventory":
            _json(remote.inventory())
        elif args.host_command == "doctor":
            _json(remote.doctor())
        elif args.host_command == "bootstrap":
            _json(remote.bootstrap())
        elif args.host_command == "authorize-sudo":
            _json(remote.authorize_sudo())
        elif args.host_command == "deploy":
            _json(remote.deploy(reset_config=args.reset_config))
        elif args.host_command == "status":
            _json(remote.status())
        elif args.host_command in {"start", "stop", "restart"}:
            _json(remote.service_action(args.host_command))
        elif args.host_command == "logs":
            print(remote.logs(lines=args.lines), end="")
        elif args.host_command == "run-phase0":
            print(
                remote.run_phase0(
                    args.config,
                    output=args.output,
                    include_curve=args.curve,
                    models=args.models,
                ),
                end="",
            )
        elif args.host_command == "verify-recovery":
            _json(remote.verify_recovery())
        elif args.host_command == "reboot":
            print(remote.reboot())
        elif args.host_command == "reboot-acceptance":
            _json(
                remote.reboot_acceptance(
                    cycles=args.cycles,
                    timeout_seconds=args.timeout,
                    require_appliance=args.require_appliance,
                )
            )
        elif args.host_command == "boot-profile":
            _json(remote.boot_profile())
        elif args.host_command == "services":
            _json(remote.services())
        elif args.host_command == "noise-baseline":
            _json(remote.noise_baseline())
        elif args.host_command == "cpu-profile":
            _json(remote.cpu_profile())
        elif args.host_command == "verify-boot":
            _json(remote.verify_boot())
        elif args.host_command == "headless":
            _json(remote.set_headless())
        elif args.host_command == "isolate-target":
            _json(remote.isolate_sensetrace_target())
        elif args.host_command == "set-target":
            _json(remote.set_sensetrace_default())
        elif args.host_command == "watchdog-profile":
            _json(remote.watchdog_profile())
        elif args.host_command == "run-phase1a":
            print(
                remote.run_phase1a(args.config, args.phase0_report, output=args.output),
                end="",
            )
        elif args.host_command == "calibrate-phase0":
            print(
                remote.run_phase0_calibration(
                    args.config,
                    output=args.output,
                    null_replicates=args.null_replicates,
                    shuffled_replicates=args.shuffled_replicates,
                    injected_replicates=args.injected_replicates,
                    gate_validation_replicates=args.gate_validation_replicates,
                ),
                end="",
            )
        elif args.host_command == "calibrate-native-sensitivity":
            print(
                remote.run_native_sensitivity_calibration(
                    args.config,
                    output=args.output,
                    development_magnitudes=args.development_magnitudes,
                    development_replicates=args.development_replicates,
                    validation_replicates=args.validation_replicates,
                ),
                end="",
            )
        elif args.host_command == "characterize-primitive":
            print(
                remote.characterize_primitive(args.config, output=args.output),
                end="",
            )
        elif args.host_command == "characterize-multiboot":
            _json(
                remote.characterize_multiboot(
                    args.config,
                    output=args.output,
                    boots=args.boots,
                    timeout_seconds=args.timeout,
                    resume=args.resume,
                )
            )
        return 0
    if args.command == "results":
        remote = RemoteHost(args.host)
        if args.results_command == "latest":
            _json(remote.latest_result())
        else:
            print(remote.fetch_results(args.destination, run_id=args.run_id))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
