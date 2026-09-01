"""Independent empirical calibration for the complete Phase 0 pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition.synthetic import SyntheticBackend
from .config import config_fingerprint
from .datasets import build_feature_matrix, load_dataset, write_dataset_manifest
from .inventory import collect_inventory
from .journal import Journal
from .metrics import (
    CHANCE_LEVEL,
    SUPPORTED_METRICS,
    empirical_p_value,
    empirical_percentile,
    max_statistic,
    metric_value,
    monte_carlo_permutation_test,
    wilson_interval,
)
from .models import fit_model
from .phase0 import _enabled_models
from .protocol import phase0_protocol, phase0_protocol_hash
from .runner import _git_commit, new_run_id
from .splits import grouped_split, partition_indices, write_split
from .storage import ShardWriter, validate_all_shards


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _seed_bundle(base_seed: int, replicate: int, *, fresh: bool = False) -> dict[str, int]:
    sequence = np.random.SeedSequence([base_seed, 0xF00D if fresh else 0xCAFE, replicate])
    values = sequence.generate_state(6, dtype=np.uint32)
    names = [
        "acquisition_seed",
        "label_seed",
        "trace_seed",
        "split_seed",
        "model_seed",
        "permutation_seed",
    ]
    return {name: int(value) for name, value in zip(names, values, strict=True)}


def _calibration_config(config: dict[str, Any], mode: str, seeds: dict[str, int]) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    value.setdefault("acquisition", {})["synthetic_balance_mode"] = mode
    calibration = value.setdefault("calibration", {})
    if calibration.get("samples") is not None:
        value.setdefault("data", {})["samples"] = int(calibration["samples"])
    if calibration.get("trace_length") is not None:
        trace_length = int(calibration["trace_length"])
        value.setdefault("data", {})["trace_length"] = trace_length
        injection = value.setdefault("controls", {}).setdefault("injected_weak_signal", {})
        start_index = int(injection.get("start_index", trace_length // 3))
        width = int(injection.get("width", max(4, trace_length // 16)))
        if start_index + width > trace_length:
            injection["start_index"] = trace_length // 3
            injection["width"] = max(4, trace_length // 16)
    value["experiment"] = {
        **value.get("experiment", {}),
        "acquisition_seed": seeds["acquisition_seed"],
        "label_seed": seeds["label_seed"],
        "trace_seed": seeds["trace_seed"],
        "dataset_id": f"phase0-{seeds['acquisition_seed']:010d}",
    }
    return value


def _permutation_strata(config: dict[str, Any]) -> list[str]:
    configured = config.get("calibration", {}).get("permutation_strata")
    if configured:
        return list(configured)
    mode = config.get("acquisition", {}).get("synthetic_balance_mode", "global_balance_only")
    by_mode = config.get("calibration", {}).get("permutation_strata_by_balance_mode", {})
    if mode in by_mode:
        return list(by_mode[mode])
    return [
        "synthetic_location_id" if mode == "group_stratified_balance" else "synthetic_dataset_id"
    ]


def _materialize_synthetic(
    config: dict[str, Any],
    condition_dir: Path,
    condition: str,
    amplitude: float,
    seeds: dict[str, int],
) -> dict[str, Any]:
    data = config.get("data", {})
    controls = config.get("controls", {}).get("injected_weak_signal", {})
    acquisition = config.get("acquisition", {})
    backend = SyntheticBackend(
        count=int(data.get("samples", 1000)),
        trace_length=int(data.get("trace_length", 128)),
        seed=seeds["acquisition_seed"],
        acquisition_seed=seeds["acquisition_seed"],
        label_seed=seeds["label_seed"],
        trace_seed=seeds["trace_seed"],
        dataset_id=str(config.get("experiment", {}).get("dataset_id", "phase0")),
        condition=condition,
        amplitude_sigma=amplitude,
        start_index=int(controls.get("start_index", int(data.get("trace_length", 128)) // 3)),
        width=int(controls.get("width", max(4, int(data.get("trace_length", 128)) // 16))),
        session_count=int(acquisition.get("session_count", 4)),
        device_count=int(acquisition.get("device_count", 2)),
        balance_mode=str(acquisition.get("synthetic_balance_mode", "global_balance_only")),
        observations_per_location=int(acquisition.get("observations_per_location", 4)),
        permute_seed=seeds["permutation_seed"],
    )
    condition_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(
        condition_dir,
        shard_target_mb=float(acquisition.get("shard_target_mb", 512)),
        max_samples_per_shard=acquisition.get("max_samples_per_shard"),
    )
    journal = Journal(condition_dir / "events.jsonl")
    for sample in backend.samples():
        info = writer.add(sample.trace, sample.label, sample.metadata)
        if info:
            journal.append("shard_finalized", **info.as_dict())
    info = writer.finalize()
    if info:
        journal.append("shard_finalized", **info.as_dict())
    manifest = write_dataset_manifest(
        condition_dir,
        config=config,
        condition=condition,
        shard_infos=validate_all_shards(condition_dir),
        label_stream_fingerprint=backend.label_stream_fingerprint,
        class_balance={
            "0": int(np.sum(backend._labels == 0)),
            "1": int(np.sum(backend._labels == 1)),
        },
        provenance={
            "observation_semantics": (
                "label-independent synthetic noise"
                if condition == "null"
                else "known label-dependent synthetic signal"
            ),
            "seed_references": seeds,
            "balance_mode": backend.balance_mode,
            "observations_per_location": backend.observations_per_location,
            "synthetic_dataset_id": backend.synthetic_dataset_id,
            "independent_materialized_replicate": True,
        },
    )
    journal.append("condition_acquisition_completed", rows=backend.count, condition=condition)
    return manifest


def _materialize_shuffled(
    source_dir: Path,
    condition_dir: Path,
    config: dict[str, Any],
    permutation_seed: int,
) -> dict[str, Any]:
    traces, labels, metadata, _shards, source_manifest = load_dataset(source_dir)
    strata_keys = _permutation_strata(config)
    permutation = np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(permutation_seed)
    strata: dict[tuple[str, ...], list[int]] = {}
    keys = list(zip(*(np.asarray(metadata[key]).astype(str) for key in strata_keys), strict=True))
    for index, key in enumerate(keys):
        strata.setdefault(key, []).append(index)
    for indices in strata.values():
        indexes = np.asarray(indices, dtype=np.int64)
        permutation[indexes] = rng.permutation(indexes)
    shuffled_labels = labels[permutation]
    condition_dir.mkdir(parents=True, exist_ok=True)
    acquisition = config.get("acquisition", {})
    writer = ShardWriter(
        condition_dir,
        shard_target_mb=float(acquisition.get("shard_target_mb", 512)),
        max_samples_per_shard=acquisition.get("max_samples_per_shard"),
    )
    for index, (trace, label) in enumerate(zip(traces, shuffled_labels, strict=True)):
        writer.add(trace, int(label), {key: values[index] for key, values in metadata.items()})
    writer.finalize()
    return write_dataset_manifest(
        condition_dir,
        config=config,
        condition="shuffled",
        shard_infos=validate_all_shards(condition_dir),
        label_stream_fingerprint=hashlib.sha256(shuffled_labels.tobytes()).hexdigest(),
        class_balance={
            "0": int(np.sum(shuffled_labels == 0)),
            "1": int(np.sum(shuffled_labels == 1)),
        },
        provenance={
            "observation_semantics": "exact parent observation materialization with randomized labels",
            "parent_dataset_fingerprint": source_manifest["dataset_fingerprint"],
            "original_label_stream_fingerprint": source_manifest["label_stream_fingerprint"],
            "permutation_seed": permutation_seed,
            "permutation_reference": hashlib.sha256(permutation.tobytes()).hexdigest(),
            "permutation_strata": strata_keys,
            "only_changed_variable": "label association",
        },
    )


def _evaluate_dataset(
    config: dict[str, Any], dataset_dir: Path, seeds: dict[str, int]
) -> dict[str, Any]:
    traces, labels, metadata, _shards, manifest = load_dataset(dataset_dir)
    split_config = config.get("splits", {}).get("primary", {})
    group_keys = list(
        split_config.get(
            "group_keys",
            ["synthetic_dataset_id", "synthetic_session_id", "synthetic_location_id"],
        )
    )
    split = grouped_split(
        metadata,
        dataset_fingerprint=manifest["dataset_fingerprint"],
        group_keys=group_keys,
        seed=seeds["split_seed"],
        train_fraction=float(split_config.get("train_fraction", 0.7)),
        validation_fraction=float(split_config.get("validation_fraction", 0.15)),
        test_fraction=float(split_config.get("test_fraction", 0.15)),
    )
    write_split(dataset_dir / "split.json", split)
    partitions = partition_indices(metadata, split)
    features = build_feature_matrix(traces, metadata)
    training = config.get("training", {})
    configured_seeds = [int(seed) for seed in training.get("seeds", [11, 23, 37])]
    model_seeds = [seeds["model_seed"] + offset for offset in configured_seeds]
    models: dict[str, Any] = {}
    for model_name in _enabled_models(config):
        runs: list[dict[str, Any]] = []
        for model_seed in model_seeds:
            fitted = fit_model(
                model_name,
                traces,
                features,
                labels,
                partitions,
                seed=model_seed,
                epochs=int(training.get("epochs", 30)),
                patience=int(training.get("early_stopping_patience", 5)),
                batch_size=int(training.get("batch_size", 256)),
            )
            test = partitions["test"]
            input_values = features[test] if model_name != "tiny_cnn" else traces[test]
            probabilities = fitted.predict(input_values)
            runs.append(
                {
                    "model": model_name,
                    "model_seed": model_seed,
                    "balanced_accuracy": metric_value(
                        labels[test], probabilities, "balanced_accuracy"
                    ),
                    "auroc": metric_value(labels[test], probabilities, "auroc"),
                    "parameter_count": fitted.parameter_count,
                    "model_hash": fitted.model_hash,
                    "test_rows": len(test),
                }
            )
        models[model_name] = {
            "model": model_name,
            "seeds": model_seeds,
            "runs": runs,
            "summary": {
                metric: float(np.mean([run[metric] for run in runs]))
                for metric in SUPPORTED_METRICS
            },
        }
    metric_values: dict[str, float] = {}
    for model_name, model_result in models.items():
        for metric in SUPPORTED_METRICS:
            metric_values[f"{model_name}.{metric}"] = float(model_result["summary"][metric])
    statistic, component = max_statistic(metric_values)
    return {
        "dataset": manifest,
        "split": split,
        "models": models,
        "metric_values": metric_values,
        "max_statistic": statistic,
        "max_component": component,
        "seed_references": seeds,
        "balance_mode": config.get("acquisition", {}).get("synthetic_balance_mode"),
    }


def _decision(
    record: dict[str, Any],
    null_max: np.ndarray,
    null_metrics: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    metric_assessments: dict[str, Any] = {}
    for name, observed in record["metric_values"].items():
        null_values = null_metrics.get(name, np.asarray([], dtype=np.float64))
        raw_p = empirical_p_value(observed - CHANCE_LEVEL, null_values - CHANCE_LEVEL)
        adjusted_p = empirical_p_value(observed - CHANCE_LEVEL, null_max)
        metric_assessments[name] = {
            "observed_statistic": float(observed - CHANCE_LEVEL),
            "observed_score": float(observed),
            "null_percentile": empirical_percentile(observed, null_values),
            "raw_empirical_p": raw_p,
            "familywise_adjusted_p": adjusted_p,
            "decision": bool(np.isfinite(adjusted_p) and adjusted_p <= alpha),
        }
    adjusted_p = empirical_p_value(record["max_statistic"], null_max)
    max_component = record.get("max_component")
    component_name = max_component if isinstance(max_component, str) else ""
    component_null = null_metrics.get(component_name, np.asarray([], dtype=np.float64))
    raw_max_p = empirical_p_value(record["max_statistic"], component_null - CHANCE_LEVEL)
    return {
        "observed_statistic": float(record["max_statistic"]),
        "max_component": record["max_component"],
        "null_percentile": empirical_percentile(record["max_statistic"], null_max),
        "raw_empirical_p": raw_max_p,
        "familywise_adjusted_p": adjusted_p,
        "decision": bool(np.isfinite(adjusted_p) and adjusted_p <= alpha),
        "metric_assessments": metric_assessments,
    }


def _condition_distributions(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    maximum = np.asarray([record["max_statistic"] for record in records], dtype=np.float64)
    names = sorted({name for record in records for name in record["metric_values"]})
    per_metric = {
        name: np.asarray(
            [record["metric_values"].get(name, np.nan) for record in records], dtype=np.float64
        )
        for name in names
    }
    return maximum, per_metric


def _false_positive_rate(
    statistics: np.ndarray, critical_value: float, alpha: float
) -> dict[str, Any]:
    finite = statistics[np.isfinite(statistics)]
    positives = int(np.sum(finite >= critical_value))
    return {
        "target_alpha": alpha,
        "positive_replicates": positives,
        "replicates": int(len(finite)),
        "false_positive_rate": float(positives / len(finite)) if len(finite) else float("nan"),
        "wilson_interval_95": wilson_interval(positives, len(finite))
        if len(finite)
        else [float("nan"), float("nan")],
        "critical_max_statistic": float(critical_value),
    }


def _historical_investigation(
    null_metrics: dict[str, np.ndarray], null_max: np.ndarray
) -> list[dict[str, Any]]:
    historical = [
        ("historical boosted-tree BA", "boosted_trees.balanced_accuracy", 0.5318),
        ("historical boosted-tree AUROC", "boosted_trees.auroc", 0.5486),
        ("historical shuffled logistic BA", "logistic_regression.balanced_accuracy", 0.5400),
    ]
    result = []
    for label, metric_name, observed in historical:
        metric_null = null_metrics.get(metric_name, np.asarray([], dtype=np.float64))
        result.append(
            {
                "label": label,
                "metric": metric_name,
                "observed_score": observed,
                "observed_statistic": observed - CHANCE_LEVEL,
                "marginal_null_percentile": empirical_percentile(observed, metric_null),
                "marginal_raw_empirical_p": empirical_p_value(
                    observed - CHANCE_LEVEL, metric_null - CHANCE_LEVEL
                ),
                "familywise_max_statistic_percentile": empirical_percentile(
                    observed - CHANCE_LEVEL, null_max
                ),
                "familywise_adjusted_p": empirical_p_value(observed - CHANCE_LEVEL, null_max),
                "interpretation": (
                    "consistent with the calibrated null unless the corrected p-value is <= alpha; "
                    "this is a historical comparison, not a re-analysis of the original raw dataset"
                ),
            }
        )
    return result


def _permutation_record(
    config: dict[str, Any], record: dict[str, Any], dataset_dir: Path, seeds: dict[str, int]
) -> dict[str, Any]:
    traces, labels, metadata, _shards, _manifest = load_dataset(dataset_dir)
    features = build_feature_matrix(traces, metadata)
    partitions = partition_indices(metadata, record["split"])
    model_name = str(config.get("calibration", {}).get("permutation_model", "logistic_regression"))
    model_seed = seeds["model_seed"]

    def evaluate(candidate_labels: np.ndarray) -> float:
        fitted = fit_model(
            model_name,
            traces,
            features,
            candidate_labels,
            partitions,
            seed=model_seed,
            epochs=int(config.get("training", {}).get("epochs", 30)),
            patience=int(config.get("training", {}).get("early_stopping_patience", 5)),
            batch_size=int(config.get("training", {}).get("batch_size", 256)),
        )
        test = partitions["test"]
        values = features[test] if model_name != "tiny_cnn" else traces[test]
        probabilities = fitted.predict(values)
        scores = {
            metric: metric_value(candidate_labels[test], probabilities, metric)
            for metric in SUPPORTED_METRICS
        }
        statistic, _ = max_statistic(scores)
        return statistic

    observed = evaluate(labels)
    return monte_carlo_permutation_test(
        labels,
        metadata,
        strata_keys=_permutation_strata(config),
        observed_statistic=observed,
        evaluator=evaluate,
        repetitions=int(config.get("calibration", {}).get("permutation_repetitions", 100)),
        seed=seeds["permutation_seed"],
    ) | {"model": model_name, "dataset_fingerprint": record["dataset"]["dataset_fingerprint"]}


def run_native_calibration(output_root: str | Path, *, repetitions: int = 200) -> dict[str, Any]:
    """Calibrate the optional native timing kernel without making a physical claim."""

    from .acquisition.native import NativeMeasurementKernel, summarize_measurements

    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        return {
            "schema": "sensetrace.native-calibration.v1",
            "status": "unavailable",
            "reason": "native library is not built; run make -C native",
        }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    address = kernel.calibration_address()
    cached = kernel.measure_cached(address, repetitions)
    flushed = kernel.measure_flushed(address, repetitions)
    timer = kernel.timer_calibration(repetitions)
    ffi = np.asarray([kernel.timer_calibration(1)[0] for _ in range(repetitions)], dtype=np.float64)
    idle = kernel.idle_calibration(repetitions)
    report = {
        "schema": "sensetrace.native-calibration.v1",
        "status": "complete",
        "kernel": kernel.provenance(),
        "repetitions": repetitions,
        "distributions": {
            "timer_only": summarize_measurements(timer),
            "ffi_call_overhead_proxy": summarize_measurements(ffi),
            "cached_load": summarize_measurements(cached),
            "flushed_load": summarize_measurements(flushed),
            "idle_control": summarize_measurements(idle),
        },
        "claim_boundary": "flushed means CLFLUSH followed by a timed load; it does not prove DRAM access",
    }
    _write_json(output / "native-calibration.json", report)
    return report


def run_phase0_calibration(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    run_id: str | None = None,
    null_replicates: int | None = None,
    shuffled_replicates: int | None = None,
    injected_replicates: int | None = None,
    gate_validation_replicates: int | None = None,
) -> dict[str, Any]:
    """Materialize calibration and fresh validation ensembles with a frozen rule."""

    calibration_config = config.get("calibration", {})
    alpha = float(calibration_config.get("alpha", 0.05))
    modes = list(
        calibration_config.get("balance_modes", ["global_balance_only", "group_stratified_balance"])
    )
    null_count = int(null_replicates or calibration_config.get("null_replicates", 50))
    shuffled_count = int(
        shuffled_replicates or calibration_config.get("shuffled_replicates", null_count)
    )
    injected_count = int(
        injected_replicates or calibration_config.get("injected_replicates", null_count)
    )
    fresh_count = int(
        gate_validation_replicates
        or calibration_config.get("gate_validation_replicates", null_count)
    )
    if min(null_count, shuffled_count, injected_count, fresh_count) < 1:
        raise ValueError("calibration replicate counts must be positive")
    run_id = run_id or new_run_id("phase0-calibration")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol = phase0_protocol(config)
    protocol_hash = phase0_protocol_hash(config)
    _write_json(
        run_dir / "run.json",
        {
            "schema": "sensetrace.run.v2",
            "run_id": run_id,
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(config),
            "protocol_version": protocol["version"],
            "protocol_hash": protocol_hash,
            "claim_scope": "empirically calibrated synthetic Phase 0 controls only",
        },
    )
    _write_json(run_dir / "host.json", collect_inventory())
    _write_json(run_dir / "protocol.json", {**protocol, "protocol_hash": protocol_hash})
    _write_json(run_dir / "config.json", config)
    journal = Journal(run_dir / "events.jsonl")
    journal.append("calibration_started", protocol_hash=protocol_hash, balance_modes=modes)
    amplitude = float(
        config.get("controls", {}).get("injected_weak_signal", {}).get("amplitude_sigma", 0.1)
    )
    signal_levels = [
        float(level)
        for level in calibration_config.get(
            "injected_levels",
            config.get("controls", {}).get("injected_weak_signal", {}).get("levels", [amplitude]),
        )
    ]
    if amplitude not in signal_levels:
        signal_levels.append(amplitude)
    conditions: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {"null": [], "shuffled": [], "injected": []} for mode in modes
    }
    signal_curve: dict[str, dict[str, list[dict[str, Any]]]] = {mode: {} for mode in modes}
    calibration_root = run_dir / "calibration"
    for mode in modes:
        for replicate in range(max(null_count, shuffled_count, injected_count)):
            seeds = _seed_bundle(int(config.get("experiment", {}).get("seed", 1337)), replicate)
            condition_config = _calibration_config(config, mode, seeds)
            parent = calibration_root / mode / "injected" / f"replicate-{replicate:04d}"
            injected_dir = parent
            injected_manifest = _materialize_synthetic(
                condition_config, injected_dir, "injected", amplitude, seeds
            )
            injected_record = _evaluate_dataset(condition_config, injected_dir, seeds)
            injected_record["dataset"]["dataset_fingerprint"] = injected_manifest[
                "dataset_fingerprint"
            ]
            injected_record["injected_amplitude_sigma"] = amplitude
            if replicate < injected_count:
                conditions[mode]["injected"].append(injected_record)
            for level in signal_levels:
                if level == amplitude or replicate >= injected_count:
                    continue
                level_name = f"injected_{level:g}"
                level_dir = calibration_root / mode / level_name / f"replicate-{replicate:04d}"
                level_manifest = _materialize_synthetic(
                    condition_config, level_dir, "injected", level, seeds
                )
                level_record = _evaluate_dataset(condition_config, level_dir, seeds)
                level_record["dataset"]["dataset_fingerprint"] = level_manifest[
                    "dataset_fingerprint"
                ]
                level_record["injected_amplitude_sigma"] = level
                signal_curve[mode].setdefault(level_name, []).append(level_record)
            if replicate < shuffled_count:
                shuffled_dir = calibration_root / mode / "shuffled" / f"replicate-{replicate:04d}"
                shuffled_manifest = _materialize_shuffled(
                    injected_dir, shuffled_dir, condition_config, seeds["permutation_seed"]
                )
                shuffled_record = _evaluate_dataset(condition_config, shuffled_dir, seeds)
                shuffled_record["dataset"]["dataset_fingerprint"] = shuffled_manifest[
                    "dataset_fingerprint"
                ]
                conditions[mode]["shuffled"].append(shuffled_record)
            if replicate < null_count:
                null_dir = calibration_root / mode / "null" / f"replicate-{replicate:04d}"
                null_manifest = _materialize_synthetic(
                    condition_config, null_dir, "null", 0.0, seeds
                )
                null_record = _evaluate_dataset(condition_config, null_dir, seeds)
                null_record["dataset"]["dataset_fingerprint"] = null_manifest["dataset_fingerprint"]
                conditions[mode]["null"].append(null_record)
        journal.append("balance_mode_completed", mode=mode)

    all_null = [record for mode in modes for record in conditions[mode]["null"]]
    null_max, null_metrics = _condition_distributions(all_null)
    critical = float(np.quantile(null_max, 1.0 - alpha))
    calibration_fpr = _false_positive_rate(null_max, critical, alpha)
    historical = _historical_investigation(null_metrics, null_max)

    fresh: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {"null": [], "shuffled": [], "injected": []} for mode in modes
    }
    for mode in modes:
        for replicate in range(fresh_count):
            seeds = _seed_bundle(
                int(config.get("experiment", {}).get("seed", 1337)),
                100000 + replicate,
                fresh=True,
            )
            condition_config = _calibration_config(config, mode, seeds)
            injected_dir = (
                run_dir / "gate-validation" / mode / "injected" / f"replicate-{replicate:04d}"
            )
            injected_manifest = _materialize_synthetic(
                condition_config, injected_dir, "injected", amplitude, seeds
            )
            injected_record = _evaluate_dataset(condition_config, injected_dir, seeds)
            injected_record["dataset"]["dataset_fingerprint"] = injected_manifest[
                "dataset_fingerprint"
            ]
            injected_record["injected_amplitude_sigma"] = amplitude
            injected_record["calibrated_decision"] = _decision(
                injected_record, null_max, null_metrics, alpha=alpha
            )
            fresh[mode]["injected"].append(injected_record)
            shuffled_dir = (
                run_dir / "gate-validation" / mode / "shuffled" / f"replicate-{replicate:04d}"
            )
            shuffled_manifest = _materialize_shuffled(
                injected_dir, shuffled_dir, condition_config, seeds["permutation_seed"]
            )
            shuffled_record = _evaluate_dataset(condition_config, shuffled_dir, seeds)
            shuffled_record["dataset"]["dataset_fingerprint"] = shuffled_manifest[
                "dataset_fingerprint"
            ]
            shuffled_record["calibrated_decision"] = _decision(
                shuffled_record, null_max, null_metrics, alpha=alpha
            )
            fresh[mode]["shuffled"].append(shuffled_record)
            null_dir = run_dir / "gate-validation" / mode / "null" / f"replicate-{replicate:04d}"
            null_manifest = _materialize_synthetic(condition_config, null_dir, "null", 0.0, seeds)
            null_record = _evaluate_dataset(condition_config, null_dir, seeds)
            null_record["dataset"]["dataset_fingerprint"] = null_manifest["dataset_fingerprint"]
            null_record["calibrated_decision"] = _decision(
                null_record, null_max, null_metrics, alpha=alpha
            )
            fresh[mode]["null"].append(null_record)

    fresh_null = [record for mode in modes for record in fresh[mode]["null"]]
    fresh_shuffled = [record for mode in modes for record in fresh[mode]["shuffled"]]
    fresh_injected = [record for mode in modes for record in fresh[mode]["injected"]]
    fresh_null_stats = np.asarray(
        [record["max_statistic"] for record in fresh_null], dtype=np.float64
    )
    fresh_fpr = _false_positive_rate(fresh_null_stats, critical, alpha)
    fresh_shuffled_stats = np.asarray(
        [record["max_statistic"] for record in fresh_shuffled], dtype=np.float64
    )
    shuffled_fpr = _false_positive_rate(fresh_shuffled_stats, critical, alpha)

    def rate_is_calibrated(rate: dict[str, Any]) -> bool:
        value = float(rate["false_positive_rate"])
        return bool(
            np.isfinite(value)
            and value <= alpha + max(0.05, alpha)
            and rate["wilson_interval_95"][0] <= alpha
        )

    # A validation ensemble is itself a repeated experiment. One significant
    # null replicate is expected at alpha; rejecting because any one fires would
    # reintroduce the multiple-replicate problem this calibration fixes.
    null_pass = rate_is_calibrated(fresh_fpr)
    shuffled_pass = rate_is_calibrated(shuffled_fpr)
    injected_positives = sum(
        bool(record["calibrated_decision"]["decision"]) for record in fresh_injected
    )
    detection_rate = injected_positives / len(fresh_injected)
    minimum_detection_rate = float(calibration_config.get("minimum_injected_detection_rate", 0.8))
    injected_pass = detection_rate >= minimum_detection_rate
    fpr_pass = rate_is_calibrated(fresh_fpr)
    permutation_tests = []
    for mode in modes:
        if conditions[mode]["injected"]:
            record = conditions[mode]["injected"][0]
            permutation_tests.append(
                _permutation_record(
                    _calibration_config(config, mode, record["seed_references"]),
                    record,
                    calibration_root / mode / "injected" / "replicate-0000",
                    record["seed_references"],
                )
            )
    report = {
        "schema": "sensetrace.phase0-calibration-report.v1",
        "run_id": run_id,
        "protocol_version": protocol["version"],
        "protocol_hash": protocol_hash,
        "alpha": alpha,
        "multiple_comparison_policy": "empirical maximum statistic across enabled models and metrics; the maximum distribution includes all balance variants",
        "statistic": "max(score - 0.5) across model/metric summaries for one independent materialized replicate",
        "counts": {
            "balance_modes": modes,
            "calibration_null_replicates": len(all_null),
            "calibration_shuffled_replicates": sum(
                len(conditions[mode]["shuffled"]) for mode in modes
            ),
            "calibration_injected_replicates": sum(
                len(conditions[mode]["injected"]) for mode in modes
            ),
            "fresh_gate_validation_null_replicates": len(fresh_null),
            "fresh_gate_validation_shuffled_replicates": len(fresh_shuffled),
            "fresh_gate_validation_injected_replicates": len(fresh_injected),
        },
        "calibration_ensemble": conditions,
        "injected_signal_strengths": signal_levels,
        "injected_signal_curve": signal_curve,
        "empirical_null": {
            "max_statistic_distribution": null_max.tolist(),
            "metric_distributions": {
                name: values.tolist() for name, values in null_metrics.items()
            },
            "critical_max_statistic": critical,
            "calibration_false_positive_rate": calibration_fpr,
        },
        "permutation_tests": permutation_tests,
        "historical_null_investigation": historical,
        "fresh_gate_validation": {
            "conditions": fresh,
            "false_positive_rate": fresh_fpr,
            "shuffled_false_positive_rate": shuffled_fpr,
            "fresh_dataset_fingerprints": [
                record["dataset"]["dataset_fingerprint"] for record in fresh_null
            ],
        },
        "calibration_dataset_fingerprints": {
            mode: {
                condition: [
                    record["dataset"]["dataset_fingerprint"]
                    for record in conditions[mode][condition]
                ]
                for condition in ["null", "shuffled", "injected"]
            }
            for mode in modes
        },
        "acceptance": {
            "null_pass": null_pass,
            "shuffled_pass": shuffled_pass,
            "injected_pass": injected_pass,
            "injected_detection_rate": detection_rate,
            "minimum_injected_detection_rate": minimum_detection_rate,
            "pipeline_false_positive_rate_pass": bool(fpr_pass),
            "phase1_gate": bool(null_pass and shuffled_pass and injected_pass and fpr_pass),
            "decision_rule_frozen": True,
        },
        "claim_boundary": "This calibrates the software pipeline on synthetic data. It does not establish a physical DRAM-state inference result.",
    }
    _write_json(run_dir / "metrics.json", report)
    journal.append("calibration_completed", acceptance=report["acceptance"])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "acceptance": report["acceptance"],
        }
    )
    _write_json(run_dir / "run.json", run)
    return report
