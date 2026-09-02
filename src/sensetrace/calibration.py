"""Independent empirical calibration for the complete Phase 0 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition.primitive import TimingPerturbationCalibration
from .acquisition.synthetic import SyntheticBackend
from .audits import run_leakage_audits
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
    paired_delta_analysis,
    wilson_interval,
)
from .models import fit_model, train_and_evaluate
from .phase0 import _enabled_models
from .protocol import phase0_protocol, phase0_protocol_hash
from .runner import _git_commit, new_run_id
from .splits import (
    grouped_split,
    partition_indices,
    phase1a_split_hierarchy,
    validate_phase1a_split_hierarchy,
    write_split,
)
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
        "false_positive_denominator": int(len(finite)),
        "false_positive_rate": float(positives / len(finite)) if len(finite) else float("nan"),
        "wilson_interval_95": wilson_interval(positives, len(finite))
        if len(finite)
        else [float("nan"), float("nan")],
        "critical_max_statistic": float(critical_value),
    }


def _empirical_null_threshold(
    statistics: np.ndarray | list[float], *, alpha: float
) -> dict[str, Any]:
    """Choose a conservative finite-sample threshold from an empirical null.

    The threshold uses the plus-one Monte Carlo p-value convention used by
    :func:`empirical_p_value`.  It is the next representable value above the
    selected null order statistic, so a null value tied at the order statistic
    is not silently counted as a rejection.  No interpolation or parametric
    tail model is assumed.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    values = np.asarray(statistics, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("null statistics must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("null statistics contain missing or non-finite replicates")
    count = int(len(values))
    if count == 0:
        return {
            "requested_alpha": float(alpha),
            "null_replicates": 0,
            "minimum_resolvable_tail_probability": float("nan"),
            "empirical_rate_resolution": float("nan"),
            "method": "conservative empirical order statistic with plus-one p-values",
            "alpha_supported": False,
            "critical_statistic": float("nan"),
            "order_statistic": float("nan"),
            "order_statistic_rank": None,
            "allowed_null_exceedances": None,
            "interpretation": "pilot threshold unavailable: development null ensemble is empty",
        }

    # With p=(exceedances+1)/(N+1), the smallest non-zero p-value is 1/(N+1).
    minimum_tail = 1.0 / (count + 1)
    allowed_exceedances = max(
        0,
        min(count - 1, int(math.floor(alpha * (count + 1) - 1.0 + 1e-12))),
    )
    ordered = np.sort(values)
    order_index = count - allowed_exceedances - 1
    order_statistic = float(ordered[order_index])
    critical = float(np.nextafter(order_statistic, np.inf))
    alpha_supported = bool(minimum_tail <= alpha)
    return {
        "requested_alpha": float(alpha),
        "null_replicates": count,
        "minimum_resolvable_tail_probability": minimum_tail,
        "empirical_rate_resolution": 1.0 / count,
        "empirical_p_value_resolution": minimum_tail,
        "method": "conservative empirical order statistic with plus-one p-values",
        "alpha_supported": alpha_supported,
        "critical_statistic": critical,
        "order_statistic": order_statistic,
        "order_statistic_rank": int(order_index + 1),
        "allowed_null_exceedances": int(allowed_exceedances),
        "interpretation": (
            "empirically alpha-calibrated detection threshold"
            if alpha_supported
            else "pilot threshold only"
        ),
    }


def _replicate_quality(
    *, statistics: np.ndarray, alpha: float, minimum_recommended: int
) -> dict[str, Any]:
    """Explain how finely a repeated control can resolve an empirical rate."""

    count = int(np.sum(np.isfinite(statistics)))
    resolution = float(1.0 / count) if count else float("nan")
    poorly_resolved = count < minimum_recommended or (
        count > 0 and resolution > max(alpha / 2.0, 1e-12)
    )
    return {
        "replicates": count,
        "target_alpha": alpha,
        "empirical_rate_resolution": resolution,
        "empirical_p_value_resolution": (float(1.0 / (count + 1)) if count else float("nan")),
        "minimum_recommended_replicates": minimum_recommended,
        "precision_warning": bool(poorly_resolved),
        "interpretation": (
            "pipeline_sanity_check: the empirical tail is coarse and should not be presented "
            "as a high-precision false-positive estimate"
            if poorly_resolved
            else "quantitative_sensitivity_estimate: replicate count resolves the configured tail more finely"
        ),
    }


def _record_statistics(
    records: list[dict[str, Any]],
    *,
    field: str = "max_statistic",
    require_complete: bool = False,
    expected_count: int | None = None,
    ensemble: str = "ensemble",
) -> np.ndarray:
    def finite_record(record: dict[str, Any]) -> bool:
        try:
            return bool(np.isfinite(record.get(field, np.nan)))
        except (TypeError, ValueError):
            return False

    if require_complete:
        failures = [
            {
                "index": index,
                "status": record.get("status"),
                "statistic": record.get(field),
            }
            for index, record in enumerate(records)
            if record.get("status") != "available" or not finite_record(record)
        ]
        count_mismatch = expected_count is not None and len(records) != expected_count
        if count_mismatch or failures:
            expected = "unknown" if expected_count is None else str(expected_count)
            raise RuntimeError(
                f"{ensemble} is incomplete: expected {expected} records, found {len(records)}; "
                f"failed records={failures}"
            )
        return np.asarray([record[field] for record in records], dtype=np.float64)
    return np.asarray(
        [
            record[field]
            for record in records
            if record.get("status") == "available" and finite_record(record)
        ],
        dtype=np.float64,
    )


def _ensemble_provenance(
    records: list[dict[str, Any]], *, ensemble: str, expected_count: int
) -> dict[str, Any]:
    """Make the source of a statistic auditable without relying on variable names."""

    return {
        "ensemble": ensemble,
        "statistic_field": "max_statistic",
        "expected_replicates": int(expected_count),
        "record_count": int(len(records)),
        "available_record_count": int(
            sum(record.get("status") == "available" for record in records)
        ),
        "dataset_fingerprints": [
            record.get("dataset", {}).get("dataset_fingerprint", "unavailable")
            for record in records
        ],
    }


def _shuffled_false_positive_summary(
    records: list[dict[str, Any]],
    *,
    critical_max_statistic: float,
    source: str,
    ensemble: str,
    alpha: float,
    minimum_recommended_null: int = 20,
    expected_replicates: int | None = None,
) -> dict[str, Any]:
    """Summarize one explicitly named shuffled-control ensemble."""

    statistics = _record_statistics(
        records,
        require_complete=expected_replicates is not None,
        expected_count=expected_replicates,
        ensemble=f"{ensemble} shuffled-label controls",
    )
    positives = int(np.sum(statistics >= critical_max_statistic))
    return {
        "source": source,
        "ensemble": ensemble,
        "condition": "same-observation label permutation",
        "rate": float(positives / len(statistics)) if len(statistics) else float("nan"),
        "replicates": int(len(statistics)),
        "false_positive_denominator": int(len(statistics)),
        "positive_replicates": positives,
        "wilson_interval_95": wilson_interval(positives, len(statistics))
        if len(statistics)
        else [float("nan"), float("nan")],
        "statistics_quality": _replicate_quality(
            statistics=statistics,
            alpha=alpha,
            minimum_recommended=minimum_recommended_null,
        ),
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
    get_cpu = getattr(os, "sched_getcpu", None)
    cpu_before = get_cpu() if get_cpu is not None else "unavailable"
    cached = kernel.measure_cached(address, repetitions)
    flushed = kernel.measure_flushed(address, repetitions)
    timer = kernel.timer_calibration(repetitions)
    ffi = np.asarray([kernel.timer_calibration(1)[0] for _ in range(repetitions)], dtype=np.float64)
    idle = kernel.idle_calibration(repetitions)
    cpu_after = get_cpu() if get_cpu is not None else "unavailable"
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
        "measurement_quality_diagnostics": {
            "cpu_affinity_at_calibration": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else "unavailable"
            ),
            "cpu_at_call_boundaries": {
                "before": cpu_before,
                "after": cpu_after,
                "interpretation": (
                    "backend samples also record sched_getcpu at sample boundaries; the native "
                    "library does not silently discard migration or interrupt-contaminated traces"
                ),
            },
            "frequency_and_governor": (
                "recorded in host inventory and per-sample backend metadata; not a model feature"
            ),
            "thermal_and_interrupts": (
                "recorded as available by host inventory; interrupt/context-switch contamination "
                "is not filtered from raw timing values"
            ),
            "cache_control_separation": {
                "cached_median_cycles": float(np.median(cached)),
                "flushed_median_cycles": float(np.median(flushed)),
                "flushed_minus_cached_cycles": float(np.median(flushed) - np.median(cached)),
            },
            "timer_overhead_median_cycles": float(np.median(timer)),
            "autocorrelation_is_audit_only": True,
        },
        "claim_boundary": "flushed means CLFLUSH followed by a timed load; it does not prove DRAM access",
    }
    _write_json(output / "native-calibration.json", report)
    return report


def _native_sensitivity_config(
    config: dict[str, Any],
    *,
    cycles: int,
    namespace: str,
    seed: int,
) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    physical = value.setdefault("phase1a", {})
    physical.update(
        {
            "timing_perturbation_cycles": int(cycles),
            "timing_perturbation_label": 1,
            "calibration_namespace": namespace,
            "use_native_kernel": True,
            "require_native_kernel": True,
        }
    )
    value.setdefault("experiment", {})["seed"] = int(seed)
    return value


def _materialize_native_sensitivity_dataset(
    config: dict[str, Any], dataset_dir: Path, *, condition: str, seed: int
) -> dict[str, Any]:
    """Acquire one independently seeded native calibration dataset.

    This deliberately uses the same CommodityDramBackend and per-session
    source-manifest path as Phase 1A, but writes under a separate calibration
    namespace and never changes any Phase 1A run directory.
    """

    from .phase1a import _materialize_session

    physical = config.get("phase1a", {})
    calibration_context = TimingPerturbationCalibration(
        namespace=str(
            physical.get("calibration_namespace", "native-sensitivity-internal-calibration")
        ),
        cycles=int(physical.get("timing_perturbation_cycles", 0)),
        label=int(physical.get("timing_perturbation_label", 1)),
    )
    session_count = int(
        config.get("native_sensitivity", {}).get("session_count", physical.get("session_count", 4))
    )
    campaign_id = f"native-sensitivity-{seed:010d}"
    source_root = dataset_dir / "sessions"
    source_dirs: list[Path] = []
    for session_index in range(session_count):
        session_id = f"session-{seed:010d}-{session_index:04d}"
        source_dir = source_root / session_id
        materialized = _materialize_session(
            config,
            source_dir,
            condition=condition,
            campaign_id=campaign_id,
            session_index=session_index,
            session_id=session_id,
            host_inventory_snapshot=collect_inventory(),
            calibration_context=calibration_context,
        )
        source_dirs.append(Path(materialized.pop("_materialized_source_dir", str(source_dir))))
    from .datasets import combine_datasets

    return combine_datasets(
        source_dirs,
        dataset_dir,
        config=config,
        condition=condition,
        campaign_id=campaign_id,
        source_manifest_paths=[str(path / "dataset.json") for path in source_dirs],
    )


def _session_dependence(
    traces: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    session_field = (
        "acquisition_session_id" if "acquisition_session_id" in metadata else "session_id"
    )
    if session_field not in metadata:
        return {"status": "unavailable", "reason": "acquisition session metadata is absent"}
    measurement = np.median(np.asarray(traces[indices], dtype=np.float64), axis=1)
    sessions = np.asarray(metadata[session_field][indices]).astype(str)
    local_labels = np.asarray(labels[indices], dtype=np.uint8)
    result: dict[str, Any] = {"status": "reported_diagnostic", "sessions": {}}
    for session in sorted(set(sessions)):
        mask = sessions == session
        values = {
            str(label): float(np.mean(measurement[mask & (local_labels == label)]))
            for label in [0, 1]
            if np.any(mask & (local_labels == label))
        }
        result["sessions"][session] = {
            "sample_count": int(np.sum(mask)),
            "label_means": values,
            "label_1_minus_label_0": (
                values.get("1", float("nan")) - values.get("0", float("nan"))
                if "0" in values and "1" in values
                else float("nan")
            ),
        }
    result["session_count"] = len(result["sessions"])
    return result


def _timing_perturbation_observation(
    traces: np.ndarray, labels: np.ndarray, metadata: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Summarize the measured, paired effect of the artificial delay.

    The requested TSC cycles are an input to the native primitive, not a
    latency claim.  This summary is computed from the retained raw trace
    rows, pairing the two observations that share a trial-pair identity.
    Invalid or incomplete pairs are counted and reported; their raw rows are
    not deleted or rewritten.
    """

    from .acquisition.native import summarize_measurements

    pair_field = "trial_pair_id" if "trial_pair_id" in metadata else "pair_id"
    if pair_field not in metadata:
        return {
            "status": "unavailable",
            "reason": "timing perturbation pairing metadata is absent",
            "raw_measurements_retained": True,
            "raw_measurement_rows": int(len(traces)),
        }
    traces = np.asarray(traces, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    pair_values = np.asarray(metadata[pair_field]).astype(str)
    if len(traces) != len(labels) or len(pair_values) != len(labels):
        raise ValueError("timing perturbation inputs must have matching rows")

    requested_values = {
        int(value)
        for value in np.asarray(metadata.get("timing_perturbation_cycles", []), dtype=object)
    }
    perturbation_labels = {
        int(value)
        for value in np.asarray(metadata.get("timing_perturbation_label", []), dtype=object)
    }
    if not requested_values:
        return {
            "status": "unavailable",
            "reason": "requested timing perturbation metadata is absent",
            "raw_measurements_retained": True,
            "raw_measurement_rows": int(len(traces)),
        }
    if len(requested_values) != 1 or len(perturbation_labels) > 1:
        return {
            "status": "unavailable",
            "reason": "timing perturbation metadata is not constant within the dataset",
            "requested_delay_cycles": sorted(requested_values),
            "raw_measurements_retained": True,
            "raw_measurement_rows": int(len(traces)),
        }
    requested_delay = next(iter(requested_values))
    perturbation_label = next(iter(perturbation_labels), 1)
    applied_values = np.asarray(metadata.get("timing_perturbation_applied", []), dtype=object)

    def as_bool(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().casefold() in {"true", "1", "yes"}

    applied = np.asarray([as_bool(value) for value in applied_values], dtype=bool)
    if len(applied) != len(labels):
        applied = (
            labels == perturbation_label if requested_delay else np.zeros(len(labels), dtype=bool)
        )

    rows_by_pair: dict[str, list[int]] = {}
    for index, pair in enumerate(pair_values):
        rows_by_pair.setdefault(pair, []).append(index)
    deltas: list[float] = []
    excluded_pairs = 0
    for indices in rows_by_pair.values():
        if len(indices) != 2 or set(labels[indices].tolist()) != {0, 1}:
            excluded_pairs += 1
            continue
        applied_indices = [index for index in indices if applied[index]]
        if len(applied_indices) == 1:
            perturbed_index = applied_indices[0]
            control_index = indices[0] if indices[1] == perturbed_index else indices[1]
        else:
            perturbed = [index for index in indices if labels[index] == perturbation_label]
            if len(perturbed) != 1:
                excluded_pairs += 1
                continue
            perturbed_index = perturbed[0]
            control_index = indices[0] if indices[1] == perturbed_index else indices[1]
        deltas.append(float(np.median(traces[perturbed_index]) - np.median(traces[control_index])))

    delta_values = np.asarray(deltas, dtype=np.float64)
    observed_summary = summarize_measurements(delta_values)
    error_values = delta_values - requested_delay
    error_summary = summarize_measurements(error_values)
    return {
        "status": "available" if len(delta_values) else "unavailable",
        "pair_field": pair_field,
        "requested_delay_cycles": requested_delay,
        "perturbation_label": perturbation_label,
        "pair_count": int(len(rows_by_pair)),
        "paired_observation_count": int(len(delta_values)),
        "excluded_pair_count": int(excluded_pairs),
        "observed_added_latency_cycles": observed_summary,
        "observed_added_latency_distribution": observed_summary,
        "observed_minus_requested_latency_cycles": error_summary,
        "raw_measurements_retained": True,
        "raw_measurement_rows": int(len(traces)),
        "outlier_filtering": "none; all raw trace rows remain in the dataset",
    }


def _evaluate_native_sensitivity_dataset(
    config: dict[str, Any], dataset_dir: Path, *, seed: int
) -> dict[str, Any]:
    traces, labels, metadata, _shards, manifest = load_dataset(dataset_dir)
    hierarchy = phase1a_split_hierarchy(
        metadata, dataset_fingerprint=manifest["dataset_fingerprint"], seed=seed
    )
    invariants = validate_phase1a_split_hierarchy(metadata, hierarchy)
    record: dict[str, Any] = {
        "dataset": manifest,
        "split_hierarchy": {
            name: {"status": item.get("status"), "reason": item.get("reason")}
            for name, item in hierarchy.items()
        },
        "split_hierarchy_invariants": invariants,
        "feature_policy": "trace-derived features only; all identity/audit metadata excluded",
    }
    d_record = hierarchy.get("D_unseen_acquisition_session", {})
    if d_record.get("status") != "available":
        record.update(
            {
                "status": "unavailable",
                "reason": d_record.get("reason", "D split unavailable"),
            }
        )
        return record
    split = d_record["split"]
    partitions = partition_indices(metadata, split)
    features = build_feature_matrix(traces, metadata)
    ci_unit = str(config.get("reporting", {}).get("ci_unit", "acquisition_session_id"))
    if ci_unit != "sample" and ci_unit not in metadata:
        ci_unit = "sample"
    model_results = {
        name: train_and_evaluate(
            name,
            traces,
            features,
            labels,
            partitions,
            seeds=[int(value) for value in config.get("training", {}).get("seeds", [11])],
            dataset_fingerprint=manifest["dataset_fingerprint"],
            split_fingerprint=split["split_fingerprint"],
            epochs=int(config.get("training", {}).get("epochs", 10)),
            patience=int(config.get("training", {}).get("early_stopping_patience", 2)),
            batch_size=int(config.get("training", {}).get("batch_size", 128)),
            groups=None if ci_unit == "sample" else metadata[ci_unit],
            ci_unit=ci_unit,
            bootstrap_repetitions=int(
                config.get("reporting", {}).get("bootstrap_repetitions", 100)
            ),
        )
        for name in _enabled_models(config)
    }
    metric_values = {
        f"{name}.{metric}": float(result["summary"][f"{metric}_mean"])
        for name, result in model_results.items()
        for metric in SUPPORTED_METRICS
    }
    maximum, component = max_statistic(metric_values)
    test_indices = partitions["test"]
    test_metadata = {key: values[test_indices] for key, values in metadata.items()}
    paired = paired_delta_analysis(
        traces[test_indices],
        labels[test_indices],
        test_metadata,
        repetitions=int(config.get("reporting", {}).get("paired_repetitions", 500)),
        seed=seed + 5000,
    )
    audits = run_leakage_audits(
        labels,
        metadata,
        features,
        partitions,
        dataset_fingerprint=manifest["dataset_fingerprint"],
        split_fingerprint=split["split_fingerprint"],
        traces=traces,
        seed=seed + 7000,
    )
    boot_field = (
        np.asarray(metadata["boot_id"][test_indices]).astype(str)
        if "boot_id" in metadata
        else np.asarray([])
    )
    record.update(
        {
            "status": "available",
            "split": split,
            "models": model_results,
            "metric_values": metric_values,
            "max_statistic": maximum,
            "max_component": component,
            "paired_statistics": paired,
            "audits": audits,
            "session_dependence": _session_dependence(traces, labels, metadata, test_indices),
            "timing_perturbation_observation": _timing_perturbation_observation(
                traces, labels, metadata
            ),
            "boot_dependence": {
                "status": "reported_diagnostic" if len(set(boot_field)) > 1 else "unavailable",
                "boot_count": len(set(boot_field)),
                "reason": "calibration uses one OS boot unless a multi-boot dataset is supplied",
            },
            "test_composition": {
                "rows": int(len(test_indices)),
                "session_count": int(
                    len(
                        set(
                            np.asarray(
                                metadata.get("acquisition_session_id", metadata["session_id"])
                            )[test_indices].astype(str)
                        )
                    )
                ),
                "boot_count": len(set(boot_field)),
            },
        }
    )
    return record


def _sensitivity_curve(
    records: dict[float, list[dict[str, Any]]],
    *,
    critical_max_statistic: float,
    alpha: float,
) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for magnitude, values in sorted(records.items()):
        finite = np.asarray(
            [float(item["max_statistic"]) for item in values if item.get("status") == "available"],
            dtype=np.float64,
        )
        positives = int(np.sum(finite >= critical_max_statistic))
        model_metrics: dict[str, Any] = {}
        for item in values:
            for name, score in item.get("metric_values", {}).items():
                model_metrics.setdefault(name, []).append(float(score))
        curve[str(magnitude)] = {
            "magnitude_cycles": magnitude,
            "replicates": int(len(finite)),
            "power_denominator": int(len(finite)),
            "detected_replicates": positives,
            "empirical_power": float(positives / len(finite)) if len(finite) else float("nan"),
            "empirical_power_wilson_interval_95": wilson_interval(positives, len(finite))
            if len(finite)
            else [float("nan"), float("nan")],
            "critical_max_statistic": critical_max_statistic,
            "alpha": alpha,
            "metrics": {
                name: {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                }
                for name, scores in model_metrics.items()
            },
            "paired_statistics": [item.get("paired_statistics", {}) for item in values],
            "records": values,
        }
    return curve


def run_native_sensitivity_calibration(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    run_id: str | None = None,
    development_magnitudes: list[int] | None = None,
    development_replicates: int | None = None,
    validation_replicates: int | None = None,
) -> dict[str, Any]:
    """Measure the native worker pipeline's artificial timing detection floor.

    Development magnitudes are predeclared and used only to select a frozen
    candidate.  Fresh validation uses new seeds and the development critical
    value exactly once; it never retunes the threshold or model.
    """

    from .acquisition.native import NativeMeasurementKernel
    from .phase1a import _materialize_label_permutation

    sensitivity = config.get("native_sensitivity", {})
    magnitudes = sorted(
        set(
            int(value)
            for value in (
                development_magnitudes
                if development_magnitudes is not None
                else sensitivity.get("development_magnitudes_cycles", [0, 32, 64, 128, 256, 512])
            )
        )
    )
    if not magnitudes or magnitudes[0] != 0 or any(value < 0 for value in magnitudes):
        raise ValueError("native sensitivity magnitudes must include zero and be non-negative")
    dev_replicates = int(
        development_replicates
        if development_replicates is not None
        else sensitivity.get("development_replicates", 3)
    )
    dev_null_replicates = int(sensitivity.get("development_null_replicates", dev_replicates))
    dev_shuffled_replicates = int(
        sensitivity.get("development_shuffled_replicates", dev_replicates)
    )
    fresh_replicates = int(
        validation_replicates
        if validation_replicates is not None
        else sensitivity.get("validation_replicates", 5)
    )
    if min(dev_replicates, dev_null_replicates, dev_shuffled_replicates, fresh_replicates) < 2:
        raise ValueError("native sensitivity development and validation replicates must be >= 2")
    kernel = NativeMeasurementKernel.load()
    if kernel is None or not kernel.supports_clflush:
        return {
            "schema": "sensetrace.native-sensitivity-report.v3",
            "status": "unavailable",
            "reason": "native x86 CLFLUSH measurement path is unavailable",
        }

    run_id = run_id or new_run_id("native-sensitivity")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    base_seed = int(config.get("experiment", {}).get("seed", 1337))
    alpha = float(sensitivity.get("alpha", 0.05))
    if not 0.0 < alpha < 1.0:
        raise ValueError("native sensitivity alpha must be strictly between zero and one")
    target_power = float(sensitivity.get("target_power", 0.8))
    protocol = {
        "version": "native-sensitivity-protocol-v2",
        "namespace": "native_path_sensitivity_calibration",
        "mechanism": (
            "native timed load uses one delayed-capable primitive; an LFENCE follows the "
            "volatile load before the TSC-deadline delay for label=1"
        ),
        "cache_control": "CLFLUSH plus MFENCE, using the same worker acquisition path",
        "magnitudes_cycles": magnitudes,
        "null_magnitude_cycles": 0,
        "development_replicates": dev_replicates,
        "development_null_replicates": dev_null_replicates,
        "development_shuffled_replicates": dev_shuffled_replicates,
        "fresh_validation_replicates": fresh_replicates,
        "holdout": "D_unseen_acquisition_session",
        "model_rule": "empirical maximum statistic across enabled model/metric summaries",
        "alpha": alpha,
        "target_power": target_power,
        "selection_rule": "smallest positive development magnitude with empirical power >= target_power",
        "threshold_method": "conservative empirical order statistic with plus-one p-values",
        "threshold_alpha_support_rule": (
            "alpha is supportable only when 1/(development_null_replicates + 1) <= alpha"
        ),
        "seed_streams": {
            "null": "replicate_seed",
            "positive": "replicate_seed + 1000003 + requested_magnitude_cycles",
            "shuffled": "label permutation of its positive observation stream; evaluator seed is positive seed + 7919",
        },
        "fresh_validation_is_frozen": True,
        "shuffled_labels": "same-observation label permutation control, never a model feature",
        "replicate_quality": "report empirical tail resolution and warn when null counts are coarse",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(
        run_dir / "run.json",
        {
            "schema": "sensetrace.run.v2",
            "run_id": run_id,
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(config),
            "protocol_hash": protocol_hash,
            "claim_scope": "native instrumentation sensitivity only; no physical DRAM evidence",
        },
    )
    _write_json(run_dir / "host.json", collect_inventory())
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "protocol.json", {**protocol, "protocol_hash": protocol_hash})
    journal = Journal(run_dir / "events.jsonl")
    journal.append("native_sensitivity_calibration_started", protocol_hash=protocol_hash)

    # Keep these ensembles separate by construction.  In particular, a
    # shuffled-label statistic from frozen validation must never be able to
    # populate a development report through a reused generic variable.
    development_null_records: list[dict[str, Any]] = []
    development_positive_records: dict[str, list[dict[str, Any]]] = {}
    development_shuffled_records_by_magnitude: dict[str, list[dict[str, Any]]] = {}
    for replicate in range(max(dev_replicates, dev_null_replicates, dev_shuffled_replicates)):
        replicate_seed = base_seed + 100003 * (replicate + 1)
        if replicate < dev_null_replicates:
            null_config = _native_sensitivity_config(
                config,
                cycles=0,
                namespace=f"native-sensitivity:development:null:{replicate:04d}",
                seed=replicate_seed,
            )
            null_dir = run_dir / "development" / "null" / f"replicate-{replicate:04d}"
            null_manifest = _materialize_native_sensitivity_dataset(
                null_config, null_dir, condition="native_sensitivity_null", seed=replicate_seed
            )
            null_record = _evaluate_native_sensitivity_dataset(
                null_config, null_dir, seed=replicate_seed
            )
            null_record["dataset"]["dataset_fingerprint"] = null_manifest["dataset_fingerprint"]
            development_null_records.append(null_record)
        for magnitude in magnitudes:
            if replicate >= max(dev_replicates, dev_shuffled_replicates):
                continue
            positive_seed = replicate_seed + 1_000_003 + magnitude
            positive_dir = (
                run_dir / "development" / f"injected-{magnitude:08d}" / f"replicate-{replicate:04d}"
            )
            positive_config = _native_sensitivity_config(
                config,
                cycles=magnitude,
                namespace=f"native-sensitivity:development:{magnitude}:{replicate:04d}",
                seed=positive_seed,
            )
            positive_manifest = _materialize_native_sensitivity_dataset(
                positive_config,
                positive_dir,
                condition=f"native_sensitivity_injected_{magnitude}",
                seed=positive_seed,
            )
            if replicate < dev_replicates:
                positive_record = _evaluate_native_sensitivity_dataset(
                    positive_config, positive_dir, seed=positive_seed
                )
                positive_record["dataset"]["dataset_fingerprint"] = positive_manifest[
                    "dataset_fingerprint"
                ]
                positive_record["perturbation_cycles"] = magnitude
                development_positive_records.setdefault(str(magnitude), []).append(positive_record)
            if replicate < dev_shuffled_replicates:
                shuffled_dir = (
                    run_dir
                    / "development"
                    / f"shuffled-{magnitude:08d}"
                    / f"replicate-{replicate:04d}"
                )
                shuffled_manifest = _materialize_label_permutation(
                    positive_dir,
                    shuffled_dir,
                    positive_config,
                    positive_seed + 7919,
                )
                shuffled_record = _evaluate_native_sensitivity_dataset(
                    positive_config, shuffled_dir, seed=positive_seed + 7919
                )
                shuffled_record["dataset"]["dataset_fingerprint"] = shuffled_manifest[
                    "dataset_fingerprint"
                ]
                shuffled_record["perturbation_cycles"] = magnitude
                development_shuffled_records_by_magnitude.setdefault(str(magnitude), []).append(
                    shuffled_record
                )

    development_null_statistics = _record_statistics(
        development_null_records,
        require_complete=True,
        expected_count=dev_null_replicates,
        ensemble="development null",
    )
    development_positive_statistics: dict[float, np.ndarray] = {}
    for magnitude in magnitudes:
        development_positive_statistics[float(magnitude)] = _record_statistics(
            development_positive_records.get(str(magnitude), []),
            require_complete=True,
            expected_count=dev_replicates,
            ensemble=f"development positive magnitude {magnitude}",
        )
    development_shuffled_records = [
        record
        for magnitude in magnitudes
        for record in development_shuffled_records_by_magnitude.get(str(magnitude), [])
    ]
    development_shuffled_statistics = _record_statistics(
        development_shuffled_records,
        require_complete=True,
        expected_count=len(magnitudes) * dev_shuffled_replicates,
        ensemble="development shuffled-label controls",
    )
    threshold_calibration = _empirical_null_threshold(development_null_statistics, alpha=alpha)
    critical = float(threshold_calibration["critical_statistic"])
    dev_positive = {
        float(magnitude): development_positive_records[str(magnitude)] for magnitude in magnitudes
    }
    power_curve = _sensitivity_curve(dev_positive, critical_max_statistic=critical, alpha=alpha)
    selected = next(
        (
            magnitude
            for magnitude in magnitudes
            if magnitude > 0
            and np.isfinite(power_curve[str(float(magnitude))]["empirical_power"])
            and power_curve[str(float(magnitude))]["empirical_power"] >= target_power
        ),
        None,
    )
    frozen_levels = sorted(
        set(
            [0]
            + ([selected] if selected is not None else [])
            + ([magnitudes[-1]] if magnitudes[-1] != selected else [])
        )
    )
    # Materialize immutable local copies before entering fresh validation.  No
    # fresh statistic can retune either the critical value or the selected
    # perturbation magnitude.
    frozen_validation_critical = critical
    frozen_validation_magnitudes = tuple(frozen_levels)
    frozen_validation_null_records: list[dict[str, Any]] = []
    frozen_validation_positive_records: dict[str, list[dict[str, Any]]] = {}
    frozen_validation_shuffled_records_by_magnitude: dict[str, list[dict[str, Any]]] = {}
    for replicate in range(fresh_replicates):
        replicate_seed = base_seed + 9000001 + 100003 * replicate
        null_config = _native_sensitivity_config(
            config,
            cycles=0,
            namespace=f"native-sensitivity:frozen:null:{replicate:04d}",
            seed=replicate_seed,
        )
        null_dir = run_dir / "frozen-validation" / "null" / f"replicate-{replicate:04d}"
        null_manifest = _materialize_native_sensitivity_dataset(
            null_config, null_dir, condition="native_sensitivity_null", seed=replicate_seed
        )
        null_record = _evaluate_native_sensitivity_dataset(
            null_config, null_dir, seed=replicate_seed
        )
        null_record["dataset"]["dataset_fingerprint"] = null_manifest["dataset_fingerprint"]
        frozen_validation_null_records.append(null_record)
        for magnitude in frozen_validation_magnitudes:
            positive_seed = replicate_seed + 1_000_003 + magnitude
            positive_config = _native_sensitivity_config(
                config,
                cycles=magnitude,
                namespace=f"native-sensitivity:frozen:{magnitude}:{replicate:04d}",
                seed=positive_seed,
            )
            positive_dir = (
                run_dir
                / "frozen-validation"
                / f"injected-{magnitude:08d}"
                / f"replicate-{replicate:04d}"
            )
            positive_manifest = _materialize_native_sensitivity_dataset(
                positive_config,
                positive_dir,
                condition=f"native_sensitivity_injected_{magnitude}",
                seed=positive_seed,
            )
            positive_record = _evaluate_native_sensitivity_dataset(
                positive_config, positive_dir, seed=positive_seed
            )
            positive_record["dataset"]["dataset_fingerprint"] = positive_manifest[
                "dataset_fingerprint"
            ]
            positive_record["perturbation_cycles"] = magnitude
            frozen_validation_positive_records.setdefault(str(magnitude), []).append(
                positive_record
            )
            shuffled_dir = (
                run_dir
                / "frozen-validation"
                / f"shuffled-{magnitude:08d}"
                / f"replicate-{replicate:04d}"
            )
            shuffled_manifest = _materialize_label_permutation(
                positive_dir, shuffled_dir, positive_config, positive_seed + 7919
            )
            shuffled_record = _evaluate_native_sensitivity_dataset(
                positive_config, shuffled_dir, seed=positive_seed + 7919
            )
            shuffled_record["dataset"]["dataset_fingerprint"] = shuffled_manifest[
                "dataset_fingerprint"
            ]
            shuffled_record["perturbation_cycles"] = magnitude
            frozen_validation_shuffled_records_by_magnitude.setdefault(str(magnitude), []).append(
                shuffled_record
            )

    frozen_validation_null_statistics = _record_statistics(
        frozen_validation_null_records,
        require_complete=True,
        expected_count=fresh_replicates,
        ensemble="frozen-validation null",
    )
    frozen_validation_positive_records_by_float = {
        float(magnitude): frozen_validation_positive_records[str(magnitude)]
        for magnitude in frozen_validation_magnitudes
    }
    frozen_validation_positive_statistics: dict[float, np.ndarray] = {
        magnitude: _record_statistics(
            frozen_validation_positive_records[str(int(magnitude))],
            require_complete=True,
            expected_count=fresh_replicates,
            ensemble=f"frozen-validation positive magnitude {int(magnitude)}",
        )
        for magnitude in frozen_validation_magnitudes
    }
    frozen_validation_shuffled_records = [
        record
        for magnitude in frozen_validation_magnitudes
        for record in frozen_validation_shuffled_records_by_magnitude[str(int(magnitude))]
    ]
    frozen_validation_shuffled_statistics = _record_statistics(
        frozen_validation_shuffled_records,
        require_complete=True,
        expected_count=len(frozen_validation_magnitudes) * fresh_replicates,
        ensemble="frozen-validation shuffled-label controls",
    )
    fresh_curve = _sensitivity_curve(
        frozen_validation_positive_records_by_float,
        critical_max_statistic=frozen_validation_critical,
        alpha=alpha,
    )
    minimum_recommended_null = int(sensitivity.get("minimum_recommended_null_replicates", 20))
    development_shuffled_summary = _shuffled_false_positive_summary(
        development_shuffled_records,
        critical_max_statistic=critical,
        source="development shuffled-label controls only",
        ensemble="development",
        alpha=alpha,
        minimum_recommended_null=minimum_recommended_null,
        expected_replicates=len(magnitudes) * dev_shuffled_replicates,
    )
    fresh_shuffled_summary = _shuffled_false_positive_summary(
        frozen_validation_shuffled_records,
        critical_max_statistic=frozen_validation_critical,
        source="fresh/frozen shuffled-label controls only",
        ensemble="fresh_frozen_validation",
        alpha=alpha,
        minimum_recommended_null=minimum_recommended_null,
        expected_replicates=len(frozen_validation_magnitudes) * fresh_replicates,
    )
    development_false_positive_rate = _false_positive_rate(
        development_null_statistics, critical, alpha
    ) | {
        "source": "development zero-magnitude null controls only",
        "ensemble": "development_null",
    }
    frozen_validation_false_positive_rate = _false_positive_rate(
        frozen_validation_null_statistics, frozen_validation_critical, alpha
    ) | {
        "source": "fresh/frozen zero-magnitude null controls only",
        "ensemble": "frozen_validation_null",
    }
    ensemble_provenance = {
        "development_null": _ensemble_provenance(
            development_null_records,
            ensemble="development_null",
            expected_count=dev_null_replicates,
        ),
        "development_positive": {
            str(magnitude): _ensemble_provenance(
                development_positive_records[str(magnitude)],
                ensemble=f"development_positive_{magnitude}_cycles",
                expected_count=dev_replicates,
            )
            for magnitude in magnitudes
        },
        "development_shuffled": _ensemble_provenance(
            development_shuffled_records,
            ensemble="development_shuffled",
            expected_count=len(magnitudes) * dev_shuffled_replicates,
        ),
        "frozen_validation_null": _ensemble_provenance(
            frozen_validation_null_records,
            ensemble="frozen_validation_null",
            expected_count=fresh_replicates,
        ),
        "frozen_validation_positive": {
            str(magnitude): _ensemble_provenance(
                frozen_validation_positive_records[str(int(magnitude))],
                ensemble=f"frozen_validation_positive_{int(magnitude)}_cycles",
                expected_count=fresh_replicates,
            )
            for magnitude in frozen_validation_magnitudes
        },
        "frozen_validation_shuffled": _ensemble_provenance(
            frozen_validation_shuffled_records,
            ensemble="frozen_validation_shuffled",
            expected_count=len(frozen_validation_magnitudes) * fresh_replicates,
        ),
    }
    empirically_calibrated_floor = selected if threshold_calibration["alpha_supported"] else None
    report = {
        "schema": "sensetrace.native-sensitivity-report.v3",
        "status": "complete",
        "run_id": run_id,
        "protocol": {**protocol, "protocol_hash": protocol_hash},
        "kernel": kernel.provenance(),
        "threshold_calibration": threshold_calibration,
        "pilot_detection_floor": selected,
        "empirically_alpha_calibrated_detection_floor": empirically_calibrated_floor,
        "detection_floor_interpretation": (
            "empirically alpha-calibrated detection floor"
            if empirically_calibrated_floor is not None
            else "pilot threshold only; the development null ensemble cannot resolve the requested alpha"
        ),
        "raw_measurements_retained": True,
        "timing_perturbation_observations": {
            "development_positive": {
                str(magnitude): [
                    record.get("timing_perturbation_observation", {})
                    for record in development_positive_records[str(magnitude)]
                ]
                for magnitude in magnitudes
            },
            "frozen_validation_positive": {
                str(int(magnitude)): [
                    record.get("timing_perturbation_observation", {})
                    for record in frozen_validation_positive_records[str(int(magnitude))]
                ]
                for magnitude in frozen_validation_magnitudes
            },
        },
        "ensemble_provenance": ensemble_provenance,
        "ensemble_statistics": {
            "development_null": development_null_statistics.tolist(),
            "development_positive": {
                str(magnitude): statistics.tolist()
                for magnitude, statistics in development_positive_statistics.items()
            },
            "development_shuffled": development_shuffled_statistics.tolist(),
            "frozen_validation_null": frozen_validation_null_statistics.tolist(),
            "frozen_validation_positive": {
                str(magnitude): statistics.tolist()
                for magnitude, statistics in frozen_validation_positive_statistics.items()
            },
            "frozen_validation_shuffled": frozen_validation_shuffled_statistics.tolist(),
        },
        "development": {
            "null_max_statistic": development_null_statistics.tolist(),
            "null_statistics_source": "ensemble_provenance.development_null",
            "shuffled_statistics_source": "ensemble_provenance.development_shuffled",
            "positive_statistics_source": "ensemble_provenance.development_positive",
            "threshold_calibration": threshold_calibration,
            "critical_max_statistic": critical,
            "false_positive_rate": development_false_positive_rate,
            "power_curve": power_curve,
            "shuffled_false_positive_rate": development_shuffled_summary,
            "statistics_quality": {
                "null": _replicate_quality(
                    statistics=development_null_statistics,
                    alpha=alpha,
                    minimum_recommended=minimum_recommended_null,
                ),
                "shuffled": _replicate_quality(
                    statistics=development_shuffled_statistics,
                    alpha=alpha,
                    minimum_recommended=minimum_recommended_null,
                ),
            },
        },
        "frozen_selection": {
            "selected_magnitude_cycles": selected,
            "pilot_detection_floor_cycles": selected,
            "empirically_alpha_calibrated_detection_floor_cycles": empirically_calibrated_floor,
            "frozen_validation_magnitudes_cycles": list(frozen_validation_magnitudes),
            "selection_rule_applied_once": True,
            "selection_uses_fresh_validation": False,
            "selection_frozen_before_fresh_validation": True,
        },
        "fresh_frozen_validation": {
            "critical_max_statistic_from_development": frozen_validation_critical,
            "critical_value_source": "development zero-magnitude null ensemble",
            "threshold_reused_without_recalibration": True,
            "positive_statistics_source": "ensemble_provenance.frozen_validation_positive",
            "shuffled_statistics_source": "ensemble_provenance.frozen_validation_shuffled",
            "power_curve": fresh_curve,
            "null_false_positive_rate": frozen_validation_false_positive_rate,
            "shuffled_control_false_positive_rate": fresh_shuffled_summary,
            "statistics_quality": {
                "null": _replicate_quality(
                    statistics=frozen_validation_null_statistics,
                    alpha=alpha,
                    minimum_recommended=minimum_recommended_null,
                ),
                "shuffled": fresh_shuffled_summary["statistics_quality"],
            },
            "datasets_are_fresh_and_separately_seeded": True,
        },
        "replicate_counts": {
            "development_positive_per_magnitude": dev_replicates,
            "development_null": int(len(development_null_statistics)),
            "development_shuffled_total": int(len(development_shuffled_statistics)),
            "fresh_null": int(len(frozen_validation_null_statistics)),
            "fresh_shuffled_total": fresh_shuffled_summary["replicates"],
        },
        "claim_boundary": (
            "This is a positive-control calibration of the native timing and analysis path. "
            "It is artificial timing sensitivity, not evidence of physical DRAM-state inference."
        ),
    }
    _write_json(run_dir / "metrics.json", report)
    journal.append("native_sensitivity_calibration_completed", selected_magnitude_cycles=selected)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run.update({"status": "completed", "completed_at": datetime.now(UTC).isoformat()})
    _write_json(run_dir / "run.json", run)
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
    calibration_role: str = "final_gate_validation",
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
        "schema": (
            "sensetrace.phase0-calibration-report.v2"
            if protocol["version"] == "phase0-protocol-v2"
            else "sensetrace.phase0-calibration-report.v1"
        ),
        "calibration_role": calibration_role,
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
            "injected_detection_wilson_interval_95": wilson_interval(
                injected_positives, len(fresh_injected)
            ),
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


def run_phase0_power_study(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    sample_counts: list[int] | None = None,
    candidate_replicates: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Select Phase 0 v2 power from development ensembles only.

    Each candidate is a complete, independently materialized calibration with
    its own null, shuffled, injected, and fresh validation ensembles. Candidate
    reports are development evidence; the selected sample count must be used in
    a subsequent separately seeded final gate run.
    """

    base_calibration = config.get("calibration", {})
    configured_study = base_calibration.get("power_study", {})
    counts = [
        int(value)
        for value in (sample_counts or configured_study.get("sample_counts", [1000, 2000, 4000]))
    ]
    if not counts or any(value < 2 or value % 2 for value in counts):
        raise ValueError("power-study sample_counts must be positive even integers")
    replicates = int(candidate_replicates or configured_study.get("replicates", 20))
    if replicates < 2:
        raise ValueError("power-study replicates must be at least two")
    run_id = run_id or new_run_id("phase0-power-study")
    root = Path(output_root) / run_id
    root.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, Any]] = []
    base_seed = int(config.get("experiment", {}).get("seed", 1337))
    target_rate = float(base_calibration.get("minimum_injected_detection_rate", 0.8))
    target_strength = float(
        base_calibration.get(
            "target_injection_strength",
            config.get("controls", {}).get("injected_weak_signal", {}).get("amplitude_sigma", 0.1),
        )
    )
    development_permutations = int(
        configured_study.get(
            "permutation_repetitions",
            min(int(base_calibration.get("permutation_repetitions", 100)), 20),
        )
    )
    for index, count in enumerate(sorted(set(counts))):
        candidate = json.loads(json.dumps(config))
        candidate.setdefault("calibration", {})["protocol_version"] = "phase0-protocol-v2"
        candidate["calibration"]["samples"] = count
        candidate["calibration"]["null_replicates"] = replicates
        candidate["calibration"]["shuffled_replicates"] = replicates
        candidate["calibration"]["injected_replicates"] = replicates
        candidate["calibration"]["gate_validation_replicates"] = replicates
        # The development study is about power at one declared target effect;
        # the optional strength curve belongs outside candidate selection.
        candidate["calibration"]["injected_levels"] = [target_strength]
        candidate["calibration"]["permutation_repetitions"] = development_permutations
        candidate["calibration"]["_power_study_candidate"] = True
        candidate.setdefault("experiment", {})["seed"] = base_seed + 1000003 * (index + 1)
        report = run_phase0_calibration(
            candidate,
            root / "candidates",
            run_id=f"sample-{count:06d}",
            calibration_role="development_power_candidate",
        )
        acceptance = report["acceptance"]
        injected_replicates = int(report["counts"]["fresh_gate_validation_injected_replicates"])
        detection_rate = float(acceptance["injected_detection_rate"])
        detection_interval = wilson_interval(
            int(round(detection_rate * injected_replicates)), injected_replicates
        )
        candidates.append(
            {
                "sample_count": count,
                "run_id": report["run_id"],
                "protocol_hash": report["protocol_hash"],
                "fresh_injected_detection_rate": detection_rate,
                "fresh_injected_detection_wilson_interval_95": detection_interval,
                "fresh_null_false_positive_rate": report["fresh_gate_validation"][
                    "false_positive_rate"
                ],
                "fresh_shuffled_false_positive_rate": report["fresh_gate_validation"][
                    "shuffled_false_positive_rate"
                ],
                "selection_eligible": bool(
                    detection_rate >= target_rate
                    and acceptance["null_pass"]
                    and acceptance["shuffled_pass"]
                ),
                "candidate_report_path": str(
                    root / "candidates" / report["run_id"] / "metrics.json"
                ),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["selection_eligible"]]
    selected = (
        min(eligible, key=lambda item: item["sample_count"])
        if eligible
        else max(candidates, key=lambda item: item["sample_count"])
    )
    result = {
        "schema": "sensetrace.phase0-power-study.v1",
        "run_id": run_id,
        "protocol_version": "phase0-protocol-v2",
        "study_role": "development_only; no candidate gate result authorizes Phase 1A",
        "sample_count_grid": sorted(set(counts)),
        "candidate_replicates": replicates,
        "target_injection_strength": target_strength,
        "development_permutation_repetitions": development_permutations,
        "minimum_injected_detection_rate": target_rate,
        "candidates": candidates,
        "selected_design": selected,
        "selection_rule": (
            "smallest candidate meeting target injected detection and development null/shuffled checks; "
            "if none qualifies, largest candidate is recorded but the final gate remains a separate decision"
        ),
        "final_gate_requirement": "run one separately seeded phase0-protocol-v2 calibration after this study; do not reuse candidate validation data",
        "claim_boundary": "development power evidence only; no physical DRAM-state or Phase 1A claim",
    }
    _write_json(root / "power-study.json", result)
    return result
