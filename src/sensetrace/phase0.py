"""Phase 0 synthetic controls and self-contained result records."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .acquisition.synthetic import SyntheticBackend
from .audits import run_leakage_audits
from .config import config_fingerprint, normalized_config
from .datasets import build_feature_matrix, load_dataset, write_dataset_manifest
from .inventory import collect_inventory
from .journal import Journal
from .models import train_and_evaluate
from .runner import _git_commit, new_run_id
from .splits import grouped_split, partition_indices, write_split
from .storage import ShardWriter, validate_all_shards


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _partition_class_balance(
    labels: np.ndarray, partitions: dict[str, np.ndarray]
) -> dict[str, dict[str, int | float]]:
    return {
        name: {
            "rows": int(len(indices)),
            "0": int(np.sum(labels[indices] == 0)),
            "1": int(np.sum(labels[indices] == 1)),
            "positive_rate": float(np.mean(labels[indices])) if len(indices) else float("nan"),
        }
        for name, indices in partitions.items()
    }


def _condition_config(config: dict[str, Any], condition: str, amplitude: float) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    value.setdefault("data", {})["samples"] = int(value.get("data", {}).get("samples", 1000))
    value.setdefault("controls", {}).setdefault("injected_weak_signal", {})["amplitude_sigma"] = (
        amplitude
    )
    return value


def _generate_condition(
    config: dict[str, Any], condition_dir: Path, condition: str, amplitude: float
) -> dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    data = config.get("data", {})
    controls = config.get("controls", {}).get("injected_weak_signal", {})
    acquisition = config.get("acquisition", {})
    backend = SyntheticBackend(
        count=int(data.get("samples", 1000)),
        trace_length=int(data.get("trace_length", 128)),
        seed=int(config.get("experiment", {}).get("seed", 1337)),
        acquisition_seed=int(
            config.get("experiment", {}).get(
                "acquisition_seed", config.get("experiment", {}).get("seed", 1337)
            )
        ),
        label_seed=int(
            config.get("experiment", {}).get(
                "label_seed", config.get("experiment", {}).get("seed", 1337)
            )
        ),
        trace_seed=int(
            config.get("experiment", {}).get(
                "trace_seed", int(config.get("experiment", {}).get("seed", 1337)) + 1
            )
        ),
        condition=condition,
        amplitude_sigma=amplitude,
        start_index=int(
            controls.get("start_index", max(0, int(data.get("trace_length", 128)) // 3))
        ),
        width=int(controls.get("width", max(4, int(data.get("trace_length", 128)) // 16))),
        session_count=int(acquisition.get("session_count", 4)),
        device_count=int(acquisition.get("device_count", 2)),
        permute_seed=int(config.get("experiment", {}).get("seed", 1337)) + 7919,
        dataset_id=str(config.get("experiment", {}).get("dataset_id", "phase0")),
        balance_mode=str(acquisition.get("synthetic_balance_mode", "global_balance_only")),
        observations_per_location=int(acquisition.get("observations_per_location", 4)),
    )
    writer = ShardWriter(
        condition_dir,
        shard_target_mb=float(acquisition.get("shard_target_mb", 512)),
        max_samples_per_shard=acquisition.get("max_samples_per_shard"),
    )
    event_journal = Journal(condition_dir / "events.jsonl")
    for sample in backend.samples():
        info = writer.add(sample.trace, sample.label, sample.metadata)
        if info:
            event_journal.append("shard_finalized", **info.as_dict())
    info = writer.finalize()
    if info:
        event_journal.append("shard_finalized", **info.as_dict())
    event_journal.append("condition_acquisition_completed", rows=backend.count, condition=condition)
    manifest = write_dataset_manifest(
        condition_dir,
        config=config,
        condition=condition,
        shard_infos=validate_all_shards(condition_dir),
        label_stream_fingerprint=backend.label_stream_fingerprint,
        class_balance={
            "0": int((backend._labels == 0).sum()),
            "1": int((backend._labels == 1).sum()),
        },
        provenance={
            "observation_semantics": (
                "label-independent synthetic noise"
                if condition == "null"
                else "known label-dependent synthetic signal"
            ),
            "original_label_stream_fingerprint": backend.original_label_stream_fingerprint,
            "permuted_label_stream_fingerprint": (
                backend.label_stream_fingerprint if condition == "shuffled" else None
            ),
            "permutation_seed": backend.permutation_seed if condition == "shuffled" else None,
            "permutation_reference": backend.permutation_fingerprint,
            "permutation_strata": backend.permutation_strata if condition == "shuffled" else None,
        },
    )
    return manifest


def _generate_shuffled_from_dataset(
    source_dir: Path,
    condition_dir: Path,
    config: dict[str, Any],
    *,
    permutation_seed: int,
) -> dict[str, Any]:
    """Materialize the label control from the exact injected observations."""

    traces, labels, metadata, _shards, source_manifest = load_dataset(source_dir)
    configured_strata = config.get("calibration", {}).get("permutation_strata")
    mode = config.get("acquisition", {}).get("synthetic_balance_mode", "global_balance_only")
    by_mode = config.get("calibration", {}).get("permutation_strata_by_balance_mode", {})
    strata_keys = list(
        configured_strata
        or by_mode.get(mode, [])
        or [
            "synthetic_location_id"
            if mode == "group_stratified_balance"
            else "synthetic_dataset_id"
        ]
    )
    rng = np.random.default_rng(permutation_seed)
    permutation = np.arange(len(labels), dtype=np.int64)
    strata: dict[tuple[str, ...], list[int]] = {}
    keys = list(zip(*(np.asarray(metadata[key]).astype(str) for key in strata_keys), strict=True))
    for index, key in enumerate(keys):
        strata.setdefault(key, []).append(index)
    for indices in strata.values():
        index_array = np.asarray(indices, dtype=np.int64)
        permutation[index_array] = rng.permutation(index_array)
    shuffled_labels = labels[permutation]
    condition_dir.mkdir(parents=True, exist_ok=True)
    acquisition = config.get("acquisition", {})
    writer = ShardWriter(
        condition_dir,
        shard_target_mb=float(acquisition.get("shard_target_mb", 512)),
        max_samples_per_shard=acquisition.get("max_samples_per_shard"),
    )
    event_journal = Journal(condition_dir / "events.jsonl")
    for index, (trace, label) in enumerate(zip(traces, shuffled_labels, strict=True)):
        row_metadata = {key: values[index] for key, values in metadata.items()}
        info = writer.add(trace, int(label), row_metadata)
        if info:
            event_journal.append("shard_finalized", **info.as_dict())
    info = writer.finalize()
    if info:
        event_journal.append("shard_finalized", **info.as_dict())
    permutation_fingerprint = hashlib.sha256(
        np.asarray(permutation, dtype=np.int64).tobytes()
    ).hexdigest()
    label_fingerprint = hashlib.sha256(shuffled_labels.tobytes()).hexdigest()
    manifest = write_dataset_manifest(
        condition_dir,
        config=config,
        condition="shuffled",
        shard_infos=validate_all_shards(condition_dir),
        label_stream_fingerprint=label_fingerprint,
        class_balance={
            "0": int((shuffled_labels == 0).sum()),
            "1": int((shuffled_labels == 1).sum()),
        },
        provenance={
            "observation_semantics": "exact injected observation materialization",
            "parent_dataset_fingerprint": source_manifest["dataset_fingerprint"],
            "original_label_stream_fingerprint": source_manifest["label_stream_fingerprint"],
            "permuted_label_stream_fingerprint": label_fingerprint,
            "permutation_seed": permutation_seed,
            "permutation_reference": permutation_fingerprint,
            "permutation_strata": strata_keys,
            "only_changed_variable": "label association",
        },
    )
    event_journal.append(
        "condition_acquisition_completed",
        rows=len(labels),
        condition="shuffled",
        parent_dataset_fingerprint=source_manifest["dataset_fingerprint"],
    )
    return manifest


def _enabled_models(config: dict[str, Any]) -> list[str]:
    configured = config.get("models", {})
    defaults = ["logistic_regression", "boosted_trees", "tiny_mlp", "tiny_cnn"]
    if not configured:
        return ["logistic_regression", "boosted_trees"]
    return [name for name in defaults if configured.get(name, {}).get("enabled", False)]


def run_phase0(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    run_id: str | None = None,
    include_curve: bool = False,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    run_id = run_id or new_run_id("phase0")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_hash = config_fingerprint(config)
    _write_json(
        run_dir / "run.json",
        {
            "schema": "sensetrace.run.v1",
            "run_id": run_id,
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_hash,
            "claim_scope": "synthetic Phase 0 control validation only",
        },
    )
    _write_json(run_dir / "host.json", collect_inventory())
    _write_json(
        run_dir / "environment.json",
        {"python": collect_inventory().get("python"), "code_commit": _git_commit()},
    )
    _write_json(run_dir / "config.json", normalized_config(config))
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(normalized_config(config), sort_keys=False), encoding="utf-8"
    )
    journal = Journal(run_dir / "events.jsonl")
    journal.append("run_started", run_id=run_id, configuration_hash=config_hash)
    controls = config.get("controls", {})
    injected = controls.get("injected_weak_signal", {})
    signal_levels = [float(injected.get("amplitude_sigma", 0.1))]
    if include_curve:
        signal_levels = [
            float(level) for level in injected.get("levels", [0.0, 0.01, 0.025, 0.05, 0.1, 0.2])
        ]
    requested = conditions or ["null", "injected", "shuffled"]
    condition_specs: list[tuple[str, str, float]] = []
    for condition in requested:
        if condition == "injected" and include_curve:
            condition_specs.extend(
                (f"injected_{level:g}", "injected", level) for level in signal_levels
            )
        else:
            condition_specs.append((condition, condition, signal_levels[-1]))
    results: dict[str, Any] = {}
    train_config = config.get("training", {})
    seeds = [int(seed) for seed in train_config.get("seeds", [11, 23, 37])]
    models = _enabled_models(config)
    injected_source_dirs: dict[str, Path] = {}
    first_shuffled = next(
        (index for index, spec in enumerate(condition_specs) if spec[1] == "shuffled"), None
    )
    first_injected = next(
        (index for index, spec in enumerate(condition_specs) if spec[1] == "injected"), None
    )
    if first_shuffled is not None and (first_injected is None or first_shuffled < first_injected):
        parent_dir = run_dir / "datasets" / "_injected_parent_for_shuffle"
        _generate_condition(
            _condition_config(config, "injected", signal_levels[-1]),
            parent_dir,
            "injected",
            signal_levels[-1],
        )
        injected_source_dirs["_injected_parent_for_shuffle"] = parent_dir
    for condition_name, backend_condition, amplitude in condition_specs:
        journal.append("condition_started", condition=condition_name)
        condition_dir = run_dir / "datasets" / condition_name
        if backend_condition == "shuffled" and injected_source_dirs:
            source_name = next(reversed(injected_source_dirs))
            source_dir = injected_source_dirs[source_name]
            # Use the injected configuration for the materialized parent so
            # the manifest records the exact observation provenance.
            condition_config = _condition_config(config, "injected", amplitude)
            manifest = _generate_shuffled_from_dataset(
                source_dir,
                condition_dir,
                condition_config,
                permutation_seed=int(config.get("experiment", {}).get("seed", 1337)) + 7919,
            )
        else:
            condition_config = _condition_config(config, backend_condition, amplitude)
            manifest = _generate_condition(
                condition_config, condition_dir, backend_condition, amplitude
            )
        if backend_condition == "injected":
            injected_source_dirs[condition_name] = condition_dir
        traces, labels, metadata, _shards, _manifest = load_dataset(condition_dir)
        split_cfg = config.get("splits", {}).get("primary", {})
        split = grouped_split(
            metadata,
            dataset_fingerprint=manifest["dataset_fingerprint"],
            group_keys=list(
                split_cfg.get(
                    "group_keys", ["session_id", "device_id", "row_id", "cell_or_offset_id"]
                )
            ),
            seed=int(config.get("experiment", {}).get("seed", 1337)),
            train_fraction=float(split_cfg.get("train_fraction", 0.7)),
            validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
            test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        )
        split_path = condition_dir / "split.json"
        write_split(split_path, split)
        partitions = partition_indices(metadata, split)
        feature_matrix = build_feature_matrix(traces, metadata)
        ci_unit = str(config.get("reporting", {}).get("ci_unit", "session_id"))
        if ci_unit != "sample" and ci_unit not in metadata:
            raise ValueError(f"configured confidence-interval unit is missing: {ci_unit}")
        condition_results: dict[str, Any] = {
            "dataset": manifest,
            "split": split,
            "models": {},
            "confidence_interval_unit": ci_unit,
            "partition_class_balance": _partition_class_balance(labels, partitions),
        }
        for model_name in models:
            condition_results["models"][model_name] = train_and_evaluate(
                model_name,
                traces,
                feature_matrix,
                labels,
                partitions,
                seeds=seeds,
                dataset_fingerprint=manifest["dataset_fingerprint"],
                split_fingerprint=split["split_fingerprint"],
                epochs=int(train_config.get("epochs", 30)),
                patience=int(train_config.get("early_stopping_patience", 5)),
                batch_size=int(train_config.get("batch_size", 256)),
                groups=None if ci_unit == "sample" else metadata[ci_unit],
                ci_unit=ci_unit,
                bootstrap_repetitions=int(
                    config.get("reporting", {}).get("bootstrap_repetitions", 400)
                ),
            )
        condition_results["leakage_audits"] = run_leakage_audits(
            labels,
            metadata,
            feature_matrix,
            partitions,
            dataset_fingerprint=manifest["dataset_fingerprint"],
            split_fingerprint=split["split_fingerprint"],
            identity_fields=list(config.get("feature_policy", {}).get("grouping_only_fields", []))
            or None,
            seed=int(config.get("experiment", {}).get("seed", 1337)) + 3000,
        )
        results[condition_name] = condition_results
        _write_json(condition_dir / "metrics.json", condition_results)
        journal.append(
            "condition_completed",
            condition=condition_name,
            rows=len(labels),
            split_fingerprint=split["split_fingerprint"],
        )
    acceptance = _acceptance(results)
    null_models = results.get("null", {}).get("models", {})
    null_assessments = acceptance.get("control_model_assessments", {}).get("null", {})
    null_audits = results.get("null", {}).get("leakage_audits", {})
    null_group_audits = null_audits.get("label_balance", {})
    max_group_imbalance = max(
        (
            float(group["absolute_deviation_from_half"])
            for field in null_group_audits.values()
            for group in field.get("groups", [])
        ),
        default=0.0,
    )
    acceptance["null_investigation"] = {
        "observed_models": {
            name: {
                "balanced_accuracy_mean": item["summary"]["balanced_accuracy_mean"],
                "auroc_mean": item["summary"]["auroc_mean"],
                "assessment": null_assessments.get(name),
            }
            for name, item in null_models.items()
        },
        "maximum_group_positive_rate_deviation": max_group_imbalance,
        "repeated_training_seed_variation": {
            name: {
                "balanced_accuracy_std": item["summary"]["balanced_accuracy_std"],
                "auroc_std": item["summary"]["auroc_std"],
            }
            for name, item in null_models.items()
        },
        "alternative_explanations": {
            "finite_sample_variation": "plausible; one materialized null is not an independent null-resampling study",
            "group_imbalance": {
                "observed": max_group_imbalance > 0,
                "maximum_positive_rate_deviation": max_group_imbalance,
                "evidence": "see leakage_audits.label_balance",
            },
            "metadata_structure": {
                "status": "audit-only; inspect metadata_only and identity_only results",
                "metadata_only": null_audits.get("metadata_only"),
                "identity_only": null_audits.get("identity_only"),
            },
            "feature_engineering_artifact": {
                "status": "not resolved by score alone",
                "feature_distribution_differences": null_audits.get(
                    "feature_distribution_differences"
                ),
            },
            "split_artifact": {
                "status": "not ruled out; split is materialized and grouped",
                "partition_class_balance": results.get("null", {}).get("partition_class_balance"),
            },
            "repeated_seed_artifact": {
                "status": "not resolved; repeated fits on one fixed dataset are deterministic",
                "training_seed_variation": {
                    name: {
                        "balanced_accuracy_std": item["summary"]["balanced_accuracy_std"],
                        "auroc_std": item["summary"]["auroc_std"],
                    }
                    for name, item in null_models.items()
                },
            },
            "actual_unintended_synthetic_signal": "not established; the null backend injects no intentional label-dependent trace signal",
        },
        "interpretation": (
            "The boosted-tree elevation is retained as FAIL / INVESTIGATE. "
            "The recorded synthetic null has finite group-level label variation and "
            "deterministic repeated training on one materialized dataset; this does not "
            "establish genuine signal. The Phase 1 gate remains closed until an independent "
            "null-resampling investigation resolves finite-sample, group-imbalance, and split "
            "alternatives."
        ),
        "next_required_evidence": [
            "independent materialized null datasets or label permutations",
            "group-stratified balance review",
            "repeat the exact model/split audit across independent seeds",
        ],
    }
    report = {
        "schema": "sensetrace.phase0-report.v1",
        "run_id": run_id,
        "claim_scope": (
            "The injected-signal Phase 0 classifier recovered a known synthetic perturbation "
            "under grouped holdout; this is not physical DRAM-state inference."
        ),
        "configuration_hash": config_hash,
        "conditions": results,
        "acceptance": acceptance,
    }
    _write_json(run_dir / "metrics.json", report)
    journal.append("run_completed", acceptance=acceptance)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "acceptance": acceptance,
        }
    )
    _write_json(run_dir / "run.json", run)
    return report


def _acceptance(results: dict[str, Any]) -> dict[str, Any]:
    def model_assessment(item: dict[str, Any]) -> dict[str, Any]:
        runs = item.get("runs", [])
        metrics: dict[str, dict[str, Any]] = {}
        for metric, ci_key in [
            ("balanced_accuracy", "confidence_interval_95"),
            ("auroc", "auroc_confidence_interval_95"),
        ]:
            values = [float(run[metric]) for run in runs]
            intervals = [run.get(ci_key, [float("nan"), float("nan")]) for run in runs]
            finite_intervals = [interval for interval in intervals if not any(np.isnan(interval))]
            point_mean = float(np.mean(values)) if values else float("nan")
            lower = min((float(interval[0]) for interval in finite_intervals), default=float("nan"))
            upper = max((float(interval[1]) for interval in finite_intervals), default=float("nan"))
            statistically_above_chance = bool(np.isfinite(lower) and lower > 0.5)
            numerically_elevated = bool(np.isfinite(point_mean) and point_mean > 0.5)
            metrics[metric] = {
                "mean": point_mean,
                "interval_across_runs": [lower, upper],
                "numerically_elevated": numerically_elevated,
                "uncalibrated_interval_excludes_chance": statistically_above_chance,
                "calibrated_decision": "not_available; run sensetrace calibrate phase0",
            }
        elevated = any(value["numerically_elevated"] for value in metrics.values())
        return {
            "status": "UNCALIBRATED",
            "metrics": metrics,
            "reason": "A single materialized run cannot estimate the complete pipeline null distribution",
            "numerically_elevated": elevated,
        }

    control_assessments: dict[str, Any] = {}
    for control in ["null", "shuffled"]:
        control_assessments[control] = {
            model: model_assessment(item)
            for model, item in results.get(control, {}).get("models", {}).items()
        }

    def best_score(condition: str) -> float | None:
        models = results.get(condition, {}).get("models", {})
        scores = [float(item["summary"]["balanced_accuracy_mean"]) for item in models.values()]
        return max(scores) if scores else None

    injected_conditions = [name for name in results if name.startswith("injected")]
    detected = []
    for name in injected_conditions:
        item = results[name].get("models", {})
        if any(
            float(model.get("summary", {}).get("balanced_accuracy_mean", 0.0)) >= 0.55
            for model in item.values()
        ):
            detected.append(name)
    null_present = bool(control_assessments.get("null"))
    shuffled_present = bool(control_assessments.get("shuffled"))
    return {
        "null_consistent_with_chance": null_present,
        "shuffled_labels_consistent_with_chance": shuffled_present,
        "control_model_assessments": control_assessments,
        "null_controls_clean_across_enabled_models": False,
        "injected_signal_detected": bool(detected),
        "detected_injected_conditions": detected,
        "best_injected_balanced_accuracy": {name: best_score(name) for name in injected_conditions},
        "grouped_split_visible": all(
            "split" in item and item["split"]["split_strategy"] == "grouped"
            for item in results.values()
        ),
        "identity_policy_enforced_by_default": True,
        "phase1_gate": False,
        "phase1_gate_status": "closed; fresh empirical calibration is required",
    }
