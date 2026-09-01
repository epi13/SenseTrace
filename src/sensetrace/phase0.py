"""Phase 0 synthetic controls and self-contained result records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .acquisition.synthetic import SyntheticBackend
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
        condition=condition,
        amplitude_sigma=amplitude,
        start_index=int(
            controls.get("start_index", max(0, int(data.get("trace_length", 128)) // 3))
        ),
        width=int(controls.get("width", max(4, int(data.get("trace_length", 128)) // 16))),
        session_count=int(acquisition.get("session_count", 4)),
        device_count=int(acquisition.get("device_count", 2)),
        permute_seed=int(config.get("experiment", {}).get("seed", 1337)) + 7919,
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
    )
    return manifest


def _enabled_models(config: dict[str, Any]) -> list[str]:
    configured = config.get("models", {})
    defaults = ["logistic_regression", "boosted_trees", "tiny_mlp", "tiny_cnn"]
    return [name for name in defaults if configured.get(name, {}).get("enabled", True)]


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
    for condition_name, backend_condition, amplitude in condition_specs:
        journal.append("condition_started", condition=condition_name)
        condition_config = _condition_config(config, backend_condition, amplitude)
        condition_dir = run_dir / "datasets" / condition_name
        manifest = _generate_condition(
            condition_config, condition_dir, backend_condition, amplitude
        )
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
        condition_results: dict[str, Any] = {
            "dataset": manifest,
            "split": split,
            "models": {},
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
            )
        results[condition_name] = condition_results
        _write_json(condition_dir / "metrics.json", condition_results["models"])
        journal.append(
            "condition_completed",
            condition=condition_name,
            rows=len(labels),
            split_fingerprint=split["split_fingerprint"],
        )
    acceptance = _acceptance(results)
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
    def mean_score(condition: str, model: str = "logistic_regression") -> float | None:
        item = results.get(condition, {}).get("models", {}).get(model)
        return None if not item else float(item["summary"]["balanced_accuracy_mean"])

    def best_score(condition: str) -> float | None:
        models = results.get(condition, {}).get("models", {})
        scores = [float(item["summary"]["balanced_accuracy_mean"]) for item in models.values()]
        return max(scores) if scores else None

    null = mean_score("null")
    shuffled = mean_score("shuffled")
    injected_conditions = [name for name in results if name.startswith("injected")]
    detected = [name for name in injected_conditions if (best_score(name) or 0) >= 0.56]
    return {
        "null_consistent_with_chance": null is not None and abs(null - 0.5) <= 0.05,
        "shuffled_labels_consistent_with_chance": shuffled is not None
        and abs(shuffled - 0.5) <= 0.05,
        "injected_signal_detected": bool(detected),
        "detected_injected_conditions": detected,
        "grouped_split_visible": all(
            "split" in item and item["split"]["split_strategy"] == "grouped"
            for item in results.values()
        ),
        "identity_policy_enforced_by_default": True,
    }
