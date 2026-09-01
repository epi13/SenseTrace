"""Conservative Phase 1A commodity-memory campaign orchestration."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition.commodity import CommodityDramBackend
from .acquisition.native import NativeMeasurementKernel
from .config import config_fingerprint, normalized_config
from .datasets import build_feature_matrix, load_dataset, write_dataset_manifest
from .inventory import collect_inventory
from .journal import Journal
from .models import train_and_evaluate
from .phase0 import _enabled_models
from .runner import _git_commit, new_run_id
from .splits import partition_indices, phase1a_split_hierarchy, write_split
from .storage import ShardWriter, load_shards, validate_all_shards


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _backend_from_config(config: dict[str, Any]) -> CommodityDramBackend:
    data = config.get("data", {})
    physical = config.get("phase1a", {})
    return CommodityDramBackend(
        count=int(physical.get("samples", min(int(data.get("samples", 128)), 256))),
        trace_length=int(physical.get("trace_length", min(int(data.get("trace_length", 32)), 64))),
        seed=int(config.get("experiment", {}).get("seed", 1337)),
        pattern=str(physical.get("pattern", "single_bit")),
        target_bit=int(physical.get("target_bit", 0)),
        word_count=int(physical.get("word_count", 1024)),
        lock_memory=bool(physical.get("lock_memory", True)),
        cache_control=str(physical.get("cache_control", "eviction_buffer")),
        operation=str(physical.get("operation", "memory_read")),
        eviction_bytes=int(physical.get("eviction_bytes", 4 * 1024 * 1024)),
        cpu_affinity=physical.get("cpu_affinity"),
        location_count=(
            int(physical["location_count"]) if physical.get("location_count") is not None else None
        ),
        trials_per_location=int(physical.get("trials_per_location", 64)),
        labels_per_location=int(
            physical.get("labels_per_location", int(physical.get("trials_per_location", 64)) // 2)
        ),
        session_count=int(physical.get("session_count", 1)),
        use_native_kernel=bool(physical.get("use_native_kernel", True)),
    )


def _materialize_backend(
    config: dict[str, Any],
    condition_dir: Path,
    *,
    cache_control: str | None = None,
    operation: str | None = None,
    pattern: str | None = None,
) -> dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    if cache_control is not None or operation is not None or pattern is not None:
        physical = dict(config.get("phase1a", {}))
        if cache_control is not None:
            physical["cache_control"] = cache_control
        if operation is not None:
            physical["operation"] = operation
        if pattern is not None:
            physical["pattern"] = pattern
        config = json.loads(json.dumps(config))
        config["phase1a"] = physical
    backend = _backend_from_config(config)
    writer = ShardWriter(
        condition_dir,
        shard_target_mb=float(config.get("acquisition", {}).get("shard_target_mb", 1)),
        max_samples_per_shard=config.get("acquisition", {}).get("max_samples_per_shard", 256),
    )
    journal = Journal(condition_dir / "events.jsonl")
    try:
        for sample in backend.samples():
            info = writer.add(sample.trace, sample.label, sample.metadata)
            if info:
                journal.append("shard_finalized", **info.as_dict())
        info = writer.finalize()
        if info:
            journal.append("shard_finalized", **info.as_dict())
    finally:
        backend.close()
    traces, labels, metadata, _shards = load_shards(condition_dir)
    manifest = write_dataset_manifest(
        condition_dir,
        config=config,
        condition="physical",
        shard_infos=validate_all_shards(condition_dir),
        label_stream_fingerprint=hashlib.sha256(labels.tobytes()).hexdigest(),
        class_balance={"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))},
        provenance={
            "backend": "CommodityDramBackend",
            "operation": config.get("phase1a", {}).get(
                "operation", "ordinary user-space write then timed read"
            ),
            "cache_control": config.get("phase1a", {}).get("cache_control", "eviction_buffer"),
            "digital_read_is_audit_only": True,
            "physical_topology": "unknown",
        },
    )
    journal.append("condition_acquisition_completed", rows=len(labels), condition="physical")
    return manifest


def _materialize_label_permutation(
    source_dir: Path, condition_dir: Path, config: dict[str, Any], permutation_seed: int
) -> dict[str, Any]:
    traces, labels, metadata, _shards, source_manifest = load_dataset(source_dir)
    permutation = np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(permutation_seed)
    strata_keys = list(config.get("calibration", {}).get("permutation_strata", ["location_id"]))
    strata: dict[tuple[str, ...], list[int]] = {}
    keys = list(zip(*(np.asarray(metadata[key]).astype(str) for key in strata_keys), strict=True))
    for index, key in enumerate(keys):
        strata.setdefault(key, []).append(index)
    for indices in strata.values():
        index_array = np.asarray(indices, dtype=np.int64)
        permutation[index_array] = rng.permutation(index_array)
    permuted = labels[permutation]
    writer = ShardWriter(condition_dir, shard_target_mb=1, max_samples_per_shard=256)
    for index, (trace, label) in enumerate(zip(traces, permuted, strict=True)):
        info = writer.add(trace, int(label), {key: value[index] for key, value in metadata.items()})
        if info:
            pass
    writer.finalize()
    return write_dataset_manifest(
        condition_dir,
        config=config,
        condition="shuffled",
        shard_infos=validate_all_shards(condition_dir),
        label_stream_fingerprint=hashlib.sha256(permuted.tobytes()).hexdigest(),
        class_balance={"0": int(np.sum(permuted == 0)), "1": int(np.sum(permuted == 1))},
        provenance={
            "parent_dataset_fingerprint": source_manifest["dataset_fingerprint"],
            "original_label_stream_fingerprint": source_manifest["label_stream_fingerprint"],
            "permutation_seed": permutation_seed,
            "permutation_reference": hashlib.sha256(
                np.asarray(permutation, dtype=np.int64).tobytes()
            ).hexdigest(),
            "permutation_strata": strata_keys,
            "only_changed_variable": "label association",
        },
    )


def run_phase1a(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    phase0_report: str | Path | dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run a small exploratory campaign only after an explicit Phase 0 gate."""

    if phase0_report is None:
        raise RuntimeError(
            "Phase 1A is gated: provide a Phase 0 report whose acceptance.phase1_gate is true"
        )
    report = (
        json.loads(Path(phase0_report).read_text(encoding="utf-8"))
        if isinstance(phase0_report, (str, Path))
        else phase0_report
    )
    if not report.get("acceptance", {}).get("phase1_gate", False):
        raise RuntimeError("Phase 0 gate is not PASS; physical acquisition was not started")
    root = Path(output_root)
    run_id = run_id or new_run_id("phase1a")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "run.json",
        {
            "schema": "sensetrace.run.v1",
            "run_id": run_id,
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(config),
            "claim_scope": "exploratory safe commodity-memory host observables only",
            "phase0_gate_report": str(phase0_report)
            if not isinstance(phase0_report, dict)
            else "embedded",
        },
    )
    _write_json(run_dir / "host.json", collect_inventory())
    _write_json(run_dir / "config.json", normalized_config(config))
    journal = Journal(run_dir / "events.jsonl")
    journal.append("run_started", run_id=run_id, phase="phase1a", status="exploratory")

    datasets_dir = run_dir / "datasets"
    if (
        bool(config.get("phase1a", {}).get("require_native_kernel", True))
        and NativeMeasurementKernel.load() is None
    ):
        raise RuntimeError("Phase 1A requires the built native timing kernel; run make -C native")
    primary_dir = datasets_dir / "paired_single_bit"
    primary_manifest = _materialize_backend(config, primary_dir, pattern="single_bit")
    shuffled_dir = datasets_dir / "paired_single_bit_shuffled"
    shuffled_manifest = _materialize_label_permutation(
        primary_dir,
        shuffled_dir,
        config,
        int(config.get("experiment", {}).get("seed", 1337)) + 7919,
    )
    cache_dir = datasets_dir / "cache_hit_control"
    cache_manifest = _materialize_backend(
        config, cache_dir, cache_control="none", pattern="single_bit"
    )
    idle_dir = datasets_dir / "idle_no_memory_operation"
    idle_manifest = _materialize_backend(
        config, idle_dir, operation="idle", cache_control="none", pattern="single_bit"
    )
    all_zero_one_dir = datasets_dir / "all_zero_vs_all_one"
    all_zero_one_manifest = _materialize_backend(config, all_zero_one_dir, pattern="all_zero_one")
    random_word_dir = datasets_dir / "random_word_null"
    random_word_manifest = _materialize_backend(config, random_word_dir, pattern="random_word")

    result_conditions: dict[str, Any] = {}
    for name, condition_dir, manifest in [
        ("all_zero_vs_all_one", all_zero_one_dir, all_zero_one_manifest),
        ("paired_single_bit", primary_dir, primary_manifest),
        ("paired_single_bit_shuffled", shuffled_dir, shuffled_manifest),
        ("random_word_null", random_word_dir, random_word_manifest),
        ("cache_hit_control", cache_dir, cache_manifest),
        ("idle_no_memory_operation", idle_dir, idle_manifest),
    ]:
        traces, labels, metadata, _shards, _manifest = load_dataset(condition_dir)
        hierarchy = phase1a_split_hierarchy(
            metadata,
            dataset_fingerprint=manifest["dataset_fingerprint"],
            seed=int(config.get("experiment", {}).get("seed", 1337)),
        )
        primary_split_record = hierarchy.get("B_unseen_location")
        if primary_split_record is None or primary_split_record["status"] != "available":
            primary_split_record = hierarchy["A_repeated_trial_holdout"]
        if primary_split_record["status"] != "available":
            raise RuntimeError(f"no usable Phase 1A split: {primary_split_record}")
        split = primary_split_record["split"]
        write_split(condition_dir / "split.json", split)
        partitions = partition_indices(metadata, split)
        features = build_feature_matrix(traces, metadata)
        ci_unit = str(
            config.get("phase1a", {}).get(
                "ci_unit", config.get("reporting", {}).get("ci_unit", "sample")
            )
        )
        model_results = {
            model: train_and_evaluate(
                model,
                traces,
                features,
                labels,
                partitions,
                seeds=[int(seed) for seed in config.get("training", {}).get("seeds", [11, 23])],
                dataset_fingerprint=manifest["dataset_fingerprint"],
                split_fingerprint=split["split_fingerprint"],
                epochs=int(config.get("training", {}).get("epochs", 10)),
                patience=int(config.get("training", {}).get("early_stopping_patience", 2)),
                batch_size=int(config.get("training", {}).get("batch_size", 128)),
                groups=None if ci_unit == "sample" else metadata[ci_unit],
                ci_unit=ci_unit,
                bootstrap_repetitions=int(
                    config.get("reporting", {}).get("bootstrap_repetitions", 200)
                ),
            )
            for model in _enabled_models(config)
        }
        result_conditions[name] = {
            "dataset": manifest,
            "split": split,
            "models": model_results,
            "claim_status": "exploratory; not confirmatory evidence",
            "split_hierarchy": hierarchy,
            "primary_split": split["split_name"],
        }
    journal.append("run_completed", status="exploratory")
    final = {
        "schema": "sensetrace.phase1a-report.v2",
        "run_id": run_id,
        "status": "exploratory",
        "conditions": result_conditions,
        "measurement_provenance": {
            "backend": "CommodityDramBackend",
            "ordinary_read_value": "ground-truth verification only",
            "physical_address_or_row_claim": "not available",
            "paired_target_construction": "one random base word per pair; target bit forced to 0/1; pair order randomized",
            "location_design": {
                "location_count": config.get("phase1a", {}).get("location_count"),
                "trials_per_location": config.get("phase1a", {}).get("trials_per_location"),
                "labels_per_location": config.get("phase1a", {}).get("labels_per_location"),
            },
            "native_kernel_required": bool(
                config.get("phase1a", {}).get("require_native_kernel", True)
            ),
            "next_step": "freeze method and collect a new session before any confirmatory claim",
        },
    }
    _write_json(run_dir / "metrics.json", final)
    return final
