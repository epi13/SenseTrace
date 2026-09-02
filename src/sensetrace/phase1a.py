"""Phase 1A acquisition sessions, campaign manifests, and complete analysis."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition.commodity import CommodityDramBackend
from .acquisition.native import NativeMeasurementKernel
from .acquisition.primitive import TimingPerturbationCalibration
from .audits import run_leakage_audits
from .config import config_fingerprint, normalized_config
from .datasets import build_feature_matrix, combine_datasets, load_dataset, write_dataset_manifest
from .errors import ConfigError
from .hashing import sha256_json
from .inventory import collect_inventory
from .journal import Journal
from .metrics import paired_delta_analysis
from .models import train_and_evaluate
from .phase0 import _enabled_models
from .protocol import (
    phase1a_commodity_baseline_protocol,
    phase1a_commodity_baseline_protocol_hash,
)
from .runner import _git_commit, new_run_id
from .splits import (
    partition_indices,
    phase1a_split_hierarchy,
    split_composition,
    validate_phase1a_split_hierarchy,
    write_split,
)
from .storage import (
    ShardWriter,
    load_shards,
    quarantine_invalid_shards,
    quarantine_temporary_shards,
    validate_all_shards,
)


def _write_json(path: Path, value: Any) -> None:
    """Write a small ledger record atomically enough for crash recovery."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _session_count(config: dict[str, Any]) -> int:
    physical = config.get("phase1a", {})
    return int(physical.get("session_count", physical.get("acquisition_session_count", 1)))


def _validate_physical_phase1a_config(config: dict[str, Any]) -> None:
    """Reject calibration-only controls before a physical run can be created."""

    physical = config.get("phase1a", {})
    if physical.get("timing_perturbation_cycles", 0) != 0:
        raise ConfigError(
            "physical Phase 1A forbids artificial timing perturbation; use the explicit "
            "native sensitivity calibration path"
        )
    if physical.get("timing_perturbation_label", 1) != 1:
        raise ConfigError(
            "physical Phase 1A forbids non-default artificial perturbation-label configuration"
        )
    if "calibration_namespace" in physical:
        raise ConfigError(
            "physical Phase 1A forbids calibration_namespace; use the explicit calibration path"
        )


def _acquisition_protocol(
    config: dict[str, Any], calibration_context: TimingPerturbationCalibration | None
) -> tuple[str, str]:
    if calibration_context is None:
        return (
            phase1a_commodity_baseline_protocol(config)["version"],
            phase1a_commodity_baseline_protocol_hash(config),
        )
    identity = "native-sensitivity-calibration-v2"
    return identity, sha256_json(
        {
            "version": identity,
            "configuration_hash": config_fingerprint(config),
            "calibration_context": calibration_context.as_dict(),
        }
    )


def _backend_from_config(
    config: dict[str, Any],
    *,
    session_id: str | None = None,
    session_index: int = 0,
    session_seed: int | None = None,
    campaign_id: str | None = None,
    host_inventory_snapshot: dict[str, Any] | None = None,
    session_started_at: str | None = None,
    parent_session_id: str | None = None,
    recovery_reason: str | None = None,
    calibration_context: TimingPerturbationCalibration | None = None,
) -> CommodityDramBackend:
    data = config.get("data", {})
    physical = config.get("phase1a", {})
    if calibration_context is None:
        _validate_physical_phase1a_config(config)
    return CommodityDramBackend(
        count=int(physical.get("samples", min(int(data.get("samples", 128)), 256))),
        trace_length=int(physical.get("trace_length", min(int(data.get("trace_length", 32)), 64))),
        seed=int(
            session_seed
            if session_seed is not None
            else config.get("experiment", {}).get("seed", 1337)
        ),
        pattern=str(physical.get("pattern", "single_bit")),
        target_bit=int(physical.get("target_bit", 0)),
        word_count=int(physical.get("word_count", 1024)),
        lock_memory=bool(physical.get("lock_memory", True)),
        cache_control=str(physical.get("cache_control", "eviction_buffer")),
        operation=str(physical.get("operation", "memory_read")),
        measurement_primitive=str(
            physical.get("measurement_primitive", "commodity-clflush-timed-load")
        ),
        eviction_bytes=int(physical.get("eviction_bytes", 4 * 1024 * 1024)),
        cpu_affinity=physical.get("cpu_affinity"),
        location_count=(
            int(physical["location_count"]) if physical.get("location_count") is not None else None
        ),
        trials_per_location=int(physical.get("trials_per_location", 64)),
        labels_per_location=int(
            physical.get("labels_per_location", int(physical.get("trials_per_location", 64)) // 2)
        ),
        # A backend instance is one session. This legacy argument is retained
        # only for source compatibility and no longer partitions one stream.
        session_count=1,
        use_native_kernel=bool(physical.get("use_native_kernel", True)),
        calibration_context=calibration_context,
        acquisition_session_id=session_id,
        session_index=session_index,
        campaign_id=campaign_id,
        session_started_at=session_started_at,
        host_inventory_snapshot=host_inventory_snapshot,
        code_commit=_git_commit(),
        configuration_hash=config_fingerprint(config),
    )


def _session_record_from_file(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _materialize_session(
    config: dict[str, Any],
    session_dir: Path,
    *,
    condition: str,
    campaign_id: str,
    session_index: int,
    session_id: str,
    host_inventory_snapshot: dict[str, Any],
    parent_session_id: str | None = None,
    recovery_reason: str | None = None,
    calibration_context: TimingPerturbationCalibration | None = None,
) -> dict[str, Any]:
    """Acquire one genuine session into its own crash-safe source directory."""

    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "session.json"
    existing_record = _session_record_from_file(session_path)
    if existing_record and existing_record.get("status") == "completed":
        dataset_path = session_dir / "dataset.json"
        if dataset_path.exists():
            _traces, _labels, _metadata, _shards, manifest = load_dataset(session_dir)
            return manifest

    if existing_record:
        status = str(existing_record.get("status", "unknown"))
        if status == "completed":
            raise RuntimeError("completed session unexpectedly reached recovery path")

        # A finalized shard is immutable evidence from the old physical
        # allocation.  It cannot be resumed by constructing a new backend in
        # this directory: doing so would join two allocations under one
        # acquisition-session identity.  Leave an append-only decision in the
        # old ledger and materialize a sibling session with new identities.
        old_session_id = str(existing_record.get("acquisition_session_id", session_id))
        old_allocation_id = str(
            existing_record.get("controlled_memory_region", {}).get("allocation_id", "unavailable")
        )
        old_journal = Journal(session_dir / "events.jsonl")
        recovered = old_journal.recover()
        replacement_id = f"session-{uuid.uuid4().hex}"
        replacement_dir = session_dir.parent / replacement_id
        interruption_event = old_journal.append(
            "acquisition_session_interrupted",
            acquisition_session_id=old_session_id,
            allocation_id=old_allocation_id,
            finalized_shard_count=len(validate_all_shards(session_dir))
            if list(session_dir.glob("shard-*.npz"))
            else 0,
            decision="finalized shards remain immutable and are excluded from the replacement session",
            replacement_session_id=replacement_id,
            recovery_reason=recovery_reason or "restart after incomplete physical acquisition",
            recovered_event_count=len(recovered.events),
        )
        existing_record.update(
            {
                "status": "interrupted",
                "interrupted_at": datetime.now(UTC).isoformat(),
                "interruption_reason": recovery_reason
                or "restart after incomplete physical acquisition",
                "recovery_decision": "fail_closed_new_allocation",
                "recovery_replacement_session_id": replacement_id,
                "recovery_replacement_dir": str(replacement_dir),
                "interruption_event": interruption_event,
            }
        )
        _write_json(session_path, existing_record)
        replacement_snapshot = collect_inventory()
        replacement = _materialize_session(
            config,
            replacement_dir,
            condition=condition,
            campaign_id=campaign_id,
            session_index=session_index,
            session_id=replacement_id,
            host_inventory_snapshot=replacement_snapshot,
            parent_session_id=old_session_id,
            recovery_reason="replacement for interrupted physical acquisition session",
            calibration_context=calibration_context,
        )
        replacement["_materialized_source_dir"] = str(replacement_dir)
        return replacement

    session_started_at = datetime.now(UTC).isoformat()

    backend = _backend_from_config(
        config,
        session_id=session_id,
        session_index=session_index,
        session_seed=int(config.get("experiment", {}).get("seed", 1337)) + session_index * 104729,
        campaign_id=campaign_id,
        host_inventory_snapshot=host_inventory_snapshot,
        session_started_at=session_started_at,
        calibration_context=calibration_context,
    )
    session_record = backend.session_provenance()
    protocol_identity, protocol_hash = _acquisition_protocol(config, calibration_context)
    session_record.update(
        {
            "protocol_identity": protocol_identity,
            "protocol_hash": protocol_hash,
            "acquisition_scope": (
                "explicit native timing calibration"
                if calibration_context is not None
                else "physical Phase 1A commodity baseline"
            ),
        }
    )
    if existing_record:
        session_record["restart_count"] = int(existing_record.get("restart_count", 0)) + 1
        session_record["prior_session_ledger"] = {
            "status": existing_record.get("status", "unknown"),
            "allocation_id": existing_record.get("controlled_memory_region", {}).get(
                "allocation_id", "unavailable"
            ),
            "rows": existing_record.get("rows", "unavailable"),
        }
    session_record.update(
        {
            "condition": condition,
            "status": "active",
            "restart_count": int(session_record.get("restart_count", 0)),
            "config_hash": config_fingerprint(config),
            "code_commit": _git_commit(),
        }
    )
    if parent_session_id is not None:
        session_record["recovery"] = {
            "parent_session_id": parent_session_id,
            "reason": recovery_reason or "replacement session",
            "allocation_boundary": "new allocation and new acquisition-session identity",
        }
    _write_json(session_path, session_record)
    journal = Journal(session_dir / "events.jsonl")
    recovered = journal.recover()
    quarantined = quarantine_temporary_shards(session_dir)
    quarantined.extend(quarantine_invalid_shards(session_dir))
    infos = validate_all_shards(session_dir)
    completed_rows = sum(info.rows for info in infos)
    if completed_rows > backend.count:
        backend.close()
        raise RuntimeError("session shards contain more rows than the configured acquisition")
    started_event = journal.append(
        "acquisition_session_started",
        acquisition_session_id=backend.acquisition_session_id,
        condition=condition,
        session_index=session_index,
        resume_from_sample=completed_rows,
        quarantined_paths=[path.name for path in quarantined],
        recovered_event_count=len(recovered.events),
    )

    writer = ShardWriter(
        session_dir,
        shard_target_mb=float(config.get("acquisition", {}).get("shard_target_mb", 1)),
        max_samples_per_shard=config.get("acquisition", {}).get("max_samples_per_shard", 256),
    )
    try:
        for sample in backend.samples(start_index=completed_rows):
            info = writer.add(sample.trace, sample.label, sample.metadata)
            if info is not None:
                journal.append("shard_finalized", **info.as_dict())
        info = writer.finalize()
        if info is not None:
            journal.append("shard_finalized", **info.as_dict())
    finally:
        backend.close()

    _traces, labels, _metadata, _shards = load_shards(session_dir)
    finalized = validate_all_shards(session_dir)
    initial_manifest = write_dataset_manifest(
        session_dir,
        config=config,
        condition=condition,
        shard_infos=finalized,
        label_stream_fingerprint=hashlib.sha256(labels.tobytes()).hexdigest(),
        class_balance={"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))},
        provenance={
            "backend": "CommodityDramBackend",
            "session_scope": "one independently started backend and newly allocated controlled buffer",
            "physical_topology": "unknown; virtual buffer locations only",
            "protocol_identity": protocol_identity,
            "protocol_hash": protocol_hash,
            "artificial_timing_perturbation": {
                "allowed": calibration_context is not None,
                "timing_perturbation_cycles": (
                    calibration_context.cycles if calibration_context is not None else 0
                ),
                "timing_perturbation_label": (
                    calibration_context.label if calibration_context is not None else 1
                ),
                "label_correlated": bool(
                    calibration_context is not None and calibration_context.cycles > 0
                ),
                "applied": bool(calibration_context is not None and calibration_context.cycles > 0),
                "physical_phase1a_forbidden": True,
                "calibration_namespace": (
                    calibration_context.namespace
                    if calibration_context is not None
                    else "forbidden"
                ),
                "scope": (
                    "explicit calibration only"
                    if calibration_context is not None
                    else "forbidden and recorded as zero"
                ),
            },
        },
        acquisition_sessions=[session_record],
        campaign_id=campaign_id,
    )
    completed_event = journal.append(
        "acquisition_session_completed",
        acquisition_session_id=session_record["acquisition_session_id"],
        rows=len(labels),
        dataset_fingerprint=initial_manifest["dataset_fingerprint"],
    )
    session_record.update(
        {
            "status": "completed",
            "ended_at": datetime.now(UTC).isoformat(),
            "rows": len(labels),
            "dataset_fingerprint": initial_manifest["dataset_fingerprint"],
            "label_stream_fingerprint": hashlib.sha256(labels.tobytes()).hexdigest(),
            "journal_boundaries": {
                "path": "events.jsonl",
                "started_event": started_event,
                "completed_event": completed_event,
            },
        }
    )
    _write_json(session_path, session_record)
    # Refresh the manifest so its embedded source ledger includes the final
    # fingerprint and journal boundary without changing dataset identity.
    return write_dataset_manifest(
        session_dir,
        config=config,
        condition=condition,
        shard_infos=finalized,
        label_stream_fingerprint=hashlib.sha256(labels.tobytes()).hexdigest(),
        class_balance={"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))},
        provenance={
            "backend": "CommodityDramBackend",
            "session_scope": "one independently started backend and newly allocated controlled buffer",
            "physical_topology": "unknown; virtual buffer locations only",
            "protocol_identity": protocol_identity,
            "protocol_hash": protocol_hash,
            "artificial_timing_perturbation": {
                "allowed": calibration_context is not None,
                "timing_perturbation_cycles": (
                    calibration_context.cycles if calibration_context is not None else 0
                ),
                "timing_perturbation_label": (
                    calibration_context.label if calibration_context is not None else 1
                ),
                "label_correlated": bool(
                    calibration_context is not None and calibration_context.cycles > 0
                ),
                "applied": bool(calibration_context is not None and calibration_context.cycles > 0),
                "physical_phase1a_forbidden": True,
                "calibration_namespace": (
                    calibration_context.namespace
                    if calibration_context is not None
                    else "forbidden"
                ),
                "scope": (
                    "explicit calibration only"
                    if calibration_context is not None
                    else "forbidden and recorded as zero"
                ),
            },
        },
        acquisition_sessions=[session_record],
        campaign_id=campaign_id,
    )


def _existing_session_for_index(source_root: Path, session_index: int) -> Path | None:
    """Find the latest valid candidate for a logical session slot.

    Completed replacement sessions take precedence over interrupted parents;
    this prevents a second restart from attempting another recovery of the
    immutable parent directory.
    """

    candidates: list[tuple[Path, dict[str, Any]]] = []
    if not source_root.exists():
        return None
    for path in sorted(item for item in source_root.iterdir() if item.is_dir()):
        record = _session_record_from_file(path / "session.json")
        if record is not None and int(record.get("session_index", -1)) == session_index:
            candidates.append((path, record))
    for desired_status in ("completed", "active", "interrupted"):
        matching = [path for path, record in candidates if record.get("status") == desired_status]
        if matching:
            return matching[-1]
    return None


def _materialize_backend(
    config: dict[str, Any],
    condition_dir: Path,
    *,
    cache_control: str | None = None,
    operation: str | None = None,
    pattern: str | None = None,
) -> dict[str, Any]:
    """Compatibility helper that materializes exactly one source session."""

    effective = json.loads(json.dumps(config))
    physical = dict(effective.get("phase1a", {}))
    if cache_control is not None:
        physical["cache_control"] = cache_control
    if operation is not None:
        physical["operation"] = operation
    if pattern is not None:
        physical["pattern"] = pattern
    effective["phase1a"] = physical
    campaign_id = f"compat-{uuid.uuid4().hex}"
    session_id = f"session-{uuid.uuid4().hex}"
    source_dir = condition_dir / "sessions" / session_id
    materialized = _materialize_session(
        effective,
        source_dir,
        condition="physical",
        campaign_id=campaign_id,
        session_index=0,
        session_id=session_id,
        host_inventory_snapshot=collect_inventory(),
    )
    source_dir = Path(materialized.pop("_materialized_source_dir", str(source_dir)))
    return combine_datasets(
        [source_dir],
        condition_dir,
        config=effective,
        condition="physical",
        campaign_id=campaign_id,
        source_manifest_paths=[str(source_dir / "dataset.json")],
    )


def _materialize_acquired_condition(
    config: dict[str, Any],
    condition_dir: Path,
    *,
    condition: str,
    campaign_id: str,
    host_inventory_snapshot: dict[str, Any],
    session_count: int,
) -> dict[str, Any]:
    source_root = condition_dir / "sessions"
    source_dirs: list[Path] = []
    for session_index in range(session_count):
        source_dir = _existing_session_for_index(source_root, session_index)
        session_id = source_dir.name if source_dir is not None else f"session-{uuid.uuid4().hex}"
        source_dir = source_dir or source_root / session_id
        # Capture the ledger at the actual independent session boundary. The
        # campaign-level host snapshot remains useful context, but must not be
        # mistaken for six identical session snapshots.
        session_inventory = collect_inventory()
        materialized = _materialize_session(
            config,
            source_dir,
            condition=condition,
            campaign_id=campaign_id,
            session_index=session_index,
            session_id=session_id,
            host_inventory_snapshot=session_inventory,
        )
        actual_source_dir = Path(materialized.pop("_materialized_source_dir", str(source_dir)))
        source_dirs.append(actual_source_dir)
    return combine_datasets(
        source_dirs,
        condition_dir,
        config=config,
        condition=condition,
        campaign_id=campaign_id,
        source_manifest_paths=[str(path / "dataset.json") for path in source_dirs],
    )


def _materialize_label_permutation(
    source_dir: Path, condition_dir: Path, config: dict[str, Any], permutation_seed: int
) -> dict[str, Any]:
    if (condition_dir / "dataset.json").exists():
        _traces, _labels, _metadata, _shards, manifest = load_dataset(condition_dir)
        return manifest
    traces, labels, metadata, _shards, source_manifest = load_dataset(source_dir)
    permutation = np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(permutation_seed)
    strata_keys = list(config.get("calibration", {}).get("permutation_strata", ["location_id"]))
    if strata_keys == ["location_id"] and "virtual_location_id" in metadata:
        strata_keys = ["virtual_location_id"]
    strata: dict[tuple[str, ...], list[int]] = {}
    keys = list(zip(*(np.asarray(metadata[key]).astype(str) for key in strata_keys), strict=True))
    for index, key in enumerate(keys):
        strata.setdefault(key, []).append(index)
    for indices in strata.values():
        index_array = np.asarray(indices, dtype=np.int64)
        permutation[index_array] = rng.permutation(index_array)
    permuted = labels[permutation]
    writer = ShardWriter(
        condition_dir,
        shard_target_mb=float(config.get("acquisition", {}).get("shard_target_mb", 1)),
        max_samples_per_shard=config.get("acquisition", {}).get("max_samples_per_shard", 256),
    )
    for index, (trace, label) in enumerate(zip(traces, permuted, strict=True)):
        writer.add(trace, int(label), {key: value[index] for key, value in metadata.items()})
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
            "source_manifest": source_manifest,
        },
        acquisition_sessions=source_manifest.get("acquisition_sessions", []),
        campaign_id=source_manifest.get("campaign_id"),
    )


def _claim_for_split(name: str) -> str:
    return {
        "A_repeated_trial_holdout": "pipeline/paired-repeat claim only; virtual locations may be memorized",
        "B_unseen_location": "candidate relationship beyond tested virtual buffer locations; not session/device generalization",
        "C_unseen_acquisition_block": "candidate relationship beyond acquisition blocks; physical topology remains unknown",
        "D_unseen_acquisition_session": "candidate relationship across unseen acquisition sessions on this host/device",
        "E_unseen_boot": "candidate relationship across genuinely unseen OS boot IDs; not device-independent evidence",
    }.get(name, "claim boundary unavailable")


def _analyze_condition(
    condition_dir: Path, manifest: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    traces, labels, metadata, _shards, _loaded_manifest = load_dataset(condition_dir)
    hierarchy = phase1a_split_hierarchy(
        metadata,
        dataset_fingerprint=manifest["dataset_fingerprint"],
        seed=int(config.get("experiment", {}).get("seed", 1337)),
    )
    hierarchy_invariants = validate_phase1a_split_hierarchy(metadata, hierarchy)
    if hierarchy_invariants["status"] != "pass":
        raise RuntimeError("Phase 1A split hierarchy invariants failed")
    features = build_feature_matrix(traces, metadata)
    requested_ci_unit = str(
        config.get("phase1a", {}).get(
            "ci_unit", config.get("reporting", {}).get("ci_unit", "sample")
        )
    )
    enabled_models = _enabled_models(config)
    split_analyses: dict[str, Any] = {}
    for split_name, record in hierarchy.items():
        if record["status"] != "available":
            split_analyses[split_name] = {
                "status": "unavailable",
                "available": False,
                "grouping_keys": record.get("grouping_keys", []),
                "reason": record.get("reason", "unavailable"),
                "claim_allowed": record.get("claim_boundary", "no claim"),
                **(
                    {"boot_provenance": record["boot_provenance"]}
                    if split_name == "E_unseen_boot" and "boot_provenance" in record
                    else {}
                ),
            }
            continue
        split = record["split"]
        split_dir = condition_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        write_split(split_dir / f"{split_name}.json", split)
        # Preserve the historical single split path for readers that expect it,
        # while the report below always includes every available hierarchy level.
        if split_name == "B_unseen_location":
            write_split(condition_dir / "split.json", split)
        partitions = partition_indices(metadata, split)
        composition = split_composition(metadata, split, labels)
        if requested_ci_unit != "sample" and requested_ci_unit not in metadata:
            model_results: dict[str, Any] = {
                "status": "unavailable",
                "reason": f"requested CI grouping field {requested_ci_unit!r} is absent",
            }
        else:
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
                    groups=None if requested_ci_unit == "sample" else metadata[requested_ci_unit],
                    ci_unit=requested_ci_unit,
                    bootstrap_repetitions=int(
                        config.get("reporting", {}).get("bootstrap_repetitions", 200)
                    ),
                )
                for model in enabled_models
            }
        test_indices = partitions["test"]
        test_metadata = {key: values[test_indices] for key, values in metadata.items()}
        audits = run_leakage_audits(
            labels,
            metadata,
            features,
            partitions,
            dataset_fingerprint=manifest["dataset_fingerprint"],
            split_fingerprint=split["split_fingerprint"],
            traces=traces,
        )
        paired = paired_delta_analysis(
            traces[test_indices],
            labels[test_indices],
            test_metadata,
            repetitions=int(config.get("reporting", {}).get("paired_repetitions", 2000)),
            seed=int(config.get("experiment", {}).get("seed", 1337)) + 5000,
        )
        split_analyses[split_name] = {
            "status": "available",
            "available": True,
            "split": split,
            "grouping_keys": split.get("declared_grouping_keys", split["grouping_keys"]),
            "composition": composition,
            "models": model_results,
            "paired_statistics": paired,
            "audits": audits,
            "claim_allowed": _claim_for_split(split_name),
        }
    available = [
        analysis for analysis in split_analyses.values() if analysis.get("status") == "available"
    ]
    # B is retained as the compatibility view only when B actually exists.
    # Never replace it with A or another weaker split without saying so.
    primary = split_analyses.get("B_unseen_location")
    primary_available = primary is not None and primary.get("status") == "available"
    if not primary_available:
        primary = None
    return {
        "dataset": manifest,
        "split_hierarchy": hierarchy,
        "split_hierarchy_invariants": hierarchy_invariants,
        "split_analyses": split_analyses,
        "analysis_summary": {
            "available_split_count": len(available),
            "unavailable_splits": [
                name
                for name, analysis in split_analyses.items()
                if analysis.get("status") != "available"
            ],
            "all_available_splits_evaluated_independently": True,
            "identical_nominal_levels": hierarchy_invariants["identical_materialized_partitions"],
        },
        # Compatibility view: callers of the pre-campaign report can still
        # find the B result, but it is explicitly not the sole analysis.
        "split": primary.get("split") if primary else None,
        "models": primary.get("models") if primary else {},
        "primary_split": "B_unseen_location" if "B_unseen_location" in split_analyses else None,
        "primary_split_status": (
            split_analyses.get("B_unseen_location", {}).get("status", "unavailable")
        ),
        "claim_status": "exploratory; each split has its own claim boundary",
    }


def run_phase1a_campaign(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    phase0_report: str | Path | dict[str, Any],
    run_id: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Run a multi-session Phase 1A campaign after the frozen Phase 0 gate."""

    _validate_physical_phase1a_config(config)
    report = (
        json.loads(Path(phase0_report).read_text(encoding="utf-8"))
        if isinstance(phase0_report, (str, Path))
        else phase0_report
    )
    if not report.get("acceptance", {}).get("phase1_gate", False):
        raise RuntimeError("Phase 0 gate is not PASS; physical acquisition was not started")
    root = Path(output_root)
    run_id = run_id or new_run_id("phase1a")
    campaign_id = campaign_id or f"campaign-{run_id}"
    protocol = phase1a_commodity_baseline_protocol(config)
    protocol_hash = phase1a_commodity_baseline_protocol_hash(config)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "run.json",
        {
            "schema": "sensetrace.run.v1",
            "run_id": run_id,
            "campaign_id": campaign_id,
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": _git_commit(),
            "configuration_hash": config_fingerprint(config),
            "protocol_version": protocol["version"],
            "protocol_hash": protocol_hash,
            "claim_scope": "exploratory safe commodity-memory host observables only",
            "phase0_gate_report": str(phase0_report)
            if not isinstance(phase0_report, dict)
            else "embedded",
        },
    )
    host_snapshot = collect_inventory()
    _write_json(run_dir / "host.json", host_snapshot)
    _write_json(run_dir / "config.json", normalized_config(config))
    _write_json(run_dir / "protocol.json", {**protocol, "protocol_hash": protocol_hash})
    _write_json(
        run_dir / "campaign.json",
        {
            "schema": "sensetrace.phase1a-campaign.v1",
            "campaign_id": campaign_id,
            "run_id": run_id,
            "status": "active",
            "session_count_per_acquired_condition": _session_count(config),
            "source_manifests_preserved": True,
            "protocol_version": protocol["version"],
            "protocol_hash": protocol_hash,
            "acquisition_graph": "campaign -> boot -> acquisition session -> acquisition block -> virtual location -> pair -> trial",
        },
    )
    journal = Journal(run_dir / "events.jsonl")
    journal.append(
        "campaign_started",
        campaign_id=campaign_id,
        run_id=run_id,
        session_count_per_acquired_condition=_session_count(config),
    )
    if (
        bool(config.get("phase1a", {}).get("require_native_kernel", True))
        and NativeMeasurementKernel.load() is None
    ):
        raise RuntimeError("Phase 1A requires the built native timing kernel; run make -C native")

    datasets_dir = run_dir / "datasets"
    acquired_specs = [
        ("paired_single_bit", {}),
        ("cache_hit_control", {"cache_control": "none", "pattern": "single_bit"}),
        (
            "idle_no_memory_operation",
            {"operation": "idle", "cache_control": "none", "pattern": "single_bit"},
        ),
        ("all_zero_vs_all_one", {"pattern": "all_zero_one"}),
        ("random_word_null", {"pattern": "random_word"}),
    ]
    manifests: dict[str, dict[str, Any]] = {}
    for name, overrides in acquired_specs:
        condition_config = json.loads(json.dumps(config))
        physical = dict(condition_config.get("phase1a", {}))
        physical.update(overrides)
        condition_config["phase1a"] = physical
        manifests[name] = _materialize_acquired_condition(
            condition_config,
            datasets_dir / name,
            condition=name,
            campaign_id=campaign_id,
            host_inventory_snapshot=host_snapshot,
            session_count=_session_count(config),
        )

    shuffled_dir = datasets_dir / "paired_single_bit_shuffled"
    manifests["paired_single_bit_shuffled"] = _materialize_label_permutation(
        datasets_dir / "paired_single_bit",
        shuffled_dir,
        config,
        int(config.get("experiment", {}).get("seed", 1337)) + 7919,
    )

    result_conditions: dict[str, Any] = {}
    for name, manifest in manifests.items():
        result_conditions[name] = _analyze_condition(datasets_dir / name, manifest, config)
        journal.append(
            "condition_analysis_completed",
            condition=name,
            dataset_fingerprint=manifest["dataset_fingerprint"],
            available_splits=result_conditions[name]["analysis_summary"]["available_split_count"],
        )
    journal.append("campaign_completed", campaign_id=campaign_id, status="exploratory")
    final = {
        "schema": "sensetrace.phase1a-report.v4",
        "run_id": run_id,
        "campaign_id": campaign_id,
        "status": "exploratory",
        "protocol": {**protocol, "protocol_hash": protocol_hash},
        "conditions": result_conditions,
        "campaign": {
            "session_count_per_acquired_condition": _session_count(config),
            "source_manifests_preserved": True,
            "combined_dataset_provenance": "each combined condition manifest references every finalized source session manifest",
        },
        "measurement_provenance": {
            "protocol_identity": protocol["version"],
            "protocol_hash": protocol_hash,
            "backend": "CommodityDramBackend",
            "ordinary_read_value": "ground-truth verification only",
            "artificial_timing_perturbation_invariant": {
                "allowed": False,
                "timing_perturbation_cycles": 0,
                "label_correlated": False,
                "calibration_namespace": "forbidden",
                "enforcement": (
                    "ordinary Phase 1A rejects nonzero cycles, non-default labels, and any "
                    "calibration namespace before creating a run"
                ),
            },
            "physical_address_or_row_claim": "not available",
            "virtual_location_definition": "controlled offset in a fresh anonymous virtual buffer; not a known DRAM row, cell, bank, subarray, chip, or DIMM location",
            "paired_target_construction": "one random base word per pair; target bit forced to 0/1; exact half of pairs use each label order, with pair types randomized",
            "acquisition_session_definition": "one independently started backend with its own UUID, start time, host snapshot, boot ID, fresh controlled buffer, label stream, journal, kernel provenance, and hashes",
            "clflush_provenance": {
                "primitive": "_mm_clflush(address) followed by _mm_mfence() before each timed load",
                "timing_fences": "LFENCE/RDTSC at start and RDTSCP/LFENCE at end",
                "guarantee": "requests invalidation of the addressed cache line when native CPU support is reported",
                "limitations": "does not prove a DRAM access or expose physical address, row, bank, subarray, chip, or DIMM identity",
            },
            "negative_controls": [
                "paired_single_bit_shuffled",
                "random_word_null",
                "cache_hit_control",
            ],
            "next_step": "freeze method and collect a new session before any confirmatory claim",
        },
    }
    _write_json(run_dir / "metrics.json", final)
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_record.update({"status": "completed", "updated_at": datetime.now(UTC).isoformat()})
    _write_json(run_dir / "run.json", run_record)
    return final


def run_phase1a(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    phase0_report: str | Path | dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible single entry point for a first-class campaign."""

    if phase0_report is None:
        raise RuntimeError(
            "Phase 1A is gated: provide a Phase 0 report whose acceptance.phase1_gate is true"
        )
    return run_phase1a_campaign(config, output_root, phase0_report=phase0_report, run_id=run_id)
