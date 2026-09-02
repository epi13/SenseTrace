"""Dataset manifests, fingerprints, and model feature construction."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import config_fingerprint
from .errors import IntegrityError, SchemaError
from .hashing import sha256_bytes, sha256_json, sha256_text
from .protocol import PHASE1A_COMMODITY_BASELINE_VERSION
from .schema import SCHEMA_VERSION, FeaturePolicy
from .storage import ShardInfo, dataset_fingerprint, load_shards


def write_dataset_manifest(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    condition: str,
    shard_infos: list[ShardInfo],
    label_stream_fingerprint: str,
    class_balance: dict[str, int | str] | None = None,
    provenance: dict[str, Any] | None = None,
    acquisition_sessions: list[dict[str, Any]] | None = None,
    campaign_id: str | None = None,
    dataset_purpose: str | None = None,
    protocol_identity: str | None = None,
    protocol_hash: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    config_hash = config_fingerprint(config)
    manifest_provenance = dict(provenance or {})
    effective_purpose = str(
        dataset_purpose or manifest_provenance.get("dataset_purpose", "generic")
    )
    effective_protocol_identity = protocol_identity or manifest_provenance.get("protocol_identity")
    effective_protocol_hash = protocol_hash or manifest_provenance.get("protocol_hash")
    fingerprint = dataset_fingerprint(shard_infos, config_hash=config_hash)
    manifest = {
        "schema": "sensetrace.dataset-manifest.v1",
        "dataset_schema": SCHEMA_VERSION,
        "condition": condition,
        "dataset_purpose": effective_purpose,
        "created_at": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "dataset_fingerprint": fingerprint,
        "label_stream_fingerprint": label_stream_fingerprint,
        "class_balance": class_balance or {"0": "unavailable", "1": "unavailable"},
        "rows": sum(info.rows for info in shard_infos),
        "shards": [info.as_dict() for info in shard_infos],
    }
    if effective_protocol_identity is not None:
        manifest["protocol_identity"] = str(effective_protocol_identity)
    if effective_protocol_hash is not None:
        manifest["protocol_hash"] = str(effective_protocol_hash)
    if manifest_provenance:
        manifest["provenance"] = manifest_provenance
    if acquisition_sessions is not None:
        manifest["acquisition_sessions"] = acquisition_sessions
    if campaign_id is not None:
        manifest["campaign_id"] = campaign_id
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_dataset_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "dataset.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read dataset manifest {path}") from exc
    if manifest.get("schema") != "sensetrace.dataset-manifest.v1":
        raise IntegrityError("unsupported dataset manifest schema")
    return manifest


def load_dataset(
    run_dir: str | Path,
    *,
    expected_purpose: str | None = None,
    expected_protocol_identity: str | None = None,
    expected_protocol_hash: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[ShardInfo], dict[str, Any]]:
    root = Path(run_dir)
    manifest = read_dataset_manifest(root)
    traces, labels, metadata, shards = load_shards(root)
    expected = dataset_fingerprint(shards, config_hash=manifest["config_hash"])
    if expected != manifest.get("dataset_fingerprint"):
        raise IntegrityError("dataset fingerprint does not match finalized shard evidence")
    if manifest.get("rows") != len(labels):
        raise IntegrityError("dataset manifest row count does not match shards")
    if expected_purpose == "physical_phase1a":
        _validate_physical_phase1a_dataset(
            metadata,
            manifest,
            root,
            expected_protocol_identity=expected_protocol_identity,
            expected_protocol_hash=expected_protocol_hash,
        )
    else:
        _validate_physical_timing_provenance(metadata, manifest, root)
        if expected_purpose is not None and manifest.get("dataset_purpose") != expected_purpose:
            raise IntegrityError(
                f"dataset {root} purpose {manifest.get('dataset_purpose')!r} does not match "
                f"expected purpose {expected_purpose!r}"
            )
    _validate_allocation_provenance(metadata, manifest, root)
    return traces, labels, metadata, shards, manifest


def _metadata_values(metadata: dict[str, np.ndarray], field: str) -> list[str]:
    if field not in metadata:
        raise IntegrityError(f"physical Phase 1A dataset is missing required shard metadata {field!r}")
    return [str(value) for value in np.asarray(metadata[field], dtype=object)]


def _require_all_values(values: list[str], expected: str, field: str) -> None:
    if not values or any(value != expected for value in values):
        raise IntegrityError(
            f"physical Phase 1A shard metadata {field!r} is not uniformly {expected!r}"
        )


def _validate_physical_phase1a_dataset(
    metadata: dict[str, np.ndarray],
    manifest: dict[str, Any],
    source: Path,
    *,
    expected_protocol_identity: str | None,
    expected_protocol_hash: str | None,
) -> None:
    """Validate the complete physical Phase 1A identity and scope boundary."""

    if manifest.get("dataset_purpose") != "physical_phase1a":
        raise IntegrityError(
            f"dataset {source} is not explicitly marked for physical Phase 1A analysis"
        )
    identity = manifest.get("protocol_identity")
    protocol_hash = manifest.get("protocol_hash")
    if identity != PHASE1A_COMMODITY_BASELINE_VERSION:
        raise IntegrityError(
            "physical Phase 1A requires protocol identity "
            f"{PHASE1A_COMMODITY_BASELINE_VERSION!r}; found {identity!r}"
        )
    if not isinstance(protocol_hash, str) or not protocol_hash:
        raise IntegrityError("physical Phase 1A requires a non-empty protocol hash")
    if expected_protocol_identity is not None and identity != expected_protocol_identity:
        raise IntegrityError("physical Phase 1A protocol identity does not match the analysis boundary")
    if expected_protocol_hash is not None and protocol_hash != expected_protocol_hash:
        raise IntegrityError("physical Phase 1A protocol hash does not match the analysis boundary")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise IntegrityError("physical Phase 1A requires a manifest provenance mapping")
    if provenance.get("protocol_identity") != identity:
        raise IntegrityError("manifest and provenance protocol identities disagree")
    if provenance.get("protocol_hash") != protocol_hash:
        raise IntegrityError("manifest and provenance protocol hashes disagree")
    scope = provenance.get("artificial_timing_perturbation")
    if not isinstance(scope, dict):
        raise IntegrityError("physical Phase 1A requires an explicit timing scope contract")
    required_scope = {
        "allowed": False,
        "timing_perturbation_cycles": 0,
        "timing_perturbation_label": 1,
        "label_correlated": False,
        "applied": False,
        "calibration_namespace": "forbidden",
        "physical_phase1a_forbidden": True,
    }
    if any(scope.get(field) != value for field, value in required_scope.items()):
        raise IntegrityError("physical Phase 1A manifest timing scope is missing or contradictory")

    _require_all_values(_metadata_values(metadata, "protocol_identity"), str(identity), "protocol_identity")
    _require_all_values(_metadata_values(metadata, "protocol_hash"), str(protocol_hash), "protocol_hash")
    _require_all_values(
        _metadata_values(metadata, "timing_perturbation_cycles"), "0", "timing_perturbation_cycles"
    )
    _require_all_values(
        _metadata_values(metadata, "timing_perturbation_label"), "1", "timing_perturbation_label"
    )
    _require_all_values(
        _metadata_values(metadata, "timing_perturbation_applied"), "False", "timing_perturbation_applied"
    )
    _require_all_values(
        _metadata_values(metadata, "calibration_namespace"), "not_calibration", "calibration_namespace"
    )
    _require_all_values(
        _metadata_values(metadata, "artificial_timing_perturbation_allowed"),
        "False",
        "artificial_timing_perturbation_allowed",
    )

    ledgers = manifest.get("acquisition_sessions")
    if not isinstance(ledgers, list) or not ledgers or any(not isinstance(item, dict) for item in ledgers):
        raise IntegrityError("physical Phase 1A requires complete source session ledgers")
    ledger_ids: set[str] = set()
    for ledger in ledgers:
        ledger_identity = ledger.get("protocol_identity")
        ledger_hash = ledger.get("protocol_hash")
        if ledger_identity != identity or ledger_hash != protocol_hash:
            raise IntegrityError("source session ledger protocol identity/hash disagrees with manifest")
        if ledger.get("acquisition_scope") != "physical Phase 1A commodity baseline":
            raise IntegrityError("source session ledger is outside the physical Phase 1A scope")
        ledger_scope = ledger.get("timing_perturbation")
        if not isinstance(ledger_scope, dict) or any(
            ledger_scope.get(field) != value
            for field, value in {
                "cycles": 0,
                "label": 1,
                "allowed": False,
                "physical_phase1a_forbidden": True,
                "namespace": "not_calibration",
            }.items()
        ):
            raise IntegrityError("source session ledger timing scope is missing or contradictory")
        ledger_id = str(ledger.get("acquisition_session_id", ""))
        allocation = ledger.get("controlled_memory_region", {})
        if not ledger_id or not isinstance(allocation, dict) or not allocation.get("allocation_id"):
            raise IntegrityError("source session ledger lacks acquisition or allocation identity")
        ledger_ids.add(ledger_id)
    metadata_session_values = set(_metadata_values(metadata, "acquisition_session_id"))
    if metadata_session_values != ledger_ids:
        raise IntegrityError("shard metadata sessions do not match the source session ledgers")

    source_manifest_file = source / "source-manifests.json"
    if source_manifest_file.exists():
        try:
            source_record = json.loads(source_manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("cannot read combined dataset source-manifests.json") from exc
        embedded = source_record.get("source_manifests")
        if not isinstance(embedded, list) or not embedded:
            raise IntegrityError("combined physical dataset has no embedded source manifests")
        if any(
            not isinstance(item, dict)
            or item.get("dataset_purpose") != "physical_phase1a"
            or item.get("protocol_identity") != identity
            or item.get("protocol_hash") != protocol_hash
            for item in embedded
        ):
            raise IntegrityError("combined source manifests disagree with physical protocol scope")


def _validate_physical_timing_provenance(
    metadata: dict[str, np.ndarray], manifest: dict[str, Any], source: Path
) -> None:
    """Fail closed when baseline evidence carries a calibration perturbation."""

    provenance = manifest.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    protocol_identity = manifest.get("protocol_identity") or provenance.get("protocol_identity")
    if protocol_identity != PHASE1A_COMMODITY_BASELINE_VERSION:
        return

    violations: list[str] = []

    def values(name: str) -> list[object]:
        return list(np.asarray(metadata.get(name, []), dtype=object))

    cycles = values("timing_perturbation_cycles")
    for value in cycles:
        try:
            if int(str(value)) != 0:
                violations.append(f"timing_perturbation_cycles={value}")
        except (TypeError, ValueError):
            violations.append(f"invalid timing_perturbation_cycles={value!r}")
    labels = values("timing_perturbation_label")
    if any(str(value) != "1" for value in labels):
        violations.append("non-default timing_perturbation_label")

    def is_true(value: object) -> bool:
        return (
            isinstance(value, (bool, np.bool_))
            and bool(value)
            or str(value).lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    if any(is_true(value) for value in values("timing_perturbation_applied")):
        violations.append("timing_perturbation_applied=true")
    namespaces = values("calibration_namespace")
    if any(
        str(value) not in {"", "none", "not_calibration", "unavailable"} for value in namespaces
    ):
        violations.append("calibration namespace present")

    timing_contract = provenance.get("artificial_timing_perturbation")
    if isinstance(timing_contract, dict):
        contract_cycles = timing_contract.get("timing_perturbation_cycles", 0)
        try:
            contract_nonzero = int(str(contract_cycles)) != 0
        except (TypeError, ValueError):
            contract_nonzero = True
        if (
            is_true(timing_contract.get("allowed"))
            or is_true(timing_contract.get("applied"))
            or contract_nonzero
        ):
            violations.append("manifest artificial-timing contract is not physical-zero")
    if violations:
        raise IntegrityError(
            f"physical commodity dataset {source} contains calibration contamination: "
            + ", ".join(sorted(set(violations)))
        )


def _validate_allocation_provenance(
    metadata: dict[str, np.ndarray], manifest: dict[str, Any], source: Path
) -> None:
    """Reject a source dataset that blurs one session across allocations."""

    session_field = (
        "acquisition_session_id" if "acquisition_session_id" in metadata else "session_id"
    )
    if session_field not in metadata or "allocation_id" not in metadata:
        return
    session_values = np.asarray(metadata[session_field]).astype(str)
    allocation_values = np.asarray(metadata["allocation_id"]).astype(str)
    allocations: dict[str, set[str]] = {}
    for session, allocation in zip(session_values, allocation_values, strict=True):
        allocations.setdefault(str(session), set()).add(str(allocation))
    blurred = {session: values for session, values in allocations.items() if len(values) != 1}
    if blurred:
        raise IntegrityError(
            f"dataset {source} has session identities spanning multiple allocation IDs: {blurred}"
        )
    ledgers = manifest.get("acquisition_sessions", [])
    expected: dict[str, str] = {}
    for ledger in ledgers:
        if not isinstance(ledger, dict):
            continue
        status = ledger.get("status")
        if status in {"active", "interrupted", "incomplete"}:
            raise IntegrityError(
                f"dataset {source} references an incomplete acquisition session ({status})"
            )
        session = str(ledger.get("acquisition_session_id", ledger.get("session_id", "")))
        allocation = str(ledger.get("controlled_memory_region", {}).get("allocation_id", ""))
        if session and allocation and allocation not in {"unavailable", "unknown"}:
            expected[session] = allocation
    mismatched = {
        session: (next(iter(values)), expected[session])
        for session, values in allocations.items()
        if session in expected and next(iter(values)) != expected[session]
    }
    if mismatched:
        raise IntegrityError(
            f"dataset {source} allocation metadata disagrees with its session ledger: {mismatched}"
        )


def trace_features(traces: np.ndarray, *, bins: int = 32) -> np.ndarray:
    """Create fixed, transparent summary features without fitting on test data."""

    if traces.ndim != 2:
        raise SchemaError("traces must be [samples, time]")
    if bins < 1 or bins > traces.shape[1]:
        raise SchemaError("feature bins must fit within trace length")
    chunks = np.array_split(traces.astype(np.float32, copy=False), bins, axis=1)
    features = [
        traces.mean(axis=1),
        traces.std(axis=1),
        traces.min(axis=1),
        traces.max(axis=1),
        np.sqrt(np.mean(np.square(traces), axis=1)),
    ]
    features.extend(chunk.mean(axis=1) for chunk in chunks)
    features.extend(chunk.std(axis=1) for chunk in chunks)
    return np.column_stack(features).astype(np.float32, copy=False)


def build_feature_matrix(
    traces: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    feature_fields: Iterable[str] = (),
    policy: FeaturePolicy | None = None,
    allow_identity: bool = False,
) -> np.ndarray:
    fields = list(feature_fields)
    (policy or FeaturePolicy()).validate(fields, allow_identity=allow_identity)
    matrix = [trace_features(traces)]
    for field in fields:
        if field not in metadata:
            raise SchemaError(f"requested feature field not present: {field}")
        values = np.asarray(metadata[field])
        if values.dtype.kind in "iufb":
            encoded = values.astype(np.float32)
        elif allow_identity and field in (policy or FeaturePolicy()).prohibited:
            encoded = np.asarray(
                [int(sha256_text(str(value))[:8], 16) / 2**32 for value in values],
                dtype=np.float32,
            )
        else:
            raise SchemaError(f"feature field {field} is not numeric")
        matrix.append(encoded.reshape(-1, 1))
    return np.concatenate(matrix, axis=1)


def ensure_sample_ids(metadata: dict[str, np.ndarray]) -> list[str]:
    if "sample_id" not in metadata:
        raise SchemaError("dataset has no sample_id metadata")
    return [str(value) for value in metadata["sample_id"]]


def fingerprint_split(split: dict[str, Any]) -> str:
    material = dict(split)
    material.pop("split_fingerprint", None)
    return sha256_json(material)


def combine_datasets(
    source_dirs: Iterable[str | Path],
    target_dir: str | Path,
    *,
    config: dict[str, Any],
    condition: str,
    campaign_id: str,
    source_manifest_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Combine finalized session datasets without discarding source provenance.

    Source session manifests remain embedded in ``source_manifests.json`` and
    referenced from the combined manifest. Existing finalized rows are keyed
    by globally unique sample_id, so a crash/restart during combination can
    safely resume without duplicating samples.
    """

    from .storage import ShardWriter, validate_all_shards

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    sources = [Path(path) for path in source_dirs]
    if not sources:
        raise IntegrityError("cannot combine an empty source dataset list")
    loaded = [load_dataset(path) for path in sources]
    source_manifests = [item[4] for item in loaded]
    source_manifest_paths = source_manifest_paths or [
        str(path / "dataset.json") for path in sources
    ]
    source_purposes = {str(manifest.get("dataset_purpose", "generic")) for manifest in source_manifests}
    if len(source_purposes) != 1:
        raise IntegrityError("cannot combine source datasets with different dataset purposes")
    combined_purpose = next(iter(source_purposes))
    if combined_purpose == "physical_phase1a":
        # Re-open through the strict boundary before any rows are copied.
        loaded = [load_dataset(path, expected_purpose="physical_phase1a") for path in sources]
        source_manifests = [item[4] for item in loaded]
    source_protocols = {
        (manifest.get("protocol_identity"), manifest.get("protocol_hash"))
        for manifest in source_manifests
    }
    if len(source_protocols) != 1:
        raise IntegrityError("cannot combine source datasets with different protocol identities/hashes")
    combined_protocol_identity, combined_protocol_hash = next(iter(source_protocols))
    if combined_purpose == "physical_phase1a" and (
        combined_protocol_identity != PHASE1A_COMMODITY_BASELINE_VERSION
        or not isinstance(combined_protocol_hash, str)
        or not combined_protocol_hash
    ):
        raise IntegrityError("physical Phase 1A source datasets require explicit protocol identity/hash")

    existing_ids: set[str] = set()
    seen_session_ids: dict[str, Path] = {}
    for source, manifest in zip(sources, source_manifests, strict=True):
        for ledger in manifest.get("acquisition_sessions", []):
            if not isinstance(ledger, dict):
                continue
            session_id = str(ledger.get("acquisition_session_id", ledger.get("session_id", "")))
            if not session_id:
                continue
            if session_id in seen_session_ids and seen_session_ids[session_id] != source:
                raise IntegrityError(
                    "one acquisition_session_id appears in multiple source datasets; "
                    "combine only independently materialized sessions"
                )
            seen_session_ids[session_id] = source
    if sorted(target.glob("shard-*.npz")):
        _existing_traces, _existing_labels, existing_metadata, _existing_shards = load_shards(
            target
        )
        existing_ids.update(ensure_sample_ids(existing_metadata))
    row_refs: list[tuple[str, int, int]] = []
    source_ids: set[str] = set()
    for dataset_index, (_traces, _labels, metadata, _shards, _manifest) in enumerate(loaded):
        ids = ensure_sample_ids(metadata)
        for row_index, sample_id in enumerate(ids):
            if sample_id in source_ids:
                raise IntegrityError(
                    f"sample_id {sample_id!r} appears more than once across source datasets"
                )
            source_ids.add(sample_id)
            row_refs.append((sample_id, dataset_index, row_index))
    # Source session directories are not necessarily ordered by sample_id (the
    # IDs include independently generated session UUIDs). Canonicalize the
    # merged stream before writing so validate_all_shards can enforce its
    # monotonic shard-boundary invariant without weakening that invariant.
    row_refs.sort(key=lambda item: item[0])

    writer = ShardWriter(
        target,
        shard_target_mb=float(config.get("acquisition", {}).get("shard_target_mb", 512)),
        max_samples_per_shard=config.get("acquisition", {}).get("max_samples_per_shard"),
    )
    for sample_id, dataset_index, row_index in row_refs:
        if sample_id in existing_ids:
            continue
        traces, labels, metadata, _shards, _manifest = loaded[dataset_index]
        row_metadata = {key: values[row_index] for key, values in metadata.items()}
        writer.add(traces[row_index], int(labels[row_index]), row_metadata)
        existing_ids.add(sample_id)
    writer.finalize()
    shards = validate_all_shards(target)
    _combined_traces, combined_labels, _combined_metadata, _combined_shards = load_shards(target)
    sessions: list[dict[str, Any]] = []
    for manifest in source_manifests:
        sessions.extend(manifest.get("acquisition_sessions", []))
    combined = write_dataset_manifest(
        target,
        config=config,
        condition=condition,
        shard_infos=shards,
        label_stream_fingerprint=sha256_bytes(combined_labels.tobytes()),
        class_balance={
            "0": int(np.sum(combined_labels == 0)),
            "1": int(np.sum(combined_labels == 1)),
        },
        provenance={
            "campaign_combination": "finalized source session datasets combined by globally unique sample_id",
            "source_manifest_paths": source_manifest_paths,
            "source_dataset_fingerprints": [
                manifest.get("dataset_fingerprint", "unavailable") for manifest in source_manifests
            ],
            **(
                {
                    "protocol_identity": combined_protocol_identity,
                    "protocol_hash": combined_protocol_hash,
                    "artificial_timing_perturbation": {
                        "allowed": False,
                        "timing_perturbation_cycles": 0,
                        "timing_perturbation_label": 1,
                        "label_correlated": False,
                        "applied": False,
                        "calibration_namespace": "forbidden",
                        "physical_phase1a_forbidden": True,
                    },
                }
                if combined_purpose == "physical_phase1a"
                else {}
            ),
        },
        acquisition_sessions=sessions,
        campaign_id=campaign_id,
        dataset_purpose=combined_purpose,
        protocol_identity=(str(combined_protocol_identity) if combined_protocol_identity else None),
        protocol_hash=(str(combined_protocol_hash) if combined_protocol_hash else None),
    )
    (target / "source-manifests.json").write_text(
        json.dumps(
            {
                "schema": "sensetrace.campaign-source-manifests.v1",
                "campaign_id": campaign_id,
                "source_manifest_paths": source_manifest_paths,
                "source_manifests": source_manifests,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return combined
