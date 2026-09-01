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
from .hashing import sha256_json, sha256_text
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
) -> dict[str, Any]:
    root = Path(run_dir)
    config_hash = config_fingerprint(config)
    fingerprint = dataset_fingerprint(shard_infos, config_hash=config_hash)
    manifest = {
        "schema": "sensetrace.dataset-manifest.v1",
        "dataset_schema": SCHEMA_VERSION,
        "condition": condition,
        "created_at": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "dataset_fingerprint": fingerprint,
        "label_stream_fingerprint": label_stream_fingerprint,
        "class_balance": class_balance or {"0": "unavailable", "1": "unavailable"},
        "rows": sum(info.rows for info in shard_infos),
        "shards": [info.as_dict() for info in shard_infos],
    }
    if provenance:
        manifest["provenance"] = provenance
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
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[ShardInfo], dict[str, Any]]:
    root = Path(run_dir)
    manifest = read_dataset_manifest(root)
    traces, labels, metadata, shards = load_shards(root)
    expected = dataset_fingerprint(shards, config_hash=manifest["config_hash"])
    if expected != manifest.get("dataset_fingerprint"):
        raise IntegrityError("dataset fingerprint does not match finalized shard evidence")
    if manifest.get("rows") != len(labels):
        raise IntegrityError("dataset manifest row count does not match shards")
    return traces, labels, metadata, shards, manifest


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
