"""Crash-safe sequential NPZ shard storage."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IntegrityError, SchemaError
from .hashing import sha256_file, sha256_json
from .schema import SCHEMA_VERSION, validate_arrays


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    metadata_path: Path
    shard_index: int
    rows: int
    first_sample_id: str
    last_sample_id: str
    sha256: str
    schema: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "metadata_path": self.metadata_path.name,
            "shard_index": self.shard_index,
            "rows": self.rows,
            "first_sample_id": self.first_sample_id,
            "last_sample_id": self.last_sample_id,
            "sha256": self.sha256,
            "schema": self.schema,
        }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _string_array(values: list[object]) -> np.ndarray:
    """Encode one metadata column without reserving a huge fixed width.

    NumPy's Unicode dtype is fixed-width, but its inferred width is based on
    the longest value in this column.  The previous rescue attempt used
    ``U65536`` for every field so nested controller provenance could not be
    truncated.  That made an ordinary 1,600-row Phase 0 shard allocate many
    gigabytes before compression (each Unicode code point uses four bytes).
    Inferring the smallest lossless width preserves the provenance while
    keeping resident memory proportional to the actual data.
    """

    strings = [str(value) for value in values]
    return np.asarray(strings, dtype=np.str_)


class ShardWriter:
    """Bounded-memory writer that finalizes payload and metadata atomically."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        shard_target_mb: float = 512,
        max_samples_per_shard: int | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.shard_target_bytes = int(shard_target_mb * 1024 * 1024)
        self.max_samples_per_shard = max_samples_per_shard
        self._traces: list[np.ndarray] = []
        self._labels: list[int] = []
        self._metadata: dict[str, list[object]] = {}
        self._next_shard = self._discover_next_shard()

    def _discover_next_shard(self) -> int:
        indexes = []
        for path in self.run_dir.glob("shard-*.npz"):
            try:
                indexes.append(int(path.stem.split("-")[-1]))
            except ValueError:
                continue
        return max(indexes, default=-1) + 1

    @property
    def buffered_rows(self) -> int:
        return len(self._labels)

    @property
    def estimated_buffer_bytes(self) -> int:
        """Approximate resident payload size before the next shard flush."""

        trace_bytes = sum(item.nbytes for item in self._traces)
        metadata_bytes = sum(
            sum(len(str(value).encode("utf-8")) for value in values)
            for values in self._metadata.values()
        )
        return int(trace_bytes + metadata_bytes + len(self._labels))

    def add(self, trace: np.ndarray, label: int, metadata: dict[str, object]) -> ShardInfo | None:
        if trace.ndim != 1:
            raise SchemaError("each trace must be one-dimensional")
        if label not in (0, 1):
            raise SchemaError("each label must be 0 or 1")
        if not self._metadata:
            self._metadata = {key: [] for key in metadata}
        if set(metadata) != set(self._metadata):
            raise SchemaError("metadata keys changed within a shard")
        self._traces.append(np.asarray(trace, dtype=np.float32))
        self._labels.append(label)
        for key, value in metadata.items():
            self._metadata[key].append(value)
        estimated_bytes = self.estimated_buffer_bytes
        if self.max_samples_per_shard and len(self._labels) >= self.max_samples_per_shard:
            return self.finalize()
        if estimated_bytes >= self.shard_target_bytes:
            return self.finalize()
        return None

    def finalize(self) -> ShardInfo | None:
        if not self._labels:
            return None
        traces = np.stack(self._traces).astype(np.float32, copy=False)
        labels = np.asarray(self._labels, dtype=np.uint8)
        metadata = {key: _string_array(values) for key, values in self._metadata.items()}
        metadata["trial_index"] = np.asarray(self._metadata["trial_index"], dtype=np.int64)
        validate_arrays(traces, labels, metadata)
        index = self._next_shard
        stem = f"shard-{index:06d}"
        payload_tmp = self.run_dir / f"{stem}.npz.tmp"
        metadata_tmp = self.run_dir / f"{stem}.json.tmp"
        payload = self.run_dir / f"{stem}.npz"
        metadata_path = self.run_dir / f"{stem}.json"
        payload_arrays: dict[str, Any] = {"trace": traces, "label": labels, **metadata}
        np.savez_compressed(payload_tmp, **payload_arrays)
        # numpy appends .npz when passed a suffix it does not recognize.
        actual_payload_tmp = (
            payload_tmp if payload_tmp.exists() else self.run_dir / f"{stem}.npz.tmp.npz"
        )
        if actual_payload_tmp != payload_tmp:
            actual_payload_tmp.replace(payload_tmp)
        with payload_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        checksum = sha256_file(payload_tmp)
        sidecar = {
            "schema": SCHEMA_VERSION,
            "created_at": time.time(),
            "rows": len(labels),
            "trace_shape": list(traces.shape),
            "arrays": ["trace", "label", *sorted(metadata)],
            "first_sample_id": str(metadata["sample_id"][0]),
            "last_sample_id": str(metadata["sample_id"][-1]),
            "sha256": checksum,
        }
        with metadata_tmp.open("w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(payload_tmp, payload)
        os.replace(metadata_tmp, metadata_path)
        _fsync_directory(self.run_dir)
        info = ShardInfo(
            path=payload,
            metadata_path=metadata_path,
            shard_index=index,
            rows=len(labels),
            first_sample_id=str(metadata["sample_id"][0]),
            last_sample_id=str(metadata["sample_id"][-1]),
            sha256=checksum,
        )
        self._next_shard += 1
        self._traces.clear()
        self._labels.clear()
        self._metadata.clear()
        return info


def list_finalized_shards(run_dir: str | Path) -> list[Path]:
    return sorted(Path(run_dir).glob("shard-*.npz"))


def quarantine_temporary_shards(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    quarantine = root / "quarantine"
    moved: list[Path] = []
    for path in sorted(root.glob("*.tmp")):
        quarantine.mkdir(exist_ok=True)
        target = quarantine / f"{path.name}.{int(time.time() * 1000)}"
        shutil.move(str(path), str(target))
        moved.append(target)
    return moved


def quarantine_invalid_shards(run_dir: str | Path) -> list[Path]:
    """Move invalid finalized payloads aside so recovery never consumes them silently."""
    root = Path(run_dir)
    quarantine = root / "quarantine"
    moved: list[Path] = []
    for path in list_finalized_shards(root):
        try:
            validate_shard(path)
        except IntegrityError:
            quarantine.mkdir(exist_ok=True)
            stamp = int(time.time() * 1000)
            target = quarantine / f"{path.name}.invalid.{stamp}"
            shutil.move(str(path), str(target))
            moved.append(target)
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                side_target = quarantine / f"{sidecar.name}.invalid.{stamp}"
                shutil.move(str(sidecar), str(side_target))
                moved.append(side_target)
    return moved


def validate_shard(payload: str | Path) -> ShardInfo:
    path = Path(payload)
    if path.suffix != ".npz":
        raise IntegrityError(f"not a finalized NPZ shard: {path.name}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise IntegrityError(f"missing shard sidecar: {metadata_path.name}")
    try:
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid shard sidecar: {metadata_path.name}") from exc
    if sidecar.get("schema") != SCHEMA_VERSION:
        raise IntegrityError(f"unsupported shard schema in {path.name}")
    actual_checksum = sha256_file(path)
    if actual_checksum != sidecar.get("sha256"):
        raise IntegrityError(f"checksum mismatch for {path.name}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"trace", "label"}.issubset(archive.files):
                raise IntegrityError(f"missing required arrays in {path.name}")
            trace = archive["trace"]
            labels = archive["label"]
            metadata = {key: archive[key] for key in archive.files if key not in {"trace", "label"}}
            validate_arrays(trace, labels, metadata)
    except (OSError, ValueError, KeyError) as exc:
        raise IntegrityError(f"cannot read shard {path.name}: {exc}") from exc
    rows = int(sidecar.get("rows", -1))
    if rows != len(labels):
        raise IntegrityError(f"row count mismatch for {path.name}")
    return ShardInfo(
        path=path,
        metadata_path=metadata_path,
        shard_index=int(path.stem.split("-")[-1]),
        rows=rows,
        first_sample_id=str(sidecar["first_sample_id"]),
        last_sample_id=str(sidecar["last_sample_id"]),
        sha256=actual_checksum,
    )


def validate_all_shards(run_dir: str | Path) -> list[ShardInfo]:
    infos = [validate_shard(path) for path in list_finalized_shards(run_dir)]
    previous_last: str | None = None
    for info in infos:
        if previous_last is not None and info.first_sample_id <= previous_last:
            raise IntegrityError("finalized shards have overlapping or unsorted sample ranges")
        previous_last = info.last_sample_id
    return infos


def load_shards(
    run_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[ShardInfo]]:
    infos = validate_all_shards(run_dir)
    if not infos:
        raise IntegrityError(f"no finalized shards in {run_dir}")
    traces: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata_rows: dict[str, list[np.ndarray]] = {}
    for info in infos:
        with np.load(info.path, allow_pickle=False) as archive:
            traces.append(archive["trace"])
            labels.append(archive["label"])
            for key in archive.files:
                if key not in {"trace", "label"}:
                    metadata_rows.setdefault(key, []).append(archive[key])
    metadata = {key: np.concatenate(values) for key, values in metadata_rows.items()}
    return np.concatenate(traces), np.concatenate(labels), metadata, infos


def iter_shards(
    run_dir: str | Path,
) -> Iterable[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], ShardInfo]]:
    """Yield one validated shard at a time.

    This is the preferred loader for large corpora.  Unlike ``load_shards`` it
    never concatenates the dataset or retains arrays from earlier shards.
    """

    for info in validate_all_shards(run_dir):
        try:
            with np.load(info.path, allow_pickle=False) as archive:
                traces = np.asarray(archive["trace"], dtype=np.float32).copy()
                labels = np.asarray(archive["label"], dtype=np.uint8).copy()
                metadata = {
                    key: np.asarray(archive[key]).copy()
                    for key in archive.files
                    if key not in {"trace", "label"}
                }
        except (OSError, ValueError, KeyError) as exc:
            raise IntegrityError(f"cannot stream shard {info.path.name}: {exc}") from exc
        yield traces, labels, metadata, info


def iter_sample_rows(
    run_dir: str | Path,
) -> Iterable[tuple[np.ndarray, int, dict[str, object]]]:
    """Yield individual rows while retaining at most one shard in memory."""

    for traces, labels, metadata, _info in iter_shards(run_dir):
        keys = tuple(metadata)
        for index, trace in enumerate(traces):
            yield trace, int(labels[index]), {key: metadata[key][index] for key in keys}


def dataset_fingerprint(
    shards: Iterable[ShardInfo], *, config_hash: str, schema: str = SCHEMA_VERSION
) -> str:
    material = {
        "schema": schema,
        "config_hash": config_hash,
        "shards": [info.as_dict() for info in shards],
    }
    return sha256_json(material)
