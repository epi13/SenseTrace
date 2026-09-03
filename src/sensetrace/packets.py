"""Streaming broken-packet evidence contracts.

An :class:`EvidencePacket` is the unit of a controlled experiment.  It may
contain many weak observations, and each observation is allowed to be absent,
partial, failed, or corrupted.  The packet representation deliberately keeps
model inputs separate from audit metadata and labels.

Packet shards are append-only JSONL files.  A writer holds only the current
line in memory and a loader yields one packet at a time, so corpus size does
not determine resident memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .errors import IntegrityError

FragmentStatus = Literal["observed", "unavailable", "failed", "partial", "corrupted"]
TargetRole = Literal["target", "reference", "paired_target", "paired_reference", "context"]
_FRAGMENT_STATUSES = frozenset({"observed", "unavailable", "failed", "partial", "corrupted"})
_TARGET_ROLES = frozenset({"target", "reference", "paired_target", "paired_reference", "context"})


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _identity(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty string without surrounding whitespace")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} contains control characters")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProbeFragment:
    """One weak observation in an evidence packet.

    ``missing_mask`` is explicit: an observed numeric zero is represented by a
    zero with a false mask, not by the same value used for an unavailable
    fragment.  ``audit_metadata`` is retained for review but is never exposed
    by :meth:`EvidencePacket.model_arrays`.
    """

    fragment_id: str
    probe_type: str
    probe_version: str
    sequence_position: int
    target_role: TargetRole
    payload: np.ndarray | None
    status: FragmentStatus = "observed"
    timing_ticks: int | None = None
    quality: float | None = None
    missing_mask: np.ndarray | None = None
    excitation_code: tuple[int, ...] = ()
    model_eligible: bool = True
    audit_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _identity(self.fragment_id, "fragment_id")
        _identity(self.probe_type, "probe_type")
        _identity(self.probe_version, "probe_version")
        if isinstance(self.sequence_position, bool) or self.sequence_position < 0:
            raise ValueError("fragment sequence_position must be non-negative")
        if self.target_role not in _TARGET_ROLES:
            raise ValueError(f"unsupported fragment target_role {self.target_role!r}")
        if self.status not in _FRAGMENT_STATUSES:
            raise ValueError(f"unsupported fragment status {self.status!r}")
        if self.payload is None:
            if self.status in {"observed", "partial"}:
                raise ValueError("observed or partial fragments require a payload")
        else:
            payload = np.asarray(self.payload)
            if payload.ndim != 1 or payload.size == 0:
                raise ValueError("fragment payload must be a non-empty one-dimensional array")
            if not np.issubdtype(payload.dtype, np.number):
                raise ValueError("fragment payload must be numeric")
            if not np.isfinite(payload.astype(np.float32, copy=False)).all():
                raise ValueError("fragment payload must be finite")
        if self.missing_mask is not None:
            mask = np.asarray(self.missing_mask)
            if mask.ndim != 1 or self.payload is None or len(mask) != len(self.payload):
                raise ValueError("fragment missing_mask must match the one-dimensional payload")
            if mask.dtype.kind not in "biu":
                raise ValueError("fragment missing_mask must be boolean-like")
        if self.status in {"unavailable", "failed", "corrupted"} and self.model_eligible:
            raise ValueError("unavailable, failed, or corrupted fragments cannot be model-eligible")
        if self.timing_ticks is not None and (
            isinstance(self.timing_ticks, bool) or self.timing_ticks < 0
        ):
            raise ValueError("fragment timing_ticks must be a non-negative integer")
        if (
            self.quality is not None
            and not 0.0 <= _finite_float(self.quality, "fragment quality") <= 1.0
        ):
            raise ValueError("fragment quality must be between zero and one")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.excitation_code
        ):
            raise ValueError("fragment excitation_code must contain integers")
        try:
            json.dumps(self.audit_metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("fragment audit_metadata must be JSON-compatible") from exc

    @property
    def payload_length(self) -> int:
        return 0 if self.payload is None else int(np.asarray(self.payload).size)

    def effective_mask(self) -> np.ndarray:
        self.validate()
        if self.payload is None:
            return np.zeros(0, dtype=bool)
        if self.missing_mask is None:
            return np.zeros(self.payload_length, dtype=bool)
        return np.asarray(self.missing_mask, dtype=bool).copy()

    def as_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "fragment_id": self.fragment_id,
            "probe_type": self.probe_type,
            "probe_version": self.probe_version,
            "sequence_position": self.sequence_position,
            "target_role": self.target_role,
            "payload": None
            if self.payload is None
            else np.asarray(self.payload, dtype=np.float32).tolist(),
            "status": self.status,
            "timing_ticks": self.timing_ticks,
            "quality": self.quality,
            "missing_mask": None
            if self.missing_mask is None
            else np.asarray(self.missing_mask, dtype=bool).tolist(),
            "excitation_code": list(self.excitation_code),
            "model_eligible": self.model_eligible,
            "audit_metadata": self.audit_metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ProbeFragment:
        if not isinstance(record, dict):
            raise ValueError("fragment record must be an object")
        payload = record.get("payload")
        mask = record.get("missing_mask")
        fragment = cls(
            fragment_id=record["fragment_id"],
            probe_type=record["probe_type"],
            probe_version=record["probe_version"],
            sequence_position=record["sequence_position"],
            target_role=record["target_role"],
            payload=None if payload is None else np.asarray(payload, dtype=np.float32),
            status=record.get("status", "observed"),
            timing_ticks=record.get("timing_ticks"),
            quality=record.get("quality"),
            missing_mask=None if mask is None else np.asarray(mask, dtype=bool),
            excitation_code=tuple(record.get("excitation_code", ())),
            model_eligible=bool(record.get("model_eligible", True)),
            audit_metadata=dict(record.get("audit_metadata", {})),
        )
        fragment.validate()
        return fragment


@dataclass(frozen=True)
class EvidencePacket:
    """Many weak, possibly damaged observations for one controlled target."""

    packet_id: str
    target_reference: str
    acquisition_id: str
    protocol_id: str
    fragments: tuple[ProbeFragment, ...]
    controls: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    label: int | None = None

    def validate(self) -> None:
        for name, value in {
            "packet_id": self.packet_id,
            "target_reference": self.target_reference,
            "acquisition_id": self.acquisition_id,
            "protocol_id": self.protocol_id,
        }.items():
            _identity(value, name)
        if not self.fragments:
            raise ValueError("evidence packet requires at least one fragment")
        positions: set[int] = set()
        for fragment in self.fragments:
            fragment.validate()
            if fragment.sequence_position in positions:
                raise ValueError("evidence packet fragment sequence positions must be unique")
            positions.add(fragment.sequence_position)
        if self.label is not None and self.label not in (0, 1):
            raise ValueError("evidence packet label must be zero, one, or None")
        json_objects: tuple[tuple[str, object], ...] = (
            ("controls", self.controls),
            ("provenance", self.provenance),
        )
        for json_name, json_value in json_objects:
            try:
                json.dumps(json_value, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"packet {json_name} must be JSON-compatible") from exc

    def as_record(self, *, include_label: bool = True) -> dict[str, Any]:
        self.validate()
        record: dict[str, Any] = {
            "schema": "sensetrace.evidence-packet.v1",
            "packet_id": self.packet_id,
            "target_reference": self.target_reference,
            "acquisition_id": self.acquisition_id,
            "protocol_id": self.protocol_id,
            "fragments": [fragment.as_record() for fragment in self.fragments],
            "controls": self.controls,
            "provenance": self.provenance,
        }
        if include_label and self.label is not None:
            record["label"] = self.label
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EvidencePacket:
        if record.get("schema") != "sensetrace.evidence-packet.v1":
            raise ValueError("unsupported evidence packet schema")
        packet = cls(
            packet_id=record["packet_id"],
            target_reference=record["target_reference"],
            acquisition_id=record["acquisition_id"],
            protocol_id=record["protocol_id"],
            fragments=tuple(ProbeFragment.from_record(item) for item in record["fragments"]),
            controls=dict(record.get("controls", {})),
            provenance=dict(record.get("provenance", {})),
            label=record.get("label"),
        )
        packet.validate()
        return packet

    def model_arrays(
        self, *, max_fragments: int, max_payload_length: int, excitation_width: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return only model-eligible arrays with explicit masks.

        The returned tuple is ``values, observed_mask, fragment_mask,
        excitation, quality``.  Padding values are zero, but every padded or missing
        value has a false mask and therefore cannot be confused with an
        observed zero.
        """

        self.validate()
        if max_fragments < 1 or max_payload_length < 1 or excitation_width < 0:
            raise ValueError("batch dimensions must be positive")
        values = np.zeros((max_fragments, max_payload_length), dtype=np.float32)
        observed = np.zeros_like(values, dtype=bool)
        fragment_mask = np.zeros(max_fragments, dtype=bool)
        excitation = np.zeros((max_fragments, excitation_width), dtype=np.float32)
        quality = np.zeros(max_fragments, dtype=np.float32)
        eligible = [fragment for fragment in self.fragments if fragment.model_eligible]
        if len(eligible) > max_fragments:
            raise ValueError("packet has more model-eligible fragments than max_fragments")
        for row, fragment in enumerate(eligible):
            if fragment.payload is None:
                continue
            payload = np.asarray(fragment.payload, dtype=np.float32)
            if payload.size > max_payload_length:
                raise ValueError("fragment payload exceeds max_payload_length")
            length = min(payload.size, max_payload_length)
            values[row, :length] = payload[:length]
            missing = fragment.effective_mask()[:length]
            observed[row, :length] = ~missing
            fragment_mask[row] = bool(np.any(observed[row]))
            quality[row] = 1.0 if fragment.quality is None else float(fragment.quality)
            if excitation_width:
                code = np.asarray(fragment.excitation_code[:excitation_width], dtype=np.float32)
                excitation[row, : code.size] = code
        return values, observed, fragment_mask, excitation, quality


@dataclass(frozen=True)
class PacketBatch:
    """A bounded in-memory batch emitted by a streaming packet loader."""

    values: np.ndarray
    observed_mask: np.ndarray
    fragment_mask: np.ndarray
    excitation: np.ndarray
    quality: np.ndarray
    labels: np.ndarray | None
    packet_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.values.ndim != 3 or self.values.dtype != np.float32:
            raise ValueError("packet batch values must be float32 [batch, fragments, payload]")
        expected = self.values.shape
        if self.observed_mask.shape != expected or self.observed_mask.dtype != bool:
            raise ValueError("packet batch observed_mask shape/dtype mismatch")
        if self.fragment_mask.shape != expected[:2] or self.fragment_mask.dtype != bool:
            raise ValueError("packet batch fragment_mask shape/dtype mismatch")
        if (
            self.excitation.ndim != 3
            or self.excitation.shape[:2] != expected[:2]
            or self.excitation.dtype != np.float32
        ):
            raise ValueError("packet batch excitation shape mismatch")
        if self.quality.shape != expected[:2] or self.quality.dtype != np.float32:
            raise ValueError("packet batch quality shape/dtype mismatch")
        if self.labels is not None:
            if (
                self.labels.ndim != 1
                or len(self.labels) != expected[0]
                or not np.isin(self.labels, [0, 1]).all()
            ):
                raise ValueError("packet batch labels shape or values are invalid")
        if len(self.packet_ids) != expected[0]:
            raise ValueError("packet batch packet_ids length mismatch")

    @classmethod
    def from_packets(
        cls,
        packets: list[EvidencePacket],
        *,
        max_fragments: int,
        max_payload_length: int,
        excitation_width: int = 0,
        include_labels: bool = True,
    ) -> PacketBatch:
        if not packets:
            raise ValueError("cannot create a batch from no packets")
        arrays = [
            packet.model_arrays(
                max_fragments=max_fragments,
                max_payload_length=max_payload_length,
                excitation_width=excitation_width,
            )
            for packet in packets
        ]
        batch = cls(
            values=np.stack([item[0] for item in arrays]).astype(np.float32, copy=False),
            observed_mask=np.stack([item[1] for item in arrays]),
            fragment_mask=np.stack([item[2] for item in arrays]),
            excitation=np.stack([item[3] for item in arrays]).astype(np.float32, copy=False),
            quality=np.stack([item[4] for item in arrays]).astype(np.float32, copy=False),
            labels=(
                np.asarray([packet.label for packet in packets], dtype=np.uint8)
                if include_labels and all(packet.label is not None for packet in packets)
                else None
            ),
            packet_ids=tuple(packet.packet_id for packet in packets),
        )
        batch.validate()
        return batch


@dataclass(frozen=True)
class PacketShardInfo:
    path: Path
    packets: int
    first_packet_id: str
    last_packet_id: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "packets": self.packets,
            "first_packet_id": self.first_packet_id,
            "last_packet_id": self.last_packet_id,
            "sha256": self.sha256,
        }


def packet_dataset_fingerprint(infos: Iterable[PacketShardInfo], *, config_hash: str) -> str:
    material = {
        "schema": "sensetrace.evidence-packet-dataset.v1",
        "config_hash": config_hash,
        "shards": [info.as_dict() for info in infos],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def write_packet_manifest(
    root: str | Path,
    *,
    config_hash: str,
    infos: list[PacketShardInfo],
    protocol_id: str,
    purpose: str = "fragmented-evidence",
    additional_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a small immutable packet-dataset manifest from shard summaries."""

    if not config_hash or not protocol_id:
        raise ValueError("packet manifests require config_hash and protocol_id")
    manifest = {
        "schema": "sensetrace.evidence-packet-dataset.v1",
        "dataset_purpose": purpose,
        "protocol_id": protocol_id,
        "config_hash": config_hash,
        "packet_count": sum(info.packets for info in infos),
        "dataset_fingerprint": packet_dataset_fingerprint(infos, config_hash=config_hash),
        "shards": [info.as_dict() for info in infos],
        "claim_boundary": "fragmented evidence contract; no physical DRAM claim by itself",
        "model_input_policy": "stream bounded batches; identifiers, provenance, and labels stay separate",
    }
    if additional_fields:
        manifest.update(additional_fields)
    Path(root).mkdir(parents=True, exist_ok=True)
    path = Path(root) / "packet-dataset.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("cannot read existing packet dataset manifest") from exc
        if existing != manifest:
            raise IntegrityError("packet dataset manifest is immutable")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    return manifest


class PacketShardWriter:
    """Append packets one at a time and atomically finalize bounded shards."""

    def __init__(self, root: str | Path, *, max_packets_per_shard: int = 256) -> None:
        if max_packets_per_shard < 1:
            raise ValueError("max_packets_per_shard must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_packets_per_shard = max_packets_per_shard
        self._next_index = self._discover_next_index()
        self._handle: Any | None = None
        self._temporary_path: Path | None = None
        self._count = 0
        self._first_packet_id: str | None = None
        self._last_packet_id: str | None = None

    def _discover_next_index(self) -> int:
        indexes: list[int] = []
        for path in self.root.glob("packet-shard-*.jsonl"):
            try:
                indexes.append(int(path.stem.split("-")[-1]))
            except ValueError:
                continue
        return max(indexes, default=-1) + 1

    def _open(self) -> None:
        if self._handle is not None:
            return
        stem = f"packet-shard-{self._next_index:06d}"
        self._temporary_path = self.root / f"{stem}.jsonl.tmp"
        self._handle = self._temporary_path.open("w", encoding="utf-8")

    @property
    def buffered_packets(self) -> int:
        return self._count

    def add(self, packet: EvidencePacket) -> PacketShardInfo | None:
        packet.validate()
        self._open()
        assert self._handle is not None
        self._handle.write(json.dumps(packet.as_record(), allow_nan=False, sort_keys=True) + "\n")
        self._handle.flush()
        self._count += 1
        self._first_packet_id = self._first_packet_id or packet.packet_id
        self._last_packet_id = packet.packet_id
        if self._count >= self.max_packets_per_shard:
            return self.finalize()
        return None

    def finalize(self) -> PacketShardInfo | None:
        if self._count == 0:
            return None
        assert self._handle is not None and self._temporary_path is not None
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        checksum = _sha256_file(self._temporary_path)
        final_path = self.root / self._temporary_path.name.removesuffix(".tmp")
        sidecar_path = final_path.with_suffix(".json")
        sidecar = {
            "schema": "sensetrace.evidence-packet-shard.v1",
            "created_at": time.time(),
            "packets": self._count,
            "first_packet_id": self._first_packet_id,
            "last_packet_id": self._last_packet_id,
            "sha256": checksum,
        }
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(self._temporary_path, final_path)
        self._next_index += 1
        info = PacketShardInfo(
            final_path,
            self._count,
            str(self._first_packet_id),
            str(self._last_packet_id),
            checksum,
        )
        self._handle = None
        self._temporary_path = None
        self._count = 0
        self._first_packet_id = None
        self._last_packet_id = None
        return info


def list_packet_shards(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("packet-shard-*.jsonl"))


def validate_packet_shard(path: str | Path) -> PacketShardInfo:
    shard = Path(path)
    sidecar_path = shard.with_suffix(".json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        checksum = _sha256_file(shard)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read packet shard {shard}") from exc
    if sidecar.get("schema") != "sensetrace.evidence-packet-shard.v1":
        raise IntegrityError(f"unsupported packet shard schema in {shard.name}")
    if checksum != sidecar.get("sha256"):
        raise IntegrityError(f"packet shard checksum mismatch for {shard.name}")
    count = 0
    first: str | None = None
    last: str | None = None
    try:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("blank packet line")
                packet = EvidencePacket.from_record(json.loads(line))
                count += 1
                first = first or packet.packet_id
                last = packet.packet_id
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"invalid packet shard {shard.name}: {exc}") from exc
    if (
        count != sidecar.get("packets")
        or first != sidecar.get("first_packet_id")
        or last != sidecar.get("last_packet_id")
    ):
        raise IntegrityError(f"packet shard sidecar disagrees with {shard.name}")
    return PacketShardInfo(shard, count, str(first), str(last), checksum)


def iter_packets(root: str | Path, *, validate: bool = True) -> Iterator[EvidencePacket]:
    """Yield packets shard-by-shard without materializing a corpus."""

    for path in list_packet_shards(root):
        if validate:
            validate_packet_shard(path)
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        yield EvidencePacket.from_record(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise IntegrityError(f"invalid packet in {path.name}: {exc}") from exc
        except OSError as exc:
            raise IntegrityError(f"cannot stream packet shard {path.name}") from exc


def iter_packet_batches(
    packets: Iterable[EvidencePacket],
    *,
    batch_size: int,
    max_fragments: int,
    max_payload_length: int,
    excitation_width: int = 0,
    include_labels: bool = True,
) -> Iterator[PacketBatch]:
    """Convert a packet stream into bounded batches and release each batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending: list[EvidencePacket] = []
    for packet in packets:
        pending.append(packet)
        if len(pending) == batch_size:
            yield PacketBatch.from_packets(
                pending,
                max_fragments=max_fragments,
                max_payload_length=max_payload_length,
                excitation_width=excitation_width,
                include_labels=include_labels,
            )
            pending = []
    if pending:
        yield PacketBatch.from_packets(
            pending,
            max_fragments=max_fragments,
            max_payload_length=max_payload_length,
            excitation_width=excitation_width,
            include_labels=include_labels,
        )


def validate_packet_stream(root: str | Path) -> list[PacketShardInfo]:
    """Validate all packet sidecars while retaining only tiny shard summaries."""

    infos = [validate_packet_shard(path) for path in list_packet_shards(root)]
    previous: str | None = None
    for info in infos:
        if previous is not None and info.first_packet_id <= previous:
            raise IntegrityError("packet shards have overlapping or unsorted packet ranges")
        previous = info.last_packet_id
    return infos
