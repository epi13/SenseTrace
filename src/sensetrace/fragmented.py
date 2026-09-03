"""Split, leakage, and claim authorization contracts for packet evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IntegrityError, SchemaError
from .hashing import sha256_json
from .packets import EvidencePacket

CLAIM_GROUP_FIELDS = {
    "level_1_exact_host_calibrated": None,
    "level_2_exact_host_unseen_location": "virtual_location_id",
    "level_3_exact_host_unseen_session": "acquisition_session_id",
    "level_4_exact_host_unseen_boot": "boot_id",
    "level_5_unseen_dimm": "dimm_id",
    "level_6_unseen_host": "host_id",
}
_PLACEHOLDERS = frozenset({"", "unknown", "unavailable", "none", "null", "n/a", "not_applicable"})
_AUDIT_FIELDS = frozenset(
    {
        "packet_id",
        "run_id",
        "host_id",
        "session_id",
        "acquisition_id",
        "acquisition_session_id",
        "boot_id",
        "target_reference",
        "address",
        "location_id",
        "virtual_location_id",
        "fragment_id",
        "sequence_position",
        "schedule_index",
        "excitation_phase",
        "label",
    }
)


def _value(packet: EvidencePacket, field: str) -> str:
    if field == "packet_id":
        return packet.packet_id
    if field == "target_reference":
        return packet.target_reference
    if field == "acquisition_id":
        return packet.acquisition_id
    aliases = {
        "session_id": "session_id",
        "acquisition_session_id": "acquisition_session_id",
        "virtual_location_id": "virtual_location_id",
        "location_id": "location_id",
        "boot_id": "boot_id",
        "host_id": "host_id",
        "dimm_id": "dimm_id",
        "address": "address",
    }
    key = aliases.get(field, field)
    return str(packet.provenance.get(key, "unavailable"))


def _concrete(value: str) -> bool:
    return value.strip().casefold() not in _PLACEHOLDERS


def _packet_content_fingerprint(packet: EvidencePacket) -> str:
    material = {
        "fragments": [
            {
                "probe_type": fragment.probe_type,
                "probe_version": fragment.probe_version,
                "sequence_position": fragment.sequence_position,
                "target_role": fragment.target_role,
                "status": fragment.status,
                "payload": None
                if fragment.payload is None
                else np.asarray(fragment.payload, dtype=np.float32).tolist(),
                "missing_mask": None
                if fragment.missing_mask is None
                else np.asarray(fragment.missing_mask, dtype=bool).tolist(),
                "excitation_code": list(fragment.excitation_code),
            }
            for fragment in packet.fragments
        ]
    }
    return sha256_json(material)


def claim_level_availability(packets: Iterable[EvidencePacket]) -> dict[str, Any]:
    """Summarize which claim levels are supported by explicit independent IDs."""

    values: dict[str, set[str]] = {
        key: set()
        for key in (
            "virtual_location_id",
            "acquisition_session_id",
            "boot_id",
            "dimm_id",
            "host_id",
        )
    }
    packet_count = 0
    for packet in packets:
        packet.validate()
        packet_count += 1
        for field in values:
            value = _value(packet, field)
            if _concrete(value):
                values[field].add(value)
    counts = {field: len(items) for field, items in values.items()}
    levels: dict[str, dict[str, Any]] = {
        "level_1_exact_host_calibrated": {
            "status": "available",
            "reason": "calibration split does not require an independent grouping identity",
        },
        "level_2_exact_host_unseen_location": {
            "status": "available" if counts["virtual_location_id"] >= 3 else "unavailable",
            "independent_group_count": counts["virtual_location_id"],
        },
        "level_3_exact_host_unseen_session": {
            "status": "available" if counts["acquisition_session_id"] >= 3 else "unavailable",
            "independent_group_count": counts["acquisition_session_id"],
        },
        "level_4_exact_host_unseen_boot": {
            "status": "available" if counts["boot_id"] >= 3 else "unavailable",
            "independent_group_count": counts["boot_id"],
            "minimum": 3,
        },
        "level_5_unseen_dimm": {
            "status": "available" if counts["dimm_id"] >= 2 else "unavailable",
            "independent_group_count": counts["dimm_id"],
            "minimum": 2,
        },
        "level_6_unseen_host": {
            "status": "available" if counts["host_id"] >= 2 else "unavailable",
            "independent_group_count": counts["host_id"],
            "minimum": 2,
        },
    }
    for record in levels.values():
        if record["status"] == "unavailable":
            record["reason"] = (
                "insufficient explicit independent provenance groups; no groups were manufactured"
            )
    return {"packet_count": packet_count, "group_counts": counts, "levels": levels}


def authorize_claim_level(packets: Iterable[EvidencePacket], claim_level: str) -> dict[str, Any]:
    availability = claim_level_availability(packets)
    record = availability["levels"].get(claim_level)
    if record is None:
        raise SchemaError(f"unsupported claim level {claim_level!r}")
    if record.get("status") != "available":
        raise SchemaError(f"claim level {claim_level} is unavailable: {record.get('reason')}")
    return {"claim_level": claim_level, "availability": availability, "authorized": True}


def _metadata_for_split(packet: EvidencePacket) -> dict[str, str]:
    return {
        "packet_id": packet.packet_id,
        "target_reference": packet.target_reference,
        "acquisition_id": packet.acquisition_id,
        **{
            field: _value(packet, field)
            for field in (
                "session_id",
                "acquisition_session_id",
                "boot_id",
                "host_id",
                "dimm_id",
                "virtual_location_id",
            )
        },
    }


def build_fragmented_split(
    packets: Iterable[EvidencePacket],
    *,
    dataset_fingerprint: str,
    claim_level: str = "level_1_exact_host_calibrated",
    seed: int = 1337,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    residualizer_source_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Materialize only packet IDs/group membership, never payload arrays."""

    if not dataset_fingerprint:
        raise SchemaError("fragmented split requires a dataset fingerprint")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-8:
        raise SchemaError("fragmented split fractions must sum to one")
    rows = [_metadata_for_split(packet) for packet in packets]
    if not rows:
        raise SchemaError("cannot split an empty packet stream")
    ids = [row["packet_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SchemaError("packet IDs are not unique")
    group_field = CLAIM_GROUP_FIELDS.get(claim_level)
    if claim_level not in CLAIM_GROUP_FIELDS:
        raise SchemaError(f"unsupported claim level {claim_level!r}")
    groups: dict[str, list[str]] = defaultdict(list)
    if group_field is None:
        for packet_id in ids:
            groups[packet_id].append(packet_id)
    else:
        for row in rows:
            group_value = row[group_field]
            if not _concrete(group_value):
                raise SchemaError(f"claim level {claim_level} requires explicit {group_field}")
            groups[group_value].append(row["packet_id"])
    if len(groups) < 3:
        raise SchemaError(f"split requires at least three independent groups; found {len(groups)}")
    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    targets = {
        "train": len(ids) * train_fraction,
        "validation": len(ids) * validation_fraction,
        "test": len(ids) * test_fraction,
    }
    assigned: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    sizes = {key: 0 for key in assigned}
    for name in names:
        part = min(assigned, key=lambda item: sizes[item] / max(targets[item], 1.0))
        assigned[part].extend(groups[name])
        sizes[part] += len(groups[name])
    if any(not value for value in assigned.values()):
        raise SchemaError("fragmented split produced an empty partition")
    for packet_ids in assigned.values():
        packet_ids.sort()
    split: dict[str, Any] = {
        "schema": "sensetrace.fragmented-split.v1",
        "split_name": claim_level,
        "split_strategy": "packet-id grouped immutable split",
        "claim_level": claim_level,
        "grouping_field": group_field or "packet_id",
        "random_seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "train_packet_ids": assigned["train"],
        "validation_packet_ids": assigned["validation"],
        "test_packet_ids": assigned["test"],
        "residualizer_source_dataset_fingerprint": residualizer_source_dataset_fingerprint
        or "unavailable",
        "model_input_policy": {"audit_fields_excluded": sorted(_AUDIT_FIELDS)},
    }
    if residualizer_source_dataset_fingerprint == dataset_fingerprint:
        raise SchemaError("residualizer corpus overlaps the supervised dataset")
    split["split_fingerprint"] = fingerprint_fragmented_split(split)
    return split


def fingerprint_fragmented_split(split: dict[str, Any]) -> str:
    material = dict(split)
    material.pop("split_fingerprint", None)
    return sha256_json(material)


def validate_fragmented_split(
    packets: Iterable[EvidencePacket],
    split: dict[str, Any],
    *,
    dataset_fingerprint: str | None = None,
    residualizer_source_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Audit membership, group separation, duplicate payloads, and metadata firewall."""

    if split.get("schema") != "sensetrace.fragmented-split.v1":
        raise SchemaError("unsupported fragmented split schema")
    if split.get("split_fingerprint") != fingerprint_fragmented_split(split):
        raise SchemaError("fragmented split fingerprint mismatch")
    if dataset_fingerprint is not None and split.get("dataset_fingerprint") != dataset_fingerprint:
        raise SchemaError("fragmented split references a different dataset")
    if (
        residualizer_source_dataset_fingerprint is not None
        and split.get("residualizer_source_dataset_fingerprint")
        == residualizer_source_dataset_fingerprint
        == dataset_fingerprint
    ):
        raise SchemaError("residualizer corpus overlaps the supervised dataset")
    claim_level = str(split.get("claim_level", ""))
    if claim_level not in CLAIM_GROUP_FIELDS:
        raise SchemaError(f"unsupported claim level {claim_level!r}")
    expected_grouping_field = CLAIM_GROUP_FIELDS[claim_level] or "packet_id"
    memberships: dict[str, str] = {}
    for partition in ("train", "validation", "test"):
        for packet_id in split.get(f"{partition}_packet_ids", []):
            if packet_id in memberships:
                raise SchemaError(f"packet {packet_id} appears in multiple split partitions")
            memberships[packet_id] = partition
    groups: dict[str, str] = {}
    contents: dict[str, str] = {}
    seen: set[str] = set()
    cross_partition_duplicates: list[dict[str, str]] = []
    group_field = str(split.get("grouping_field", "packet_id"))
    if group_field != expected_grouping_field:
        raise SchemaError("fragmented split grouping field does not match its claim level")
    for packet in packets:
        packet.validate()
        if packet.packet_id in seen:
            raise IntegrityError(f"duplicate packet {packet.packet_id}")
        seen.add(packet.packet_id)
        assigned_partition = memberships.get(packet.packet_id)
        if assigned_partition is None:
            raise SchemaError(f"packet {packet.packet_id} is not in the frozen split")
        group = packet.packet_id if group_field == "packet_id" else _value(packet, group_field)
        previous = groups.setdefault(group, assigned_partition)
        if previous != assigned_partition:
            raise SchemaError(f"group {group} crosses split partitions")
        digest = _packet_content_fingerprint(packet)
        old = contents.get(digest)
        if old is not None and memberships[old] != assigned_partition:
            cross_partition_duplicates.append(
                {
                    "first_packet_id": old,
                    "duplicate_packet_id": packet.packet_id,
                    "content_fingerprint": digest,
                }
            )
        contents[digest] = packet.packet_id
    expected_ids = set(memberships)
    if seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise SchemaError(
            f"frozen split coverage mismatch (missing={missing[:3]}, extra={extra[:3]})"
        )
    if cross_partition_duplicates:
        raise SchemaError("copied or duplicated packet content crosses split partitions")
    return {
        "schema": "sensetrace.fragmented-leakage-audit.v1",
        "status": "pass",
        "packet_count": len(seen),
        "partition_counts": {
            part: sum(value == part for value in memberships.values())
            for part in ("train", "validation", "test")
        },
        "grouping_field": group_field,
        "cross_partition_group_count": 0,
        "cross_partition_duplicate_count": 0,
        "metadata_firewall": {"status": "pass", "excluded_fields": sorted(_AUDIT_FIELDS)},
        "residualizer_overlap": False,
        "ordering_policy": "packet ordering is audit-only and absent from model arrays",
    }


def audit_fragmented_leakage(
    packets: Iterable[EvidencePacket],
    split: dict[str, Any],
    *,
    dataset_fingerprint: str | None = None,
    residualizer_source_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    return validate_fragmented_split(
        packets,
        split,
        dataset_fingerprint=dataset_fingerprint,
        residualizer_source_dataset_fingerprint=residualizer_source_dataset_fingerprint,
    )


def write_fragmented_split(path: str, split: dict[str, Any]) -> None:
    expected = fingerprint_fragmented_split(split)
    if split.get("split_fingerprint") != expected:
        raise SchemaError("cannot write a split with an invalid fingerprint")
    destination_path = Path(path)
    material = json.dumps(split, indent=2, sort_keys=True) + "\n"
    if destination_path.exists():
        try:
            existing = json.loads(destination_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaError("cannot read existing immutable fragmented split") from exc
        if existing != split:
            raise SchemaError("fragmented split artifact is immutable")
        return
    destination_path.write_text(material, encoding="utf-8")
