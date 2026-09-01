"""Materialized grouped train/validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import ensure_sample_ids, fingerprint_split
from .errors import SchemaError


def _group_key(metadata: dict[str, np.ndarray], index: int, keys: list[str]) -> str:
    values = []
    for key in keys:
        if key not in metadata:
            raise SchemaError(f"grouping field not present: {key}")
        values.append(str(metadata[key][index]))
    return "|".join(values)


def grouped_split(
    metadata: dict[str, np.ndarray],
    *,
    dataset_fingerprint: str,
    group_keys: list[str],
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    sample_ids = ensure_sample_ids(metadata)
    count = len(sample_ids)
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-8:
        raise SchemaError("split fractions must sum to one")
    groups: dict[str, list[int]] = {}
    for index in range(count):
        groups.setdefault(_group_key(metadata, index, group_keys), []).append(index)
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    targets = {
        "train": count * train_fraction,
        "validation": count * validation_fraction,
        "test": count * test_fraction,
    }
    assigned: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    sizes = {name: 0 for name in assigned}
    # Greedily fill the partition furthest below its target, while keeping groups intact.
    for name in group_names:
        destination = min(assigned, key=lambda part: sizes[part] / max(targets[part], 1.0))
        assigned[destination].extend(groups[name])
        sizes[destination] += len(groups[name])
    for values in assigned.values():
        values.sort()
    if any(not assigned[name] for name in assigned):
        raise SchemaError("grouped split produced an empty partition")
    split: dict[str, Any] = {
        "schema": "sensetrace.split.v1",
        "split_name": "primary-grouped",
        "split_strategy": "grouped",
        "random_seed": seed,
        "grouping_keys": group_keys,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "train_sample_ids": [sample_ids[index] for index in assigned["train"]],
        "validation_sample_ids": [sample_ids[index] for index in assigned["validation"]],
        "test_sample_ids": [sample_ids[index] for index in assigned["test"]],
    }
    split["split_fingerprint"] = fingerprint_split(split)
    return split


def write_split(path: str | Path, split: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_split(
    path: str | Path, *, expected_dataset_fingerprint: str | None = None
) -> dict[str, Any]:
    try:
        split = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read split {path}") from exc
    expected = fingerprint_split(split)
    if split.get("split_fingerprint") != expected:
        raise SchemaError("split fingerprint mismatch")
    if (
        expected_dataset_fingerprint
        and split.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise SchemaError("split references a different dataset fingerprint")
    required = {"train_sample_ids", "validation_sample_ids", "test_sample_ids"}
    if not required.issubset(split):
        raise SchemaError("split is missing a materialized partition")
    return split


def partition_indices(
    metadata: dict[str, np.ndarray], split: dict[str, Any]
) -> dict[str, np.ndarray]:
    ids = ensure_sample_ids(metadata)
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    partitions = {}
    for name in ["train", "validation", "test"]:
        try:
            partitions[name] = np.asarray(
                [lookup[sample_id] for sample_id in split[f"{name}_sample_ids"]], dtype=np.int64
            )
        except KeyError as exc:
            raise SchemaError(f"split references unknown sample id {exc.args[0]}") from exc
    joined = np.concatenate(list(partitions.values()))
    if len(np.unique(joined)) != len(ids) or len(joined) != len(ids):
        raise SchemaError("split partitions do not cover each sample exactly once")
    return partitions
