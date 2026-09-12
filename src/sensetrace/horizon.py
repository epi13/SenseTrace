"""Causal present-to-future forecasting experiments.

This module is deliberately independent of the physical-memory acquisition
backends.  It provides the analysis contract needed to ask whether a state at
an origin contains information about a later state, while keeping the
forecast passive: the forecast is computed from a copy of the origin state
and is never passed back into the trajectory that produced the target.

The first implementation supports binary and scalar continuous future-state
targets.  The trajectory/pair representation is intentionally more general
than either target so categorical future events and compressed vector targets
can be added without changing horizon alignment or split semantics.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .errors import IntegrityError, SchemaError
from .hashing import sha256_bytes, sha256_json

TargetKind = Literal["binary", "continuous"]
AlignmentMode = Literal["index", "position"]
_IDENTITY_METADATA_FIELDS = frozenset(
    {
        "trajectory_id",
        "pair_id",
        "run_id",
        "session_id",
        "acquisition_session_id",
        "boot_id",
        "host_id",
        "device_id",
        "dimm_id",
        "address",
        "location_id",
        "virtual_location_id",
        "origin_index",
        "target_index",
        "target_position",
    }
)


def _jsonable_scalar(value: object, field: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise SchemaError(f"trajectory metadata {field} must be finite")
        return value
    raise SchemaError(f"trajectory metadata {field} must be scalar")


def _array_digest(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array)
    return sha256_bytes(normalized.tobytes())


@dataclass(frozen=True)
class Horizon:
    """A future distance and its alignment semantics.

    ``index`` treats distance as a count of trajectory states.  ``position``
    treats it as a delta in the trajectory's monotone position coordinate and
    selects the first observed state at or after the requested position.  The
    unit is descriptive (for example ``layer``, ``token``, or ``wall_clock``)
    and is retained in the manifest rather than being interpreted implicitly.
    """

    distance: int | float
    unit: str = "step"
    alignment: AlignmentMode = "index"

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, (int, float)):
            raise SchemaError("horizon distance must be numeric")
        if not np.isfinite(float(self.distance)) or float(self.distance) <= 0:
            raise SchemaError("horizon distance must be positive and finite")
        if self.alignment == "index" and (
            not isinstance(self.distance, int) or isinstance(self.distance, bool)
        ):
            raise SchemaError("index-aligned horizon distance must be an integer")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise SchemaError("horizon unit must be a non-empty string")
        if self.alignment not in {"index", "position"}:
            raise SchemaError(f"unsupported horizon alignment {self.alignment!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "unit": self.unit,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class TargetSpec:
    """Define a target extracted from one component of a future state."""

    name: str
    kind: TargetKind
    state_index: int = 0
    threshold: float = 0.0
    positive_if: Literal["ge", "gt", "le", "lt"] = "ge"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SchemaError("target name must be non-empty")
        if self.kind not in {"binary", "continuous"}:
            raise SchemaError(f"unsupported target kind {self.kind!r}")
        if isinstance(self.state_index, bool) or not isinstance(self.state_index, int) or self.state_index < 0:
            raise SchemaError("target state_index must be a non-negative integer")
        if not np.isfinite(self.threshold):
            raise SchemaError("target threshold must be finite")
        if self.positive_if not in {"ge", "gt", "le", "lt"}:
            raise SchemaError(f"unsupported positive_if comparison {self.positive_if!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "state_index": self.state_index,
            "threshold": self.threshold,
            "positive_if": self.positive_if,
        }

    def extract(self, future_states: np.ndarray) -> np.ndarray:
        states = np.asarray(future_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] <= self.state_index:
            raise SchemaError("future states do not contain the target state component")
        values = states[:, self.state_index]
        if self.kind == "continuous":
            return values.astype(np.float64, copy=True)
        if self.positive_if == "ge":
            result = values >= self.threshold
        elif self.positive_if == "gt":
            result = values > self.threshold
        elif self.positive_if == "le":
            result = values <= self.threshold
        else:
            result = values < self.threshold
        return result.astype(np.uint8)


@dataclass(frozen=True)
class StateTrajectory:
    """One ordered computational trajectory.

    ``states[i]`` is the complete feature-visible state at ``positions[i]``.
    A trajectory's metadata is audit context only and is never implicitly
    included in model features.
    """

    trajectory_id: str
    states: np.ndarray
    positions: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.trajectory_id or self.trajectory_id != self.trajectory_id.strip():
            raise SchemaError("trajectory_id must be a non-empty string")
        states = np.asarray(self.states, dtype=np.float32)
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
            raise SchemaError("trajectory states must have shape [time >= 2, state_dim >= 1]")
        if not np.isfinite(states).all():
            raise SchemaError("trajectory states must be finite")
        positions = (
            np.arange(states.shape[0], dtype=np.float64)
            if self.positions is None
            else np.asarray(self.positions, dtype=np.float64)
        )
        if positions.ndim != 1 or len(positions) != len(states):
            raise SchemaError("trajectory positions must match the state time dimension")
        if not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
            raise SchemaError("trajectory positions must be finite and strictly increasing")
        metadata = dict(self.metadata or {})
        for field, value in metadata.items():
            _jsonable_scalar(value, field)
        states.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "metadata", metadata)

    def fingerprint(self) -> str:
        metadata = {
            key: _jsonable_scalar(value, key)
            for key, value in sorted((self.metadata or {}).items())
        }
        return sha256_json(
            {
                "trajectory_id": self.trajectory_id,
                "states_shape": list(self.states.shape),
                "states_sha256": _array_digest(self.states),
                "positions_sha256": _array_digest(np.asarray(self.positions)),
                "metadata": metadata,
            }
        )


@dataclass(frozen=True)
class ForecastPairs:
    """Causally aligned present-state/future-target rows for one horizon."""

    horizon: Horizon
    target: TargetSpec
    features: np.ndarray
    targets: np.ndarray
    pair_ids: np.ndarray
    metadata: Mapping[str, np.ndarray]
    source_fingerprint: str
    alignment_rule: str

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        targets = np.asarray(self.targets)
        pair_ids = np.asarray(self.pair_ids).astype(str)
        if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
            raise SchemaError("forecast features must be a non-empty finite matrix")
        if targets.ndim != 1 or len(targets) != len(features):
            raise SchemaError("forecast targets must be a vector aligned to features")
        if self.target.kind == "continuous":
            targets = targets.astype(np.float64)
            if not np.isfinite(targets).all():
                raise SchemaError("continuous forecast targets must be finite")
        else:
            if not set(np.unique(targets).tolist()).issubset({0, 1}):
                raise SchemaError("binary forecast targets must contain only 0 and 1")
            targets = targets.astype(np.uint8)
        if pair_ids.ndim != 1 or len(pair_ids) != len(features) or len(set(pair_ids)) != len(pair_ids):
            raise SchemaError("forecast pair IDs must be unique and aligned to features")
        metadata = {str(key): np.asarray(value) for key, value in self.metadata.items()}
        for key, value in metadata.items():
            if value.ndim != 1 or len(value) != len(features):
                raise SchemaError(f"forecast metadata field {key} is not row-aligned")
        required = {"trajectory_id", "origin_index", "target_index", "origin_position", "target_position"}
        if not required.issubset(metadata):
            raise SchemaError(f"forecast metadata is missing {sorted(required - set(metadata))}")
        if np.any(np.asarray(metadata["target_index"]) <= np.asarray(metadata["origin_index"])):
            raise SchemaError("forecast target index must be later than its origin index")
        features.setflags(write=False)
        targets.setflags(write=False)
        pair_ids.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(len(self.features))

    def fingerprint(self) -> str:
        metadata = {
            key: np.asarray(value).astype(str).tolist()
            for key, value in sorted(self.metadata.items())
        }
        return sha256_json(
            {
                "horizon": self.horizon.as_dict(),
                "target": self.target.as_dict(),
                "source_fingerprint": self.source_fingerprint,
                "features_sha256": _array_digest(self.features),
                "targets_sha256": _array_digest(np.asarray(self.targets)),
                "pair_ids": self.pair_ids.tolist(),
                "metadata": metadata,
                "alignment_rule": self.alignment_rule,
            }
        )


def trajectory_collection_fingerprint(trajectories: Sequence[StateTrajectory]) -> str:
    if not trajectories:
        raise SchemaError("at least one trajectory is required")
    ids = [trajectory.trajectory_id for trajectory in trajectories]
    if len(ids) != len(set(ids)):
        raise SchemaError("trajectory IDs must be unique")
    return sha256_json({"trajectories": [trajectory.fingerprint() for trajectory in trajectories]})


def _target_index(trajectory: StateTrajectory, origin_index: int, horizon: Horizon) -> int | None:
    if horizon.alignment == "index":
        target = origin_index + int(horizon.distance)
        return target if target < len(trajectory.states) else None
    assert trajectory.positions is not None
    desired = float(trajectory.positions[origin_index]) + float(horizon.distance)
    target = int(np.searchsorted(trajectory.positions, desired, side="left"))
    return target if target < len(trajectory.states) else None


def build_forecast_pairs(
    trajectories: Iterable[StateTrajectory],
    horizon: Horizon,
    target: TargetSpec,
) -> ForecastPairs:
    """Align current states with later targets without exposing later states.

    The only feature assignment in this function is ``features <- states[t]``;
    future states are passed solely to ``TargetSpec.extract``.  This is the
    central causal invariant for the passive forecasting phase.
    """

    materialized = list(trajectories)
    source_fingerprint = trajectory_collection_fingerprint(materialized)
    feature_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    pair_ids: list[str] = []
    metadata: dict[str, list[Any]] = {
        "trajectory_id": [],
        "origin_index": [],
        "target_index": [],
        "origin_position": [],
        "target_position": [],
    }
    for trajectory in materialized:
        for origin_index in range(len(trajectory.states)):
            target_index = _target_index(trajectory, origin_index, horizon)
            if target_index is None or target_index <= origin_index:
                continue
            feature_rows.append(np.asarray(trajectory.states[origin_index], dtype=np.float32).copy())
            future_rows.append(np.asarray(trajectory.states[target_index], dtype=np.float32).copy())
            pair_ids.append(
                sha256_json(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "origin_index": origin_index,
                        "target_index": target_index,
                        "horizon": horizon.as_dict(),
                    }
                )
            )
            metadata["trajectory_id"].append(trajectory.trajectory_id)
            metadata["origin_index"].append(origin_index)
            metadata["target_index"].append(target_index)
            assert trajectory.positions is not None
            metadata["origin_position"].append(float(trajectory.positions[origin_index]))
            metadata["target_position"].append(float(trajectory.positions[target_index]))
    if not feature_rows:
        raise SchemaError(f"horizon {horizon.as_dict()} has no valid present/future pairs")
    future = np.stack(future_rows).astype(np.float32)
    return ForecastPairs(
        horizon=horizon,
        target=target,
        features=np.stack(feature_rows),
        targets=target.extract(future),
        pair_ids=np.asarray(pair_ids),
        metadata={key: np.asarray(value) for key, value in metadata.items()},
        source_fingerprint=source_fingerprint,
        alignment_rule=(
            "target=t+distance by state index"
            if horizon.alignment == "index"
            else "target=first observed position at or after origin_position+distance"
        ),
    )


def build_horizon_split(
    pairs: ForecastPairs,
    *,
    seed: int,
    group_key: str = "trajectory_id",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    """Create an immutable whole-trajectory split for one horizon."""

    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise SchemaError("horizon split fractions must be positive and sum to one")
    if group_key not in pairs.metadata:
        raise SchemaError(f"horizon split group key {group_key!r} is missing")
    group_values = np.asarray(pairs.metadata[group_key]).astype(str)
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(group_values):
        groups.setdefault(group, []).append(index)
    if len(groups) < 3:
        raise SchemaError("horizon split requires at least three independent trajectory groups")
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    targets = {
        "train": pairs.row_count * train_fraction,
        "validation": pairs.row_count * validation_fraction,
        "test": pairs.row_count * test_fraction,
    }
    assigned: dict[str, list[int]] = {name: [] for name in targets}
    sizes = {name: 0 for name in targets}
    for group in group_names:
        destination = min(assigned, key=lambda name: sizes[name] / max(targets[name], 1.0))
        assigned[destination].extend(groups[group])
        sizes[destination] += len(groups[group])
    if any(not assigned[name] for name in assigned):
        raise SchemaError("horizon split produced an empty partition")
    for indices in assigned.values():
        indices.sort()
    split: dict[str, Any] = {
        "schema": "sensetrace.horizon-split.v1",
        "split_strategy": "whole-trajectory grouped holdout",
        "group_key": group_key,
        "random_seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "pair_dataset_fingerprint": pairs.fingerprint(),
        "source_trajectory_fingerprint": pairs.source_fingerprint,
        "train_pair_ids": pairs.pair_ids[assigned["train"]].tolist(),
        "validation_pair_ids": pairs.pair_ids[assigned["validation"]].tolist(),
        "test_pair_ids": pairs.pair_ids[assigned["test"]].tolist(),
    }
    split["split_fingerprint"] = fingerprint_horizon_split(split)
    return split


def fingerprint_horizon_split(split: Mapping[str, Any]) -> str:
    material = dict(split)
    material.pop("split_fingerprint", None)
    return sha256_json(material)


def horizon_partition_indices(pairs: ForecastPairs, split: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if split.get("schema") != "sensetrace.horizon-split.v1":
        raise SchemaError("unsupported horizon split schema")
    if split.get("split_fingerprint") != fingerprint_horizon_split(split):
        raise SchemaError("horizon split fingerprint mismatch")
    if split.get("pair_dataset_fingerprint") != pairs.fingerprint():
        raise SchemaError("horizon split references a different pair dataset")
    lookup = {pair_id: index for index, pair_id in enumerate(pairs.pair_ids.tolist())}
    result: dict[str, np.ndarray] = {}
    all_indices: list[int] = []
    for partition in ("train", "validation", "test"):
        try:
            values = [lookup[pair_id] for pair_id in split[f"{partition}_pair_ids"]]
        except KeyError as exc:
            raise SchemaError(f"horizon split references unknown pair ID {exc.args[0]}") from exc
        result[partition] = np.asarray(values, dtype=np.int64)
        all_indices.extend(values)
    if len(all_indices) != pairs.row_count or len(set(all_indices)) != pairs.row_count:
        raise SchemaError("horizon split partitions do not cover every pair exactly once")
    return result


def validate_horizon_alignment(pairs: ForecastPairs, split: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit causal ordering, duplicate rows, and whole-trajectory separation."""

    origins = np.asarray(pairs.metadata["origin_index"])
    targets = np.asarray(pairs.metadata["target_index"])
    causal = bool(np.all(targets > origins))
    result: dict[str, Any] = {
        "schema": "sensetrace.horizon-causal-audit.v1",
        "status": "pass" if causal else "fail",
        "pair_count": pairs.row_count,
        "all_targets_strictly_after_origins": causal,
        "feature_source": "trajectory.states[origin_index] only",
        "target_source": "trajectory.states[target_index] only",
        "feedback": "disabled; forecast is not supplied to trajectory generation or target acquisition",
        "duplicate_pair_id_count": int(pairs.row_count - len(set(pairs.pair_ids.tolist()))),
    }
    if split is not None:
        partitions = horizon_partition_indices(pairs, split)
        groups = np.asarray(pairs.metadata["trajectory_id"]).astype(str)
        group_partitions: dict[str, set[str]] = {}
        for partition, indices in partitions.items():
            for group in np.unique(groups[indices]):
                group_partitions.setdefault(str(group), set()).add(partition)
        crossing = sorted(group for group, values in group_partitions.items() if len(values) > 1)
        result["whole_trajectory_holdout"] = not crossing
        result["cross_partition_trajectory_ids"] = crossing
        result["status"] = "pass" if causal and not crossing else "fail"
    return result


def numeric_metadata_matrix(
    pairs: ForecastPairs, fields: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Build an explicit metadata-only view, rejecting identity/future fields."""

    if not fields:
        return np.empty((pairs.row_count, 0), dtype=np.float64), []
    forbidden = sorted(set(fields) & _IDENTITY_METADATA_FIELDS)
    if forbidden:
        raise SchemaError(
            "metadata-only baseline cannot use identity or future-index fields: "
            + ", ".join(forbidden)
        )
    columns: list[np.ndarray] = []
    for field in fields:
        if field not in pairs.metadata:
            raise SchemaError(f"metadata-only field {field!r} is missing")
        values = np.asarray(pairs.metadata[field])
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"metadata-only field {field!r} must be numeric") from exc
        if not np.isfinite(numeric).all():
            raise SchemaError(f"metadata-only field {field!r} must be finite")
        columns.append(numeric)
    return np.column_stack(columns), list(fields)


def _constant_prediction(train_targets: np.ndarray, target_kind: TargetKind) -> np.ndarray:
    if target_kind == "continuous":
        return np.full(1, float(np.mean(train_targets)), dtype=np.float64)
    return np.full(1, float(np.mean(train_targets)), dtype=np.float64)


def _fit_predict(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    values: np.ndarray,
    *,
    seed: int,
    target_kind: TargetKind,
    shuffle_labels: bool = False,
    empirical_cdf_tie_policy: str = "zero_is_positive",
) -> np.ndarray:
    if shuffle_labels:
        train_y = np.asarray(train_y).copy()
        np.random.default_rng(seed).shuffle(train_y)
    if target_kind == "continuous" and model_name in {
        "reverse_delta_sign",
        "sign_transition",
        "training_median_current_level",
        "empirical_cdf",
        "current_level_logistic",
        "current_delta_logistic",
    }:
        raise SchemaError(f"{model_name} is only defined for binary timing-sign targets")
    if model_name == "reverse_delta_sign":
        if train_x.shape[1] < 2:
            raise SchemaError("reverse_delta_sign requires [current level, current delta] features")
        fallback = float(np.mean(train_y))
        delta = np.asarray(values[:, 1], dtype=np.float64)
        return np.where(delta < 0.0, 1.0, np.where(delta > 0.0, 0.0, fallback))
    if model_name == "training_median_current_level":
        if train_x.shape[1] < 1:
            raise SchemaError("training_median_current_level requires a current-level feature")
        median = float(np.median(train_x[:, 0]))
        fallback = float(np.mean(train_y))
        level = np.asarray(values[:, 0], dtype=np.float64)
        return np.where(level < median, 1.0, np.where(level > median, 0.0, fallback))
    if model_name == "sign_transition":
        if train_x.shape[1] < 2:
            raise SchemaError("sign_transition requires [current level, current delta] features")
        delta = np.asarray(train_x[:, 1], dtype=np.float64)
        value_delta = np.asarray(values[:, 1], dtype=np.float64)
        categories = np.sign(delta).astype(np.int8)
        value_categories = np.sign(value_delta).astype(np.int8)
        fallback = float(np.mean(train_y))
        probabilities = np.full(len(values), fallback, dtype=np.float64)
        for category in (-1, 0, 1):
            mask = categories == category
            if np.any(mask):
                probabilities[value_categories == category] = float(np.mean(train_y[mask]))
        return probabilities
    if model_name == "empirical_cdf":
        if train_x.shape[1] < 1:
            raise SchemaError("empirical_cdf requires a current-level feature")
        training_levels = np.sort(np.asarray(train_x[:, 0], dtype=np.float64))
        side = "left" if empirical_cdf_tie_policy == "zero_is_positive" else "right"
        if empirical_cdf_tie_policy not in {"zero_is_positive", "zero_is_negative", "exclude"}:
            raise SchemaError(f"unsupported empirical CDF tie policy {empirical_cdf_tie_policy!r}")
        ranks = np.searchsorted(training_levels, np.asarray(values[:, 0]), side=side)
        return np.clip(1.0 - ranks / max(len(training_levels), 1), 0.0, 1.0)
    if model_name in {"current_level_logistic", "current_delta_logistic"}:
        column = 0 if model_name == "current_level_logistic" else 1
        if train_x.shape[1] <= column:
            raise SchemaError(f"{model_name} requires feature column {column}")
        train_x = train_x[:, [column]]
        values = values[:, [column]]
        model_name = "linear_logistic"
    if model_name == "combined_logistic":
        model_name = "linear_logistic"
    if model_name in {"majority", "random"}:
        if model_name == "majority":
            constant = _constant_prediction(train_y, target_kind)[0]
            return np.full(len(values), constant, dtype=np.float64)
        rng = np.random.default_rng(seed)
        if target_kind == "continuous":
            return rng.choice(np.asarray(train_y, dtype=np.float64), size=len(values), replace=True)
        return rng.random(len(values))
    if train_x.ndim != 2 or train_x.shape[1] == 0:
        raise SchemaError(f"{model_name} requires a non-empty feature matrix")
    scaler = StandardScaler().fit(train_x)
    x_train = scaler.transform(train_x)
    x_values = scaler.transform(values)
    if target_kind == "continuous":
        if model_name == "linear_ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "nearest_neighbor":
            model = KNeighborsRegressor(n_neighbors=min(5, len(x_train)), weights="distance")
        else:
            raise SchemaError(f"unsupported continuous forecaster {model_name!r}")
        model.fit(x_train, train_y)
        return np.asarray(model.predict(x_values), dtype=np.float64)
    if len(np.unique(train_y)) < 2:
        return np.full(len(values), float(np.asarray(train_y)[0]), dtype=np.float64)
    if model_name == "linear_logistic":
        model = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="lbfgs")
    elif model_name == "nearest_neighbor":
        model = KNeighborsClassifier(n_neighbors=min(5, len(x_train)), weights="distance")
    else:
        raise SchemaError(f"unsupported binary forecaster {model_name!r}")
    model.fit(x_train, train_y.astype(np.uint8))
    probabilities = np.asarray(model.predict_proba(x_values), dtype=np.float64)
    classes = np.asarray(model.classes_)
    if len(classes) == 1:
        return np.full(len(values), float(classes[0]), dtype=np.float64)
    positive = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, positive]


def _bootstrap_metric(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    metric: str,
    repetitions: int,
) -> list[float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        y = targets[indices]
        p = predictions[indices]
        if metric == "balanced_accuracy":
            values.append(_binary_balanced_accuracy(y, p >= 0.5))
        elif metric == "auroc":
            if len(np.unique(y)) > 1:
                values.append(float(roc_auc_score(y, p)))
        elif metric == "mae":
            values.append(float(mean_absolute_error(y, p)))
        elif metric == "rmse":
            values.append(float(np.sqrt(mean_squared_error(y, p))))
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _bootstrap_difference(
    targets: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> list[float]:
    """Bootstrap a paired correctness difference over complete trajectories."""

    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        left = np.asarray(left_predictions[indices]) >= 0.5
        right = np.asarray(right_predictions[indices]) >= 0.5
        values.append(float(np.mean(left == targets[indices]) - np.mean(right == targets[indices])))
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(mask):
            total += float(np.sum(mask)) / len(labels) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return total


def _binary_balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    recalls = []
    for value in (0, 1):
        denominator = int(np.sum(labels == value))
        recalls.append(float(np.sum((labels == value) & (predictions == value))) / max(denominator, 1))
    return float(np.mean(recalls))


def _evaluate(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    target_kind: TargetKind,
    groups: np.ndarray,
    seed: int,
    bootstrap_repetitions: int,
    baseline_value: float | None = None,
) -> dict[str, Any]:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(targets) != len(predictions):
        raise SchemaError("forecast predictions and targets are not aligned")
    if target_kind == "binary":
        probabilities = np.clip(predictions, 0.0, 1.0)
        binary = (probabilities >= 0.5).astype(np.uint8)
        auroc = float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) > 1 else float("nan")
        return {
            "sample_count": int(len(targets)),
            "class_balance": {"0": int(np.sum(targets == 0)), "1": int(np.sum(targets == 1))},
            "balanced_accuracy": _binary_balanced_accuracy(targets, binary),
            "auroc": auroc,
            "brier_score": float(brier_score_loss(targets, probabilities)),
            "calibration_error_ece_10": _calibration_error(targets, probabilities),
            "confidence_interval_95": _bootstrap_metric(
                targets,
                probabilities,
                groups,
                seed=seed,
                metric="balanced_accuracy",
                repetitions=bootstrap_repetitions,
            ),
            "auroc_confidence_interval_95": _bootstrap_metric(
                targets,
                probabilities,
                groups,
                seed=seed + 1,
                metric="auroc",
                repetitions=bootstrap_repetitions,
            ),
            "confidence_interval_unit": "trajectory_id",
            "chance_reference": 0.5,
        }
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    mae = float(mean_absolute_error(targets, predictions))
    constant = float(np.mean(targets) if baseline_value is None else baseline_value)
    baseline_rmse = float(np.sqrt(mean_squared_error(targets, np.full(len(targets), constant))))
    return {
        "sample_count": int(len(targets)),
        "target_mean": float(np.mean(targets)),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(targets, predictions)) if len(np.unique(targets)) > 1 else float("nan"),
        "skill_over_constant_mean": (
            float(1.0 - rmse / baseline_rmse) if baseline_rmse > 0 else float("nan")
        ),
        "constant_baseline": constant,
        "confidence_interval_95": _bootstrap_metric(
            targets,
            predictions,
            groups,
            seed=seed,
            metric="rmse",
            repetitions=bootstrap_repetitions,
        ),
        "mae_confidence_interval_95": _bootstrap_metric(
            targets,
            predictions,
            groups,
            seed=seed + 1,
            metric="mae",
            repetitions=bootstrap_repetitions,
        ),
        "confidence_interval_unit": "trajectory_id",
        "chance_reference": "constant train-target mean; report skill against it",
    }


def _group_preserving_permutation(
    targets: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    shifts: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Circularly shift target rows within each natural trajectory group.

    A row-wise permutation would destroy the trajectory boundary structure and
    can make a temporal null too easy.  Circular shifts retain each group's
    target distribution and row count while breaking present/future alignment.
    The optional common shifts are used by the max-statistic family test so
    the null preserves dependence across horizons and models.
    """

    values = np.asarray(targets).copy()
    group_values = np.asarray(groups).astype(str)
    for group in np.unique(group_values):
        indices = np.flatnonzero(group_values == group)
        if len(indices) < 2:
            continue
        shift = int(shifts[group]) % len(indices) if shifts is not None else int(rng.integers(len(indices)))
        if shift:
            values[indices] = values[indices][np.roll(np.arange(len(indices)), shift)]
    return values


def _effect_statistic(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    target_kind: TargetKind,
    baseline_value: float | None = None,
) -> float:
    if target_kind == "binary":
        return float(_binary_balanced_accuracy(targets, predictions >= 0.5) - 0.5)
    baseline = float(np.mean(targets) if baseline_value is None else baseline_value)
    baseline_rmse = float(np.sqrt(mean_squared_error(targets, np.full(len(targets), baseline))))
    if baseline_rmse <= 0:
        return float("nan")
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    return float(1.0 - rmse / baseline_rmse)


def _group_shift_matrix(
    targets: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    shifts_by_group: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Vectorize circular within-group shifts over randomization replicates."""

    targets = np.asarray(targets)
    group_values = np.asarray(groups).astype(str)
    result = np.empty((repetitions, len(targets)), dtype=targets.dtype)
    for group in np.unique(group_values):
        indices = np.flatnonzero(group_values == group)
        if len(indices) < 2:
            result[:, indices] = targets[indices]
            continue
        shifts = (
            np.asarray(shifts_by_group[group], dtype=np.int64) % len(indices)
            if shifts_by_group is not None
            else rng.integers(0, len(indices), size=repetitions, dtype=np.int64)
        )
        source = (np.arange(len(indices), dtype=np.int64)[None, :] - shifts[:, None]) % len(indices)
        result[:, indices] = targets[indices][source]
    return result


def _vectorized_effects(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    groups: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    target_kind: TargetKind,
    baseline_value: float | None = None,
    shifts_by_group: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    permuted = _group_shift_matrix(
        targets,
        groups,
        repetitions,
        rng,
        shifts_by_group=shifts_by_group,
    )
    if target_kind == "binary":
        predicted_positive = np.asarray(predictions) >= 0.5
        actual_positive = permuted == 1
        positive_denominator = np.sum(actual_positive, axis=1)
        negative_denominator = np.sum(~actual_positive, axis=1)
        true_positive = np.sum(actual_positive & predicted_positive[None, :], axis=1)
        true_negative = np.sum((~actual_positive) & (~predicted_positive[None, :]), axis=1)
        return 0.5 * (
            true_positive / np.maximum(positive_denominator, 1)
            + true_negative / np.maximum(negative_denominator, 1)
        ) - 0.5
    predictions = np.asarray(predictions, dtype=np.float64)
    baseline = float(np.mean(targets) if baseline_value is None else baseline_value)
    baseline_rmse = np.sqrt(np.mean((permuted - baseline) ** 2, axis=1))
    rmse = np.sqrt(np.mean((permuted - predictions[None, :]) ** 2, axis=1))
    return np.where(baseline_rmse > 0.0, 1.0 - rmse / baseline_rmse, np.nan)


def _permutation_p_value(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    groups: np.ndarray,
    seed: int,
    repetitions: int,
    target_kind: TargetKind,
    baseline_value: float | None = None,
) -> float:
    observed = _effect_statistic(
        targets,
        predictions,
        target_kind=target_kind,
        baseline_value=baseline_value,
    )
    rng = np.random.default_rng(seed)
    if target_kind == "binary":
        effects = _vectorized_effects(
            targets,
            predictions,
            groups=groups,
            repetitions=repetitions,
            rng=rng,
            target_kind=target_kind,
            baseline_value=baseline_value,
        )
        return float((1 + np.sum(effects >= observed)) / (repetitions + 1))
    null: list[float] = []
    for _ in range(repetitions):
        permuted = _group_preserving_permutation(targets, groups, rng)
        null.append(
            _effect_statistic(
                permuted,
                predictions,
                target_kind=target_kind,
                baseline_value=baseline_value,
            )
        )
    if not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(np.asarray(null) >= observed)) / (repetitions + 1))


def _max_statistic_permutation_p_values(
    families: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    """Return max-statistic adjusted p-values for one declared test family.

    The family is supplied by the caller and normally contains all requested
    horizons and non-control probes for one target/condition.  One common
    trajectory-level shift per permutation is reused across that family,
    preserving the dependence created by shared trajectories and horizons.
    """

    if not families:
        return {}
    observed = {
        str(item["key"]): _effect_statistic(
            np.asarray(item["targets"]),
            np.asarray(item["predictions"]),
            target_kind=cast(TargetKind, item["target_kind"]),
            baseline_value=item.get("baseline_value"),
        )
        for item in families
    }
    rng = np.random.default_rng(seed)
    all_groups = sorted(
        {
            str(group)
            for item in families
            for group in np.asarray(item["groups"]).astype(str)
        }
    )
    shifts = {
        group: rng.integers(0, 2**32, size=repetitions, dtype=np.uint64)
        for group in all_groups
    }
    null_array = np.full(repetitions, -np.inf, dtype=np.float64)
    for item in families:
        effects = _vectorized_effects(
            np.asarray(item["targets"]),
            np.asarray(item["predictions"]),
            groups=np.asarray(item["groups"]).astype(str),
            repetitions=repetitions,
            rng=rng,
            target_kind=cast(TargetKind, item["target_kind"]),
            baseline_value=item.get("baseline_value"),
            shifts_by_group=shifts,
        )
        null_array = np.maximum(null_array, np.where(np.isfinite(effects), effects, -np.inf))
    return {
        key: float((1 + np.sum(null_array >= effect)) / (repetitions + 1))
        if np.isfinite(effect)
        else float("nan")
        for key, effect in observed.items()
    }


def evaluate_horizon_curve(
    pairs_by_horizon: Mapping[str, ForecastPairs],
    splits_by_horizon: Mapping[str, Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    feature_view: str = "state",
    metadata_features: np.ndarray | Mapping[str, np.ndarray] | None = None,
    control_features: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    control_targets: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    seeds: Sequence[int] = (11, 23, 37),
    bootstrap_repetitions: int = 400,
    permutation_repetitions: int = 400,
    practical_balanced_accuracy: float = 0.55,
    practical_continuous_skill: float = 0.05,
    significance_alpha: float = 0.05,
    empirical_cdf_tie_policy: str = "zero_is_positive",
) -> dict[str, Any]:
    """Evaluate fixed baselines and probes for every requested horizon."""

    allowed = {
        "majority",
        "random",
        "shuffled_labels",
        "metadata_only",
        "linear_logistic",
        "nearest_neighbor",
        "reverse_delta_sign",
        "sign_transition",
        "training_median_current_level",
        "empirical_cdf",
        "current_level_logistic",
        "current_delta_logistic",
        "combined_logistic",
        "linear_ridge",
        "temporally_shuffled_states",
        "wrong_trajectory_pairing",
        "reversed_alignment",
        "same_state",
    }
    unknown = sorted(set(model_names) - allowed)
    if unknown:
        raise SchemaError(f"unsupported horizon models: {unknown}")
    report: dict[str, Any] = {
        "schema": "sensetrace.predictive-horizon-curve.v1",
        "feature_view": feature_view,
        "models": list(model_names),
        "empirical_cdf_tie_policy": empirical_cdf_tie_policy,
        "horizons": {},
        "selection_policy": "all predeclared models are reported; selected_model uses validation only",
        "test_policy": "test predictions are generated once per predeclared model/seed and never used for selection",
        "multiplicity": {
            "method": "group-preserving max-statistic permutation",
            "family": "all predeclared non-control models multiplied by all requested horizons within one target and condition",
            "replicate_unit": "training seed; repeated seeds are reported as replications, not extra hypothesis families",
            "raw_p_value_field": "permutation_p_value",
            "corrected_p_value_field": "permutation_p_value_max_statistic",
        },
    }
    control_models = {
        "majority",
        "random",
        "shuffled_labels",
        "metadata_only",
        "temporally_shuffled_states",
        "wrong_trajectory_pairing",
        "reversed_alignment",
        "same_state",
        "reverse_delta_sign",
        "sign_transition",
        "training_median_current_level",
        "empirical_cdf",
    }
    test_families: list[dict[str, Any]] = []
    simple_baseline_names = (
        "reverse_delta_sign",
        "sign_transition",
        "training_median_current_level",
        "empirical_cdf",
        "current_level_logistic",
        "current_delta_logistic",
    )
    for horizon_key, pairs in pairs_by_horizon.items():
        split = splits_by_horizon[horizon_key]
        partitions = horizon_partition_indices(pairs, split)
        audit = validate_horizon_alignment(pairs, split)
        if audit["status"] != "pass":
            raise IntegrityError(f"causal horizon audit failed for {horizon_key}")
        train = partitions["train"]
        validation = partitions["validation"]
        test = partitions["test"]
        groups = np.asarray(pairs.metadata["trajectory_id"]).astype(str)
        if feature_view == "metadata_only":
            if metadata_features is None:
                raise SchemaError("metadata_only feature view requires metadata_features")
            features = np.asarray(
                metadata_features[horizon_key]
                if isinstance(metadata_features, Mapping)
                else metadata_features,
                dtype=np.float64,
            )
        else:
            features = pairs.features.astype(np.float64)
        if len(features) != pairs.row_count:
            raise SchemaError("metadata features are not aligned to forecast pairs")
        is_binary = pairs.target.kind == "binary"
        models_report: dict[str, Any] = {}
        validation_scores: dict[str, float] = {}
        prediction_cache: dict[str, dict[int, dict[str, Any]]] = {}
        for model_name in model_names:
            if (control_features is not None and model_name in control_features) or (
                control_targets is not None and model_name in control_targets
            ):
                if (control_features is None or model_name not in control_features) and model_name not in {
                    "same_state",
                    "reversed_alignment",
                }:
                    raise SchemaError(f"control features are missing for {model_name!r}")
            if model_name == "metadata_only":
                if metadata_features is None:
                    models_report[model_name] = {"status": "unavailable", "reason": "no metadata feature view"}
                    continue
                model_features = np.asarray(
                    metadata_features[horizon_key]
                    if isinstance(metadata_features, Mapping)
                    else metadata_features,
                    dtype=np.float64,
                )
            elif model_name in {"majority", "random", "shuffled_labels"}:
                model_features = features
            else:
                model_features = features
            if model_name in {"temporally_shuffled_states", "wrong_trajectory_pairing"}:
                if control_features is None or model_name not in control_features:
                    raise SchemaError(f"control features are required for {model_name!r}")
                model_features = np.asarray(control_features[model_name][horizon_key], dtype=np.float64)
            model_targets = pairs.targets
            if control_targets is not None and model_name in control_targets:
                model_targets = np.asarray(control_targets[model_name][horizon_key])
            if len(model_targets) != pairs.row_count:
                raise SchemaError(f"control targets are not aligned for {model_name!r}")
            if model_name == "metadata_only" and model_features.shape[1] == 0:
                models_report[model_name] = {"status": "unavailable", "reason": "empty metadata feature view"}
                continue
            if not is_binary and model_name in {
                "linear_logistic",
                "combined_logistic",
                "reverse_delta_sign",
                "sign_transition",
                "training_median_current_level",
                "empirical_cdf",
                "current_level_logistic",
                "current_delta_logistic",
            }:
                models_report[model_name] = {
                    "status": "unavailable",
                    "reason": "binary classifier is incompatible with continuous target",
                }
                continue
            if is_binary and model_name == "linear_ridge":
                models_report[model_name] = {
                    "status": "unavailable",
                    "reason": "continuous regressor is incompatible with binary target",
                }
                continue
            runs: list[dict[str, Any]] = []
            for seed in seeds:
                train_y = model_targets[train]
                if model_name == "shuffled_labels":
                    control_model = "linear_ridge" if not is_binary else "linear_logistic"
                    validation_predictions = _fit_predict(
                        control_model,
                        model_features[train],
                        train_y,
                        model_features[validation],
                        seed=seed,
                        target_kind=pairs.target.kind,
                        shuffle_labels=True,
                        empirical_cdf_tie_policy=empirical_cdf_tie_policy,
                    )
                    test_predictions = _fit_predict(
                        control_model,
                        model_features[train],
                        train_y,
                        model_features[test],
                        seed=seed,
                        target_kind=pairs.target.kind,
                        shuffle_labels=True,
                        empirical_cdf_tie_policy=empirical_cdf_tie_policy,
                    )
                else:
                    fit_name = (
                        "linear_ridge"
                        if model_name == "metadata_only" and not is_binary
                        else "linear_logistic"
                        if model_name == "metadata_only"
                        else "linear_ridge"
                        if model_name in {
                            "temporally_shuffled_states",
                            "wrong_trajectory_pairing",
                            "reversed_alignment",
                            "same_state",
                        }
                        and not is_binary
                        else "linear_logistic"
                        if model_name in {
                            "temporally_shuffled_states",
                            "wrong_trajectory_pairing",
                            "reversed_alignment",
                            "same_state",
                        }
                        else model_name
                    )
                    validation_predictions = _fit_predict(
                        fit_name,
                        model_features[train],
                        train_y,
                        model_features[validation],
                        seed=seed,
                        target_kind=pairs.target.kind,
                        empirical_cdf_tie_policy=empirical_cdf_tie_policy,
                    )
                    test_predictions = _fit_predict(
                        fit_name,
                        model_features[train],
                        train_y,
                        model_features[test],
                        seed=seed,
                        target_kind=pairs.target.kind,
                        empirical_cdf_tie_policy=empirical_cdf_tie_policy,
                    )
                validation_result = _evaluate(
                    model_targets[validation],
                    validation_predictions,
                    target_kind=pairs.target.kind,
                    groups=groups[validation],
                    seed=seed,
                    bootstrap_repetitions=max(20, min(bootstrap_repetitions, 100)),
                    baseline_value=float(np.mean(train_y)) if not is_binary else None,
                )
                test_result = _evaluate(
                    model_targets[test],
                    test_predictions,
                    target_kind=pairs.target.kind,
                    groups=groups[test],
                    seed=seed,
                    bootstrap_repetitions=bootstrap_repetitions,
                    baseline_value=float(np.mean(train_y)) if not is_binary else None,
                )
                if is_binary:
                    validation_scores.setdefault(model_name, 0.0)
                    validation_scores[model_name] += float(validation_result["balanced_accuracy"]) / len(seeds)
                    test_result["permutation_p_value"] = (
                        float("nan")
                        if model_name in control_models
                        else _permutation_p_value(
                            model_targets[test],
                            test_predictions,
                            groups=groups[test],
                            seed=seed + 10_000,
                            repetitions=permutation_repetitions,
                            target_kind=pairs.target.kind,
                        )
                    )
                else:
                    validation_scores.setdefault(model_name, 0.0)
                    validation_scores[model_name] += float(validation_result["rmse"]) / len(seeds)
                    test_result["permutation_p_value"] = (
                        float("nan")
                        if model_name in control_models
                        else _permutation_p_value(
                            model_targets[test],
                            test_predictions,
                            groups=groups[test],
                            seed=seed + 10_000,
                            repetitions=permutation_repetitions,
                            target_kind=pairs.target.kind,
                            baseline_value=float(np.mean(train_y)),
                        )
                    )
                prediction_cache.setdefault(model_name, {})[int(seed)] = {
                    "validation_score": (
                        float(validation_result["balanced_accuracy"])
                        if is_binary
                        else float(validation_result["rmse"])
                    ),
                    "test_predictions": np.asarray(test_predictions),
                    "targets": np.asarray(model_targets[test]),
                    "control_target": model_name in (control_targets or {}),
                }
                if model_name not in control_models:
                    test_families.append(
                        {
                            "key": f"{horizon_key}:{model_name}:{seed}",
                            "horizon_key": horizon_key,
                            "model_name": model_name,
                            "seed": int(seed),
                            "targets": model_targets[test],
                            "predictions": test_predictions,
                            "groups": groups[test],
                            "target_kind": pairs.target.kind,
                            "baseline_value": float(np.mean(train_y)) if not is_binary else None,
                            "result": test_result,
                        }
                    )
                runs.append({"seed": int(seed), "validation": validation_result, "test": test_result})
            models_report[model_name] = {
                "status": "evaluated",
                "control": model_name in control_models,
                "runs": runs,
            }
        available_simple = [
            name
            for name in simple_baseline_names
            if name in prediction_cache
            and any(not item["control_target"] for item in prediction_cache[name].values())
        ]
        for model_name, model_runs in models_report.items():
            if model_runs.get("status") != "evaluated" or model_name not in prediction_cache:
                continue
            for run in model_runs["runs"]:
                seed = int(run["seed"])
                current = prediction_cache[model_name].get(seed)
                if current is None or current["control_target"] or not is_binary:
                    continue
                candidates = [
                    name
                    for name in available_simple
                    if seed in prediction_cache[name] and not prediction_cache[name][seed]["control_target"]
                ]
                if not candidates:
                    continue
                baseline_name = max(
                    candidates,
                    key=lambda name: prediction_cache[name][seed]["validation_score"],
                )
                baseline = prediction_cache[baseline_name][seed]
                target_values = np.asarray(current["targets"], dtype=np.uint8)
                current_predictions = np.asarray(current["test_predictions"])
                baseline_predictions = np.asarray(baseline["test_predictions"])
                current_correct = (current_predictions >= 0.5) == target_values
                baseline_correct = (baseline_predictions >= 0.5) == target_values
                run["paired_baseline_contrast"] = {
                    "selection": "strongest declared simple baseline by validation balanced accuracy",
                    "baseline_model": baseline_name,
                    "balanced_accuracy_difference": float(
                        _binary_balanced_accuracy(target_values, current_predictions >= 0.5)
                        - _binary_balanced_accuracy(target_values, baseline_predictions >= 0.5)
                    ),
                    "paired_correctness_difference": float(
                        np.mean(current_correct.astype(np.float64) - baseline_correct.astype(np.float64))
                    ),
                    "paired_correctness_confidence_interval_95": _bootstrap_difference(
                        target_values,
                        current_predictions,
                        baseline_predictions,
                        groups[test],
                        seed=seed + 30_000,
                        repetitions=bootstrap_repetitions,
                    ),
                    "confidence_interval_unit": "trajectory_id",
                }
        available_for_selection = [
            name
            for name in model_names
            if name in validation_scores and name not in {"majority", "random", "shuffled_labels", "metadata_only"}
        ]
        if available_for_selection:
            selected = (
                max(available_for_selection, key=lambda name: validation_scores[name])
                if is_binary
                else min(available_for_selection, key=lambda name: validation_scores[name])
            )
        else:
            selected = None
        report["horizons"][horizon_key] = {
            "horizon": pairs.horizon.as_dict(),
            "target": pairs.target.as_dict(),
            "pair_count": pairs.row_count,
            "split_fingerprint": split["split_fingerprint"],
            "alignment_audit": audit,
            "validation_selection": {
                "selected_model": selected,
                "validation_scores": validation_scores,
                "candidate_models": available_for_selection,
            },
            "models": models_report,
        }
    corrected = _max_statistic_permutation_p_values(
        test_families,
        seed=10_000_019,
        repetitions=permutation_repetitions,
    )
    for family in test_families:
        family["result"]["permutation_p_value_max_statistic"] = corrected[family["key"]]
    report["useful_lead_summary"] = summarize_useful_lead(
        report,
        practical_balanced_accuracy=practical_balanced_accuracy,
        practical_continuous_skill=practical_continuous_skill,
        significance_alpha=significance_alpha,
    )
    return report


def summarize_useful_lead(
    report: Mapping[str, Any],
    *,
    practical_balanced_accuracy: float = 0.55,
    practical_continuous_skill: float = 0.05,
    significance_alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize practical lead time separately from raw predictability.

    A practical horizon is the largest tested distance whose mean held-out
    score clears a predeclared effect threshold.  A statistically supported
    horizon additionally requires a positive mean effect and at least half of
    the repeated test seeds to pass the per-run permutation threshold.  These
    are descriptive finite-grid summaries, not optional stopping procedures.
    """

    by_model: dict[str, Any] = {}
    horizon_records = report.get("horizons", {})
    model_names = sorted(
        {
            model_name
            for record in horizon_records.values()
            for model_name in record.get("models", {})
        }
    )
    for model_name in model_names:
        curve: list[dict[str, Any]] = []
        for record in horizon_records.values():
            model = record.get("models", {}).get(model_name, {})
            if model.get("status") != "evaluated" or not model.get("runs"):
                continue
            test_runs = [run["test"] for run in model["runs"]]
            horizon = record["horizon"]
            distance = float(horizon["distance"])
            if record["target"]["kind"] == "binary":
                scores = [float(run["balanced_accuracy"]) for run in test_runs]
                p_values = [float(run.get("permutation_p_value", float("nan"))) for run in test_runs]
                corrected_p_values = [
                    float(run.get("permutation_p_value_max_statistic", float("nan")))
                    for run in test_runs
                ]
                finite_p_values = [p for p in p_values if np.isfinite(p)]
                finite_corrected_p_values = [p for p in corrected_p_values if np.isfinite(p)]
                score = float(np.mean(scores))
                effect = score - 0.5
                practical = score >= practical_balanced_accuracy
                supported = effect > 0 and sum(
                    p <= significance_alpha for p in corrected_p_values if np.isfinite(p)
                ) >= max(1, int(np.ceil(len(p_values) / 2)))
                curve.append(
                    {
                        "distance": distance,
                        "score": score,
                        "metric": "balanced_accuracy",
                        "effect_over_chance": effect,
                        "practical_threshold": practical_balanced_accuracy,
                        "practical": practical,
                        "statistically_supported": supported,
                        "permutation_p_value": float(np.mean(finite_p_values)) if finite_p_values else None,
                        "permutation_p_value_max_statistic": (
                            float(np.mean(finite_corrected_p_values))
                            if finite_corrected_p_values
                            else None
                        ),
                    }
                )
            else:
                scores = [float(run["rmse"]) for run in test_runs]
                baseline = float(
                    np.mean(
                        [
                            item["test"]["rmse"]
                            for item in record.get("models", {}).get("majority", {}).get("runs", [])
                        ]
                    )
                ) if record.get("models", {}).get("majority", {}).get("runs") else float("nan")
                skill = float(1.0 - np.mean(scores) / baseline) if baseline > 0 else float("nan")
                practical = bool(np.isfinite(skill) and skill >= practical_continuous_skill)
                p_values = [float(run.get("permutation_p_value", float("nan"))) for run in test_runs]
                corrected_p_values = [
                    float(run.get("permutation_p_value_max_statistic", float("nan")))
                    for run in test_runs
                ]
                finite_p_values = [p for p in p_values if np.isfinite(p)]
                finite_corrected_p_values = [p for p in corrected_p_values if np.isfinite(p)]
                supported = bool(
                    np.isfinite(skill)
                    and sum(
                        p <= significance_alpha for p in corrected_p_values if np.isfinite(p)
                    )
                    >= max(1, int(np.ceil(len(corrected_p_values) / 2)))
                )
                curve.append(
                    {
                        "distance": distance,
                        "score": float(np.mean(scores)),
                        "metric": "rmse",
                        "skill_over_constant_mean": skill,
                        "practical_threshold": practical_continuous_skill,
                        "practical": practical,
                        "statistically_supported": supported,
                        "permutation_p_value": float(np.mean(finite_p_values)) if finite_p_values else None,
                        "permutation_p_value_max_statistic": (
                            float(np.mean(finite_corrected_p_values))
                            if finite_corrected_p_values
                            else None
                        ),
                    }
                )
        curve.sort(key=lambda item: item["distance"])
        if not curve:
            continue
        target_kind = curve[0]["metric"]
        values = np.asarray(
            [
                item.get("effect_over_chance", item.get("skill_over_constant_mean", float("nan")))
                for item in curve
            ],
            dtype=np.float64,
        )
        distances = np.asarray([item["distance"] for item in curve], dtype=np.float64)
        positive = np.maximum(values, 0.0)
        area = float(np.trapezoid(positive, distances)) if len(curve) > 1 else float(positive[0])
        span = float(distances[-1] - distances[0]) if len(curve) > 1 else 1.0
        practical_rows = [item for item in curve if item["practical"]]
        supported_rows = [item for item in curve if item["statistically_supported"]]
        by_model[model_name] = {
            "curve": curve,
            "metric": target_kind,
            "area_under_positive_effect": area,
            "normalized_area_under_positive_effect": float(area / span),
            "maximum_practical_horizon": (
                max(item["distance"] for item in practical_rows) if practical_rows else None
            ),
            "maximum_statistically_supported_horizon": (
                max(item["distance"] for item in supported_rows) if supported_rows else None
            ),
        }
    selected = [
        record.get("validation_selection", {}).get("selected_model")
        for record in horizon_records.values()
        if record.get("validation_selection", {}).get("selected_model")
    ]
    return {
        "schema": "sensetrace.useful-lead-summary.v1",
        "practical_balanced_accuracy": practical_balanced_accuracy,
        "practical_continuous_skill": practical_continuous_skill,
        "significance_alpha": significance_alpha,
        "statistical_rule": "positive mean effect and at least half of repeated test seeds pass the permutation threshold",
        "model_summaries": by_model,
        "selected_models_by_horizon": selected,
    }


def _git_commit() -> str:
    declared = os.environ.get("SENSETRACE_COMMIT")
    if declared:
        return declared
    # Remote editable installs are intentionally deployed without .git.  The
    # host deployment writes this marker next to the installed source so
    # analysis artifacts still bind to the exact source that ran them.
    source_root = Path(__file__).resolve().parents[2]
    deployed_marker = source_root / ".sensetrace-commit"
    if deployed_marker.is_file():
        try:
            value = deployed_marker.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if value:
            return value
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    material = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"cannot read immutable horizon artifact {path}") from exc
        if existing != json.loads(material):
            raise IntegrityError(f"immutable horizon artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(material, encoding="utf-8")


def write_horizon_run(
    output: str | Path,
    *,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    source_fingerprint: str,
    split_fingerprints: Mapping[str, str],
    splits: Mapping[str, Mapping[str, Any]] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    claim_boundary: str = "synthetic trajectory-analysis validation; no physical DRAM or model-inference claim",
) -> dict[str, Any]:
    """Persist machine-readable reproducibility metadata and curve results."""

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    metadata = dict(run_metadata or {})
    serialized_report = json.dumps(dict(report), indent=2, sort_keys=True, default=str) + "\n"
    serialized_splits = (
        json.dumps({key: dict(value) for key, value in splits.items()}, indent=2, sort_keys=True) + "\n"
        if splits is not None
        else None
    )
    manifest: dict[str, Any] = {
        "schema": "sensetrace.predictive-horizon-manifest.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "sensetrace_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "source_trajectory_fingerprint": source_fingerprint,
        "split_fingerprints": dict(split_fingerprints),
        "results_sha256": sha256_bytes(serialized_report.encode("utf-8")),
        "splits_sha256": (
            sha256_bytes(serialized_splits.encode("utf-8")) if serialized_splits is not None else None
        ),
        "config": dict(config),
        "run_metadata": metadata,
        "causal_policy": {
            "mode": "passive_observation",
            "forecast_changes_target_pipeline": False,
            "feature_source": "present state at forecast origin only",
            "future_state_used_for": "target extraction only",
        },
        "claim_boundary": claim_boundary,
    }
    _immutable_json(root / "manifest.json", manifest)
    if splits is not None:
        _immutable_json(root / "splits.json", {key: dict(value) for key, value in splits.items()})
    _immutable_json(root / "results.json", dict(report))
    return {"manifest": manifest, "report": dict(report), "output": str(root)}


def load_horizon_run(output: str | Path) -> dict[str, Any]:
    """Read and verify a persisted horizon condition before using its result."""

    root = Path(output)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        report_text = (root / "results.json").read_text(encoding="utf-8")
        report = json.loads(report_text)
        split_text = (root / "splits.json").read_text(encoding="utf-8")
        splits = json.loads(split_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read horizon run artifacts under {root}") from exc
    if manifest.get("schema") != "sensetrace.predictive-horizon-manifest.v1":
        raise IntegrityError("unsupported predictive-horizon manifest schema")
    if manifest.get("results_sha256") != sha256_bytes(report_text.encode("utf-8")):
        raise IntegrityError("predictive-horizon results hash does not match its manifest")
    if manifest.get("splits_sha256") != sha256_bytes(split_text.encode("utf-8")):
        raise IntegrityError("predictive-horizon splits hash does not match its manifest")
    if report.get("schema") != "sensetrace.predictive-horizon-curve.v1":
        raise IntegrityError("unsupported predictive-horizon result schema")
    return {"manifest": manifest, "report": report, "splits": splits}


def generate_synthetic_trajectories(
    *,
    condition: Literal["predictable", "null"],
    trajectory_count: int,
    length: int,
    state_dim: int,
    seed: int,
    autoregressive_coefficient: float = 0.85,
) -> list[StateTrajectory]:
    """Generate a falsifiable control pair for the horizon analysis.

    ``predictable`` uses a stationary AR(1) process.  ``null`` draws every
    state independently.  Both conditions share the same dimensions,
    trajectory lengths, labels-by-construction, and metadata shape.
    """

    if condition not in {"predictable", "null"}:
        raise SchemaError(f"unsupported synthetic horizon condition {condition!r}")
    if trajectory_count < 6 or length < 4 or state_dim < 1:
        raise SchemaError("synthetic horizon data requires >=6 trajectories, length >=4, dimension >=1")
    if not 0.0 <= autoregressive_coefficient < 1.0:
        raise SchemaError("autoregressive_coefficient must be in [0, 1)")
    rng = np.random.default_rng(seed)
    trajectories: list[StateTrajectory] = []
    innovation_scale = float(np.sqrt(max(1e-12, 1.0 - autoregressive_coefficient**2)))
    for index in range(trajectory_count):
        if condition == "null":
            states = rng.normal(size=(length, state_dim)).astype(np.float32)
        else:
            states = np.empty((length, state_dim), dtype=np.float32)
            states[0] = rng.normal(size=state_dim)
            for time_index in range(1, length):
                states[time_index] = (
                    autoregressive_coefficient * states[time_index - 1]
                    + innovation_scale * rng.normal(size=state_dim)
                )
        trajectories.append(
            StateTrajectory(
                trajectory_id=f"{condition}-trajectory-{index:04d}",
                states=states,
                metadata={
                    "condition": condition,
                    "generator": "synthetic-horizon-ar1-v1",
                    "replicate": index,
                },
            )
        )
    return trajectories


def run_synthetic_horizon_experiment(
    config: Mapping[str, Any],
    output: str | Path,
    *,
    conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize predictable/null controls and write one report per condition."""

    data = dict(config.get("data", {}))
    horizon_config = dict(config.get("horizon", {}))
    target_config = dict(config.get("target", {}))
    trajectory_count = int(data.get("trajectories", 48))
    length = int(data.get("length", 48))
    state_dim = int(data.get("state_dim", 8))
    seed = int(config.get("experiment", {}).get("seed", 1337))
    requested_horizons = horizon_config.get("distances", [1, 2, 4, 8, 16])
    if not isinstance(requested_horizons, list) or not requested_horizons:
        raise SchemaError("horizon.distances must be a non-empty list")
    unit = str(horizon_config.get("unit", "step"))
    alignment = str(horizon_config.get("alignment", "index"))
    horizons = [
        Horizon(distance=value, unit=unit, alignment=cast(AlignmentMode, alignment))
        for value in requested_horizons
    ]
    target = TargetSpec(
        name=str(target_config.get("name", "future_state_0_sign")),
        kind=cast(TargetKind, str(target_config.get("kind", "binary"))),
        state_index=int(target_config.get("state_index", 0)),
        threshold=float(target_config.get("threshold", 0.0)),
        positive_if=cast(
            Literal["ge", "gt", "le", "lt"], str(target_config.get("positive_if", "ge"))
        ),
    )
    models = tuple(
        str(item)
        for item in config.get(
            "models",
            ["majority", "random", "shuffled_labels", "metadata_only", "linear_logistic", "nearest_neighbor"],
        )
    )
    seeds = tuple(int(item) for item in config.get("training", {}).get("seeds", [11, 23, 37]))
    split_config = dict(config.get("splits", {}).get("primary", {}))
    selected_conditions = tuple(conditions or config.get("conditions", ["predictable", "null"]))
    root = Path(output)
    condition_reports: dict[str, Any] = {}
    for condition_offset, condition in enumerate(selected_conditions):
        if condition not in {"predictable", "null"}:
            raise SchemaError(f"unsupported horizon condition {condition!r}")
        trajectories = generate_synthetic_trajectories(
            condition=cast(Literal["predictable", "null"], condition),
            trajectory_count=trajectory_count,
            length=length,
            state_dim=state_dim,
            seed=seed + condition_offset,
            autoregressive_coefficient=float(data.get("autoregressive_coefficient", 0.85)),
        )
        pairs_by_horizon: dict[str, ForecastPairs] = {}
        splits_by_horizon: dict[str, dict[str, Any]] = {}
        metadata_by_horizon: dict[str, np.ndarray] = {}
        for horizon in horizons:
            key = f"{horizon.unit}:{horizon.distance:g}:{horizon.alignment}"
            pairs = build_forecast_pairs(trajectories, horizon, target)
            pairs_by_horizon[key] = pairs
            splits_by_horizon[key] = build_horizon_split(
                pairs,
                seed=seed,
                group_key="trajectory_id",
                train_fraction=float(split_config.get("train_fraction", 0.7)),
                validation_fraction=float(split_config.get("validation_fraction", 0.15)),
                test_fraction=float(split_config.get("test_fraction", 0.15)),
            )
            # Position is an explicit, non-state metadata-only control.  It
            # is fitted through the same train-only scaler as every probe.
            metadata_by_horizon[key], _ = numeric_metadata_matrix(pairs, ["origin_position"])
        report = evaluate_horizon_curve(
            pairs_by_horizon,
            splits_by_horizon,
            model_names=models,
            metadata_features=metadata_by_horizon,
            seeds=seeds,
            bootstrap_repetitions=int(config.get("reporting", {}).get("bootstrap_repetitions", 400)),
            permutation_repetitions=int(config.get("reporting", {}).get("permutation_repetitions", 400)),
            practical_balanced_accuracy=float(
                config.get("reporting", {}).get("practical_balanced_accuracy", 0.55)
            ),
            practical_continuous_skill=float(
                config.get("reporting", {}).get("practical_continuous_skill", 0.05)
            ),
            significance_alpha=float(config.get("reporting", {}).get("significance_alpha", 0.05)),
        )
        condition_config = dict(config)
        condition_config["condition"] = condition
        saved = write_horizon_run(
            root / condition,
            report=report,
            config=condition_config,
            source_fingerprint=pairs_by_horizon[next(iter(pairs_by_horizon))].source_fingerprint,
            split_fingerprints={key: value["split_fingerprint"] for key, value in splits_by_horizon.items()},
            splits=splits_by_horizon,
            run_metadata={
                "condition": condition,
                "execution_host": platform.node() or "unavailable",
                "requested_node": config.get("run_metadata", {}).get("node", "controller"),
                "target_model": config.get("run_metadata", {}).get("target_model", "synthetic-ar1-v1"),
            },
        )
        condition_reports[condition] = saved
    collection = {
        "schema": "sensetrace.predictive-horizon-experiment.v1",
        "conditions": list(selected_conditions),
        "condition_reports": {
            condition: {"output": record["output"], "manifest": record["manifest"]}
            for condition, record in condition_reports.items()
        },
        "passive_observation": True,
        "claim_boundary": "synthetic trajectory-analysis validation; no physical DRAM or model-inference claim",
    }
    _immutable_json(root / "experiment.json", collection)
    return collection
