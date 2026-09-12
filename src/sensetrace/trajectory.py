"""Adapters from SenseTrace acquisition observations to ordered trajectories.

The adapter deliberately exposes only the ordered primitive-observation trace.
``Sample.label`` and label-bearing metadata never enter the trajectory state or
its audit metadata.  This makes a complete acquisition sample the natural
trajectory boundary while preserving the existing acquisition contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .acquisition.base import Sample
from .errors import SchemaError
from .horizon import ForecastPairs, StateTrajectory

_PROVENANCE_FIELDS = (
    "session_id",
    "acquisition_session_id",
    "boot_id",
    "allocation_id",
    "physical_allocation_id",
    "measurement_primitive",
    "cache_control_method",
    "physical_operation",
)


def sample_to_trajectory(sample: Sample) -> StateTrajectory:
    """Convert one complete SenseTrace measurement trace into a trajectory.

    State component 0 is the measured trace value.  Component 1 is the
    causal first difference, with the first position defined as zero.  The
    difference uses only the current and preceding observation, so a future
    state cannot enter the present-state feature vector.
    """

    metadata = dict(sample.metadata)
    sample_id = metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SchemaError("real trajectory adaptation requires a non-empty sample_id")
    trace = np.asarray(sample.trace, dtype=np.float64)
    if trace.ndim != 1 or len(trace) < 2:
        raise SchemaError("real trajectory adaptation requires a 1-D trace with >=2 observations")
    if not np.isfinite(trace).all():
        raise SchemaError("real trajectory traces must be finite")
    delta = np.empty(len(trace), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = np.diff(trace)
    states = np.column_stack((trace, delta)).astype(np.float32)

    audit_metadata: dict[str, Any] = {
        "trajectory_boundary": "one_complete_acquisition_sample",
        "state_representation": "trace_level_and_causal_first_difference_v1",
        "position_unit": "native_measurement_repetition",
        "label_used": False,
    }
    for field in _PROVENANCE_FIELDS:
        value = metadata.get(field)
        if value is not None:
            if not isinstance(value, (str, int, float, bool)):
                raise SchemaError(f"sample provenance field {field!r} must be scalar")
            audit_metadata[field] = value
    return StateTrajectory(trajectory_id=sample_id.strip(), states=states, metadata=audit_metadata)


def samples_to_trajectories(samples: Iterable[Sample]) -> list[StateTrajectory]:
    """Adapt a complete, ordered sample collection with unique IDs."""

    trajectories: list[StateTrajectory] = []
    seen: set[str] = set()
    for sample in samples:
        trajectory = sample_to_trajectory(sample)
        if trajectory.trajectory_id in seen:
            raise SchemaError(f"duplicate real trajectory ID {trajectory.trajectory_id!r}")
        seen.add(trajectory.trajectory_id)
        trajectories.append(trajectory)
    if len(trajectories) < 6:
        raise SchemaError("real horizon analysis requires at least six complete trajectories")
    return trajectories


def _group_rows(pairs: ForecastPairs) -> dict[str, np.ndarray]:
    groups = np.asarray(pairs.metadata["trajectory_id"]).astype(str)
    return {group: np.flatnonzero(groups == group) for group in np.unique(groups)}


def build_real_controls(
    pairs_by_horizon: Mapping[str, ForecastPairs],
    *,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Build declared negative controls without changing primary pair IDs.

    The returned feature controls retain the primary split and row metadata.
    They deliberately destroy one relationship at a time: within-trajectory
    temporal order, trajectory identity, or origin/target order.  The
    ``same_state`` target is intentionally non-temporal and is kept separate
    from inferential max-statistic families by the evaluator.
    """

    feature_controls: dict[str, dict[str, np.ndarray]] = {
        "temporally_shuffled_states": {},
        "wrong_trajectory_pairing": {},
        "reversed_alignment": {},
    }
    target_controls: dict[str, dict[str, np.ndarray]] = {
        "same_state": {},
        "reversed_alignment": {},
    }
    for horizon_offset, (key, pairs) in enumerate(pairs_by_horizon.items()):
        rng = np.random.default_rng(seed + horizon_offset * 7919)
        groups = _group_rows(pairs)
        temporal = pairs.features.astype(np.float64, copy=True)
        for indices in groups.values():
            temporal[indices] = temporal[indices][rng.permutation(len(indices))]

        group_names = sorted(groups)
        if len(group_names) < 2:
            raise SchemaError("wrong-trajectory control requires at least two trajectory groups")
        permutation = rng.permutation(len(group_names))
        if np.any(permutation == np.arange(len(group_names))):
            permutation = np.roll(np.arange(len(group_names)), 1)
        wrong = pairs.features.astype(np.float64, copy=True)
        origin_positions = np.asarray(pairs.metadata["origin_position"], dtype=np.float64)
        for destination_index, destination_group in enumerate(group_names):
            source_group = group_names[int(permutation[destination_index])]
            destination_rows = groups[destination_group]
            source_rows = groups[source_group]
            source_by_position = {
                float(origin_positions[index]): int(index) for index in source_rows
            }
            try:
                wrong[destination_rows] = np.asarray(
                    [pairs.features[source_by_position[float(origin_positions[index])]] for index in destination_rows]
                )
            except KeyError as exc:
                raise SchemaError(
                    "wrong-trajectory control requires matching origin positions"
                ) from exc

        reversed_features = pairs.features.astype(np.float64, copy=True)
        reversed_targets = pairs.targets.copy()
        for indices in groups.values():
            reversed_targets[indices] = reversed_targets[indices][::-1]

        feature_controls["temporally_shuffled_states"][key] = temporal
        feature_controls["wrong_trajectory_pairing"][key] = wrong
        feature_controls["reversed_alignment"][key] = reversed_features
        target_controls["same_state"][key] = pairs.target.extract(pairs.features.astype(np.float64))
        target_controls["reversed_alignment"][key] = reversed_targets
    return feature_controls, target_controls
