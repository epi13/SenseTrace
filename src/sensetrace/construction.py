"""Construction-preserving nulls for the timing-delta horizon result.

The historical horizon experiment used states ``[x_t, x_t - x_(t-1)]`` and
labelled a row with the future state's delta sign.  This module keeps that
production representation, but generates independent raw ``x`` observations
before rebuilding the state, target, split, and diagnostics.  It therefore
tests the feature--target construction itself rather than replacing the
target with an independent label.
"""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .acquisition.base import Sample
from .errors import IntegrityError, SchemaError
from .hashing import sha256_bytes, sha256_json
from .horizon import (
    ForecastPairs,
    Horizon,
    StateTrajectory,
    TargetSpec,
    _git_commit,
    _immutable_json,
    _target_index,
    build_horizon_split,
    evaluate_horizon_curve,
    numeric_metadata_matrix,
)
from .phase1a import _backend_from_config
from .trajectory import build_real_controls, sample_to_trajectory

ZeroPolicy = Literal["zero_is_positive", "zero_is_negative", "exclude"]
TimingSemantics = Literal["future_adjacent_delta_sign", "current_to_future_delta_sign"]

_SIGN_LABELS = (-1, 0, 1)
_SIMPLE_MODELS = (
    "reverse_delta_sign",
    "sign_transition",
    "training_median_current_level",
    "empirical_cdf",
    "current_level_logistic",
    "current_delta_logistic",
)
_DEFAULT_MODELS = (
    "majority",
    "random",
    "shuffled_labels",
    *_SIMPLE_MODELS,
    "linear_logistic",
    "nearest_neighbor",
)


@dataclass(frozen=True)
class TimingTargetDefinition:
    """Make the two possible timing target indexings explicit."""

    semantics: TimingSemantics
    zero_policy: ZeroPolicy = "zero_is_positive"

    def __post_init__(self) -> None:
        if self.semantics not in {
            "future_adjacent_delta_sign",
            "current_to_future_delta_sign",
        }:
            raise SchemaError(f"unsupported timing target semantics {self.semantics!r}")
        if self.zero_policy not in {"zero_is_positive", "zero_is_negative", "exclude"}:
            raise SchemaError(f"unsupported timing target zero policy {self.zero_policy!r}")

    def as_dict(self) -> dict[str, str]:
        return {"semantics": self.semantics, "zero_policy": self.zero_policy}

    @property
    def name(self) -> str:
        return self.semantics


def _raw_values(trajectory: StateTrajectory) -> np.ndarray:
    if trajectory.states.shape[1] < 2:
        raise SchemaError("timing construction controls require [level, causal_delta] states")
    values = np.asarray(trajectory.states[:, 0], dtype=np.float64)
    expected_delta = np.zeros(len(values), dtype=np.float64)
    expected_delta[1:] = np.diff(values)
    if not np.allclose(trajectory.states[:, 1], expected_delta, rtol=0.0, atol=1e-5):
        raise SchemaError(
            f"trajectory {trajectory.trajectory_id!r} does not contain the production causal first difference"
        )
    return values


def _label_from_delta(delta: float, zero_policy: ZeroPolicy) -> int | None:
    if delta == 0.0:
        if zero_policy == "exclude":
            return None
        return 1 if zero_policy == "zero_is_positive" else 0
    return int(delta > 0.0)


def build_timing_pairs(
    trajectories: Sequence[StateTrajectory],
    horizon: Horizon,
    definition: TimingTargetDefinition,
) -> ForecastPairs:
    """Build A or B timing targets from raw values with explicit tie handling.

    A is ``sign(x[t+h] - x[t+h-1])``.  B is ``sign(x[t+h] - x[t])``.  For
    ``zero_policy=exclude`` tied target rows are omitted before splitting; all
    other policies retain every valid origin/target pair.
    """

    if not trajectories:
        raise SchemaError("at least one timing trajectory is required")
    source_fingerprint = sha256_json(
        {
            "trajectories": [trajectory.fingerprint() for trajectory in trajectories],
            "construction": definition.as_dict(),
        }
    )
    target_spec = TargetSpec(
        name=definition.name,
        kind="binary",
        state_index=1,
        threshold=0.0,
        positive_if="ge" if definition.zero_policy == "zero_is_positive" else "gt",
    )
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    pair_ids: list[str] = []
    metadata: dict[str, list[Any]] = {
        "trajectory_id": [],
        "origin_index": [],
        "target_index": [],
        "origin_position": [],
        "target_position": [],
        "target_delta": [],
        "target_tied": [],
    }
    for trajectory in trajectories:
        raw = _raw_values(trajectory)
        for origin_index in range(len(raw)):
            target_index = _target_index(trajectory, origin_index, horizon)
            if target_index is None or target_index <= origin_index:
                continue
            target_delta = (
                raw[target_index] - raw[target_index - 1]
                if definition.semantics == "future_adjacent_delta_sign"
                else raw[target_index] - raw[origin_index]
            )
            label = _label_from_delta(float(target_delta), definition.zero_policy)
            if label is None:
                continue
            feature_rows.append(np.asarray(trajectory.states[origin_index], dtype=np.float32).copy())
            labels.append(label)
            pair_ids.append(
                sha256_json(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "origin_index": origin_index,
                        "target_index": target_index,
                        "horizon": horizon.as_dict(),
                        "target": definition.as_dict(),
                    }
                )
            )
            metadata["trajectory_id"].append(trajectory.trajectory_id)
            metadata["origin_index"].append(origin_index)
            metadata["target_index"].append(target_index)
            assert trajectory.positions is not None
            metadata["origin_position"].append(float(trajectory.positions[origin_index]))
            metadata["target_position"].append(float(trajectory.positions[target_index]))
            metadata["target_delta"].append(float(target_delta))
            metadata["target_tied"].append(bool(target_delta == 0.0))
    if not feature_rows:
        raise SchemaError(
            f"timing target {definition.as_dict()} has no valid present/future pairs"
        )
    rule = (
        "A: target=sign(raw[target_index]-raw[target_index-1])"
        if definition.semantics == "future_adjacent_delta_sign"
        else "B: target=sign(raw[target_index]-raw[origin_index])"
    )
    return ForecastPairs(
        horizon=horizon,
        target=target_spec,
        features=np.stack(feature_rows),
        targets=np.asarray(labels, dtype=np.uint8),
        pair_ids=np.asarray(pair_ids),
        metadata={key: np.asarray(value) for key, value in metadata.items()},
        source_fingerprint=source_fingerprint,
        alignment_rule=(
            f"{rule}; {definition.zero_policy}; "
            + (
                "tied target rows excluded"
                if definition.zero_policy == "exclude"
                else "tied target rows retained"
            )
        ),
    )


def trajectories_from_samples(samples: Sequence[Sample]) -> list[StateTrajectory]:
    """Use the same real acquisition adapter as the historical production path."""

    trajectories = [sample_to_trajectory(sample) for sample in samples]
    if len({item.trajectory_id for item in trajectories}) != len(trajectories):
        raise SchemaError("construction campaign source trajectories must have unique sample IDs")
    if len(trajectories) < 6:
        raise SchemaError("construction campaign requires at least six trajectories")
    return trajectories


def trajectories_from_raw_values(
    values: Sequence[np.ndarray],
    *,
    trajectory_ids: Sequence[str],
    condition: str,
) -> list[StateTrajectory]:
    if len(values) != len(trajectory_ids):
        raise SchemaError("raw timing values and trajectory IDs must have equal length")
    trajectories: list[StateTrajectory] = []
    for raw, trajectory_id in zip(values, trajectory_ids, strict=True):
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
            raise SchemaError("raw timing surrogate values must be finite 1-D arrays of length >=2")
        delta = np.zeros(len(array), dtype=np.float64)
        delta[1:] = np.diff(array)
        trajectories.append(
            StateTrajectory(
                trajectory_id=str(trajectory_id),
                states=np.column_stack((array, delta)).astype(np.float32),
                metadata={
                    "condition": condition,
                    "generator": "construction-preserving-timing-null-v1",
                    "trajectory_boundary": "one_complete_acquisition_sample",
                    "state_representation": "trace_level_and_causal_first_difference_v1",
                    "position_unit": "native_measurement_repetition",
                    "label_used": False,
                },
            )
        )
    return trajectories


def raw_order_shuffle(
    trajectories: Sequence[StateTrajectory], *, seed: int
) -> list[StateTrajectory]:
    """Shuffle raw observations within each trajectory, then rebuild states."""

    result: list[StateTrajectory] = []
    for offset, trajectory in enumerate(trajectories):
        rng = np.random.default_rng(seed + offset * 1_000_003)
        raw = _raw_values(trajectory).copy()
        rng.shuffle(raw)
        result.append(
            StateTrajectory(
                trajectory_id=trajectory.trajectory_id,
                states=np.column_stack((raw, np.r_[0.0, np.diff(raw)])).astype(np.float32),
                positions=trajectory.positions,
                metadata={
                    **dict(trajectory.metadata or {}),
                    "condition": "raw_order_shuffle",
                    "generator": "within-trajectory-raw-order-shuffle-v1",
                },
            )
        )
    return result


def generate_timing_surrogate(
    reference: Sequence[StateTrajectory],
    *,
    condition: str,
    seed: int,
    quantization_step: float = 1.0,
) -> list[StateTrajectory]:
    """Generate independent raw observations while retaining lengths and IDs."""

    if not reference:
        raise SchemaError("timing surrogate generation requires reference trajectories")
    pool = np.concatenate([_raw_values(item) for item in reference])
    if not np.isfinite(pool).all() or np.std(pool) <= 0:
        raise SchemaError("reference timing pool must have non-zero finite variation")
    rng = np.random.default_rng(seed)
    mean = float(np.mean(pool))
    scale = float(np.std(pool))
    raw_values: list[np.ndarray] = []
    for trajectory in reference:
        length = len(trajectory.states)
        if condition == "iid_continuous":
            values = rng.normal(mean, scale, size=length)
        elif condition == "iid_skewed":
            # Standardize a log-normal draw to the observed level/scale so the
            # null exercises skew without introducing a scale shortcut.
            skewed = rng.lognormal(mean=0.0, sigma=0.8, size=length)
            skewed_mean = float(np.mean(skewed))
            skewed_scale = float(np.std(skewed))
            values = mean + scale * (skewed - skewed_mean) / max(skewed_scale, 1e-12)
        elif condition in {"iid_quantized", "iid_discrete"}:
            values = rng.choice(pool, size=length, replace=True)
            step = float(quantization_step)
            if step <= 0 or not np.isfinite(step):
                raise SchemaError("quantization_step must be positive and finite")
            values = np.round(values / step) * step
        elif condition == "empirical_marginal":
            # This is a diagnostic surrogate: its generator is allowed to
            # condition on the complete observed marginal, while every
            # deployable baseline still fits on the training partition only.
            values = rng.choice(pool, size=length, replace=True)
        else:
            raise SchemaError(f"unsupported timing surrogate condition {condition!r}")
        raw_values.append(np.asarray(values, dtype=np.float64))
    return trajectories_from_raw_values(
        raw_values,
        trajectory_ids=[item.trajectory_id for item in reference],
        condition=condition,
    )


def _autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if lag <= 0 or len(values) <= lag:
        return None
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _lag_summary(per_trajectory: Sequence[dict[str, Any]], field: str, lag: int) -> dict[str, Any]:
    values = [item[field][str(lag)] for item in per_trajectory if item[field].get(str(lag)) is not None]
    if not values:
        return {"valid_trajectories": 0, "mean": None, "median": None, "sd": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "valid_trajectories": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def timing_diagnostics(
    trajectories: Sequence[StateTrajectory], *, max_lag: int = 8
) -> dict[str, Any]:
    """Report pooled and per-trajectory timing dependence without hiding heterogeneity."""

    per_trajectory: list[dict[str, Any]] = []
    transition = np.zeros((3, 3), dtype=np.int64)
    for trajectory in trajectories:
        raw = _raw_values(trajectory)
        delta = np.zeros(len(raw), dtype=np.float64)
        delta[1:] = np.diff(raw)
        signs = np.sign(delta).astype(np.int8)
        for current, future in zip(signs[:-1], signs[1:], strict=True):
            transition[_SIGN_LABELS.index(int(current)), _SIGN_LABELS.index(int(future))] += 1
        per_trajectory.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "length": int(len(raw)),
                "raw_mean": float(np.mean(raw)),
                "raw_sd": float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0,
                "tie_rate_raw_delta": float(np.mean(delta == 0.0)),
                "raw_autocorrelation": {
                    str(lag): _autocorrelation(raw, lag) for lag in range(1, max_lag + 1)
                },
                "delta_autocorrelation": {
                    str(lag): _autocorrelation(delta, lag) for lag in range(1, max_lag + 1)
                },
                "delta_sign_autocorrelation": {
                    str(lag): _autocorrelation(signs.astype(np.float64), lag)
                    for lag in range(1, max_lag + 1)
                },
            }
        )
    return {
        "schema": "sensetrace.timing-diagnostics.v1",
        "trajectory_count": len(trajectories),
        "lags": list(range(1, max_lag + 1)),
        "aggregation": "per-trajectory summaries plus across-trajectory distribution; no pooled row-only estimate",
        "per_trajectory": per_trajectory,
        "lag_summary": {
            field: {
                str(lag): _lag_summary(per_trajectory, field, lag)
                for lag in range(1, max_lag + 1)
            }
            for field in ("raw_autocorrelation", "delta_autocorrelation", "delta_sign_autocorrelation")
        },
        "delta_sign_transition_matrix": {
            "labels": list(_SIGN_LABELS),
            "counts": transition.tolist(),
            "row_labels": "current delta sign (-1, 0, +1)",
            "column_labels": "next delta sign (-1, 0, +1)",
            "zero_handling": "zero is an explicit state in diagnostics; target tie policy is recorded separately",
        },
    }


def save_source_trajectories(
    output: str | Path, trajectories: Sequence[StateTrajectory]
) -> dict[str, Any]:
    """Persist compact raw trajectories so the campaign can be rerun later."""

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    values = np.concatenate([_raw_values(item) for item in trajectories]).astype(np.float64)
    offsets = np.cumsum([0, *[len(_raw_values(item)) for item in trajectories]], dtype=np.int64)
    ids = np.asarray([item.trajectory_id for item in trajectories], dtype=str)
    npz_path = root / "raw_trajectories.npz"
    if not npz_path.exists():
        temporary = root / "raw_trajectories.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, raw_values=values, offsets=offsets, trajectory_ids=ids)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, npz_path)
    raw_hash = sha256_bytes(npz_path.read_bytes())
    existing_manifest: dict[str, Any] | None = None
    manifest_path = root / "raw_trajectories.json"
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_manifest = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            existing_manifest = None
    manifest = {
        "schema": "sensetrace.raw-timing-trajectories.v1",
        "path": npz_path.name,
        "sha256": raw_hash,
        "trajectory_count": len(trajectories),
        "trajectory_ids": ids.tolist(),
        "lengths": np.diff(offsets).tolist(),
        "representation": "raw timing values only; production states are rebuilt as [x, causal diff]",
        "sensitive_data": False,
        "created_at": (existing_manifest or {}).get("created_at", datetime.now(UTC).isoformat()),
    }
    _immutable_json(root / "raw_trajectories.json", manifest)
    return manifest


def _horizon_key(horizon: Horizon) -> str:
    return f"{horizon.unit}:{horizon.distance:g}:{horizon.alignment}"


def _config_value(config: Mapping[str, Any], path: str, default: Any) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def analyze_construction_conditions(
    reference: Sequence[StateTrajectory],
    config: Mapping[str, Any],
    output: str | Path,
    *,
    acquisition_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all construction controls over a supplied acquisition."""

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    construction = config.get("construction", {})
    if not isinstance(construction, Mapping):
        raise SchemaError("construction configuration must be a mapping")
    seed = int(_config_value(config, "experiment.seed", 1337))
    conditions = tuple(
        str(value)
        for value in construction.get(
            "conditions",
            [
                "real_observed",
                "iid_continuous",
                "iid_skewed",
                "iid_quantized",
                "empirical_marginal",
                "raw_order_shuffle",
            ],
        )
    )
    distances = construction.get("horizon_distances", [1, 2, 4, 8, 16])
    if not isinstance(distances, list) or not distances:
        raise SchemaError("construction.horizon_distances must be a non-empty list")
    unit = str(construction.get("horizon_unit", "native_measurement_repetition"))
    alignment = cast(Any, str(construction.get("horizon_alignment", "index")))
    horizons = [Horizon(distance=value, unit=unit, alignment=alignment) for value in distances]
    zero_policy = cast(ZeroPolicy, str(construction.get("zero_policy", "zero_is_positive")))
    definitions = [
        TimingTargetDefinition("future_adjacent_delta_sign", zero_policy),
        TimingTargetDefinition("current_to_future_delta_sign", zero_policy),
    ]
    models = tuple(str(value) for value in construction.get("models", _DEFAULT_MODELS))
    reporting = config.get("reporting", {})
    if not isinstance(reporting, Mapping):
        raise SchemaError("reporting configuration must be a mapping")
    bootstrap_repetitions = int(reporting.get("bootstrap_repetitions", 500))
    permutation_repetitions = int(reporting.get("permutation_repetitions", 2000))
    training_seeds = tuple(int(value) for value in _config_value(config, "training.seeds", [11, 23, 37]))
    source_manifest = save_source_trajectories(root, reference)
    base_metadata = {
        "schema": "sensetrace.construction-preserving-campaign.v1",
        "reference_source_fingerprint": sha256_json([item.fingerprint() for item in reference]),
        "source_artifact": source_manifest,
        "target_semantics": {
            "A_future_adjacent_difference": "sign(x[t+h] - x[t+h-1])",
            "B_current_to_future_difference": "sign(x[t+h] - x[t])",
        },
        "zero_policy": zero_policy,
        "feature_construction": "production [x_t, x_t-x_(t-1)] rebuilt from raw observations",
        "diagnostic_surrogate_policy": "empirical marginal uses the complete observed marginal for generation; model fitting remains train-only",
        "split_policy": "same trajectory IDs, whole-trajectory grouped holdout, no observations moved between partitions",
        "controls": {
            "construction_preserving": "independent raw values or within-trajectory raw-order shuffle, then production reconstruction",
            "label_shuffle": "retained legacy control; breaks shared-value feature-target relationship",
            "wrong_trajectory": "retained legacy control; breaks shared-value feature-target relationship",
            "circular_target": "retained evaluator permutation; breaks alignment while preserving group structure",
        },
        "reference_acquisition": dict(acquisition_record or {}),
    }
    _immutable_json(root / "campaign.json", base_metadata)
    condition_reports: dict[str, Any] = {}
    for condition_index, condition in enumerate(conditions):
        if condition == "real_observed":
            trajectories = list(reference)
            generator = "observed production acquisition"
        elif condition == "raw_order_shuffle":
            trajectories = raw_order_shuffle(reference, seed=seed + condition_index * 1009)
            generator = "within-trajectory raw-order shuffle followed by production reconstruction"
        else:
            trajectories = generate_timing_surrogate(
                reference,
                condition=condition,
                seed=seed + condition_index * 1009,
                quantization_step=float(construction.get("quantization_step", 1.0)),
            )
            generator = f"independent raw-value generator: {condition}"
        condition_dir = root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        _immutable_json(
            condition_dir / "condition.json",
            {
                "schema": "sensetrace.construction-condition.v1",
                "condition": condition,
                "generator": generator,
                "trajectory_ids": [item.trajectory_id for item in trajectories],
                "trajectory_lengths": [len(item.states) for item in trajectories],
                "source_trajectory_fingerprint": sha256_json([item.fingerprint() for item in trajectories]),
                "diagnostic_only_generation": condition in {"iid_continuous", "iid_skewed", "iid_quantized", "iid_discrete", "empirical_marginal"},
            },
        )
        diagnostics = timing_diagnostics(
            trajectories, max_lag=int(construction.get("diagnostic_max_lag", 8))
        )
        _immutable_json(condition_dir / "diagnostics.json", diagnostics)
        target_reports: dict[str, Any] = {}
        for target_index, definition in enumerate(definitions):
            pairs_by_horizon: dict[str, ForecastPairs] = {}
            splits_by_horizon: dict[str, dict[str, Any]] = {}
            metadata_by_horizon: dict[str, np.ndarray] = {}
            for horizon in horizons:
                key = _horizon_key(horizon)
                pairs = build_timing_pairs(trajectories, horizon, definition)
                pairs_by_horizon[key] = pairs
                splits_by_horizon[key] = build_horizon_split(
                    pairs,
                    seed=seed,
                    group_key="trajectory_id",
                    train_fraction=float(_config_value(config, "splits.primary.train_fraction", 0.7)),
                    validation_fraction=float(_config_value(config, "splits.primary.validation_fraction", 0.15)),
                    test_fraction=float(_config_value(config, "splits.primary.test_fraction", 0.15)),
                )
                metadata_by_horizon[key], _ = numeric_metadata_matrix(pairs, ["origin_position"])
            if definition.semantics == "future_adjacent_delta_sign":
                control_features, control_targets = build_real_controls(
                    pairs_by_horizon, seed=seed + target_index * 104729
                )
                selected_models = models
            else:
                # The B target has a meaningful shared-current-value baseline
                # at every horizon; same-state/reversed controls from the old
                # A adapter would not have the same target semantics.
                control_features, control_targets = None, None
                selected_models = tuple(
                    value
                    for value in models
                    if value
                    not in {
                        "temporally_shuffled_states",
                        "wrong_trajectory_pairing",
                        "reversed_alignment",
                        "same_state",
                    }
                )
            report = evaluate_horizon_curve(
                pairs_by_horizon,
                splits_by_horizon,
                model_names=selected_models,
                metadata_features=metadata_by_horizon,
                control_features=control_features,
                control_targets=control_targets,
                seeds=training_seeds,
                bootstrap_repetitions=bootstrap_repetitions,
                permutation_repetitions=permutation_repetitions,
                practical_balanced_accuracy=float(reporting.get("practical_balanced_accuracy", 0.55)),
                practical_continuous_skill=float(reporting.get("practical_continuous_skill", 0.05)),
                significance_alpha=float(reporting.get("significance_alpha", 0.05)),
                empirical_cdf_tie_policy=zero_policy,
            )
            report["construction_control"] = {
                "condition": condition,
                "generator": generator,
                "construction_preserving": True,
                "target_definition": definition.as_dict(),
                "reference_trajectory_ids_preserved": [
                    item.trajectory_id for item in trajectories
                ]
                == [item.trajectory_id for item in reference],
                "diagnostics_file": "../diagnostics.json",
                # ``target_dir`` is root/condition/target; resolve from that
                # artifact's directory rather than the condition directory.
                "raw_source_artifact": "../../raw_trajectories.npz",
            }
            target_dir = condition_dir / definition.name
            from .horizon import write_horizon_run

            saved_record = write_horizon_run(
                target_dir,
                report=report,
                config={**dict(config), "construction_target": definition.as_dict(), "condition": condition},
                source_fingerprint=pairs_by_horizon[next(iter(pairs_by_horizon))].source_fingerprint,
                split_fingerprints={key: value["split_fingerprint"] for key, value in splits_by_horizon.items()},
                splits=splits_by_horizon,
                run_metadata={
                    "condition": condition,
                    "generator": generator,
                    "target_definition": definition.as_dict(),
                    "execution_host": platform.node() or "unavailable",
                    "requested_node": _config_value(config, "run_metadata.node", "controller"),
                    "acquisition_session_id": (
                        dict(acquisition_record or {}).get("session_provenance", {}).get(
                            "acquisition_session_id", "unavailable"
                        )
                    ),
                    "raw_source_artifact_sha256": source_manifest["sha256"],
                    "construction_preserving": True,
                },
                claim_boundary=(
                    "commodity timing feature-target construction characterization; no temporal-memory, "
                    "hidden physical state, model-foresight, or precognition claim"
                ),
            )
            target_reports[definition.name] = saved_record
        condition_reports[condition] = {
            "condition": condition,
            "diagnostics": str(condition_dir / "diagnostics.json"),
            "targets": target_reports,
        }
    collection = {
        "schema": "sensetrace.construction-preserving-campaign-result.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "sensetrace_commit": _git_commit(),
        "conditions": list(conditions),
        "targets": [definition.name for definition in definitions],
        "condition_reports": condition_reports,
        "campaign": base_metadata,
        "claim_boundary": (
            "commodity timing feature-target construction characterization; no temporal-memory, "
            "hidden physical state, model-foresight, or precognition claim"
        ),
    }
    _immutable_json(root / "experiment.json", collection)
    return collection


def run_construction_falsification_experiment(
    config: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    """Acquire a fresh worker/controller trace and run the frozen campaign."""

    acquisition = config.get("acquisition", {})
    physical = config.get("phase1a", {})
    if not isinstance(acquisition, Mapping) or acquisition.get("backend") != "commodity":
        raise SchemaError("construction falsification requires acquisition.backend=commodity")
    if not isinstance(physical, Mapping) or physical.get("pattern", "single_bit") != "random_word":
        raise SchemaError("construction falsification requires phase1a.pattern=random_word")
    backend = _backend_from_config(dict(config))
    try:
        samples = list(backend.samples())
        session_provenance = backend.session_provenance()
    finally:
        backend.close()
    trajectories = trajectories_from_samples(samples)
    acquisition_record = {
        "schema": "sensetrace.construction-acquisition.v1",
        "backend": "commodity-dram",
        "sample_count": len(samples),
        "trajectory_count": len(trajectories),
        "session_provenance": session_provenance,
        "execution_host": platform.node() or "unavailable",
        "requested_node": _config_value(config, "run_metadata.node", "controller"),
        "sample_label_used_for_features": False,
        "sample_label_used_for_targets": False,
        "raw_artifact_policy": "compact raw timing trajectories are retained in campaign output; labels never serialized into state source",
    }
    root = Path(output)
    _immutable_json(root / "acquisition.json", acquisition_record)
    return analyze_construction_conditions(
        trajectories,
        config,
        root,
        acquisition_record=acquisition_record,
    )


def load_raw_source(path: str | Path) -> list[StateTrajectory]:
    """Load a saved raw source and rebuild production states for reanalysis."""

    root = Path(path)
    try:
        manifest = json.loads((root / "raw_trajectories.json").read_text(encoding="utf-8"))
        with np.load(root / manifest["path"], allow_pickle=False) as archive:
            raw = np.asarray(archive["raw_values"], dtype=np.float64)
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
            ids = np.asarray(archive["trajectory_ids"]).astype(str)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot load raw timing source under {root}") from exc
    if manifest.get("sha256") != sha256_bytes((root / manifest["path"]).read_bytes()):
        raise IntegrityError("raw timing source hash does not match its manifest")
    if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(raw):
        raise IntegrityError("raw timing source offsets are invalid")
    return trajectories_from_raw_values(
        [raw[offsets[index] : offsets[index + 1]] for index in range(len(ids))],
        trajectory_ids=ids.tolist(),
        condition="loaded_raw_source",
    )
