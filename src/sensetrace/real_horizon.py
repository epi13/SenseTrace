"""Passive horizon experiments over SenseTrace's real acquisition traces."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .errors import SchemaError
from .horizon import (
    ForecastPairs,
    Horizon,
    TargetKind,
    TargetSpec,
    _immutable_json,
    build_forecast_pairs,
    build_horizon_split,
    evaluate_horizon_curve,
    write_horizon_run,
)
from .phase1a import _backend_from_config
from .trajectory import build_real_controls, samples_to_trajectories


def _target_specs(config: Mapping[str, Any]) -> list[TargetSpec]:
    raw_targets = config.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SchemaError("real horizon configuration requires a non-empty targets list")
    specs: list[TargetSpec] = []
    names: set[str] = set()
    for raw in raw_targets:
        if not isinstance(raw, Mapping):
            raise SchemaError("real horizon target entries must be mappings")
        spec = TargetSpec(
            name=str(raw.get("name", "")),
            kind=cast(TargetKind, str(raw.get("kind", "continuous"))),
            state_index=int(raw.get("state_index", 0)),
            threshold=float(raw.get("threshold", 0.0)),
            positive_if=cast(Any, str(raw.get("positive_if", "ge"))),
        )
        if spec.name in names:
            raise SchemaError(f"duplicate real horizon target {spec.name!r}")
        names.add(spec.name)
        specs.append(spec)
    return specs


def _horizons(config: Mapping[str, Any]) -> list[Horizon]:
    horizon_config = config.get("horizon", {})
    if not isinstance(horizon_config, Mapping):
        raise SchemaError("real horizon configuration horizon must be a mapping")
    distances = horizon_config.get("distances", [1, 2, 4, 8, 16])
    if not isinstance(distances, list) or not distances:
        raise SchemaError("real horizon distances must be a non-empty list")
    unit = str(horizon_config.get("unit", "native_measurement_repetition"))
    alignment = cast(Any, str(horizon_config.get("alignment", "index")))
    return [Horizon(distance=value, unit=unit, alignment=alignment) for value in distances]


def run_real_trace_horizon_experiment(
    config: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Acquire fresh commodity traces and evaluate several future targets.

    The acquisition remains the existing safe ordinary-user-space commodity
    backend.  This is a measurement-trajectory experiment, not a hidden-bit,
    DRAM-origin, or model-inference experiment.
    """

    acquisition = config.get("acquisition", {})
    physical = config.get("phase1a", {})
    if not isinstance(acquisition, Mapping) or acquisition.get("backend") != "commodity":
        raise SchemaError("real trace horizon experiments require acquisition.backend=commodity")
    if not isinstance(physical, Mapping) or physical.get("pattern", "single_bit") != "random_word":
        raise SchemaError(
            "real trace horizon experiments require phase1a.pattern=random_word so labels are not encoded"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    backend = _backend_from_config(dict(config))
    try:
        samples = list(backend.samples())
        session_provenance = backend.session_provenance()
    finally:
        backend.close()
    trajectories = samples_to_trajectories(samples)
    horizons = _horizons(config)
    targets = _target_specs(config)
    split_config = config.get("splits", {}).get("primary", {})
    if not isinstance(split_config, Mapping):
        raise SchemaError("real horizon primary split must be a mapping")
    seed = int(config.get("experiment", {}).get("seed", 1337))
    horizon_config = config.get("horizon", {})
    configured_models = (
        horizon_config.get("models")
        if isinstance(horizon_config, Mapping) and "models" in horizon_config
        else config.get("models")
    )
    model_names = tuple(
        str(item)
        for item in configured_models or (
            "majority",
            "random",
            "shuffled_labels",
            "metadata_only",
            "temporally_shuffled_states",
            "wrong_trajectory_pairing",
            "reversed_alignment",
            "same_state",
            "linear_logistic",
            "linear_ridge",
            "nearest_neighbor",
        )
    )
    if isinstance(configured_models, Mapping):
        model_names = tuple(
            str(name)
            for name, options in configured_models.items()
            if isinstance(options, Mapping) and options.get("enabled", False)
        )
    if not model_names:
        raise SchemaError("real horizon configuration selected no models")
    training = config.get("training", {})
    seeds = tuple(int(item) for item in training.get("seeds", [11, 23, 37]))
    reporting = config.get("reporting", {})
    if not isinstance(reporting, Mapping):
        raise SchemaError("real horizon reporting must be a mapping")
    acquisition_record = {
        "schema": "sensetrace.real-horizon-acquisition.v1",
        "backend": "commodity-dram",
        "sample_count": len(samples),
        "trajectory_count": len(trajectories),
        "trajectory_boundary": "one_complete_acquisition_sample",
        "state_representation": "trace_level_and_causal_first_difference_v1",
        "sample_label_used_for_features": False,
        "sample_label_used_for_target": False,
        "session_provenance": session_provenance,
        "execution_host": platform.node() or "unavailable",
        "requested_node": config.get("run_metadata", {}).get("node", "controller"),
    }
    _immutable_json(root / "acquisition.json", acquisition_record)

    target_reports: dict[str, Any] = {}
    for target_index, target in enumerate(targets):
        pairs_by_horizon: dict[str, ForecastPairs] = {}
        splits_by_horizon: dict[str, dict[str, Any]] = {}
        metadata_by_horizon: dict[str, Any] = {}
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
            metadata_by_horizon[key], _ = _metadata_feature(pairs)
        control_features, control_targets = build_real_controls(
            pairs_by_horizon,
            seed=seed + target_index * 104729,
        )
        report = evaluate_horizon_curve(
            pairs_by_horizon,
            splits_by_horizon,
            model_names=model_names,
            metadata_features=metadata_by_horizon,
            control_features=control_features,
            control_targets=control_targets,
            seeds=seeds,
            bootstrap_repetitions=int(reporting.get("bootstrap_repetitions", 400)),
            permutation_repetitions=int(reporting.get("permutation_repetitions", 400)),
            practical_balanced_accuracy=float(reporting.get("practical_balanced_accuracy", 0.55)),
            practical_continuous_skill=float(reporting.get("practical_continuous_skill", 0.05)),
            significance_alpha=float(reporting.get("significance_alpha", 0.05)),
        )
        target_config = dict(config)
        target_config["target"] = target.as_dict()
        target_config["target_name"] = target.name
        target_dir = root / target.name
        saved = write_horizon_run(
            target_dir,
            report=report,
            config=target_config,
            source_fingerprint=pairs_by_horizon[next(iter(pairs_by_horizon))].source_fingerprint,
            split_fingerprints={key: value["split_fingerprint"] for key, value in splits_by_horizon.items()},
            splits=splits_by_horizon,
            run_metadata={
                "condition": "real_commodity_trace",
                "execution_host": platform.node() or "unavailable",
                "requested_node": config.get("run_metadata", {}).get("node", "controller"),
                "target_model": config.get("run_metadata", {}).get(
                    "target_model", "sensetrace-commodity-trace-v1"
                ),
                "target_name": target.name,
                "trajectory_boundary": "one_complete_acquisition_sample",
                "state_representation": "trace_level_and_causal_first_difference_v1",
                "sample_label_used_for_features": False,
                "acquisition_session_id": session_provenance.get("acquisition_session_id", "unavailable"),
            },
            claim_boundary=(
                "real SenseTrace commodity measurement-trajectory analysis; no hidden physical state, "
                "DRAM-origin, or model-inference claim"
            ),
        )
        target_reports[target.name] = saved
    collection = {
        "schema": "sensetrace.real-predictive-horizon-experiment.v1",
        "targets": list(target_reports),
        "target_reports": {
            target_name: {"output": record["output"], "manifest": record["manifest"]}
            for target_name, record in target_reports.items()
        },
        "acquisition": "existing CommodityDramBackend; labels and label metadata excluded from trajectory states/features",
        "passive_observation": True,
        "claim_boundary": (
            "real SenseTrace commodity measurement-trajectory analysis; no hidden physical state, "
            "DRAM-origin, or model-inference claim"
        ),
    }
    _immutable_json(root / "experiment.json", collection)
    return collection


def _metadata_feature(pairs: ForecastPairs) -> tuple[Any, list[str]]:
    """Use only the allowed origin position for the metadata-only baseline."""

    from .horizon import numeric_metadata_matrix

    return numeric_metadata_matrix(pairs, ["origin_position"])
