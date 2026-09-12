from __future__ import annotations

import json

import numpy as np
import pytest

from sensetrace.errors import IntegrityError, SchemaError
from sensetrace.horizon import (
    Horizon,
    StateTrajectory,
    TargetSpec,
    build_forecast_pairs,
    build_horizon_split,
    evaluate_horizon_curve,
    generate_synthetic_trajectories,
    load_horizon_run,
    numeric_metadata_matrix,
    run_synthetic_horizon_experiment,
    validate_horizon_alignment,
    write_horizon_run,
)


def _trajectories(count: int = 9) -> list[StateTrajectory]:
    return [
        StateTrajectory(
            trajectory_id=f"trajectory-{index}",
            states=np.asarray(
                [[index + time, 10 * index + time] for time in range(8)], dtype=np.float32
            ),
        )
        for index in range(count)
    ]


def test_index_horizon_uses_current_state_only_and_future_state_only_for_target():
    pairs = build_forecast_pairs(
        _trajectories(6),
        Horizon(2, unit="layer", alignment="index"),
        TargetSpec("future_sign", "binary", threshold=3.5),
    )
    assert pairs.row_count == 36
    assert pairs.metadata["origin_index"][0] == 0
    assert pairs.metadata["target_index"][0] == 2
    assert np.array_equal(pairs.features[0], np.asarray([0.0, 0.0], dtype=np.float32))
    assert pairs.targets[0] == 0
    assert validate_horizon_alignment(pairs)["status"] == "pass"


def test_position_alignment_selects_first_observed_position_at_or_after_target():
    trajectory = StateTrajectory(
        "irregular",
        np.arange(10, dtype=np.float32).reshape(5, 2),
        positions=np.asarray([0.0, 0.5, 2.0, 4.5, 7.0]),
    )
    pairs = build_forecast_pairs(
        [trajectory], Horizon(2.1, unit="wall_clock", alignment="position"), TargetSpec("value", "continuous")
    )
    assert pairs.metadata["origin_index"].tolist() == [0, 1, 2, 3]
    assert pairs.metadata["target_index"].tolist() == [3, 3, 3, 4]
    assert pairs.targets.tolist() == [6.0, 6.0, 6.0, 8.0]
    assert "first observed position" in pairs.alignment_rule


def test_horizon_split_is_whole_trajectory_and_rejects_cross_partition_tampering():
    pairs = build_forecast_pairs(_trajectories(), Horizon(1), TargetSpec("value", "continuous"))
    split = build_horizon_split(pairs, seed=22)
    audit = validate_horizon_alignment(pairs, split)
    assert audit["status"] == "pass"
    assert audit["whole_trajectory_holdout"] is True
    mutated = json.loads(json.dumps(split))
    moved = mutated["test_pair_ids"].pop()
    mutated["train_pair_ids"].append(moved)
    from sensetrace.horizon import fingerprint_horizon_split

    mutated["split_fingerprint"] = fingerprint_horizon_split(mutated)
    assert validate_horizon_alignment(pairs, mutated)["status"] == "fail"


def test_metadata_firewall_rejects_identity_and_future_fields():
    pairs = build_forecast_pairs(_trajectories(), Horizon(1), TargetSpec("value", "continuous"))
    with pytest.raises(SchemaError, match="identity or future-index"):
        numeric_metadata_matrix(pairs, ["trajectory_id"])
    with pytest.raises(SchemaError, match="identity or future-index"):
        numeric_metadata_matrix(pairs, ["target_position"])
    matrix, fields = numeric_metadata_matrix(pairs, ["origin_position"])
    assert matrix.shape == (pairs.row_count, 1)
    assert fields == ["origin_position"]


def test_predictable_and_null_controls_have_distinct_horizon_behavior():
    target = TargetSpec("future_sign", "binary")
    reports = {}
    for condition in ("predictable", "null"):
        trajectories = generate_synthetic_trajectories(
            condition=condition, trajectory_count=30, length=32, state_dim=3, seed=100
        )
        pairs_by_horizon = {}
        splits = {}
        metadata = {}
        for horizon in (Horizon(1), Horizon(8)):
            key = f"step:{horizon.distance:g}:index"
            pairs_by_horizon[key] = build_forecast_pairs(trajectories, horizon, target)
            splits[key] = build_horizon_split(pairs_by_horizon[key], seed=9)
            metadata[key] = np.asarray(
                pairs_by_horizon[key].metadata["origin_position"], dtype=np.float64
            )[:, None]
        reports[condition] = evaluate_horizon_curve(
            pairs_by_horizon,
            splits,
            model_names=["majority", "random", "shuffled_labels", "metadata_only", "linear_logistic"],
            metadata_features=metadata,
            seeds=[11],
            bootstrap_repetitions=20,
            permutation_repetitions=20,
        )
    predictable_short = reports["predictable"]["horizons"]["step:1:index"]["models"]["linear_logistic"]["runs"][0]["validation"]["balanced_accuracy"]
    predictable_long = reports["predictable"]["horizons"]["step:8:index"]["models"]["linear_logistic"]["runs"][0]["validation"]["balanced_accuracy"]
    null_short = reports["null"]["horizons"]["step:1:index"]["models"]["linear_logistic"]["runs"][0]["validation"]["balanced_accuracy"]
    assert predictable_short > 0.5
    assert predictable_short > predictable_long
    assert abs(null_short - 0.5) < 0.2


def test_continuous_future_state_target_reports_regression_skill():
    trajectories = generate_synthetic_trajectories(
        condition="predictable", trajectory_count=18, length=20, state_dim=2, seed=5
    )
    pairs = build_forecast_pairs(trajectories, Horizon(1), TargetSpec("future_value", "continuous"))
    key = "step:1:index"
    split = build_horizon_split(pairs, seed=3)
    metadata, _ = numeric_metadata_matrix(pairs, ["origin_position"])
    report = evaluate_horizon_curve(
        {key: pairs},
        {key: split},
        model_names=["majority", "random", "shuffled_labels", "metadata_only", "linear_ridge", "nearest_neighbor"],
        metadata_features={key: metadata},
        seeds=[11],
        bootstrap_repetitions=20,
        permutation_repetitions=20,
    )
    test = report["horizons"][key]["models"]
    assert test["linear_ridge"]["status"] == "evaluated"
    assert "skill_over_constant_mean" in report["useful_lead_summary"]["model_summaries"]["linear_ridge"]["curve"][0]
    assert test["linear_ridge"]["runs"][0]["test"]["rmse"] < test["majority"]["runs"][0]["test"]["rmse"]


def test_horizon_run_writes_reproducible_manifest_and_immutable_artifacts(tmp_path):
    config = {
        "experiment": {"name": "fixture", "seed": 4},
        "data": {"trajectories": 9, "length": 12, "state_dim": 2, "target_balance": 0.5},
        "horizon": {"distances": [1], "unit": "step", "alignment": "index"},
        "target": {"name": "future_sign", "kind": "binary"},
        "models": ["majority", "linear_logistic"],
        "training": {"seeds": [1]},
        "reporting": {"bootstrap_repetitions": 20, "permutation_repetitions": 20},
    }
    result = run_synthetic_horizon_experiment(config, tmp_path)
    manifest_path = tmp_path / "predictable" / "manifest.json"
    assert result["passive_observation"] is True
    manifest = json.loads(manifest_path.read_text())
    assert manifest["causal_policy"]["forecast_changes_target_pipeline"] is False
    assert (tmp_path / "predictable" / "splits.json").exists()
    loaded = load_horizon_run(tmp_path / "predictable")
    assert loaded["manifest"]["results_sha256"]
    with pytest.raises(IntegrityError, match="already differs"):
        write_horizon_run(
            tmp_path / "predictable",
            report={"changed": True},
            config=config,
            source_fingerprint="different",
            split_fingerprints={},
        )
