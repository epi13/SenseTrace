from __future__ import annotations

import numpy as np
import pytest

from sensetrace.cli import build_parser
from sensetrace.controlled_forecast import (
    CausalForecastInterface,
    _feature_batch,
    _fit_model,
    _record_from_journal,
    _synthetic_record,
    _target_values,
    validate_controlled_forecast_config,
)
from sensetrace.errors import SchemaError


def _config() -> dict[str, object]:
    return {
        "experiment": {"name": "controlled-test", "seed": 7},
        "data": {"target_balance": 0.5, "samples": 8, "trace_length": 32},
        "splits": {
            "primary": {
                "train_fraction": 0.7,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            }
        },
        "training": {"seeds": [11]},
        "reporting": {"ci_unit": "session_id"},
        "acquisition": {"backend": "commodity"},
        "phase1a": {
            "campaign_intent": "measurement_characterization",
            "protocol_version": "phase1a-commodity-baseline-v1",
            "measurement_primitive": "commodity-clflush-timed-load",
            "pattern": "random_word",
            "cache_control": "eviction_buffer",
            "operation": "memory_read",
            "timing_perturbation_cycles": 0,
            "timing_perturbation_label": 1,
            "trials_per_location": 4,
            "labels_per_location": 2,
            "session_count": 1,
        },
        "controlled_forecast": {
            "protocol_version": "controlled-predictive-state-v1",
            "sessions_per_cell": 3,
            "trajectories_per_session": 1,
            "baseline_repetitions": 4,
            "excitation_length": 2,
            "repetitions_per_phase": 2,
            "quiet_future_repetitions": 12,
            "future_block_length": 2,
            "history_length": 4,
            "phase_duration_us": 1,
            "eviction_bytes": 64,
            "word_count": 32,
            "excitation_families": ["read_pressure", "active_quiet", "sham", "passive"],
            "measurement_conditions": ["cached_preloaded", "clflush", "eviction", "timer_only"],
            "horizons": [1, 4],
            "primary": {
                "excitation_family": "read_pressure",
                "measurement_condition": "cached_preloaded",
                "target": "future_block_mean",
                "horizon": 1,
            },
        },
    }


def test_controlled_forecast_cli_and_config_are_versioned():
    args = build_parser().parse_args(["run", "controlled-forecast", "--stage", "confirmation"])
    assert args.run_command == "controlled-forecast"
    assert args.stage == "confirmation"
    validate_controlled_forecast_config(_config())


def test_causal_interface_rejects_future_and_identity_inputs():
    records = [_synthetic_record("partially_observed", index, np.random.default_rng(index)) for index in range(12)]
    config = _config()
    config["controlled_forecast"]["future_block_length"] = 2  # type: ignore[index]
    model = _fit_model("current_observation", "future_block_mean", 1, records, range(8), config)
    stream = CausalForecastInterface({1: model}, history_length=4)
    with pytest.raises(SchemaError, match="forbidden"):
        stream.update({"target_ticks": 100, "future_observation": 101}, {})
    stream.update({"target_ticks": 100}, {"workload_active": 0.0})
    with pytest.raises(SchemaError, match="declared quiet policy"):
        stream.forecast([1], "known_future_schedule")


def test_streaming_feature_replay_uses_only_origin_and_matches_offline_features():
    records = [_synthetic_record("partially_observed", index, np.random.default_rng(index)) for index in range(8)]
    record = records[0]
    model = _fit_model("arx_history", "future_timing_level", 1, records, range(6), _config())
    stream = CausalForecastInterface({1: model}, history_length=4)
    for position in range(record.origin_index + 1):
        stream.update(
            {
                "target_ticks": int(record.target_ticks[position]),
                "reference_ticks": int(record.reference_ticks[position]),
                "reference_available": True,
            },
            {"workload_active": float(record.workload_history[position])},
        )
    online = stream.forecast([1], "quiet")[1]
    offline = model.predict_batch(_feature_batch([record], 4))[0]
    assert online == pytest.approx(float(offline))
    values, valid = _target_values([record], "future_timing_level", 1, 2)
    assert valid.tolist() == [True]
    assert values[0] == pytest.approx(float(record.target_ticks[record.origin_index + 1]))


def test_journal_round_trip_preserves_raw_integer_channels():
    record = _synthetic_record("iid_quantized", 0, np.random.default_rng(3))
    decoded = _record_from_journal(record.as_journal_record())
    assert decoded.target_ticks.dtype == np.uint64
    assert decoded.reference_ticks.dtype == np.int64
    assert np.array_equal(decoded.target_ticks, record.target_ticks)
    assert decoded.origin_index == record.origin_index


def test_configuration_rejects_uncovered_horizon():
    config = _config()
    config["controlled_forecast"]["horizons"] = [20]  # type: ignore[index]
    with pytest.raises(SchemaError, match="cover the largest target block"):
        validate_controlled_forecast_config(config)
