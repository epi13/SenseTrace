from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sensetrace.config import load_config, validate_config
from sensetrace.errors import SchemaError
from sensetrace.latent_system_state import (
    LATENT_SYSTEM_STATE_PROTOCOL_VERSION,
    _evaluate_ladder,
    _feature_batch,
    _record_from_journal,
    _synthetic_record,
    assert_feature_firewall,
    validate_latent_system_state_config,
)


def _config() -> dict[str, object]:
    return validate_config(load_config("configs/latent-system-state-worker03.example.yaml"))


def test_latent_protocol_is_separate_and_versioned():
    config = _config()
    assert (
        validate_latent_system_state_config(config)["latent_system_state"]["protocol_version"]
        == LATENT_SYSTEM_STATE_PROTOCOL_VERSION
    )
    assert config["latent_system_state"]["target_conditions"] == ["cached_preloaded", "timer_only"]  # type: ignore[index]


def test_channel_contract_declares_source_units_availability_and_eligibility():
    record = _synthetic_record("aliasing", 0, np.random.default_rng(10))
    specs = record.metadata["witness_provenance"]["channel_specs"]
    assert {
        "name",
        "raw_source",
        "availability",
        "causal_eligible",
        "units",
        "observer_tier",
    } <= set(specs[0])
    assert specs[0]["causal_eligible"] is True


def test_journal_round_trip_preserves_raw_channels_and_missing_values():
    record = _synthetic_record("leakage_trap", 1, np.random.default_rng(11))
    decoded = _record_from_journal(record.as_journal_record())
    assert decoded.target_ticks.dtype == np.uint64
    assert np.array_equal(decoded.target_ticks, record.target_ticks)
    assert not np.isnan(decoded.witness_values[record.origin_index + 1, 0])
    assert not bool(decoded.witness_causal_eligible[record.origin_index + 1, 0])


def test_post_origin_witness_values_are_not_feature_inputs():
    record = _synthetic_record("leakage_trap", 2, np.random.default_rng(12))
    changed = replace(record, witness_values=record.witness_values.copy())
    changed.witness_values[record.origin_index + 1 :, :] = 10_000_000.0
    first = _feature_batch([record], 1, 2, 4)
    second = _feature_batch([changed], 1, 2, 4)
    assert_feature_firewall(first)
    for name in first.features:
        assert np.array_equal(first.features[name], second.features[name], equal_nan=True)


def test_post_origin_causal_marking_fails_closed():
    record = _synthetic_record("iid", 3, np.random.default_rng(13))
    causal = record.witness_causal_eligible.copy()
    causal[record.origin_index + 1 :, :] = True
    invalid = replace(record, witness_causal_eligible=causal)
    with pytest.raises(SchemaError, match="post-origin witness"):
        invalid.validate()


def test_synthetic_aliasing_is_resolved_by_current_multichannel_state():
    records = [
        _synthetic_record("aliasing", index, np.random.default_rng(100 + index))
        for index in range(30)
    ]
    report = _evaluate_ladder(records, _config(), horizon=1)
    assert report["test_mse"]["current_multichannel"] < report["test_mse"]["current_primary"]
    assert report["primary_comparison"]["test_loss_improvement"] > 20.0


def test_synthetic_iid_does_not_create_large_witness_gain():
    records = [
        _synthetic_record("iid", index, np.random.default_rng(200 + index)) for index in range(30)
    ]
    report = _evaluate_ladder(records, _config(), horizon=1)
    assert abs(report["primary_comparison"]["test_loss_improvement"]) < 50.0


def test_synthetic_workload_effect_is_not_called_system_witness_value():
    records = [
        _synthetic_record("workload_only", index, np.random.default_rng(300 + index))
        for index in range(30)
    ]
    report = _evaluate_ladder(records, _config(), horizon=1)
    assert report["test_mse"]["current_plus_workload"] < report["test_mse"]["current_primary"]
    assert report["test_mse"]["current_multichannel"] > report["test_mse"]["current_plus_workload"]


def test_record_validation_requires_aligned_witness_matrix():
    record = _synthetic_record("iid", 4, np.random.default_rng(400))
    invalid = replace(record, witness_valid=np.zeros((len(record.target_ticks), 2), dtype=bool))
    with pytest.raises(SchemaError, match="witness channel"):
        invalid.validate()
