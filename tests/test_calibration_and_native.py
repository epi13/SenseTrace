from __future__ import annotations

from collections import Counter

import numpy as np

from sensetrace.acquisition.commodity import CommodityDramBackend
from sensetrace.acquisition.native import NativeMeasurementKernel
from sensetrace.acquisition.synthetic import SyntheticBackend
from sensetrace.calibration import run_phase0_calibration
from sensetrace.config import validate_config
from sensetrace.metrics import max_statistic, monte_carlo_permutation_test
from sensetrace.splits import phase1a_split_hierarchy


def _calibration_config() -> dict:
    return validate_config(
        {
            "experiment": {"name": "calibration-test", "seed": 9},
            "data": {"target_balance": 0.5, "samples": 40, "trace_length": 32},
            "controls": {
                "injected_weak_signal": {"amplitude_sigma": 1.0, "start_index": 8, "width": 4}
            },
            "splits": {
                "primary": {
                    "group_keys": [
                        "synthetic_dataset_id",
                        "synthetic_session_id",
                        "synthetic_location_id",
                    ],
                    "train_fraction": 0.6,
                    "validation_fraction": 0.2,
                    "test_fraction": 0.2,
                }
            },
            "models": {
                "logistic_regression": {"enabled": True},
                "boosted_trees": {"enabled": False},
                "tiny_mlp": {"enabled": False},
                "tiny_cnn": {"enabled": False},
            },
            "training": {"seeds": [11], "epochs": 3, "early_stopping_patience": 1},
            "acquisition": {
                "shard_target_mb": 1,
                "max_samples_per_shard": 40,
                "observations_per_location": 4,
            },
            "calibration": {
                "alpha": 0.05,
                "balance_modes": ["global_balance_only", "group_stratified_balance"],
                "permutation_repetitions": 2,
                "minimum_injected_detection_rate": 0.0,
            },
        }
    )


def test_calibration_materializes_independent_ensembles_and_freezes_gate(tmp_path):
    report = run_phase0_calibration(
        _calibration_config(),
        tmp_path,
        null_replicates=2,
        shuffled_replicates=2,
        injected_replicates=2,
        gate_validation_replicates=2,
    )
    assert report["protocol_version"] == "phase0-protocol-v1"
    assert report["acceptance"]["decision_rule_frozen"]
    assert report["counts"]["calibration_null_replicates"] == 4
    assert report["counts"]["fresh_gate_validation_null_replicates"] == 4
    assert len(set(report["fresh_gate_validation"]["fresh_dataset_fingerprints"])) == 4
    assert report["empirical_null"]["max_statistic_distribution"]
    assert report["permutation_tests"][0]["strata_keys"] == ["synthetic_dataset_id"]


def test_synthetic_group_balance_and_seed_provenance():
    samples = list(
        SyntheticBackend(
            count=40,
            trace_length=16,
            seed=1,
            acquisition_seed=10,
            label_seed=11,
            trace_seed=12,
            balance_mode="group_stratified_balance",
        ).samples()
    )
    for start in range(0, 40, 4):
        assert Counter(sample.label for sample in samples[start : start + 4]) == {0: 2, 1: 2}
    assert len({sample.metadata["synthetic_dataset_id"] for sample in samples}) == 1
    assert samples[0].metadata["acquisition_seed"] == 10
    assert samples[0].metadata["label_seed"] == 11
    assert samples[0].metadata["trace_seed"] == 12


def test_max_statistic_and_permutation_preserve_strata():
    statistic, component = max_statistic({"logistic.ba": 0.52, "tree.auroc": 0.54})
    assert statistic == 0.040000000000000036
    assert component == "tree.auroc"
    labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    metadata = {"location_id": np.asarray(["a", "a", "b", "b"])}
    result = monte_carlo_permutation_test(
        labels,
        metadata,
        strata_keys=["location_id"],
        observed_statistic=0.5,
        evaluator=lambda candidate: float(candidate[0]),
        repetitions=3,
        seed=2,
    )
    assert result["strata_count"] == 2
    assert result["repetitions"] == 3


def test_paired_backend_balances_each_location_and_exposes_native_provenance():
    backend = CommodityDramBackend(
        count=48,
        location_count=3,
        trials_per_location=16,
        trace_length=4,
        word_count=8,
        lock_memory=False,
        cache_control="none",
        use_native_kernel=False,
    )
    try:
        samples = list(backend.samples())
    finally:
        backend.close()
    for start in range(0, 48, 16):
        assert Counter(sample.label for sample in samples[start : start + 16]) == {0: 8, 1: 8}
    metadata = {
        key: np.asarray([sample.metadata[key] for sample in samples]) for key in samples[0].metadata
    }
    hierarchy = phase1a_split_hierarchy(metadata, dataset_fingerprint="test", seed=3)
    assert hierarchy["A_repeated_trial_holdout"]["status"] == "available"
    assert hierarchy["B_unseen_location"]["status"] == "available"
    assert samples[0].metadata["label_semantics"] == "target bit equals label"


def test_native_kernel_is_optional_but_has_serialized_contract():
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        return
    address = kernel.calibration_address()
    cached = kernel.measure_cached(address, 8)
    flushed = kernel.measure_flushed(address, 8) if kernel.supports_clflush else np.asarray([])
    assert len(cached) == 8
    if len(flushed):
        assert len(flushed) == 8
    assert "RDTSC" in kernel.provenance()["timer_source"]
