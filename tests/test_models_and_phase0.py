from __future__ import annotations

from sensetrace.config import validate_config
from sensetrace.phase0 import run_phase0


def _config(models):
    return validate_config(
        {
            "experiment": {"name": "phase0-test", "seed": 1337},
            "data": {"target_balance": 0.5, "samples": 1600, "trace_length": 128},
            "controls": {
                "injected_weak_signal": {"amplitude_sigma": 0.1, "start_index": 40, "width": 8}
            },
            "splits": {
                "primary": {
                    "group_keys": ["session_id", "device_id", "row_id", "cell_or_offset_id"],
                    "train_fraction": 0.7,
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
                }
            },
            "models": {
                name: {"enabled": name in models}
                for name in ["logistic_regression", "boosted_trees", "tiny_mlp", "tiny_cnn"]
            },
            "training": {
                "seeds": [11, 23],
                "epochs": 10,
                "early_stopping_patience": 2,
                "batch_size": 128,
            },
            "acquisition": {"shard_target_mb": 1, "max_samples_per_shard": 256},
        }
    )


def test_phase0_negative_controls_and_injected_signal(tmp_path):
    report = run_phase0(_config(["logistic_regression"]), tmp_path)
    assert report["acceptance"]["null_consistent_with_chance"]
    assert report["acceptance"]["shuffled_labels_consistent_with_chance"]
    assert report["acceptance"]["injected_signal_detected"]
    for condition in ["null", "injected", "shuffled"]:
        assert report["conditions"][condition]["split"]["split_strategy"] == "grouped"


def test_repeated_seed_metadata_is_present(tmp_path):
    report = run_phase0(_config(["logistic_regression"]), tmp_path)
    model = report["conditions"]["injected"]["models"]["logistic_regression"]
    assert model["seeds"] == [11, 23]
    assert model["summary"]["balanced_accuracy_std"] >= 0
    assert all("confidence_interval_95" in run for run in model["runs"])
