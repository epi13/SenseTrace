from __future__ import annotations

from sensetrace.acquisition.commodity import CommodityDramBackend
from sensetrace.config import validate_config
from sensetrace.datasets import combine_datasets, write_dataset_manifest
from sensetrace.phase1a import _analyze_condition, run_phase1a_campaign
from sensetrace.storage import ShardWriter, validate_all_shards


def _campaign_config() -> dict:
    return validate_config(
        {
            "experiment": {"name": "phase1a-campaign-test", "seed": 17},
            "data": {"target_balance": 0.5, "samples": 8, "trace_length": 32},
            "models": {
                "logistic_regression": {"enabled": True},
                "boosted_trees": {"enabled": False},
                "tiny_mlp": {"enabled": False},
                "tiny_cnn": {"enabled": False},
            },
            "training": {"seeds": [3], "epochs": 2, "early_stopping_patience": 1},
            "acquisition": {"shard_target_mb": 1, "max_samples_per_shard": 8},
            "phase1a": {
                "samples": 8,
                "trace_length": 32,
                "location_count": 1,
                "trials_per_location": 8,
                "labels_per_location": 4,
                "word_count": 8,
                "lock_memory": False,
                "cache_control": "none",
                "use_native_kernel": False,
                "require_native_kernel": False,
                "ci_unit": "acquisition_block",
            },
            "reporting": {"bootstrap_repetitions": 5, "paired_repetitions": 20},
        }
    )


def test_all_available_phase1a_splits_are_evaluated_and_strict_levels_are_visible(tmp_path):
    config = _campaign_config()
    source_dirs = []
    for session_index in range(4):
        session_id = f"session-{session_index}"
        backend = CommodityDramBackend(
            count=8,
            location_count=1,
            trials_per_location=8,
            trace_length=32,
            word_count=8,
            lock_memory=False,
            cache_control="none",
            use_native_kernel=False,
            acquisition_session_id=session_id,
            session_index=session_index,
        )
        source = tmp_path / session_id
        writer = ShardWriter(source, max_samples_per_shard=8)
        session_ledger = backend.session_provenance()
        for sample in backend.samples():
            writer.add(sample.trace, sample.label, sample.metadata)
        backend.close()
        writer.finalize()
        write_dataset_manifest(
            source,
            config=config,
            condition="paired_single_bit",
            shard_infos=validate_all_shards(source),
            label_stream_fingerprint=session_ledger["label_stream_fingerprint"],
            class_balance={"0": 4, "1": 4},
            acquisition_sessions=[session_ledger],
        )
        source_dirs.append(source)
    target = tmp_path / "combined"
    manifest = combine_datasets(
        source_dirs,
        target,
        config=config,
        condition="paired_single_bit",
        campaign_id="campaign-test",
    )
    result = _analyze_condition(target, manifest, config)
    analyses = result["split_analyses"]
    assert all(analyses[name]["status"] == "available" for name in analyses)
    assert result["analysis_summary"]["all_available_splits_evaluated_independently"] is True
    assert all("models" in analyses[name] for name in analyses)
    assert all("paired_statistics" in analyses[name] for name in analyses)
    assert len(manifest["acquisition_sessions"]) == 4
    assert len({item["acquisition_session_id"] for item in manifest["acquisition_sessions"]}) == 4


def test_phase1a_campaign_refuses_a_closed_phase0_gate(tmp_path):
    try:
        run_phase1a_campaign(
            _campaign_config(),
            tmp_path,
            phase0_report={"acceptance": {"phase1_gate": False}},
            run_id="closed-gate",
        )
    except RuntimeError as exc:
        assert "gate is not PASS" in str(exc)
    else:
        raise AssertionError("a closed Phase 0 gate must prevent Phase 1A acquisition")
    assert not (tmp_path / "closed-gate").exists()
