from __future__ import annotations

import numpy as np
import pytest

from sensetrace.acquisition.commodity import CommodityDramBackend
from sensetrace.acquisition.primitive import TimingPerturbationCalibration
from sensetrace.config import validate_config
from sensetrace.datasets import combine_datasets, write_dataset_manifest
from sensetrace.errors import ConfigError
from sensetrace.phase1a import _analyze_condition, run_phase1a_campaign
from sensetrace.protocol import (
    phase1a_commodity_baseline_protocol,
    phase1a_commodity_baseline_protocol_hash,
)
from sensetrace.splits import partition_indices, phase1a_split_hierarchy
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


def test_same_boot_sessions_leave_cross_boot_split_unavailable(tmp_path):
    config = _campaign_config()
    protocol = phase1a_commodity_baseline_protocol(config)
    protocol_hash = phase1a_commodity_baseline_protocol_hash(config)
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
                protocol_identity=protocol["version"],
                protocol_hash=protocol_hash,
            )
        source = tmp_path / session_id
        writer = ShardWriter(source, max_samples_per_shard=8)
        session_ledger = backend.session_provenance()
        session_ledger.update(
            {
                "protocol_identity": protocol["version"],
                "protocol_hash": protocol_hash,
                "acquisition_scope": "physical Phase 1A commodity baseline",
            }
        )
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
                dataset_purpose="physical_phase1a",
                protocol_identity=protocol["version"],
                protocol_hash=protocol_hash,
                provenance={
                    "protocol_identity": protocol["version"],
                    "protocol_hash": protocol_hash,
                    "artificial_timing_perturbation": {
                        "allowed": False,
                        "timing_perturbation_cycles": 0,
                        "timing_perturbation_label": 1,
                        "label_correlated": False,
                        "applied": False,
                        "calibration_namespace": "forbidden",
                        "physical_phase1a_forbidden": True,
                    },
                },
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
    assert all(
        analyses[name]["status"] == "available" for name in analyses if name != "E_unseen_boot"
    )
    assert analyses["E_unseen_boot"]["status"] == "unavailable"
    assert "insufficient independent OS boot groups" in analyses["E_unseen_boot"]["reason"]
    assert result["analysis_summary"]["all_available_splits_evaluated_independently"] is True
    assert all("models" in analyses[name] for name in analyses if analyses[name]["available"])
    assert all(
        "paired_statistics" in analyses[name] for name in analyses if analyses[name]["available"]
    )
    assert len(manifest["acquisition_sessions"]) == 4
    assert len({item["acquisition_session_id"] for item in manifest["acquisition_sessions"]}) == 4


def test_multiple_genuine_boots_make_e_available_and_keep_boots_disjoint():
    all_samples = []
    for boot_index in range(4):
        backend = CommodityDramBackend(
            count=8,
            location_count=1,
            trials_per_location=8,
            trace_length=32,
            word_count=8,
            lock_memory=False,
            cache_control="none",
            use_native_kernel=False,
            acquisition_session_id=f"boot-session-{boot_index}",
            session_index=boot_index,
        )
        try:
            samples = list(backend.samples())
        finally:
            backend.close()
        for sample in samples:
            sample.metadata["boot_id"] = f"genuine-boot-{boot_index}"
        all_samples.extend(samples)
    metadata = {
        key: np.asarray([sample.metadata[key] for sample in all_samples])
        for key in all_samples[0].metadata
    }
    hierarchy = phase1a_split_hierarchy(metadata, dataset_fingerprint="test", seed=19)
    record = hierarchy["E_unseen_boot"]
    assert record["status"] == "available"
    assert "OS boot IDs" in record["split"]["claim_boundary"]
    partitions = partition_indices(metadata, record["split"])
    boot_membership = {}
    for part, indices in partitions.items():
        for boot_id in np.unique(metadata["boot_id"][indices]):
            assert boot_id not in boot_membership
            boot_membership[str(boot_id)] = part
    assert len(boot_membership) == 4


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


def test_physical_phase1a_rejects_artificial_timing_perturbation_before_run_creation(tmp_path):
    for field, value in (
        ("timing_perturbation_cycles", 32),
        ("timing_perturbation_label", 0),
        ("calibration_namespace", "legacy"),
    ):
        config = _campaign_config()
        config["phase1a"][field] = value
        run_id = f"contaminated-physical-{field}"
        with pytest.raises(ConfigError, match="physical Phase 1A forbids"):
            run_phase1a_campaign(
                config,
                tmp_path,
                phase0_report={"acceptance": {"phase1_gate": True}},
                run_id=run_id,
            )
        assert not (tmp_path / run_id).exists()


def test_normal_zero_perturbation_remains_physical_and_unlabeled():
    backend = CommodityDramBackend(
        count=4,
        location_count=1,
        trials_per_location=4,
        trace_length=8,
        word_count=4,
        lock_memory=False,
        cache_control="none",
        use_native_kernel=False,
    )
    try:
        sample = next(backend.samples())
    finally:
        backend.close()
    assert sample.metadata["timing_perturbation_cycles"] == 0
    assert sample.metadata["timing_perturbation_label"] == 1
    assert sample.metadata["timing_perturbation_applied"] is False
    assert sample.metadata["artificial_timing_perturbation_allowed"] is False


def test_explicit_calibration_context_is_the_only_backend_delay_path():
    for legacy_fields in (
        {"timing_perturbation_cycles": 1},
        {"timing_perturbation_label": 0},
        {"calibration_namespace": "legacy"},
    ):
        with pytest.raises(ValueError, match="calibration-only"):
            CommodityDramBackend(
                count=4,
                location_count=1,
                trials_per_location=4,
                trace_length=8,
                word_count=4,
                lock_memory=False,
                cache_control="none",
                use_native_kernel=False,
                **legacy_fields,
            )
    backend = CommodityDramBackend(
        count=4,
        location_count=1,
        trials_per_location=4,
        trace_length=8,
        word_count=4,
        lock_memory=False,
        cache_control="none",
        use_native_kernel=False,
        calibration_context=TimingPerturbationCalibration("test-calibration", 1),
    )
    try:
        sample = next(backend.samples())
    finally:
        backend.close()
    assert sample.metadata["timing_perturbation_cycles"] == 1
    assert sample.metadata["calibration_namespace"] == "test-calibration"
    assert sample.metadata["artificial_timing_perturbation_allowed"] is True


def test_frozen_baseline_protocol_forbids_artificial_perturbation():
    protocol = phase1a_commodity_baseline_protocol(_campaign_config())
    assert protocol["artificial_timing_perturbation"] == {
        "allowed": False,
        "timing_perturbation_cycles": 0,
        "label_correlated": False,
        "calibration_namespace": "forbidden",
        "enforcement": (
            "physical Phase 1A rejects nonzero artificial timing cycles, non-default "
            "perturbation labels, and calibration namespaces before acquisition"
        ),
    }
