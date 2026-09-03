"""Frozen warmup-repeat firewall: first-touch control and multiboot-v2 gates."""

from __future__ import annotations

import numpy as np
import pytest

from sensetrace.acquisition.commodity import ControlledMemoryBuffer
from sensetrace.characterization import (
    acquisition_order_diagnostics,
    characterization_protocol,
    parse_allocation_warmup,
    run_allocation_warmup,
)
from sensetrace.config import validate_config
from sensetrace.errors import ConfigError
from sensetrace.hashing import sha256_json
from sensetrace.multiboot import (
    MULTIBOOT_REQUIRED_WARMUP,
    combine_multiboot_reports,
    multiboot_protocol,
)


def _base_config(**overrides) -> dict:
    config = validate_config(
        {
            "experiment": {"name": "warmup-test", "seed": 11},
            "data": {"target_balance": 0.5, "samples": 8, "trace_length": 32},
            "acquisition": {"backend": "commodity"},
            "phase1a": {
                "protocol_version": "phase1a-commodity-baseline-v1",
                "measurement_primitive": "commodity-clflush-timed-load",
                "location_count": 1,
                "trials_per_location": 8,
                "labels_per_location": 4,
                "cache_control": "none",
                "use_native_kernel": False,
                "require_native_kernel": False,
            },
        }
    )
    config.setdefault("characterization", {}).update(overrides.pop("characterization", {}))
    for key, value in overrides.items():
        config[key] = value
    return validate_config(config)


def test_warmup_defaults_to_disabled_legacy():
    assert parse_allocation_warmup({}) == {
        "enabled": False,
        "touch_pages": True,
        "dummy_loads": 0,
    }


def test_warmup_rejects_unknown_fields_and_empty_enabled():
    with pytest.raises(ValueError, match="unknown fields"):
        parse_allocation_warmup(
            {"characterization": {"allocation_warmup": {"enabled": True, "bogus_field": 1}}}
        )
    # enabled=True with defaults touch_pages=True is valid; empty work is rejected only
    # when both touch is off and no dummy loads are requested.
    with pytest.raises(ValueError, match="requires touch_pages or dummy_loads"):
        parse_allocation_warmup(
            {
                "characterization": {
                    "allocation_warmup": {
                        "enabled": True,
                        "touch_pages": False,
                        "dummy_loads": 0,
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="dummy_loads"):
        parse_allocation_warmup(
            {"characterization": {"allocation_warmup": {"enabled": True, "dummy_loads": -1}}}
        )


def test_warmup_touch_is_deterministic_and_complete():
    first = ControlledMemoryBuffer(16, lock_memory=False)
    second = ControlledMemoryBuffer(16, lock_memory=False)
    try:
        touch_first = first.warmup_touch()
        touch_second = second.warmup_touch()
        assert touch_first == touch_second
        assert touch_first["words_touched"] == 16
        assert touch_first["bytes_touched"] == 16 * 8
        # Every word was actually written: checksum is non-trivial and stable.
        assert touch_first["checksum_hex"] != hex(0)
    finally:
        first.close()
        second.close()


def test_run_allocation_warmup_disabled_is_explicit():
    buffer = ControlledMemoryBuffer(8, lock_memory=False)
    try:
        assert run_allocation_warmup(buffer, None, {"enabled": False}) == {
            "enabled": False,
            "status": "disabled",
        }
    finally:
        buffer.close()


def test_run_allocation_warmup_falls_back_without_native_kernel():
    buffer = ControlledMemoryBuffer(8, lock_memory=False)
    try:
        provenance = run_allocation_warmup(
            buffer, None, {"enabled": True, "touch_pages": True, "dummy_loads": 4}
        )
        assert provenance["status"] == "complete"
        assert provenance["page_touch"]["words_touched"] == 8
        assert provenance["dummy_loads"]["executed"] == 4
        assert provenance["dummy_loads"]["path"] == "python_read_fallback_no_native_kernel"
        assert "outside any operation-scoped perf window" in provenance["pmu_window"]
    finally:
        buffer.close()


def test_run_allocation_warmup_uses_native_path_when_available():
    class FakeKernel:
        def measure_cached(self, address, repetitions, extra_delay_cycles=0):
            assert repetitions == 5
            return np.zeros(repetitions)

    buffer = ControlledMemoryBuffer(8, lock_memory=False)
    try:
        provenance = run_allocation_warmup(
            buffer,
            FakeKernel(),  # type: ignore[arg-type]
            {"enabled": True, "touch_pages": True, "dummy_loads": 5},
        )
        assert provenance["dummy_loads"] == {
            "requested": 5,
            "executed": 5,
            "path": "native_cached_load",
        }
    finally:
        buffer.close()


def test_characterization_protocol_hashes_warmup_and_witness():
    disabled = characterization_protocol(_base_config())
    enabled = characterization_protocol(
        _base_config(
            characterization={
                "allocation_warmup": {"enabled": True, "touch_pages": True, "dummy_loads": 64}
            },
            witness={"requirement": "disabled", "hooks": []},
        )
    )
    assert disabled["version"] == "measurement-primitive-characterization-v3"
    assert disabled["sample_design"]["allocation_warmup"]["enabled"] is False
    assert enabled["sample_design"]["allocation_warmup"] == {
        "enabled": True,
        "touch_pages": True,
        "dummy_loads": 64,
    }
    assert enabled["witness"]["requirement"] == "disabled"
    assert enabled["witness"]["automatic_sample_veto"] is False
    assert sha256_json(disabled) != sha256_json(enabled)


def test_multiboot_v2_requires_frozen_warmup_witness_and_null_rule():
    valid = validate_config(
        {
            "experiment": {"name": "multiboot-warmup", "seed": 20260904},
            "data": {"target_balance": 0.5, "samples": 64, "trace_length": 32},
            "acquisition": {"backend": "commodity"},
            "phase1a": {
                "protocol_version": "phase1a-commodity-baseline-v1",
                "measurement_primitive": "commodity-clflush-timed-load",
                "location_count": 4,
                "trials_per_location": 16,
                "labels_per_location": 8,
                "cache_control": "clflush",
                "use_native_kernel": True,
                "require_native_kernel": True,
            },
            "characterization": {
                "replicates": 3,
                "multiboot_boots": 3,
                "location_count": 4,
                "trials_per_location": 16,
                "trace_length": 32,
                "weak_positive_control_cycles": [0, 32, 64, 128],
                "scoped_perf_event": "cpu/cache-misses/",
                "allocation_warmup": dict(MULTIBOOT_REQUIRED_WARMUP),
            },
            "witness": {"requirement": "disabled", "hooks": []},
        }
    )
    protocol = multiboot_protocol(valid)
    assert protocol["version"] == "measurement-primitive-multiboot-v2"
    assert protocol["sample_design"]["allocation_warmup"] == dict(MULTIBOOT_REQUIRED_WARMUP)
    assert protocol["witness"]["requirement"] == "disabled"

    # Warmup disabled must not satisfy the v2 gate.
    disabled = validate_config(
        {
            **valid,
            "characterization": {
                **valid["characterization"],
                "allocation_warmup": {"enabled": False},
            },
        }
    )
    with pytest.raises(ValueError, match="allocation_warmup"):
        multiboot_protocol(disabled)

    # Witness must stay disabled for this PMU gate.
    witnessed = validate_config({**valid, "witness": {"requirement": "optional"}})
    with pytest.raises(ValueError, match="witness.requirement"):
        multiboot_protocol(witnessed)

    # Null-rule retuning after seeing v1 instability is forbidden.
    retuned = validate_config(
        {
            **valid,
            "characterization": {
                **valid["characterization"],
                "null_stability": {"max_relative_deviation": 0.5},
            },
        }
    )
    with pytest.raises(ValueError, match="null rule"):
        multiboot_protocol(retuned)


def _multiboot_report(
    boot_id: str,
    warmup: object,
    protocol_hash: str = "ph-warmup",
):
    return {
        "protocol": {
            "protocol_hash": protocol_hash,
            "sample_design": {
                "replicates": 3,
                "scoped_perf_event": "cpu/cache-misses/",
                "allocation_warmup": warmup,
            },
            "witness": {"requirement": "disabled"},
        },
        "controls": {
            "boot_dependence": {"unique_boots": [boot_id]},
            "operation_scoped_perf_oracle": {
                "agreement": {
                    "status": "pass",
                    "agreement_count": 3,
                    "sample_count": 3,
                    "multiplex_veto": False,
                    "paired_differences": [],
                },
                "null_stability": {
                    "status": "pass",
                    "completeness": {"status": "pass"},
                    "finite_value_validity": {
                        "status": "pass",
                        "raw_replicate_medians": [10.0, 10.5, 9.5],
                    },
                    "stability": {
                        "status": "pass",
                        "center": 10.0,
                        "median_absolute_deviation": 0.5,
                    },
                },
                "stability_pass": True,
            },
        },
        "decision_gate": {"outcome": "B_observable_available_but_oracle_weak"},
    }


def test_multiboot_combine_fails_provenance_on_mixed_warmup():
    warmup = {"enabled": True, "touch_pages": True, "dummy_loads": 64}
    reports = [
        _multiboot_report("boot-0", warmup),
        _multiboot_report("boot-1", warmup),
        _multiboot_report("boot-2", {"enabled": False, "touch_pages": True, "dummy_loads": 0}),
    ]
    combined = combine_multiboot_reports(reports, expected_boots=3)
    assert combined["allocation_warmup_agreement"] is False
    assert combined["provenance_complete"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"


def test_config_validation_rejects_bad_warmup_and_witness():
    with pytest.raises(ConfigError, match="allocation_warmup"):
        validate_config(
            {
                "experiment": {"name": "bad", "seed": 1},
                "characterization": {"allocation_warmup": {"enabled": True, "dummy_loads": 99999}},
            }
        )
    with pytest.raises(ConfigError, match="witness.requirement"):
        validate_config(
            {
                "experiment": {"name": "bad", "seed": 1},
                "witness": {"requirement": "best_effort"},
            }
        )


def _diagnostic_record(replicate_id: str, order: int, median: float) -> dict:
    return {
        "replicate_id": replicate_id,
        "acquisition_order_index": order,
        "operation_scoped_perf": {
            "status": "complete",
            "raw_count_summary": {"median": median},
            "scaled_count_summary": {"median": median},
            "multiplexed_fraction": 0.0,
        },
    }


def test_acquisition_order_diagnostics_detects_order_zero_inflation_without_gating():
    records = {
        "null_control": [
            _diagnostic_record("replicate-0000", 0, 25.0),
            _diagnostic_record("replicate-0001", 1, 3.0),
        ],
        "cached_control": [
            _diagnostic_record("replicate-0000", 1, 4.0),
            _diagnostic_record("replicate-0001", 0, 22.0),
        ],
        "requested_clflush_control": [
            _diagnostic_record("replicate-0000", 2, 20.0),
            _diagnostic_record("replicate-0001", 2, 34.0),
        ],
    }
    diagnostics = acquisition_order_diagnostics(records)
    assert diagnostics["status"] == "diagnostic_only_no_gate_effect"
    assert diagnostics["summary_by_acquisition_order"]["0"]["count"] == 2
    # replicate-0000 order-0 (25.0) beats orders 1-2; replicate-0001 order-0
    # (22.0) does not beat the flushed 34.0 control.
    assert diagnostics["order_zero_highest"] == {
        "replicates_with_order_zero_highest": 1,
        "replicates_evaluated": 2,
    }
    assert (
        records["null_control"][0]["operation_scoped_perf"]["raw_count_summary"]["median"] == 25.0
    )


def test_end_to_end_characterization_retains_warmup_provenance(tmp_path, monkeypatch):
    import sensetrace.characterization as characterization

    class FakeKernel:
        supports_clflush = True

        def measure_cached(self, address, repetitions, extra_delay_cycles=0):
            return np.zeros(repetitions, dtype=np.float64)

        def measure_flushed(self, address, repetitions, extra_delay_cycles=0):
            return np.full(repetitions, 200.0, dtype=np.float64)

        def provenance(self):
            return {
                "implementation": "fake",
                "version": "fake-v1",
                "library": "fake",
                "library_sha256": "f" * 64,
                "timer_source": "fake",
                "clflush_supported": True,
                "raw_units": "fake",
                "guarantees": [],
                "limitations": ["fake kernel for warmup test"],
            }

    monkeypatch.setattr(
        characterization.NativeMeasurementKernel, "load", classmethod(lambda cls: FakeKernel())
    )
    config = validate_config(
        {
            "experiment": {"name": "warmup-e2e", "seed": 41},
            "data": {"target_balance": 0.5, "samples": 8, "trace_length": 8},
            "acquisition": {"backend": "commodity"},
            "phase1a": {
                "protocol_version": "phase1a-commodity-baseline-v1",
                "measurement_primitive": "commodity-clflush-timed-load",
                "location_count": 1,
                "trials_per_location": 4,
                "labels_per_location": 2,
                "word_count": 8,
                "lock_memory": False,
                "cache_control": "clflush",
                "eviction_bytes": 1024,
                "use_native_kernel": True,
                "require_native_kernel": True,
            },
            "characterization": {
                "replicates": 2,
                "location_count": 1,
                "trials_per_location": 4,
                "trace_length": 8,
                "weak_positive_control_cycles": [0],
                "allocation_warmup": {"enabled": True, "touch_pages": True, "dummy_loads": 4},
            },
            "witness": {"requirement": "disabled", "hooks": []},
        }
    )
    report = characterization.run_measurement_primitive_characterization(
        config, tmp_path, run_id="warmup-e2e"
    )
    assert report["status"] == "complete"
    assert report["protocol"]["sample_design"]["allocation_warmup"]["enabled"] is True
    assert report["protocol"]["witness"]["requirement"] == "disabled"
    by_replicate = report["controls"]["allocation_warmup"]["by_replicate"]
    assert set(by_replicate) == {"replicate-0000", "replicate-0001"}
    for provenance in by_replicate.values():
        assert provenance["enabled"] is True
        assert provenance["status"] == "complete"
        assert provenance["page_touch"]["words_touched"] >= 8
    # Acquisition order remains retained for cold-transient diagnostics.
    for summary in report["controls"]["by_control"].values():
        for record in summary["records"]:
            assert "acquisition_order_index" in record
