"""Adversarial firewall: multiboot-v2 validates against frozen protocol.

Three consistently wrong reports must never satisfy the gate merely by
agreeing with each other. Each report must match the authoritative frozen
v2 protocol in requested design, executed native warmup, witness policy,
multiplex telemetry, hashes, and genuine boots.
"""

from __future__ import annotations

import copy
from pathlib import Path

from sensetrace.characterization import (
    DEFAULT_NULL_STABILITY_RULE,
    is_native_warmup_compliant,
)
from sensetrace.config import load_config
from sensetrace.hashing import sha256_json
from sensetrace.multiboot import (
    MULTIBOOT_CANDIDATE_EVENT,
    MULTIBOOT_REQUIRED_WARMUP,
    combine_multiboot_reports,
    validate_report_against_frozen,
)

FROZEN_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "worker03-multiboot-scoped-perf-warmup.example.yaml"
)


def _frozen_config() -> dict:
    return load_config(str(FROZEN_CONFIG_PATH))


def _frozen_hash(frozen_config: dict) -> str:
    from sensetrace.characterization import characterization_protocol

    return sha256_json(characterization_protocol(frozen_config))


def _compliant_warmup_by_replicate() -> dict:
    required = dict(MULTIBOOT_REQUIRED_WARMUP)
    return {
        f"replicate-{index:04d}": {
            "enabled": True,
            "requested": dict(required),
            "page_touch": {
                "words_touched": 64,
                "bytes_touched": 512,
                "pattern_base_hex": "0x9e3779b97f4a7c15",
                "checksum_hex": "0x1234",
                "touch_method": "deterministic write plus read-back of every word",
            },
            "dummy_loads": {
                "requested": 64,
                "executed": 64,
                "path": "native_cached_load",
            },
            "status": "complete",
            "pmu_window": "none; warmup runs outside any operation-scoped perf window",
        }
        for index in range(3)
    }


def _compliant_report(boot_id: str, frozen_hash: str) -> dict:
    null_medians = [10.0, 10.5, 9.5]
    return {
        "protocol": {
            "version": "measurement-primitive-characterization-v3",
            "protocol_hash": frozen_hash,
            "sample_design": {
                "replicates": 3,
                "location_count": 4,
                "trials_per_location": 16,
                "weak_positive_control_cycles": [0, 32, 64, 128],
                "scoped_perf_event": MULTIBOOT_CANDIDATE_EVENT,
                "allocation_warmup": dict(MULTIBOOT_REQUIRED_WARMUP),
            },
            "witness": {
                "version": "ebpf-witness-protocol-v1",
                "requirement": "disabled",
                "requested_hooks": [],
            },
            "analysis": {"null_stability": dict(DEFAULT_NULL_STABILITY_RULE)},
        },
        "witness_evidence": {
            "requirement": "disabled",
            "collection": "not_collected",
            "session": None,
            "session_status": None,
        },
        "controls": {
            "allocation_warmup": {
                "design": dict(MULTIBOOT_REQUIRED_WARMUP),
                "by_replicate": _compliant_warmup_by_replicate(),
            },
            "boot_dependence": {"unique_boots": [boot_id]},
            "operation_scoped_perf_oracle": {
                "agreement": {
                    "status": "pass",
                    "agreement_count": 3,
                    "sample_count": 3,
                    "multiplex_veto": False,
                    "multiplex_telemetry_present": True,
                    "multiplex_telemetry_complete": True,
                    "paired_differences": [],
                },
                "null_stability": {
                    "status": "pass",
                    "completeness": {"status": "pass"},
                    "finite_value_validity": {
                        "status": "pass",
                        "raw_replicate_medians": list(null_medians),
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
        "decision_gate": {"outcome": "A_usable_auditable_primitive"},
    }


def _combine_with_frozen(reports: list[dict]) -> dict:
    return combine_multiboot_reports(reports, expected_boots=3, frozen_config=_frozen_config())


def test_compliant_reports_satisfy_frozen_gate():
    frozen = _frozen_config()
    frozen_hash = _frozen_hash(frozen)
    reports = [_compliant_report(f"boot-{index}", frozen_hash) for index in range(3)]
    for report in reports:
        assert (
            validate_report_against_frozen(
                report,
                frozen_config=frozen,
                frozen_characterization_hash=frozen_hash,
                expected_replicates=3,
            )
            == []
        )
    combined = _combine_with_frozen(reports)
    assert combined["frozen_protocol_validation"]["compliant"] is True
    assert combined["provenance_complete"] is True


def test_native_warmup_validator_accepts_compliant_and_rejects_fallback():
    required = dict(MULTIBOOT_REQUIRED_WARMUP)
    compliant, _ = is_native_warmup_compliant(
        _compliant_warmup_by_replicate()["replicate-0000"], required
    )
    assert compliant is True
    fallback = copy.deepcopy(_compliant_warmup_by_replicate()["replicate-0000"])
    fallback["status"] = "fallback_complete"
    fallback["dummy_loads"]["path"] = "python_read_fallback_no_native_kernel"
    compliant, reason = is_native_warmup_compliant(fallback, required)
    assert compliant is False
    assert "not native" in reason or "fallback" in reason or "complete" in reason
    missing_compliant, _ = is_native_warmup_compliant(None, required)
    assert missing_compliant is False


def _assert_fails_provenance(reports: list[dict], fragment: str) -> dict:
    combined = _combine_with_frozen(reports)
    assert combined["provenance_complete"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"
    assert "frozen" in combined["reason"]
    violations = [
        violation
        for boot in combined["frozen_protocol_validation"]["per_boot_violations"]
        for violation in boot["violations"]
    ]
    assert any(fragment in violation for violation in violations), violations
    return combined


def test_1_three_reports_with_same_wrong_warmup_fail():
    frozen_hash = _frozen_hash(_frozen_config())
    wrong = {"enabled": True, "touch_pages": True, "dummy_loads": 32}
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        report["protocol"]["sample_design"]["allocation_warmup"] = dict(wrong)
        for provenance in report["controls"]["allocation_warmup"]["by_replicate"].values():
            provenance["requested"] = dict(wrong)
            provenance["dummy_loads"] = {
                "requested": 32,
                "executed": 32,
                "path": "native_cached_load",
            }
        reports.append(report)
    _assert_fails_provenance(reports, "warmup")


def test_2_reports_claiming_64_but_python_fallback_fail():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        for provenance in report["controls"]["allocation_warmup"]["by_replicate"].values():
            provenance["status"] = "fallback_complete"
            provenance["dummy_loads"]["path"] = "python_read_fallback_no_native_kernel"
        reports.append(report)
    _assert_fails_provenance(reports, "complete")


def test_3_witness_optional_fails():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        report["protocol"]["witness"]["requirement"] = "optional"
        report["witness_evidence"]["requirement"] = "optional"
        reports.append(report)
    _assert_fails_provenance(reports, "witness")


def test_4_missing_witness_policy_fails():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        del report["protocol"]["witness"]
        del report["witness_evidence"]
        reports.append(report)
    _assert_fails_provenance(reports, "witness")


def test_5_same_wrong_pmu_event_fails():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        report["protocol"]["sample_design"]["scoped_perf_event"] = "cpu/cache-references/"
        reports.append(report)
    _assert_fails_provenance(reports, "scoped_perf_event")


def test_6_mixed_characterization_versions_fail():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = [_compliant_report(f"boot-{index}", frozen_hash) for index in range(3)]
    reports[2]["protocol"]["version"] = "measurement-primitive-characterization-v2"
    _assert_fails_provenance(reports, "characterization version")


def test_7_missing_executed_warmup_fails():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        del report["controls"]["allocation_warmup"]
        reports.append(report)
    _assert_fails_provenance(reports, "warmup")


def test_8_mixed_native_and_fallback_paths_fail():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = [_compliant_report(f"boot-{index}", frozen_hash) for index in range(3)]
    for provenance in reports[2]["controls"]["allocation_warmup"]["by_replicate"].values():
        provenance["status"] = "fallback_complete"
        provenance["dummy_loads"]["path"] = "python_read_fallback_after_native_failure"
    combined = _combine_with_frozen(reports)
    assert combined["provenance_complete"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"
    per_boot = combined["frozen_protocol_validation"]["per_boot_violations"]
    assert per_boot[0]["violations"] == []
    assert per_boot[1]["violations"] == []
    assert len(per_boot[2]["violations"]) > 0


def test_9_incomplete_multiplex_telemetry_fails():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = []
    for index in range(3):
        report = _compliant_report(f"boot-{index}", frozen_hash)
        agreement = report["controls"]["operation_scoped_perf_oracle"]["agreement"]
        del agreement["multiplex_telemetry_present"]
        agreement["multiplex_telemetry_complete"] = False
        reports.append(report)
    _assert_fails_provenance(reports, "multiplex")


def test_10_protocol_hash_disagreement_fails():
    frozen_hash = _frozen_hash(_frozen_config())
    reports = [_compliant_report(f"boot-{index}", frozen_hash) for index in range(3)]
    reports[2]["protocol"]["protocol_hash"] = "deadbeef" * 8
    _assert_fails_provenance(reports, "hash")


def test_11_reused_and_invalid_boot_ids_fail():
    frozen_hash = _frozen_hash(_frozen_config())
    reused = [_compliant_report("same-boot", frozen_hash) for _ in range(3)]
    combined = _combine_with_frozen(reused)
    assert combined["boots_distinct_and_genuine"] is False
    assert combined["provenance_complete"] is False
    assert combined["outcome"] == "C_primitive_unsuitable"
    invalid = [_compliant_report("unavailable", frozen_hash) for _ in range(3)]
    combined_invalid = _combine_with_frozen(invalid)
    assert combined_invalid["provenance_complete"] is False
    assert combined_invalid["outcome"] == "C_primitive_unsuitable"
