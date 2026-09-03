"""Bounded genuine-multi-boot PMU characterization.

This is a characterization experiment, not a hidden-bit campaign. One small
scoped-PMU characterization is repeated across genuinely distinct OS boots;
a reboot is a grouping variable verified from ``/proc/.../boot_id``, never an
incremented software label.

Statistical firewall: the protocol below (candidate event, controls, null
rule, decision tree) is frozen and hashed *before* new measurements are
collected. The candidate set is a single event, ``cpu/cache-misses/``, so no
counter search or cherry-picking is possible. ``cpu/cache-references/`` is
known-available but out of scope for this gate and must not be substituted
post hoc.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .characterization import DEFAULT_NULL_STABILITY_RULE, _null_stability_analysis
from .hashing import sha256_json

MULTIBOOT_PROTOCOL_VERSION = "measurement-primitive-multiboot-v1"
MULTIBOOT_CANDIDATE_EVENT = "cpu/cache-misses/"
MULTIBOOT_DIAGNOSTIC_EVENT = "cpu/cache-references/"


def multiboot_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen multi-boot design and decision tree."""
    characterization = config.get("characterization", {})
    replicates = int(characterization.get("replicates", 3))
    boots = int(characterization.get("multiboot_boots", 3))
    return {
        "version": MULTIBOOT_PROTOCOL_VERSION,
        "candidate_event": MULTIBOOT_CANDIDATE_EVENT,
        "diagnostic_event_out_of_scope": MULTIBOOT_DIAGNOSTIC_EVENT,
        "candidate_policy": (
            "single predeclared event; no counter search; cache-references is "
            "diagnostic-only and cannot satisfy this gate"
        ),
        "sample_design": {
            "boots": boots,
            "replicates_per_boot": replicates,
            "location_count": int(characterization.get("location_count", 4)),
            "trials_per_location": int(characterization.get("trials_per_location", 16)),
            "trace_length": int(characterization.get("trace_length", 32)),
            "scoped_perf_event": characterization.get(
                "scoped_perf_event", MULTIBOOT_CANDIDATE_EVENT
            ),
        },
        "null_stability_rule": dict(DEFAULT_NULL_STABILITY_RULE),
        "decision_tree": {
            "A_usable_auditable_primitive": (
                "every boot shows 3/3 directional PMU agreement with no "
                "multiplexing, cross-boot null stability passes, and provenance "
                "(distinct genuine boot IDs, fresh sessions/allocations, protocol "
                "agreement) is complete"
            ),
            "B_observable_available_but_oracle_weak": (
                "directional contrast survives in every boot but cross-boot null "
                "stability or independence remains insufficient"
            ),
            "C_primitive_unsuitable": (
                "directional contrast fails in any boot, provenance is incomplete, "
                "or boot IDs are not genuinely distinct"
            ),
        },
        "claim_boundary": (
            "multi-boot characterization only; no hidden-bit inference; a cache-miss "
            "count is access-path evidence, not DRAM access or topology evidence"
        ),
    }


def multiboot_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(multiboot_protocol(config))


def _report_boot_id(report: dict[str, Any]) -> str:
    for key in ("boot_dependence", "controls"):
        _ = key
    boot_section = report.get("controls", {}).get("boot_dependence", {})
    boots = boot_section.get("unique_boots", [])
    if len(boots) == 1 and boots[0] not in {"", None, "unavailable", "unknown"}:
        return str(boots[0])
    return "unavailable"


def _null_replicate_medians(report: dict[str, Any]) -> list[float]:
    oracle = report.get("controls", {}).get("operation_scoped_perf_oracle", {})
    stability = oracle.get("null_stability", {})
    raw = stability.get("finite_value_validity", {}).get("raw_replicate_medians", [])
    values: list[float] = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def combine_multiboot_reports(
    reports: list[dict[str, Any]], *, expected_boots: int = 3
) -> dict[str, Any]:
    """Combine one characterization report per genuine boot into a decision."""
    protocol_hashes = {
        str(report.get("protocol", {}).get("protocol_hash", "unavailable")) for report in reports
    }
    boot_ids = [_report_boot_id(report) for report in reports]
    distinct_boots = sorted(set(boot_ids))
    genuine_boots = [boot for boot in distinct_boots if boot not in {"unavailable", "unknown", ""}]
    boots_distinct = len(genuine_boots) == len(reports) == expected_boots
    protocol_agreement = len(protocol_hashes) == 1 and "unavailable" not in protocol_hashes
    per_boot_agreement = []
    for report in reports:
        oracle = report.get("controls", {}).get("operation_scoped_perf_oracle", {})
        agreement = oracle.get("agreement", {})
        per_boot_agreement.append(
            {
                "boot_id": _report_boot_id(report),
                "status": agreement.get("status", "unavailable"),
                "agreement_count": agreement.get("agreement_count", 0),
                "sample_count": agreement.get("sample_count", 0),
                "multiplex_veto": agreement.get("multiplex_veto", None),
                "decision_outcome": report.get("decision_gate", {}).get("outcome", "unavailable"),
            }
        )
    all_medians: list[float] = []
    for report in reports:
        all_medians.extend(_null_replicate_medians(report))
    expected_replicates = expected_boots * 3
    cross_boot_records = [
        {"status": "complete", "sample_median_summary": {"median": value}} for value in all_medians
    ]
    cross_boot_stability = _null_stability_analysis(
        cross_boot_records,
        expected_replicates=expected_replicates,
        rule=dict(DEFAULT_NULL_STABILITY_RULE),
    )
    cross_boot_stability["value_definition"] = (
        "median of all per-operation raw cache-miss counts within each null "
        f"replicate, pooled across {len(reports)} genuine boots"
    )
    directional_every_boot = bool(per_boot_agreement) and all(
        item["status"] == "pass" and item["agreement_count"] == item["sample_count"] >= 3
        for item in per_boot_agreement
    )
    multiplex_clean = all(
        item.get("multiplex_veto") in {False, None} for item in per_boot_agreement
    )
    # None means legacy telemetry absent; new runs must carry an explicit False.
    provenance_complete = bool(
        boots_distinct and protocol_agreement and len(reports) == expected_boots
    )
    cross_boot_stable = cross_boot_stability.get("status") == "pass"
    if not provenance_complete or not directional_every_boot:
        outcome = "C_primitive_unsuitable"
        reason = (
            "boot provenance incomplete or directional contrast failed in at least one genuine boot"
        )
    elif not cross_boot_stable or not multiplex_clean:
        outcome = "B_observable_available_but_oracle_weak"
        reason = (
            "directional contrast survived every boot but cross-boot null stability "
            "or multiplex cleanliness remains insufficient"
        )
    else:
        outcome = "A_usable_auditable_primitive"
        reason = (
            "directional agreement in every genuine boot with cross-boot null "
            "stability and complete provenance"
        )
    return {
        "schema": "sensetrace.multiboot-characterization-report.v1",
        "protocol_version": MULTIBOOT_PROTOCOL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "expected_boots": expected_boots,
        "observed_reports": len(reports),
        "protocol_hashes": sorted(protocol_hashes),
        "protocol_agreement": protocol_agreement,
        "boot_ids": boot_ids,
        "distinct_genuine_boot_ids": genuine_boots,
        "boots_distinct_and_genuine": boots_distinct,
        "per_boot_agreement": per_boot_agreement,
        "directional_agreement_every_boot": directional_every_boot,
        "cross_boot_null_stability": cross_boot_stability,
        "cross_boot_null_stable": cross_boot_stable,
        "cross_boot_null_medians": all_medians,
        "provenance_complete": provenance_complete,
        "outcome": outcome,
        "reason": reason,
        "claim_boundary": (
            "multi-boot characterization only; no hidden-bit inference; cache-miss "
            "counts are access-path evidence, not DRAM access or topology evidence"
        ),
    }


def write_combined_report(reports: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    combined = combine_multiboot_reports(reports)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "multiboot-report.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return combined
