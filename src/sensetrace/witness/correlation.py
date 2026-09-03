"""Deterministic, uncertainty-aware sample/witness correlation."""

from __future__ import annotations

from typing import Any

from sensetrace.acquisition.probe_contract import ProbeSampleRecord

from .models import WitnessSession


def correlate_witness(
    samples: list[ProbeSampleRecord], session: WitnessSession
) -> list[dict[str, Any]]:
    """Attach events using inclusive, uncertainty-expanded sample windows.

    Events are never used to discard samples.  A stale session or an
    unavailable clock alignment fails closed instead of guessing an overlap.
    """

    results: list[dict[str, Any]] = []
    for sample in samples:
        request = sample.request
        if request.session_id != session.experiment_id:
            raise ValueError("stale witness/session data: experiment identities differ")
        base = {
            "sample_index": request.sample_index,
            "correlation_id": request.correlation_id,
            "observer_session_id": session.session_id,
            "clock_alignment": session.alignment.as_dict(),
            "sample_window": {
                "start_ns": sample.monotonic_start_ns,
                "end_ns": sample.monotonic_end_ns,
                "clock_domain": sample.clock_domain,
                "boundary_rule": "inclusive after bounded alignment and uncertainty expansion",
            },
            "sample_veto": False,
        }
        if session.status in {"unavailable", "failed"}:
            results.append({**base, "state": "witness_unavailable", "events": []})
            continue
        if session.alignment.status != "bounded" or session.alignment.offset_ns is None:
            results.append({**base, "state": "incomplete_witness", "events": []})
            continue
        uncertainty = session.alignment.uncertainty_ns or 0
        lower = sample.monotonic_start_ns - uncertainty
        upper = sample.monotonic_end_ns + uncertainty
        matched: list[dict[str, Any]] = []
        for event in session.events:
            if event.session_id != session.session_id:
                raise ValueError("stale witness event rejected")
            aligned = event.timestamp_ns + session.alignment.offset_ns
            if lower <= aligned <= upper:
                matched.append({**event.as_dict(), "aligned_sample_clock_ns": aligned})
        if session.status == "incomplete":
            state = "incomplete_witness"
        elif matched:
            state = "witness_event_present"
        else:
            state = "clean"
        counts: dict[str, int] = {}
        for matched_event in matched:
            name = str(matched_event["event_type"])
            counts[name] = counts.get(name, 0) + 1
        results.append({**base, "state": state, "event_counts": counts, "events": matched})
    return results
