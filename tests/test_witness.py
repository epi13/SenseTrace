from __future__ import annotations

import copy
import io
import json
import time
from pathlib import Path

import pytest

from sensetrace.acquisition.probe_contract import (
    ProbeImplementation,
    ProbeRequest,
    ProbeSampleRecord,
)
from sensetrace.witness.correlation import correlate_witness
from sensetrace.witness.models import (
    ClockAlignment,
    WitnessEvent,
    WitnessHookCapability,
    WitnessSession,
)
from sensetrace.witness.observer import (
    KERNEL_CLOCK_DOMAIN,
    USER_CLOCK_DOMAIN,
    BpftraceWitnessObserver,
    discover_witness_capabilities,
)
from sensetrace.witness.protocol import enforce_witness_requirement, witness_protocol


def test_tracefs_permission_denial_is_reported_as_unavailable(monkeypatch):
    def deny(_path: Path) -> bool:
        raise PermissionError("tracefs access denied")

    monkeypatch.setattr(Path, "is_dir", deny)
    capabilities = discover_witness_capabilities(("context_switch",))
    assert capabilities["status"] == "unavailable"
    assert capabilities["tracefs_root"] == "unavailable"
    assert capabilities["hooks"][0]["status"] == "unavailable"


def _probe(index: int = 0, start: int = 100, end: int = 200) -> ProbeSampleRecord:
    implementation = ProbeImplementation(
        implementation_id="test",
        implementation_version="1",
        backend_kind="test",
        artifact_sha256="a" * 64,
        architecture="test",
        kernel_release="test",
        compatibility_status="available",
        timing_source="test",
        result_units="ticks",
    )
    return ProbeSampleRecord(
        implementation=implementation,
        request=ProbeRequest("experiment-1", index, "test", {}, f"sample-{index}"),
        status="complete",
        monotonic_start_ns=start,
        monotonic_end_ns=end,
        clock_domain=USER_CLOCK_DOMAIN,
        raw_result=[1],
        result_units="ticks",
        cpu_before=0,
        cpu_after=0,
        requested_affinity=None,
        effective_affinity=(0,),
    )


def _session(
    *events: WitnessEvent,
    status: str = "operational",
    alignment_status: str = "bounded",
) -> WitnessSession:
    return WitnessSession(
        session_id="observer-1",
        experiment_id="experiment-1",
        status=status,  # type: ignore[arg-type]
        target_pid=1,
        target_tid=1,
        requested_hooks=("context_switch", "cpu_migration", "page_fault"),
        attached_hooks=("context_switch", "cpu_migration", "page_fault"),
        unavailable_hooks=(),
        events=tuple(events),
        provenance={},
        alignment=ClockAlignment(
            KERNEL_CLOCK_DOMAIN,
            USER_CLOCK_DOMAIN,
            0 if alignment_status == "bounded" else None,
            0 if alignment_status == "bounded" else None,
            alignment_status,  # type: ignore[arg-type]
            "test",
        ),
        started_at="start",
        ended_at="end",
    )


def _event(kind: str, timestamp: int, **fields: object) -> WitnessEvent:
    return WitnessEvent(
        session_id="observer-1",
        event_type=kind,
        timestamp_ns=timestamp,
        clock_domain=KERNEL_CLOCK_DOMAIN,
        cpu=0,
        pid=1,
        tid=1,
        fields=dict(fields),
    )


def test_ebpf_and_btf_unavailable_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("sensetrace.witness.observer.shutil.which", lambda _name: None)
    capabilities = discover_witness_capabilities(
        ("context_switch",), tracefs_root=tmp_path / "missing", btf_path=tmp_path / "btf"
    )
    assert capabilities["status"] == "unavailable"
    assert capabilities["backend_path"] == "unavailable"
    assert capabilities["btf"]["status"] == "unavailable"
    assert capabilities["hooks"][0]["status"] == "unsupported"


def test_one_requested_tracepoint_unavailable_is_not_zero(tmp_path, monkeypatch):
    (tmp_path / "sched" / "sched_switch").mkdir(parents=True)
    monkeypatch.setattr("sensetrace.witness.observer.shutil.which", lambda _name: "/bin/true")
    capabilities = discover_witness_capabilities(
        ("context_switch", "page_fault"), tracefs_root=tmp_path
    )
    by_name = {item["name"]: item for item in capabilities["hooks"]}
    assert by_name["context_switch"]["status"] == "supported"
    assert by_name["page_fault"]["status"] == "unsupported"
    assert "never interpreted as zero" in capabilities["missing_semantics"]


def test_correlation_includes_exact_window_boundaries_and_event_types():
    session = _session(
        _event("context_switch", 100),
        _event("cpu_migration", 150),
        _event("page_fault", 200),
        _event("page_fault", 201),
    )
    result = correlate_witness([_probe()], session)[0]
    assert result["state"] == "witness_event_present"
    assert result["event_counts"] == {
        "context_switch": 1,
        "cpu_migration": 1,
        "page_fault": 1,
    }
    assert result["sample_veto"] is False


def test_partial_or_uncertain_witness_is_never_called_clean():
    partial = correlate_witness([_probe()], _session(status="incomplete"))[0]
    uncertain = correlate_witness(
        [_probe()], _session(status="operational", alignment_status="uncertain")
    )[0]
    assert partial["state"] == "incomplete_witness"
    assert uncertain["state"] == "incomplete_witness"


def test_observer_unavailable_correlates_as_unavailable():
    result = correlate_witness([_probe()], _session(status="unavailable"))[0]
    assert result["state"] == "witness_unavailable"


def test_operational_observer_with_no_events_is_clean_not_unavailable():
    result = correlate_witness([_probe()], _session())[0]
    assert result["state"] == "clean"


def test_stale_session_and_stale_event_are_rejected():
    stale_session = _session()
    object.__setattr__(stale_session, "experiment_id", "old-experiment")
    with pytest.raises(ValueError, match="stale witness/session"):
        correlate_witness([_probe()], stale_session)
    stale_event = _event("page_fault", 150)
    object.__setattr__(stale_event, "session_id", "old-observer")
    with pytest.raises(ValueError, match="stale witness event"):
        correlate_witness([_probe()], _session(stale_event))


def test_missing_optional_witness_event_fields_remain_valid():
    event = WitnessEvent(
        session_id="observer-1",
        event_type="page_fault",
        timestamp_ns=150,
        clock_domain=KERNEL_CLOCK_DOMAIN,
        cpu=None,
        pid=None,
        tid=None,
    )
    assert correlate_witness([_probe()], _session(event))[0]["state"] == "witness_event_present"


def test_witness_does_not_mutate_historical_evidence_or_claim_dram():
    historical = {"outcome": "B_observable_available_but_oracle_weak", "witness": "not_collected"}
    before = copy.deepcopy(historical)
    session = _session(_event("page_fault", 150))
    correlate_witness([_probe()], session)
    assert historical == before
    rendered = session.as_dict()
    assert "not direct DRAM" in rendered["claim_boundary"]


def test_frozen_witness_protocol_requires_explicit_requirement_and_never_auto_vetoes():
    protocol = witness_protocol(
        {"witness": {"requirement": "required", "hooks": ["context_switch"]}}
    )
    assert protocol["requirement"] == "required"
    assert protocol["automatic_sample_veto"] is False
    enforce_witness_requirement(protocol, _session())
    with pytest.raises(RuntimeError, match="requires operational"):
        enforce_witness_requirement(protocol, _session(status="incomplete"))
    with pytest.raises(ValueError, match="disabled, optional, or required"):
        witness_protocol({"witness": {"requirement": "best_effort"}})


def _capability_fixture() -> dict[str, object]:
    return {
        "schema": "sensetrace.ebpf-witness-capabilities.v1",
        "status": "supported",
        "kernel_release": "test",
        "architecture": "test",
        "backend": "bpftrace",
        "backend_path": "/fake/bpftrace",
        "backend_version": "test",
        "tracefs_root": "/test",
        "btf": {"status": "available"},
        "privilege": {},
        "requested_hooks": ["context_switch"],
        "hooks": [
            WitnessHookCapability(
                "context_switch",
                "sched:sched_switch",
                "supported",
                "test",
                "/test/sched/sched_switch",
            ).as_dict()
        ],
        "collection": "not_collected",
    }


def test_observer_permission_failure_is_classified(tmp_path, monkeypatch):
    class DeniedProcess:
        returncode = 1
        stderr = io.StringIO("permission denied while loading BPF")
        pid = 44

        def poll(self):
            return 1

    monkeypatch.setattr(
        "sensetrace.witness.observer.discover_witness_capabilities",
        lambda *args, **kwargs: _capability_fixture(),
    )
    monkeypatch.setattr(
        "sensetrace.witness.observer.subprocess.Popen", lambda *a, **k: DeniedProcess()
    )
    observer = BpftraceWitnessObserver(
        session_id="observer-1",
        experiment_id="experiment-1",
        target_pid=1,
        target_tid=1,
        output_dir=tmp_path,
        requested_hooks=("context_switch",),
    )
    assert observer.start(timeout=0.1) is False
    session = observer.stop()
    assert session.status == "unavailable"
    assert session.failure["kind"] == "permission_denied"  # type: ignore[index]


def test_observer_generic_startup_failure_is_distinct(tmp_path, monkeypatch):
    class FailedProcess:
        returncode = 2
        stderr = io.StringIO("syntax or verifier failure")
        pid = 46

        def poll(self):
            return 2

    monkeypatch.setattr(
        "sensetrace.witness.observer.discover_witness_capabilities",
        lambda *args, **kwargs: _capability_fixture(),
    )
    monkeypatch.setattr(
        "sensetrace.witness.observer.subprocess.Popen", lambda *a, **k: FailedProcess()
    )
    observer = BpftraceWitnessObserver(
        session_id="observer-1",
        experiment_id="experiment-1",
        target_pid=1,
        target_tid=1,
        output_dir=tmp_path,
        requested_hooks=("context_switch",),
    )
    assert observer.start(timeout=0.1) is False
    session = observer.stop()
    assert session.status == "failed"
    assert session.failure["kind"] == "startup_failure"  # type: ignore[index]


def test_observer_clean_shutdown_and_timestamp_metadata(tmp_path, monkeypatch):
    class RunningProcess:
        returncode = None
        stderr = io.StringIO("")
        pid = 45

        def __init__(self, command):
            self.output = command[command.index("-o") + 1]
            with open(self.output, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "sensetrace.witness-ready.v1",
                            "session_id": "observer-1",
                            "timestamp_ns": time.monotonic_ns(),
                        }
                    )
                    + "\n"
                )

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "sensetrace.witness.observer.discover_witness_capabilities",
        lambda *args, **kwargs: _capability_fixture(),
    )
    monkeypatch.setattr(
        "sensetrace.witness.observer.subprocess.Popen",
        lambda command, **kwargs: RunningProcess(command),
    )
    observer = BpftraceWitnessObserver(
        session_id="observer-1",
        experiment_id="experiment-1",
        target_pid=1,
        target_tid=1,
        output_dir=tmp_path,
        requested_hooks=("context_switch",),
    )
    assert observer.start(timeout=0.2) is True
    with observer.event_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "sensetrace.witness-event.v1",
                    "session_id": "observer-1",
                    "event_type": "context_switch",
                    "timestamp_ns": time.monotonic_ns(),
                    "cpu": 0,
                    "pid": 1,
                    "tid": 1,
                }
            )
            + "\n"
        )
    session = observer.stop()
    assert session.status == "operational"
    assert session.alignment.status == "bounded"
    assert session.provenance["clock_source"] == KERNEL_CLOCK_DOMAIN
    assert session.provenance["program_sha256"] != "not_collected"
    assert len(session.events) == 1
