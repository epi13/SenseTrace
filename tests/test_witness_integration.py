from __future__ import annotations

import os
from uuid import uuid4

import pytest

from sensetrace.witness.observer import (
    BpftraceWitnessObserver,
    discover_witness_capabilities,
)


def test_live_ebpf_observer_when_host_supports_it(tmp_path):
    capabilities = discover_witness_capabilities(("context_switch",))
    if capabilities["status"] != "supported":
        pytest.skip(
            "live eBPF witness unavailable: "
            f"backend={capabilities['backend_path']}, hooks={capabilities['hooks']}"
        )
    privilege = capabilities["privilege"]
    if os.geteuid() != 0 and not (privilege.get("cap_bpf") and privilege.get("cap_perfmon")):
        pytest.skip(
            "live eBPF witness permission unavailable: requires root or CAP_BPF+CAP_PERFMON"
        )
    observer = BpftraceWitnessObserver(
        session_id=f"integration-{uuid4().hex}",
        experiment_id="integration-test",
        target_pid=os.getpid(),
        target_tid=None,
        output_dir=tmp_path,
        requested_hooks=("context_switch",),
    )
    if not observer.start():
        session = observer.stop()
        pytest.fail(f"detected live eBPF capability but observer failed: {session.failure}")
    session = observer.stop()
    assert session.status == "operational"
    assert session.attached_hooks == ("context_switch",)
