"""Bounded pilot correlating existing native probes with witness evidence."""

from __future__ import annotations

import json
import mmap
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sensetrace.acquisition.native import NativeMeasurementKernel
from sensetrace.acquisition.probe_contract import ProbeRequest, ProbeSampleRecord

from .correlation import correlate_witness
from .observer import BpftraceWitnessObserver, discover_witness_capabilities

PILOT_HOOKS = (
    "context_switch",
    "cpu_migration",
    "page_fault",
    "page_allocation",
    "direct_reclaim",
    "compaction",
    "numa_migration",
)


def _positive_control(
    stop: threading.Event,
    tid_ready: threading.Event,
    churn_start: threading.Event,
    tid_box: list[int],
) -> None:
    """Generate bounded, process-owned page faults and scheduler activity.

    Runs on the worker thread whose TID is observed by the witness program.
    Memory churn (first-touch faults/allocations) is process-wide and visible
    via the PID filter; scheduler yields run on this watched TID so the
    scheduler hooks are actually exercised. Waits for ``churn_start`` so the
    baseline sample observes an idle worker.
    """

    tid_box.append(threading.get_native_id())
    tid_ready.set()
    churn_start.wait(timeout=5)
    pages = mmap.mmap(-1, 4 * 1024 * 1024)
    try:
        while not stop.is_set():
            for offset in range(0, len(pages), mmap.PAGESIZE):
                pages[offset] = (pages[offset] + 1) % 256
                if offset % (256 * mmap.PAGESIZE) == 0:
                    os.sched_yield()
            pages.close()
            if stop.is_set():
                break
            pages = mmap.mmap(-1, 4 * 1024 * 1024)
    finally:
        try:
            pages.close()
        except (BufferError, OSError):
            pass


def run_witness_pilot(
    output_dir: str | Path,
    *,
    use_sudo: bool = False,
    repetitions: int = 20_000,
) -> dict[str, object]:
    """Run two native samples, one overlapped by an explicit positive control."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    experiment_id = f"witness-pilot-{uuid4().hex}"
    observer_id = f"observer-{uuid4().hex}"
    capabilities = discover_witness_capabilities(PILOT_HOOKS)
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        report: dict[str, object] = {
            "schema": "sensetrace.ebpf-witness-pilot.v1",
            "status": "observer_unavailable",
            "reason": "native measurement kernel is unavailable",
            "capabilities": capabilities,
        }
        (destination / "pilot-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    main_tid = threading.get_native_id()
    stop = threading.Event()
    tid_ready = threading.Event()
    churn_start = threading.Event()
    tid_box: list[int] = []
    worker = threading.Thread(
        target=_positive_control, args=(stop, tid_ready, churn_start, tid_box), daemon=True
    )
    worker.start()
    if not tid_ready.wait(timeout=5) or not tid_box:
        stop.set()
        worker.join(timeout=3)
        report = {
            "schema": "sensetrace.ebpf-witness-pilot.v1",
            "status": "observer_startup_failed",
            "experiment_id": experiment_id,
            "reason": "positive-control worker TID was unavailable",
            "question": "Does kernel-side witness telemetry provide useful context for native memory-probe measurements?",
            "answer": "not_evaluable",
            "claim_boundary": "no hardware result; positive control did not start",
        }
        (destination / "pilot-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return report
    worker_tid = int(tid_box[0])
    observed_tids = tuple(sorted({main_tid, worker_tid}))
    observer = BpftraceWitnessObserver(
        session_id=observer_id,
        experiment_id=experiment_id,
        target_pid=os.getpid(),
        target_tid=main_tid,
        target_tids=observed_tids,
        output_dir=destination / "observer",
        requested_hooks=PILOT_HOOKS,
        use_sudo=use_sudo,
    )
    operational = observer.start()
    if not operational:
        stop.set()
        churn_start.set()
        worker.join(timeout=3)
        session = observer.stop()
        report = {
            "schema": "sensetrace.ebpf-witness-pilot.v1",
            "status": "observer_unavailable"
            if session.status == "unavailable"
            else "observer_startup_failed",
            "experiment_id": experiment_id,
            "observer": session.as_dict(),
            "question": "Does kernel-side witness telemetry provide useful context for native memory-probe measurements?",
            "answer": "not_evaluable",
            "claim_boundary": "no hardware result; observer did not become operational",
        }
        (destination / "pilot-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return report

    address = NativeMeasurementKernel.calibration_address()
    samples: list[ProbeSampleRecord] = []
    samples.append(
        kernel.execute(
            ProbeRequest(
                session_id=experiment_id,
                sample_index=0,
                operation="cached_load",
                parameters={"repetitions": repetitions},
                correlation_id="baseline-native-sample",
            ),
            address=address,
        )
    )
    churn_start.set()
    try:
        samples.append(
            kernel.execute(
                ProbeRequest(
                    session_id=experiment_id,
                    sample_index=1,
                    operation="flushed_load",
                    parameters={
                        "repetitions": repetitions,
                        "positive_control": (
                            "concurrent 4 MiB first-touch churn and sched_yield "
                            "on watched worker TID"
                        ),
                    },
                    correlation_id="positive-control-native-sample",
                ),
                address=address,
            )
        )
    finally:
        stop.set()
        worker.join(timeout=3)
    time.sleep(0.1)
    session = observer.stop()
    correlation = correlate_witness(samples, session)
    positive_events = len(correlation[1]["events"])
    if session.status == "operational":
        state = (
            "observer_operational" if positive_events else "observer_operational_no_relevant_events"
        )
    elif session.status == "incomplete":
        state = "observer_incomplete"
    else:
        state = "observer_unavailable"
    report = {
        "schema": "sensetrace.ebpf-witness-pilot.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": state,
        "experiment_id": experiment_id,
        "question": "Does kernel-side witness telemetry provide useful context for native memory-probe measurements?",
        "answer": (
            "yes_context_observed" if positive_events else "not_demonstrated_in_this_bounded_run"
        ),
        "probe_samples": [sample.as_dict() for sample in samples],
        "observer": session.as_dict(),
        "correlation": correlation,
        "positive_control": {
            "predeclared": True,
            "condition": (
                "concurrent process-owned 4 MiB page first-touch churn plus "
                "scheduler yields on watched worker TID"
            ),
            "purpose": "verify confounder visibility, not create or test a DRAM claim",
            "observed_tids": list(observed_tids),
            "main_tid": main_tid,
            "worker_tid": worker_tid,
            "scheduler_validation": (
                "sched_yield runs on the watched worker TID so scheduler hooks "
                "are actually exercised; memory churn is process-wide via PID filter"
            ),
        },
        "interpretation": (
            "Witness events are contextual confounder evidence. They are not direct DRAM evidence "
            "and did not automatically veto either sample."
        ),
    }
    (destination / "pilot-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report
