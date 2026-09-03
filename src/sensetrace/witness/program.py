"""Small bpftrace program generator with one fragment per detected hook."""

from __future__ import annotations

import json

HOOK_TRACEPOINTS = {
    "context_switch": "sched:sched_switch",
    "cpu_migration": "sched:sched_migrate_task",
    "page_fault": "exceptions:page_fault_user",
    "page_allocation": "kmem:mm_page_alloc",
    "page_free": "kmem:mm_page_free",
    "direct_reclaim": "vmscan:mm_vmscan_direct_reclaim_begin",
    "compaction": "compaction:mm_compaction_begin",
    "numa_migration": "migrate:mm_migrate_pages",
}
PROGRAM_VERSION = "sensetrace-bpftrace-witness-v1"


def _line(
    session_id: str,
    event_type: str,
    extra_format: str = "",
    extra_arguments: str = "",
) -> str:
    return (
        f'printf("{{\\"schema\\":\\"sensetrace.witness-event.v1\\",'
        f'\\"session_id\\":\\"{session_id}\\",'
        f'\\"event_type\\":\\"{event_type}\\",'
        '\\"timestamp_ns\\":%llu,\\"cpu\\":%d,\\"pid\\":%d,\\"tid\\":%d'
        f'{extra_format}}}\\n", nsecs, cpu, pid, tid{extra_arguments});'
    )


def render_bpftrace_program(session_id: str, hooks: tuple[str, ...]) -> str:
    """Render only available tracepoints so partial kernels can still observe."""

    if not session_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in session_id
    ):
        raise ValueError("witness session_id contains characters unsafe for generated source")
    blocks = [
        "BEGIN",
        "{",
        f'  printf("{{\\"schema\\":\\"sensetrace.witness-ready.v1\\",\\"session_id\\":\\"{session_id}\\",\\"timestamp_ns\\":%llu}}\\n", nsecs);',
        "}",
    ]
    for hook in hooks:
        tracepoint = HOOK_TRACEPOINTS[hook]
        if hook == "context_switch":
            predicate = "/args->prev_pid == $2 || args->next_pid == $2/"
            statement = _line(
                session_id,
                hook,
                ',\\"prev_tid\\":%d,\\"next_tid\\":%d',
                ", args->prev_pid, args->next_pid",
            )
        elif hook == "cpu_migration":
            predicate = "/args->pid == $2/"
            statement = _line(
                session_id,
                hook,
                ',\\"migrated_tid\\":%d,\\"source_cpu\\":%d,\\"destination_cpu\\":%d',
                ", args->pid, args->orig_cpu, args->dest_cpu",
            )
        else:
            predicate = "/pid == $1 || tid == $2/"
            statement = _line(session_id, hook)
        blocks.extend(
            [
                f"tracepoint:{tracepoint}",
                predicate,
                "{",
                f"  {statement}",
                "}",
            ]
        )
    return "\n".join(blocks) + "\n"


def render_bpftrace_program_for_targets(
    session_id: str,
    hooks: tuple[str, ...],
    *,
    target_tids: tuple[int, ...],
) -> str:
    """Render predicates covering every watched TID with literal identities.

    The single-TID ``render_bpftrace_program`` uses ``$2`` for one TID. The
    pilot positive control churns on a worker thread while native samples run
    on the main thread, so scheduler hooks must cover both TIDs. Literal TIDs
    keep the program bounded and auditable; the source hash records exactly
    which TIDs were watched. ``pid == $1`` is retained for process-wide
    memory hooks so the caller still passes the target PID as ``$1``.
    """

    tids = tuple(sorted({int(tid) for tid in target_tids}))
    if not tids or any(tid <= 0 for tid in tids):
        raise ValueError("target_tids must be non-empty positive thread IDs")
    if not session_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in session_id
    ):
        raise ValueError("witness session_id contains characters unsafe for generated source")
    for hook in hooks:
        if hook not in HOOK_TRACEPOINTS:
            raise ValueError(f"unknown witness hook: {hook}")
    blocks = [
        "BEGIN",
        "{",
        f'  printf("{{\\"schema\\":\\"sensetrace.witness-ready.v1\\",\\"session_id\\":\\"{session_id}\\",\\"timestamp_ns\\":%llu}}\\n", nsecs);',
        "}",
    ]
    switch_alternatives = " || ".join(
        f"args->prev_pid == {tid} || args->next_pid == {tid}" for tid in tids
    )
    migrate_alternatives = " || ".join(f"args->pid == {tid}" for tid in tids)
    tid_alternatives = " || ".join(f"tid == {tid}" for tid in tids)
    for hook in hooks:
        tracepoint = HOOK_TRACEPOINTS[hook]
        if hook == "context_switch":
            predicate = f"/{switch_alternatives}/"
            statement = _line(
                session_id,
                hook,
                ',\\"prev_tid\\":%d,\\"next_tid\\":%d',
                ", args->prev_pid, args->next_pid",
            )
        elif hook == "cpu_migration":
            predicate = f"/{migrate_alternatives}/"
            statement = _line(
                session_id,
                hook,
                ',\\"migrated_tid\\":%d,\\"source_cpu\\":%d,\\"destination_cpu\\":%d',
                ", args->pid, args->orig_cpu, args->dest_cpu",
            )
        else:
            predicate = f"/pid == $1 || {tid_alternatives}/"
            statement = _line(session_id, hook)
        blocks.extend(
            [
                f"tracepoint:{tracepoint}",
                predicate,
                "{",
                f"  {statement}",
                "}",
            ]
        )
    return "\n".join(blocks) + "\n"


def observed_tids_from_program(source: str) -> tuple[int, ...]:
    """Extract literal watched TIDs from a multi-target program for testing."""

    import re

    found = sorted({int(value) for value in re.findall(r"== (\d+)", source) if int(value) > 0})
    return tuple(found)


def program_manifest(session_id: str, hooks: tuple[str, ...]) -> dict[str, object]:
    return {
        "backend": "bpftrace",
        "implementation_version": PROGRAM_VERSION,
        "source_kind": "generated from versioned per-tracepoint fragments",
        "session_id": session_id,
        "hooks": list(hooks),
        "tracepoints": [HOOK_TRACEPOINTS[hook] for hook in hooks],
        "event_schema": json.loads('{"schema":"sensetrace.witness-event.v1"}')["schema"],
    }
