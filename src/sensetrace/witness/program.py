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
