# worker-03 bounded eBPF witness pilot — 2026-09-03

## Question and boundary

Question: does kernel-side witness telemetry provide useful contextual
information for interpreting the existing native memory-probe measurements?

This was an instrumentation pilot, not a hidden-bit, biological, DRAM-topology,
or controlled-command experiment. It did not rerun or alter the historical
three-boot PMU campaign. No sample was automatically discarded.

## Host and implementation

- Host: `worker-03`, Fedora, Linux `6.17.1-300.fc43.x86_64`, x86_64.
- BTF: `/sys/kernel/btf/vmlinux` available; recorded but not required by this
  tracepoint-only backend.
- Backend: bpftrace `v0.24.2`; bpftool `v7.6.0`; clang `21.1.8`.
- The Fedora packages `bpftrace`, `bpftool`, and `clang` were installed. The BPF
  programs were loaded only for the pilot and unloaded on clean shutdown; no
  persistent kernel settings or programs were created.
- Experiment ID: `witness-pilot-0936803e17b249519b69baef66730c38`.
- Boot ID: `8118368e-e85c-4244-978b-3434c37dadce`.
- Observer interval: `2026-09-03T03:43:23.046689+00:00` through
  `2026-09-03T03:43:24.198092+00:00`.
- Generated observer source/program SHA-256:
  `4c1001ae6b42965ceb628dae059ee09aa2381c883e1bad9bca00cafbb88aa28c`.
- Native artifact SHA-256:
  `9c650336b966d7ee7029220a57d2862a430f27f0400a2000f3e86d1efb3d5f43`.
- Raw retained artifacts on worker-03:
  `/home/worker-03/.local/share/sensetrace/runs/witness-pilot-architecture-v4-20260903`.
- Pilot report SHA-256:
  `807bb7d592a808a438e65dc82f32c606ab0dabb80ce20c1952894f6f6c01b5ec`.
- Raw event stream SHA-256:
  `d85a813e1e16a9c31c3fe1328b4e7bd340d9ffb916988383c9b0996ccfa56a11`.

## Hooks and capability result

All requested tracepoints were present and attached:

- `sched:sched_switch`
- `sched:sched_migrate_task`
- `exceptions:page_fault_user`
- `kmem:mm_page_alloc`
- `vmscan:mm_vmscan_direct_reclaim_begin`
- `compaction:mm_compaction_begin`
- `migrate:mm_migrate_pages`

The observer ended `operational`, with zero unavailable hooks, zero malformed
event lines, and 4,402 retained events. Totals were 56 context switches, 2,177
page-allocation events, and 2,169 user page faults. The attached CPU-migration,
direct-reclaim, compaction, and NUMA-migration hooks produced no events in this
bounded session. This means operational with no relevant events for those
hooks, not proof that those phenomena were absent beyond the observed target
and interval.

## Correlation result

Both native samples used 20,000 raw TSC-cycle measurements. The native timing
source was LFENCE/RDTSC at start and RDTSCP/LFENCE at end. CPU 0 was recorded
before and after both samples; the process affinity allowed CPUs 0–7.

Correlation used Linux monotonic clock identity between `bpf_ktime_get_ns` and
userspace `CLOCK_MONOTONIC`, zero offset, inclusive boundaries, and an explicit
100 microsecond boundary uncertainty.

| Sample | Condition | Correlated context switches | Page allocations | User page faults | State |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Cached-load baseline | 0 | 75 | 74 | `witness_event_present` |
| 1 | Flushed-load plus concurrent 4 MiB first-touch/sched-yield positive control | 1 | 1,069 | 1,066 | `witness_event_present` |

The positive control produced a large, directionally sensible increase in the
specific host events it was intended to create. The baseline also shows why
the witness is useful: even an apparently ordinary native sample can overlap a
small number of observable faults/allocations.

## Conclusion

Result: `yes_context_observed`. Kernel-side witness telemetry supplied useful
context for these native measurements and distinguished the positive-control
window from the baseline. The result validates observer operation and
correlation semantics on this host only.

It does **not** establish direct DRAM access, commands, refresh behavior,
DIMM/channel/rank/bank/row or physical-cell identity, address-scrambling
knowledge, cache/MMU bypass, or analog DRAM measurement. The evidence remains
kernel-side context about possible confounders.
