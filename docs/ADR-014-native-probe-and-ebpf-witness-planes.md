# ADR-014: Native probe and eBPF witness planes

- Status: accepted
- Date: 2026-09-03

## Decision

SenseTrace keeps five epistemically distinct planes:

1. **Research/control — Python.** Freezes protocols, orchestrates runs, records
   provenance and evidence, performs statistics, recovers interrupted work,
   and enforces claims.
2. **Native CPU probe — C with a narrow Python ABI.** Performs timing-sensitive
   loads, fences, cache operations, and future bounded PMU operations. The
   `sensetrace.probe-sample.v1` contract records implementation/version and
   artifact hash, compatibility, timing source, affinity, request parameters,
   sample index/window, raw result, status/failure, provenance, and optional
   witness correlation identity.
3. **Experimental eBPF witness — bpftrace backend.** Temporarily attaches only
   detected tracepoints and records low-interference scheduling and
   memory-management context for a target PID/TID. It is an observer, never an
   actuator.
4. **Privileged host probe.** Reserved for bounded privileged allocation,
   legitimate physical-page provenance, privileged telemetry, or a future
   kernel driver. Privilege does not by itself establish DRAM topology.
5. **Controlled external memory hardware.** A qualitatively different boundary
   that must identify the target/device, controller command/address, timing,
   refresh relationship, trigger, channels, clocks, firmware/configuration,
   and calibration.

The level numbers identify access and claim boundaries. They do not guarantee
that a numerically higher layer is better in every measurement dimension.

## Initial witness backend and hooks

The bounded backend generates explicit bpftrace source from versioned
fragments. It requests only configured hooks and attaches only tracepoints
present in tracefs:

- `sched:sched_switch` — switches involving the target TID;
- `sched:sched_migrate_task` — target-TID CPU migration;
- `exceptions:page_fault_user` — user page faults in target process context;
- `kmem:mm_page_alloc` and `kmem:mm_page_free` — selected allocation activity;
- `vmscan:mm_vmscan_direct_reclaim_begin` — direct reclaim activity;
- `compaction:mm_compaction_begin` — compaction activity;
- `migrate:mm_migrate_pages` — NUMA/page migration context where exposed.

Tracepoint availability and fields vary by kernel. Capability discovery records
the kernel/architecture, bpftrace identity, BTF state, tracefs source, effective
capabilities, every requested/attached/unavailable hook, source/program hash,
observer process identity, target PID/TID, timestamps, clock domain, and parse
failures. A missing hook is `unsupported` or `unavailable`; a load failure is
`permission_denied` or `failed`; telemetry is never silently treated as zero.
BTF is recorded but is not required by this tracepoint-only bpftrace backend.

## Correlation and interpretation

Native samples use userspace `CLOCK_MONOTONIC` windows. Witness events use
`bpf_ktime_get_ns`. On Linux these have a semantic monotonic-clock identity;
the correlator records zero offset plus a conservative 100 microsecond boundary
uncertainty. Inclusive, uncertainty-expanded boundaries are deterministic.
If alignment is unavailable or uncertain, the sample is
`incomplete_witness`, not guessed clean.

Samples retain one of `clean`, `witness_event_present`, `incomplete_witness`,
or `witness_unavailable`. No state automatically discards a sample. A frozen
protocol must explicitly say whether witness evidence is disabled, optional,
or required and how analysis uses it. Old evidence remains immutable and is
not reinterpreted as if a witness had been present.

## Claim boundary

eBPF does **not** inherently provide direct DRAM commands; ACTIVATE, PRECHARGE,
or REFRESH control; reliable DIMM/channel/rank/bank/row identity; controller
scrambling knowledge; physical-cell identity; analog DRAM measurements; or a
direct bypass of caches, the MMU, or the memory controller. It provides
contextual host evidence only.

Native C/Rust/assembly can improve timing and hardware-adjacent execution, but
does not by itself provide any of those controlled-DRAM properties either.
Neither plane may populate controlled-hardware topology fields.
