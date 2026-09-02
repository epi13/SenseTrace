# ADR-012: Measurement-primitive characterization boundaries

- Status: accepted
- Date: 2026-09-02

## Decision

Characterization is executed through a `MeasurementPrimitive` contract. Each
primitive declares its controls, required contrasts, replication and allocation
matching, oracle gate, scope boundary, and claim boundary. The shared engine
randomizes declared controls, preserves raw observations and acquisition order,
and emits exactly one evidence outcome:

- `A_usable_auditable_primitive` when the observable, controls, provenance, and
  independent oracle gates pass;
- `B_observable_available_but_oracle_weak` when the observable is controlled but
  an independent oracle is absent or insufficient; or
- `C_primitive_unsuitable` when an observable, control, provenance, or scope gate
  fails.

The commodity timed-load primitive remains a cache-path/timing observable. Its
characterization controls may use artificial delay only through the explicit
`TimingPerturbationCalibration` context. Ordinary physical Phase 1A rejects
nonzero delay, non-default perturbation labels, and calibration namespaces, and
its frozen protocol records that zero-only invariant. Characterization protocol
v2 has a machine-readable robust null rule: median of replicate sample
medians, median absolute deviation, relative to the absolute median center,
with maximum relative deviation 0.25, maximum relative MAD 0.10, and at least
three complete finite replicates. Completeness, finite validity, and stability
are reported separately; insufficient evidence never becomes a silent pass.

Contrasts are joined by explicit replicate IDs rather than filtered-list
position. Each report retains matched IDs, missing-side IDs, paired
differences, and a summary. Physical Phase 1A analysis requires an explicit
`physical_phase1a` dataset purpose, the frozen
`phase1a-commodity-baseline-v1` identity, a non-empty matching protocol hash,
zero timing perturbation, no calibration namespace, and agreement across shard
metadata, session ledgers, combined manifests, and embedded source manifests.
Calibration datasets remain loadable through their explicit calibration path.

Linux PMU discovery and the optional reader use explicitly serialized
`perf_event_attr` records with disabled creation, kernel and hypervisor
exclusion, the calling-thread PID, `cpu=-1`, `inherit=0`, and explicit
time-enabled/time-running read format. The reader resets and enables only
around a SenseTrace-owned callback, then disables, reads, and closes the FD.
Sysfs event records preserve PMU device, source type, full-width config, and
format fields; ambiguous bare aliases are never selected. Uncore events remain
unusable for this thread-scoped primitive. A PMU observation may support only
the documented cache/access-path statement; it is never called a DRAM access
or hidden-bit oracle by name alone.

## Consequences

This keeps calibration data out of physical Phase 1A evidence and prevents the
characterization engine from smuggling commodity-specific assumptions into a
new primitive. The current commodity characterization can therefore support a
controlled observable result while remaining outcome B until independent
access-state instrumentation is available. A PMU candidate that cannot pass its
permission and scope boundary is not characterized or promoted by inference.
