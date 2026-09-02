# ADR-012: Measurement-primitive characterization boundaries

- Status: accepted
- Date: 2026-09-01

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
its frozen protocol records that zero-only invariant.

Linux PMU discovery uses an explicitly serialized `perf_event_attr` with
disabled creation, kernel and hypervisor exclusion, the calling-thread PID, and
`cpu=-1`. Discovery is vocabulary and permission evidence only: it never
collects system-wide counters, reads an operation-scoped counter, or treats a
timing shift as a DRAM oracle. A future PMU primitive must provide its own event
encoding, scoped reader, multiplexing evidence, and independent access-state
oracle before it can enter the A/B/C gate.

## Consequences

This keeps calibration data out of physical Phase 1A evidence and prevents the
characterization engine from smuggling commodity-specific assumptions into a
new primitive. The current commodity characterization can therefore support a
controlled observable result while remaining outcome B until independent
access-state instrumentation is available. A PMU candidate that cannot pass its
permission and scope boundary is not characterized or promoted by inference.
