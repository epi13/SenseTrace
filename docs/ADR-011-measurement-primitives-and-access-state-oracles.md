# ADR-011: Measurement primitives and access-state oracles

## Context

The native positive control shows that SenseTrace can detect an artificial
timing perturbation as small as the current 32-TSC-cycle control. The corrected
three-boot commodity validation was near chance. CLFLUSH plus a timed load is
therefore an ambiguous cache/memory-hierarchy observable, not proof that a load
reached physical DRAM.

## Decision

Phase 1A commodity acquisition is frozen as
`phase1a-commodity-baseline-v1`. New physical work must select a named
measurement primitive through the acquisition factory rather than changing the
meaning of the baseline in place.

Each primitive separates target-state preparation, the operation under test,
access-state/event provenance, physical observation, audit-only metadata, and
model-eligible features. A primitive declares capabilities with explicit
`known`, `unknown`, or `unsupported` values. Addresses, allocation/session/boot
identities, oracle identity, and oracle results are excluded from model inputs
by default.

An access-state oracle is recorded separately from timing. Its strength is
`exact`, `probabilistic`, `partial`, or `unavailable`; a requested cache state
is not promoted to an independent oracle. Commodity PMU discovery is scoped to
documenting the host's vocabulary and permission boundary. Future counter
readers must scope collection to SenseTrace-owned processes/threads and must
record permission failures without broadening host security policy.

Before hidden-bit inference, each new primitive must pass null, strong and weak
positive controls, session/boot/order/allocation checks, and oracle-agreement
characterization where an oracle exists. A high model score alone is not a
characterization gate.

## Consequence

The current commodity primitive remains useful as a reproducible comparison
baseline and instrumentation control. Its current decision is **B: observable
available but oracle weak**. If local instrumentation cannot provide a
meaningful independent access-state oracle, commodity Phase 1 should stop and
the architecture should transition toward controlled-memory-interface research
instead of scaling the same uncertain signal.

