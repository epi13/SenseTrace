# ADR-006: Conservative Phase 1A measurement provenance

## Decision

Phase 1A begins with ordinary user-space accesses to an anonymous page-aligned
buffer. `mlock` and CPU affinity are best-effort operations whose actual result
is recorded. Timing uses the host's `perf_counter_ns` path and preserves raw
observations. Cache controls are named and scoped: `none` is a cache-hit
control, `eviction_buffer` is best-effort eviction, and `clflush` uses the
native `_mm_clflush` plus `_mm_mfence` sequence before the timed load, with
explicit compiler barriers around the LFENCE/RDTSC/RDTSCP sequence. Neither
eviction method proves a DRAM access. Whole-word contrast, single-bit contrast,
and randomized words are separate patterns.

The ordinary digital read verifies the controlled write but is excluded from
the feature matrix. Physical address, row, bank, voltage, refresh, and
disturbance claims are not exposed. Package/RAPL, PMU, environmental, CPU
frequency, boot, session, and cache-control provenance are recorded when
available.

Phase 1A is exploratory and requires an explicit passing Phase 0 report. It
includes same-observation label permutation, exact pair-order counterbalance,
trial-order/metadata/drift audits, cache-hit, random-word, and
idle/no-memory-operation controls. Confirmation requires a new acquisition
after freezing method, preprocessing, model family, and evaluation.

## Consequence

Chance results are valid outcomes. Any positive result is bounded by the
accessible host measurement channel and cannot be described as direct DRAM
state inference without independent evidence.
