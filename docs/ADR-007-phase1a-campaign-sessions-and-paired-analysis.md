# ADR-007: Genuine Phase 1A sessions and complete holdout reporting

## Decision

One `CommodityDramBackend` instance represents one acquisition session. A
session has an explicit UUID, start time, host inventory snapshot, boot ID,
fresh anonymous controlled buffer, label stream, journal, environment and
measurement provenance, and configuration/code hashes. `session_count` is a
campaign setting that creates multiple backend instances; it must not be used
to label sections of one stream as independent sessions.

Phase 1A combines finalized per-session source datasets into a campaign
condition while retaining every source manifest and fingerprint. Sample,
session, pair, and virtual-location identifiers are globally unambiguous in a
combined dataset. Legacy `session_id` and `location_id` fields remain readable
aliases, with documentation that they do not identify physical DRAM topology.

Pair order is exactly counterbalanced per virtual location and randomized at
the pair level. Every available A–E split is materialized and evaluated
independently. The report includes a predeclared paired sample-median timing
analysis, cluster-aware uncertainty, and nuisance/drift audits; those audits
are not model features.

CLFLUSH provenance names the actual `_mm_clflush` and `_mm_mfence` sequence and
the LFENCE/RDTSC/RDTSCP timing fences. It is a cache-control primitive only and
does not establish a DRAM access or physical location.

## Consequence

Single-session campaigns may produce useful A/B diagnostics but cannot silently
support D/E claims. A positive result on one split is bounded by that split's
grouping and claim boundary. A clean null, an unavailable strict split, or a
failed paired diagnostic is a valid scientific outcome.
