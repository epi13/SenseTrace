# ADR-008: Genuine cross-boot holdout semantics

## Decision

`E_unseen_boot` groups samples only by the recorded OS `boot_id`. It is
available only when at least three non-placeholder boot IDs are present, so
train, validation, and test can each contain a genuinely independent boot
group. `D_unseen_acquisition_session` remains a separate holdout grouped only
by `acquisition_session_id`.

The analyzer materializes all available hierarchy levels and emits automated
invariants for exact coverage, group disjointness, sample-ID uniqueness, and
identical grouping or materialized partitions. A nested boot-by-session
diagnostic may be reported as provenance, but it is not a cross-boot claim.

## Consequence

Several sessions collected during one OS boot can support D but must leave E
unavailable with an explicit insufficient-boot-groups reason. This downgrades
the historical Phase 1A E result without changing the retained raw evidence.
