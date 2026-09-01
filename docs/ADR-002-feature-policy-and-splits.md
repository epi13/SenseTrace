# ADR-002: Default-deny identity fields and materialized grouped splits

## Decision

`host_id`, `device_id`, `bank_id`, `row_id`, `cell_or_offset_id`, `session_id`, `trial_index`, and `sample_id` are audit/grouping fields. The feature policy rejects them as model inputs unless a caller explicitly opts into a controlled leakage test. Primary splits materialize exact sample IDs and group on configured topology/session fields.

## Rationale

Remembering identity is an easier explanation than learning a general state-correlated physical channel. The policy is enforced at feature construction so correctness does not depend on every researcher remembering a manual filter. Materialized fingerprints prevent a changed manifest or ad-hoc resplit from being silently reused.

## Consequence

Missing physical topology limits the strongest defensible holdout. The absence is recorded rather than replaced with an inferred address mapping.
