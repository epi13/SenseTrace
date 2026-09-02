# ADR-009: Physical allocation boundary on recovery

## Decision

An interrupted commodity acquisition session fails closed at the physical
allocation boundary. Finalized shards remain immutable evidence in the
interrupted directory and an append-only journal event records the decision.
Recovery creates a sibling session with a new acquisition-session UUID, fresh
host snapshot, fresh anonymous allocation ID, and an explicit parent-session
relationship. The replacement, and only completed scientifically valid source
sessions, may enter a combined campaign dataset.

`allocation_id` is persisted in every new sample, and virtual location, pair,
cell/offset, and buffer identifiers include both session and allocation
identity. A pair cannot span allocations: a partially finalized pair remains
in the excluded interrupted source, while the replacement starts a new pair
identity. The general commodity runner also refuses in-place resume when
finalized rows already exist and directs the operator to a new run/session.

## Consequence

The project never silently joins observations from different anonymous
allocations under one session or location identity. Interrupted evidence is
not discarded, but it is not silently promoted into a complete campaign.
