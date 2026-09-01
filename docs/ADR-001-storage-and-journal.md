# ADR-001: Sequential NPZ shards and append-only journal

## Decision

Phase 0 stores buffered samples in compressed NPZ shards with a JSON sidecar. The payload is written to `shard-XXXXXX.npz.tmp`, flushed and fsynced, checksummed, described by a fsynced sidecar, and finalized with atomic renames. A JSONL journal records lifecycle, shard, interruption, recovery, and storage-guard events; every journal record is flushed and fsynced.

## Rationale

This is transparent to Python and scientific tooling, supports array-level validation without loading the entire dataset, favors sequential HDD writes, and avoids one-file-per-trace overhead. The journal is append-only so a crash does not require rewriting a giant manifest. Incomplete temporary or invalid finalized shards are quarantined before resume.

## Consequence

Loading a full dataset for training still materializes arrays in memory. A future high-volume reader can stream validated shards without changing the on-disk contract.
