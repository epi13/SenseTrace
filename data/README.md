# SenseTrace Data Directory

SenseTrace datasets are generated experimental artifacts and should generally **not** be committed directly to Git.

Use this directory for local acquisition output, manifests, processed datasets, materialized splits, and crash-recovery journals.

Recommended layout:

```text
data/
├── manifests/
├── raw/
│   └── run-<id>/
│       ├── shard-000001.<format>
│       ├── shard-000002.<format>
│       └── ...
├── processed/
├── splits/
└── journals/
```

## Acquisition storage rules

- Do **not** create one file per trace for large campaigns.
- Buffer samples in memory and write sequential dataset shards.
- Start with shard targets around 256 MB-1 GB and tune from measured acquisition throughput.
- Write incomplete shards using a temporary name such as `.tmp`.
- Flush payload and metadata before finalization.
- Compute and record a checksum for finalized shards.
- Finalize with an atomic rename where the storage/filesystem permits it.
- On restart, validate the last finalized shard and quarantine/discard incomplete temporary shards before resuming.
- Keep enough journal/checkpoint state to resume deterministically without silently duplicating samples.

This design makes HDD-backed `worker-03` suitable for early unattended acquisition by favoring large sequential writes rather than high rates of small random I/O.

Raw waveform volume can grow quickly. One million traces containing 2,048 32-bit samples is approximately 8 GB per channel before metadata, so channel count, trace length, sample width, and retention policy must be recorded as experiment parameters.

Commit only small fixtures, schemas, hashes, and metadata required to reproduce an experiment. Large raw traces should live in external/local experiment storage.

See [`docs/DATASET.md`](../docs/DATASET.md) for the dataset contract and feature/splitting rules, and [`docs/WORKER03.md`](../docs/WORKER03.md) for the unattended acquisition and recovery plan.
