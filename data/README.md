# SenseTrace Data Directory

SenseTrace datasets are generated experimental artifacts and should generally **not** be committed directly to Git.

Use this directory for local acquisition output, manifests, processed datasets, and materialized splits.

Recommended layout:

```text
data/
├── manifests/
├── raw/
├── processed/
└── splits/
```

Commit only small fixtures, schemas, hashes, and metadata required to reproduce an experiment. Large raw traces should live in external/local experiment storage.

See [`docs/DATASET.md`](../docs/DATASET.md) for the dataset contract and feature/splitting rules.
