# SenseTrace Dataset Contract

## Purpose

The dataset format must make experimental leakage easy to detect and hard to introduce accidentally. Raw measurements, labels, physical grouping metadata, and model features should remain conceptually separate.

## Canonical sample

Each sample represents one controlled write-and-measure cycle.

Recommended metadata fields:

| Field | Purpose | Model input by default? |
| --- | --- | --- |
| `sample_id` | Unique sample identifier | No |
| `label` | Ground-truth hidden bit (`0`/`1`) | Target only |
| `trace_ref` | Reference to raw trace data | Yes, through trace loader |
| `session_id` | Acquisition session grouping | No |
| `device_id` | DIMM/chip grouping identifier | No |
| `bank_id` | Physical grouping where known | No |
| `row_id` | Physical grouping where known | No |
| `cell_or_offset_id` | Fine physical grouping where known | No |
| `trial_index` | Acquisition ordering/audit | No |
| `temperature_c` | Environmental measurement | Optional |
| `vdd_v` | Supply measurement | Optional |
| `refresh_age_ns` | Time since relevant refresh event | Optional |
| `wait_ns` | Controlled delay before observation | Optional |
| `timing_profile` | Acquisition timing configuration | Optional/encoded |
| `channel_mask` | Which measurement channels are valid | No/direct loader use |
| `seed_id` | Reproducibility reference, not raw secret/random state | No |

Exact physical topology fields may be unavailable on early hardware. Missing topology is acceptable, but the absence must be recorded because it limits which holdout claims can be made.

## Raw traces

Raw time-series measurements should be preserved before feature extraction.

A trace should carry:

- sample rate;
- channel names;
- units;
- trigger/alignment information;
- acquisition window;
- instrument configuration where applicable.

Do not retain only engineered features if raw traces can reasonably be stored. Future analysis may discover signal characteristics not anticipated during acquisition.

## Storage recommendation

For early experiments:

```text
data/
├── manifests/
│   └── run-YYYYMMDD-*.parquet|csv
├── raw/
│   └── <session>/<sample-or-shard>.*
├── processed/
│   └── <pipeline-version>/...
└── splits/
    └── <split-name>.json
```

Large generated datasets should not be committed directly to Git. Commit schemas, small fixtures, manifests, hashes, and scripts needed to reproduce them.

## Label generation

The target stream must be generated independently of physical location and acquisition order.

Requirements:

- approximately 50/50 globally;
- approximately balanced within each major physical group when sample counts allow;
- no deterministic alternating pattern such as `010101...`;
- no relationship between label and device, row, address, session, or timing profile;
- retain enough provenance to audit generation without making the random generator itself an unintended model feature.

The model must never receive the label generator state or any field from which the next label can be reconstructed.

## Feature policy

### Allowed by default

- raw physical trace channels;
- explicitly selected environmental measurements;
- explicitly selected controlled timing/refresh variables.

### Excluded by default

- physical address;
- row/cell identifier;
- device identity;
- acquisition order;
- filename/path encodings that contain the label;
- post-read digital value;
- ECC result that trivially reveals the target unless ECC itself is the declared channel under study;
- any feature produced after consulting the ground-truth target.

Excluded fields may still exist in metadata because they are essential for group-aware splitting and auditing.

## Preprocessing

All preprocessing learned from data must be fit on the training partition only.

This includes:

- normalization statistics;
- PCA or learned dimensionality reduction;
- feature selection;
- imputation values;
- trace templates;
- threshold selection.

A global normalization pass over train and test data can create subtle leakage and should be avoided.

## Split materialization

For serious results, persist exact sample IDs for train/validation/test rather than recreating splits ad hoc.

A split record should state:

```text
split_name
split_strategy
random_seed
grouping_keys
train_sample_ids
validation_sample_ids
test_sample_ids
dataset_hash
```

This makes model comparisons use exactly the same evidence.

## Dataset versioning

Every published or compared dataset should have a stable fingerprint derived from its manifest and relevant acquisition metadata. Changing filtering, alignment, labels, or preprocessing should produce a new dataset version.

## Sensitive data policy

Initial and baseline datasets use generated random target patterns. Do not collect unrelated application memory, credentials, personal data, or third-party content for SenseTrace experiments.
