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
| `acquisition_session_id` | Globally unambiguous actual acquisition-session UUID | No |
| `acquisition_block` | Logical block within an acquisition session | No |
| `device_id` | DIMM/chip grouping identifier | No |
| `bank_id` | Physical grouping where known | No |
| `row_id` | Physical grouping where known | No |
| `cell_or_offset_id` | Fine physical grouping where known | No |
| `allocation_id` | Fresh anonymous allocation identity for one physical session incarnation | No |
| `virtual_location_id` | Controlled location in the anonymous virtual buffer | No |
| `buffer_offset_id` | Controlled buffer-word offset identifier | No |
| `pair_id` / `trial_pair_id` | Matched label-0/label-1 pair grouping | No |
| `pair_order` / `pair_position` | Counterbalance and temporal-position audit fields | No |
| `trial_index` | Acquisition ordering/audit | No |
| `temperature_c` | Environmental measurement | Optional |
| `vdd_v` | Supply measurement | Optional |
| `refresh_age_ns` | Time since relevant refresh event | Optional |
| `wait_ns` | Controlled delay before observation | Optional |
| `timing_profile` | Acquisition timing configuration | Optional/encoded |
| `channel_mask` | Which measurement channels are valid | No/direct loader use |
| `seed_id` | Reproducibility reference, not raw secret/random state | No |
| `measurement_primitive` | Named operation/observation implementation | No |
| `measurement_primitive_capabilities` | Known/unknown/unsupported capability contract | No |
| `access_state_oracle_provenance` | Independent-oracle status and strength | No |
| `physical_observation_semantics` | Conservative description of what the trace measures | No |

Exact physical topology fields may be unavailable on early hardware. Missing topology is acceptable, but the absence must be recorded because it limits which holdout claims can be made.

Primitive provenance is a separate namespace from model features. An oracle may
be exact, probabilistic, partial, or unavailable. A requested cache control is
not evidence of the resulting physical memory layer. Addresses, allocation and
session identities, oracle identity/results, and boot/order metadata remain
audit-only unless a future protocol explicitly justifies an ablation.

`location_id` and `cell_or_offset_id` are retained for compatibility with older
datasets. In Phase 1A they are virtual buffer identifiers, not known physical
DRAM locations. New code should use `virtual_location_id`, `buffer_offset_id`,
and `acquisition_session_id` explicitly.

## Acquisition-session and campaign provenance

An acquisition session is one independently started backend with its own fresh
controlled memory allocation. It is not a slice of a continuous stream. Its
`session.json` records the UUID, timestamps, host inventory snapshot, boot ID,
allocation/locking result, label-stream fingerprint, measurement-kernel and
cache-control provenance, journal boundaries, environment snapshot, and
configuration/code hashes.

If a physical session is interrupted, finalized shards remain in its immutable
interrupted source directory. Recovery records an append-only interruption
decision and starts a fresh session/allocation with a new UUID, allocation ID,
host snapshot, and parent-session reference. A completed campaign never
silently combines an interrupted source with its replacement.

A Phase 1A campaign may contain multiple source session datasets. The combined
condition manifest records all source dataset fingerprints and embeds the
individual source manifests in `source-manifests.json`; finalized source shards
are never treated as anonymous rows. The intended conceptual hierarchy is:

```text
campaign -> boot -> acquisition session -> acquisition block
         -> virtual location -> pair -> trial
```

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
