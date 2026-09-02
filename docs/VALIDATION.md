# SenseTrace Validation and Anti-Leakage Rules

## Why validation is the experiment

For SenseTrace, a model score is meaningful only if the split and controls rule out easier explanations. Repeated measurements of the same memory locations can create strong fingerprints unrelated to the current stored state.

The validation design therefore matters at least as much as the model architecture.

## Required split hierarchy

Results should progress through increasingly strict holdouts.

### Split A — sample holdout

Random samples are held out while physical locations may appear in both train and test.

Use only as an early pipeline check. Do not treat this as evidence of general state inference.

### Split B — unseen physical locations

All measurements associated with a target cell/offset are assigned to only one partition.

This tests whether the model can generalize beyond memorized cell fingerprints.

### Split C — unseen acquisition blocks / available regions

Hold out complete rows, banks, subarrays, or the closest available grouping.

This holds out complete logical acquisition blocks. Commodity SenseTrace does
not know DRAM topology, so no row/bank/subarray claim is made unless a future
instrument explicitly supplies those grouping fields.

### Split D — unseen acquisition sessions

Train and test data come from different collection sessions, preferably separated in time and after reinitialization of the acquisition setup.

This tests session-specific drift and instrument artifacts.

### Split E — unseen OS boot

Complete genuine OS boot groups are held out from each other and used only for
final evaluation. SenseTrace requires at least three real boot groups before
this split is available. This is not an unseen-device split; device-independent
evidence requires a separately declared cross-device campaign.

This is the strongest currently implemented commodity-host boot-boundary test.

## Forbidden shortcuts

Unless a field is the explicit subject of an ablation study, the model should not receive:

- target address;
- row/cell identifier;
- device identifier;
- acquisition order;
- filenames that encode target state;
- the normal digital read value;
- target-derived preprocessing outputs;
- generator state that can reconstruct labels.

## Negative controls

Every serious run should include controls capable of falsifying the pipeline.

### Label permutation

Shuffle labels after acquisition while preserving all features and split grouping.

Expected result: chance.

### Synthetic null

Generate features statistically independent of balanced labels.

Expected result: chance.

### Injected weak signal

Add a small, known label-dependent feature to otherwise null traces.

Expected result: the pipeline detects it, and ablation of the injected region removes the gain.

### Metadata-only model

Train a model using only fields that should contain no target-state information, such as permitted non-identifying run metadata.

Unexpected above-chance performance is a warning of dataset construction bias.

### Address-only audit

As an explicit diagnostic, test whether grouping/location metadata predicts the label. This model is never considered a valid SenseTrace inference result; it is used to reveal accidental label imbalance across locations.

## Balance checks

Before training, report label frequencies for:

- entire dataset;
- each split;
- each device;
- each session;
- each major physical group;
- each acquisition/timing configuration.

Any strong local imbalance should be corrected or explicitly controlled.

## Statistical reporting

Small effects near 50% are expected to be scientifically interesting, so uncertainty must be visible.

At minimum report:

- number of independent test samples;
- balanced accuracy;
- AUROC;
- confidence interval;
- repeated training seeds;
- repeated acquisition sessions where possible.

Because many measurements may come from the same physical location or session, do not assume every sample is statistically independent. Where practical, bootstrap or aggregate by the grouping unit used for the claim.

SenseTrace records the CI unit explicitly. Phase 0 defaults to a session-level bootstrap; a result must say `CI unit: session_id` (or `sample` when sample independence is the intended claim). Group-aware intervals resample complete groups rather than treating repeated observations as independent.

Phase 1A materializes and independently evaluates every available split in the
hierarchy. The report includes availability, grouping keys, partition sample
and class composition, group counts, dataset/split fingerprints, model seeds,
metrics, uncertainty, and the claim boundary. An unavailable session or boot
split remains unavailable; it is never replaced by Split B without saying so.

Phase 1A also reports a predeclared paired diagnostic: the within-pair change in
sample median timing, `median(label=1) - median(label=0)`, with pair-level
sign-flip testing and a confidence interval from complete session/block
clusters. Trace mean, first-access latency, and p95 are secondary diagnostics
and are labeled exploratory rather than selected as alternate primary outcomes.

Every Phase 0 condition exposes construction audits for metadata-only prediction, identity-only prediction in a controlled audit mode, trial-order prediction, label balance by device/session/row/cell, and train/test feature-distribution differences. These are audit artifacts only and cannot establish SenseTrace inference.

Phase 1A adds exact pair-order balance and audit channels for pair position,
acquisition drift, CPU migration, frequency/governor regime, thermal state,
cache state, boot/session identity, and block identity. These channels are
reported to diagnose nuisance explanations and remain excluded from the
default feature matrix.

### Phase 1A holdout boundary and native sensitivity

The strict levels have distinct declared identities: A groups repeated trials
within a virtual location and pair, B groups virtual locations, C groups
acquisition blocks, D groups acquisition sessions, and E groups only OS boot
IDs. E is unavailable with fewer than three real boot groups. A session ID or
allocation ID cannot be used to manufacture a cross-boot split. Every run
records split invariants for exact coverage, group disjointness, unique sample
IDs, and duplicated nominal partitions.

The exact native timing path is calibrated separately with a positive control
that adds a known TSC-deadline delay after the volatile load inside the native
timing window. The sweep includes a zero null, multiple predeclared magnitudes,
fresh independently seeded datasets, the D holdout, the same model/metric
maximum-statistic rule, paired statistics, uncertainty, session/boot
dependence, and shuffled-label controls. Development data selects a frozen
candidate; fresh validation uses new seeds and the development critical value
without retuning. This estimates an instrumentation detection floor only. It
does not constitute a physical DRAM-state result.

Use the separate namespace with:

```bash
sensetrace calibrate native-sensitivity \
  --config configs/worker03.example.yaml --output runs/native-sensitivity
```

Raw timing values are retained. Quantiles, autocorrelation, outlier fractions,
CPU/frequency/thermal state, cache-control separation, and acquisition-order
drift are audit diagnostics; no noisy trace is removed merely because it hurts
a result.

The worker-03 validation evidence is split by claim scope: [genuine multi-boot
Phase 1A validation](evidence/worker03-multiboot-validation-2026-09-01.md)
documents the corrected E boundary and physical result, while [native-path
sensitivity evidence](evidence/native-sensitivity-worker03-2026-09-01.md)
documents the artificial timing positive control. The former was near chance on
the small three-boot test; the latter is not evidence of DRAM-state inference.
Given that result, the next physical experiment changes the measurement
primitive toward a controlled memory interface or hardware-counter access-state
oracle before increasing Phase 1A sample counts.

The worker-03 characterization result is recorded in [measurement-primitive
evidence](evidence/worker03-measurement-primitive-characterization-2026-09-02.md):
the strong cache-path control was observed, but the independent access-state
oracle remained unavailable because scoped PMU access was permission-denied.
This is the recorded B outcome, not a hidden-bit result.

Sensitivity reports distinguish development shuffled-label controls from
fresh/frozen shuffled-label controls by source ensemble. Replicate counts and
the empirical tail resolution are reported; six-replicate estimates are
pipeline sanity checks with broad Wilson intervals, not high-precision 5% tail
certification. Configure separate development null and shuffled counts when
needed with `development_null_replicates` and
`development_shuffled_replicates`.

## Measurement-primitive decision gates

The characterization suite records the host CPU/PMU vocabulary and checks
whether standard perf-visible events are discoverable without attaching to
unrelated processes. It does not broaden perf permissions. A primitive may
advance only when its null behavior, strong/weak controls, session/boot/order
and allocation dependence, and oracle status are recorded. The current
commodity result is **B — observable available but oracle weak**; a future
worker result can become **A** only if a meaningful independent access-state
oracle is actually demonstrated. Otherwise the valid outcome is **C** and the
commodity Phase 1 line transitions toward controlled-memory-interface work.

## Multiple comparisons

Searching hundreds of channels, windows, architectures, and timing settings can eventually produce an above-chance result by luck.

Treat exploratory analysis and confirmatory testing separately:

1. discover candidate signals on development data;
2. freeze preprocessing, features, architecture, and evaluation rule;
3. evaluate once on a held-out confirmation set or new acquisition session.

For stronger claims, repeat confirmation on a newly collected dataset.

Phase 0 now freezes this policy as `phase0-protocol-v1`. The calibration command
materializes independent null, shuffled, and injected datasets, records separate
acquisition/label/trace/split/model/permutation seeds, and estimates the
complete pipeline false-positive rate. It uses the empirical maximum statistic
across enabled model/metric combinations to control family-wise error at the
configured alpha (0.05 by default). The fresh gate-validation ensemble is not
the calibration ensemble. See [the frozen protocol](PHASE0-PROTOCOL-V1.md).

Use:

```bash
sensetrace calibrate phase0 --config configs/phase0.example.yaml --output runs
```

The report exposes observed statistics, empirical null percentiles, raw and
family-wise adjusted empirical p-values, the null maximum-statistic distribution,
the Wilson interval for the false-positive rate, and a within-stratum Monte Carlo
permutation test. A score above 0.5 is expected under a true null and is not by
itself a gate failure.

## Model-capacity audit

Track accuracy as capacity increases.

Example:

```text
model             parameters     balanced accuracy
logistic          ~100           0.501
CNN-small         8K             0.527
CNN-medium        42K            0.531
TCN                160K          0.530
```

A signal that saturates in a small model is often easier to interpret than one appearing only after a very large model is introduced.

## Channel ablation

For a successful multichannel model:

- remove one channel at a time;
- train single-channel models;
- mask temporal regions of raw traces;
- test physically motivated engineered features separately.

Ablation should answer not just *whether* prediction works, but *which measurement carries the information*.

## Acceptance ladder

Use conservative language tied to the strongest split passed.

| Result | Interpretation |
| --- | --- |
| Above chance only on Split A | Pipeline signal; may be memorization |
| Above chance on unseen locations | Candidate state-correlated channel |
| Above chance on unseen regions and sessions | Stronger physical evidence |
| Above chance on unseen OS boot | Cross-boot evidence on the recorded host |
| Above chance on unseen device | Cross-device generalized evidence only with a separate device campaign |
| Replicated on new hardware/acquisition | Reproducible phenomenon |

## Failure is useful

A clean 50% result under a well-powered, carefully instrumented experiment is valuable. It establishes an upper bound on what the tested channel and measurement sensitivity can reveal.

SenseTrace should preserve negative results with the same rigor as positive ones.
