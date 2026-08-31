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

### Split C — unseen physical regions

Hold out complete rows, banks, subarrays, or the closest available grouping.

This reduces the chance that nearby manufacturing variation is functioning as an identity signal.

### Split D — unseen acquisition sessions

Train and test data come from different collection sessions, preferably separated in time and after reinitialization of the acquisition setup.

This tests session-specific drift and instrument artifacts.

### Split E — unseen device

One or more complete DIMMs/chips are absent from training and used only for final evaluation.

This is the strongest early test of a device-independent physical relationship.

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

## Multiple comparisons

Searching hundreds of channels, windows, architectures, and timing settings can eventually produce an above-chance result by luck.

Treat exploratory analysis and confirmatory testing separately:

1. discover candidate signals on development data;
2. freeze preprocessing, features, architecture, and evaluation rule;
3. evaluate once on a held-out confirmation set or new acquisition session.

For stronger claims, repeat confirmation on a newly collected dataset.

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
| Above chance on unseen device | Cross-device generalized evidence |
| Replicated on new hardware/acquisition | Reproducible phenomenon |

## Failure is useful

A clean 50% result under a well-powered, carefully instrumented experiment is valuable. It establishes an upper bound on what the tested channel and measurement sensitivity can reveal.

SenseTrace should preserve negative results with the same rigor as positive ones.
