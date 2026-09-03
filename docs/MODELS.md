# SenseTrace Model Strategy

## Design principle

SenseTrace should prefer the smallest model that can reliably detect a physical information channel.

The objective is not to maximize benchmark accuracy with unrestricted capacity. It is to answer:

> What is the lowest-complexity function that extracts reproducible target-state information from the measured signal?

## Model ladder

The fragmented-evidence path compares receivers on the same immutable packet
dataset and split fingerprints. Its ladder is:

1. logistic regression on bounded packet summaries;
2. boosted trees on those summaries;
3. small CNN/TCN fragment encoder;
4. weak-evidence aggregator;
5. JEPA-like encoder with a linear probe;
6. JEPA-like encoder with a tiny MLP probe;
7. predictive-coding latent refinement;
8. JEPA + predictive-coding hybrid.

The hybrid keeps latent width, predictor width, and refinement steps explicit.
Training consumes a re-openable batch factory, and evaluation uses fixed-size
histograms rather than retaining a full prediction vector. Self-supervised
losses and supervised state metrics remain separate.

### 0. Logistic regression

Use on engineered scalar features and simple summary statistics.

Purpose:

- establish a linear baseline;
- expose obvious leakage;
- provide a highly interpretable reference.

### 1. Boosted trees

Use on engineered physical features such as:

- peak amplitude;
- minimum/maximum;
- rise or fall time;
- settling time;
- integrated power/voltage;
- selected derivatives;
- spectral-band energy;
- timing/refresh/environmental variables.

Purpose:

- detect nonlinear relationships without deep sequence models;
- rank feature importance;
- generate hypotheses about physical mechanisms.

### 2. Tiny MLP

Use for low-dimensional tabular measurements.

Suggested initial budget:

```text
1K-10K parameters
```

Avoid using an MLP directly on long raw traces unless it is serving as an intentional control.

### 3. Tiny 1D CNN

Primary first architecture for raw physical traces.

A starting topology:

```text
input: channels x time
  -> Conv1D(16, kernel=11)
  -> Conv1D(32, kernel=7)
  -> Conv1D(32, kernel=5)
  -> Conv1D(64, kernel=3)
  -> global average pooling
  -> Dense(32-64)
  -> sigmoid
```

Use normalization and pooling choices that do not accidentally mix train/test statistics.

Expected initial parameter range:

```text
10K-100K
```

Why CNN first:

- likely useful signals are local temporal structures;
- convolution naturally detects small shifts, slopes, ringing, transients, and settling behavior;
- parameter count remains small;
- temporal saliency/ablation is straightforward.

### 4. Residual 1D CNN

If the tiny CNN detects a signal but underfits, add shallow residual blocks rather than immediately increasing width dramatically.

Target range:

```text
20K-150K parameters
```

This is the expected primary architecture for the first serious trace experiments.

### 5. Temporal Convolutional Network

Use dilated causal/non-causal convolutions when the useful relationship spans a wider time window than a compact CNN captures efficiently.

Target range:

```text
30K-250K parameters
```

A TCN should be justified by evidence that long-range temporal context improves strict-holdout performance.

### 6. Multi-branch trace + metadata model

When environmental or controlled experimental variables are useful, process them separately from the waveform.

```text
raw trace ----------------> 1D CNN ----\
                                      concatenate -> small head -> P(bit=1)
allowed numeric metadata -> small MLP --/
```

This keeps waveform representation distinct from scalar experimental context and makes ablation easier.

## Architectures not favored initially

### Transformer

Do not begin with a transformer. The expected task is narrow, data are regular time series, and the experiment benefits from low model capacity and interpretability.

A transformer-like model becomes interesting only if:

- long-range interactions matter;
- CNN/TCN models plateau for defensible reasons;
- strict holdouts still show signal;
- parameter growth is controlled.

### Large foundation model

Out of scope for initial SenseTrace inference. A large model would make nuisance-feature memorization easier and physical interpretation harder.

## Capacity sweep

For any positive result, train a controlled capacity series where practical:

```text
~1K
~5K
~10K
~25K
~50K
~100K
~250K
```

Plot performance against parameter count under the same split.

A result that appears at low capacity and saturates is especially interesting because it suggests a relatively simple physical mapping.

## Input strategy

Start with the most direct measured signal and add context incrementally.

Recommended progression:

1. one raw trace channel;
2. each channel independently;
3. all raw trace channels;
4. trace plus controlled timing variables;
5. trace plus environmental variables;
6. complete permitted feature set.

Never include address/device identity in the primary inference input. Those belong to grouping and diagnostics.

## Training objective

Initial task:

```text
binary classification
loss: binary cross entropy
output: calibrated or calibratable P(bit = 1)
```

Class weighting should normally be unnecessary because acquisition is balanced. If weighting is needed, treat that as a dataset-quality warning and document it.

## Reproducibility

Record for every training run:

- dataset fingerprint;
- split fingerprint;
- architecture name/version;
- parameter count;
- optimizer and learning rate;
- batch size;
- epoch/early-stop rule;
- random seeds;
- preprocessing version;
- software/hardware environment.

## Interpretation tools

For positive models, favor tools that help locate the signal:

- temporal masking/window ablation;
- channel ablation;
- feature permutation for tabular baselines;
- saliency only as a hypothesis generator, not proof;
- retraining on isolated trace windows;
- comparison against physically motivated engineered features.

The goal is to turn a predictive result into a testable explanation of the underlying measurement channel.

## Preregistered worker-03 tournament

`worker03-fragmented-exact-host-v1` runs every declared candidate against the
same immutable packet fingerprint, split fingerprint, preprocessing contract,
and feature policy. The bounded CPU implementation uses packet-summary
features for all candidates when a native deep-learning implementation is not
explicitly enabled; it reports that fallback rather than calling it a raw
trace CNN or JEPA result. Validation balanced accuracy selects the declared
candidate, and the unmodified selected test partition is evaluated once.

The report also includes an artificial positive sensitivity control, a
label-independent null, keyed shuffled labels, fragment-relation rotation,
metadata-input invariance, and a single-fragment baseline. These controls can
falsify or qualify the pipeline; none upgrades native CPU timing into physical
DRAM or hidden-state evidence.
