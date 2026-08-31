# SenseTrace Experimental Protocol

## 1. Research question

SenseTrace tests whether physical observations surrounding controlled DRAM operations contain reproducible information about a hidden, balanced random bit beyond the ordinary digital value returned by a memory read.

The experiment is intentionally framed as a binary inference problem:

```text
Y = measured physical observations
X = hidden randomly written bit
model(Y) -> P(X = 1)
```

The model is an instrument for detecting weak relationships. It does not create information that is absent from `Y`.

## 2. Primary hypotheses

### Null hypothesis H0

After controlling for experimental leakage and class imbalance, measurements carry no usable information about the hidden state and held-out prediction remains at chance.

```text
balanced accuracy ~= 0.50
```

### Alternative hypothesis H1

One or more measured channels contain state-correlated information and a model can predict held-out random bits above chance with reproducible statistical confidence.

The strongest version of H1 is generalization to a physical DRAM device not represented in training.

## 3. Experimental principles

1. **Random targets.** Use generated random bits so language, file formats, program structure, and other semantic priors cannot help prediction.
2. **Balanced targets.** Maintain approximately 50/50 labels globally and within important physical groups.
3. **Known writes.** The experiment controls the written target state so ground truth is unambiguous.
4. **Separate observation from label.** The ordinary digital read result must never enter model features.
5. **Group-aware holdouts.** Random sample splitting is insufficient because repeated measurements of the same physical cell can leak identity.
6. **Small models first.** Start with interpretable baselines before increasing capacity.
7. **Negative controls.** A valid pipeline must recover chance performance when the physical relationship is intentionally removed.
8. **Reproducibility.** Every run records hardware, acquisition, environment, split, model, seed, and software configuration.

## 4. Acquisition cycle

A canonical sample should follow this conceptual sequence:

```text
choose target location
        |
generate balanced random label
        |
write known state
        |
wait controlled interval / establish condition
        |
perform controlled DRAM operation
        |
record physical observations
        |
record ground-truth label separately
        |
persist sample + experiment metadata
```

The exact operation and available observations depend on the experimental phase.

## 5. Candidate observation channels

SenseTrace may investigate, where experimentally accessible:

- operation timing and latency;
- refresh age and controlled retention interval;
- supply-current or board-level power traces;
- board-level voltage traces;
- temperature;
- command timing parameters;
- activation/precharge history;
- neighboring-row activity under controlled conditions;
- analog transient features captured by suitable instrumentation;
- derived waveform features such as slope, peak, settling time, energy, or spectral components.

Location identifiers and device identifiers may be stored for grouping and audit purposes but are **not model inputs by default**.

## 6. Experimental phases

### Phase 0 — pipeline controls

Goal: prove that the software pipeline can distinguish a genuine signal from no signal before interpreting any DRAM result.

Required datasets:

1. **Pure null:** random labels independent of random/synthetic traces. Expected result: chance.
2. **Injected weak signal:** insert a small known label-dependent perturbation into synthetic traces. Expected result: detectable above chance.
3. **Shuffled-label control:** real or synthetic measurements with labels randomly permuted. Expected result: chance.

Phase 0 is complete only when the pipeline passes all three controls.

### Phase 1 — accessible DRAM observables

Use non-invasive measurements and controllable DRAM conditions on owned research hardware. Establish whether timing, refresh, power, temperature, or other accessible channels contain detectable state information.

The priority is clean methodology rather than maximizing measurement depth.

### Phase 2 — controlled memory interface

Introduce tighter command-level control when suitable research hardware is available, for example an FPGA-based DRAM controller. This phase can systematically vary activation, precharge, refresh, and timing conditions while preserving the same validation rules.

### Phase 3 — deeper physical instrumentation

If earlier results justify it, capture richer electrical behavior from owned lab hardware with appropriate instrumentation. Any deeper measurement should be treated as a new acquisition domain and revalidated from null controls upward.

## 7. Minimum experiment matrix

For every candidate channel, collect enough data to answer at least:

- Does the channel work on repeated observations of known cells?
- Does performance survive a holdout of physical cells?
- Does it survive a holdout of rows/banks or other physical regions?
- Does it survive a later acquisition session?
- Does it survive a new DIMM/device?
- Does removing the channel destroy the gain?
- Does label shuffling return the model to chance?

## 8. Baselines

Every experiment should report at minimum:

1. majority-class baseline;
2. logistic regression;
3. a nonlinear tabular baseline when engineered features exist;
4. a small neural model appropriate to the raw measurement type.

A complex model is not evidence by itself. Results are strongest when simple and complex models independently detect the same channel.

## 9. Primary metrics

Report:

- balanced accuracy;
- AUROC;
- confusion matrix;
- sample count and label balance;
- repeated-run mean and variance;
- uncertainty/confidence interval;
- model parameter count.

For very small effects near 50%, statistical testing and repeated acquisition sessions are mandatory. A tiny numerical gain on a huge correlated dataset is not enough by itself.

## 10. Channel ablation

Once a multichannel model performs above chance, retrain while removing one channel at a time.

Example:

```text
all channels                  0.612
- power                       0.611
- timing                      0.553
- refresh age                 0.610
timing only                   0.548
```

The goal is to discover *where the information lives*, not merely to maximize the score.

## 11. Interpretation standard

Results should be described conservatively:

- **Chance under strict holdout:** no detected channel at current measurement sensitivity.
- **Above chance on known cells only:** possible cell-specific physical fingerprint or experimental leakage; not yet a general state channel.
- **Above chance on unseen cells/regions:** evidence of a more general relationship.
- **Above chance on unseen DIMMs:** strong evidence that the measured phenomenon generalizes across devices.

No result should be described as predicting arbitrary memory unless the tested protocol actually supports that claim.

## 12. Research scope

Use hardware the researcher owns or is explicitly authorized to test. Initial experiments should use generated patterns rather than sensitive application data. The project is aimed at physical characterization of memory systems and reproducible measurement science.
