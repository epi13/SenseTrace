# SenseTrace

**SenseTrace** is an experimental research project for measuring how much information about DRAM state exists outside the conventional digital read interface.

The core question is deliberately narrow:

> Can weak physical observations of DRAM behavior predict a balanced, randomly written hidden bit above chance without directly consuming the ordinary digital read result as an input?

SenseTrace treats a learned model as an **information detector**, not as an oracle. If a model generalizes above 50% on genuinely unseen random bits, unseen physical locations, and eventually unseen DIMMs, that is evidence that the measured channel carries information about the stored state.

## Research hypothesis

Let:

- `X` be a hidden, balanced random DRAM state (`0` or `1`).
- `Y` be measurements available around a controlled DRAM operation: timing, power, voltage, temperature, refresh age, analog trace features, or other physical observations.

SenseTrace asks whether:

```text
I(X; Y) > 0
```

in a reproducible setting.

A classifier is used as a practical probe of that relationship:

```text
physical observations -> model -> P(bit = 1)
```

A result above chance is interesting only if it survives strict controls against address memorization, device fingerprinting, temporal leakage, label imbalance, and train/test contamination.

## What SenseTrace is not

SenseTrace does **not** assume that arbitrary unknown memory can be predicted from nothing. Information-theoretic limits still apply. Any successful inference must come from measurable information correlated with the target state or from prior knowledge about the data.

The initial experiment therefore uses **random, balanced labels** so semantic structure cannot help the model reconstruct the answer.

## Initial experiment

1. Generate a cryptographically random, balanced stream of `0` and `1` labels.
2. Write known target states to controlled DRAM locations on research hardware.
3. Perform a controlled operation while collecting permitted physical observations.
4. Store the observation and hidden label as one sample.
5. Train deliberately small models first.
6. Evaluate progressively harder holdouts:
   - repeated measurements on known cells;
   - unseen cells;
   - unseen rows/banks/subarrays where topology is available;
   - unseen acquisition sessions;
   - unseen DIMMs/devices.
7. Ablate input channels to identify which measurements actually carry predictive information.

## Model ladder

SenseTrace starts with the simplest model capable of detecting a signal:

| Stage | Model | Typical purpose |
| --- | --- | --- |
| 0 | Logistic regression | Linear leakage baseline |
| 1 | Boosted trees | Nonlinear engineered-feature baseline |
| 2 | Tiny MLP | Tabular physical measurements |
| 3 | Tiny 1D CNN | Raw waveform / time-series traces |
| 4 | Residual 1D CNN | Primary learned trace model |
| 5 | Small TCN | Longer temporal relationships |
| 6 | Multi-branch CNN + metadata MLP | Trace plus environmental metadata |

The expected useful range for the first neural experiments is roughly **10K-100K parameters**. Larger models are not a goal; model capacity is increased only when controlled results justify it.

## Success criteria

The first meaningful milestone is not 90% or 99% accuracy. It is:

> **Statistically reproducible performance above 50% on balanced random bits under a holdout that prevents the model from memorizing the tested physical locations.**

The strongest early result would additionally generalize to a DIMM not represented during training.

Every reported result should include:

- sample count;
- class balance;
- exact split strategy;
- device/session composition;
- model parameter count;
- balanced accuracy and AUROC;
- confidence interval or equivalent uncertainty estimate;
- repeated-run variance;
- channel ablations;
- negative controls.

## Dedicated experiment host

Early unattended experiments are intended to run on `worker-03`, a dedicated 32 GB RAM / 2 GB VRAM system with HDD-backed storage and otherwise closely matched hardware to the primary workstation.

The runner should be designed as a resumable experiment appliance rather than an interactive script:

```text
boot
  -> start runner
  -> recover experiment journal
  -> validate last completed shard
  -> resume acquisition
  -> train/evaluate configured baselines when requested
  -> continue until experiment completion or explicit stop
```

The HDD is not expected to be a limiting factor if traces are buffered and written as large sequential shards rather than one file per sample. Future timing, refresh, or disturbance experiments should assume the host may crash and must preserve already completed acquisition work through temporary files, checksums, atomic shard finalization, and recovery journaling.

The small initial models are CPU-capable, so the 2 GB GPU is optional rather than a requirement. Acquisition integrity and validation take priority over training throughput.

Because the primary workstation is closely matched to `worker-03`, it may later provide a useful small cross-machine holdout dataset to test whether a learned signal transfers beyond a specific DIMM or host. Prolonged or intentionally destabilizing tests should remain on dedicated research hardware.

See [worker-03 experiment host](docs/WORKER03.md) for the runner, storage, crash-recovery, reproducibility, and cross-machine validation plan.

## Repository layout

```text
SenseTrace/
├── README.md
├── configs/
│   ├── phase0.example.yaml
│   └── worker03.example.yaml
├── data/
│   └── README.md
└── docs/
    ├── DATASET.md
    ├── EXPERIMENT.md
    ├── MODELS.md
    ├── VALIDATION.md
    └── WORKER03.md
```

## Experimental phases

**Phase 0 — pipeline validation**  
Synthetic/no-signal and intentionally leaked controls prove that the collection, splitting, training, and metrics pipeline behaves correctly.

**Phase 1 — accessible DRAM observables**  
Collect timing, refresh-age, environmental, and board-level measurements available without invasive chip modification.

**Phase 2 — controlled memory interface**  
Use research hardware such as an FPGA-based memory controller where appropriate to gain tighter control over command timing and acquisition.

**Phase 3 — deeper physical instrumentation**  
Investigate richer electrical traces on owned lab hardware when instrumentation makes this scientifically justified.

Each phase must retain the same random-label and holdout discipline so increased instrumentation does not weaken the experimental standard.

## Safety and research scope

SenseTrace is intended for controlled experiments on hardware the researcher owns or is explicitly authorized to test. Initial datasets should contain generated random patterns rather than user, application, credential, or other sensitive memory contents.

The purpose of the project is to characterize physical information channels in memory systems, not to develop tooling for extracting third-party memory.

## Documentation

- [Experiment protocol](docs/EXPERIMENT.md)
- [Dataset contract](docs/DATASET.md)
- [Model strategy](docs/MODELS.md)
- [Validation and anti-leakage rules](docs/VALIDATION.md)
- [worker-03 experiment host](docs/WORKER03.md)

## Status

**Experiment definition / scaffold.** No claim of DRAM-state inference has been established yet. The first objective is to build a measurement and validation pipeline capable of proving both positive and negative results cleanly.
