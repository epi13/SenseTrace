# ADR-010: Native timing-path sensitivity calibration

## Decision

SenseTrace calibrates the exact native worker timing path with a separately
namespaced positive control. The native kernel can add a predeclared TSC-
deadline delay after the volatile load and before the serialized end timestamp
for label 1. The perturbation is measured in requested TSC cycles, is explicit
in provenance, and is not a physical-memory effect.

The protocol sweeps a fixed magnitude grid including zero, uses independent
freshly seeded datasets, evaluates the same D session holdout and model/metric
maximum statistic, and retains paired median-latency diagnostics, uncertainty,
session dependence, boot dependence, and shuffled-label controls. Development
data selects the smallest magnitude meeting the predeclared power target. A
fresh frozen validation ensemble uses new seeds and the development critical
value exactly once; it cannot retune the model, threshold, or magnitude.

## Consequence

The result is a detection floor for the instrumentation and analysis path, not
evidence of DRAM-state inference. The physical Phase 1A dataset is never
modified or contaminated by calibration controls. A noisy or failed
calibration is evidence to improve measurement fidelity, not a reason to
increase physical sample counts blindly.

The report schema now keeps development shuffled-label statistics separate from
fresh/frozen shuffled-label statistics. The retained 2026-09-01 worker report
has the same numeric rate for both ensembles, so its historical values remain
unchanged; future reports name the source ensemble explicitly. Six-replicate
tail estimates are marked as pipeline sanity checks with broad uncertainty.
