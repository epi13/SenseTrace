# Phase 0 v2 development power study — 2026-09-01

## Purpose

The frozen v1 evidence detected the declared 0.10-sigma synthetic signal in
68% of fresh replicates, below the predeclared 80% requirement. This study was
therefore run as a development/design study, not as a rerun of the v1 gate and
not as permission to start Phase 1A.

Run: `phase0-power-study-20260901T155735Z-b0ef3d89`.

- Candidate sample counts: 1,000, 2,000, and 4,000.
- Independent replicates per candidate and balance mode: 20.
- Conditions: independent null, exact-parent shuffled, injected 0.10-sigma,
  and fresh gate-validation ensembles.
- Balance modes: `global_balance_only` and `group_stratified_balance`.
- Development permutations: 20, explicitly recorded for candidate evaluation;
  the final v2 gate retains 100.
- Alpha: 0.05; maximum-statistic family-wise rule; Wilson intervals reported.

## Observed results

| candidate samples | injected detection | detection Wilson 95% | fresh null FPR | fresh shuffled FPR | eligible |
|---:|---:|---:|---:|---:|:---:|
| 1,000 | 40.0% | [26.3%, 55.4%] | 20.0% | 10.0% | no |
| 2,000 | 92.5% | [80.1%, 97.4%] | 7.5% | 12.5% | no |
| 4,000 | 100.0% | [91.2%, 100.0%] | 25.0% | 7.5% | no |

No candidate met both the 80% injected-detection target and the development
null/shuffled calibration checks. Consistent with the frozen v2 selection rule,
4,000 samples is recorded as the largest conservative candidate, but this is
not a passing gate and does not authorize physical acquisition.

The candidate artifacts are retained under the ignored local `runs/` directory.
They include distinct materialized dataset fingerprints for each ensemble and
the complete per-candidate reports. The final v2 gate must use a new seed and a
new run directory; it must not reuse any candidate validation data.

## Claim boundary

This is synthetic development/power evidence only. It does not establish a
physical DRAM-state inference result, DRAM topology, or cross-host/device
generalization. Phase 1A remains closed pending the separate final v2 gate.
