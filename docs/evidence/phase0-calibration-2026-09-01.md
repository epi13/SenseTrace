# Phase 0 calibration evidence — 2026-09-01

## Observed fact

Full example-config run ID: `phase0-calibration-20260901T051610Z-65c134bc`.

- Protocol: `phase0-protocol-v1`.
- Protocol hash: `8bd9ce4b32704dfcd9998ca07549fb4507d090045dd479b50695fbe8b6fe7939`.
- Calibration: 50 null, 50 shuffled, and 50 injected replicates for each of
  `global_balance_only` and `group_stratified_balance` (100 per condition total).
- Fresh validation: 100 null, 100 shuffled, and 100 injected datasets, all with
  fingerprints distinct from the calibration ensemble.
- The injected-strength curve materialized 0.05, 0.10, and 0.20 sigma levels,
  with 50 replicates per non-primary curve level and balance mode.
- Alpha was 0.05. The empirical null critical maximum statistic was
  `0.0840286`.
- Calibration null: 5/100 positive, estimated false-positive rate 5.0%, Wilson
  95% interval `[0.0215, 0.1118]`.
- Fresh null: 7/100 positive, estimated false-positive rate 7.0%, Wilson 95%
  interval `[0.0343, 0.1375]`.
- Fresh shuffled: 7/100 positive, with the same 7.0% rate and Wilson interval.
- Fresh injected detection was 68/100 (68%), below the configured minimum of
  80%. Null, shuffled, and pipeline false-positive checks passed, but the
  overall Phase 1 gate correctly remained closed.

## Historical null investigation

The old boosted-tree and shuffled-logistic values are ordinary under the new
empirical null:

| historical value | marginal null percentile | raw empirical p | max-statistic percentile | family-wise p |
| --- | ---: | ---: | ---: | ---: |
| boosted-tree BA 0.5318 | 86% | 0.149 | 61% | 0.396 |
| boosted-tree AUROC 0.5486 | 91% | 0.099 | 79% | 0.218 |
| shuffled logistic BA 0.5400 | 89% | 0.119 | 74% | 0.267 |

These values are not re-read from the old raw dataset; they are fixed historical
comparators evaluated against the new null distribution. The evidence supports
finite-sample/model-search variation as a sufficient explanation. It does not
prove that every possible construction or feature family is null.

## Randomization and native controls

The fixed-dataset Monte Carlo permutation tests used 100 permutations and plus-one
empirical p-values. Global-balance data permuted within the materialized dataset
(`p = 0.1386`); group-stratified data permuted within synthetic location
(`p = 0.0099`). The latter is the expected randomization evidence for the known
injected signal when its within-location balance is preserved, not an independent
physical result.

Native calibration run: 200 repetitions on the controller's x86 host. Cached
load median was 22 TSC cycles; CLFLUSH load median was 188 cycles. Timer-only
median was 24 cycles and the idle control median was 24 cycles. These are raw
cycle-path controls; CLFLUSH does not establish a DRAM or DRAM-row access.

## Interpretation and unresolved confounds

The calibrated decision rule no longer treats every score above 0.5 as a
failure, and it does not reject a validation ensemble because one null replicate
fires at alpha. The maximum statistic controls the tested model/metric family;
the validation false-positive rate estimates how often the complete pipeline
opens its positive decision under known null data. The full campaign therefore
closed Phase 1 for a substantive reason: the configured 0.10-sigma injection
was detected in only 68% of fresh replicates, below the predeclared 80% power
criterion. The implementation did not weaken the control to force a pass.

The 100-replicate intervals are still not a substitute for a power analysis over
the intended physical effect size. Training-seed means are used in the replicate
statistic and each seed is recorded; repeated fits on one dataset are not treated
as independent datasets.

## Claim boundary

This evidence validates a synthetic statistical gate and native timing controls.
It establishes no physical DRAM-state inference result, no DRAM-row mapping, and
no cross-host generalization.
