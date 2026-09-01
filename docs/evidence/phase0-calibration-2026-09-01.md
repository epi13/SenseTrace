# Phase 0 calibration evidence — 2026-09-01

## Observed fact

Local run ID: `phase0-calibration-local-20260901-v3`.

- Protocol: `phase0-protocol-v1`.
- Protocol hash: `940213259a7482f07864e7129b01482347ebdd1a7798e349c12f4f3f8fcc55d7`.
- Calibration: 10 null, 10 shuffled, and 10 injected replicates for each of
  `global_balance_only` and `group_stratified_balance` (20 per condition total).
- Fresh validation: 20 null, 20 shuffled, and 20 injected datasets, all with
  fingerprints distinct from the calibration ensemble.
- The injected-strength curve materialized 0.05, 0.10, and 0.20 sigma levels.
- Alpha was 0.05. The empirical null critical maximum statistic was
  `0.0906474`.
- Fresh null: 1/20 positive, estimated false-positive rate 5.0%, Wilson 95%
  interval `[0.0089, 0.2361]`.
- Fresh shuffled: 1/20 positive, estimated false-positive rate 5.0%, with the
  same Wilson interval.
- The configured local engineering gate passed: null, shuffled, injected
  detection rate 50% against a local minimum of 50%, and false-positive-rate
  check all passed. This run used 20 rather than the example configuration's
  50 replicates and is therefore a calibration milestone, not a high-powered
  final campaign.

## Historical null investigation

The old boosted-tree values are ordinary under the new empirical null:

| historical value | marginal null percentile | raw empirical p | max-statistic percentile | family-wise p |
| --- | ---: | ---: | ---: | ---: |
| boosted-tree BA 0.5318 | 75% | 0.286 | 45% | 0.571 |
| boosted-tree AUROC 0.5486 | 70% | 0.333 | 65% | 0.381 |
| shuffled logistic BA 0.5400 | 70% | 0.333 | 55% | 0.476 |

These values are not re-read from the old raw dataset; they are fixed historical
comparators evaluated against the new null distribution. The evidence supports
finite-sample/model-search variation as a sufficient explanation. It does not
prove that every possible construction or feature family is null.

## Randomization and native controls

The fixed-dataset Monte Carlo permutation tests used 20 permutations and plus-one
empirical p-values. Global-balance data permuted within the materialized dataset;
group-stratified data permuted within synthetic location. Both representative
injected tests returned `p = 0.0476`, consistent with the intentionally injected
signal at this small permutation count.

Native calibration run: 200 repetitions on the controller's x86 host. Cached
load median was 22 TSC cycles; CLFLUSH load median was 188 cycles. Timer-only
median was 24 cycles and the idle control median was 24 cycles. These are raw
cycle-path controls; CLFLUSH does not establish a DRAM or DRAM-row access.

## Interpretation and unresolved confounds

The calibrated decision rule no longer treats every score above 0.5 as a
failure, and it does not reject a validation ensemble because one null replicate
fires at alpha. The maximum statistic controls the tested model/metric family;
the validation false-positive rate estimates how often the complete pipeline
opens its positive decision under known null data.

The 20-replicate interval is wide. Larger unattended campaigns (50–200 or more)
are required before making a precise claim about the long-run false-positive
rate. Training-seed means are used in the replicate statistic and each seed is
recorded; repeated fits on one dataset are not treated as independent datasets.

## Claim boundary

This evidence validates a synthetic statistical gate and native timing controls.
It establishes no physical DRAM-state inference result, no DRAM-row mapping, and
no cross-host generalization.
