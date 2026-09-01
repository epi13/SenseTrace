# Phase 0 protocol v2

`phase0-protocol-v2` is the frozen protocol family for a power-calibrated
Phase 1A gate. It exists because the v1 evidence detected the declared
0.10-sigma synthetic signal in only 68% of fresh validation replicates, below
the predeclared 80% requirement. That result is treated as an underpowered
design finding, not as permission to rerun until it passes.

## Design and freeze procedure

1. Run `sensetrace calibrate phase0-power` with a predeclared sample-count grid
   around the v1 1,000-sample design.
2. Use independent materialized calibration and fresh validation ensembles for
   every candidate, with independent acquisition, label, trace, split, model,
   and permutation seeds.
3. Choose the smallest candidate that reaches the predeclared 80% injected
   detection target while its development null and shuffled controls remain
   calibrated. If none qualifies, record the largest candidate but leave the
   gate closed.
4. Freeze the selected sample count, model/metric family, alpha, balance modes,
   split, and decision rule in a v2 configuration and protocol hash.
5. Run exactly one separately seeded final v2 calibration. Its fresh
   gate-validation ensemble is not part of the power study.

Candidate runs test only the target strength rather than spending compute on a
signal-strength curve. They may use the explicitly recorded development
permutation count in `calibration.power_study.permutation_repetitions`; the
final v2 configuration retains its full frozen permutation count.

The power study is development evidence only. It cannot open Phase 1A. The
final gate retains `alpha = 0.05`, empirical maximum-statistic family-wise
control, independent null/shuffled/injected materializations, Wilson reporting
of the false-positive rate, and the 80% minimum injected detection rate.

## Final gate rule

The final v2 report opens the Phase 1A gate only when all of the following are
true:

- fresh null and shuffled false-positive rates satisfy the frozen tolerance;
- the 0.10-sigma injected condition reaches at least 80% calibrated detection;
- the maximum-statistic rule and protocol hash are present;
- all materialized ensembles have independent dataset fingerprints.

If the final v2 gate fails, Phase 1A remains closed. A new gate requires a
documented implementation or operational failure, or a separately designed
future protocol; favorable random variation is not a rerun reason.

In the 2026-09-01 development study, no candidate qualified: 1,000 samples
missed power, 2,000 failed the shuffled false-positive check, and 4,000 failed
the null false-positive check. The selected 4,000-sample value is therefore a
conservative candidate for the one final gate, not evidence that the v2 gate
has passed. See [the study evidence](evidence/phase0-power-study-v2-2026-09-01.md).

## Claim boundary

Protocol v2 calibrates the software pipeline on synthetic data only. It does
not establish a physical DRAM-state inference result, a DRAM topology mapping,
or a cross-host/device result.
