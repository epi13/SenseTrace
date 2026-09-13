# Construction-null falsification result — 2026-09-12

Status: completed. The fresh worker-03 campaign explains the replicated
horizon-1 `future_timing_delta_sign` result as a consequence of target/feature
construction and ordinary value sharing. It does not support a residual
physical or model-computation forecasting effect.

## Scope and execution

The production adapter exposes each timing trajectory as:

```text
state[t] = [x[t], x[t] - x[t-1]]
```

The historical target is A, the future state's delta sign:

```text
A: sign(x[t+h] - x[t+h-1])
```

At `h=1`, A shares `x[t]` with the current feature vector. The campaign also
reports B, an explicit indexing diagnostic:

```text
B: sign(x[t+h] - x[t])
```

B shares the current value at every horizon by definition and is not a
replacement for the historical A result. All targets use the frozen
`zero_is_positive` policy, and all splits are complete-trajectory grouped
holdouts.

The clean v4 run used 128 complete `random_word` samples, 32 native
repetitions per trajectory, horizons 1/2/4/8/16, three training seeds (11,
23, 37), 500 trajectory-bootstrap repetitions, and 2,000
trajectory-preserving randomizations per inferential probe. It evaluated the
observed trajectories plus continuous-IID, skewed-IID, quantized/discrete,
empirical-marginal, and raw-order-shuffle conditions. Every condition was
rebuilt through production `[x, causal difference]` construction.

The raw trajectory artifact is retained at
`evidence/self-forecasting-construction-worker03-20260912/raw_trajectories.npz`
with immutable manifest `raw_trajectories.json`; it contains timing values
only, not labels or memory contents.

## Main result

The table gives held-out balanced accuracy averaged over the three training
seeds for the current-level-only logistic probe. It is the transparent
baseline that exposes the construction effect. The combined logistic probe
matches it exactly at the key observed A horizon.

| Condition | A h=1 | A h=2 | A h=4 | A h=8 | A h=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| real observed | 0.697 | 0.500 | 0.500 | 0.500 | 0.500 |
| IID continuous | 0.754 | 0.496 | 0.500 | 0.462 | 0.514 |
| IID skewed | 0.735 | 0.473 | 0.500 | 0.523 | 0.514 |
| IID quantized | 0.598 | 0.500 | 0.500 | 0.500 | 0.500 |
| empirical marginal | 0.698 | 0.500 | 0.500 | 0.500 | 0.500 |
| raw-order shuffle | 0.724 | 0.500 | 0.500 | 0.500 | 0.500 |

The observed A result is therefore reproduced by a current-level baseline and
by raw-order-shuffled/IID trajectories. Its apparent effect disappears after
the one-step overlap. The difference-only logistic probe is weaker at A h=1
(0.596 on the observed condition) and does not restore a later-horizon
effect. The nearest-neighbor probe is not a meaningful improvement: its A h=1
score is 0.713, while its paired correctness difference against the selected
median baseline is negative with a trajectory-bootstrap interval below zero.

For B, current-level predictability persists across horizons in both the real
and construction-null data:

| Condition | B h=1 | B h=2 | B h=4 | B h=8 | B h=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| real observed | 0.697 | 0.703 | 0.699 | 0.708 | 0.678 |
| IID continuous | 0.754 | 0.731 | 0.733 | 0.742 | 0.707 |
| IID skewed | 0.735 | 0.754 | 0.763 | 0.759 | 0.717 |
| IID quantized | 0.598 | 0.591 | 0.603 | 0.601 | 0.605 |
| empirical marginal | 0.698 | 0.700 | 0.695 | 0.686 | 0.707 |
| raw-order shuffle | 0.724 | 0.705 | 0.671 | 0.704 | 0.692 |

This is the expected current-value comparison effect, not evidence of a
long-range physical or computational signal. The quantized condition is
lower and class-imbalanced because exact ties are labelled positive under the
declared policy; that is a boundary-behavior result, not a new effect.

At real A h=1, the combined/current-level logistic score is `0.697` with a
representative 95% trajectory-bootstrap interval `[0.665, 0.736]`, Brier
score `0.151`, and ECE-10 `0.046`. Its paired balanced-accuracy difference
from the training-median current-level rule is exactly `0.000` with paired
correctness difference `0.000` and interval `[0.000, 0.000]`. The raw and
max-statistic circular-shift p-values are `0.0005`, the resolution floor for
2,000 randomizations, but the same low p-value occurs for the corresponding
current-level probe in construction-null data. Those p-values reject a
different alignment null; they do not reject the construction explanation.

## Diagnostics

The observed delta-sign transition count matrix uses rows for current sign
and columns for next sign, ordered `(-1, 0, +1)`:

```text
[[209, 300, 818],
 [438, 497, 358],
 [728, 411, 209]]
```

Across the 128 trajectories, the observed mean/median lag-1
autocorrelations were:

| Series | Mean | Median |
| --- | ---: | ---: |
| raw `x` | -0.018 | -0.035 |
| causal difference | -0.470 | -0.472 |
| difference sign | -0.416 | -0.415 |

The continuous-IID condition produced difference lag-1 mean/median
`-0.471/-0.473`; raw-order shuffle produced `-0.469/-0.476`. This is the
expected negative autocorrelation induced by differencing adjacent IID
values. It is not evidence of physical memory. The observed raw-delta tie
rate was 0.326 on average across trajectories; quantized IID was 0.665,
confirming that tie handling materially changes prevalence and baseline
scores.

The full JSON reports include per-trajectory uncertainty, class prevalence,
transition matrices, raw/delta/sign autocorrelations through lag 8, Brier
scores, calibration error, AUROC, bootstrap intervals, paired contrasts, all
declared transparent baselines, and legacy label/wrong-trajectory/
reversed-alignment/same-state controls. Those legacy controls test alignment
or feature-target relationship failure; they are not substitutes for the
construction-preserving nulls.

## Historical reanalysis and claim boundary

The earlier discovery and same-boot confirmation artifacts contain aggregate
JSON, manifests, splits, and results but no raw timing trajectories or
reconstructible shards. Their historical A scores remain `0.689` and `0.709`
at horizon 1, respectively, but neither can be retrospectively rerun through
the construction-null pipeline. The fresh v4 acquisition is the first run
with retained raw trajectories for this analysis.

The correct conclusion is that the replicated one-step observation is
explained by shared-value target construction plus ordinary timing-trace
statistics. No residual remains that justifies a reboot, cross-boot
replication, CLFLUSH/eviction sweep, cached-load campaign, target-address
campaign, or other hardware follow-up for this question. The result makes no
claim about hidden DRAM state, DRAM-origin information, model-computation
foresight, or precognition.

## Provenance

- Worker: `worker-03`; session `session-eb35f3a028574ad597c4d38ae59d837a`.
- Boot: `92cdb521-ac36-4649-a35c-bfb55c6ac870` (same boot as the historical
  confirmation; this result is a construction falsification, not a new
  cross-boot replication).
- Source commit: `be8ea99be2fcf4309d83aa7e445473c54c322572`.
- Configuration hash:
  `1ec1e0221bbe0aa7344d76d3739d6175b1f6d2a23988f0afb52c70e7f64a86b9`.
- Raw trajectory SHA-256:
  `abd356c7608dc7bc44fa22264a8d5763af0f1e59dd2376e02f5d888c3beffe24`.
- Campaign artifact directory:
  `evidence/self-forecasting-construction-worker03-20260912/`.
- All 12 result hashes and the raw artifact hash were rechecked after fetch.

