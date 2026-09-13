# Construction-preserving falsification campaign — 2026-09-12

Status: completed frozen campaign and evidence record. This document corrects
the interpretation of the replicated horizon-1 `future_timing_delta_sign`
observation without changing the historical result files. Final results are
summarized in `docs/evidence/self-forecasting-construction-null-20260912.md`.

## Target and indexing audit

The production real-trace adapter converts each complete acquisition sample
into one trajectory whose state at repetition `t` is:

```text
state[t] = [x[t], delta[t]]
delta[t] = x[t] - x[t-1]
delta[0] = 0
```

The historical binary target uses state component 1 at the future target
index. Therefore its actual semantics are:

```text
A: sign(x[t+h] - x[t+h-1])
```

with the historical `positive_if=ge` policy: an exactly zero difference is
labelled positive. At horizon 1, A is `sign(x[t+1]-x[t])`, which shares
`x[t]` with the feature vector. At horizons greater than 1, A uses raw
observations that do not overlap the current feature under IID sampling.

The tempting but different target is:

```text
B: sign(x[t+h] - x[t])
```

The campaign implements both targets explicitly in
`sensetrace.construction.build_timing_pairs`. B retains the shared-current
value relationship at every horizon under IID sampling. B is a diagnostic
indexing control and must not be substituted for the historical A result.

Each trajectory is one complete acquisition sample, and no pair crosses a
sample, session, allocation, or boot boundary. Features are copied from the
origin state only. Future values are read only to derive the declared target.
The standard scaler and all fitted baselines are fit on the training
partition. Grouped splits use complete `trajectory_id` values, so overlapping
within-trajectory rows never cross train, validation, and test.

## Controls

The production-path nulls first generate or reorder raw `x` values and then
rebuild `[x, causal difference]`, pairs, labels, and splits:

| Condition | Raw-value operation | Purpose |
| --- | --- | --- |
| `iid_continuous` | independent normal draws matched to observed mean/scale | analytical IID check |
| `iid_skewed` | independent standardized log-normal draws | skew robustness |
| `iid_quantized` | independent empirical draws rounded to a declared step | ties and quantization |
| `empirical_marginal` | independent resampling with replacement from the observed raw marginal | data-shaped diagnostic null |
| `raw_order_shuffle` | within-trajectory raw-order permutation, then full reconstruction | destroys temporal order while retaining each trajectory's marginal and shared-value construction |
| `real_observed` | untouched production trajectory | reference condition |

`empirical_marginal` is explicitly diagnostic: its generator may condition on
the complete observed marginal. It does not leak evaluation labels into model
fitting; every deployable predictor is still fit on the training rows only.
The raw-order shuffle preserves trajectory IDs and lengths, and uses the same
split seed, so observations are not moved between partitions.

The retained label-shuffle, wrong-trajectory, reversed-alignment, and circular
target controls answer different questions. They break the shared-value
feature-target relationship (or the time alignment), so chance performance on
them cannot establish that the production result exceeds the construction
null.

## Transparent baselines and diagnostics

The frozen model list contains majority, random, and shuffled-label controls;
the sign-reversal rule; a training-fitted delta-sign transition model; a
training-median current-level rule; a training-fitted empirical-CDF predictor;
current-level-only and current-difference-only logistic probes; the combined
logistic probe; and nearest neighbor. The empirical-CDF tie policy is recorded
with the run. The report includes paired held-out differences against the
strongest declared simple baseline chosen by validation, with trajectory-level
bootstrap intervals.

The campaign also records class prevalence, Brier score, calibration error,
AUROC where defined, trajectory bootstrap intervals, an explicit 3×3
`(-1, 0, +1)` delta-sign transition matrix, and per-trajectory plus
across-trajectory summaries of raw, difference, and difference-sign
autocorrelation. Negative lag-one autocorrelation in differences from IID raw
values is expected from differencing and is not interpreted as physical
memory.

For continuous IID observations the analytical checks are approximately:

```text
A at h=1, reverse current delta sign:       balanced accuracy 2/3
A at h>1, reverse current delta sign:       no shared-value advantage
A at h=1, training-median current level:    balanced accuracy 3/4
B at every h, training-median current level: balanced accuracy 3/4
```

Finite samples, skew, quantization, ties, and grouped holdout produce
deviations. The tests use tolerances rather than an exact Monte Carlo target.

## Frozen worker-03 protocol

The executable configuration is
`configs/self-forecasting-construction-worker03.example.yaml`:

- 128 complete `CommodityDramBackend` samples, 32 native repetitions each;
- `random_word`, so acquisition labels are independent of the written word;
- the existing `phase1a-commodity-baseline-v1` eviction/native path;
- horizons 1, 2, 4, 8, and 16 native measurement repetitions;
- `zero_is_positive`, retained to match the historical target exactly;
- three training seeds (11, 23, 37);
- 500 trajectory bootstrap repetitions and 2,000 trajectory-preserving
  randomizations; attainable randomization p-value resolution is 1/2001;
- one whole-trajectory grouped split with 70/15/15 proportions;
- all listed targets, baselines, probes, and legacy controls reported;
- no stopping, target selection, or model selection from held-out test scores.

The frozen primary construction contrast is the held-out combined probe versus
the strongest simple baseline chosen from validation. A result is considered
explained by construction when the real and raw-shuffled/IID curves are
consistent with the simple baseline family, the combined probe adds no
material paired held-out accuracy, and the A curve loses the effect after the
shared observation at horizon 1. The p-value against shuffled labels is not a
test of excess predictability beyond this construction null.

Run and fetch commands:

```bash
python -m sensetrace.cli host doctor worker-03
python -m sensetrace.cli host status worker-03
python -m sensetrace.cli host deploy worker-03
python -m sensetrace.cli host run-construction-falsification worker-03 \
  --config configs/self-forecasting-construction-worker03.example.yaml \
  --output /home/worker-03/.local/share/sensetrace/runs/self-forecasting-construction-20260912-v4
python -m sensetrace.cli results fetch-horizon --host worker-03 \
  --output /home/worker-03/.local/share/sensetrace/runs/self-forecasting-construction-20260912-v4 \
  --destination evidence/self-forecasting-construction-worker03-20260912
```

The campaign stores `raw_trajectories.npz` and its hash manifest alongside
the JSON reports. This is compact timing evidence, contains no labels or
user memory contents, and is ignored by the repository's evidence policy.
Each target manifest records the code revision, normalized configuration,
source fingerprint, split fingerprints, execution/requested node, boot and
session provenance, and raw-source hash.

## Historical evidence and claim boundary

The discovery and same-boot confirmation directories contain only acquisition
and analysis JSON, splits, manifests, and aggregate results. They do not
contain the raw timing trajectory or a reconstructible shard. They can verify
the reported `0.689` and `0.709` horizon-1 balanced-accuracy results and their
provenance, but they cannot support a retrospective construction-null rerun.
The campaign therefore does not relabel those results as construction-null
analyses. The fresh worker run is the first acquisition for which the raw
trajectory is retained for this purpose.

If the real effect is explained by these controls, no reboot, cross-boot
replication, CLFLUSH/eviction sweep, cached-load, or target-address campaign
is scientifically warranted for this question. If and only if a reproducible
residual remains after the construction controls and simple baselines, a
separate bounded physical-artifact protocol may be justified. Any such result
would still concern commodity timing trajectories only; it would not establish
hidden DRAM-state recovery, PMU claims, model-computation foresight, or
precognition.

The v4 result is explained by the construction controls. The current-level
baseline matches the real A horizon-1 score, the raw-order/IID controls
reproduce the same short-range pattern, and A returns to chance after the
one-step shared-value overlap. The distinct B target remains predictable at
all tested horizons in matched nulls because it compares each future value to
the current value by definition. No hardware follow-up is warranted for this
question. See
`docs/evidence/self-forecasting-construction-null-20260912.md` for the
complete result and provenance.
