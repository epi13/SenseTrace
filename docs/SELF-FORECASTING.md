# Passive self-forecasting and predictive horizons

SenseTrace now has an analysis path for a narrow, testable question:

> Given the state visible at an origin `t`, how much information is available
> about a measurable property of the later state at `t+k`?

This is future-state prediction, not precognition. The initial implementation
is intentionally independent of the physical-memory claim. It uses ordered
synthetic trajectories to validate the causal alignment, split, controls, and
reporting machinery before an acquisition backend is allowed to supply a
computational-state stream.

## Causal contract

For every forecast row the framework constructs:

```text
features = trajectory.states[origin_index]
target   = target_spec(trajectory.states[target_index])
```

The future state is used only to derive the target. A forecast is not passed
back to trajectory generation, acquisition, or target computation. The
manifest records this as `passive_observation` and rejects an alignment where
`target_index <= origin_index`.

Horizon distance is explicit. `alignment=index` means a state-count offset;
`alignment=position` means the first observed state at or after the requested
position delta. The unit is retained as provenance, so layer, token, event,
and wall-clock horizons are not silently conflated.

## Targets and baselines

The current target adapter supports:

- binary future-state properties, such as the sign of a later latent
  component; and
- scalar continuous future-state properties, such as a later norm or latent
  component.

Every horizon reports the predeclared majority/constant baseline, seeded
random control, shuffled-label control, an explicit numeric metadata-only
control, and simple probes where compatible: logistic/ridge and nearest
neighbor. The metadata-only view rejects identity fields and future-index
fields. It is intentionally useful for exposing an ordering or bookkeeping
shortcut.

Probe selection is validation-only. All predeclared model/seed test scores are
retained as fixed comparisons, and test scores are never used to choose a
model. Confidence intervals bootstrap complete trajectory IDs rather than
individual overlapping windows.

## Useful lead time

Accuracy alone does not say whether a forecast can be operationally useful.
The report therefore includes `useful_lead_summary`:

- `maximum_practical_horizon`: largest tested distance above the configured
  practical effect threshold (`0.55` balanced accuracy by default, or `0.05`
  skill over a constant mean for continuous targets);
- `maximum_statistically_supported_horizon`: largest tested distance with a
  positive mean effect and at least half of repeated test seeds passing the
  per-run permutation threshold;
- area and normalized area under the positive effect curve.

These are finite-grid summaries, not optional stopping. The full curve,
sample counts, class balance, confidence intervals, AUROC, Brier score,
calibration error, permutation p-values, and repeated-seed results remain the
primary record.

## Synthetic falsification controls

`predictable` trajectories are stationary AR(1) state streams. `null`
trajectories draw each state independently with the same dimensions and
metadata shape. A valid implementation should recover a short-horizon signal
for `predictable`, show deterioration as distance grows, and remain near
baseline for `null` and `shuffled_labels`. These controls demonstrate the
analysis behavior only; they do not establish a physical or model-inference
result.

Run locally:

```bash
sensetrace run horizon \
  --config configs/self-forecasting-worker03.example.yaml \
  --output runs/self-forecasting-horizon-v1
```

The same bounded synthetic control can run on worker-03 through the existing
controller path:

```bash
sensetrace host run-horizon worker-03 \
  --config configs/self-forecasting-worker03.example.yaml \
  --output /home/worker-03/.local/share/sensetrace/runs/self-forecasting-horizon-v1
```

The remote output is disposable experiment data, but it is not automatically
treated as physical worker evidence. Retrieve the JSON artifacts to durable
controller storage and preserve the node identity, commit, config, split
fingerprints, and result hashes.

```bash
sensetrace results fetch-horizon --host worker-03 \
  --output /home/worker-03/.local/share/sensetrace/runs/self-forecasting-horizon-v1 \
  --destination evidence/self-forecasting-worker03-20260912
```

Each condition directory contains:

```text
manifest.json   # environment, config, fingerprints, and passive causal policy
splits.json     # immutable pair IDs and whole-trajectory partitions
results.json    # predictive-horizon curve and useful-lead summary
```

## Interpretation boundary and next work

The first worker-03 run is a software and synthetic validation of the
self-forecasting analysis path. It must not be combined with the historical
commodity PMU result or described as hidden-bit, DRAM-origin, or model
precognition evidence.

The next scientifically useful extension is a state adapter that emits
trajectories from an explicitly defined computation (for example, a model
layer/token trace or controlled-memory-interface trace), while retaining the
same whole-run holdout and passive causal contract. Only after a passive signal
survives natural-unit holdouts, null controls, and independent seeds should an
offline speculative-use simulation be added. Closed-loop interventions remain
out of scope for this phase.
