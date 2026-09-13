# Controlled predictive-state campaign

This is the executable protocol and analysis contract for the focused
worker-03 experiment. It asks whether observations through a quiet forecast
origin predict a disjoint future timing level or future-block mean beyond the
current observation, apparatus position, actual workload history, and the
declared quiet policy.

## Commands

Build and test the native artifact locally:

```bash
make -C native clean all test
PYTHONPATH=src .venv/bin/python -m sensetrace.cli validate-controlled-forecast \
  --config configs/controlled-predictive-state-worker03.example.yaml \
  --output runs/controlled-predictive-state-synthetic-validation
```

On the resolved SSH alias `worker-03`, deploy the branch, calibrate the
instrument, then run development and confirmation into separate directories:

```bash
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host deploy worker-03
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host calibrate-controlled-forecast worker-03 \
  --config configs/controlled-predictive-state-worker03.example.yaml \
  --output /home/worker-03/.local/share/sensetrace/runs/controlled-predictive-state-calibration
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host run-controlled-forecast worker-03 \
  --config configs/controlled-predictive-state-worker03.example.yaml \
  --stage development \
  --output /home/worker-03/.local/share/sensetrace/runs/controlled-predictive-state-development
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host run-controlled-forecast worker-03 \
  --config configs/controlled-predictive-state-worker03.example.yaml \
  --stage confirmation \
  --output /home/worker-03/.local/share/sensetrace/runs/controlled-predictive-state-confirmation
```

Fetch the three artifact namespaces with the explicit remote output path:

```bash
PYTHONPATH=src .venv/bin/python -m sensetrace.cli results fetch-controlled-forecast \
  --host worker-03 --output /home/worker-03/.local/share/sensetrace/runs/controlled-predictive-state-development \
  --destination evidence/controlled-predictive-state-worker03/development
```

Repeat the fetch for calibration and confirmation, then analyze locally:

```bash
PYTHONPATH=src .venv/bin/python -m sensetrace.cli analyze controlled-forecast \
  --config configs/controlled-predictive-state-worker03.example.yaml \
  --development evidence/controlled-predictive-state-worker03/development \
  --confirmation evidence/controlled-predictive-state-worker03/confirmation \
  --output evidence/controlled-predictive-state-worker03/analysis
```

All acquisition stages are resumable after a completed trajectory. The journal
is append-only, the compressed raw artifact is rewritten atomically from the
completed journal, and the manifest hash is checked before analysis.

## Causality and quiet policy

Each trajectory has one forecast origin. The origin is the first quiet
measurement after every excitation worker has joined and a native stop clock
has been recorded. The target begins at `origin + horizon`; block targets use
only later observations. The quiet policy is known context: no excitation
workers are launched after stop confirmation. This is controlled-response
forecasting, not a claim that future workload is unknown.

The streaming interface is:

```python
interface.update(observation, available_inputs)
interface.forecast(horizons, declared_future_policy="quiet")
```

The replay path updates through the origin, emits every horizon before reading
the target, and requires numerical agreement with offline inference. State
warmup is causal and excluded from scoring. Scaling, PCA, ridge fits, model
selection, and any target threshold are development-only. Confirmation is
never an unlabeled preprocessing corpus.

## Measurement and provenance

The v5 native ABI keeps raw integer start/end TSC values, endpoint TSC_AUX,
quality bits, condition codes, and first/second acquisition order. A paired
reference uses the next observed cache-line-sized word. `reference_ticks -
target_ticks` is optional derived data; raw channels remain primary. An AUX
mismatch marks endpoint identity disagreement, while an equal pair is only
endpoint evidence and cannot prove no migration between endpoints.

The native trajectory duration includes the v5 RDTSCP/LFENCE boundary and the
timed volatile load (or timer-only boundary). CLFLUSH/MFENCE and eviction
sweep happen before the timed load. TSC ticks are not silently relabeled as
core cycles. A separate TSC-to-monotonic calibration supplies elapsed-time
lead estimates.

The calibration namespace contains real cached/flushed/eviction/timer
controls and an explicitly labeled artificial delay control. It is not used as
a model reference corpus. Worker inventory, boot, allocation, configuration,
code revision, binary hash, affinity, and witness status are audit metadata,
not model features.

## Evidence boundary

The primary statistic is the paired held-out squared-error improvement:

`loss(strongest development-selected eligible baseline) - loss(candidate)`.

Intervals resample complete session IDs, not rows or random seeds. Controlled
families and passive forecasting are reported separately. Workload-only and
current-plus-workload baselines determine whether an apparent observation
history effect is actually a controlled workload-response effect.

This protocol can establish ordinary machine-state predictive value if it
replicates. It cannot establish hidden DRAM information, DRAM origin, physical
bank/row/channel identity, or a unique mechanism. A fitted low-dimensional
state is a predictive representation only.
