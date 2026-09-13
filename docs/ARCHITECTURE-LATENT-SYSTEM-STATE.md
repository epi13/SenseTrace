# Latent system-state witness architecture

Protocol: `latent-system-state-v1`

This is a new experiment namespace.  The historical `controlled-predictive-state-v1`
artifacts and conclusions are not rewritten.

## Observability audit

The existing controlled trajectory already records raw `uint64` TSC starts and
ends, endpoint `TSC_AUX`, quality bits, target/reference durations, paired
reference channels, randomized acquisition order, process affinity, requested
and actual pressure-worker execution, native pressure start flags, session and
trajectory IDs, boot IDs, fresh allocation identity, excitation schedule and
execution, origin index, stop confirmation, timing gaps through retained TSC
endpoints, cache-line spacing metadata, monotonic/TSC conversion, and a
claim-boundary/feature-firewall record.  Existing host inventory records CPU,
SMT/core shape, cache topology, affinity, cpufreq policy, boot/kernel, memory,
systemd, storage/network, thermal and PMU capability metadata where available.

The current trajectory does not provide a synchronized per-origin vector of
thread resource/scheduler counters, PMU deltas, `/proc` VM/IRQ/softirq state,
or a causal versus witness-only availability mask for those channels.  The
existing eBPF pilot is a separate contextual observer and its events are not
automatically predictors.  Existing perf support is operation-scoped and
records capability, event encoding, permission, and multiplex state, but was
not part of the old trajectory feature plane.

## Tier contract

* Tier 0 retains the native timing control with no system-witness snapshots.
* Tier 1 can add a deliberately selected local/native subset.  The deployed
  worker configuration uses only RDTSCP/AUX on every block and opens generic
  thread-scoped `cache-references`/`cache-misses` PMU windows at the origin
  bundle (`pmu_scope: origin_only`).  Each event records raw/scaled counts,
  time enabled/running, and multiplex status.  Thread resource, context-switch,
  `schedstat`, and frequency witnesses remain implemented but are not silently
  enabled in the low-overhead Tier 1 protocol.
* Tier 2 adds bounded `/proc/stat`, `/proc/vmstat`, and memory PSI witnesses and
  the full local witness set, and records the existing bpftrace tracepoint
  capability result.  It does not broaden privileges or treat unavailable eBPF
  hooks as zero events.  Tier 2 is a mechanism/witness tier until its measured
  observer effect is acceptable.

Each channel is described by `WitnessChannelSpec`: raw source, acquisition
interval, availability rule, causal eligibility, overhead description, units,
normalization and missing-value policy, and observer tier.  Every trajectory
stores capture intervals, availability timestamps, validity, and a causal mask
per row and channel.  Values after the origin remain reconstructable witnesses
but cannot enter a predictor.

## Causal origin and probes

One trajectory measures the selected timing condition through baseline and
controlled excitation, then acquires a sequential origin bundle containing the
configured cached/timer/flush/eviction projections.  The bundle order is
retained because sequential probes can perturb one another.  The forecast
origin is the end of that bundle; the target is a disjoint quiet future.  Raw
timing arrays, raw endpoints, raw AUX values, bundle order, witness arrays,
validity and availability masks are retained, so derived features can be
reconstructed.

## Analysis firewall and model question

The ladder is training mean, persistence, current primary timing, current
multichannel timing plus eligible witnesses, current timing plus declared
workload, primary history, multichannel history, regularized linear state, and
rank-bounded PCA/delay state.  Imputation/scaling/PCA/ridge fitting is derived
from development sessions only.  Session IDs, seeds, schedules, boot and
allocation identity, future values, future witness state, and whole-trajectory
normalization are forbidden.

The preregistered incremental questions are separate:

1. Does current multichannel state improve held-out future prediction over
   current primary timing?
2. Does multichannel history improve over current multichannel state?

Matched-state diagnostics search for cross-family pairs with similar present
primary timing but different causal multichannel state and report future
divergence with session-level bootstrap intervals.  They are mechanism/state
aliasing diagnostics, not a substitute for held-out prediction.

The development-only timescale sweep is explicitly `[50, 100, 1000]` µs in the
example protocol and uses two independent sessions per cell.  It cannot alter
the frozen 100 µs confirmation protocol.

## Worker-03 boundary

The live worker currently exposes generic thread-scoped cache-reference and
cache-miss PMU probes, but richer retired-load/LLC/TLB/stall vocabulary is not
available through its permitted interface.  The perf executable, tracefs and
bpftrace are unavailable in the reported worker inventory; those absences are
recorded explicitly.  No kernel/eBPF facility is silently substituted.

Commands:

```bash
PYTHONPATH=src .venv/bin/python -m sensetrace.cli validate-latent-system-state \
  --config configs/latent-system-state-worker03.example.yaml \
  --output runs/latent-system-state-synthetic-validation
PYTHONPATH=src .venv/bin/python -m sensetrace.cli characterize-latent-observer \
  --config configs/latent-system-state-worker03.example.yaml \
  --output runs/latent-system-state-observer-effect
PYTHONPATH=src .venv/bin/python -m sensetrace.cli run latent-timescale-sweep \
  --config configs/latent-system-state-worker03.example.yaml \
  --output runs/latent-system-state-timescale-sweep
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host deploy worker-03
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host run-latent-system-state worker-03 \
  --config configs/latent-system-state-worker03.example.yaml \
  --stage development \
  --output /home/worker-03/.local/share/sensetrace/runs/latent-system-state-development
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host run-latent-system-state worker-03 \
  --config configs/latent-system-state-worker03.example.yaml \
  --stage confirmation \
  --output /home/worker-03/.local/share/sensetrace/runs/latent-system-state-confirmation
PYTHONPATH=src .venv/bin/python -m sensetrace.cli host run-latent-timescale-sweep worker-03 \
  --config configs/latent-system-state-worker03.example.yaml \
  --output /home/worker-03/.local/share/sensetrace/runs/latent-system-state-timescale-sweep
```
