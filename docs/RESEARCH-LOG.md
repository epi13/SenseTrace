# Self-forecasting research log

This log keeps the new computational-foresight line falsifiable. Results are
classified as confirmed findings, promising observations, negative results,
hypotheses, or speculation. A synthetic control result is never promoted to a
physical-memory or model-inference claim.

## 2026-09-12 — framework bring-up

Status: confirmed software behavior, pending worker-03 run.

- Added immutable present→future pair construction with explicit index- and
  position-defined horizons.
- Added whole-trajectory train/validation/test holdouts, causal alignment
  audits, metadata firewall, bootstrap uncertainty over trajectory IDs,
  permutation controls, and reproducibility manifests.
- Added binary and scalar continuous future-state targets.
- Added predictable AR(1) and independent null synthetic controls.
- Local deterministic smoke tests recover a short-horizon predictable signal,
  show deterioration at a longer horizon, and keep the null near baseline.

## Open questions

- Which naturally occurring computational states are predictable from earlier
  states, and at what horizon does the signal decay?
- Does a signal generalize across complete prompts, runs, seeds, models, or
  worker boots rather than only across rows from one generated corpus?
- Does future-state predictability survive controls for deterministic ordering,
  identifiers, normalization, overlapping windows, and shared samples?
- Which latent components or future events become predictable earliest?
- Can forecast uncertainty itself be predicted and calibrated?
- What is the maximum practical lead time after accounting for false
  speculation and preparation cost?

## Guardrails

The current phase is passive observation only. Forecasts do not alter the
trajectory that produces the target. Offline speculative-use simulation and
closed-loop intervention require a separate explicitly labeled phase after
credible passive evidence.

## 2026-09-12 — worker-03 availability check

Status: infrastructure blocker; no worker result was recorded.

The existing controller path was used as requested:

```text
sensetrace host doctor worker-03
```

The SSH alias resolved to `192.168.1.113`, but both ICMP and SSH were
unreachable from the controller (`No route to host` on TCP port 22). The
failure occurred before authentication, deployment, or experiment startup.
No local run is being relabeled as worker-03 evidence. Retry after the node or
network route is restored, then deploy the exact committed source and preserve
the remote manifest alongside the controller copy.

## 2026-09-12 — controller synthetic full curve

Status: confirmed synthetic-control behavior; not a SenseTrace physical or
model-inference finding.

The committed framework was exercised locally with the worker-03 protocol
shape, 48 generated trajectories, six horizons (`1, 2, 4, 8, 16, 32`), three
probe seeds, 120 trajectory-bootstrap repetitions, and 120 permutation
repetitions. The durable local artifacts are under
`runs/self-forecasting-horizon-local-20260912/`.

For the predictable AR(1) condition, the linear probe's held-out balanced
accuracy was `0.811, 0.721, 0.629, 0.541, 0.523, 0.527` across those horizons;
the configured practical threshold gave a maximum practical horizon of 4.
For the independent null condition the corresponding values were
`0.525, 0.458, 0.517, 0.484, 0.542, 0.422`, with no practical horizon. Every
causal alignment audit passed. The null curve's finite-sample deviations and
occasional control p-values are retained as a warning that multiple horizons,
models, and seeds require cautious interpretation; they are not evidence of a
future-state signal.

The manifests record commit `f14ddcc5200db6ed8953854537d0ecbf687dee9a`,
execution host `fedora`, requested node `worker-03`, source trajectory
fingerprints, split fingerprints, and result hashes. The requested worker run
remains pending because the node was unreachable.
