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
