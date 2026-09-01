# Phase 0 v2 final gate — 2026-09-01

This is the one fresh final-gate run required by the frozen Phase 0 v2
protocol. It is a synthetic software-pipeline calibration and does not
establish a physical DRAM-state inference result.

- Run: `phase0-final-v2-20260901`
- Protocol: `phase0-protocol-v2`
- Protocol hash: `b774b0f314dbcf55b13211779f2c8d0e0a7cd38ef9b8ab3d08996d57652d6168`
- Decision rule: alpha `0.05`; empirical maximum statistic across the enabled
  models and metrics, with the frozen critical value `0.04347277777777777`.
- Independent replicates: 100 calibration and 100 fresh gate-validation
  replicates for each of null, shuffled-label, and 0.10-sigma injected
  conditions, across both declared balance modes.
- Calibration null FPR: `5/100 = 0.05` (Wilson 95% interval
  `[0.02154, 0.11175]`).
- Fresh null FPR: `2/100 = 0.02` (Wilson 95% interval
  `[0.00550, 0.07001]`).
- Fresh shuffled-label FPR: `6/100 = 0.06` (Wilson 95% interval
  `[0.02779, 0.12477]`).
- Injected detection: `100/100 = 1.00` (Wilson 95% interval
  `[0.96301, 1.00000]`), above the frozen minimum of `0.80`.
- Frozen outcome: `acceptance.phase1_gate = true`.

The gate therefore authorizes the explicitly requested safe Phase 1A worker
campaign. It does not authorize a claim about physical DRAM rows, banks,
subarrays, chips, DIMMs, or DRAM-resident state; those remain outside the
available measurement provenance.
