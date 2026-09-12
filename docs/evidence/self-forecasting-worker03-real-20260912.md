# Worker-03 real measurement-trajectory evidence — 2026-09-12

This note indexes the fetched worker artifacts for the real SenseTrace
measurement-trajectory experiment. The raw JSON is retained locally under
`evidence/`; it is intentionally not committed because the trajectory and
split files are large. The committed implementation and frozen protocol are
the durable provenance record.

## Scope and provenance

- backend: `CommodityDramBackend` with the commodity eviction-buffer/native
  measurement path
- state: measured timing level plus causal first difference
- targets: future timing level and future timing-delta sign
- trajectories: one complete `Sample` per trajectory; no cross-sample windows
- source: `4b9b2de592b88a098c16efcd21a1efa66bf431a6`
- worker: `worker-03` for execution host and requested node
- claim boundary: measured commodity timing traces only; no DRAM, hidden-state,
  model-foresight, or precognition claim

## Discovery run

Directory: `evidence/self-forecasting-trace-worker03-20260912-4b9b2de/`

The discovery acquisition used session
`session-fa0248b94c5a407a91804c692ae82a48`, boot
`92cdb521-ac36-4649-a35c-bfb55c6ac870`, and source trajectory fingerprint
`693261b5a57765e17a96e897483fcd4b0a3449c5e2e5b4b780cbc993ef5486fa`.

The timing-level ridge skill was effectively null across horizons 1–16. The
timing-delta-sign logistic balanced accuracy was `0.689` at horizon 1 and
`0.500` at horizons 2, 4, 8, and 16. Horizon 1 had a 95% trajectory-bootstrap
interval of `[0.648, 0.728]`, with raw and max-statistic corrected p-values
`0.0025`. The temporal-shuffle, wrong-trajectory, and reversed-alignment
controls were chance.

Result hashes:

- `future_timing_level/results.json`:
  `3f6a2365cf63b3dbaf7ab887872f76914f19b7897dc49da051b84c12c1fa4a85`
- `future_timing_delta_sign/results.json`:
  `2883824aa2696993539bde23ca3ccf92c9140ee6b99a2dba885e0b73758e86c0`
- `future_timing_level/splits.json`:
  `ad8b50732150ecd79cb8bbbb9ebec427f07098fba357a0e8e312b9f1dd0a9c52`
- `future_timing_delta_sign/splits.json`:
  `a06ddd7047c5b6e7e6666932d41ebf072b3293f6758234917a576c77f5cf1a89`

## Independent confirmation

Directory: `evidence/self-forecasting-trace-worker03-confirmation-20260912/`

The confirmation acquisition used fresh session
`session-197994c67b214e4289c2ce1db7a17449` and distinct source trajectory
fingerprint
`e8a91611a686d38a9db60c0abd33eadf8f368421d577bc6e6e5e41413c94ea76`.
It used the same boot ID as discovery, so it is same-boot independent
replication, not cross-boot replication.

The timing-level ridge skill was `-0.001, -0.002, -0.001, -0.006, +0.002`
at horizons 1, 2, 4, 8, and 16. Timing-delta-sign logistic balanced accuracy
was `0.709` at horizon 1 and `0.500` thereafter; the horizon-1 interval was
`[0.674, 0.739]`, with raw and max-statistic corrected p-values `0.0025`.
Controls remained null except for same-state, which is a construction check
and scored 1.0 by design.

Artifact hashes:

- `acquisition.json`:
  `11c6bfbfb4e60bc00e2e92bb5a56be727104385b4578c178fe4ddc52b3b4f9b3`
- `experiment.json`:
  `e3bff2dd6b96c3108d4c090758b78be46faab6e8ebf739537a00a775933f2552`
- `future_timing_level/manifest.json`:
  `bf63ed78f3a65e0b156aee27f7224180a99cc0406cb544759ec6396e4a97a360`
- `future_timing_level/results.json`:
  `417e41d51c7f7a7a8f9c74aa50b701c1920c4f11b4d96bbe0ac11e8a91f6579c`
- `future_timing_level/splits.json`:
  `465c00ed89d20b5db2d04c3d2174dd86d3e4cd9e87c6148372db753fba5f4f59`
- `future_timing_delta_sign/manifest.json`:
  `85f0ac8bf1304db98b52879ec6cb0ff480a45e06f05e786be165d1b9344bfc57`
- `future_timing_delta_sign/results.json`:
  `814cda78c07b8105e07843a5db3c2e02b3bfc8eb1e1d768680064e90805e7188`
- `future_timing_delta_sign/splits.json`:
  `7983eaa9afc44287e9158ad727bfb254f883e2bb00ea43194db89670754fae63`

## Interpretation

The repeated one-step effect is bounded to a naturally generated commodity
timing measurement trace. Because the target is the next first difference and
the state includes the current difference, ordinary local temporal
autocorrelation or an acquisition artifact is the strongest interpretation.
The result does not justify operational speculative computation. Cross-boot
replication and dedicated autocorrelation/phase controls are required before
making a stronger statement.
