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

## 2026-09-12 — worker-03 synthetic validation

Status: confirmed worker execution and synthetic-control behavior; not a real
SenseTrace self-forecasting finding.

The worker became available through the existing MNCS/Fabric rendezvous and
reported authenticated identity `worker-03`, session generation 63, hostname
`worker-03`, address `192.168.1.114`, Fedora kernel `6.17.1-300.fc43`, eight
logical CPUs, and approximately 32 GiB of memory. The SSH diagnostic initially
disagreed because the local alias still resolved to the worker's previous
address; after resolution recovered, `sensetrace host doctor` passed with an
authoritative system service and one active runner. No alternate transport was
introduced.

The exact committed source at `12814f9c4fc6c3f14a44725294de51849f3351a1` was
deployed, the native library rebuilt, and the full configured synthetic run
completed on worker-03 with 72 trajectories, six horizons (`1, 2, 4, 8, 16,
32`), three training seeds, and 400 bootstrap/permutation repetitions. The
linear-logistic predictable curve was `0.822, 0.755, 0.684, 0.564, 0.529,
0.497`; the null curve was `0.513, 0.479, 0.519, 0.493, 0.492, 0.487`. Every
causal audit passed. The fetched result hashes are recorded in the immutable
worker manifests: predictable results
`2b59aa70a8b32361ab172c87180ba5b278555d11f5a506c7641855f7cb84af4c`, null
results `75f4c892e78b708625bf52d77f25d8ba837d1eb3edfae0135ce5ca537908f009`.

The controller smoke run and worker run differ in sample count and therefore
are not byte-identical, but both recover the expected short-horizon AR(1)
decay and independent-null behavior. The worker manifest initially recorded
`sensetrace_commit=unavailable` because the horizon writer only tried local Git
discovery; deployment provenance separately confirmed the exact SHA. The
writer has now been corrected to read the deployed source marker, and future
runs will bind the manifest directly to that SHA.

## 2026-09-12 — real measurement-trajectory adapter

Status: confirmed framework behavior; local smoke acquisition only, pending
the worker real run.

The first naturally generated non-synthetic trajectory is one complete
`Sample` from the existing `CommodityDramBackend`. The trace is retained in
its native measurement-repetition order; no windows cross samples, sessions,
allocations, or boots. The adapter exposes the measured timing level and a
causal first difference. It deliberately excludes `Sample.label`, label
semantics, and target-adjacent metadata from state features and the
metadata-only baseline. The real configuration uses `random_word` so the
balanced label stream is not encoded in the observed word.

The local native smoke path acquired two target families (future timing level
and future timing-delta sign), completed the full adapter/control round trip,
and retained the expected source commit in its manifests. This is not worker
evidence and makes no physical DRAM, hidden-state, or model inference claim.
The worker run must be interpreted with the same boundary.

The evaluator now reports raw group-preserving permutation p-values and
max-statistic corrected p-values. The correction family is all predeclared
non-control model×horizon tests for one target and condition, with common
trajectory-level circular shifts preserving group structure; repeated training
seeds remain replications. Effect sizes and trajectory-bootstrap intervals are
primary, and controls are not treated as positive evidence.

## 2026-09-12 — first real worker measurement run

Status: real experimental finding, exploratory only; independent confirmation
required.

Worker-03 acquired 64 complete `CommodityDramBackend` samples in fresh session
`session-fa0248b94c5a407a91804c692ae82a48`, boot
`92cdb521-ac36-4649-a35c-bfb55c6ac870`, using the committed
`random_word`/eviction-buffer configuration. The two target reports share
source trajectory fingerprint
`693261b5a57765e17a96e897483fcd4b0a3449c5e2e5b4b780cbc993ef5486fa` and are
stored under `evidence/self-forecasting-trace-worker03-20260912-4b9b2de/`.
Execution host and requested node both report `worker-03`; both manifests bind
to source commit `4b9b2de592b88a098c16efcd21a1efa66bf431a6`.

For `future_timing_level`, the linear-ridge held-out skill over the constant
training mean was `-0.002, -0.002, +0.001, +0.002, -0.000` at horizons
`1, 2, 4, 8, 16`; no horizon cleared the 0.05 practical threshold or the
corrected significance rule. The nearest-neighbor probe was worse than the
constant baseline. Temporal-shuffle, wrong-trajectory, reversed-alignment,
and shuffled-label controls were similarly null; metadata-only skill was below
0.005 at every horizon.

For `future_timing_delta_sign`, linear-logistic balanced accuracy was
`0.689, 0.500, 0.500, 0.500, 0.500` at horizons `1, 2, 4, 8, 16`. At horizon
1 the effect over chance was `+0.189`, the 95% trajectory-bootstrap interval
was `[0.648, 0.728]`, and both raw and model×horizon max-statistic corrected
permutation p-values were `0.0025`. The same-state control was 1.0 by design;
temporal-shuffle, wrong-trajectory, and reversed-alignment controls were 0.5.
The signal disappears after one native measurement repetition and was observed
by both the linear and nearest-neighbor probes at that first horizon.

This is a real, short-range predictability result for a commodity timing
measurement trace. It is not evidence that SenseTrace predicts its own model
computation, hidden physical state, DRAM-origin information, or precognition.
The natural interpretation to test next is ordinary local timing
autocorrelation or an acquisition artifact. The confirmation protocol is
frozen in `docs/evidence/self-forecasting-confirmation-20260912.md`; no
operational speculative computation is justified.

## 2026-09-12 — independent real-trace confirmation

Status: replicated bounded measurement-trace finding; no higher-level
self-forecasting claim.

The frozen confirmation protocol completed on worker-03 with a fresh session
`session-197994c67b214e4289c2ce1db7a17449` and a distinct source trajectory
fingerprint
`e8a91611a686d38a9db60c0abd33eadf8f368421d577bc6e6e5e41413c94ea76`.
The boot ID was unchanged
(`92cdb521-ac36-4649-a35c-bfb55c6ac870`), so this is an independent
same-boot acquisition rather than a cross-boot replication. The run used the
same frozen source commit `4b9b2de592b88a098c16efcd21a1efa66bf431a6` and
reported worker-03 for both execution host and requested node.

The `future_timing_level` ridge probe remained null: skill was `-0.001,
-0.002, -0.001, -0.006, +0.002` at horizons `1, 2, 4, 8, 16`, with no
corrected significance. The `future_timing_delta_sign` logistic probe
reproduced the one-step effect: balanced accuracy was `0.709` at horizon 1
(95% trajectory-bootstrap interval `[0.674, 0.739]`; raw and max-statistic
corrected permutation p-values `0.0025`) and `0.500` at every later horizon.
The temporal-shuffle, wrong-trajectory, and reversed-alignment controls were
chance; the same-state control was 1.0 by construction. The first-run effect
(`0.689`) and confirmation effect (`0.709`) therefore reproduce only a
short-range autocorrelation-like property of the measured timing trace.

Confirmation artifacts are retained under
`evidence/self-forecasting-trace-worker03-confirmation-20260912/`; hashes and
the bounded interpretation are summarized in
`docs/evidence/self-forecasting-worker03-real-20260912.md`. This supports a
replicated measurement-trace observation, not a claim about SenseTrace
predicting model computation, hidden physical state, DRAM-origin information,
or precognition. Cross-boot replication and targeted autocorrelation controls
remain the appropriate next tests.

## 2026-09-12 — construction-preserving falsification protocol

Status: completed; construction explanation supported and hardware follow-up
closed for this question.

The historical result's actual target is A, `sign(x[t+h]-x[t+h-1])`, because
the real adapter exposes `[x[t], x[t]-x[t-1]]` and the target extracts the
future state's delta component. At horizon 1 this shares `x[t]` with the
feature vector. The distinct B target, `sign(x[t+h]-x[t])`, is implemented as
an explicit indexing diagnostic because its shared-current-value effect would
persist at every IID horizon.

The new campaign generates independent raw measurements (continuous,
skewed, quantized, and empirical-marginal), or shuffles raw observations
within each trajectory, then rebuilds production features and targets. It
reports the sign-reversal, transition, median-level, empirical-CDF,
current-level-only, current-difference-only, combined, and nearest-neighbor
baselines; trajectory-level uncertainty; paired contrasts; tie-aware sign
transitions; and per-trajectory lag diagnostics. See
`docs/SELF-FORECASTING-CONSTRUCTION-CAMPAIGN-20260912.md` for the frozen
protocol and interpretation criteria. Historical worker JSON lacks raw timing
trajectories, so it is retained as historical evidence rather than silently
reprocessed through the new null.

The clean v4 worker run completed all 12 target/condition reports with 128
complete `random_word` samples of 32 native repetitions, horizons 1/2/4/8/16,
three training seeds, 500 trajectory bootstraps, and 2,000
trajectory-preserving randomizations. The run used source commit
`be8ea99be2fcf4309d83aa7e445473c54c322572`, session
`session-eb35f3a028574ad597c4d38ae59d837a`, and the unchanged boot
`92cdb521-ac36-4649-a35c-bfb55c6ac870`. Artifacts are under
`evidence/self-forecasting-construction-worker03-20260912/`; raw trajectories
are retained with SHA-256
`abd356c7608dc7bc44fa22264a8d5763af0f1e59dd2376e02f5d888c3beffe24`.

For historical target A (`sign(x[t+h]-x[t+h-1])`), the real observed
current-level logistic score was `0.697` at h=1 and `0.500` at h=2/4/8/16.
The construction-preserving current-level baseline scored `0.754` on
continuous IID h=1 and `0.724` after raw-order shuffle; the corresponding
combined probe matched the current-level baseline at real h=1. The A score
returned to chance after the one-step shared-value overlap. The current-to-
future diagnostic target B remained around `0.68–0.76` across horizons in
both real and null trajectories, as expected from its explicit comparison to
the current value. The quantized condition showed the declared tie/prevalence
effect.

Observed lag-1 raw/delta/delta-sign autocorrelation means were
`-0.018/-0.470/-0.416`; continuous-IID values were
`-0.038/-0.471/-0.330`, and raw-order-shuffle values were
`-0.049/-0.469/-0.418`. The negative delta autocorrelation is the expected
mathematical consequence of differencing, not physical memory. The paired
combined-versus-median contrast at real A h=1 was exactly zero. The raw and
max-statistic circular-shift p-value floor of `0.0005` also appeared in the
construction nulls, so it does not distinguish construction from a physical
effect.

The final evidence summary is
`docs/evidence/self-forecasting-construction-null-20260912.md`. The
historical discovery (`0.689`) and same-boot confirmation (`0.709`) remain
valid aggregate records, but their raw trajectories are unavailable and were
not silently reconstructed. The one-step observation is explained by
shared-value target construction and ordinary timing-trace statistics. No
reboot, cross-boot, CLFLUSH/eviction, cached-load, or target-address hardware
follow-up is warranted for this question.
