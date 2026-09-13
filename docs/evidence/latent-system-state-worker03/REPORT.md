# `latent-system-state-v1` worker-03 report

Date: 2026-09-13 UTC  
Worker: `worker-03`, Intel Core i7-9700, 8 physical cores, SMT disabled  
Boot: `92cdb521-ac36-4649-a35c-bfb55c6ac870`  
Protocol/configuration hash: `0ed6802e9c276ef51502edf98b574e98cb808a4c6a820895c40ea048b35882b9`  
Acquisition code: `46c8378e7b17e7d5b0dc6cdd5f8bb537d28d27a8`

## Decision

The primary latent-state hypothesis did not replicate. On untouched
confirmation data, `read_pressure / cached_preloaded / future_block_mean /
horizon 8`, current multichannel state was worse than current primary timing:

| comparison | loss improvement, ticks² | session bootstrap 95% interval |
| --- | ---: | ---: |
| current primary → current multichannel | -183.687 | [-588.543, 42.158] |
| current multichannel → multichannel history | +254.662 | [37.038, 640.398] |
| current primary → multichannel history | +70.975 | [40.946, 99.713] |

The multichannel history result is a separate temporal-history signal, not
evidence that the added system witnesses reconstructed a latent machine state.
The frozen primary comparison itself failed the practical confirmation gate.

## Phase 1 audit and implementation

The historical controlled trajectory already retained raw TSC start/end,
`TSC_AUX`, quality flags, paired target/reference timing, acquisition order,
affinity, requested and actual excitation execution, session/trajectory IDs,
boot IDs, allocation identity, conditions, excitation timing, timing gaps,
native provenance, and host inventory. The new campaign therefore did not
duplicate those fields. Its gap-filling layer adds explicit per-channel source,
units, capture interval, availability timestamp, causal eligibility, validity,
observer tier, PMU provenance, and raw witness preservation.

Tier 0 remains a no-witness native timing control. The frozen acquisition used
Tier 1 with:

- native RDTSCP/AUX CPU identity at the origin;
- calling-thread `cache-references` and `cache-misses` PMU events, opened only
  around the origin bundle, with raw/scaled counts, enabled/running time, and
  multiplex/validity state;
- a sequential origin bundle containing cached, timer-only, CLFLUSH, and
  eviction timing projections;
- explicit origin-time causal masks. Non-origin Tier 1 rows are retained as
  raw timing but do not receive predictor-eligible witness reads.

Tier 2 code implements bounded `/proc/stat`, `/proc/vmstat`, memory PSI,
thread resource/status/schedstat, and frequency witnesses as mechanism
witnesses. It was not used as the frozen predictor tier because its measured
observer effect is large. The worker inventory reports no `perf` executable,
tracefs, or bpftrace backend. Richer requested PMU vocabulary—retired-load,
memory-stall, DTLB, and uncore-memory candidates—was unavailable through the
permitted worker interface. These channels were recorded as unavailable, not
substituted or zero-filled.

Every predictor is built through the explicit channel contract. Future rows,
future witnesses, identity metadata, confirmation-derived scaling, target
values, and whole-trajectory normalization are rejected or excluded. Raw
integer channels and the JSONL journal round-trip are hash-checked.

## Observer effect

Thresholds were frozen before confirmation: maximum 2× wall latency, 2× timing
variance, 20 TSC ticks median shift, and 25% thread CPU utilization.

| tier | added median wall time | wall ratio | timing variance ratio | median timing shift | thread CPU utilization | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tier 1 | 127,636 ns | 5.41× | 0.61× | 0 ticks | 0.98 | failed |
| Tier 2 | 705,726 ns | 25.38× | 0.93× | 0 ticks | 0.99 | failed |

This benchmark includes the configured origin PMU window; it is therefore an
empirical warning about the PMU witness path, not a claim that PMU overhead is
zero. The non-origin Tier 1 path without PMU/snapshot capture was near the
Tier 0 native control in the local smoke test (~31.8 vs ~31.0 µs median), but
the origin witness path remains materially more expensive.

## Synthetic validation

The suite covered IID noise, persistence-only behavior, hidden-state aliasing,
workload-only effects, drift, instrumentation-correlated noise, and a
post-origin leakage trap. It detected the injected aliasing witness gain,
did not produce a large IID witness gain, separated workload-only signal from
system-witness value, and kept future witness rows out of features. The
leakage trap reported `future_witness_used: false`.

## Development and confirmation

Authoritative development-r2 used 48 independent sessions, one trajectory per
session, with six sessions per cell. At horizon 8, its primary current-primary
to-current-multichannel improvement was -3446.098 ticks² (session interval
[-5602.278, -1289.917]). The development timer-only anomaly did not remain
stable enough to retune the protocol.

Confirmation used 160 fresh independent sessions, one trajectory per session,
20 per family/condition cell, a fresh confirmation seed namespace, and the
same boot/configuration. Confirmation raw artifacts report three valid origin
witness channels for every trajectory: 48/48 in development-r2 and 160/160 in
confirmation. No post-origin causal witness was accepted.

At confirmation horizon 8:

| cell | current primary → multichannel | multichannel → history | interpretation |
| --- | ---: | ---: | --- |
| read_pressure / cached_preloaded | -183.687 [-588.543, 42.158] | +254.662 [37.038, 640.398] | primary fails; timing history helps over the weak multichannel model |
| read_pressure / timer_only | +12.485 [-72.905, 117.682] | +5.267 [-78.830, 82.972] | prior anomaly not confirmed |
| active_quiet / cached_preloaded | -277.939 [-404.977, -162.135] | +324.200 [64.841, 757.967] | added channels harm; history partially recovers |
| active_quiet / timer_only | +42.944 [-128.888, 254.885] | -480.945 [-1244.660, 31.060] | uncertain secondary cell |

For the primary cell, test MSE was 114.275 ticks² for current primary,
297.962 for current multichannel, and 43.300 for multichannel history. Skills
relative to the training-mean baseline were +0.004, -1.596, and +0.623,
respectively. These are held-out confirmation statistics with development-only
preprocessing.

## Matched-state divergence

The diagnostic found many cross-family pairs with nearly equal current primary
timing, but this is not itself a predictive result. For cached-load pairs at
horizon 8, there were 316 pair rows across 58 sessions, mean residual primary
difference 3.943 ticks, mean broader-state raw absolute difference 3342.011
(mixed timing/count units), and future divergence 11.775 ticks with session
bootstrap interval [9.787, 13.741]. State/future absolute correlation was
-0.138. Timer-only matching gave 582 pair rows across 66 sessions, residual
4.244 ticks, broader-state raw difference 2509.833, future divergence 11.812
ticks with interval [10.387, 12.893], and correlation +0.075.

Thus the present scalar timing value can be matched while the retained PMU/
origin-bundle witness vector differs, but the divergence diagnostic did not
show a useful causal state partition. Pair rows are not independent; session
clusters remain the uncertainty unit.

## Timer-only mechanism result

The focused timer-only control was included beside all memory timing channels.
The earlier `read_pressure / timer_only / horizon 8` development anomaly was
not confirmed: the confirmation gain from current multichannel over scalar
timing was only +12.485 ticks² with an interval spanning zero, and the
history-over-multichannel gain was also compatible with zero. With scheduler,
IRQ/softirq, VM, and frequency witnesses not acceptable for frozen Tier 1
acquisition, this campaign cannot identify a specific timer-only mechanism.

## Timescale exploration

The development-only sweep acquired 50, 100, and 1000 µs intervals with two
sessions per cell. The sweep is descriptive only because its cell cluster count
is insufficient for confirmation. Its current-to-future primary MSEs were
smallest and most stable in some 50 µs cells, but the pattern was not a frozen
causal effect and did not trigger protocol retuning.

## Evidence and next step

Raw trajectories, journals, layouts, manifests, hashes, worker provenance,
observer characterization, synthetic controls, timescale outputs, and both
development/confirmation analyses are under this directory. The development
freeze is recorded in `DEVELOPMENT-FREEZE.md`; the architecture is documented
in `docs/ARCHITECTURE-LATENT-SYSTEM-STATE.md`.

Cross-boot testing was not triggered: the primary same-boot confirmation did
not replicate. The evidence-supported next step is a narrowly preregistered
replication of the confirmed temporal-history signal using the Tier 0/native
primary channel, with session-level splits and no PMU/system-witness retuning.
Do not expand the witness stack or chase the timer-only secondary until that
history result is separated from session drift and reproduced on untouched
sessions.
