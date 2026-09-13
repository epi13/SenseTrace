# Controlled predictive-state campaign — worker-03

Date: 2026-09-12/13 UTC
Protocol: `controlled-predictive-state-v1`
Execution host: `worker-03`
Code deployed for acquisition: `670d6ae5846ebc176ca56b5359630ae237db37a9`
Boot: `92cdb521-ac36-4649-a35c-bfb55c6ac870`

## Executive result

The preregistered primary comparison did not show reproducible incremental
value from observation history. The primary cell was `read_pressure` with the
cached/preloaded measurement, targeting the future eight-repetition block mean
at horizon 8. Development selected `current_observation` as the strongest
eligible baseline and `delay_dmd_control` as the history candidate. The paired
held-out loss improvement was `-7.35 TSC-tick²` in development and
`-6.74 TSC-tick²` in untouched confirmation, with confirmation normalized
skill `-0.040` and session-bootstrap 95% interval `[-20.09, 2.38]`. Negative
means that the history candidate was worse than the selected baseline.

On confirmation, the primary-cell MSEs were:

| model | MSE (TSC-tick²) |
| --- | ---: |
| persistence | 6.73 |
| current observation | 7.22 |
| workload history + current observation | 7.22 |
| delay-embedded state | 13.96 |
| ARX observation/workload history | 18.64 |
| training mean / workload-only | 262.62 |

Thus the simplest adequate description for the frozen primary is current
timing persistence/current observation. Workload history alone did not explain
the result, and adding it to current observation changed no primary MSE at
the reported precision. There is no evidence here that observation-derived
state adds value beyond present information and the declared quiet context.

## Acquisition actually executed

Development and confirmation each completed 192 trajectories in 48 grouped
session IDs: three sessions per family/condition cell and four trajectories per
session. Each trajectory retained 113 raw observations: 16 baseline, 64
excitation-phase observations, one forecast-origin observation, and 32 quiet
future observations. The origin was index 80 in every trajectory, and every
target began strictly after that origin.

The four experiment families were `read_pressure`, `active_quiet`, `sham`, and
`passive`; the four instrument conditions were cached/preloaded, CLFLUSH,
eviction sweep, and timer-only. The `passive` family used the same acquisition
and forecast contract with no excitation workers. Development and confirmation
used different session IDs and confirmation schedule seeds (the confirmation
seed range is offset by 10,000,000); no schedule identifier was exposed as a
feature.

All 48 sessions in each stage were compliant. Observed active fractions were
1.0 for `read_pressure`, 0–0.2 for the `active_quiet` pulse family, and 0 for
sham/passive. The requested codebook and actual worker records are separate in
each trajectory. No eBPF witness was attached; synchronization evidence is the
native pressure-start flag, thread barrier, worker endpoint record, and worker
join before stop/origin.

The raw audit found 21,696 of 21,696 target observations with AUX-present
quality and zero endpoint AUX mismatches in each stage. This is endpoint
evidence only: equal AUX values do not prove that no migration happened between
endpoints. Requested and observed measurement affinity was `[0]`; the evidence
does not claim more than that process-mask observation.

The minimum measured TSC gap from stop confirmation to the origin was 735,345
ticks in development and 726,019 ticks in confirmation; the
minimum origin-to-next-observation endpoint gap was 117,873 ticks in
development and 105,294 ticks in confirmation. All were positive. Repetition
start gaps were retained in raw TSC units; the 5th/50th/95th percentile ranges
were `[194, 1,368, 2,044,006]` ticks in development and
`[211, 1,343, 2,168,974]` ticks in confirmation. The large upper tail reflects
the recorded excitation/quiet orchestration intervals, so forecasts are
reported in both repetition horizon and elapsed time.

## Instrument calibration

The separate worker calibration used 128 repetitions per real control and a
labeled artificial timing control. Median raw durations were:

| control | median TSC ticks |
| --- | ---: |
| timer-only | 30 |
| cached/preloaded load | 38 |
| eviction sweep then load | 64 |
| CLFLUSH/MFENCE then load | 210 |
| artificial 256-tick delay | 394 |

The artificial delay is calibration-only and was not used as a physical
reference corpus or a campaign feature. The worker conversion was
`2.9982837313 TSC ticks/ns` (`0.3335241390 ns/tick`) from a paired monotonic/TSC
calibration. TSC ticks remain the raw unit and are not relabeled as core
cycles.

The trajectory primitive retains both raw paired channel durations and the
acquisition order. It uses the next observed 64-bit word for the reference,
with observed 64-byte cache-line spacing; this establishes distinct observed
cache lines only. It does not establish separate DRAM banks, rows, channels,
or physical addresses. Sequential paired probes can perturb one another.

The worker binary reports legacy `sensetrace-native-kernel-v4` for historical
entry points and `sensetrace-native-trajectory-v1` for the new ABI. The deployed
binary SHA-256 is
`44c74a61d4e334f4ebffbece6f542f4486938786421f06ab0ac07877afdd94ad`.
The native build used `cc -O3 -std=c11 -Wall -Wextra -fPIC -shared`.

Assembly was audited on the deployed binary with:

```text
objdump -d -Mintel /opt/sensetrace/source/native/libsensetrace_measurement.so
```

The disassembly contains the v1 trajectory symbol and shows RDTSCP/LFENCE
endpoint boundaries, `clflush` followed by `mfence` on the CLFLUSH branch, and
the native condition/order loop. The pressure symbol shows its native RDTSCP
start, immediate `started = 1` store, bounded TSC-deadline read/write loop,
and affinity restore. The measured load is inside the retained endpoint
interval; the eviction sweep and CLFLUSH/MFENCE precede that interval.

## Forecast evaluation

The target contract was either the future timing level or a future block mean;
the preregistered primary was the block mean. Historical delta-sign targets
remain regression controls and were not used as the primary endpoint.

Every feature declares origin-inclusive availability. The analysis used raw
target/reference durations, optional paired difference, actual workload active
fraction through origin, quality masks, and origin position. It excluded future
observations, future witness/scheduler state, IDs, seeds, full excitation
schedules, confirmation values, and whole-trajectory normalization. Model
scaling, PCA/state fitting, regularization, and baseline/candidate selection
were development-only. The confirmation set was not used as an unlabeled
reference corpus.

The ladder was training mean, persistence, current observation plus position,
workload-only diagnostic, current observation plus workload history, regularized
ARX history, and one rank-bounded delay-embedded PCA/state model. For every
forecast origin, the streaming interface emitted all declared horizons before
the target was revealed. Offline and streaming predictions agreed for every
replayed comparison in the development and confirmation reports.

The passive primary cell (`passive`, cached/preloaded, block mean, horizon 8)
also failed to replicate a history gain: development selected persistence vs
delay state with `+822.17 TSC-tick²`, while confirmation was `-246.53` with
95% session-bootstrap interval `[-325.89, -188.53]`. This is reported
separately from controlled response forecasting.

Some secondary cells were positive in confirmation—for example,
`active_quiet`/eviction at horizon 1 (`+50.62`, interval `[1.63, 77.44]`) and
`read_pressure`/timer-only at horizon 8 (`+11.51`, `[4.26, 18.09]`). They were
not the frozen primary, coexist with negative conditions and passive controls,
and do not establish a general observation-history result. The active-quiet
eviction horizon-8 contrast, for example, was only `+29.07` with interval
`[-18.18, 107.28]`. The campaign therefore stops model expansion.

The primary confirmation candidate was not early enough to be actionable. For
all declared horizons, the confirmation replay measured median future elapsed
time of about 126–128 microseconds and 5th-percentile lead of about 79–80
microseconds from the origin endpoint to the first target endpoint. The
delay-state candidate's inference p95 was about 1.73 ms; ARX was about 1.14 ms;
persistence was about 0.29 ms. The lower-tail lead did not exceed inference
cost for any model/horizon. These are offline replay timings, not a live-shadow
claim; the measured gap makes a live online prediction at this sampling
interval non-actionable.

## Synthetic and construction controls

The independent synthetic validation artifact exercised IID quantized noise,
current-state Markov dynamics, a partially observed dynamical process,
workload-history-only response, and drift/position nuisance. It was run with
separate deterministic simulation seeds and no worker data. It is a calibration
of false-positive/power behavior, not hardware evidence. The historical
construction-null conclusion and its artifacts remain unchanged; the confirmed
raw artifact reference audit now resolves target directories with
`../../raw_trajectories.npz`, covered by a regression test.

## Artifact manifest

Fetched artifacts are retained locally under:

- `evidence/controlled-predictive-state-worker03/calibration/`
- `evidence/controlled-predictive-state-worker03/development/`
- `evidence/controlled-predictive-state-worker03/confirmation/`
- `evidence/controlled-predictive-state-worker03/analysis/`

Key SHA-256 values:

| artifact | SHA-256 |
| --- | --- |
| calibration raw | `4d7f2b9dd66bb73adc575ce0af2f8fce4a6029ae703f3a3a0292239bc016cec1` |
| development raw | `109070682070c7facdb510a1100335c3a3da5313d3e25064f13b5ed1a9767196` |
| development journal | `69165bf9421a9999554e2738aba25227a215a08031f5edd9c623aa19f0aee5af` |
| confirmation raw | `daa009a58d585d08cecfe2ef499978f5f884fbe238b133f265d3259e24a6fa1f` |
| confirmation journal | `7ae197d11c75af071d1b780caa758c4974ca21c304c48431209866acb9319d39` |
| analysis | `e2ff357ec5e27e556d22fc2a859119038c75d890720c3c36a380d85d5657b9cf` |

The raw files are integer-preserving compressed NPZ artifacts; the append-only
JSONL journals and metadata contain provenance, requested/actual execution,
quality, timing, session, boot, affinity, allocation, and claim-boundary
records.

## Conclusion and next step

On one worker boot, controlled excitation produced a reproducible ordinary
timing response, but the frozen primary did not show that observation history
improves prediction beyond current observation and permitted workload context.
Workload history alone did not explain the primary; current timing persistence
was adequate. Same-boot session replication supports this bounded conclusion;
cross-boot testing was not triggered because the useful primary effect did not
replicate. The next evidence-supported step is a targeted mechanism check on
the remaining secondary anomaly—same protocol, fresh sessions, one condition at
a time, with timer-only/paired-differential and slightly changed sampling
interval controls—before any model expansion or physical DRAM claim.
