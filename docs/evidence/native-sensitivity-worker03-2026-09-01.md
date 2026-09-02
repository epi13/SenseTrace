# worker-03 native timing sensitivity calibration — 2026-09-01

This is a separately namespaced instrumentation positive-control calibration.
It measures the sensitivity of the native acquisition and analysis path to an
artificial TSC-deadline delay after the volatile load. It is not a physical
DRAM-state experiment and must not be combined with the Phase 1A physical
dataset.

## Protocol and implementation

- Run: `native-sensitivity-20260902T031046Z-b67e61c1`
- Code commit: `40532af8534f68b3a3f14f30af3d0fdc35a9ac84`
- Protocol: `native-sensitivity-protocol-v1`
- Protocol hash: `472616b52d988a1a9b01a7b350d3627ee1ae5bc5e31f629a19865065c76c897f`
- Development grid: `0, 32, 64, 128, 256, 512` requested TSC cycles
- Development replicates: 6 per magnitude; frozen validation replicates: 6
- Holdout: D, unseen acquisition session
- Enabled models: logistic regression and boosted trees
- Selection: smallest positive development magnitude reaching empirical power 0.80
- Fresh validation: new seeds, fixed development critical statistic, no retuning
- Controls: zero null and shuffled-label controls on the same observations

The development null maximum-statistic critical value was `0.0968323`. The
development rule selected `32` cycles. Fresh validation used magnitudes `0`,
`32`, and `512` only.

## Detection curve

| requested delay (TSC cycles) | development power | fresh frozen power | fresh logistic AUROC | fresh tree AUROC |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0/6 | 0/6 | 0.4988 | 0.4935 |
| 32 | 6/6 | 6/6 | 0.5590 | 0.9274 |
| 64 | 6/6 | not selected | — | — |
| 128 | 6/6 | not selected | — | — |
| 256 | 6/6 | not selected | — | — |
| 512 | 6/6 | 6/6 | 0.9974 | 0.9987 |

Development AUROC means for logistic/tree at 32 cycles were `0.5854/0.9169`;
at 512 cycles they were `0.9967/0.9974`. The selected fresh-control paired
median-latency delta was `+46.5391` cycles, median `+33`, with sign-flip
p-value `0.001996`. This is the expected artificial timing response, not a
physical-memory effect.

## False-positive and dependence controls

- Development zero-null FPR: `1/6 = 0.1667`, Wilson 95% interval
  `[0.0301, 0.5635]`.
- Fresh frozen zero-null FPR: `0/6 = 0.0000`.
- Development shuffled-label FPR: `2/18 = 0.1111`.
- Fresh shuffled-label FPR: `2/18 = 0.1111`.
- The fresh selected positive record had one held-out acquisition session,
  with label means `276.6953` and `323.2344` cycles; session dependence is
  reported, not removed.
- Boot dependence is explicitly unavailable for this calibration because each
  calibration replicate runs within one OS boot (`boot_count=1`).
- The six-replicate Wilson intervals are broad; this is a path-sensitivity
  calibration, not a final effect-size or false-positive certification.

## Native timing audit

The worker used `sensetrace-native-kernel-v2`, library SHA-256
`34d2ee12c4cd29991965534f9a9da7f245bdf3656235d11f1e0226a770345de4`, with
explicit compiler barriers around LFENCE/RDTSC and RDTSCP/LFENCE. In a separate
200-repetition native audit, the timer-only median was `22` cycles, cached-load
median `22`, and flushed-load median `179` (difference `157`). Flushed-load
raw values were retained; their mean was `193.99`, p99 `1216.03`, and the
lag-1 autocorrelation was `-0.0124`. The apparent outlier fraction is an audit
summary only and was not used to filter observations. The worker's Python build
does not expose `sched_getcpu`; per-sample CPU-boundary provenance remains
recorded and migration/interrupt-contaminated traces are not silently dropped.

The native audit claim is limited to “CLFLUSH followed by a timed load.” It does
not establish that any load reached DRAM, nor does it expose physical row, bank,
subarray, chip, or DIMM identity.

## Reproduction commands

```bash
python3 -m sensetrace.cli host deploy worker-03
python3 -m sensetrace.cli host calibrate-native-sensitivity worker-03 \
  --config configs/worker03-native-sensitivity-validation.example.yaml \
  --output /home/worker-03/.local/share/sensetrace/runs/native-sensitivity-worker03-validation
python3 -m sensetrace.cli calibrate native --output runs/native-calibration --repetitions 200
```

The complete raw calibration report and per-dataset ledgers remain under the
ignored local `runs/worker03-native-sensitivity-validation-20260901/` evidence
directory and the corresponding worker run directory; this tracked summary
preserves the run ID, hashes, protocol, and measured outcomes.
