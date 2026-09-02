# worker-03 measurement-primitive characterization — 2026-09-02

This is a small, non-inference characterization of the commodity
`commodity-clflush-timed-load` primitive. It is not a Phase 1A hidden-bit
inference run and must not be combined with the Phase 1A physical dataset.

## Run and protocol

- Run: `primitive-characterization-20260902T043906Z-bb9cd8d6`
- Code commit: `13a57812440faff259683198afc4176543766d18`
- Configuration: `configs/worker03-primitive-characterization.example.yaml`
- Characterization protocol: `measurement-primitive-characterization-v1`
- Protocol hash: `e6f659fa4d610c265f04a773a6a511a99d690759cceb9cd1c72e2945ad41900d`
- Frozen commodity protocol: `phase1a-commodity-baseline-v1`
- Frozen commodity protocol hash: `24f0081004dc612fd77294528b0f4756464b416acfb3878ff06863dd654bf6c0`
- Design: 3 replicates, 4 virtual locations, 16 trials per location, trace
  length 32, weak-control delays `0, 32, 64, 128` requested TSC cycles
- Run status: completed

The characterization artifact was retained on worker-03 at:

`/home/worker-03/.local/share/sensetrace/runs/primitive-characterization-20260902/primitive-characterization-20260902T043906Z-bb9cd8d6/`

The remote `metrics.json` SHA-256 is
`5a528f5cae39e58cf0614b9983f267dd4ff122aaf7d7b31d08b7e5a7b089722a`.
The remote `protocol.json` SHA-256 is
`150961ecaa640b392c537b288547f445aff7b6cd4e3c455e4b264ede6fa6cf49`.

## Worker provenance and capability result

- Host: `worker-03`
- Host ID: `f9c883e925544e25a88cabd27688c724`
- Genuine OS boot ID: `23286929-5e51-423d-8de6-e945177f4c2a`
- Kernel: `6.17.1-300.fc43.x86_64`
- CPU: Intel Core i7-9700 @ 3.00 GHz, x86_64
- Acquisition sessions: 3 unique
- Anonymous allocations: 3 unique
- Genuine boot groups: 1

Safe capability discovery found PMU sysfs devices, including `cpu` and
`uncore_imc`. The `perf` executable was unavailable. Generic
`cache-references` and `cache-misses` event vocabulary was visible, but the
SenseTrace-owned scoped `perf_event_open` probes returned
`permission_denied`. No event was collected, interpreted as an access-state
oracle, or used as a model feature. Discovery did not broaden kernel
permissions or collect system-wide/unrelated-process counters.

The primitive therefore records:

- cache-residency control: known;
- physical address and row/bank/channel topology: unsupported;
- independent access-state oracle: unsupported/unavailable;
- translation state and replay across sessions/boots/devices: unknown;
- model-eligible inputs: trace-derived features only by default.

## Characterization observations

The strong control compared a cached control with the requested CLFLUSH path.
Across three paired replicate summaries, the requested-CLFLUSH-minus-cached
median latency difference had mean `+236.0` TSC cycles, median `+235.75`, and
standard deviation `10.13`; all three differences were positive. This
demonstrates that the timing path responds to the declared cache-path control.
It does not demonstrate that a load reached DRAM.

The no-artificial-delay null had median-of-replicate-medians `46.0` cycles and
replicate-median standard deviation `3.61` across three replicates. The
predeclared artificial timing-control curve produced label-median deltas of
`+1.5`, `+54.25`, `+49.5`, and `+54.25` cycles at requested delays `0`, `32`,
`64`, and `128`, respectively. These are instrumentation controls only.

Raw values and acquisition-order diagnostics were retained. Flushed-control
order/median correlations were `-0.9434`, `-0.9421`, and `-0.4510`; these are
audit diagnostics and were not filtered or converted into a physical claim.

## Decision gate

The recorded outcome is **B — observable available but oracle weak**:

- strong cache-path control observed: yes;
- meaningfully independent access-state oracle: no;
- hidden-bit model training: not performed;
- next action: improve access-state instrumentation before hidden-bit
  inference or a larger commodity sample count.

The result establishes only a commodity host timing/cache-path observation. It
does not establish physical DRAM access, physical address or row/bank/channel
identity, a device-independent signal, or information about a hidden bit.
The recommended path is to transition toward a controlled memory-interface or
appropriately scoped hardware-counter primitive if its access-state semantics
can be independently audited. If that cannot be achieved, stop the commodity
Phase 1 line rather than increasing `N` under the same uncertain observable.

## Reproduction commands

```bash
python3 -m sensetrace.cli host inventory worker-03
python3 -m sensetrace.cli host deploy worker-03
python3 -m sensetrace.cli host characterize-primitive worker-03 \
  --config configs/worker03-primitive-characterization.example.yaml \
  --output /home/worker-03/.local/share/sensetrace/runs/primitive-characterization-20260902
```

The complete raw characterization artifacts remain on worker-03; this tracked
summary preserves the run identity, code/protocol hashes, capability result,
observations, and decision boundary.
