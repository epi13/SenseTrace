# worker-03 genuine multi-boot scoped-PMU characterization — 2026-09-03

## Decision

`B_observable_available_but_oracle_weak`

The predeclared directional PMU contrast reproduced 3/3 in **every one of
three genuinely distinct OS boots** with no multiplexing and full protocol
agreement, but cross-boot PMU null stability failed the frozen rule by a wide
margin. This is a characterization result only: no hidden-bit classifier was
trained and no larger commodity Phase 1A campaign was started.

## Statistical firewall

The protocol was frozen before any new measurement:

- multiboot protocol: `measurement-primitive-multiboot-v1`
- multiboot hash: `b745f520ceb73a4490ab400fc0e91791b247e991befbb9858f23fc89e1ec7ad4`
- per-boot characterization protocol: `measurement-primitive-characterization-v2`
- per-boot protocol hash: `367203b868076246ee52fd9b5323c87eb1cb5cbab9f9c5ed0c0c5575a8074b6c`
  (identical to the 2026-09-02 single-boot run, so the per-boot runs are
  directly comparable history)
- config: `configs/worker03-multiboot-scoped-perf.example.yaml`
- config hash: `c47a0064bc990e4489c11b93abbc289c268b3b249e5dd367641baeacf2f94af5`
- candidate event (single, predeclared): `cpu/cache-misses/`
- `cpu/cache-references/` known-available but diagnostic-only; not gated,
  not substituted.
- null rule (frozen): max relative deviation ≤ 0.25, max relative MAD ≤ 0.10,
  minimum 3 complete finite replicates; cross-boot stability pools all 9 null
  replicate medians under the same rule.
- decision tree frozen in `src/sensetrace/multiboot.py::multiboot_protocol`.

No threshold, event, replicate count, or boot count was changed after seeing
measurements.

## Identity and host

- Code commit (deployed, all three boots): `f4f335cb2f43c59b31e44891200611888357359e`
- Live operator config preserved (`fd04b5ed…` before and after deploy);
  no `perf_event_paranoid`, capability, or sysctl change was made.
- Host: `worker-03`, Fedora Linux 43, kernel `6.17.1-300.fc43.x86_64`,
  Intel Core i7-9700.
- `perf_event_paranoid=2`, no `CAP_PERFMON`/`CAP_SYS_ADMIN`; calling-thread
  probes opened successfully under that boundary.
- Orchestration: `sensetrace host characterize-multiboot` (new), which
  verifies the actual `/proc/sys/kernel/random/boot_id` before every boot,
  refuses reused boot IDs, fetches evidence per boot, and reboots between
  boots. The local manifest was written incrementally after every boot.

## Genuine boots and runs

| Boot index | Boot ID | Run ID | metrics.json SHA-256 |
| --- | --- | --- | --- |
| 0 | `23286929-5e51-423d-8de6-e945177f4c2a` | `primitive-characterization-20260903T023848Z-b9c79aee` | `20687b90937a6a604f234c55bf0a655a1b02f1d7ae964f74351b25609788c06a` |
| 1 | `f42323ab-e97e-4ff1-a726-1016986f6874` | `primitive-characterization-20260903T023958Z-65ad5ddd` | `9aeed46b2a359043f82d831b920fbd519f8a0e9e187ce6a5adfa65898c069c39` |
| 2 | `8118368e-e85c-4244-978b-3434c37dadce` | `primitive-characterization-20260903T024132Z-f4a8df62` | `48eee9961aa8acd663e815ac7ee4bd707abff49628a12f31af1fc95731ce94a4` |

All three runs report protocol hash
`367203b868076246ee52fd9b5323c87eb1cb5cbab9f9c5ed0c0c5575a8074b6c`
(protocol agreement: true) and configuration hash `c47a0064…`.
Two reboots were requested and verified (new boot ID + system service active
with exactly one runner after each); `verify-boot` passes on the final boot.
Immutable worker runs remain under
`/home/worker-03/.local/share/sensetrace/runs/multiboot-boot-{00,01,02}/`.

## Observations

Per-boot PMU agreement (requested-CLFLUSH above cached, raw medians):

| Boot | Cached medians | CLFLUSH medians | Agreement | Multiplex veto |
| --- | --- | --- | --- | --- |
| 0 (`23286929`) | 3.0, 0.0, 3.0 | 35.0, 33.5, 83.5 | 3/3 pass | false |
| 1 (`f42323ab`) | 4.0, 5.5, 9.0 | 33.0, 33.0, 37.0 | 3/3 pass | false |
| 2 (`8118368e`) | 8.0, 5.0, 5.0 | 35.0, 32.0, 38.5 | 3/3 pass | false |

Every per-operation read reported `time_enabled == time_running`
(no multiplexing). Latency contrast passed in every boot
(cached ≈ 44 TSC cycles, requested-CLFLUSH ≈ 216–224).

PMU null medians (per boot): `[24.5, 2.5, 2.0]`, `[21.0, 4.0, 4.0]`,
`[29.5, 8.0, 3.5]` — stability **fail** within every boot.
Cross-boot pool of all 9 medians
`[24.5, 2.5, 2.0, 21.0, 4.0, 4.0, 29.5, 8.0, 3.5]`:
center `4.0`, MAD `2.0`, relative MAD `0.5`, maximum relative deviation
`6.375` against limits 0.10 / 0.25 — **fail**.

## Characterized cause (post-hoc analysis, not a gate change)

The retained acquisition-order indices show a systematic first-use effect:
in all 9 replicates across all 3 boots, the control executed **first**
(`acquisition_order_index=0`) on each fresh allocation carries the highest
PMU median (9/9), regardless of which control it is:

- boot-00/01/02 null at order 0: 24.5 / 21.0 / 29.5 (vs 2.0–8.0 later);
- `artificial_delay_64` at order 0: 60.0 / 68.0 / 80.5;
- `artificial_delay_32` at order 0: 55.5 / 62.5 / 84.0.

Within-control per-operation reads confirm the transient: the first
operation on a fresh allocation read **2426** cache misses, decaying to
single digits over subsequent operations, while the TSC-latency sample
median stays 44.0 (median hides the spike; the PMU window sums it).
Mechanism: fresh-allocation first touch (page-table/TLB/cache/code-path
cold state) is counted inside the operation-scoped PMU window. Latency
medians are insensitive to it; summed event counts are not.

This explains the null instability as acquisition-order cold start, not boot
identity — but it was found post hoc and therefore motivates, but does not
itself constitute, a new experiment.

## Limitations

- Per-operation `cpu_id` telemetry was `unavailable` on all worker samples:
  the worker Python build lacks `os.sched_getcpu`. A libc fallback was added
  after these runs (not deployed before them), so CPU migration cannot be
  excluded as a secondary contributor from this dataset. The 9/9 order-0
  pattern cannot be explained by random migration.
- One CLFLUSH replicate read 83.5 at order 6 (boot-00); residual noise beyond
  the order-0 effect remains uncharacterized.

## Claim boundary and next action

Justified: across three genuine boots with frozen protocol and full
agreement, the qualified `cpu/cache-misses/` reader reproduced a directional
cached-vs-requested-CLFLUSH contrast 9/9 with no multiplexing, while its null
proved unstable within and across boots; the instability is characterized as
a first-use-of-allocation cold transient.

Not justified: DRAM access, physical address/row/bank/channel identity,
hidden-bit signal, cross-device transfer, or an A-level primitive.

Next step (single highest value, predeclared but **not executed** in this
push to preserve the firewall): one bounded repeat with an explicit,
pre-frozen warm-up/first-touch-discard control and the same frozen null rule
and decision tree. If that fails, stop the commodity line and continue Phase 2
controlled-memory-interface work, whose software boundary and mock backend
are implemented in this push.
