# worker-03 warm-up-controlled multi-boot scoped-PMU characterization — 2026-09-03

## Decision

`C_primitive_unsuitable`

The frozen warm-up follow-up completed across three genuinely distinct OS
boots with complete native warm-up provenance and no PMU multiplexing. The
latency control remained directional, but the operation-scoped
`cpu/cache-misses/` contrast passed only 2/3, 3/3, and 2/3 replicates by boot.
The pooled PMU null also failed the unchanged stability rule. The commodity
measurement line therefore stops; no hidden-bit classifier or larger Phase 1A
campaign was started.

## Frozen protocol and identity

The protocol and decision tree were frozen before this run:

- multiboot protocol: `measurement-primitive-multiboot-v2`
- multiboot protocol hash: `80c116fab5bba3e5dc1e33ce12d5f20d2905c4ca9050cea56fa4a98ee6a56589`
- per-boot protocol: `measurement-primitive-characterization-v3`
- per-boot protocol hash: `31fff040115ce370627eda4c3fd0fb6d6a4700d06e92159ed0bbfa8e546b5ab3`
- config: `configs/worker03-multiboot-scoped-perf-warmup.example.yaml`
- raw config SHA-256: `a38a573d5e633797679deb0cc5904e168eee07259b09b21b58afb8605df6c257`
- normalized configuration hash retained in records: `7220ce77f8366a14ff4eead1746ea5b374c6da3d1c652c4d7b0591777a787d1c`
- candidate event: qualified `cpu/cache-misses/`
- diagnostic-only event: `cpu/cache-references/`
- null rule: maximum relative deviation `0.25`, maximum relative MAD `0.10`,
  minimum three complete finite replicates; all nine null medians were pooled
  for the cross-boot check
- warm-up: deterministic write/read-back of all 64 words plus 64 native cached
  dummy loads, outside every operation-scoped PMU window
- witness: disabled, as required by the frozen PMU gate

No threshold, event, replicate count, boot count, or warm-up behavior changed
after measurements were observed.

## Host and scope

- deployed code commit: `cd294626fe1f4cc9293fd3f99bf6bb0f6f16c05a`
- host: `worker-03`, Fedora Linux 43, kernel `6.17.1-300.fc43.x86_64`
- CPU: Intel Core i7-9700, x86_64
- PMU boundary: `/proc/sys/kernel/perf_event_paranoid=2`, no
  `CAP_PERFMON`/`CAP_SYS_ADMIN`, calling-thread probes opened successfully
- event source: PMU device `cpu`, source type `4` (`PERF_TYPE_RAW`), config
  `0x412e` (`event=0x2e,umask=0x41`), 64-bit config representation
- reader scope: SenseTrace-owned calling thread, `pid` set to that thread,
  `cpu=-1`, `inherit=0`, disabled creation, kernel/hypervisor excluded,
  reset/enable immediately before the callback, disable/read/close afterward,
  read format `3` with `time_enabled` and `time_running`
- worker state after completion: final boot verification passed, system
  service active/enabled, dedicated `sensetrace.target` active, exactly one
  runner; no worker PMU or sysctl permission change was made

## Genuine boots and retained runs

| Boot | Boot ID | Run ID | Metrics SHA-256 |
| --- | --- | --- | --- |
| 0 | `8118368e-e85c-4244-978b-3434c37dadce` | `primitive-characterization-20260903T174159Z-c400a78f` | `f76da4c21bb105024bfe65a5d6a94ab6d518059adab464bd3a763685c5db334b` |
| 1 | `a4805d43-8bee-4dda-ad03-d7d4a4c96a75` | `primitive-characterization-20260903T174337Z-05efcdd2` | `7eb7c9054c923c92bce58ea30e5dce634f84d348f8db424b578971fe04b0a665` |
| 2 | `77793d9c-8d5c-422a-805d-021141ad83dc` | `primitive-characterization-20260903T174515Z-e04b1281` | `450d79e934b1a20d186f82b7da158abbc425471c142baf1d99bf10f0e1e8357f` |

The controller verified a new boot ID after each of the two reboot requests.
All three reports passed frozen protocol validation, had three complete native
warm-ups, and retained all per-operation PMU readings. Raw worker runs remain
under `/home/worker-03/.local/share/sensetrace/runs/multiboot-boot-{00,01,02}/`.

## Observations

The declared latency control passed in all nine matched replicate contrasts:

| Boot | Cached medians | Requested-CLFLUSH medians | Differences |
| --- | --- | --- | --- |
| 0 | 44, 44, 44 | 224, 220, 218 | 180, 176, 174 |
| 1 | 44, 44, 44 | 220, 226, 218 | 176, 182, 174 |
| 2 | 44, 44, 44 | 222, 220, 218 | 178, 176, 174 |

The independent PMU contrast was not stable enough for the gate:

| Boot | Cached PMU medians | Requested-CLFLUSH PMU medians | Paired differences | Agreement |
| --- | --- | --- | --- | --- |
| 0 | 5.5, 74.5, 4.5 | 36, 49, 36 | 30.5, -25.5, 31.5 | 2/3 fail |
| 1 | 3, 3.5, 3 | 34, 41, 35.5 | 31, 37.5, 32.5 | 3/3 pass |
| 2 | 105, 3, 3 | 46, 38.5, 33.5 | -59, 35.5, 30.5 | 2/3 fail |

Every operation reported `time_enabled == time_running`; multiplex veto was
false for all three boots. The warm-up compliance gate passed for all nine
replicates with `path=native_cached_load` and `status=complete`. Acquisition
order remained diagnostic only: order zero was the highest PMU order in 2/3,
3/3, and 2/3 replicates by boot, so the warm-up did not justify changing the
predeclared analysis after the fact.

The nine retained null replicate medians were
`[5.0, 18.0, 13.5, 4.0, 11.0, 16.0, 10.0, 10.0, 6.0]`. The frozen pooled
stability calculation produced center `10.0`, MAD `4.0`, relative MAD `0.4`,
and maximum relative deviation `0.8`, failing the limits `0.10` and `0.25`.
Completeness and finite-value validity passed; this was a true stability
failure, not missing or non-finite evidence.

## Raw evidence artifacts

The controller manifest SHA-256 is
`dca1a576f342d2600584ce73fa0865a1a083d394a18363a34fa5a6d747736498` and the
combined report SHA-256 is
`bbcae87305328f22fb183e7dbc592c79a2a1e9e269fdba3b07a315d972260b68f`.
The fetched local copy is under
`evidence/multiboot-warmup-20260904/`; the worker copies are the immutable
source artifacts. The controller also retained per-boot hashes for
`run.json`, `host.json`, `config.json`, `protocol.json`, `metrics.json`, and
`events.jsonl` in the manifest.

## Claim boundary and next action

Justified: on this worker and frozen protocol, a thread-scoped raw CPU
cache-miss counter can be collected with complete scope/provenance and shows a
repeatable cache-path response in some controls, but the warm-up-controlled
three-boot characterization fails the predeclared directional and null
stability gates.

Not justified: DRAM access, physical address/row/bank/channel/rank/DIMM
identity, hidden-bit recovery, device-independent generalization, or an
`A_usable_auditable_primitive`.

The single highest-value next action is to stop commodity PMU/CLFLUSH scaling
and continue Phase 2 controlled-memory-interface work under ADR-013. Any new
measurement must introduce a new frozen protocol identity rather than retuning
this failed gate.
