# worker-03 scoped PMU characterization — 2026-09-02

## Decision

`B_observable_available_but_oracle_weak`

The existing latency observable and the operation-scoped PMU reader both
completed within the declared scope. The PMU cache-miss direction repeated in
all three matched replicates, but the PMU null was not stable under the frozen
robust rule. This is a characterization result only: no hidden-bit classifier
was trained and no larger commodity Phase 1A campaign was started.

## Identity and host

- Code commit: `7387b57128c5d31e98f08e0eaf5a50e98a98e2d4`
- Configuration hash: `b62e5251e1922ac5822f71b6b472592255f01ceebb724efcf4bb248ac7dedd0f`
- Protocol identity: `measurement-primitive-characterization-v2`
- Protocol hash: `367203b868076246ee52fd9b5323c87eb1cb5cbab9f9c5ed0c0c5575a8074b6c`
- Run ID: `primitive-characterization-20260902T150747Z-bf7336ba`
- Host: `worker-03`, Fedora Linux 43, kernel `6.17.1-300.fc43.x86_64`
- CPU: Intel(R) Core(TM) i7-9700, x86_64
- Genuine boot ID: `23286929-5e51-423d-8de6-e945177f4c2a`

The live operator config was preserved during deployment. No worker PMU
permission or sysctl change was made. The fresh inventory recorded
`/proc/sys/kernel/perf_event_paranoid=2`, effective capabilities
`0x0000000000000000`, and no `CAP_PERFMON` or `CAP_SYS_ADMIN` bit. The
calling-thread probes opened successfully under that boundary.

## PMU event and scope

The run selected the qualified sysfs event `cpu/cache-misses/`:

- PMU device: `cpu`
- event source type: `4` (`PERF_TYPE_RAW`)
- config: `0x412e` (`event=0x2e,umask=0x41`)
- config width represented: 64 bits
- preserved format fields: `event=config:0-7`, `umask=config:8-15`, plus the
  other exposed CPU format fields in the raw artifact
- read format: `3` (`total_time_enabled` and `total_time_running`)

`cpu/cache-references/` was also available (`type=4`, config `0x4f2e`) but was
not used. Uncore PMU devices were inventoried but not selected because an
uncore count cannot satisfy this primitive's calling-thread scope.

Each operation used the current native calling-thread ID, `cpu=-1`,
`inherit=0`, and no system-wide or unrelated-process attachment. The event was
created disabled with kernel and hypervisor counting excluded; the reader
reset/enabled immediately before the SenseTrace-owned controlled operation,
disabled immediately after it, read one scoped event, and deterministically
closed the FD. Every PMU reading reported `time_enabled == time_running`, so
the 64 observations per control/replicate were not multiplexed.

## Predeclared design and identities

The run used 3 replicates, 4 virtual locations, and 16 trials per location
(64 controlled operations per control and replicate). Controls were:

- `null_control`: repeated memory read with no requested eviction;
- `cached_control`: the low-latency side of the cache-path contrast;
- `requested_clflush_control`: CLFLUSH/MFENCE requested before the timed load;
- artificial delay controls at 0, 32, 64, and 128 TSC cycles, used only as
  calibration controls.

Controls shared one fresh allocation within each replicate and used the same
deterministic target stream. Replicate allocation IDs were:

| Replicate | Session ID | Allocation ID |
| --- | --- | --- |
| `replicate-0000` | `characterization-replicate-6d4dbbd7519f4c8db70ee03b5b1251fb` | `characterization-allocation-8ee126f8e09b44b185549033be34fefb` |
| `replicate-0001` | `characterization-replicate-70128df098e54279aa59ff9b87a0c0df` | `characterization-allocation-004bb2a8ba014216b24cc56f891f1b3a` |
| `replicate-0002` | `characterization-replicate-dda5bc5a8e0f41e59845a4bcee57d261` | `characterization-allocation-46bd2e6d964e4f4887e3504824106e27` |

The randomized acquisition-order indices (replicate rows 0–2) were:

| Control | Order indices |
| --- | --- |
| `null_control` | 2, 1, 3 |
| `cached_control` | 1, 4, 6 |
| `requested_clflush_control` | 4, 5, 5 |
| `artificial_delay_0_cycles` | 3, 6, 1 |
| `artificial_delay_32_cycles` | 0, 2, 0 |
| `artificial_delay_64_cycles` | 5, 3, 4 |
| `artificial_delay_128_cycles` | 6, 0, 2 |

## Observations

The latency null replicate medians were `44.0, 44.0, 44.0`. The predeclared
rule passed completeness, finite validity, and stability (center `44.0`, MAD
`0.0`, maximum relative deviation `0.0`). The matched latency contrast
(`requested_clflush - cached`) was:

| Replicate | Cached median | CLFLUSH median | Difference |
| --- | ---: | ---: | ---: |
| `replicate-0000` | 44.0 | 216.0 | 172.0 |
| `replicate-0001` | 44.0 | 217.0 | 173.0 |
| `replicate-0002` | 44.0 | 224.0 | 180.0 |

All three replicate IDs matched; no left or right replicate was missing.

The PMU raw-count medians, computed from all 64 retained per-operation reads
within each control/replicate, were:

| Replicate | Cached PMU median | CLFLUSH PMU median | Difference |
| --- | ---: | ---: | ---: |
| `replicate-0000` | 8.0 | 33.5 | 25.5 |
| `replicate-0001` | 3.5 | 34.0 | 30.5 |
| `replicate-0002` | 2.0 | 33.5 | 31.5 |

The directional PMU agreement was `3/3`, with no missing sides:

```json
{
  "expected_right_above_left": {
    "observed_right_above_left": 3,
    "observed_right_not_above_left": 0
  }
}
```

The PMU null medians were `4.5, 10.0, 3.0`. Completeness and finite-value
validity passed, but stability failed: center `4.5`, MAD `1.5`, relative MAD
`0.3333333333`, and maximum relative deviation `1.2222222222`, against the
predeclared limits 0.10 and 0.25. Therefore the PMU oracle gate reported
`oracle_available=true`, `oracle_independent=true`,
`oracle_agreement_pass=true`, but `oracle_stability_pass=false`.

## Raw evidence artifacts

The immutable worker run remains at:

`/home/worker-03/.local/share/sensetrace/runs/scoped-perf-characterization-20260902-v3/primitive-characterization-20260902T150747Z-bf7336ba/`

SHA-256 values captured on worker-03:

| Artifact | SHA-256 |
| --- | --- |
| `metrics.json` | `ab79e2a35747cbc4da262ad09bb60dcb7352a55ae90b5f76e5e31733c73f4499` |
| `protocol.json` | `888f83c7ab6396cf752eb4671e457d3201b1f4cf8ce08fe268325be8815cc85a` |
| `config.json` | `3bb1aa8be718f0b2408910ffb21af25c351777ae5d66871d01266f2b83126eb9` |
| `host.json` | `22c469d190a1f7feedad643870c076c1446be8adfabb841735d2ca3d76875e` |
| `events.jsonl` | `9948a671e17ed3f87af187beab473952c6d90f02a8587ddc193c80ab81bcb6fe` |

## Claim boundary and next action

Justified: on this worker, boot, configuration, and run, a qualified raw CPU
cache-miss event was opened and measured only around SenseTrace-owned
operations; its retained per-operation counts showed a repeatable directional
contrast between the cached and requested-CLFLUSH controls across three fresh
allocations. This supports a partial cache/access-path characterization.

Not justified: a cache miss means DRAM access; any physical address, row, bank,
channel, subarray, chip, DIMM, or device-independent identity; a hidden-bit
signal; uncore thread scope; or an A-level frozen physical measurement
primitive. Timing and PMU co-movement do not establish physical DRAM topology.

The single highest-value next experiment is one predeclared repeat of this same
small scoped characterization across at least three genuinely distinct OS boot
groups, retaining all per-operation counts. It should first test whether the
PMU-null instability survives fresh boots; it must not increase hidden-bit N.
