# worker-03 genuine multi-boot validation — 2026-09-01

This is a new, small validation run for the corrected Phase 1A holdout
boundary. It is not a re-write of the historical Phase 1A evidence and does
not establish physical DRAM-state inference.

## Acquisition and provenance

The predeclared configuration was
`configs/worker03-multiboot-validation.example.yaml`, gated by the retained
Phase 0 v2 report
`runs/phase0-final-v2-20260901/metrics.json`. Each run used one completed
commodity acquisition session, 64 rows, 4 virtual locations, 16 trials per
location, native CLFLUSH timing, and a fresh anonymous allocation. The three
campaigns were collected under code commit `8283b2c1de1df8b02f11a0799586b94981f9d969`.

| campaign run | UTC start | OS `boot_id` | acquisition session | allocation | status |
| --- | --- | --- | --- | --- | --- |
| `phase1a-20260902T030214Z-17056d9d` | 03:02:14 | `697e8ad6-ef00-4d2f-a9ae-c944867c9d46` | `session-59f6ef763ce44d8cb9b44d999ebe4013` | `buffer-71d296b239af48e89235317c05d27116` | completed |
| `phase1a-20260902T030421Z-0e3da06a` | 03:04:21 | `987161aa-43a2-47ca-ad44-fff858a41816` | `session-77a7403816d9489e95f2c996f03297f6` | `buffer-df15e48c93a94f3eb256b1dd41c254a7` | completed |
| `phase1a-20260902T030623Z-b82c7dab` | 03:06:23 | `23286929-5e51-423d-8de6-e945177f4c2a` | `session-29c80dc0bfe34ecca6edd5f7e4969c9f` | `buffer-0b6e7e75f230499ebfef7b9e897095b8` | completed |

The reboot transitions were `697e8ad6 → 987161aa` and `987161aa →
23286929`; both fresh-SSH acceptance checks passed. On each boot the
authoritative system service was active and enabled, the dedicated target was
active, the display manager was inactive, exactly one runner was present, and
the user-service fallback was disabled.

The combined paired dataset has 192 rows, dataset fingerprint
`38fd884783400692fc141b923ea7ae0c1bdf23787d5be2ac2a55dd23e52174da`, and was
analyzed under code commit `40532af8534f68b3a3f14f30af3d0fdc35a9ac84`. It has
three unique session IDs, three unique allocation IDs, and three unique OS
boot IDs. Source manifests remain embedded in the combined manifest.

## Corrected hierarchy result

All five levels were available and passed exact-coverage, group-disjointness,
sample-identity, and provenance invariants:

| split | independent groups | status | interpretation |
| --- | ---: | --- | --- |
| A repeated trial holdout | 96 | available | paired-repeat pipeline claim only |
| B unseen virtual location | 12 | available | candidate relationship beyond tested virtual locations |
| C unseen acquisition block | 3 | available | candidate relationship beyond blocks; topology unknown |
| D unseen acquisition session | 3 | available | candidate relationship across sessions on this host/device |
| E unseen OS boot | 3 | available | candidate relationship across genuine boot groups; not device-independent |

This small configuration has one acquisition block per boot, so C and E
materialize identical partitions. The report records that equivalence; they are
not counted as two independent confirmations.

For the corrected E test partition, logistic regression produced AUROC
`0.5029` and balanced accuracy `0.4531`; boosted trees produced AUROC
`0.4761` and balanced accuracy `0.4688`. The paired primary median-latency
delta was `-1.2656` TSC cycles with sign-flip p-value `0.6048` and null interval
`[-3.0781, 3.1406]`. The test partition contains one boot/session cluster, so
the small-sample uncertainty is substantial. This is a near-chance exploratory
result, not evidence against all possible physical channels and not evidence
for one.

## Integrity and recovery checks

An initial combine attempt correctly rejected unsorted cross-session shard
boundaries. The combiner was repaired to canonicalize the globally unique
`sample_id` order before writing merged shards; the retry passed all storage and
hierarchy invariants. No partially validated combined artifact was used.

The worker recovery self-test also passed: an interrupted run resumed at the
next finalized sample index (`13`), discarded and hashed 29 temporary bytes,
quarantined the incomplete temporary shard, and completed at index `32`. Killing
the authoritative runner caused systemd to restart it with one active process
(old PID `4277`, new PID `18240`, one restart).

## Reproduction commands

The controller operations used for this evidence were:

```bash
python3 -m sensetrace.cli host doctor worker-03
python3 -m sensetrace.cli host deploy worker-03
python3 -m sensetrace.cli host run-phase1a worker-03 \
  --config configs/worker03-multiboot-validation.example.yaml \
  --phase0-report runs/phase0-final-v2-20260901/metrics.json \
  --output /home/worker-03/.local/share/sensetrace/runs/phase1a-multiboot-validation-20260901
python3 -m sensetrace.cli host reboot-acceptance worker-03 \
  --cycles 1 --timeout 180 --require-appliance
python3 -m sensetrace.cli host verify-recovery worker-03
```

The run/reboot/run sequence was repeated until three distinct boot IDs were
collected. The corrected combine and analysis used
`sensetrace.datasets.combine_datasets` and
`sensetrace.phase1a._analyze_condition` on the finalized per-session source
directories only.

## Decision

Do not increase the Phase 1A commodity sample count yet. The native positive
control shows that the analysis path can detect an artificial timing change,
but CLFLUSH is not proof of a DRAM access and the physical multi-boot result is
near chance. The next physical experiment should change the measurement
primitive to a controlled memory-interface or hardware-counter experiment with
an explicit access-state oracle, while preserving the same three-boot,
allocation-boundary, paired-order, shuffled-label, and fail-closed recovery
rules. If that primitive cannot be made auditable, stop the physical Phase 1A
line rather than scaling its sample count.
