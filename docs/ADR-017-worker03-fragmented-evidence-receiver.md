# ADR-017: worker-03 fragmented-evidence receiver

## Status

Accepted as an implemented software milestone. No physical controlled-memory
session has been acquired by this ADR.

## Decision

SenseTrace freezes the immediate experimental target as
`worker03-hardware-v1`. The target inventory is non-destructive and records
observed CPU, cache, affinity, TSC flags, microcode, NUMA, frequency policy,
BIOS/DMI, DIMM/SPD, thermal, hugepage, kernel, and native-artifact evidence.
Unavailable observations remain unavailable. A target match is reported only
when the observed vendor/product/CPU/core/SMT profile agrees with the frozen
baseline.

The unit consumed by a receiver is an `EvidencePacket`, not a timing scalar.
Each packet contains ordered `ProbeFragment` records. A fragment preserves its
probe identity, role, timing, quality, excitation relationship, payload, and
one of the explicit states `observed`, `unavailable`, `failed`, `partial`, or
`corrupted`. Missingness is a mask; an observed zero is never converted into a
missing zero. Packet shards are append-only JSONL with atomic finalization and
streaming iteration.

The default model input is limited to payload values, observed masks, fragment
masks, quality, and an optional coded-excitation vector. Packet identifiers,
run/session/boot/host/DIMM/address identifiers, schedules, provenance, and
labels remain outside the model path. Labels are only required by supervised
heads; the unlabeled process model and JEPA target path do not consume them.

Worker-03 excitation is represented by a deterministic request codebook and a
separate execution record. The first code families are PRBS, Walsh-like,
read-pressure, write-pressure, and active/quiet schedules. Core roles are
explicitly assigned and validated for unique physical cores. Requested code,
executed code, interrupted positions, clocks, and compliance are never
collapsed into one claim.

The native kernel adds versioned, auditable x86 timing entry points for scalar
cached/flushed controls, a dependency/use chain, repeated load response, and a
paired cached differential. The C implementation uses explicit compiler
barriers and x86 fences; it does not claim DRAM reachability, physical address
knowledge, or row/bank/DIMM identity. `make -C native disassemble` provides an
audit view of the exported routines.

The initial receiver ladder is logistic regression, boosted trees, a small
CNN/TCN path, weak-evidence aggregation, JEPA-like linear/tiny-MLP probes,
predictive-coding refinement, and a JEPA + predictive-coding hybrid. The
implemented latent receiver uses a compact temporal fragment encoder, a
stop-gradient/EMA-style target encoder, a latent predictor, and a bounded
number of prediction-error refinement steps. A streaming noise residualizer
learns per-probe running means from an explicitly fingerprinted unlabeled
reference corpus and subtracts only observed values.

All training takes a re-openable batch factory. It does not move a full corpus
to a device, retain validation graphs, or retain evaluation prediction arrays.
Acquisition and training are separate modules and phases: finalized immutable
evidence must exist before receiver training begins.

## Controls and claim levels

The synthetic broken-packet control distributes an injected relational signal
over weak fragment pairs, includes missing/noisy/misleading fragments, and
reports null, shuffled-label, metadata-firewall, and relation-ablation controls.
It validates receiver mechanics only. Synthetic success is not worker-03
evidence and is not evidence of a DRAM channel.

Claims are recorded at explicit levels: Level 1 calibrated exact-host,
Level 2 unseen location on worker-03, Level 3 unseen session, Level 4 unseen
boot, Level 5 unseen DIMM, and Level 6 unseen host. A result at one level does
not open any higher level.

The historical frozen commodity PMU/warmup result remains `C: primitive
unsuitable`. This new packet receiver is a new primitive and cannot authorize
the rejected hidden-bit campaign.

## Consequences and future work

This architecture can accept future controlled-interface or FPGA evidence by
converting observations into the same packet contract. A mock packet or native
CPU packet cannot pass as physical controlled-hardware evidence. A future
physical adapter must provide its own continuity proof, hardware topology,
command/trigger/clock provenance, and complete session ledger before the
existing physical-evidence firewall can accept it.

The next experiment should first collect a long unlabeled/reference packet
baseline and then a small, preregistered worker-03 exact-host controlled-state
dataset with counterbalanced coded excitation, paired target/reference probes,
session holdout, shuffled labels, and a frozen receiver tournament. No live
worker-03 acquisition is performed by this software milestone.
