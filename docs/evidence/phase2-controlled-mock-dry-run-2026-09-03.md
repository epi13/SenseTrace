# Phase 2 controlled mock vertical dry-run — 2026-09-03

## Purpose

This is a software/evidence-contract artifact for the Phase 2 progression. No
FPGA, external controller, physical DRAM topology source, or physical
measurement claim is involved.

## Reproduction

```text
sensetrace run phase2-mock \
  --config configs/phase2-controlled-mock.example.yaml \
  --output runs/phase2-controlled-mock
```

The runner can be interrupted with `--stop-after N` and invoked again with the
same configuration and output directory. The mock's deterministic logical
session identity permits replay from the finalized shard boundary. A future
real controlled adapter must declare a different recovery capability and must
fail closed unless physical session/controller/allocation continuity is
established. The runner persists a versioned immutable resume contract: a
changed seed, controller configuration, target/firmware identity, trace design,
or other experiment-significant configuration is rejected before config or
shard evidence is rewritten. The controller conformance harness separately
rejects malformed command, timing, trace, identity, topology, and shutdown
responses.

## Contract exercised

- configuration validation selects `controlled_mock` and the versioned
  `controlled-memory-interface-mock-v1` identity;
- the runner hashes the Phase 2 mock controller configuration;
- the runner persists normalized config material, run parameters, backend
  identity, and a fail-closed recovery decision;
- each row is produced by an opaque command passed through
  `ControlledMemoryInterface.issue()`, followed by trace acquisition and
  payload retrieval through the same interface;
- command sequence, trigger, command-clock, sampling-clock, trace-channel, and
  controller configuration identities survive into ordinary SenseTrace sample
  metadata and NPZ shards;
- topology is `source=unavailable`; virtual locations and opaque tokens are
  never converted into row/bank/channel/rank/device/DIMM truth;
- journals record recovery and manifest finalization, and the completed
  manifest is `phase2_mock_controlled`;
- `validate_physical_evidence_dataset()` rejects the resulting dataset.

The complete local validation sequence is: clean run; interrupted run with
`--stop-after`; identical-config resume; changed-config resume rejection;
temporary or invalid-shard quarantine; manifest generation and ordinary dataset
load; and explicit physical-evidence rejection. The mock remains software and
evidence-contract validation only.

The successful run demonstrates that the evidence plumbing is ready for a
future adapter. It does not demonstrate a physical DRAM information channel,
and it must not be combined with the historical commodity evidence or used to
reopen the frozen `C_primitive_unsuitable` decision.
