# ADR-015: Phase 2 controlled acquisition spine

- Status: accepted
- Date: 2026-09-03

## Context

ADR-013 defined the controlled-memory interface, but the mock path bypassed
that interface and emitted ordinary samples directly. The next physical path
must be testable before a controller or FPGA is selected, without allowing a
software emulator to become physical evidence.

## Decision

The Phase 2 path is composed of:

1. `ControlledMemoryInterface`, which exposes provenance, opaque-token topology
   lookup, command issue, trace acquisition, and trace payload retrieval.
2. `SyntheticMockControlledInterface`, a deterministic emulator of that
   lifecycle. It declares `controlled-memory-interface-mock-v1`, hashes its
   controller configuration, keeps topology unavailable, and uses explicit
   synthetic identities.
3. `ControlledInterfaceAcquisitionBackend`, which validates each command
   result before converting it to the ordinary SenseTrace `Sample` and shard
   format. No parallel storage, grouping, split, or model pipeline exists.

Controlled records enforce runtime values, not only `Literal` annotations.
Results bind command sequence, trigger, refresh, timing, command clock, and
sampling clock relationships across command, provenance, acquisition, and
channel objects. Command/controller and trace sampling clocks are distinct
fields because a real system may legitimately use different clocks.

Completed mock runs use dataset purpose `phase2_mock_controlled` and are
validated as software/evidence-contract artifacts. The physical controlled
hardware gate requires a different purpose and interface identity and rejects
mock evidence. A successful dry-run therefore proves contract propagation,
hashing, journaling, shard persistence, recovery, and claim-boundary
enforcement—not a physical DRAM information channel.

Recovery is a backend capability. Deterministic synthetic and mock sessions
may replay finalized prefixes. Commodity physical acquisition and the future
real controlled adapter default to fail-closed unless they explicitly declare
continuity of the required physical session/controller/allocation provenance.

## Consequences

- `configs/phase2-controlled-mock.example.yaml` is a complete dry-run entry
  point.
- A future real adapter must implement the same interface and provide its own
  explicit protocol identity and physical evidence manifest contract.
- Mock identities, unavailable topology, and nonphysical purpose survive into
  samples and manifests, so later validation cannot silently upgrade them.
- Historical commodity functionality remains available for reproduction, but
  the failed frozen commodity scaling decision is not reopened by this path.
