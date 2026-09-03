# ADR-013: Phase 2 controlled-memory-interface boundary

- Status: accepted
- Date: 2026-09-03

## Context

The genuine three-boot scoped-PMU campaign (2026-09-03) preserved decision B:
the `cpu/cache-misses/` directional contrast reproduced 9/9 across boots, but
the null was unstable within and across boots, characterized as a
first-use-of-allocation cold transient. The one predeclared warm-up follow-up
then completed under the frozen v2 gate: all warm-ups were native and all PMU
reads were non-multiplexed, but directional agreement failed in two boots and
the pooled null stability failed. The commodity timed-load/CLFLUSH path is now
closed for scaling. SenseTrace therefore needs a clean research boundary for
controlled memory hardware before any device is connected.

## Decision

Phase 2 work starts from `src/sensetrace/acquisition/controlled.py`:

- `ControlledMemoryInterface` declares the contract a future controller must
  satisfy: physical experiment-target identity, externally controlled
  address/command identity, command timing, refresh relationship, acquisition
  trigger, analog/digital trace channels, hardware clock, controller firmware
  identity, controller configuration hash, device/DIMM identity, calibration
  state, and acquisition provenance.
- `ControlledTraceAcquisition` requires an acquisition/trigger identity,
  hardware-clock timing and uncertainty, refresh relationship, controlled
  command-sequence identity, and at least one `ControlledTraceChannel` with
  channel kind, units, clock, and calibration identity. A future implementation
  cannot return an anonymous waveform and call it controlled evidence.
  Required identities reject placeholder values (`""`, `"unavailable"`,
  `"unknown"`); synthetic/mock evidence uses explicit synthetic identities
  rather than generic missing-value placeholders.
- `ControlledMemoryTopology` carries row/bank/channel/rank/device/DIMM fields
  that are valid **only** with `source="controlled_hardware"`. Any concrete
  topology field with any other source raises; deriving topology from a
  virtual address raises unconditionally. No virtual-to-physical synthesis is
  possible through this interface.
- `SyntheticMockControlledInterface` emulates the controller lifecycle and
  `ControlledInterfaceAcquisitionBackend` adapts its validated command results
  and trace payloads into ordinary SenseTrace samples. The resulting
  `SyntheticMockControlledBackend` preserves grouped samples, label
  fingerprints, session/allocation provenance, unknown-device topology, and
  deterministic recovery. Its observation semantics say "synthetic mock
  trace" and it can never produce a physical claim.
- Synthetic traces and logical sample identities are deterministic functions of
  `(seed, sample_index, operation_identity)`. Reconstructing the backend and
  resuming at index N therefore produces the same remaining sequence as an
  uninterrupted run. Wall-clock start time remains attempt provenance only.

The existing infrastructure (journaling, sharding, hashing, provenance,
grouped splits, controls, model interfaces, recovery, evidence manifests) is
reused; no separate scientific pipeline is forked.

## Consequences

- A future controller integrates by implementing `ControlledMemoryInterface`
  and supplying hardware-sourced topology; anything less remains
  `unavailable` by construction, not by documentation alone.
- The mock backend gives the warm-up follow-up (or any future primitive) a
  tested recovery/data path without weakening commodity evidence rules.
- The native and eBPF planes remain separate from this interface: neither can
  manufacture controlled command, refresh, trigger, channel, or topology
  provenance.
- The warm-up characterization failed its frozen C gate, so the commodity
  Phase 1 line stops and Phase 2 is the only physical path forward.
