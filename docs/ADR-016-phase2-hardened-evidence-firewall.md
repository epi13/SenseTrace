# ADR-016: Hardened Phase 2 recovery and physical evidence firewall

- Status: accepted
- Date: 2026-09-03

## Context

Phase 2 has a deterministic software mock, but a controller response is not
physical evidence merely because a manifest labels it as hardware. Recovery is
also part of scientific provenance: a restarted process must not silently
continue a run under a changed experiment contract or an unproven controller
identity.

## Decision

The runner uses `sensetrace-resume-contract-v1`. Once `run.json` exists, its
configuration hash, persisted normalized `config.json`, run parameters, and
backend recovery identity are immutable. A mismatch is rejected before config
material is rewritten, shards are quarantined, acquisition is reopened, or
resumed evidence is appended. The caller must create a new run directory and
identity.

Recovery capability is an explicit backend hook. `AcquisitionBackend` and the
generic `ControlledInterfaceAcquisitionBackend` fail closed. The deterministic
synthetic and mock backends explicitly implement replay validation. A future
physical adapter must provide a continuity decision and auditable evidence for
the controller session, configuration, firmware, target/device, calibration,
clocks, protocol, and any other identities it actually observes. Missing
continuity is never interpreted as continuity.

Physical controlled-hardware datasets must declare
`physical-controlled-hardware-evidence-v1` with a fixed required-field set.
The validator checks the configured backend, manifest, session ledger, shard
metadata, and reconstructed serialized `ControlledCommand`, result,
provenance, topology, acquisition, and channel objects. Duplicated fields must
agree. Required physical identities cannot be placeholders or mock, synthetic,
virtual, or derived identities. Topology is accepted only when supplied by the
controlled hardware interface; it is never decoded from a virtual address or
opaque token.

`FaultInjectingControlledInterface` is the conformance harness. It deliberately
produces malformed timing, command, trace, identity, topology, continuity, and
shutdown responses so adapters are tested at the same backend boundary they
must satisfy. The harness does not grant physical status to the mock.

## Consequences

- Mock recovery remains deterministic and testable, but mock evidence remains
  `phase2_mock_controlled` and is rejected by the physical gate.
- A physical adapter can use any transport; PCIe, USB, serial, Ethernet, FPGA,
  and vendor-specific details remain outside the evidence model.
- A future adapter must implement explicit lifecycle/provenance methods and
  emit the required physical contract before any physical dataset can enter
  analysis.
- The project has not acquired or claimed a physical information channel.
