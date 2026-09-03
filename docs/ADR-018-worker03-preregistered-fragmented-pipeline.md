# ADR-018: Preregistered worker-03 fragmented-evidence pipeline

## Decision

The first worker-03 fragmented-evidence experiment is identified by
`worker03-fragmented-exact-host-v1` and a SHA-256 protocol fingerprint. The
protocol freezes the target inventory match, native measurement kernel and
probe versions, ordered fragment composition, requested coded excitation,
label and reference procedures, split/claim hierarchy, bounded receiver
configuration space, metrics, controls, and stop rules.

Requested excitation is never treated as executed excitation. The acquisition
adapter must supply executed positions, affinity/scheduler observations,
interruptions, timing uncertainty, and compliance. A divergence that changes
the scientific protocol requires a new experiment identity.

The executable path is an exact-host native CPU observation path. It creates
ordered `EvidencePacket` fragments and writes append-only JSONL shards without
promoting virtual addresses, cache observations, or native timing into
controlled-memory or FPGA evidence. Reference packets are unlabeled and are
the only permitted residualizer fitting corpus; the residualizer records its
immutable source fingerprint.

## Adapter trust boundary

Internal consistency of serialized controller, command, trigger, topology, and
trace records is necessary but does not prove that an adapter is attached to
physical hardware. A physical claim that relies on an adapter must include an
`controlled-adapter-attestation-v1` record, bound to controller, firmware,
configuration, target, session, and host-inventory identities. Its explicit
trust assumptions are part of the evidence. An independently corroborated
physical claim is a separate future tier.

The historical Phase 1A commodity PMU result remains **C: primitive
unsuitable**. This milestone does not reopen it and does not add physical
worker-03 or FPGA evidence.

## Lifecycle

`planned → protocol_frozen → inventory_verified → reference_acquisition →
controlled_acquisition → evidence_finalized → split_frozen → training →
validation_selection → test_evaluation → decision`

The lifecycle is persisted as an append-only event stream and immutable state
artifact. Training requires finalized evidence and a frozen split; selection
uses validation only; test evaluation is one-shot; decision output is
machine-readable.

