"""Tiny broken-packet synthetic controls.

These controls validate receiver behavior only.  They deliberately contain no
claim about DRAM, native probes, or a physical information channel.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from .metrics import metric_value
from .packets import EvidencePacket, FragmentStatus, ProbeFragment


@dataclass(frozen=True)
class SyntheticPacketControl:
    positive_packets: tuple[EvidencePacket, ...]
    null_packets: tuple[EvidencePacket, ...]
    shuffled_labels: tuple[int, ...]
    ablated_packets: tuple[EvidencePacket, ...]
    claim_boundary: str = "synthetic receiver validation only; no physical measurement claim"


def _packet_label(packet: EvidencePacket) -> int:
    signs: dict[int, int] = {}
    for fragment in packet.fragments:
        if not fragment.model_eligible or fragment.payload is None:
            continue
        observed = np.asarray(fragment.payload, dtype=np.float32)[~fragment.effective_mask()]
        if observed.size:
            signs[fragment.sequence_position] = 1 if float(np.mean(observed)) >= 0 else -1
    relations: list[int] = []
    for position in range(0, max(signs, default=-1) + 1, 2):
        if position in signs and position + 1 in signs:
            relations.append(signs[position] * signs[position + 1])
    if not relations:
        return 0
    return int(sum(value > 0 for value in relations) * 2 >= len(relations))


def generate_broken_packet_control(
    *,
    packet_count: int = 96,
    fragment_count: int = 8,
    seed: int = 2026,
    missing_rate: float = 0.2,
    misleading_rate: float = 0.1,
    noise: float = 0.35,
    signal: float = 0.55,
) -> SyntheticPacketControl:
    """Generate weak fragments whose recoverable information is a parity relation."""

    if packet_count < 8 or packet_count % 2 or fragment_count < 4 or fragment_count % 2:
        raise ValueError(
            "packet_count must be even and fragment_count must be an even number >= four"
        )
    if not 0.0 <= missing_rate < 1.0 or not 0.0 <= misleading_rate < 1.0:
        raise ValueError("missing_rate and misleading_rate must be in [0, 1)")
    if noise <= 0 or signal <= 0:
        raise ValueError("noise and signal must be positive")
    rng = np.random.default_rng(seed)
    labels = np.asarray([0] * (packet_count // 2) + [1] * (packet_count // 2), dtype=np.uint8)
    rng.shuffle(labels)
    positive: list[EvidencePacket] = []
    null: list[EvidencePacket] = []
    ablated: list[EvidencePacket] = []
    for index, label in enumerate(labels):
        packet_id = f"synthetic-broken-packet-{index:06d}"
        signs: list[int] = []
        relation = 1 if label else -1
        for _pair in range(fragment_count // 2):
            base = int(rng.choice(np.asarray([-1, 1], dtype=np.int8)))
            signs.extend((base, base * relation))
        fragments: list[ProbeFragment] = []
        null_fragments: list[ProbeFragment] = []
        ablated_fragments: list[ProbeFragment] = []
        for position, sign in enumerate(signs):
            missing = bool(rng.random() < missing_rate)
            misleading = bool(rng.random() < misleading_rate)
            effective_sign = -int(sign) if misleading else int(sign)
            payload = np.asarray(
                [effective_sign * signal + rng.normal(0.0, noise)], dtype=np.float32
            )
            fragment_id = f"{packet_id}-fragment-{position:03d}"
            status = cast(FragmentStatus, "unavailable" if missing else "observed")
            fragments.append(
                ProbeFragment(
                    fragment_id=fragment_id,
                    probe_type="synthetic-weak-fragment",
                    probe_version="synthetic-broken-packet-v1",
                    sequence_position=position,
                    target_role="target",
                    payload=None if missing else payload,
                    status=status,
                    quality=0.35 if misleading else 1.0,
                    model_eligible=not missing,
                    audit_metadata={"misleading_injection": misleading},
                )
            )
            null_fragments.append(
                replace(
                    fragments[-1],
                    fragment_id=f"{packet_id}-null-fragment-{position:03d}",
                    payload=None
                    if missing
                    else np.asarray([rng.normal(0.0, noise)], dtype=np.float32),
                    audit_metadata={"misleading_injection": False},
                )
            )
            ablated_fragments.append(
                replace(
                    fragments[-1],
                    fragment_id=f"{packet_id}-ablated-fragment-{position:03d}",
                    payload=None
                    if missing
                    else np.asarray([rng.normal(0.0, noise)], dtype=np.float32),
                    audit_metadata={"injected_relation_removed": True},
                )
            )
        positive.append(
            EvidencePacket(
                packet_id=packet_id,
                target_reference="synthetic-controlled-target",
                acquisition_id="synthetic-acquisition-000",
                protocol_id="synthetic-broken-packet-v1",
                fragments=tuple(fragments),
                controls={"injected_relation": "fragment-sign parity"},
                provenance={"synthetic": True},
                label=int(label),
            )
        )
        null.append(
            replace(
                positive[-1],
                packet_id=f"{packet_id}-null",
                fragments=tuple(null_fragments),
            )
        )
        ablated.append(
            replace(
                positive[-1],
                packet_id=f"{packet_id}-ablated",
                fragments=tuple(ablated_fragments),
            )
        )
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return SyntheticPacketControl(
        tuple(positive), tuple(null), tuple(int(value) for value in shuffled), tuple(ablated)
    )


def evaluate_broken_packet_control(control: SyntheticPacketControl) -> dict[str, Any]:
    """Evaluate the injected parity, null, shuffled-label, and ablation controls."""

    def score(packets: tuple[EvidencePacket, ...], labels: np.ndarray | None = None) -> float:
        expected = np.asarray(
            [
                packet.label if labels is None else labels[index]
                for index, packet in enumerate(packets)
            ],
            dtype=np.uint8,
        )
        probabilities = np.asarray([_packet_label(packet) for packet in packets], dtype=np.float32)
        return metric_value(expected, probabilities, "balanced_accuracy")

    labels = np.asarray(
        [int(packet.label) for packet in control.positive_packets if packet.label is not None],
        dtype=np.uint8,
    )
    shuffled = np.asarray(control.shuffled_labels, dtype=np.uint8)
    return {
        "schema": "sensetrace.synthetic-broken-packet-report.v1",
        "packet_count": len(control.positive_packets),
        "fragment_count": len(control.positive_packets[0].fragments),
        "positive_control_balanced_accuracy": score(control.positive_packets),
        "null_balanced_accuracy": score(control.null_packets),
        "shuffled_label_balanced_accuracy": score(control.positive_packets, shuffled),
        "relation_ablated_balanced_accuracy": score(control.ablated_packets),
        "single_fragment_baseline": "not used as a claim; each fragment marginal is intentionally weak",
        "metadata_only": "unavailable; identifiers are not supplied to the parity decoder",
        "information_source": "injected relational parity across surviving fragment signs",
        "label_fingerprint": hashlib.sha256(labels.tobytes()).hexdigest(),
        "claim_boundary": control.claim_boundary,
        "null_interpretation": "chance-like null is receiver validation, not a physical null experiment",
    }
