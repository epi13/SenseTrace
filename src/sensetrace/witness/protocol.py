"""Frozen protocol fragment for experiments that opt into witness evidence."""

from __future__ import annotations

from typing import Any

from sensetrace.hashing import sha256_json

from .models import WitnessSession


def witness_protocol(config: dict[str, Any]) -> dict[str, Any]:
    witness = config.get("witness", {})
    requirement = witness.get("requirement", "optional")
    if requirement not in {"disabled", "optional", "required"}:
        raise ValueError("witness.requirement must be disabled, optional, or required")
    hooks = tuple(str(value) for value in witness.get("hooks", ()))
    return {
        "version": "ebpf-witness-protocol-v1",
        "requirement": requirement,
        "requested_hooks": list(hooks),
        "sample_states": [
            "clean",
            "witness_event_present",
            "incomplete_witness",
            "witness_unavailable",
        ],
        "automatic_sample_veto": False,
        "clock_alignment": "record domains, offset method, and uncertainty; never invent precision",
        "claim_boundary": "contextual host evidence only; not direct DRAM evidence",
    }


def witness_protocol_hash(config: dict[str, Any]) -> str:
    return sha256_json(witness_protocol(config))


def enforce_witness_requirement(protocol: dict[str, Any], session: WitnessSession | None) -> None:
    """Fail closed when a frozen protocol requires complete witness evidence."""

    requirement = protocol.get("requirement")
    if requirement == "required" and (session is None or session.status != "operational"):
        status = "not_collected" if session is None else session.status
        raise RuntimeError(f"frozen protocol requires operational witness evidence; got {status}")
    if requirement == "disabled" and session is not None:
        raise RuntimeError("frozen protocol disables witness collection for this experiment")
