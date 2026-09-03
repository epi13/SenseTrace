"""Adapter attestation and evidence-source boundaries.

The evidence firewall can check that serialized records agree with one another,
but it cannot prove that an adapter is connected to the physical device it
claims to control.  This module makes that limitation explicit.  A new
physical claim must carry an adapter attestation; native CPU observations and
mock controller observations have separate, non-physical evidence tiers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .hashing import sha256_json

CONTROLLED_ADAPTER_ATTESTATION_VERSION = "controlled-adapter-attestation-v1"
UNKNOWN = "unavailable"
EvidenceTier = Literal[
    "internally_consistent",
    "adapter_attested_physical",
    "independently_corroborated_physical",
    "native_exact_host",
    "synthetic_or_mock",
]

_PLACEHOLDERS = frozenset({"", "unknown", "unavailable", "none", "null", "n/a"})
_SYNTHETIC_TOKENS = ("mock", "synthetic", "virtual", "derived", "fixture")


def _text(value: object, field_name: str, *, allow_unknown: bool = True) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} contains control characters")
    if not allow_unknown and value.casefold() in _PLACEHOLDERS:
        raise ValueError(f"{field_name} is unavailable")
    return value


def _physical_text(value: object, field_name: str) -> str:
    result = _text(value, field_name, allow_unknown=False)
    if any(token in result.casefold() for token in _SYNTHETIC_TOKENS):
        raise ValueError(f"{field_name} uses a synthetic identity")
    return result


@dataclass(frozen=True)
class ControlledAdapterAttestation:
    """What a controlled adapter itself attests, without inventing unknowns."""

    adapter_identity: str = UNKNOWN
    adapter_source_module: str = UNKNOWN
    adapter_driver_identity: str = UNKNOWN
    controller_identity: str = UNKNOWN
    controller_firmware_identity: str = UNKNOWN
    controller_configuration_fingerprint: str = UNKNOWN
    transport_type: str = UNKNOWN
    target_identity: str = UNKNOWN
    acquisition_session_identity: str = UNKNOWN
    observed_hardware_capability_record: dict[str, Any] = field(default_factory=dict)
    calibration_identity: str = UNKNOWN
    clock_identities: dict[str, Any] = field(default_factory=dict)
    topology_source_authority: str = UNKNOWN
    adapter_binary_source_fingerprint: str = UNKNOWN
    code_commit: str = UNKNOWN
    host_inventory_fingerprint: str = UNKNOWN
    created_at: str = UNKNOWN
    trust_assumptions: tuple[str, ...] = ()
    independently_corroborated: bool = False

    def validate(self, *, require_physical: bool = False) -> None:
        for name, value in asdict(self).items():
            if name in {"observed_hardware_capability_record", "clock_identities"}:
                if not isinstance(value, dict):
                    raise ValueError(f"{name} must be a mapping")
                json.dumps(value, allow_nan=False, sort_keys=True)
            elif name == "trust_assumptions":
                if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
                    raise ValueError("trust_assumptions must be a tuple of strings")
                for item in value:
                    _text(item, "trust assumption")
            elif name == "independently_corroborated":
                if not isinstance(value, bool):
                    raise ValueError("independently_corroborated must be boolean")
            else:
                _text(value, name)
        if require_physical:
            required = (
                "adapter_identity",
                "controller_identity",
                "controller_firmware_identity",
                "controller_configuration_fingerprint",
                "transport_type",
                "target_identity",
                "acquisition_session_identity",
                "topology_source_authority",
                "code_commit",
                "host_inventory_fingerprint",
            )
            for name in required:
                _physical_text(getattr(self, name), name)
            if self.topology_source_authority.casefold() != "controlled_hardware":
                raise ValueError(
                    "physical adapter attestation requires controlled_hardware topology authority"
                )
            if not self.observed_hardware_capability_record:
                raise ValueError(
                    "physical adapter attestation requires observed hardware capabilities"
                )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["schema"] = CONTROLLED_ADAPTER_ATTESTATION_VERSION
        return result

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ControlledAdapterAttestation:
        if not isinstance(record, dict):
            raise ValueError("adapter attestation must be a mapping")
        if record.get("schema") != CONTROLLED_ADAPTER_ATTESTATION_VERSION:
            raise ValueError("unsupported adapter attestation schema")
        values = dict(record)
        values.pop("schema", None)
        if "trust_assumptions" in values:
            values["trust_assumptions"] = tuple(values["trust_assumptions"])
        attestation = cls(**values)
        attestation.validate()
        return attestation

    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())

    def binds_to(
        self,
        *,
        adapter_identity: str | None = None,
        controller_identity: str,
        firmware_identity: str,
        configuration_fingerprint: str,
        target_identity: str,
        acquisition_session_identity: str,
        host_inventory_fingerprint: str | None = None,
    ) -> bool:
        """Check only identities actually supplied by the evidence contract."""

        self.validate(require_physical=True)
        expected = {
            "controller_identity": controller_identity,
            "controller_firmware_identity": firmware_identity,
            "controller_configuration_fingerprint": configuration_fingerprint,
            "target_identity": target_identity,
            "acquisition_session_identity": acquisition_session_identity,
        }
        if adapter_identity is not None:
            expected["adapter_identity"] = adapter_identity
        if host_inventory_fingerprint is not None:
            expected["host_inventory_fingerprint"] = host_inventory_fingerprint
        return all(getattr(self, key) == value for key, value in expected.items())


def evidence_source_record(
    *,
    internally_consistent: bool,
    tier: EvidenceTier,
    adapter_attestation: ControlledAdapterAttestation | None = None,
    independent_corroboration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an explicit, non-upgrading evidence-source declaration."""

    if tier == "adapter_attested_physical" and adapter_attestation is None:
        raise ValueError("adapter-attested physical evidence requires an attestation")
    if tier == "independently_corroborated_physical" and independent_corroboration is None:
        raise ValueError("independently corroborated evidence requires corroboration material")
    if tier in {"native_exact_host", "synthetic_or_mock"} and adapter_attestation is not None:
        raise ValueError("native or synthetic evidence cannot carry a physical adapter attestation")
    if tier == "adapter_attested_physical" and adapter_attestation is not None:
        adapter_attestation.validate(require_physical=True)
    record: dict[str, Any] = {
        "schema": "sensetrace.evidence-source.v1",
        "internally_consistent": bool(internally_consistent),
        "evidence_tier": tier,
        "physical_claim_authority": (
            "adapter_attestation"
            if tier == "adapter_attested_physical"
            else "independent_corroboration"
            if tier == "independently_corroborated_physical"
            else "none"
        ),
        "claim_boundary": (
            "adapter-attested physical controlled-interface evidence; hardware truth remains an adapter trust assumption"
            if tier == "adapter_attested_physical"
            else "independently corroborated physical evidence; corroboration scope must be reviewed"
            if tier == "independently_corroborated_physical"
            else "native exact-host observation; no controlled physical-memory claim"
            if tier == "native_exact_host"
            else "synthetic/mock or software-contract evidence; no physical claim"
        ),
    }
    if adapter_attestation is not None:
        adapter_attestation.validate()
        record["adapter_attestation"] = adapter_attestation.as_dict()
        record["adapter_attestation_fingerprint"] = adapter_attestation.fingerprint()
    if independent_corroboration is not None:
        json.dumps(independent_corroboration, allow_nan=False, sort_keys=True)
        record["independent_corroboration"] = independent_corroboration
    return record


def require_adapter_attestation(
    record: dict[str, Any],
    *,
    adapter_identity: str | None = None,
    controller_identity: str,
    firmware_identity: str,
    configuration_fingerprint: str,
    target_identity: str,
    acquisition_session_identity: str,
    host_inventory_fingerprint: str | None = None,
) -> ControlledAdapterAttestation:
    """Fail closed when a physical dataset lacks a bound adapter attestation."""

    source = record.get("evidence_source")
    if not isinstance(source, dict):
        raise ValueError("physical evidence requires an evidence_source record")
    if source.get("evidence_tier") != "adapter_attested_physical":
        raise ValueError(
            "physical evidence requires adapter attestation at the adapter_attested_physical tier"
        )
    raw = source.get("adapter_attestation")
    if not isinstance(raw, dict):
        raise ValueError("physical evidence requires adapter attestation")
    attestation = ControlledAdapterAttestation.from_dict(raw)
    if not attestation.binds_to(
        adapter_identity=adapter_identity,
        controller_identity=controller_identity,
        firmware_identity=firmware_identity,
        configuration_fingerprint=configuration_fingerprint,
        target_identity=target_identity,
        acquisition_session_identity=acquisition_session_identity,
        host_inventory_fingerprint=host_inventory_fingerprint,
    ):
        raise ValueError("adapter attestation identity does not bind to physical evidence")
    if source.get("adapter_attestation_fingerprint") != attestation.fingerprint():
        raise ValueError("adapter attestation fingerprint mismatch")
    return attestation
