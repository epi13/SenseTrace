"""Portable capability records for measurement primitives."""

from __future__ import annotations

from typing import Any

from .primitive import AccessStateOracle, PrimitiveCapabilities


def commodity_timing_capabilities(*, operation: str, cache_control: str) -> PrimitiveCapabilities:
    """Declare what the commodity timed-load path can actually establish."""

    return PrimitiveCapabilities(
        operation_issues_memory_access="known" if operation == "memory_read" else "known",
        independent_access_state_oracle="unsupported",
        cache_residency_control="known" if cache_control in {"none", "clflush"} else "unknown",
        translation_state="unknown",
        depends_on_virtual_addresses="known",
        physical_address_information="unsupported",
        row_bank_channel_topology="unsupported",
        privileged_counters="unsupported",
        kernel_support="known" if cache_control == "clflush" else "unknown",
        external_hardware="unsupported",
        destructive_or_state_changing="known",
        replay_across_sessions_boots_devices="unknown",
    )


def commodity_timing_oracle(*, operation: str, cache_control: str) -> AccessStateOracle:
    """Describe requested cache state without promoting it to a DRAM oracle."""

    requested = {
        "none": "no cache eviction requested; the load may be satisfied by any cache level",
        "eviction_buffer": "best-effort user-space eviction sweep; resulting cache state is not independently known",
        "clflush": "CLFLUSH/MFENCE invalidation requested; resulting memory-layer path is not independently known",
    }.get(cache_control, "cache state unavailable")
    if operation == "idle":
        requested = "no target memory operation; timer control only"
    return AccessStateOracle(
        name="commodity-requested-access-state",
        strength="unavailable",
        status="unavailable",
        observation=requested,
        source="SenseTrace primitive control metadata; no independent hardware event",
        independent_of_latency=False,
        model_feature_eligible=False,
    )


def primitive_contract(
    *,
    primitive: str,
    capabilities: PrimitiveCapabilities,
    oracle: AccessStateOracle,
) -> dict[str, Any]:
    """Return a serializable contract suitable for protocol hashing."""

    return {
        "primitive": primitive,
        "capabilities": capabilities.as_dict(),
        "access_state_oracle": oracle.as_dict(),
        "provenance_policy": {
            "oracle_identity_is_audit_only": True,
            "addresses_and_allocations_are_audit_only": True,
            "session_and_boot_ids_are_audit_only": True,
            "model_eligible_features": "trace-derived features only by default",
        },
    }
