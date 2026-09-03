from __future__ import annotations

import ctypes

import pytest

from sensetrace.acquisition.native import NativeMeasurementKernel
from sensetrace.acquisition.probe_contract import (
    ProbeFailure,
    ProbeImplementation,
    ProbeRequest,
    ProbeSampleRecord,
)


def _implementation() -> ProbeImplementation:
    return ProbeImplementation(
        implementation_id="test-probe",
        implementation_version="1",
        backend_kind="test",
        artifact_sha256="a" * 64,
        architecture="test-arch",
        kernel_release="test-kernel",
        compatibility_status="available",
        timing_source="test monotonic counter",
        result_units="ticks",
    )


def test_probe_contract_requires_failure_for_unsupported_operation():
    with pytest.raises(ValueError, match="require explicit failure"):
        ProbeSampleRecord(
            implementation=_implementation(),
            request=ProbeRequest("session", 0, "unknown", {}),
            status="unsupported",
            monotonic_start_ns=1,
            monotonic_end_ns=2,
            clock_domain="test",
            raw_result=None,
            result_units="ticks",
            cpu_before=None,
            cpu_after=None,
            requested_affinity=None,
            effective_affinity=None,
        ).validate()


def test_probe_contract_rejects_unit_mismatch():
    with pytest.raises(ValueError, match="units disagree"):
        ProbeSampleRecord(
            implementation=_implementation(),
            request=ProbeRequest("session", 0, "timer", {}),
            status="complete",
            monotonic_start_ns=1,
            monotonic_end_ns=2,
            clock_domain="test",
            raw_result=[1],
            result_units="nanoseconds",
            cpu_before=None,
            cpu_after=None,
            requested_affinity=None,
            effective_affinity=None,
        ).validate()


def test_native_probe_unknown_operation_fails_closed():
    kernel = object.__new__(NativeMeasurementKernel)
    kernel.supports_clflush = False
    kernel.implementation_contract = lambda: _implementation()  # type: ignore[method-assign]
    result = kernel.execute(ProbeRequest("session", 2, "unknown", {"repetitions": 1}))
    assert result.status == "unsupported"
    assert result.failure == ProbeFailure(
        kind="unsupported_operation",
        capability="unknown",
        message="native measurement kernel does not implement 'unknown'",
    )
    assert "direct DRAM" in result.as_dict()["claim_boundary"]


def test_native_probe_contract_with_built_library_if_available():
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        pytest.skip("native library is not built on this host")
    assert kernel is not None
    value = ctypes.c_uint64(7)
    record = kernel.execute(
        ProbeRequest("session", 0, "cached_load", {"repetitions": 2}),
        address=ctypes.addressof(value),
    )
    assert record.status == "complete"
    assert len(record.raw_result or []) == 2
    assert record.implementation.artifact_sha256
