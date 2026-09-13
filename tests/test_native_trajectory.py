from __future__ import annotations

import ctypes

import numpy as np
import pytest

from sensetrace.acquisition.native import TRAJECTORY_CONDITIONS, NativeMeasurementKernel


def test_versioned_trajectory_abi_retains_order_quality_and_pressure_start():
    kernel = NativeMeasurementKernel.load()
    if kernel is None:
        pytest.skip("native library is not built on this host")
    target = ctypes.c_uint64(0x1234)
    reference_storage = (ctypes.c_uint64 * 16)()
    reference_address = ctypes.addressof(reference_storage) + 8 * ctypes.sizeof(ctypes.c_uint64)
    eviction = bytearray(4096)
    conditions = np.asarray(
        [
            TRAJECTORY_CONDITIONS["cached_preloaded"],
            TRAJECTORY_CONDITIONS["clflush"],
            TRAJECTORY_CONDITIONS["eviction"],
            TRAJECTORY_CONDITIONS["timer_only"],
        ],
        dtype=np.uint8,
    )
    orders = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    result = kernel.measure_trajectory(
        ctypes.addressof(target),
        reference_address,
        conditions,
        orders,
        eviction_address=ctypes.addressof((ctypes.c_uint8 * len(eviction)).from_buffer(eviction)),
        eviction_bytes=len(eviction),
    )
    assert result["first_start_tsc"].dtype == np.uint64
    assert result["second_start_aux"].shape == (4,)
    assert np.array_equal(orders, np.asarray([0, 1, 0, 1], dtype=np.uint8))
    assert np.all((result["first_quality"] & 1) != 0)
    assert np.all(result["first_end_tsc"] >= result["first_start_tsc"])
    pressure = kernel.run_memory_pressure(
        ctypes.addressof(reference_storage),
        len(reference_storage),
        50_000,
        requested_cpu=-1,
    )
    assert pressure["started"] == 1
    assert pressure["end_tsc"] >= pressure["start_tsc"]
    assert kernel.provenance()["version"] == "sensetrace-native-kernel-v4"
    assert kernel.provenance()["trajectory_version"] == "sensetrace-native-trajectory-v1"
