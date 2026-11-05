import numpy as np
import pytest

from brilliant_isp import BrilliantISP
from brilliantisp_gpu.engine import GpuEngine
from tools.gpu_validation.qualification import (
    MemoryReference, _compare_value, observe_cpu, observe_gpu,
)


def test_memory_capture_is_immutable_and_deduplicated():
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    memory = MemoryReference()
    descriptor = memory.capture(data)["array"]
    assert memory.capture(data)["array"] == descriptor and len(memory.blobs) == 1
    data[:] = 65535
    np.testing.assert_array_equal(memory.read(descriptor), np.arange(6, dtype=np.uint16).reshape(2, 3))


def test_memory_stage_gates_preserve_declared_scale_and_dtype():
    memory = MemoryReference()
    expected = memory.capture(np.zeros((2, 3), np.float32))
    assert _compare_value(expected, np.ones((2, 3), np.float32),
                          memory, "white_balance", 2**24 - 1)["passed"]
    assert not _compare_value(expected, np.ones((2, 3), np.float32),
                              memory, "white_balance", 4095)["passed"]
    assert not _compare_value(expected, np.zeros((2, 3), np.float64),
                              memory, "white_balance", 4095)["passed"]
    integer = memory.capture(np.ones((2, 3), np.uint16))
    assert not _compare_value(integer, np.zeros((2, 3), np.uint16),
                              memory, "crop", 4095)["passed"]


def test_memory_control_contracts_and_tuple_structure():
    memory = MemoryReference()
    assert _compare_value(memory.capture((1., 2.)), (1. + 1e-9, 2.),
                          memory, "auto_white_balance", 4095)["passed"]
    assert not _compare_value(memory.capture(1), True,
                              memory, "auto_exposure", 4095)["passed"]
    assert not _compare_value(memory.capture((1., 2.)), (1.,),
                              memory, "auto_white_balance", 4095)["passed"]


def test_stage_observers_restore_methods_after_exception():
    cpu, gpu = BrilliantISP._run_stage, GpuEngine._run_stage
    memory = MemoryReference()
    with pytest.raises(RuntimeError):
        with observe_cpu(memory):
            raise RuntimeError("test")
    with pytest.raises(RuntimeError):
        with observe_gpu(memory, 4095, []):
            raise RuntimeError("test")
    assert BrilliantISP._run_stage is cpu and GpuEngine._run_stage is gpu
