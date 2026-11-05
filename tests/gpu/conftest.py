from __future__ import annotations

import pytest

from brilliantisp_gpu.types import GpuOptions


def pytest_addoption(parser):
    parser.addoption("--require-gpu", action="store_true", default=False,
                     help="fail instead of skipping when real GPU hardware is unavailable")
    parser.addoption("--gpu-vendor", choices=["nvidia", "amd"], default="nvidia",
                     help="required GPU vendor for hardware tests")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu_hardware: requires a real CUDA/ROCm device (not a CPU-tensor qualification)"
    )


def _resolve(request):
    from brilliantisp_gpu.types import GpuUnavailableError, GpuVendorError
    from brilliantisp_gpu import device as gpu_device

    require = request.config.getoption("--require-gpu")
    expected = request.config.getoption("--gpu-vendor")

    def _unavailable(msg: str):
        if require:
            pytest.fail(msg)
        pytest.skip(msg)

    try:
        gpu_device.torch_module()
    except GpuUnavailableError as exc:
        _unavailable(str(exc))
    try:
        gpu_device.require_vendor(expected)
    except (GpuVendorError, GpuUnavailableError) as exc:
        _unavailable(str(exc))
    try:
        return gpu_device.resolve_device()
    except GpuUnavailableError as exc:
        _unavailable(str(exc))


@pytest.fixture(scope="session", autouse=True)
def _gpu_gate(request):
    """Under --require-gpu the whole run fails fast when torch is missing,
    CPU-only, the wrong vendor, or no device is resolvable."""
    if request.config.getoption("--require-gpu"):
        _resolve(request)


@pytest.fixture(scope="session")
def gpu_options():
    return GpuOptions(device="cuda:0")


@pytest.fixture(scope="session")
def torch_mod(request):
    if request.config.getoption("--require-gpu"):
        return _resolve(request) and __import__("torch")
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def cuda_device(request):
    """Real GPU device; skips without --require-gpu, fails with it."""
    return _resolve(request)


@pytest.fixture(params=[
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu_hardware),
])
def tensor_device(request):
    """CPU-tensor coverage plus a real-device pass for helpers."""
    if request.param == "cpu":
        return "cpu"
    return _resolve(request)
