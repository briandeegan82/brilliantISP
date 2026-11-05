"""GpuOptions validation and device/vendor failure honesty."""

from __future__ import annotations

import pytest

from brilliantisp_gpu.types import (
    GpuOptions,
    GpuUnavailableError,
    GpuVendorError,
)
from brilliantisp_gpu import device as gpu_device


def test_options_defaults():
    o = GpuOptions()
    assert o.device == "cuda:0"
    assert o.max_batch_size == 4
    assert o.memory_fraction == 0.70
    assert o.prefetch == 2
    assert o.profile_stages is False
    assert o.split_on_oom is True
    assert o.device_index == 0


def test_options_rejects_cpu_and_bad_fields():
    with pytest.raises(ValueError):
        GpuOptions(device="cpu")
    with pytest.raises(ValueError):
        GpuOptions(device="cpu:0")
    with pytest.raises(ValueError):
        GpuOptions(device="")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda:")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda:x")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda:-1")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda:0:1")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda:00")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda:+1")
    with pytest.raises(ValueError):
        GpuOptions(device="cuda: 0")
    with pytest.raises(ValueError):
        GpuOptions(max_batch_size=0)
    with pytest.raises(ValueError):
        GpuOptions(memory_fraction=0.0)
    with pytest.raises(ValueError):
        GpuOptions(memory_fraction=1.5)
    with pytest.raises(ValueError):
        GpuOptions(memory_fraction=float("nan"))
    with pytest.raises(ValueError):
        GpuOptions(memory_fraction=float("inf"))
    with pytest.raises(ValueError):
        GpuOptions(prefetch=-1)


def test_options_frozen():
    o = GpuOptions()
    with pytest.raises(Exception):
        o.device = "cpu"


@pytest.mark.gpu_hardware
def test_resolve_real_device(cuda_device):
    assert cuda_device.type == "cuda"
    props = gpu_device.device_properties(cuda_device)
    assert props["vendor"] in ("nvidia", "amd")
    assert props["total_memory"] > 0
    assert (props["hip"] is not None) == (props["vendor"] == "amd")


@pytest.mark.gpu_hardware
def test_scope_restores_flags(cuda_device, torch_mod):
    torch = torch_mod
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    with gpu_device.gpu_scope(cuda_device) as dev:
        assert dev == cuda_device
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.is_inference_mode_enabled()
    assert torch.backends.cuda.matmul.allow_tf32 is True
    torch.backends.cuda.matmul.allow_tf32 = prev_m
    assert torch.backends.cudnn.allow_tf32 == prev_c


@pytest.mark.gpu_hardware
def test_scope_restores_flags_on_exception(cuda_device, torch_mod):
    torch = torch_mod
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with pytest.raises(RuntimeError, match="boom"):
        with gpu_device.gpu_scope(cuda_device):
            raise RuntimeError("boom")
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    torch.backends.cuda.matmul.allow_tf32 = prev_m
    torch.backends.cudnn.allow_tf32 = prev_c


def _run_pytest(env_extra):
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/gpu/test_imports.py", "-q",
         "--require-gpu", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=300,
    )


def test_require_gpu_fails_without_device():
    proc = _run_pytest({"CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode != 0


def test_require_gpu_fails_without_torch():
    """Actual import blocker: pytest main with torch unimportable must fail."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import pytest\n"
        "sys.exit(pytest.main(['tests/gpu/test_imports.py', '-q', '--require-gpu']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=repo, timeout=300,
    )
    assert proc.returncode != 0


def test_require_gpu_fails_wrong_vendor():
    from brilliantisp_gpu import device as gpu_device
    other = "amd" if gpu_device.backend_vendor() == "nvidia" else "nvidia"
    import os, subprocess, sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc2 = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/gpu/test_imports.py", "-q",
         "--require-gpu", "--gpu-vendor", other, "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=300,
    )
    assert proc2.returncode != 0


@pytest.mark.gpu_hardware
def test_wrong_vendor_raises(cuda_device):
    actual = gpu_device.backend_vendor()
    other = "amd" if actual == "nvidia" else "nvidia"
    with pytest.raises(GpuVendorError):
        gpu_device.require_vendor(other)


@pytest.mark.gpu_hardware
def test_out_of_range_device_raises(cuda_device, torch_mod):
    with pytest.raises(GpuUnavailableError):
        gpu_device.resolve_device(GpuOptions(device="cuda:99"))


def test_missing_torch_names_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(GpuUnavailableError):
        gpu_device.torch_module()
