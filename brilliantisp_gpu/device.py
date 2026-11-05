from __future__ import annotations

import contextlib
from typing import Any, Iterator

from .types import (
    GpuOptions,
    GpuUnavailableError,
    GpuVendorError,
    require_torch,
)


def torch_module():
    return require_torch()


def backend_vendor() -> str:
    """Runtime vendor of the installed torch build: 'amd' when built for ROCm."""
    torch = require_torch()
    if torch.version.hip is not None:
        return "amd"
    if torch.version.cuda is not None:
        return "nvidia"
    return "cpu"


def resolve_device(device: str | GpuOptions = "cuda:0"):
    """Resolve to a real CUDA/ROCm torch.device; errors are honest and named.

    Vendor is reported truthfully from ``torch.version.hip``: ROCm builds
    surface as ``cuda`` devices through the HIP shim but are reported 'amd'.
    """
    torch = require_torch()
    options = device if isinstance(device, GpuOptions) else GpuOptions(device=device)
    vendor = backend_vendor()
    if vendor == "cpu":
        raise GpuUnavailableError(
            "installed torch has no CUDA or ROCm support "
            f"(torch {torch.__version__}, cuda={torch.version.cuda}, hip={torch.version.hip})"
        )
    if not torch.cuda.is_available():
        raise GpuUnavailableError(
            f"torch {vendor} backend reports no usable device for {options.device!r} "
            f"(cuda={torch.version.cuda}, hip={torch.version.hip})"
        )
    index = options.device_index
    if index >= torch.cuda.device_count():
        raise GpuUnavailableError(
            f"device {options.device!r} out of range; {torch.cuda.device_count()} device(s) visible"
        )
    return torch.device("cuda", index)


def require_vendor(expected: str) -> None:
    actual = backend_vendor()
    if actual != expected:
        raise GpuVendorError(
            f"GPU vendor mismatch: required {expected!r}, torch backend is {actual!r}"
        )


def device_properties(device: Any | None = None) -> dict[str, Any]:
    torch = require_torch()
    dev = device if device is not None else resolve_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "name": props.name,
        "vendor": backend_vendor(),
        "capability": f"{props.major}.{props.minor}",
        "total_memory": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
    }


@contextlib.contextmanager
def gpu_scope(device: Any | None = None) -> Iterator[Any]:
    """Scoped strict-inference context on the resolved CUDA/ROCm device.

    Disables TF32 matmul/cuDNN paths and autocast for the block, restores the
    caller's previous flags afterwards, and runs under inference_mode.
    """
    torch = require_torch()
    dev = device if device is not None else resolve_device()
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.device(dev), torch.inference_mode():
                yield dev
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
