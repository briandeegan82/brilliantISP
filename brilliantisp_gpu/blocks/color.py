"""GPU colour correction matrix (normalized RGB, no clip/rescale).

Mirrors _matrix_linear + ColorCorrectionMatrixOptimized.compute_linear:
input widened directly to f64, finite 3x3 f64 coefficient rows from
config, reshape @ mat.T, checked f32 output. Disabled returns the input
as f32 and never reads the coefficients.
"""

from __future__ import annotations

import numpy as np

from ..types import GpuError, require_torch
from ..ops.array import checked_float32


def color_correction_matrix(image, sensor: dict, params: dict):
    """BHWC3 -> BHWC3 float32 via the 3x3 CCM; no clip."""
    torch = require_torch()
    arr = torch.as_tensor(image)
    if arr.ndim != 4 or arr.shape[-1] != 3 or not arr.numel():
        raise GpuError(f"CCM expects nonempty BHWC3 input, got {tuple(arr.shape)}")
    if arr.dtype == torch.bool or arr.is_complex():
        raise GpuError(f"CCM expects real input, got {arr.dtype}")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("CCM compute_linear requires a finite HWC3 array")
    if not params.get("is_enable", True):
        return checked_float32(arr, name="color_correction_matrix")
    rows = [
        np.asarray(params["corrected_red"], dtype=np.float64),
        np.asarray(params["corrected_green"], dtype=np.float64),
        np.asarray(params["corrected_blue"], dtype=np.float64),
    ]
    mat = np.stack(rows)
    if mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
        raise ValueError("CCM matrix must be finite 3x3 rows")
    f = arr.to(torch.float64)
    m = torch.as_tensor(mat, dtype=torch.float64, device=arr.device)
    out = (f.reshape(-1, 3) @ m.t()).reshape(f.shape)
    return checked_float32(out, name="color_correction_matrix")
