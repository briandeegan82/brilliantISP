from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import GpuError, require_torch


@dataclass(frozen=True)
class PcaDiagnostics:
    """On-device intermediates of one pca_means evaluation (no D2H)."""

    means: object
    unit: object
    projections: object
    order: object
    selected_indices: object
    covariance: object
    eigenvalues: object
    eigenvector: object


def pca_means(rows, percentage: float, *, return_diagnostics: bool = False):
    """Torch port of ``AutoWhiteBalance._estimate_means(rows, 'pca')``.

    ``rows`` is an (N,3) float64 CUDA tensor of rejection-filtered samples
    and ``percentage`` lies in (0,100]. Returns the absolute
    max-eigenvalue eigenvector of the selected-sample second moment,
    shape (3,). With ``return_diagnostics=True`` returns
    ``(means3, PcaDiagnostics)`` whose fields are GPU tensors.

    Projections use the runtime-compiled fused-fma kernel and ordering
    uses the runtime-compiled reference argsort matching the NumPy 1.26.4
    scalar quicksort finite-key path, so equal-projection ties follow the
    same selection as the CPU reference. CPU input is rejected with a
    named GpuError — there is no CPU fallback.
    """
    torch = require_torch()
    arr = torch.as_tensor(rows)
    if not isinstance(arr, torch.Tensor) or not arr.is_cuda:
        raise GpuError("pca_means requires CUDA/HIP device rows (no CPU fallback)")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise GpuError(f"pca_means expects (N,3) rows, got {tuple(arr.shape)}")
    if arr.shape[0] == 0:
        raise GpuError("pca_means requires at least one sample")
    if not math.isfinite(percentage) or percentage <= 0 or percentage > 100:
        raise GpuError(f"pca percentage must be in (0,100], got {percentage!r}")
    data = arr.to(torch.float64).contiguous()
    if not bool(torch.isfinite(data).all()):
        raise GpuError("pca_means requires finite rows")
    means = data.mean(dim=0)
    if bool((means <= 0).any()) or not bool(torch.isfinite(means).all()):
        raise GpuError("AWB channel means are degenerate")
    unit = means / torch.linalg.norm(means)
    from ..ops.reference_sort import fused_project, reference_argsort

    projections = fused_project(data, unit.contiguous())
    order = reference_argsort(projections)
    index = max(1, int(math.ceil(data.shape[0] * percentage / 100)))
    selected_indices = torch.cat((order[:index], order[-index:]))
    selected = data[selected_indices]
    sigma = selected.mT @ selected
    eig_value, eig_vector = torch.linalg.eigh(sigma)
    eigvec = eig_vector[:, int(torch.argmax(eig_value))].abs()
    if return_diagnostics:
        return eigvec, PcaDiagnostics(
            means=means,
            unit=unit,
            projections=projections,
            order=order,
            selected_indices=selected_indices,
            covariance=sigma,
            eigenvalues=eig_value,
            eigenvector=eigvec,
        )
    return eigvec
