"""Explicit border index maps matching SciPy/NumPy extension semantics.

Modes:
- ``reflect``: scipy.ndimage 'reflect' / np.pad 'symmetric' — edge sample
  repeated (d c b a | a b c d | d c b a), fold period 2n.
- ``mirror``: scipy.ndimage 'mirror' / np.pad 'reflect' — edge sample not
  repeated (d c b | a b c d | c b a), fold period 2n-2.
- ``nearest``/``replicate``/``edge``: clamp to the boundary sample.
- ``constant``/``zero``: positions outside the array map to sentinel -1
  (caller substitutes cval).
- ``wrap``: periodic extension.

Singleton dimensions always fold to index 0; radii may exceed the extent.
"""

from __future__ import annotations

from ..types import require_torch

_SENTINEL = -1

_ALIASES = {
    "reflect": "reflect",
    "symmetric": "reflect",
    "symm": "reflect",
    "mirror": "mirror",
    "nearest": "nearest",
    "replicate": "nearest",
    "edge": "nearest",
    "constant": "constant",
    "zero": "constant",
    "wrap": "wrap",
    "periodic": "wrap",
}


def _normalize_mode(mode: str) -> str:
    try:
        return _ALIASES[str(mode)]
    except KeyError:
        raise ValueError(f"unknown border mode {mode!r}") from None


def border_indices(length: int, radius: int, mode: str = "reflect", *, device=None):
    """Source index for every position in ``[-radius, length + radius)``.

    Returns an int64 tensor of ``length + 2 * radius`` entries; 'constant'
    positions carry the -1 sentinel. ``radius`` may be zero or exceed
    ``length``; ``length`` must be positive.
    """
    torch = require_torch()
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"length must be a positive integer, got {length!r}")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError(f"radius must be a nonnegative integer, got {radius!r}")
    mode = _normalize_mode(mode)
    positions = torch.arange(-radius, length + radius, dtype=torch.int64, device=device)
    n = length
    if mode == "constant":
        return torch.where((positions < 0) | (positions >= n), _SENTINEL * 1, positions)
    if mode == "nearest" or n == 1:
        return positions.clamp(0, n - 1)
    if mode == "wrap":
        return positions % n
    if mode == "reflect":
        period = 2 * n
        t = positions % period
        return torch.where(t < n, t, period - 1 - t)
    period = 2 * n - 2
    t = positions % period
    return torch.where(t < n, t, period - t)


def pad(x, radius, mode: str = "reflect", cval: float = 0.0):
    """Pad the last two axes of a tensor via explicit border indices.

    ``radius`` is an int or ``(rows, cols)`` pair. 'constant' fills with
    ``cval``; other modes gather through the folded index maps.
    """
    torch = require_torch()
    arr = torch.as_tensor(x)
    if arr.ndim < 2:
        raise ValueError(f"pad expects >=2 dims, got {arr.shape}")
    if isinstance(radius, int):
        ry = rx = radius
    else:
        ry, rx = radius
    h, w = arr.shape[-2], arr.shape[-1]
    rows = border_indices(h, ry, mode, device=arr.device)
    cols = border_indices(w, rx, mode, device=arr.device)
    if _normalize_mode(mode) == "constant":
        out = arr.new_full(arr.shape[:-2] + (h + 2 * ry, w + 2 * rx), cval)
        out[..., ry:ry + h, rx:rx + w] = arr
        return out
    return arr[..., rows, :][..., :, cols]
