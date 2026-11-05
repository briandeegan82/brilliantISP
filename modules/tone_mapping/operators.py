"""Pure display-linear tone curves, independent of storage and execution backend.

``aces_fitted`` is the scalar Narkowicz-style rational approximation, not a
complete ACES color transform. Exposure, normalization, and output transfer
are deliberately outside these functions.
"""

from __future__ import annotations

import numpy as np


def _nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("tone-curve input must contain only finite values")
    return np.maximum(values, 0.0)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def aces_fitted(values: np.ndarray) -> np.ndarray:
    """Evaluate ``x*(2.51*x+.03)/(x*(2.43*x+.59)+.14)``."""
    x = _nonnegative(values)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(mapped, 0.0, 1.0)


def hable_partial(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard partial curve without near-black cancellation."""
    x = _nonnegative(values)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mapped = (
            x * (0.15 * (1.0 - 0.02 / 0.30) * x + 0.50 * (0.10 - 0.02 / 0.30))
            / (x * (0.15 * x + 0.50) + 0.20 * 0.30)
        )
    return mapped


def hable(
    values: np.ndarray, *, exposure_bias: float = 2.0, white_point: float = 11.2
) -> np.ndarray:
    """Apply a linear bias and divide by the partial curve at ``white_point``."""
    bias = _positive("exposure_bias", exposure_bias)
    white = _positive("white_point", white_point)
    reference = float(hable_partial(np.asarray([white]))[0])
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("white_point has no representable positive Hable reference")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mapped = hable_partial(_nonnegative(values) * bias) / reference
    return np.clip(mapped, 0.0, 1.0)


def reinhard(
    values: np.ndarray, *, knee: float = 0.25, strength: float = 1.0
) -> np.ndarray:
    """Evaluate the generalized global curve ``x/(x+knee/strength)``."""
    knee = _positive("knee", knee)
    strength = _positive("strength", strength)
    white = knee / strength
    if not np.isfinite(white) or white <= 0:
        raise ValueError("knee/strength must have a representable positive ratio")
    x = _nonnegative(values)
    return np.clip(x / (x + white), 0.0, 1.0)
