from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from tools.gui.analysis import normalize_roi
from util.raw_decode import RawFormat


@dataclass(frozen=True)
class ComparisonIdentity:
    source_fingerprint: str
    geometry: str
    domain: str
    shape: tuple[int, ...]


def comparison_identity(
    source_fingerprint: str, config: Mapping[str, Any], shape: Sequence[int],
    *, domain: str = "display_srgb8",
) -> ComparisonIdentity:
    """Conservative registration identity: source, decode, crop and scale."""
    if not source_fingerprint or not domain:
        raise ValueError("comparison source fingerprint and domain are required")
    signature = {
        "decode": asdict(RawFormat.from_sensor_info(config["sensor_info"])),
        "crop": config.get("crop", {}), "scale": config.get("scale", {}),
        "shape": list(shape),
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return ComparisonIdentity(
        source_fingerprint, hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        domain, tuple(int(v) for v in shape),
    )


def compatibility_reason(a: ComparisonIdentity, b: ComparisonIdentity) -> str | None:
    if a.source_fingerprint != b.source_fingerprint:
        return "Different source frames; pixel registration is not established."
    if a.shape != b.shape or a.geometry != b.geometry:
        return "Different decode/crop/scale geometry; pixel registration is not established."
    if a.domain != b.domain:
        return "Different signal domains; numeric differences are not comparable."
    return None


def _pair(
    a: np.ndarray, b: np.ndarray, identity_a: ComparisonIdentity, identity_b: ComparisonIdentity,
) -> tuple[np.ndarray, np.ndarray]:
    reason = compatibility_reason(identity_a, identity_b)
    if reason:
        raise ValueError(reason)
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != identity_a.shape or bb.shape != identity_b.shape:
        raise ValueError("array geometry disagrees with its comparison identity")
    if aa.ndim not in (2, 3) or not aa.size or aa.dtype.kind not in "uif" or bb.dtype.kind not in "uif":
        raise ValueError("comparison requires nonempty real image arrays")
    if aa.ndim == 3 and aa.shape[-1] != 3:
        raise ValueError("comparison HWC arrays must have three components")
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        raise ValueError("comparison values must be finite")
    return aa, bb


def difference_statistics(
    a: np.ndarray, b: np.ndarray, *,
    identity_a: ComparisonIdentity, identity_b: ComparisonIdentity,
    roi: Sequence[int] | None = None,
) -> dict[str, Any]:
    aa, bb = _pair(a, b, identity_a, identity_b)
    bounds = normalize_roi(roi, aa.shape[1], aa.shape[0])
    x0, y0, x1, y1 = bounds
    with np.errstate(over="ignore", invalid="ignore"):
        delta = aa[y0:y1, x0:x1].astype(np.float64) - bb[y0:y1, x0:x1].astype(np.float64)
        absolute = np.abs(delta)
        mse = float(np.mean(delta * delta, dtype=np.float64))
    if not np.all(np.isfinite(delta)) or not math.isfinite(mse):
        raise ValueError("comparison exceeds supported float64 magnitude")
    changed = np.any(delta != 0, axis=2) if delta.ndim == 3 else delta != 0
    return {
        "domain": identity_a.domain, "roi": list(bounds),
        "pixel_count": int(delta.shape[0] * delta.shape[1]),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_components": int(np.count_nonzero(delta)),
        "maximum_absolute": float(absolute.max()),
        "mean_absolute": float(np.mean(absolute, dtype=np.float64)),
        "rmse": math.sqrt(mse),
        "note": "Differences in declared numeric units; not perceptual Delta-E or an image-quality score.",
    }


def compose_rgb8(
    a: np.ndarray, b: np.ndarray, *,
    identity_a: ComparisonIdentity, identity_b: ComparisonIdentity,
    mode: str, fraction: float = 0.5, difference_gain: float = 1.0,
) -> np.ndarray:
    aa, bb = _pair(a, b, identity_a, identity_b)
    if aa.ndim != 3 or aa.dtype != np.uint8 or bb.dtype != np.uint8:
        raise ValueError("display comparisons require uint8 RGB")
    if mode == "wipe":
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("wipe fraction must be in 0..1")
        split = int(math.floor(fraction * aa.shape[1]))
        result = bb.copy()
        result[:, :split] = aa[:, :split]
        return result
    if mode == "difference":
        if isinstance(difference_gain, bool) or not math.isfinite(difference_gain) or not 0 < difference_gain <= 64:
            raise ValueError("difference gain must be in (0, 64]")
        delta = np.abs(aa.astype(np.int16) - bb.astype(np.int16)).astype(np.float64)
        return np.rint(np.clip(delta * difference_gain, 0, 255)).astype(np.uint8)
    raise ValueError("comparison mode must be wipe or difference")
