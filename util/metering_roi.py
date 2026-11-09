"""Opt-in spatial metering region (ROI) for AE and AWB statistics.

Configured as an ``roi`` sub-mapping of ``auto_exposure`` and/or
``auto_white_balance``. Coordinates are normalized to the statistics image
(x: 0 left .. 1 right, y: 0 top .. 1 bottom), i.e. the frame after embedded-row
removal, sensor flips and crop. AE meters full-resolution RGB while AWB samples
half-resolution CFA planes; normalized coordinates select the same scene region
in both.

Keys (all optional except ``is_enable``)::

    roi:
      is_enable: true
      ellipse: {cx: 0.5, cy: 0.5, rx: 0.4, ry: 0.5}  # keep pixels inside
      include_rect: [x0, y0, x1, y1]                 # keep pixels inside
      exclude_rects: [[x0, y0, x1, y1], ...]
      exclude_polygons: [[[x, y], [x, y], [x, y], ...], ...]
      zone_weights: [[1, 1, 1], [2, 2, 2]]           # rows top->bottom, nearest zone

AE uses the resulting weights for its meter average and histogram moments.
AWB treats the ROI as a selection mask: pixels with a positive weight are
sampled with equal weight.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

_KEYS = frozenset({"is_enable", "ellipse", "include_rect", "exclude_rects", "exclude_polygons", "zone_weights"})
_CACHE: dict[tuple[int, int, str], np.ndarray] = {}
_CACHE_MAX = 32


def roi_config(section: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return the enabled ``roi`` sub-mapping of an AE/AWB section, else None."""
    if not isinstance(section, Mapping):
        return None
    roi = section.get("roi")
    if roi is None:
        return None
    if not isinstance(roi, Mapping):
        raise ValueError("roi must be a mapping")
    enabled = roi.get("is_enable", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("roi.is_enable must be a bool")
    return roi if enabled else None


def weight_map(shape: Sequence[int], roi: Mapping[str, Any] | None) -> np.ndarray | None:
    """Read-only float64 HxW weights for ``roi`` at ``shape``; None when inactive."""
    if roi is None:
        return None
    h, w = int(shape[0]), int(shape[1])
    key = (h, w, json.dumps(_plain(roi), sort_keys=True))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    weights = _build(h, w, roi)
    weights.setflags(write=False)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = weights
    return weights


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _num(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _rect(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_num(v, name) for v in value)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def _in_rect(x: np.ndarray, y: np.ndarray, r: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = r
    return ((y >= y0) & (y < y1)) & ((x >= x0) & (x < x1))


def _in_polygon(x: np.ndarray, y: np.ndarray, poly: Any, name: str) -> np.ndarray:
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        raise ValueError(f"{name} must be a list of at least 3 [x, y] vertices")
    pts = []
    for v in poly:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"{name} vertices must be [x, y]")
        pts.append((_num(v[0], name), _num(v[1], name)))
    inside = np.zeros((y.shape[0], x.shape[1]), dtype=bool)
    for (xi, yi), (xj, yj) in zip(pts, pts[1:] + pts[:1]):
        if yi == yj:
            continue
        crosses = (yi > y) != (yj > y)
        x_edge = xi + (xj - xi) * (y - yi) / (yj - yi)
        inside ^= crosses & (x < x_edge)
    return inside


def _build(h: int, w: int, roi: Mapping[str, Any]) -> np.ndarray:
    unknown = sorted(set(map(str, roi)) - _KEYS)
    if unknown:
        raise ValueError(f"unknown roi keys {unknown}; expected a subset of {sorted(_KEYS)}")
    if h <= 0 or w <= 0:
        raise ValueError(f"roi needs a nonempty statistics image, got {h}x{w}")
    y = ((np.arange(h, dtype=np.float64) + 0.5) / h)[:, None]
    x = ((np.arange(w, dtype=np.float64) + 0.5) / w)[None, :]
    mask = np.ones((h, w), dtype=bool)

    ellipse = roi.get("ellipse")
    if ellipse is not None:
        if not isinstance(ellipse, Mapping) or set(ellipse) != {"cx", "cy", "rx", "ry"}:
            raise ValueError("roi.ellipse must have exactly cx, cy, rx, ry")
        cx, cy, rx, ry = (_num(ellipse[k], f"roi.ellipse.{k}") for k in ("cx", "cy", "rx", "ry"))
        if rx <= 0 or ry <= 0:
            raise ValueError("roi.ellipse rx and ry must be positive")
        mask &= ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    include = roi.get("include_rect")
    if include is not None:
        mask &= _in_rect(x, y, _rect(include, "roi.include_rect"))

    rects = roi.get("exclude_rects") or []
    if not isinstance(rects, (list, tuple)):
        raise ValueError("roi.exclude_rects must be a list of [x0, y0, x1, y1]")
    for i, r in enumerate(rects):
        mask &= ~_in_rect(x, y, _rect(r, f"roi.exclude_rects[{i}]"))

    polys = roi.get("exclude_polygons") or []
    if not isinstance(polys, (list, tuple)):
        raise ValueError("roi.exclude_polygons must be a list of polygons")
    for i, p in enumerate(polys):
        mask &= ~_in_polygon(x, y, p, f"roi.exclude_polygons[{i}]")

    weights = mask.astype(np.float64)
    grid = roi.get("zone_weights")
    if grid is not None:
        if (not isinstance(grid, (list, tuple)) or not grid
                or not all(isinstance(row, (list, tuple)) and row for row in grid)
                or len({len(row) for row in grid}) != 1):
            raise ValueError("roi.zone_weights must be a nonempty rectangular list of rows")
        table = np.array([[_num(v, "roi.zone_weights") for v in row] for row in grid], dtype=np.float64)
        if np.any(table < 0) or not np.any(table > 0):
            raise ValueError("roi.zone_weights must be nonnegative with at least one positive zone")
        rows = np.minimum((y[:, 0] * table.shape[0]).astype(np.int64), table.shape[0] - 1)
        cols = np.minimum((x[0] * table.shape[1]).astype(np.int64), table.shape[1] - 1)
        weights *= table[rows[:, None], cols[None, :]]

    if not np.any(weights > 0):
        raise ValueError(f"roi selects no pixels of the {h}x{w} statistics image")
    return weights
