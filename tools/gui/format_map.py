"""Image-format UI mappings onto BrilliantISP YAML keys."""

from __future__ import annotations

from typing import Any

BAYER_PATTERNS = ("rggb", "grbg", "gbrg", "bggr")

# rggb_start index used by the Image Format window (phase at pixel 0,0).
RGGB_START_TO_PATTERN: dict[int, str] = {
    0: "rggb",
    1: "grbg",
    2: "gbrg",
    3: "bggr",
}
PATTERN_TO_RGGB_START: dict[str, int] = {v: k for k, v in RGGB_START_TO_PATTERN.items()}


def bayer_from_rggb_start(start: int) -> str:
    if start not in RGGB_START_TO_PATTERN:
        raise ValueError(f"Invalid rggb_start {start!r}; expected 0..3")
    return RGGB_START_TO_PATTERN[start]


def rggb_start_from_bayer(pattern: str) -> int:
    key = str(pattern).lower()
    if key not in PATTERN_TO_RGGB_START:
        raise ValueError(f"Unknown bayer_pattern {pattern!r}")
    return PATTERN_TO_RGGB_START[key]


def crop_from_margins(
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> dict[str, Any]:
    """Map L-R-T-B crop margins to YAML ``crop`` geometry."""
    new_width = int(width) - int(left) - int(right)
    new_height = int(height) - int(top) - int(bottom)
    enabled = any(int(v) != 0 for v in (left, right, top, bottom))
    return {
        "crop_x_start": int(left),
        "crop_y_start": int(top),
        "new_width": new_width,
        "new_height": new_height,
        "is_enable": enabled,
    }


def margins_from_crop(
    width: int,
    height: int,
    crop: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    """Inverse of :func:`crop_from_margins` → (L, R, T, B)."""
    if not crop or not bool(crop.get("is_enable")):
        return 0, 0, 0, 0
    left = int(crop.get("crop_x_start", 0) or 0)
    top = int(crop.get("crop_y_start", 0) or 0)
    new_w = int(crop.get("new_width", width) or width)
    new_h = int(crop.get("new_height", height) or height)
    right = int(width) - left - new_w
    bottom = int(height) - top - new_h
    return left, max(0, right), top, max(0, bottom)


def update_crop_from_margins(
    crop: dict[str, Any],
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> dict[str, Any]:
    """Write L-R-T-B into ``crop``. Zero margins disable crop but keep the ROI."""
    geo = crop_from_margins(width, height, left, right, top, bottom)
    if geo["is_enable"]:
        crop.update(geo)
    else:
        crop["is_enable"] = False
    return crop
