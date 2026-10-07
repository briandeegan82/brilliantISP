#!/usr/bin/env python3
"""Export a brilliantISP camera YAML into a boltISP-compatible YAML.

Usage:
  python tools/export_boltisp_yaml.py \
      --input config/AD_cam.yml \
      --output /home/brian/boltISP/config/AD_cam_from_brilliant.yml

LSC grid generation
-------------------
boltISP uses bilinear-interpolated gain tables (r_table, gr_table, gb_table,
b_table) while brilliantISP evaluates a radial polynomial at runtime.  This
script generates the tables by evaluating:

    gain(r) = max(1.0, 1 + k1*r² + k2*r⁴)

at each node of a uniform *grid_width × grid_height* grid over the sensor.
r is normalised so that r=0 at the image centre and r=1 at the corners.

The generated tables are written as inline YAML float lists (to the precision
boltISP expects: three decimal places is plenty given GainQ5.11 resolution of
~0.00049).  The keys ``lens_shading_correction.grid_width`` and
``lens_shading_correction.grid_height`` are also emitted so boltISP knows the
grid dimensions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

TONE_MAPPER_TO_OPERATOR = {
    "durand": "durand",
    "aces": "aces",
    "hable": "hable",
    "reinhard_integer": "reinhard",
    "aces_integer": "aces",
    "hable_integer": "hable",
}


def _to_inline_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def _to_inline_list(values: list[Any]) -> str:
    return "[" + ", ".join(_to_inline_scalar(v) for v in values) + "]"


def _lsc_radial_grid(
    k1: float,
    k2: float,
    grid_width: int,
    grid_height: int,
) -> list[float]:
    """Evaluate brilliantISP's radial polynomial LSC over a uniform grid.

    Returns a flat row-major list of ``grid_width * grid_height`` gain values.
    r is normalised so that r=1 at the image corners (half-diagonal = r_max).
    Gains are clamped to [1.0, 15.99] to stay within boltISP's Q5.11 range.
    """
    # Half-extents in normalised [0,1] image coordinates; centre is (0.5, 0.5)
    r_max = math.sqrt(0.5**2 + 0.5**2)  # distance from centre to corner ≈ 0.7071
    gains: list[float] = []
    for gy in range(grid_height):
        y_norm = gy / max(grid_height - 1, 1)  # [0, 1] top to bottom
        dy = y_norm - 0.5
        for gx in range(grid_width):
            x_norm = gx / max(grid_width - 1, 1)  # [0, 1] left to right
            dx = x_norm - 0.5
            r = math.sqrt(dx**2 + dy**2) / r_max  # 0 at centre, 1 at corners
            gain = 1.0 + k1 * r**2 + k2 * r**4
            gain = max(1.0, min(gain, 15.99))  # clamp to Q5.11-safe range
            gains.append(round(gain, 5))
    return gains


def _apply_lsc_table_generation(
    cfg: dict[str, Any],
    grid_width: int,
    grid_height: int,
) -> None:
    """Convert brilliantISP k1/k2 LSC params to boltISP gain grid tables.

    Reads ``lens_shading_correction.{channel}_k1`` / ``{channel}_k2`` for
    channels r, gr, gb, b.  Writes ``r_table``, ``gr_table``, ``gb_table``,
    ``b_table`` plus ``grid_width`` / ``grid_height`` into the same section.

    If the LSC section or k1/k2 keys are absent the function returns without
    modifying the config.
    """
    lsc = cfg.get("lens_shading_correction")
    if not isinstance(lsc, dict):
        return

    channels = [
        ("r", "r_k1", "r_k2", "r_table"),
        ("gr", "gr_k1", "gr_k2", "gr_table"),
        ("gb", "gb_k1", "gb_k2", "gb_table"),
        ("b", "b_k1", "b_k2", "b_table"),
    ]

    any_generated = False
    for _ch, k1_key, k2_key, table_key in channels:
        k1 = lsc.get(k1_key)
        k2 = lsc.get(k2_key)
        if not isinstance(k1, (int, float)) or not isinstance(k2, (int, float)):
            continue
        lsc[table_key] = _lsc_radial_grid(float(k1), float(k2), grid_width, grid_height)
        any_generated = True

    if any_generated:
        lsc["grid_width"] = grid_width
        lsc["grid_height"] = grid_height
        # Remove k1/k2 keys so boltISP's flat parser does not get confused by
        # unknown keys (it silently ignores them, but keep the file clean).
        for _ch, k1_key, k2_key, _table_key in channels:
            lsc.pop(k1_key, None)
            lsc.pop(k2_key, None)


def _apply_tonemap_translation(cfg: dict[str, Any]) -> None:
    tone_mapping = cfg.get("tone_mapping")
    if not isinstance(tone_mapping, dict):
        return
    tone_mapper = tone_mapping.get("tone_mapper")
    if not isinstance(tone_mapper, str):
        return
    mapped = TONE_MAPPER_TO_OPERATOR.get(tone_mapper.strip().lower())
    if mapped:
        tone_mapping["operator"] = mapped
    tm = tone_mapper.strip().lower()

    # boltISP consumes Hable parameters under hdr_tone_mapping.*.
    # For brilliant integer/float hable sections, copy nearest equivalents.
    if mapped == "hable":
        hdr_tm = cfg.get("hdr_tone_mapping")
        if not isinstance(hdr_tm, dict):
            hdr_tm = {}
            cfg["hdr_tone_mapping"] = hdr_tm
        hdr_tm["operator"] = "hable"

        if tm == "hable_integer":
            hable_integer = cfg.get("hable_integer")
            if isinstance(hable_integer, dict):
                exp = hable_integer.get("exposure_bias")
                if isinstance(exp, (int, float)):
                    hdr_tm["hable_exposure_bias"] = float(exp)
                wp = hable_integer.get("white_point")
                if isinstance(wp, (int, float)):
                    hdr_tm["hable_white_point"] = float(wp)
                use_norm = hable_integer.get("use_normalization")
                if isinstance(use_norm, bool):
                    # boltISP runs tiled; per-image normalization from brilliantISP
                    # would require global frame stats to avoid tile seams.
                    hdr_tm["hable_use_normalization"] = False
                norm_out = hable_integer.get("normalize_output")
                if isinstance(norm_out, bool):
                    hdr_tm["hable_normalize_output"] = norm_out
                hdr_scale = hable_integer.get("hdr_scale")
                if isinstance(hdr_scale, (int, float)):
                    hdr_tm["hable_hdr_scale"] = float(hdr_scale)
        elif tm == "hable":
            hable = cfg.get("hable")
            if isinstance(hable, dict):
                exp = hable.get("exposure_bias")
                if isinstance(exp, (int, float)):
                    hdr_tm["hable_exposure_bias"] = float(exp)
                wp = hable.get("white_point")
                if isinstance(wp, (int, float)):
                    hdr_tm["hable_white_point"] = float(wp)


def _render_node(node: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_render_node(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{pad}{key}: {_to_inline_list(value)}")
            else:
                lines.append(f"{pad}{key}: {_to_inline_scalar(value)}")
        return lines
    lines.append(f"{pad}{_to_inline_scalar(node)}")
    return lines


def export_bolt_yaml(
    input_path: Path,
    output_path: Path,
    lsc_grid_width: int = 17,
    lsc_grid_height: int = 17,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at YAML root: {input_path}")

    _apply_tonemap_translation(data)
    _apply_lsc_table_generation(data, lsc_grid_width, lsc_grid_height)

    rendered = "\n".join(_render_node(data)) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert brilliantISP YAML into boltISP-compatible YAML.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to brilliantISP YAML (for example config/AD_cam.yml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for boltISP-compatible YAML.",
    )
    parser.add_argument(
        "--lsc-grid-width",
        type=int,
        default=17,
        help="Number of LSC grid nodes horizontally (default: 17, matching boltISP default).",
    )
    parser.add_argument(
        "--lsc-grid-height",
        type=int,
        default=17,
        help="Number of LSC grid nodes vertically (default: 17, matching boltISP default).",
    )
    args = parser.parse_args()
    export_bolt_yaml(
        args.input,
        args.output,
        lsc_grid_width=args.lsc_grid_width,
        lsc_grid_height=args.lsc_grid_height,
    )


if __name__ == "__main__":
    main()
