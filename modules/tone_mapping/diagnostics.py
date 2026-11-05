"""Tone-mapping curve diagnostic plots.

Single helper with local Matplotlib Figure/FigureCanvasAgg imports — the
global backend is never changed. Curves are sampled through the shared
engine (no duplicated coefficients); labels are display-linear/fitted and
the Durand illustration is explicitly marked frame-dependent. Writes only
to ``platform["output_dir"]``; both flags off produces no IO.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import re

import numpy as np

from modules.tone_mapping.backends import map_luminance
from modules.tone_mapping.registry import ToneMappingSpec

_MODULE_NAME = "tone_mapping"


def save_mapped_output(
    image: np.ndarray,
    *,
    platform: Mapping[str, Any],
    input_name: str,
    module_name: str = _MODULE_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save the actual mapped array as native ``.npy`` under
    ``platform["output_dir"]`` (explicit M3 intermediate-format policy:
    native NPY preserves full precision independently of the final save
    format). ``allow_pickle=False``; names are sanitized.

    When ``metadata`` is supplied the JSON sidecar is serialized first
    (``allow_nan=False``) and NPY+JSON are written as a collision-safe
    exclusive group: both must be new, a metadata-only collision retries
    the next numbered suffix after removing only this call's new NPY,
    and failures roll back only files this call created.
    """
    out_dir = platform.get("output_dir")
    if not out_dir:
        raise ValueError("platform.output_dir is required for mapped-output saves")
    meta_text = (
        json.dumps(dict(metadata), indent=2, sort_keys=True, allow_nan=False)
        if metadata is not None else None
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(input_name).stem)
    module = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_name)
    index = 0
    while True:
        suffix = f"_{index:04d}" if index else ""
        path = out_dir / f"{stem}_{module}{suffix}.npy"
        json_path = path.with_suffix(".json") if meta_text is not None else None
        reserved = False
        try:
            with ExitStack() as rollback:
                with ExitStack() as handles:
                    npy_fh = handles.enter_context(path.open("xb"))
                    rollback.callback(path.unlink, missing_ok=True)
                    json_fh = None
                    if json_path is not None:
                        json_fh = handles.enter_context(json_path.open("x", encoding="utf-8"))
                        rollback.callback(json_path.unlink, missing_ok=True)
                    reserved = True
                    np.save(npy_fh, np.asarray(image), allow_pickle=False)
                    if json_fh is not None and meta_text is not None:
                        json_fh.write(meta_text)
                rollback.pop_all()
        except FileExistsError:
            if reserved:
                raise
            index += 1
            continue
        return path

_LABELS = {
    "aces_fitted": "ACES fitted approximation (display-linear, not full ACES)",
    "hable": "Hable/Uncharted2 display-linear curve",
    "reinhard": "Reinhard display-linear curve",
    "durand": "Durand local operator (frame-dependent illustration)",
}


def plot_curve(
    spec: ToneMappingSpec,
    platform: Mapping[str, Any],
    *,
    reference_luminance: np.ndarray | None = None,
) -> Path | None:
    """Render the effective curve; returns the written path or None."""
    if not spec.enabled:
        return None
    out_dir = platform.get("output_dir")
    if not out_dir:
        return None
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if spec.backend == "fixed":
        probe = np.linspace(0, 4095, 512, dtype=np.uint32)
    elif spec.operator == "durand":
        probe = np.linspace(0.0, 4095.0, 512, dtype=np.float64).reshape(1, -1)
    else:
        probe = np.linspace(0.0, 4095.0, 512, dtype=np.float64)
    xs_codes = np.linspace(0.0, 4095.0, 512, dtype=np.float64)
    mapped = map_luminance(
        probe, input_max=4095.0 if spec.backend != "fixed" else 4095,
        output_max=4095, config=spec,
    )
    fig = Figure(figsize=(5, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(xs_codes, np.asarray(mapped, dtype=np.float64).reshape(-1))
    ax.set_xlabel("input code (display-linear normalized domain)")
    ax.set_ylabel("output code")
    ax.set_title(
        f"{_LABELS.get(spec.operator, spec.operator)}"
        f" [backend={spec.backend}]"
    )
    if spec.operator == "durand":
        ax.text(
            0.02, 0.95, "frame-dependent: adaptive normalization applied",
            transform=ax.transAxes, fontsize=8, va="top",
        )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    base = f"tone_curve_{spec.operator}_{spec.backend}"
    while True:
        suffix = f"_{index:04d}" if index else ""
        out_path = out_dir / f"{base}{suffix}.png"
        try:
            fh = out_path.open("xb")
        except FileExistsError:
            index += 1
            continue
        try:
            with fh:
                fig.savefig(fh, format="png", dpi=120)
        except BaseException:
            out_path.unlink(missing_ok=True)
            raise
        return out_path
