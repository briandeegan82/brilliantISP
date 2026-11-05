"""Explicit output writer for native pipeline results.

Consumes the formatted native pixel array, its OutputSpec and FrameMetadata
and writes exclusive new files only — never overwrites prior outputs.
Stems are validated against path traversal. One collision-free base stem is
reserved for the image and all three sidecars so they always share a name.
On failure only files created by this call are removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from util.output_contract import OutputSpec
from util.pipeline_types import FrameMetadata


def _stem_ok(stem: str) -> str:
    if not stem or stem in (".", "..") or "/" in stem or "\\" in stem or stem.startswith("-"):
        raise ValueError(f"output stem {stem!r} is not a safe file basename")
    return stem


def _reserve_stem(directory: Path, stem: str) -> str:
    """Return a collision-free stem shared by the image and sidecars."""
    suffixes = (".png", ".jpg", ".tiff", ".yuv", ".json", "_effective.yaml", "_migration.json")
    index = 0
    candidate = stem
    while True:
        if not any((directory / f"{candidate}{sfx}").exists() for sfx in suffixes):
            return candidate
        index += 1
        candidate = f"{stem}_{index}"


def _expect_native(native: np.ndarray, spec: OutputSpec) -> np.ndarray:
    data = np.asarray(native)
    if data.size == 0:
        raise ValueError("native output must be nonempty")
    if spec.model == "ycbcr":
        # exact bit-depth enforcement BEFORE any endian cast
        if data.dtype.kind != "u" or data.dtype.itemsize * 8 != spec.bit_depth:
            raise ValueError(
                f"yuv native dtype must be unsigned {spec.bit_depth}-bit, "
                f"got {data.dtype}")
        if spec.layout == "422":
            if data.ndim != 2 or data.shape[1] % 4 or data.shape[1] < 4:
                raise ValueError("yuv 4:2:2 requires nonempty Hx(2W) packed data")
        else:
            if data.ndim != 3 or data.shape[-1] != 3:
                raise ValueError("yuv 4:4:4 requires HWC3 native data")
        if spec.bit_depth == 16:
            return np.asarray(data, dtype="<u2")
        if data.dtype.byteorder in (">", "|") and data.dtype.itemsize > 1:
            return np.asarray(data, dtype="=u1")
        return np.asarray(data, dtype="=u1")
    if spec.floating_linear:
        if data.dtype != np.float32 or data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError("float TIFF requires finite float32 HWC3 native pixels")
        if not np.all(np.isfinite(data)):
            raise ValueError("float TIFF native pixels must be finite")
    elif data.dtype.kind != "u" or data.dtype.itemsize * 8 != spec.bit_depth or data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError("native dtype/shape does not match declared output spec")
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_output(
    native: np.ndarray,
    spec: OutputSpec,
    metadata: FrameMetadata,
    *,
    output_dir: Path,
    stem: str,
    effective_config: Mapping[str, Any],
    migration_report: Mapping[str, Any],
) -> list[Path]:
    """Write the formatted output plus sidecars; returns created paths.

    PNG8/16 and JPEG8 go through cv2.imencode (RGB->BGR), TIFF8/16/F32
    through tifffile (photometric='rgb'), YUV via little-endian-16 bytes.
    Exclusive numeric-suffix naming prevents overwrite of prior outputs;
    one shared stem keeps image and sidecars consistent. Sidecar content
    is serialized before any file is touched, and on failure only files
    this call created are removed.
    """
    stem = _stem_ok(stem)
    data = _expect_native(native, spec)
    if int(metadata.height) != data.shape[0]:
        raise ValueError(
            f"metadata height {metadata.height} != actual {data.shape[0]}")
    logical_w = data.shape[1] // 2 if spec.format == "yuv" and spec.layout == "422" else data.shape[1]
    if int(metadata.width) != logical_w:
        raise ValueError(
            f"metadata width {metadata.width} != actual {logical_w}")
    # serialize before any filesystem mutation; invalid input leaves nothing
    meta_text = json.dumps(_jsonable(dict(metadata.__dict__)), indent=2, sort_keys=True)
    yaml_text = yaml.safe_dump(dict(effective_config), sort_keys=False)
    report_text = json.dumps(_jsonable(dict(migration_report)), indent=2, sort_keys=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _reserve_stem(output_dir, stem)
    image_ext = {"png": ".png", "jpeg": ".jpg", "tiff": ".tiff", "yuv": ".yuv"}[spec.format]
    image_path = output_dir / f"{base}{image_ext}"
    meta_path = output_dir / f"{base}.json"
    yaml_path = output_dir / f"{base}_effective.yaml"
    report_path = output_dir / f"{base}_migration.json"
    if spec.format in ("png", "jpeg"):
        import cv2
        # cv2 cannot encode big-endian words; normalize 16-bit to little-endian
        if data.dtype.itemsize > 1 and data.dtype.byteorder == ">":
            data = np.asarray(data, dtype="<u2")
        bgr = np.ascontiguousarray(data[..., ::-1])
        ok, buf = cv2.imencode(image_ext, bgr)
        if not ok:
            raise ValueError(f"cv2 failed to encode {spec.format}")
        image_bytes: bytes | None = buf.tobytes()
    elif spec.format == "yuv":
        image_bytes = np.ascontiguousarray(data).tobytes(order="C")
    else:
        image_bytes = None  # tiff streams via tifffile into an exclusive handle

    written: list[Path] = []
    try:
        if image_bytes is not None:
            with image_path.open("xb") as fh:
                written.append(image_path)
                fh.write(image_bytes)
        else:
            import tifffile
            with image_path.open("xb") as fh:
                written.append(image_path)
                tifffile.imwrite(fh, data, photometric="rgb")
        with meta_path.open("x", encoding="utf-8") as fh:
            written.append(meta_path)
            fh.write(meta_text)
        with yaml_path.open("x", encoding="utf-8") as fh:
            written.append(yaml_path)
            fh.write(yaml_text)
        with report_path.open("x", encoding="utf-8") as fh:
            written.append(report_path)
            fh.write(report_text)
    except Exception:
        for created in written:
            try:
                created.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return written
