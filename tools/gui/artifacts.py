from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.gui.analysis import analyze_rgb8, analyze_values, native_values, sample_tone_curve
from tools.gui.comparison import comparison_identity, compose_rgb8, difference_statistics
from util.color import encode_transfer, quantize_rgb
from util.native_output import preview_from_native
from util.output_contract import OutputSpec
from util.pipeline_types import FrameMetadata

_STAGE_DOMAINS = {
    "white_balance": ("linear_bayer_codes", "codes"),
    "demosaic": ("camera_linear_rgb", "normalized"),
    "color_correction_matrix": ("scene_linear_rgb", "normalized"),
    "linear_rgb_pre_tone_mapping": ("scene_linear_rgb", "normalized"),
    "tone_mapping_pre": ("display_linear_bayer", "normalized"),
    "tone_mapping_post": ("display_linear_rgb", "normalized"),
    "demosaic_render": ("display_linear_rgb", "normalized"),
    "color_correction_matrix_render": ("display_linear_rgb", "normalized"),
}


def _manifest(directory: str | Path) -> tuple[Path, dict[str, Any]]:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported result artifact manifest")
    return directory, manifest


def _array(directory: Path, filename: Any) -> np.ndarray:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or "\\" in filename:
        raise ValueError("artifact array must be a plain filename")
    path = directory / filename
    if not path.resolve().is_relative_to(directory):
        raise ValueError("artifact array escapes its result directory")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_artifact(
    manifest_dir: str | Path, stage: str = "preview",
) -> tuple[np.ndarray, dict[str, Any]]:
    directory, manifest = _manifest(manifest_dir)
    files = manifest["files"]
    if stage == "native":
        data = _array(directory, files["native"])
        metadata = copy.deepcopy(manifest["native_metadata"])
    elif stage == "preview":
        data = _array(directory, files["preview"])
        if data.ndim != 3 or data.shape[-1] != 3 or data.dtype != np.uint8:
            raise ValueError("artifact preview must be uint8 RGB")
        domain = manifest.get("preview_domain", "display_srgb8")
        metadata = asdict(FrameMetadata(
            width=data.shape[1], height=data.shape[0], cfa_pattern=None,
            storage_bit_depth=8, code_bit_depth=8, signal_domain=domain,
            transfer="srgb" if domain == "display_srgb8" else "visualization",
            source_identity=manifest["native_metadata"]["source_identity"],
            color_model="rgb", color_space="srgb" if domain == "display_srgb8" else None,
            value_units="codes" if domain == "display_srgb8" else "visualization_codes",
            nominal_range=(0, 255), component_ranges=((0, 255),) * 3,
        ))
    else:
        if stage not in _STAGE_DOMAINS:
            raise ValueError(f"no verified analysis domain for captured stage {stage!r}")
        entry = next((v for v in files.get("captures", []) if v["stage"] == stage), None)
        if entry is None:
            raise ValueError(f"{stage} was not captured; request it on the next Process")
        data = _array(directory, entry["path"])
        if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] != 3):
            raise ValueError("captured stage must be a Bayer plane or RGB image")
        domain, units = _STAGE_DOMAINS[stage]
        reference = float(manifest["feedback"]["linear_reference_code_max"]) if stage == "white_balance" else 1.0
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("invalid captured-stage reference white")
        metadata = asdict(FrameMetadata(
            width=data.shape[1], height=data.shape[0], cfa_pattern=None,
            storage_bit_depth=data.dtype.itemsize * 8, code_bit_depth=None,
            signal_domain=domain, transfer="linear",
            source_identity=manifest["native_metadata"]["source_identity"],
            color_model="bayer" if data.ndim == 2 else "rgb",
            value_units=units, nominal_range=(0, reference),
            component_ranges=((0, reference),) * (1 if data.ndim == 2 else 3),
            reference_white=reference, allows_signed_headroom=True,
        ))
    metadata["artifact_id"] = str(directory)
    metadata["stage"] = stage
    metadata["source_fingerprint"] = manifest["source_fingerprint"]
    return data, metadata


def artifact_preview(manifest_dir: str | Path, stage: str = "preview") -> np.ndarray:
    directory, manifest = _manifest(manifest_dir)
    data, metadata = load_artifact(directory, stage)
    if stage == "preview":
        return data
    if stage == "native" and manifest.get("kind") != "raw":
        return preview_from_native(data, OutputSpec.from_mapping(manifest["effective_config"]["output"]))
    reference = float(metadata["reference_white"])
    normalized = np.clip(np.asarray(data, dtype=np.float64) / reference, 0, 1)
    if normalized.ndim == 2:
        normalized = np.repeat(normalized[..., None], 3, axis=2)
    return quantize_rgb(encode_transfer(normalized, "srgb", dtype=np.float64), bit_depth=8)


def _comparison_key(manifest: dict[str, Any], data: np.ndarray, metadata: dict[str, Any], stage: str):
    config = copy.deepcopy(manifest["request_config"])
    if stage not in ("preview", "native"):
        config["scale"] = {"is_enable": False}
    if manifest.get("kind") == "raw":
        config["crop"] = {"is_enable": False}
        config["scale"] = {"is_enable": False}
    domain = json.dumps({
        key: metadata.get(key) for key in (
            "signal_domain", "transfer", "value_units", "code_bit_depth",
            "color_model", "color_space", "matrix", "range_mode", "component_ranges",
        )
    }, sort_keys=True, allow_nan=False)
    return comparison_identity(manifest["source_fingerprint"], config, data.shape, domain=domain)


def view_artifact(directory: str | Path):
    path, manifest = _manifest(directory)
    image, metadata = load_artifact(path, "preview")
    return image, metadata, manifest, _comparison_key(manifest, image, metadata, "preview")


def artifact_pixel(directory: str | Path, x: float, y: float, cache: dict) -> dict[str, Any] | None:
    from tools.gui.analysis import native_pixel
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    key = str(Path(directory).resolve())
    if key not in cache:
        if len(cache) >= 3:
            cache.pop(next(iter(cache)))
        cache[key] = load_artifact(directory, "native")
    values, metadata = cache[key]
    if not 0 <= x < metadata["width"] or not 0 <= y < metadata["height"]:
        return None
    return native_pixel(values, metadata, math.floor(x), math.floor(y))


def analyze_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    directory, manifest = _manifest(payload["manifest_dir"])
    stage = str(payload.get("stage", "preview"))
    statistics_stage = "native" if manifest.get("kind") == "raw" and stage == "preview" else stage
    data, metadata = load_artifact(directory, statistics_stage)
    logical = native_values(data, metadata)
    roi = payload.get("roi")
    if statistics_stage == "preview" and metadata["signal_domain"] == "display_srgb8":
        stats = analyze_rgb8(logical, roi)
    else:
        stats = analyze_values(logical, nominal_ranges=metadata["component_ranges"], roi=roi)
        stats["domain"] = metadata["signal_domain"]
        stats["units"] = metadata["value_units"]
        stats["component_names"] = ["Bayer/code"] if logical.ndim == 2 else (
            ["Y", "Cb", "Cr"] if metadata.get("color_model") == "ycbcr" else ["R", "G", "B"])
    response: dict[str, Any] = {
        "manifest_dir": str(directory), "stage": stage, "roi": roi,
        "statistics_stage": statistics_stage, "metadata": metadata, "stats": stats,
    }
    if manifest.get("kind") == "raw":
        response["note"] = "Native sensor-code statistics; the stretched CFA visualization is not a processed sRGB image."
    else:
        response["curve"] = sample_tone_curve(manifest["effective_config"])
    if payload.get("comparison_dir"):
        other_dir, other = _manifest(payload["comparison_dir"])
        other_data, other_metadata = load_artifact(other_dir, statistics_stage)
        other_logical = native_values(other_data, other_metadata)
        response["difference"] = difference_statistics(
            logical, other_logical, roi=roi,
            identity_a=_comparison_key(manifest, logical, metadata, statistics_stage),
            identity_b=_comparison_key(other, other_logical, other_metadata, statistics_stage),
        )
        response["comparison_dir"] = str(other_dir)
    return response


def render_comparison(payload: dict[str, Any], output_dir: str | Path) -> Path:
    directory_a, manifest_a = _manifest(payload["a_dir"])
    directory_b, manifest_b = _manifest(payload["b_dir"])
    a, metadata_a = load_artifact(directory_a, "preview")
    b, metadata_b = load_artifact(directory_b, "preview")
    image = compose_rgb8(
        a, b,
        identity_a=_comparison_key(manifest_a, a, metadata_a, "preview"),
        identity_b=_comparison_key(manifest_b, b, metadata_b, "preview"),
        mode=payload["mode"], fraction=payload.get("fraction", .5),
        difference_gain=payload.get("difference_gain", 1.0),
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "preview.npy.part"
    with temporary.open("xb") as fh:
        np.save(fh, image, allow_pickle=False)
    temporary.replace(directory / "preview.npy")
    manifest = {
        "schema_version": 1, "preview": "preview.npy", "descriptor": payload,
        "shape": list(image.shape), "domain": metadata_a["signal_domain"],
        "note": "Derived display-code visualization; original native results remain unchanged.",
    }
    temporary = directory / "render.json.part"
    temporary.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "render.json")
    return directory / "render.json"


def write_raw_artifact(
    source: str | Path, *, output_dir: str | Path, config: dict[str, Any],
    source_fingerprint: str, tag: dict[str, Any], decoder: Any = None,
) -> Path:
    """Record decoded input and the existing stretched CFA visualization."""
    from tools.gui.raw_preview import mosaic_preview
    from util.raw_decode import RawFormat, decode_raw

    fmt = RawFormat.from_sensor_info(config["sensor_info"])
    decoded = decode_raw(source, fmt) if decoder is None else decoder.decode(source, fmt)
    preview = mosaic_preview(decoded.pixels, decoded.bayer_pattern)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    for filename, array in (("native.npy", decoded.pixels), ("preview.npy", preview)):
        with (directory / filename).open("xb") as fh:
            np.save(fh, array, allow_pickle=False)
    maximum = float(2**decoded.code_bit_depth - 1)
    metadata = FrameMetadata(
        width=decoded.width, height=decoded.height, cfa_pattern=decoded.bayer_pattern,
        storage_bit_depth=decoded.pixels.dtype.itemsize * 8,
        code_bit_depth=decoded.code_bit_depth, signal_domain="sensor_bayer_codes",
        transfer="sensor_encoded", source_identity=decoded.source_identity,
        color_model="bayer", value_units="codes",
        nominal_range=(0, maximum), component_ranges=((0, maximum),), reference_white=maximum,
    )
    manifest = {
        "schema_version": 1, "kind": "raw", "tag": tag,
        "request_config": config, "effective_config": config,
        "source_fingerprint": source_fingerprint, "preview_domain": "raw_CFA_visualization",
        "files": {"native": "native.npy", "preview": "preview.npy", "captures": []},
        "native_metadata": asdict(metadata), "feedback": {}, "warnings": list(decoded.warnings),
        "timings": [], "migration_report": {},
    }
    temp = directory / "result.json.part"
    temp.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    temp.replace(directory / "result.json")
    return directory / "result.json"
