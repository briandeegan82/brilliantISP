from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from tools.gui.analysis import native_pixel
from tools.gui.artifacts import analyze_artifact, artifact_pixel, artifact_preview, load_artifact, render_comparison, write_raw_artifact
from util.native_output import output_metadata
from util.output_contract import OutputSpec


def _config() -> dict:
    return {
        "schema_version": 2,
        "sensor_info": {"width": 2, "height": 1, "bayer_pattern": "rggb", "bit_depth": 12},
        "crop": {"is_enable": False}, "scale": {"is_enable": False},
        "tone_mapping": {"is_enable": False},
        "output": {"format": "tiff", "bit_depth": 32, "model": "rgb", "transfer": "linear"},
    }


def _result(directory, values: np.ndarray, *, fingerprint: str = "source") -> None:
    directory.mkdir()
    cfg = _config()
    spec = OutputSpec.from_mapping(cfg["output"])
    metadata = output_metadata(values, spec, source_identity="fixture", tone_mapped=False)
    np.save(directory / "native.npy", values, allow_pickle=False)
    np.save(directory / "preview.npy", np.zeros(values.shape, np.uint8), allow_pickle=False)
    np.save(directory / "linear.npy", values, allow_pickle=False)
    manifest = {
        "schema_version": 1, "request_config": cfg, "effective_config": cfg,
        "source_fingerprint": fingerprint, "native_metadata": asdict(metadata),
        "feedback": {"linear_reference_code_max": 4095},
        "files": {"native": "native.npy", "preview": "preview.npy",
                  "captures": [{"stage": "linear_rgb_pre_tone_mapping", "path": "linear.npy"}]},
    }
    (directory / "result.json").write_text(json.dumps(manifest))


def test_native_and_captured_headroom_are_not_measured_from_preview(tmp_path) -> None:
    values = np.array([[[-.25, .5, 2], [0, 1, 4]]], np.float32)
    directory = tmp_path / "result"
    _result(directory, values)
    native = analyze_artifact({"manifest_dir": str(directory), "stage": "native"})
    assert native["stats"]["components"][0]["below_nominal"] == 1
    assert native["stats"]["components"][2]["above_nominal"] == 2
    assert [c["mean"] for c in native["stats"]["components"]] == [-.125, .75, 3]
    capture = analyze_artifact({
        "manifest_dir": str(directory), "stage": "linear_rgb_pre_tone_mapping",
        "roi": [0, 0, 1, 1],
    })
    assert [c["mean"] for c in capture["stats"]["components"]] == [-.25, .5, 2]
    assert capture["metadata"]["transfer"] == "linear"
    view = artifact_preview(directory, "linear_rgb_pre_tone_mapping")
    assert view[0, 0].tolist() == [0, 188, 255]
    original, meta = load_artifact(directory, "native")
    assert native_pixel(original, meta, 0, 0)["values"] == [-.25, .5, 2]


def test_raw_visualization_statistics_use_sensor_codes_not_false_color(tmp_path) -> None:
    source = tmp_path / "source.pgm"
    raw = np.arange(48, dtype=np.uint16).reshape(6, 8) * 10
    contents = b"P5 8 6 4095\n" + raw.astype(">u2").tobytes()
    source.write_bytes(contents)
    cfg = _config()
    cfg["sensor_info"].update({"width": 8, "height": 6, "source_format": "auto"})
    target = tmp_path / "raw_artifact"
    write_raw_artifact(source, output_dir=target, config=cfg,
                       source_fingerprint=hashlib.sha256(contents).hexdigest(), tag={})
    values, metadata = load_artifact(target, "native")
    np.testing.assert_array_equal(values, raw)
    assert metadata["cfa_pattern"] == "rggb"
    assert native_pixel(values, metadata, 2, 3)["values"] == [260]
    result = analyze_artifact({"manifest_dir": str(target), "stage": "preview", "roi": [1, 2, 3, 4]})
    assert result["statistics_stage"] == "native"
    assert result["stats"]["components"][0]["mean"] == 215
    assert result["stats"]["pixel_count"] == 4
    assert result["stats"]["domain"] == "sensor_bayer_codes"
    assert source.read_bytes() == contents


def test_artifact_comparison_rejects_unregistered_source(tmp_path) -> None:
    values = np.ones((1, 2, 3), np.float32)
    a, b = tmp_path / "a", tmp_path / "b"
    _result(a, values)
    _result(b, values, fingerprint="other")
    with pytest.raises(ValueError, match="Different source"):
        analyze_artifact({"manifest_dir": str(a), "stage": "native", "comparison_dir": str(b)})


def test_artifact_path_traversal_and_unknown_capture_fail(tmp_path) -> None:
    directory = tmp_path / "result"
    _result(directory, np.ones((1, 2, 3), np.float32))
    with pytest.raises(ValueError, match="no verified analysis domain"):
        load_artifact(directory, "unverified_stage")
    path = directory / "result.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["native"] = "../outside.npy"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="plain filename"):
        load_artifact(directory, "native")


def test_rendered_comparison_preserves_native_sources(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _result(a, np.zeros((1, 2, 3), np.float32))
    _result(b, np.ones((1, 2, 3), np.float32))
    np.save(a / "preview.npy", np.array([[[0, 10, 255], [100, 0, 12]]], np.uint8))
    np.save(b / "preview.npy", np.array([[[2, 4, 254], [99, 255, 12]]], np.uint8))
    before = [(path, path.read_bytes()) for directory in (a, b) for path in directory.iterdir()]
    payload = {"a_dir": str(a), "b_dir": str(b), "mode": "difference", "difference_gain": 2}
    result = render_comparison(payload, tmp_path / "difference")
    manifest = json.loads(result.read_text())
    np.testing.assert_array_equal(np.load(result.parent / manifest["preview"]),
                                  [[[4, 12, 2], [2, 255, 0]]])
    result = render_comparison(dict(payload, mode="wipe", fraction=.5), tmp_path / "wipe")
    np.testing.assert_array_equal(np.load(result.parent / "preview.npy"),
                                  [[[0, 10, 255], [99, 255, 12]]])
    assert all(path.read_bytes() == contents for path, contents in before)
    with pytest.raises(FileExistsError):
        render_comparison(payload, tmp_path / "difference")


def test_artifact_picker_uses_full_resolution_native_coordinates(tmp_path) -> None:
    directory = tmp_path / "result"
    _result(directory, np.array([[[-.25, .5, 2], [0, 1, 4]]], np.float32))
    cache = {}
    point = artifact_pixel(directory, 1.2, .2, cache)
    assert point is not None
    assert (point["x"], point["y"]) == (1, 0)
    assert point["values"] == [0, 1, 4]
    assert point["domain"] == "scene_linear_rgb"
    assert artifact_pixel(directory, -.1, .2, cache) is None
    assert artifact_pixel(directory, 2, .2, cache) is None
    assert artifact_pixel(directory, .2, 1, cache) is None
    assert artifact_pixel(directory, float("nan"), .2, cache) is None
    for index in range(4):
        path = tmp_path / str(index)
        _result(path, np.zeros((1, 2, 3), np.float32))
        artifact_pixel(path, .2, .2, cache)
    assert len(cache) == 3
