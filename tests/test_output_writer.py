from __future__ import annotations

import json
import numpy as np
import pytest
import tifffile
from pathlib import Path

from util.output_contract import OutputSpec
from util.output_writer import write_output
from util.native_output import output_metadata


def meta(native, spec):
    return output_metadata(native, spec, source_identity="t", tone_mapped=False)


GROUP = (".png", ".jpg", ".tiff", ".yuv", ".json", "_effective.yaml", "_migration.json")


@pytest.mark.parametrize("bit", (8, 16))
def test_png_readback_dtype_and_rgb_order(tmp_path: Path, bit: int) -> None:
    import cv2
    spec = OutputSpec.from_mapping({"format": "png", "bit_depth": bit})
    native = (np.array([[[1, 129, 255]]], np.uint8) if bit == 8
              else np.array([[[1, 1024, 65535]]], np.uint16))
    paths = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="f", effective_config={}, migration_report={})
    assert len(paths) == 4 and all(p.exists() for p in paths)
    back = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)[..., ::-1]
    assert back.dtype == native.dtype
    np.testing.assert_array_equal(back, native)


def test_tiff16_readback_exact(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping({"format": "tiff", "bit_depth": 16})
    native = np.array([[[1, 1024, 65535]]], np.uint16)
    paths = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="f", effective_config={}, migration_report={})
    back = tifffile.imread(paths[0])
    assert back.dtype == np.uint16
    np.testing.assert_array_equal(back, native)


def test_f32_tiff_exact(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping({"format": "tiff", "bit_depth": 32, "transfer": "linear"})
    native = np.array([[[-0.125, 0.5, 2.0]]], np.float32)
    paths = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="f", effective_config={}, migration_report={})
    back = tifffile.imread(paths[0])
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, native)


def test_jpeg_magic_and_decode(tmp_path: Path) -> None:
    import cv2
    spec = OutputSpec.from_mapping({"format": "jpeg", "bit_depth": 8})
    native = np.full((8, 8, 3), 200, np.uint8)
    paths = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="j", effective_config={}, migration_report={})
    raw = paths[0].read_bytes()
    assert raw[:2] == b"\xff\xd8"
    back = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    assert back.shape == (8, 8, 3)


def test_yuv16_be_input_exact_le_bytes(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping({"model": "ycbcr", "format": "yuv", "bit_depth": 16})
    be = np.array([[[1, 1024, 65535]]], ">u2")
    paths = write_output(be, spec, meta(be, spec), output_dir=tmp_path,
                         stem="y", effective_config={}, migration_report={})
    assert paths[0].read_bytes() == np.asarray(be, "<u2").tobytes(order="C")


def test_yuv8_444_bytes_and_rejects_bitdepth_mismatch(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping(
        {"model": "ycbcr", "format": "yuv", "bit_depth": 8})
    native = np.array([[[90, 240, 34]]], np.uint8)
    paths = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="y", effective_config={}, migration_report={})
    assert paths[0].read_bytes() == native.tobytes(order="C")
    spec16 = OutputSpec.from_mapping(
        {"model": "ycbcr", "format": "yuv", "bit_depth": 16})
    with pytest.raises(ValueError):
        write_output(native, spec16, meta(native, spec16), output_dir=tmp_path,
                     stem="y", effective_config={}, migration_report={})


def test_group_names_and_sidecar_only_collision(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping({"format": "png", "bit_depth": 8})
    native = np.array([[[1, 129, 255]]], np.uint8)
    # only a sidecar exists for stem g -> whole group must advance to g_1
    (tmp_path / "g_effective.yaml").write_text("preexisting", encoding="utf-8")
    paths = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="g", effective_config={}, migration_report={})
    stems = {p.name for p in paths}
    assert stems == {
        "g_1.png", "g_1.json", "g_1_effective.yaml", "g_1_migration.json"}
    assert (tmp_path / "g_effective.yaml").read_text() == "preexisting"
    # repeated write still does not overwrite
    again = write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                         stem="g", effective_config={}, migration_report={})
    assert all(p.name.startswith("g_2") for p in again)


def test_cleanup_only_own_files_on_encode_failure(tmp_path: Path, monkeypatch) -> None:
    spec = OutputSpec.from_mapping({"format": "png", "bit_depth": 8})
    native = np.array([[[1, 129, 255]]], np.uint8)
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    import cv2
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))
    with pytest.raises(ValueError):
        write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                     stem="x", effective_config={}, migration_report={})
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_cleanup_partial_after_image_open(tmp_path: Path, monkeypatch) -> None:
    spec = OutputSpec.from_mapping({"format": "tiff", "bit_depth": 16})
    native = np.zeros((2, 2, 3), np.uint16)
    import tifffile as t
    monkeypatch.setattr(t, "imwrite",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        write_output(native, spec, meta(native, spec), output_dir=tmp_path,
                     stem="x", effective_config={}, migration_report={})
    assert not list(tmp_path.iterdir())


def test_write_rejects_bad_native_and_cleans_up(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping({"format": "png", "bit_depth": 8})
    bad = np.zeros((2, 2), np.uint8)
    m = meta(np.zeros((2, 2, 3), np.uint8), spec)
    with pytest.raises(ValueError):
        write_output(bad, spec, m, output_dir=tmp_path, stem="x",
                     effective_config={}, migration_report={})
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        write_output(np.zeros((2, 2, 3), np.uint8), spec, m,
                     output_dir=tmp_path, stem="../x",
                     effective_config={}, migration_report={})


def test_metadata_shape_mismatch_rejected(tmp_path: Path) -> None:
    spec = OutputSpec.from_mapping({"format": "png", "bit_depth": 8})
    native = np.zeros((2, 4, 3), np.uint8)
    m = output_metadata(native, spec, source_identity="t", tone_mapped=False)
    import dataclasses
    m = dataclasses.replace(m, height=9)
    with pytest.raises(ValueError):
        write_output(native, spec, m, output_dir=tmp_path, stem="x",
                     effective_config={}, migration_report={})
    assert not list(tmp_path.iterdir())
