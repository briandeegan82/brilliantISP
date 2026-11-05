from __future__ import annotations

import copy
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml
from typing import Any, Literal, cast


from brilliant_isp import BrilliantISP
from tools.gui.raw_preview import load_bayer_array, preview_from_config
from tools.gui.session import GuiSession
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingOptions, ProcessingState
from util.raw_decode import (
    DecodedRaw,
    RawFormat,
    RawFormatError,
    decode_raw,
    flip_bayer_pattern,
    shift_bayer_pattern,
)

REPO = Path(__file__).resolve().parents[1]
SVS = REPO / "config" / "SVS_cam.yml"


def _cfg() -> dict:
    cfg = yaml.safe_load(SVS.read_text())
    cfg["sensor_info"].update(
        width=8, height=6, bit_depth=12, hdr_bit_depth=12,
        storage_format="uint16", endian_type="ieee-le",
        horizontal_flip=False, vertical_flip=False,
    )
    for sec in (
        "dead_pixel_correction", "bayer_noise_reduction",
        "lens_shading_correction", "auto_white_balance", "auto_exposure",
        "black_level_correction", "companding", "oecf", "ldci", "sharpen",
        "2d_noise_reduction", "scale", "crop", "tone_mapping",
        "digital_gain", "color_correction_matrix",
    ):
        if isinstance(cfg.get(sec), dict):
            cfg[sec]["is_enable"] = False
    cfg["platform"].update(render_3a=False, plot_histograms=False,
                           debug_enabled=False, short_output_names=True,
                           disable_progress_bar=True)
    cfg["yuv_conversion_format"]["is_enable"] = False
    return cfg


def _img() -> np.ndarray:
    return (np.arange(48, dtype=np.uint16).reshape(6, 8) * 80 % 4096)


def _write_raw(path: Path, img: np.ndarray, order: str = "little") -> None:
    img.astype("<u2" if order == "little" else ">u2").tofile(path)


def test_pipeline_decode_matches_decoder(tmp_path: Path) -> None:
    cfg = _cfg()
    raw = tmp_path / "f.raw"
    _write_raw(raw, _img())
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    dec = decode_raw(raw, RawFormat.from_sensor_info(cfg["sensor_info"]))
    np.testing.assert_array_equal(isp.raw, dec.pixels)
    if isp.decoded_raw is not dec:
        assert isp.decoded_raw is not None
        assert np.array_equal(isp.decoded_raw.pixels, dec.pixels)


def test_no_legacy_fallback(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg()
    raw = tmp_path / "f.raw"
    _write_raw(raw, _img())
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())

    def boom(*a, **k):
        raise AssertionError("legacy loader must not be called")

    monkeypatch.setattr(isp, "_load_raw_legacy", boom)
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert isp.raw is not None


def test_process_frame_file_uses_decoder(tmp_path: Path) -> None:
    cfg = _cfg()
    raw = tmp_path / "f.raw"
    _write_raw(raw, _img())
    result = process_frame(raw, cfg)
    dec = decode_raw(raw, RawFormat.from_sensor_info(cfg["sensor_info"]))
    cfg2 = copy.deepcopy(cfg)
    result2 = process_frame(_img(), cfg2)
    np.testing.assert_array_equal(result.preview_rgb8, result2.preview_rgb8)


def test_pgm_pipeline(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"]["endian_type"] = "ieee-le"
    img = _img()
    p = tmp_path / "f.pgm"
    header = b"P5 8 6 4095\n"
    p.write_bytes(header + img.astype(">u2").tobytes())
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.pgm"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)


def test_byte_order_override_raw_only(tmp_path: Path) -> None:
    cfg = _cfg()
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img, order="big")
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw(byte_order="big")
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)
    assert isp.sensor_info.get("endian_type") == "ieee-be"


def test_header_and_stride_via_sensor(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"]["header_bytes"] = 4
    cfg["sensor_info"]["row_stride_bytes"] = 20
    img = _img()
    raw = tmp_path / "f.raw"
    rows = b"".join(
        img[y].astype("<u2").tobytes() + b"\xAA" * 4 for y in range(6)
    )
    raw.write_bytes(b"HEAD" + rows)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)


def test_wrong_size_rejected_no_inference(tmp_path: Path) -> None:
    cfg = _cfg()
    raw = tmp_path / "f.raw"
    raw.write_bytes(b"\x00" * 50)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    with pytest.raises(Exception, match="exactly"):
        isp.load_raw()


def test_embedded_rows_loaded_geometry(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"].update(height=8, embedded_rows_top=2,
                              embedded_rows_bottom=1)
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert isp.raw.shape == (5, 8)
    assert (isp.sensor_info["width"], isp.sensor_info["height"]) == (8, 5)
    isp.run_pipeline_up_to_wb()
    assert isp.current_frame is not None
    assert isp.current_frame.shape[:2] == (5, 8)


def test_flips_effective_pattern(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"]["horizontal_flip"] = True
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, np.fliplr(img))
    assert isp.sensor_info["bayer_pattern"] == flip_bayer_pattern(
        cfg["sensor_info"]["bayer_pattern"], 8, 6, True, False
    )


def test_repeated_load_resets_decoded_state(tmp_path: Path) -> None:
    cfg = _cfg()
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    isp.run_pipeline_up_to_wb()
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert isp.pass_index == 0
    assert isp.stage_events == []
    np.testing.assert_array_equal(isp.raw, img)


def test_preview_matches_pipeline(tmp_path: Path) -> None:
    cfg = _cfg()
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    bayer = load_bayer_array(
        raw,
        width=8, height=6, endian="ieee-le", bit_depth=12,
        data_format="uint16", bayer_pattern=cfg["sensor_info"]["bayer_pattern"],
    )
    dec = decode_raw(raw, RawFormat.from_sensor_info(cfg["sensor_info"]))
    np.testing.assert_array_equal(bayer, dec.pixels)
    preview = preview_from_config(raw, cfg)
    assert preview.shape == (6, 8, 3)


def test_session_preserves_source_and_reopen(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    src = tmp_path / "frame.raw"
    _write_raw(src, _img())
    dest = session.copy_raw_from(src)
    assert dest.name == "input.raw"
    assert dest.read_bytes() == src.read_bytes()
    reopened, cfg = GuiSession.open_existing(session.paths.directory)
    assert reopened.paths.input_raw.name == "input.raw"
    assert reopened.paths.input_raw.read_bytes() == src.read_bytes()
    assert cfg["platform"]["filename"] == "input.raw"
    assert src.read_bytes() == dest.read_bytes()


def test_session_legacy_input_raw_no_field(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.ui_state.pop("input_filename", None)
    session.write_ui_state()
    reopened, _ = GuiSession.open_existing(session.paths.directory)
    assert reopened.paths.input_raw.name == "input.raw"


def test_session_unsafe_filename_rejected(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.ui_state["input_filename"] = "../escape.raw"
    session.write_ui_state()
    with pytest.raises(ValueError, match="unsafe"):
        GuiSession.open_existing(session.paths.directory)


def test_gui_feedback_keeps_source_metadata() -> None:
    from tools.gui.yaml_compat import merge_pipeline_feedback

    dst = _cfg()
    src = copy.deepcopy(dst)
    src["sensor_info"]["width"] = 4
    src["sensor_info"]["bayer_pattern"] = "bggr"
    src["sensor_info"]["storage_format"] = "packed24"
    src["sensor_info"]["bit_depth"] = 24
    src["white_balance"]["r_gain"] = 1.7
    before = copy.deepcopy(dst["sensor_info"])
    merge_pipeline_feedback(dst, src)
    assert dst["sensor_info"] == before
    assert dst["white_balance"]["r_gain"] == 1.7


def test_packed24_pipeline(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"].update(bit_depth=24, storage_format="packed24")
    codes = np.arange(48, dtype=np.uint32).reshape(6, 8) * 5000 % (1 << 24)
    raw = tmp_path / "f.raw"
    raw.write_bytes(b"".join(
        int(c).to_bytes(3, "little") for c in codes.flatten()
    ))
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, codes)


def test_word_pair_pipeline(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"].update(bit_depth=32, storage_format="word_pair32",
                              word_order="low_first", endian_type="ieee-le")
    codes = np.arange(48, dtype=np.uint32).reshape(6, 8)
    raw = tmp_path / "f.raw"
    raw.write_bytes(b"".join(
        struct.pack("<2H", c & 0xFFFF, c >> 16) for c in codes.flatten()
    ))
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, codes)


def test_load_method_rejected() -> None:
    isp = BrilliantISP("", _cfg(), outFileName="",
                       options=ProcessingOptions())
    with pytest.raises(ValueError, match="storage_format"):
        isp.execute(img_path="f.raw", load_method="3byte")


def test_decode_warnings_surface_once(tmp_path: Path) -> None:
    cfg = _cfg()
    del cfg["sensor_info"]["storage_format"]
    del cfg["sensor_info"]["data_format"]
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    dec_warns = [w for w in isp.config_notices if "defaulted" in w]
    assert len(dec_warns) == 1


def test_container_byte_order_not_rewritten(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"]["endian_type"] = "ieee-le"
    img = _img()
    p = tmp_path / "f.pgm"
    p.write_bytes(b"P5 8 6 4095\n" + img.astype(">u2").tobytes())
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.pgm"
    isp.load_raw(byte_order="big")
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)
    assert isp.sensor_info.get("endian_type") == "ieee-le"


def test_failed_load_clears_stale_state(tmp_path: Path) -> None:
    cfg = _cfg()
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert isp.raw is not None
    bad = tmp_path / "bad.raw"
    bad.write_bytes(b"\x00" * 10)
    isp.raw_file = "bad.raw"
    with pytest.raises(Exception):
        isp.load_raw()
    assert isp.raw is None
    assert isp.current_frame is None
    assert isp.decoded_raw is None
    with pytest.raises(RuntimeError, match="RAW image"):
        isp.run_pipeline()


def test_decode_warnings_reset_per_load(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    cfg = _cfg()
    cfg["sensor_info"].update(width=4, height=4, source_format="rawpy",
                              bayer_pattern="bggr")

    class _Stub:
        raw_image_visible = np.zeros((4, 4), np.uint16)
        raw_colors_visible = np.array([[0, 1], [1, 2]], np.uint8)
        raw_pattern = np.zeros((2, 2), np.uint8)
        color_desc = b"RGBG"
        white_level = 4095

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    stub = _Stub()
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: stub))
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.dng"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert any("actual" in w for w in isp.config_notices)

    img = _img()[:4, :4]
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp.sensor_info["source_format"] = "auto"
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert not any("actual" in w for w in isp.config_notices)
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    assert isp.config_notices == list(isp._base_config_notices)


def test_update_sensor_info_new_pattern_survives() -> None:
    isp = BrilliantISP("", _cfg(), outFileName="",
                       options=ProcessingOptions())
    isp.update_sensor_info({
        "width": 8, "height": 6, "bit_depth": 12, "bayer_pattern": "bggr",
    })
    assert isp._frame_reset_baseline["bayer_pattern"] == "bggr"
    assert isp._loaded_frame_geometry is None
    isp._reset_transient_frame_state()
    assert isp.sensor_info is not None
    assert isp.sensor_info["bayer_pattern"] == "bggr"


def test_crop_orig_size_exact(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["crop"]["is_enable"] = True
    cfg["crop"].update(crop_x_start=0, crop_y_start=0, new_width=4, new_height=4)
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw = img
    isp.in_file = isp.out_file = "t"
    isp.run_pipeline_up_to_wb()
    assert isp.sensor_info is not None
    assert isp.sensor_info.get("orig_size") == "(8, 6)"


def test_tiff16be_pipeline(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    cfg = _cfg()
    cfg["sensor_info"].update(bit_depth=16, source_format="tiff")
    img = np.arange(48, dtype=">u2").reshape(6, 8)
    p = tmp_path / "f.tiff"
    tifffile.imwrite(p, img)
    result = process_frame(p, cfg)
    dec = decode_raw(p, RawFormat.from_sensor_info(cfg["sensor_info"]))
    arr_result = process_frame(dec.pixels.astype(np.uint16), cfg)
    np.testing.assert_array_equal(result.preview_rgb8, arr_result.preview_rgb8)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.tiff"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)
    np.testing.assert_array_equal(isp.raw, dec.pixels)
    preview = preview_from_config(p, cfg)
    assert preview.shape == (6, 8, 3)


@pytest.mark.parametrize("order", ["little", "big"])
def test_packed24_pipeline_both_endian(tmp_path: Path, order: Literal["little", "big"]) -> None:
    cfg = _cfg()
    cfg["sensor_info"].update(bit_depth=24, storage_format="packed24",
                              endian_type=f"ieee-{'le' if order == 'little' else 'be'}")
    codes = np.arange(48, dtype=np.uint32).reshape(6, 8) * 5000 % (1 << 24)
    raw = tmp_path / "f.raw"
    raw.write_bytes(b"".join(int(c).to_bytes(3, order) for c in codes.flatten()))
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, codes)


@pytest.mark.parametrize("order", ["little", "big"])
def test_word_pair32_both_endian_distinguishable(tmp_path: Path, order: str) -> None:
    cfg = _cfg()
    cfg["sensor_info"].update(bit_depth=32, storage_format="word_pair32",
                              word_order="low_first",
                              endian_type=f"ieee-{'le' if order == 'little' else 'be'}")
    codes = (np.arange(48, dtype=np.uint32).reshape(6, 8) * 0x01010101 + 0x12345678) % (1 << 32)
    raw = tmp_path / "f.raw"
    pack = "<2H" if order == "little" else ">2H"
    raw.write_bytes(b"".join(
        struct.pack(pack, int(c) & 0xFFFF, int(c) >> 16) for c in codes.flatten()
    ))
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, codes)


def test_flip_odd_crop_full_profile_stability(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["sensor_info"]["horizontal_flip"] = True
    cfg["crop"]["is_enable"] = True
    cfg["crop"].update(crop_x_start=1, crop_y_start=1, new_width=4, new_height=4)
    img = _img()
    raw = tmp_path / "f.raw"
    _write_raw(raw, img)
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "f.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    flipped = np.fliplr(img)
    expected_phase = shift_bayer_pattern(
        flip_bayer_pattern("gbrg", 8, 6, True, False), 1, 1
    )
    isp.run_pipeline(visualize_output=False)
    assert isp.current_frame is not None
    wb1 = isp.current_frame.copy()
    assert wb1.shape == (4, 4)
    assert isp.sensor_info["bayer_pattern"] == expected_phase
    assert isp.decoded_raw is not None
    np.testing.assert_array_equal(isp.decoded_raw.pixels[1:5, 1:5],
                                  flipped[1:5, 1:5])
    isp.run_pipeline_up_to_wb()
    np.testing.assert_array_equal(isp.current_frame, wb1)
    isp.run_pipeline(visualize_output=False)
    np.testing.assert_array_equal(isp.current_frame, wb1)
    assert (isp.sensor_info["width"], isp.sensor_info["height"]) == (4, 4)


def test_array_endian_variants_equal(tmp_path: Path) -> None:
    cfg = _cfg()
    native = _img()
    result_native = process_frame(native, cfg)
    be = native.astype(">u2")
    result_be = process_frame(be, cfg)
    np.testing.assert_array_equal(result_be.preview_rgb8,
                                  result_native.preview_rgb8)
    assert be.dtype == np.dtype(">u2")


def test_preview_rejects_invalid_like_factory(tmp_path: Path) -> None:
    from tools.gui.raw_preview import preview_from_config, load_bayer_array

    cfg = _cfg()
    raw = tmp_path / "f.raw"
    _write_raw(raw, _img())
    for mutate in (
        {"width": 8.9},
        {"horizontal_flip": "false"},
        {"storage_format": None},
        {"storage_format": ""},
        {"header_bytes": -1},
        {"header_bytes": 2.5},
    ):
        bad = copy.deepcopy(cfg)
        bad["sensor_info"].update(mutate)
        with pytest.raises((RawFormatError, ValueError)):
            preview_from_config(raw, bad)
    with pytest.raises(RawFormatError):
        load_bayer_array(raw, width=cast(Any, 8.9), height=6)
    with pytest.raises(RawFormatError):
        load_bayer_array(raw, width=8, height=6, storage_format="")
    with pytest.raises(RawFormatError):
        load_bayer_array(raw, width=8, height=6, horizontal_flip=cast(Any, "false"))


def test_computed_stats_invalid_layout() -> None:
    from tools.gui.computed import compute_format_stats

    base = _cfg()
    for mutate in (
        {"storage_format": None},
        {"header_bytes": -2},
        {"row_stride_bytes": 0},
        {"row_stride_bytes": 4.5},
        {"width": 8.9},
        {"embedded_rows_top": -1},
        {"header_bytes": float("inf")},
    ):
        cfg = copy.deepcopy(base)
        cfg["sensor_info"].update(mutate)
        stats = compute_format_stats(cfg)
        assert stats["estimated_raw_size"] == 0
        assert stats["format_error"]


def test_computed_stats_valid_exact() -> None:
    from tools.gui.computed import compute_format_stats

    cfg = _cfg()
    cfg["sensor_info"].update(header_bytes=8, row_stride_bytes=20,
                              embedded_rows_top=2)
    stats = compute_format_stats(cfg)
    assert stats["format_error"] == ""
    assert stats["estimated_raw_size"] == 8 + 6 * 20
    assert stats["effective_height"] == 4


def test_session_atomic_copy_preserves_state(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    good = tmp_path / "good.pgm"
    good.write_bytes(b"P5 2 2 255\n" + bytes([1, 2, 3, 4]))
    session.copy_raw_from(good)
    good_bytes = session.paths.input_raw.read_bytes()
    bad = tmp_path / "bad.pgm"
    bad.write_bytes(b"not a pgm at all")
    before_ui = dict(session.ui_state)
    with pytest.raises(Exception):
        session.copy_raw_from(bad)
    assert session.paths.input_raw.read_bytes() == good_bytes
    assert session.ui_state == before_ui


def test_session_staged_suffixes(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    for suffix in (".pgm", ".tiff", ".dng"):
        src = tmp_path / f"src{suffix}"
        if suffix == ".pgm":
            src.write_bytes(b"P5 2 2 255\n" + bytes([1, 2, 3, 4]))
        else:
            src.write_bytes(b"\x00" * 32)
        dest = session.copy_raw_from(src)
        assert dest.name == f"input{suffix}"
        reopened, cfg = GuiSession.open_existing(session.paths.directory)
        assert reopened.paths.input_raw.name == f"input{suffix}"
        assert cfg["platform"]["filename"] == f"input{suffix}"
        reloaded = reopened.reload_raw()
        assert reloaded.read_bytes() == src.read_bytes()


def test_session_null_filename_rejected(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    for bad in (None, "", "../x.raw", "a/b", "a\\b", ".", "..", "x\x00y", 5):
        session.ui_state["input_filename"] = bad
        session.write_ui_state()
        with pytest.raises(ValueError, match="unsafe"):
            GuiSession.open_existing(session.paths.directory)
        del session.ui_state["input_filename"]
        session.write_ui_state()


def test_source_format_container_layout_fields_rejected() -> None:
    with pytest.raises(RawFormatError):
        RawFormat(width=2, height=2, bayer_pattern="rggb", bit_depth=8,
                  storage_format="uint8", source_format="pgm", header_bytes=4)
    with pytest.raises(RawFormatError):
        RawFormat(width=2, height=2, bayer_pattern="rggb", bit_depth=8,
                  storage_format="uint8", source_format="tiff",
                  row_stride_bytes=8)
    with pytest.raises(RawFormatError):
        RawFormat(width=2, height=2, bayer_pattern="rggb", bit_depth=12,
                  storage_format="uint16", source_format="rawpy",
                  data_alignment="MSB")


def test_direct_constructor_rejects_null_storage() -> None:
    with pytest.raises(RawFormatError, match="storage_format"):
        RawFormat(width=2, height=2, bayer_pattern="rggb", bit_depth=8,
                  storage_format=cast(Any, None))
    with pytest.raises(RawFormatError, match="storage_format"):
        RawFormat(width=2, height=2, bayer_pattern="rggb", bit_depth=8,
                  storage_format="")
    with pytest.raises(RawFormatError, match="bit_depth"):
        RawFormat(width=2, height=2, bayer_pattern="rggb", bit_depth=cast(Any, "12"),
                  storage_format=cast(Any, None))


def test_resolve_source_format_shared() -> None:
    from util.raw_decode import resolve_source_format
    assert resolve_source_format("a.pgm") == "pgm"
    assert resolve_source_format("a.raw") == "raw"
    assert resolve_source_format("a.raw", "pgm") == "pgm"
    with pytest.raises(RawFormatError):
        resolve_source_format("a.png")


def test_endian_override_not_sticky(tmp_path: Path) -> None:
    cfg = _cfg()
    be = tmp_path / "be.raw"
    le = tmp_path / "le.raw"
    img = _img()
    _write_raw(be, img, order="big")
    _write_raw(le, img, order="little")
    caller_snapshot = copy.deepcopy(cfg["sensor_info"])
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw_file = "be.raw"
    isp.load_raw(byte_order="big")
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)
    assert isp.sensor_info.get("endian_type") == "ieee-be"
    isp.raw_file = "le.raw"
    isp.load_raw()
    assert isp.raw is not None
    assert isp.sensor_info is not None
    np.testing.assert_array_equal(isp.raw, img)
    assert isp.sensor_info.get("endian_type") == "ieee-le"
    assert cfg["sensor_info"] == caller_snapshot
    bad = tmp_path / "bad.raw"
    bad.write_bytes(b"\x00" * 7)
    isp.raw_file = "bad.raw"
    with pytest.raises(Exception):
        isp.load_raw()
    assert isp.raw is None
    assert isp.decoded_raw is None


def test_atomic_copy_canary_and_failure(tmp_path: Path, monkeypatch) -> None:
    import shutil as shutil_mod

    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    canary = session.paths.directory / "input.pgm.tmp"
    canary.write_bytes(b"preexisting canary")
    good = tmp_path / "good.pgm"
    good.write_bytes(b"P5 2 2 255\n" + bytes([1, 2, 3, 4]))
    session.copy_raw_from(good)
    prior = session.paths.input_raw.read_bytes()
    prior_ui = dict(session.ui_state)

    def fail_copy(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("mid-copy failure")

    monkeypatch.setattr(shutil_mod, "copyfile", fail_copy)
    with pytest.raises(OSError):
        session.copy_raw_from(good)
    assert session.paths.input_raw.read_bytes() == prior
    assert session.ui_state == prior_ui
    assert canary.read_bytes() == b"preexisting canary"
    leftovers = [p.name for p in session.paths.directory.iterdir()
                 if p.name.startswith(".stage-")]
    assert leftovers == []
    assert good.read_bytes() == b"P5 2 2 255\n" + bytes([1, 2, 3, 4])


def test_session_filename_property_guard(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.paths.input_filename = "..\\evil"
    with pytest.raises(ValueError, match="unsafe"):
        _ = session.paths.input_raw
