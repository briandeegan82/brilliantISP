from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.gui.computed import compute_format_stats
from tools.gui.format_map import (
    bayer_from_rggb_start,
    crop_from_margins,
    margins_from_crop,
    rggb_start_from_bayer,
    update_crop_from_margins,
)
from tools.gui.process import prepare_process
from tools.gui.session import GuiSession, recover_last_session
from tools.gui.state import SharedConfig
from tools.gui.validation import can_process, validate_config
from tools.gui.yaml_compat import (
    REQUIRED_CONFIG_KEYS,
    contains_forbidden_aliases,
    dump_config_yaml,
    load_preset_yaml,
    merge_pipeline_feedback,
    missing_required_keys,
)

REPO = Path(__file__).resolve().parents[2]
SVS = REPO / "config" / "SVS_cam.yml"
ROD = REPO / "config" / "ROD_cam_daytime_le.yml"


def _tiny_raw(path: Path, n: int = 32) -> Path:
    path.write_bytes(bytes(range(n)))
    return path


def _cfg(**overrides: object) -> dict:
    cfg = load_preset_yaml(SVS)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)  # type: ignore[union-attr]
        else:
            cfg[key] = val
    return cfg


def test_tc_s01_new_session(tmp_path: Path) -> None:
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    assert session.paths.config_yml.is_file()
    assert session.paths.session_json.is_file()
    assert not session.has_raw
    assert "sensor_info" in config
    last = json.loads((tmp_path / "last_session.json").read_text(encoding="utf-8"))
    assert last["session_id"] == session.session_id


def test_tc_s02_load_image_copies_raw(tmp_path: Path) -> None:
    src = _tiny_raw(tmp_path / "source.raw")
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    dest = session.copy_raw_from(src)
    assert dest.is_file()
    assert dest.read_bytes() == src.read_bytes()
    original = src.read_bytes()
    dest.write_bytes(b"changed")
    assert src.read_bytes() == original
    assert config["platform"]["filename"] == "input.raw" or session.paths.input_raw.name == "input.raw"
    session.ensure_platform_filename(config)
    assert config["platform"]["filename"] == "input.raw"


def test_tc_s03_save_session(tmp_path: Path) -> None:
    src = _tiny_raw(tmp_path / "source.raw")
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.copy_raw_from(src)
    config["sensor_info"]["width"] = 640
    session.ui_state["fit_mode"] = "1:1"
    session.save(config)
    saved = yaml.safe_load(session.paths.config_yml.read_text(encoding="utf-8"))
    assert saved["sensor_info"]["width"] == 640
    ui = json.loads(session.paths.session_json.read_text(encoding="utf-8"))
    assert ui["fit_mode"] == "1:1"
    assert session.has_raw


def test_tc_s04_open_session(tmp_path: Path) -> None:
    src = _tiny_raw(tmp_path / "source.raw")
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.copy_raw_from(src)
    config["sensor_info"]["height"] = 480
    session.ui_state["zoom"] = 2.5
    png = session.paths.output_png
    png.write_bytes(b"\x89PNG")
    session.save(config)

    opened, loaded = GuiSession.open_existing(session.paths.directory, root=tmp_path)
    assert loaded["sensor_info"]["height"] == 480
    assert opened.has_raw
    assert opened.paths.output_png.is_file()
    assert opened.ui_state["zoom"] == 2.5


def test_tc_s05_auto_recover(tmp_path: Path) -> None:
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    recovered = recover_last_session(root=tmp_path)
    assert recovered is not None
    rec_session, _cfg = recovered
    assert rec_session.session_id == session.session_id


def test_tc_s06_process_writes_artifacts(tmp_path: Path) -> None:
    src = _tiny_raw(tmp_path / "source.raw")
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.copy_raw_from(src)
    ok, issues = prepare_process(session, config)
    assert ok, issues
    assert session.paths.config_yml.is_file()
    session.write_output_png(rgb_bytes=b"\x89PNG\r\n")
    assert session.paths.output_png.is_file()


def test_tc_m01_single_object() -> None:
    state = SharedConfig(_cfg())
    state.set("sensor_info.width", 1111)
    assert state.config["sensor_info"]["width"] == 1111
    dumped = state.config
    assert dumped["sensor_info"]["width"] == 1111


def test_tc_m02_ae_awb_shortcuts() -> None:
    state = SharedConfig(_cfg())
    state.apply_ae(False)
    state.apply_awb(True)
    state.set_digital_gain_index(3)
    assert state.get("auto_exposure.is_enable") is False
    assert state.get("auto_white_balance.is_enable") is True
    assert state.get("digital_gain.current_gain") == 3


@pytest.mark.parametrize("preset", [SVS, ROD])
def test_tc_m03_yaml_source_of_truth(preset: Path) -> None:
    loaded = load_preset_yaml(preset)
    missing = missing_required_keys(loaded)
    assert missing == []
    assert contains_forbidden_aliases(loaded) == []
    assert set(REQUIRED_CONFIG_KEYS).issubset(loaded.keys())


def test_tc_v01_size() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = 0
    paths = {i.path for i in validate_config(cfg)}
    assert "sensor_info.width" in paths


def test_tc_v02_crop_bounds() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = 100
    cfg["sensor_info"]["height"] = 80
    cfg["crop"]["is_enable"] = True
    cfg["crop"]["crop_x_start"] = 90
    cfg["crop"]["crop_y_start"] = 0
    cfg["crop"]["new_width"] = 20
    cfg["crop"]["new_height"] = 10
    assert any(i.path == "crop" for i in validate_config(cfg))


def test_disabled_crop_skips_size_validation() -> None:
    cfg = _cfg()
    cfg["crop"]["is_enable"] = False
    cfg["crop"]["crop_x_start"] = 1220
    cfg["crop"]["crop_y_start"] = 440
    cfg["crop"]["new_width"] = 300
    cfg["crop"]["new_height"] = 300
    assert validate_config(cfg) == []


def test_tc_v03_bayer() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["bayer_pattern"] = "rgb"
    assert any("bayer_pattern" in i.path for i in validate_config(cfg))


def test_tc_v04_bit_depth_endian() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["bit_depth"] = 0
    cfg["sensor_info"]["endian_type"] = "middle"
    paths = {i.path for i in validate_config(cfg)}
    assert "sensor_info.bit_depth" in paths
    assert "sensor_info.endian_type" in paths


def test_tc_v05_gains() -> None:
    cfg = _cfg()
    cfg["white_balance"]["r_gain"] = 0
    cfg["white_balance"]["b_gain"] = -1
    cfg["digital_gain"]["gain_array"] = [0.0]
    cfg["digital_gain"]["current_gain"] = 0
    paths = {i.path for i in validate_config(cfg)}
    assert "white_balance.r_gain" in paths
    assert "white_balance.b_gain" in paths
    assert "digital_gain.gain_array" in paths


def test_tc_v06_process_blocked_without_write(tmp_path: Path) -> None:
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    config["sensor_info"]["width"] = -5
    session.paths.config_yml.write_text("sentinel: true\n", encoding="utf-8")
    ok, issues = prepare_process(session, config)
    assert not ok
    assert issues
    assert "sentinel: true" in session.paths.config_yml.read_text(encoding="utf-8")
    ok2, _ = can_process(config, has_raw=False)
    assert ok2 is False


def test_tc_f01_cfa_mapping() -> None:
    assert bayer_from_rggb_start(0) == "rggb"
    assert bayer_from_rggb_start(1) == "grbg"
    assert bayer_from_rggb_start(2) == "gbrg"
    assert bayer_from_rggb_start(3) == "bggr"
    assert rggb_start_from_bayer("gbrg") == 2


def test_tc_f02_crop_margins() -> None:
    geo = crop_from_margins(1920, 1080, 10, 20, 30, 40)
    assert geo["crop_x_start"] == 10
    assert geo["crop_y_start"] == 30
    assert geo["new_width"] == 1890
    assert geo["new_height"] == 1010
    assert geo["is_enable"] is True


def test_disable_crop_keeps_roi_and_validates() -> None:
    crop = {
        "is_enable": True,
        "crop_x_start": 1220,
        "crop_y_start": 440,
        "new_width": 300,
        "new_height": 300,
    }
    assert margins_from_crop(1920, 1536, crop) == (1220, 400, 440, 796)
    crop["is_enable"] = False
    assert margins_from_crop(1920, 1536, crop) == (0, 0, 0, 0)
    update_crop_from_margins(crop, 1920, 1536, 0, 0, 0, 0)
    assert crop["is_enable"] is False
    assert crop["new_width"] == 300
    assert crop["crop_x_start"] == 1220
    cfg = _cfg()
    cfg["crop"].update(crop)
    assert validate_config(cfg) == []


def test_merge_feedback_does_not_shrink_sensor_when_crop_enabled() -> None:
    dst = _cfg()
    dst["crop"]["is_enable"] = True
    dst["sensor_info"]["width"] = 1920
    dst["sensor_info"]["height"] = 1536
    dst["digital_gain"]["is_enable"] = True
    dst["digital_gain"]["is_auto"] = True
    dst["digital_gain"]["current_gain"] = 5
    src = {
        "sensor_info": {"width": 300, "height": 300, "bayer_pattern": "rggb", "bit_depth": 12},
        "white_balance": {"r_gain": 1.4, "b_gain": 1.2},
        "digital_gain": {"current_gain": 2, "ae_feedback": 0},
    }
    merge_pipeline_feedback(dst, src)
    assert dst["sensor_info"]["width"] == 1920
    assert dst["sensor_info"]["height"] == 1536
    assert dst["white_balance"]["r_gain"] == 1.4
    assert dst["digital_gain"]["current_gain"] == 2


def test_merge_feedback_preserves_manual_digital_gain() -> None:
    dst = _cfg()
    dst["digital_gain"]["is_enable"] = False
    dst["digital_gain"]["is_auto"] = False
    dst["digital_gain"]["current_gain"] = 5
    src = {"digital_gain": {"current_gain": 15, "ae_feedback": -1}}
    merge_pipeline_feedback(dst, src)
    assert dst["digital_gain"]["current_gain"] == 5
    assert dst["digital_gain"]["ae_feedback"] == -1


def test_digital_gain_disabled_is_passthrough() -> None:
    import numpy as np
    from modules.digital_gain.digital_gain import DigitalGain

    img = np.array([[100, 200], [300, 400]], dtype=np.uint32)
    platform = {"debug_enabled": False, "in_file": "t", "out_file": "t"}
    sensor = {"hdr_bit_depth": 24, "bit_depth": 12, "bayer_pattern": "rggb"}
    parm = {
        "is_debug": False,
        "is_auto": True,
        "is_enable": False,
        "gain_array": [1.0, 2.0, 128.0],
        "current_gain": 2,
        "ae_feedback": None,
        "is_save": False,
    }
    out, idx = DigitalGain(img, platform, sensor, parm).execute()  # type: ignore[arg-type]
    np.testing.assert_array_equal(out, img)
    assert idx == 2


def test_digital_gain_enabled_applies_array_gain() -> None:
    import numpy as np
    from modules.digital_gain.digital_gain import DigitalGain

    img = np.array([[10, 20], [30, 40]], dtype=np.uint32)
    platform = {"debug_enabled": False, "in_file": "t", "out_file": "t"}
    sensor = {"hdr_bit_depth": 24, "bit_depth": 12, "bayer_pattern": "rggb"}
    parm = {
        "is_debug": False,
        "is_auto": False,
        "is_enable": True,
        "gain_array": [1.0, 2.0, 4.0],
        "current_gain": 2,
        "ae_feedback": None,
        "is_save": False,
    }
    out, idx = DigitalGain(img, platform, sensor, parm).execute()  # type: ignore[arg-type]
    np.testing.assert_array_equal(out, img * 4)
    assert idx == 2


def test_tc_f03_computed_fields() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = 100
    cfg["sensor_info"]["height"] = 50
    cfg["sensor_info"]["bit_depth"] = 16
    cfg["sensor_info"]["data_format"] = "uint16"
    cfg["crop"]["is_enable"] = True
    cfg["crop"]["new_width"] = 80
    cfg["crop"]["new_height"] = 40
    cfg["scale"]["is_enable"] = True
    cfg["scale"]["new_width"] = 40
    cfg["scale"]["new_height"] = 20
    stats = compute_format_stats(cfg, header_bytes=8)
    assert stats["effective_width"] == 80
    assert stats["effective_height"] == 40
    assert stats["active_pixel_count"] == 3200
    assert abs(stats["crop_percentage"] - 36.0) < 1e-6
    assert stats["estimated_raw_size"] == 8 + 100 * 50 * 2
    assert stats["output_width"] == 40
    assert stats["output_height"] == 20


def test_tc_c03_yaml_names() -> None:
    cfg = _cfg()
    bnr = cfg["bayer_noise_reduction"]
    assert "filter_window" in bnr
    assert "filt_window" not in bnr
    assert "is_enable" in cfg["sharpen"]
    gamma = cfg["gamma_correction"]
    assert "curve" in gamma
    assert "gamma" in gamma
    assert contains_forbidden_aliases(cfg) == []


def test_tc_t01_dirty_mark() -> None:
    state = SharedConfig(_cfg())
    assert state.is_dirty() is False
    state.set("white_balance.r_gain", 1.95)
    assert state.is_dirty("white_balance.r_gain")
    assert "white_balance.r_gain" in state.dirty_paths()


def test_tc_t03_reset() -> None:
    state = SharedConfig(_cfg())
    orig = state.get("white_balance.r_gain")
    state.set("white_balance.r_gain", 9.9)
    state.set("sharpen.sharpen_sigma", 8.0)
    state.reset_parameter("white_balance.r_gain")
    assert state.get("white_balance.r_gain") == orig
    state.reset_module("sharpen")
    assert state.get("sharpen.sharpen_sigma") == state.preset_snapshot["sharpen"]["sharpen_sigma"]
    state.set("sensor_info.width", 12)
    state.reset_all()
    assert state.config == state.preset_snapshot


def test_tc_i02_export_required_keys(tmp_path: Path) -> None:
    cfg = load_preset_yaml(SVS)
    out = tmp_path / "export.yml"
    dump_config_yaml(cfg, out)
    exported = load_preset_yaml(out)
    assert missing_required_keys(exported) == []


def test_tc_i03_reload_does_not_clobber_isp(tmp_path: Path) -> None:
    src = _tiny_raw(tmp_path / "source.raw", 16)
    session, config = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    session.copy_raw_from(src)
    config["sharpen"]["is_enable"] = True
    src.write_bytes(b"ABCDEFGH" * 4)
    session.reload_raw()
    assert session.paths.input_raw.read_bytes().startswith(b"ABCD")
    assert config["sharpen"]["is_enable"] is True
