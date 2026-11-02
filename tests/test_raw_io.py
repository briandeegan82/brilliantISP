from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from brilliant_isp import BrilliantISP
from tools.gui.session import GuiSession
from tools.gui.yaml_compat import load_preset_yaml, missing_required_keys
from util.raw_io import apply_data_alignment, apply_raw_flips, parse_pgm_header, read_raw_payload

REPO = Path(__file__).resolve().parents[1]
SVS = REPO / "config" / "SVS_cam.yml"
ROD = REPO / "config" / "ROD_cam_daytime_le.yml"


def _p5(path: Path, width: int, height: int, raster: bytes, maxval: int = 255) -> Path:
    path.write_bytes(f"P5\n{width} {height}\n{maxval}\n".encode("ascii") + raster)
    return path


def test_parse_p5_strips_header() -> None:
    raster = bytes([1, 2, 3, 4])
    data = b"P5\n# comment\n2 2\n255\n" + raster
    parsed = parse_pgm_header(data)
    assert parsed is not None
    magic, w, h, maxval, offset = parsed
    assert magic == "P5"
    assert (w, h, maxval) == (2, 2, 255)
    assert data[offset:] == raster


def test_read_raw_payload_p5(tmp_path: Path) -> None:
    raster = bytes(range(8))
    pgm = _p5(tmp_path / "tiny.pgm", 4, 2, raster)
    payload = read_raw_payload(pgm)
    assert payload.is_pgm
    assert payload.data == raster
    assert payload.pgm_width == 4
    assert payload.pgm_height == 2
    assert payload.header_bytes > 0


def test_apply_raw_flips() -> None:
    img = np.arange(8, dtype=np.uint16).reshape(2, 4)
    lr = apply_raw_flips(img, True, False)
    ud = apply_raw_flips(img, False, True)
    both = apply_raw_flips(img, True, True)
    np.testing.assert_array_equal(lr, np.fliplr(img))
    np.testing.assert_array_equal(ud, np.flipud(img))
    np.testing.assert_array_equal(both, np.flipud(np.fliplr(img)))


def test_session_copy_strips_pgm(tmp_path: Path) -> None:
    raster = bytes([9, 8, 7, 6])
    src = _p5(tmp_path / "source.pgm", 2, 2, raster)
    session, _ = GuiSession.create_new(root=tmp_path, preset_path=SVS)
    dest = session.copy_raw_from(src)
    assert dest.read_bytes() == raster
    assert src.read_bytes().startswith(b"P5")
    assert session.last_payload_meta is not None
    assert session.last_payload_meta.is_pgm


def test_hdr_8_3mp_yaml_schema() -> None:
    loaded = load_preset_yaml(ROD)
    assert missing_required_keys(loaded) == []
    si = loaded["sensor_info"]
    assert si["width"] == 2592
    assert si["height"] == 1536
    assert si["sensor"] == "Sony_IMX490"
    assert si["horizontal_flip"] is False
    assert si["vertical_flip"] is False


def test_apply_data_alignment_msb() -> None:
    # 12-bit code 240 left-justified in uint16 → 3840
    img = np.array([[240 << 4, 1000 << 4], [0, 4095 << 4]], dtype=np.uint16)
    out = apply_data_alignment(img, 12, "MSB")
    np.testing.assert_array_equal(out, np.array([[240, 1000], [0, 4095]], dtype=np.uint16))
    # LSB is a no-op
    np.testing.assert_array_equal(apply_data_alignment(img, 12, "LSB"), img)


def test_load_raw_msb_alignment(tmp_path: Path) -> None:
    cfg = load_preset_yaml(SVS)
    cfg["sensor_info"]["width"] = 4
    cfg["sensor_info"]["height"] = 2
    cfg["sensor_info"]["bit_depth"] = 12
    cfg["sensor_info"]["endian_type"] = "ieee-le"
    cfg["sensor_info"]["data_alignment"] = "MSB"
    cfg["sensor_info"]["horizontal_flip"] = False
    cfg["sensor_info"]["vertical_flip"] = False
    cfg["platform"]["filename"] = "msb.pgm"
    cfg["platform"]["debug_enabled"] = False
    cfg_path = tmp_path / "msb.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    codes = np.array([[240, 1000, 512, 2000], [100, 3000, 0, 4095]], dtype=np.uint16)
    raster = (codes << 4).astype("<u2").tobytes()
    _p5(tmp_path / "msb.pgm", 4, 2, raster, maxval=65535)

    isp = BrilliantISP(str(tmp_path), str(cfg_path), outFileName="", output_path=str(tmp_path / "out"))
    isp.load_raw()
    assert isp.raw is not None
    np.testing.assert_array_equal(isp.raw, codes)


def test_load_raw_pgm_with_flip(tmp_path: Path) -> None:
    cfg = load_preset_yaml(SVS)
    cfg["sensor_info"]["width"] = 4
    cfg["sensor_info"]["height"] = 2
    cfg["sensor_info"]["bit_depth"] = 12
    cfg["sensor_info"]["endian_type"] = "ieee-be"
    cfg["sensor_info"]["horizontal_flip"] = True
    cfg["sensor_info"]["vertical_flip"] = False
    cfg["platform"]["filename"] = "tiny.pgm"
    cfg["platform"]["debug_enabled"] = False
    cfg_path = tmp_path / "tiny.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    raster = np.arange(8, dtype=">u2").tobytes()
    _p5(tmp_path / "tiny.pgm", 4, 2, raster, maxval=65535)

    isp = BrilliantISP(str(tmp_path), str(cfg_path), outFileName="", output_path=str(tmp_path / "out"))
    isp.load_raw()
    assert isp.raw is not None
    assert isp.raw.shape == (2, 4)
    expected = apply_raw_flips(np.arange(8, dtype=np.uint16).reshape(2, 4), True, False)
    np.testing.assert_array_equal(isp.raw, expected)
