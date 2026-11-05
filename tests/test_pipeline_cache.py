from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from util.pipeline_cache import ArrayLru, CachedRawDecoder, stage_cache_key
from util.pipeline_types import ProcessingState
from util.raw_decode import RawFormat


def _config() -> dict:
    return {
        "sensor_info": {"width": 8, "height": 6, "bit_depth": 12, "hdr_bit_depth": 24, "bayer_pattern": "rggb"},
        "companding": {"is_enable": False}, "white_balance": {"r_gain": 1, "b_gain": 1},
        "demosaic": {"algorithm": "bilinear"}, "color_correction_matrix": {"is_enable": False},
        "tone_mapping": {"is_enable": True, "operator": "hable"},
        "output": {"transfer": "srgb"},
    }


def _key(config=None, *, stage="white_balance", source="frame-a", image=None, gains=(1, 1), dg=0, state=None) -> str:
    cfg = _config() if config is None else config
    return stage_cache_key(
        stage, np.ones((6, 8), np.float32) if image is None else image,
        source_identity=source, config=cfg, sensor_info=cfg["sensor_info"],
        awb_gains=gains, digital_gain_index=dg, processing_state=state,
    )


def test_array_budget_lru_and_immutable_storage() -> None:
    cache = ArrayLru(16, max_entries=2)
    a = np.arange(4, dtype=np.uint16)
    assert cache.put_array("a", a)
    assert cache.put_array("b", a)
    a[0] = 999
    cached = cache.get_array("a")
    assert cached is not None and cached[0] == 0
    with pytest.raises(ValueError):
        cached.setflags(write=True)
    cached.shape = (2, 2)
    setattr(cached, "dtype", np.dtype(np.uint8))
    again = cache.get_array("a")
    assert again is not None and again.shape == (4,) and again.dtype == np.uint16
    np.testing.assert_array_equal(again, [0, 1, 2, 3])
    assert cache.put_array("c", np.zeros(4, np.uint16))
    assert cache.get_array("b") is None
    assert cache.stats()["array_bytes"] == 16
    assert cache.stats()["entries"] == 2
    assert cache.stats()["evictions"] == 1
    assert not cache.put_array("large", np.zeros(9, np.uint16))
    assert cache.stats()["array_bytes"] == 16
    cache.clear()
    assert cache.stats()["entries"] == cache.stats()["array_bytes"] == 0
    assert not ArrayLru(0).put_array("none", a)


def test_stage_cache_keys_invalidate_upstream_runtime_source_and_actual_bytes() -> None:
    base = _config()
    original = _key(base)
    variants = [
        {"sensor_info": {**base["sensor_info"], "bayer_pattern": "grbg"}},
        {"crop": {"is_enable": True, "left": 1}},
        {"companding": {"is_enable": True}},
        {"digital_gain": {"current_gain": 1}},
        {"white_balance": {"r_gain": 2}},
    ]
    for update in variants:
        cfg = copy.deepcopy(base)
        cfg.update(update)
        assert _key(cfg) != original
    assert _key(source="frame-b") != original
    assert _key(gains=(2, 1)) != original
    assert _key(dg=1) != original
    assert _key(state=ProcessingState(previous_awb_gains=(2, 1))) != original
    assert _key(image=np.full((6, 8), 2, np.float32)) != original
    assert _key(image=np.ones((8, 6), np.float32)) != original
    assert _key(image=np.ones((6, 8), np.float64)) != original


def test_cache_reuses_only_appropriate_downstream_edits() -> None:
    cfg = _config()
    wb, ccm, render = _key(cfg), _key(cfg, stage="color_correction_matrix"), _key(cfg, stage="color_correction_matrix_render")
    cfg["output"]["transfer"] = "rec709"
    cfg["tone_mapping"]["operator"] = "reinhard"
    assert _key(cfg) == wb
    assert _key(cfg, stage="color_correction_matrix") == ccm
    assert _key(cfg, stage="color_correction_matrix_render") != render
    cfg["color_correction_matrix"]["is_enable"] = True
    assert _key(cfg) == wb
    assert _key(cfg, stage="color_correction_matrix") != ccm


def test_decoded_cache_uses_content_identity_and_owned_returned_pixels(tmp_path) -> None:
    path = tmp_path / "frame.pgm"
    raw = np.arange(48, dtype=np.uint16).reshape(6, 8)
    source = b"P5 8 6 4095\n" + raw.astype(">u2").tobytes()
    path.write_bytes(source)
    cache = ArrayLru(4096)
    decoder = CachedRawDecoder(cache, expected_fingerprint=hashlib.sha256(source).hexdigest())
    fmt = RawFormat.from_sensor_info(_config()["sensor_info"])
    first = decoder.decode(path, fmt)
    first.pixels[0, 0] = 123
    second = decoder.decode(path, fmt)
    np.testing.assert_array_equal(second.pixels, raw)
    assert second.pixels.flags.writeable
    assert cache.stats()["hits"] == 1
    second.pixels[0, 0] = 999
    np.testing.assert_array_equal(decoder.decode(path, fmt).pixels, raw)
    path.write_bytes(source[:-1] + bytes([source[-1] ^ 1]))
    with pytest.raises(ValueError, match="staged source changed"):
        decoder.decode(path, fmt)
