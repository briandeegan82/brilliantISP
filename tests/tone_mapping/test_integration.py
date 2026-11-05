from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from modules.tone_mapping.migration import migrate_tone_mapping
from modules.tone_mapping.registry import (
    ResolvedToneMapping,
    ToneMappingError,
    resolve_tone_mapping,
)
from modules.tone_mapping.tone_mapping import ToneMapping
from modules.tone_mapping.backends import map_luminance

REPO = Path(__file__).resolve().parents[2]
SVS = REPO / "config" / "SVS_cam.yml"


def _svs() -> dict:
    return yaml.safe_load(SVS.read_text())


def _base() -> dict:
    cfg = _svs()
    return cfg


class _Log:
    def warning(self, *a): pass
    def info(self, *a): pass


class _Ctx:
    def __init__(self, cfg):
        self.tone_mapping = cfg["tone_mapping"]
        self.platform = {"in_file": "t", "out_file": "t", "output_dir": ""}
        self.sensor_info = cfg["sensor_info"]
        self.param_durand = cfg.get("hdr_durand", {})
        self.param_aces = cfg.get("aces", {})
        self.param_aces_integer = cfg.get("aces_integer", {})
        self.param_hable = cfg.get("hable", {})
        self.param_hable_integer = cfg.get("hable_integer", {})
        self.param_integer_tmo = cfg.get("reinhard_integer", {})
        self.c_yaml = cfg
        self.logger = _Log()


def test_resolver_legacy_matrix() -> None:
    for selector, expected_op, expected_backend in (
        ("aces", "aces_fitted", "float"),
        ("aces_integer", "aces_fitted", "lut"),
        ("hable", "hable", "float"),
        ("hable_integer", "hable", "lut"),
        ("reinhard_integer", "reinhard", "fixed"),
        ("integer_tmo", "reinhard", "fixed"),
        ("hdr_durand", "durand", "cpu"),
        ("durand", "durand", "cpu"),
    ):
        cfg = _base()
        cfg["tone_mapping"]["tone_mapper"] = selector
        r = resolve_tone_mapping(cfg)
        assert r.spec.operator == expected_op
        assert r.spec.backend == expected_backend
        assert r.placement == "pre_demosaic"
        assert isinstance(r, ResolvedToneMapping)


def test_resolver_defaults_and_notices() -> None:
    cfg = _base()
    r = resolve_tone_mapping(cfg)
    assert r.spec.operator == "hable"
    assert r.spec.backend == "lut"
    cfg2 = _base()
    cfg2["tone_mapping"]["tone_mapper"] = "aces"
    r2 = resolve_tone_mapping(cfg2)
    assert r2.spec.normalization == "per_frame"
    assert r2.spec.domain_scale == 100.0
    assert r2.spec.lut_domain == 16.0
    assert any("OETF" in n for n in r2.notices)
    for selector, expected_op, expected_backend in (
        ("aces_integer", "aces_fitted", "lut"),
        ("integer_tmo", "reinhard", "fixed"),
        ("hdr_durand", "durand", "cpu"),
    ):
        cfg3 = _base()
        cfg3["tone_mapping"]["tone_mapper"] = selector
        r3 = resolve_tone_mapping(cfg3)
        assert (r3.spec.operator, r3.spec.backend) == (
            expected_op, expected_backend)


def test_resolver_canonical() -> None:
    cfg = _base()
    cfg["tone_mapping"].update(
        operator="reinhard", backend="lut", normalization="per_frame",
        exposure_ev=1.0,
    )
    r = resolve_tone_mapping(cfg)
    assert r.spec.operator == "reinhard"
    assert r.spec.backend == "lut"
    assert r.placement == "post_ccm"
    assert any("overrides" in n for n in r.notices)
    cfg["tone_mapping"].pop("backend")
    with pytest.raises(ToneMappingError):
        resolve_tone_mapping(cfg)
    cfg["tone_mapping"]["is_enable"] = False
    r3 = resolve_tone_mapping(cfg)
    assert not r3.spec.enabled


def test_resolver_disabled_skips_garbage() -> None:
    cfg = _base()
    cfg["tone_mapping"].update(is_enable=False, operator="bogus",
                               backend="bogus")
    r = resolve_tone_mapping(cfg)
    assert not r.spec.enabled


def test_adapter_pre_and_post(tmp_path) -> None:
    cfg = _base()
    ctx: Any = _Ctx(cfg)
    img = np.arange(8 * 8, dtype=np.uint32).reshape(8, 8) * 1000 % (2**24)
    tmo = ToneMapping(img, ctx)
    out = tmo.execute(save_output=False, plot_curve=False)
    assert out.dtype == np.uint16
    assert out.shape == img.shape
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = False
    rgb = np.stack([img, img, img], axis=-1).astype(np.uint16)
    tmo2 = ToneMapping(rgb, ctx)
    out2 = tmo2.execute(save_output=False, plot_curve=False)
    assert out2.shape == rgb.shape
    assert out2.dtype == np.uint16


def test_adapter_disabled_identity() -> None:
    cfg = _base()
    cfg["tone_mapping"]["is_enable"] = False
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = False
    ctx: Any = _Ctx(cfg)
    rgb = np.zeros((4, 4, 3), np.uint16)
    tmo = ToneMapping(rgb, ctx)
    out = tmo.execute(save_output=False, plot_curve=False)
    assert out is rgb


def test_wrappers_import() -> None:
    from modules.tone_mapping.aces.aces_tone_mapping import ACESToneMapping
    from modules.tone_mapping.hable.hable_tone_mapping import HableToneMapping
    from modules.tone_mapping.integer_tmo.integer_tone_mapping import IntegerReinhardToneMapping
    from modules.tone_mapping.integer_tmo.aces_integer_tone_mapping import ACESIntegerToneMapping
    from modules.tone_mapping.integer_tmo.hable_integer_tone_mapping import HableIntegerToneMapping
    from modules.tone_mapping.durand.hdr_durand_fast import HDRDurandToneMapping
    from modules.tone_mapping.durand.hdr_durand_fast_gpu import HDRDurandToneMappingGPU

    platform = {"in_file": "t", "out_file": "t"}
    sensor = {"hdr_bit_depth": 12, "output_bit_depth": 8}
    img = np.full((4, 4), 0.5)
    int_img = np.full((4, 4), 100, np.uint16)
    assert ACESToneMapping(img, platform, sensor, {}).execute().shape == (4, 4)
    assert HableToneMapping(img, platform, sensor, {}).execute().shape == (4, 4)
    assert IntegerReinhardToneMapping(int_img, platform, sensor, {}).execute().dtype == np.uint16
    assert ACESIntegerToneMapping(int_img, platform, sensor, {}).execute().dtype == np.uint16
    assert HableIntegerToneMapping(int_img, platform, sensor, {}).execute().dtype == np.uint16
    assert HDRDurandToneMapping(img, platform, sensor, {}).execute().shape == (4, 4)
    assert HDRDurandToneMappingGPU(img, platform, sensor, {}).execute().shape == (4, 4)


def test_durand_constant_regression() -> None:
    const = np.full((4, 4), 100, np.uint16)
    out = map_luminance(
        const, input_max=255, output_max=255,
        config=__import__(
            "modules.tone_mapping.registry", fromlist=["ToneMappingSpec"]
        ).ToneMappingSpec(operator="durand", backend="cpu"),
    )
    assert float(out.flat[0]) == pytest.approx(100.0)


def test_diagnostics_gating(tmp_path) -> None:
    from modules.tone_mapping import diagnostics
    from modules.tone_mapping.registry import ToneMappingSpec

    platform = {"output_dir": str(tmp_path)}
    s = ToneMappingSpec(operator="reinhard", backend="lut")
    p = diagnostics.plot_curve(s, platform)
    assert p is not None and p.is_file()
    s_off = ToneMappingSpec(operator="reinhard", enabled=False)
    assert diagnostics.plot_curve(s_off, platform) is None
    assert diagnostics.plot_curve(s, {"output_dir": ""}) is None


def test_migrate_valid_and_report(tmp_path) -> None:
    cfg = _base()
    src_copy = yaml.safe_load(yaml.safe_dump(cfg))
    migrated, report = migrate_tone_mapping(cfg)
    assert report["migration"] == "tone_mapping_m3"
    assert report["selected_legacy_section"] == "hable_integer"
    assert "hdr_durand" in report["legacy_sections"]
    assert "aces" in report["inactive_presets"]
    assert migrated["tone_mapping"]["operator"] == "hable"
    assert migrated["tone_mapping"]["backend"] == "lut"
    assert "hable_integer" not in migrated
    assert cfg == src_copy
    assert "tone_mapping.operator" not in str(report["notices"])


def test_migrate_all_selected_aliases() -> None:
    for selector in ("aces", "aces_integer", "hable", "hable_integer",
                     "reinhard_integer", "integer_tmo", "hdr_durand"):
        cfg = _base()
        cfg["tone_mapping"]["tone_mapper"] = selector
        migrated, report = migrate_tone_mapping(cfg)
        assert "operator" in migrated["tone_mapping"]


def test_migrate_disabled_and_failures() -> None:
    cfg = _base()
    cfg["tone_mapping"]["is_enable"] = False
    migrated, report = migrate_tone_mapping(cfg)
    assert migrated["tone_mapping"]["is_enable"] is False
    cfg2 = _base()
    cfg2["tone_mapping"]["tone_mapper"] = "bogus"
    cfg2["tone_mapping"]["is_enable"] = False
    with pytest.raises(ToneMappingError):
        migrate_tone_mapping(cfg2)
    cfg3 = _base()
    cfg3["tone_mapping"]["operator"] = "hable"
    cfg3["tone_mapping"]["backend"] = "float"
    canon, rep3 = migrate_tone_mapping(cfg3)
    assert canon["tone_mapping"]["operator"] == "hable"
    assert canon["schema_version"] == 1


def test_migrate_cli(tmp_path) -> None:
    import subprocess, sys
    out = tmp_path / "migrated.yml"
    rep = tmp_path / "report.json"
    cfg_file = tmp_path / "src.yml"
    cfg_file.write_text(yaml.safe_dump(_base(), sort_keys=False))
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "migrate_tone_mapping.py"),
         "--config", str(cfg_file), "--output", str(out), "--report", str(rep)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    migrated = yaml.safe_load(out.read_text())
    assert migrated["tone_mapping"]["operator"] == "hable"
    report = json.loads(rep.read_text())
    assert report["migration"] == "tone_mapping_m3"
    r2 = subprocess.run(
        [sys.executable, str(REPO / "tools" / "migrate_tone_mapping.py"),
         "--config", str(cfg_file), "--output", str(out), "--report", str(rep)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r2.returncode == 2
    src_after = yaml.safe_load(cfg_file.read_text())
    assert src_after == yaml.safe_load(yaml.safe_dump(_base()))


def test_export_canonical(tmp_path) -> None:
    import tools.export_boltisp_yaml as exporter

    cfg = _base()
    cfg["tone_mapping"].update(
        operator="hable", backend="float",
        parameters={"exposure_bias": 2.5, "white_point": 10.0},
        domain_scale=3.0, exposure_ev=1.0, normalization="per_frame",
    )
    warnings: list[str] = []
    exporter._apply_tonemap_translation(cfg, warnings)
    hdr = cfg["hdr_tone_mapping"]
    assert hdr["operator"] == "hable"
    assert hdr["hable_exposure_bias"] == 2.5
    assert hdr["hable_hdr_scale"] == 6.0
    assert hdr["hable_use_normalization"] is False
    cfg2 = _base()
    cfg2["tone_mapping"].update(operator="aces_fitted", backend="float")
    with pytest.raises(ValueError, match="verified"):
        exporter._apply_tonemap_translation(cfg2, [])
    cfg3 = _base()
    cfg3["tone_mapping"].update(
        operator="durand", backend="cpu",
        parameters={"sigma_space": 4.0, "contrast_factor": 3.0},
    )
    with pytest.raises(ValueError, match="verified"):
        exporter._apply_tonemap_translation(cfg3, [])
    cfg4 = _base()
    cfg4["tone_mapping"].update(operator="reinhard", backend="lut")
    with pytest.raises(ValueError, match="verified"):
        exporter._apply_tonemap_translation(cfg4, [])


def test_gui_strict_parser() -> None:
    from tools.gui.validation import parse_numeric_editor

    assert parse_numeric_editor("3", 2) == 3
    assert parse_numeric_editor("2.5", 2) == 2.5
    assert parse_numeric_editor("abc", 2) == "abc"
    assert parse_numeric_editor("256.5", 65536) == 256.5
    assert parse_numeric_editor("3.0", 2) == 3


def test_process_frame_canonical(tmp_path) -> None:
    from tests.test_pipeline_runner import _fast_config
    from util.pipeline_runner import process_frame

    for op, backend in (
        ("aces_fitted", "float"), ("aces_fitted", "lut"),
        ("hable", "float"), ("reinhard", "fixed"), ("durand", "cpu"),
    ):
        cfg = _fast_config()
        cfg["tone_mapping"] = {
            "is_enable": True, "operator": op, "backend": backend,
            "parameters": {},
        }
        before = yaml.safe_load(yaml.safe_dump(cfg))
        frame = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 10 % 4096
        result = process_frame(frame, cfg)
        assert result.preview_rgb8 is not None
        assert cfg == before
