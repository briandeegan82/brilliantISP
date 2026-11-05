from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from modules.tone_mapping import diagnostics
from modules.tone_mapping.backends import map_luminance
from modules.tone_mapping.local_durand import map_luminance as local_map
from modules.tone_mapping.migration import migrate_tone_mapping
from modules.tone_mapping.normalization import FrameStats, prepare_curve_input
from modules.tone_mapping.registry import ToneMappingError, ToneMappingSpec, resolve_tone_mapping
from modules.tone_mapping.tone_mapping import ToneMapping
from tests.test_pipeline_runner import _fast_config
from util.isp_types import ToneMappingContext

ALIASES = (
    "aces", "aces_integer", "hable", "hable_integer",
    "reinhard_integer", "integer_tmo", "durand", "hdr_durand",
)
BACKENDS = (
    ("aces_fitted", "float"), ("aces_fitted", "lut"),
    ("hable", "float"), ("hable", "lut"),
    ("reinhard", "float"), ("reinhard", "lut"),
    ("reinhard", "fixed"), ("durand", "cpu"),
)


def config(selector: str, **parameters) -> dict:
    cfg = _fast_config()
    cfg["tone_mapping"] = {
        "is_enable": True, "tone_mapper": selector,
        "tone_mapping_before_demosaic": True,
    }
    section = "hdr_durand" if selector in ("durand", "hdr_durand") else selector
    cfg[section] = {"is_enable": True, **parameters}
    return cfg


def context(cfg: dict, output_dir: Path | None = None) -> ToneMappingContext:
    return cast(ToneMappingContext, SimpleNamespace(
        c_yaml=cfg, tone_mapping=cfg["tone_mapping"], sensor_info=cfg["sensor_info"],
        platform={"in_file": "test.raw", "output_dir": str(output_dir) if output_dir else ""},
        logger=SimpleNamespace(warning=lambda message: None),
        param_aces=cfg.get("aces", {}), param_aces_integer=cfg.get("aces_integer", {}),
        param_hable=cfg.get("hable", {}), param_hable_integer=cfg.get("hable_integer", {}),
        param_integer_tmo=cfg.get("reinhard_integer", cfg.get("integer_tmo", {})),
        param_durand=cfg.get("hdr_durand", {}),
    ))


@pytest.mark.parametrize("selector", ("aces", "aces_integer"))
def test_aces_alias_precedence_is_lazy(selector: str) -> None:
    primary, fallback = (
        ("exposure_adjust", "exposure_adjustment") if selector == "aces"
        else ("exposure_adjustment", "exposure_adjust")
    )
    cfg = config(selector, **{primary: 0.5, fallback: "ignored-invalid-alias"})
    assert resolve_tone_mapping(cfg).spec.exposure_ev == 0.5


@pytest.mark.parametrize("selector", ("hable", "hable_integer"))
def test_hable_bias_is_not_ev_and_unused_alias_is_not_validated(selector: str) -> None:
    cfg = config(selector, exposure_bias=3.5, exposure_adjust="ignored-invalid-alias")
    resolved = resolve_tone_mapping(cfg)
    assert resolved.spec.exposure_ev == 0
    assert resolved.spec.parameters_map()["exposure_bias"] == 3.5
    cfg[selector].pop("exposure_bias")
    cfg[selector]["exposure_adjust"] = 4.5
    assert resolve_tone_mapping(cfg).spec.parameters_map()["exposure_bias"] == 4.5


@pytest.mark.parametrize("kwargs", (
    {"parameters": None}, {"parameters": 123},
    {"domain_scale": 10**400}, {"exposure_ev": 10**400},
))
def test_invalid_spec_containers_and_numbers_are_named_errors(kwargs: dict) -> None:
    with pytest.raises(ToneMappingError):
        ToneMappingSpec(operator="hable", **kwargs)


def test_disabled_spec_ignores_all_operator_controls() -> None:
    inactive: dict[str, Any] = {
        "operator": "bogus", "backend": "bogus", "enabled": False,
        "normalize_output": "invalid-inactive", "parameters": None,
    }
    spec = ToneMappingSpec(**inactive)
    data = np.ones((2, 2), np.uint16)
    assert map_luminance(data, input_max=65535, output_max=65535, config=spec) is data
    assert local_map(data, input_max=65535, output_max=65535, config=spec) is data


def test_saturation_does_not_execute_overflowing_multiply() -> None:
    with np.errstate(all="raise"):
        result = prepare_curve_input(
            np.array([0.0, 1e308]), input_max=1, normalization="declared_range",
            gain=1e10, lut_domain=16,
        )
    np.testing.assert_array_equal(result, [0, 16])


def test_unrepresentable_normalization_is_rejected() -> None:
    with pytest.raises(ValueError):
        map_luminance(
            np.array([[-1e308, 1e308]]), input_max=1, output_max=65535,
            config=ToneMappingSpec("hable", normalization="per_frame"),
            frame_stats=FrameStats(-1e308, 1e308),
        )


@pytest.mark.parametrize("input_max,output_max", ((0, 255), (255, 0), (256, 65536)))
def test_direct_local_range_validation(input_max: int, output_max: int) -> None:
    with pytest.raises(ToneMappingError):
        local_map(
            np.ones((2, 2), np.uint16), input_max=input_max, output_max=output_max,
            config=ToneMappingSpec("durand", backend="cpu"),
        )


def test_direct_local_checks_unsigned_range() -> None:
    with pytest.raises(ToneMappingError):
        local_map(
            np.full((2, 2), 256, np.uint16), input_max=255, output_max=255,
            config=ToneMappingSpec("durand", backend="cpu"),
        )


@pytest.mark.parametrize("field,value", (
    ("parameters", None), ("parameters", {"gamma": 2.4}),
    ("gamma", 2.4), ("domain_sclae", 2), ("normalize_output", "false"),
    ("placement", "pre_demosiac"),
))
def test_canonical_active_settings_are_not_silently_discarded(field: str, value) -> None:
    cfg = config("hable")
    cfg["tone_mapping"] = {"is_enable": True, "operator": "hable", "backend": "float", field: value}
    with pytest.raises(ToneMappingError):
        resolve_tone_mapping(cfg)


@pytest.mark.parametrize("selector", ALIASES)
@pytest.mark.parametrize("pre", (True, False))
def test_alias_adapter_and_migration_are_exactly_equivalent(selector: str, pre: bool) -> None:
    cfg = config(
        selector, exposure_bias=3.5, white_point=8.0, hdr_scale=2.0,
        knee=0.4, strength=1.5, sigma_space=1.3, sigma_color=0.6, contrast_factor=1.8,
    )
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = pre
    snapshot = copy.deepcopy(cfg)
    plane = np.arange(48, dtype=np.uint32).reshape(6, 8) * 71
    image = plane if pre else np.stack((plane * 3, plane * 2, plane), axis=-1).astype(np.uint16)
    migrated, report = migrate_tone_mapping(cfg)
    actual = ToneMapping(image, context(cfg)).execute(False, False)
    expected = ToneMapping(image, context(migrated)).execute(False, False)
    np.testing.assert_array_equal(actual, expected)
    assert cfg == snapshot
    assert report["legacy_sections"]
    second, _ = migrate_tone_mapping(migrated)
    assert second == migrated


def test_selected_disabled_stays_disabled_after_migration() -> None:
    cfg = config("hable", exposure_bias=5.0, white_point=7.0)
    cfg["hable"]["is_enable"] = False
    migrated, _ = migrate_tone_mapping(cfg)
    assert migrated["tone_mapping"]["is_enable"] is False
    assert migrated["tone_mapping"]["parameters"]["exposure_bias"] == 5.0
    cfg["hable"]["white_point"] = -1
    with pytest.raises(ToneMappingError):
        migrate_tone_mapping(cfg)


@pytest.mark.parametrize("operator,backend", BACKENDS)
def test_real_isolated_curve_artifact(operator: str, backend: str, tmp_path: Path) -> None:
    spec = ToneMappingSpec(operator, backend=backend)
    path = diagnostics.plot_curve(spec, {"output_dir": str(tmp_path)})
    assert path is not None and path.is_file()


def test_save_gate_and_saved_native_array(tmp_path: Path) -> None:
    cfg = config("hable", is_save=True, is_plot_curve=True)
    cfg["tone_mapping"].update(is_save=True, is_plot_curve=True)
    frame = np.arange(48, dtype=np.uint16).reshape(6, 8) * 71
    tmo = ToneMapping(frame, context(cfg, tmp_path))
    tmo.execute(save_output=False)
    assert list(tmp_path.iterdir()) == []
    mapped = tmo.execute(visualize_output=False)
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".npy"
    np.testing.assert_array_equal(np.load(files[0], allow_pickle=False), mapped)
    original_file = files[0]
    original_bytes = original_file.read_bytes()
    repeated = tmo.execute(visualize_output=False)
    files = list(tmp_path.iterdir())
    assert len(files) == 2 and all(p.suffix == ".npy" for p in files)
    assert original_file.read_bytes() == original_bytes
    for path in files:
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), repeated)


def test_integer_bayer_reaches_lut_without_float_conversion(monkeypatch) -> None:
    import modules.tone_mapping.tone_mapping as adapter

    cfg = config("hable_integer")
    frame = np.arange(48, dtype=np.uint32).reshape(6, 8)
    original = adapter.map_luminance
    inputs = []

    def capture(source, **kwargs):
        inputs.append(source)
        return original(source, **kwargs)

    monkeypatch.setattr(adapter, "map_luminance", capture)
    ToneMapping(frame, context(cfg)).execute(False, False)
    assert len(inputs) == 1 and inputs[0] is frame


def test_gui_flush_rejects_fraction_and_recovers_declared_integer() -> None:
    import tkinter as tk
    from tools.gui.state import SharedConfig
    from tools.isp_calibration_gui import CalibrationApp

    interp = tk.Tcl()
    app: Any = CalibrationApp.__new__(CalibrationApp)
    app.tk = tk
    app._syncing = False
    app.state = SharedConfig({"tone_mapping": {"lut_size": 65536}, "digital_gain": {"current_gain": 0}})
    var = tk.StringVar(interp, "256.5")
    app._isp_vars = {"tone_mapping.lut_size": var}
    app.var_dg = tk.StringVar(interp, "0")
    app._flush_isp_form_to_state()
    assert app.state.get("tone_mapping.lut_size") == 256.5
    var.set("invalid")
    app._flush_isp_form_to_state()
    assert app.state.get("tone_mapping.lut_size") == "invalid"
    var.set("1024")
    app._flush_isp_form_to_state()
    assert type(app.state.get("tone_mapping.lut_size")) is int
    assert app.state.get("tone_mapping.lut_size") == 1024


@pytest.mark.parametrize("operator", ("aces_fitted", "hable", "reinhard"))
def test_normalized_output_small_exposure_does_not_collapse_lut(operator: str) -> None:
    source = np.linspace(0, 4095, 129).astype(np.uint16)
    options: dict[str, Any] = dict(operator=operator, exposure_ev=-20, normalize_output=True)
    reference = map_luminance(
        source, input_max=4095, output_max=65535,
        config=ToneMappingSpec(**options),
    )
    actual = map_luminance(
        source, input_max=4095, output_max=65535,
        config=ToneMappingSpec(**options, backend="lut"),
    )
    np.testing.assert_allclose(actual, reference, atol=2)


@pytest.mark.parametrize("module,name,selector", (
    ("aces.aces_tone_mapping", "ACESToneMapping", "aces"),
    ("hable.hable_tone_mapping", "HableToneMapping", "hable"),
    ("integer_tmo.aces_integer_tone_mapping", "ACESIntegerToneMapping", "aces_integer"),
    ("integer_tmo.hable_integer_tone_mapping", "HableIntegerToneMapping", "hable_integer"),
    ("integer_tmo.integer_tone_mapping", "IntegerReinhardToneMapping", "reinhard_integer"),
    ("durand.hdr_durand_fast", "HDRDurandToneMapping", "durand"),
))
def test_wrappers_use_the_same_resolver(module: str, name: str, selector: str) -> None:
    cls = getattr(importlib.import_module(f"modules.tone_mapping.{module}"), name)
    params = {
        "is_enable": True, "exposure_adjust": 0.5, "exposure_adjustment": -0.5,
        "exposure_bias": 3.5, "white_point": 7.0, "sigma_space": 1.5,
        "sigma_color": 0.6, "contrast_factor": 1.7, "knee": 0.4, "strength": 2.0,
    }
    cfg = config(selector, **params)
    expected = resolve_tone_mapping(cfg).spec
    source = np.linspace(0, 1, 48).reshape(6, 8) if cls.float_mode else np.arange(48, dtype=np.uint16).reshape(6, 8)
    instance = cls(source, {"in_file": "test"}, {"hdr_bit_depth": 12}, params)
    assert instance.spec == expected
    out = instance.execute()
    direct = map_luminance(
        source, input_max=1 if cls.float_mode else 4095,
        output_max=1 if cls.float_mode else 65535, config=expected,
    )
    np.testing.assert_array_equal(out, direct)
    disabled = cls(source, {}, {"hdr_bit_depth": 12}, {
        "is_enable": False, "exposure_bias": "inactive", "white_point": "inactive",
    })
    assert not disabled.spec.enabled
    disabled.execute()


@pytest.mark.parametrize("operator,backend", BACKENDS)
@pytest.mark.parametrize("placement", ("pre_demosaic", "post_ccm"))
def test_real_pipeline_canonical_matrix_has_no_io(operator: str, backend: str, placement: str, tmp_path: Path) -> None:
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    cfg = _fast_config()
    cfg["tone_mapping"].update(
        is_enable=True, operator=operator, backend=backend, placement=placement,
        parameters={},
    )
    snapshot = copy.deepcopy(cfg)
    frame = (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) * 10 % 4096).astype(np.uint16)
    original = frame.copy()
    out_dir = tmp_path / "must_not_exist"
    result = process_frame(frame, cfg, options=ProcessingOptions(output_dir=out_dir))
    assert result.preview_rgb8.shape == (48, 64, 3)
    assert cfg == snapshot and not out_dir.exists()
    np.testing.assert_array_equal(frame, original)


def test_fixed_rgb_uses_exact_weighted_integer_domain(monkeypatch) -> None:
    import modules.tone_mapping.tone_mapping as adapter

    cfg = config("reinhard_integer", knee=0.4)
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = False
    image = np.array([[[65535, 0, 0], [0, 65535, 0], [0, 0, 65535], [17, 3, 1]]], np.uint16)
    observed = []
    original = adapter.map_luminance

    def capture(source, **kwargs):
        observed.append((source.copy(), kwargs["input_max"]))
        return original(source, **kwargs)

    monkeypatch.setattr(adapter, "map_luminance", capture)
    output = ToneMapping(image, context(cfg)).execute(False, False)
    assert observed[0][0].dtype == np.uint32
    np.testing.assert_array_equal(observed[0][0], [[2126 * 65535, 7152 * 65535, 722 * 65535, 2126 * 17 + 7152 * 3 + 722]])
    assert observed[0][1] == 65535 * 10000
    assert output.dtype == np.uint16 and output.shape == image.shape


def test_disabled_pre_uses_declared_output_max_and_canonical_ignores_legacy_garbage() -> None:
    cfg = config("hable")
    cfg["tone_mapping"].update(is_enable=False, tone_mapper="unknown")
    cfg["sensor_info"]["pipeline_rgb_bit_depth"] = 12
    image = np.array([[0, 4095]], np.uint32)
    np.testing.assert_array_equal(ToneMapping(image, context(cfg)).execute(), [[0, 4095]])
    cfg["tone_mapping"] = {
        "is_enable": True, "operator": "hable", "backend": "float",
        "tone_mapper": "hable", "placement": "pre_demosaic",
    }
    cfg["hable"] = "irrelevant-invalid-legacy-section"
    assert ToneMapping(image, context(cfg)).execute(False, False).dtype == np.uint16


def test_hable_near_black_matches_high_precision_equation() -> None:
    from decimal import Decimal, localcontext
    from modules.tone_mapping.operators import hable_partial

    with localcontext() as ctx:
        ctx.prec = 80
        for text in ("0", "1e-12", "1e-6", "0.25", "1"):
            z = Decimal(text)
            reference = (
                (Decimal(".15") * z * z + Decimal(".05") * z + Decimal(".004"))
                / (Decimal(".15") * z * z + Decimal(".5") * z + Decimal(".06"))
                - Decimal(1) / Decimal(15)
            )
            actual = hable_partial(np.array([float(z)]))[0]
            if z == 0:
                assert actual == 0
            else:
                np.testing.assert_allclose(actual, float(reference), rtol=1e-14, atol=0)


def test_real_legacy_alias_context_tracks_selected_live_parameters() -> None:
    from brilliant_isp import BrilliantISP
    from util.pipeline_types import ProcessingOptions

    cfg = config("integer_tmo", knee=0.6, strength=2.0)
    cfg["reinhard_integer"] = {"is_enable": True, "knee": 0.3}
    engine = BrilliantISP("", cfg, outFileName="", options=ProcessingOptions())
    loaded = engine.c_yaml
    assert loaded is not None and "integer_tmo" in loaded and "reinhard_integer" in loaded
    assert engine.param_integer_tmo is loaded["integer_tmo"]
    frame = np.arange(48, dtype=np.uint16).reshape(6, 8)
    adapter = ToneMapping(frame, cast(ToneMappingContext, engine))
    engine.param_integer_tmo["knee"] = 0.7
    adapter.execute(False, False)
    assert adapter.resolved_spec.parameters_map()["knee"] == 0.7
    assert loaded["reinhard_integer"].get("knee") == 0.3
    engine.tone_mapping["tone_mapper"] = "reinhard_integer"
    adapter.execute(False, False)
    assert adapter.resolved_spec.parameters_map()["knee"] == 0.3
    engine.param_integer_tmo = {"is_enable": True, "knee": 0.8}
    adapter.execute(False, False)
    assert adapter.resolved_spec.parameters_map()["knee"] == 0.8


def test_mapped_save_collision_and_partial_cleanup_preserve_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "frame_tone_mapping.npy"
    existing.write_bytes(b"existing-user-data")
    with pytest.raises(ValueError):
        diagnostics.save_mapped_output(
            np.array([object()], dtype=object),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
        )
    assert existing.read_bytes() == b"existing-user-data"
    assert set(tmp_path.iterdir()) == {existing}


def test_native_tmo_diagnostics_one_final_npy(tmp_path: Path, monkeypatch) -> None:
    import json
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    monkeypatch.chdir(tmp_path)
    cfg = config("hable", is_save=True)
    cfg["platform"]["render_3a"] = True
    frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 4096
    out_dir = tmp_path / "diagnostics"
    result = process_frame(frame, cfg, options=ProcessingOptions(
        output_dir=out_dir, diagnostics=True, capture_stages=("tone_mapping_pre",),
    ))
    npys = list(out_dir.rglob("*.npy"))
    jsons = list(out_dir.rglob("*.json"))
    active_tmo = [e for e in result.timings
                  if e.stage_id == "tone_mapping_pre" and e.status == "completed"]
    assert len(active_tmo) == 1
    assert len(npys) == 1 and len(jsons) == 1
    captured = result.captures["tone_mapping_pre"]
    saved = np.load(npys[0], allow_pickle=False)
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, captured)
    meta = json.loads(jsons[0].read_text())
    assert meta["diagnostic_schema_version"] == 1
    assert meta["stage"] == "tone_mapping_pre"
    assert meta["signal_domain"] == "tone_mapped_bayer"
    assert meta["transfer"] == "linear"
    assert meta["value_units"] == "normalized"
    assert meta["cfa_pattern"] == cfg["sensor_info"]["bayer_pattern"]
    assert meta["tone_mapping"]["operator"] == "hable"
    assert meta["tone_mapping"]["backend"] == "float"
    assert meta["tone_mapping"]["enabled"] is True
    assert meta["reference_white"] == 1.0
    assert meta["shape"] == list(captured.shape)
    assert meta["dtype"] == str(captured.dtype)
    art = {p.name for p in result.artifacts}
    assert npys[0].name in art and jsons[0].name in art


def test_native_tmo_diagnostics_repeat_non_overwrite(tmp_path: Path, monkeypatch) -> None:
    import json
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    monkeypatch.chdir(tmp_path)
    cfg = config("hable", is_save=True)
    cfg["platform"]["render_3a"] = True
    frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 4096
    out_dir = tmp_path / "diagnostics"
    options = ProcessingOptions(
        output_dir=out_dir, diagnostics=True, capture_stages=("tone_mapping_pre",))
    first = process_frame(frame, cfg, options=options)
    npys0 = sorted(out_dir.rglob("*.npy"))
    first_bytes = {p: (p.read_bytes(), p.with_suffix(".json").read_bytes())
                   for p in npys0}
    second = process_frame(frame, cfg, options=options)
    npys = sorted(out_dir.rglob("*.npy"))
    jsons = sorted(out_dir.rglob("*.json"))
    assert len(npys) == 2 and len(jsons) == 2
    assert npys[0].name != npys[1].name
    for p, (nb, jb) in first_bytes.items():
        assert p.read_bytes() == nb and p.with_suffix(".json").read_bytes() == jb
    np.testing.assert_array_equal(
        np.load(npys[0], allow_pickle=False), first.captures["tone_mapping_pre"])
    np.testing.assert_array_equal(
        np.load(npys[1], allow_pickle=False), second.captures["tone_mapping_pre"])


def test_native_tmo_diagnostics_gating(tmp_path: Path, monkeypatch) -> None:
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    monkeypatch.chdir(tmp_path)
    frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 4096

    cfg = config("hable", is_save=True)
    d0 = tmp_path / "d0"
    process_frame(frame, cfg, options=ProcessingOptions(
        output_dir=d0, diagnostics=False, save_outputs=False))
    assert not d0.exists()
    d0b = tmp_path / "d0b"
    process_frame(frame, cfg, options=ProcessingOptions(
        output_dir=d0b, diagnostics=False, save_outputs=True))
    assert list(d0b.glob("*.png")) or list(d0b.glob("*.tiff"))
    assert not (d0b / "module_output").exists()

    cfg2 = config("hable")
    d1 = tmp_path / "d1"
    process_frame(frame, cfg2, options=ProcessingOptions(
        output_dir=d1, diagnostics=True))
    assert not list(d1.rglob("*.npy")) and not list(d1.rglob("*.json")) \
        and not list(d1.rglob("*.png"))

    cfg3 = config("hable", is_save=True, is_plot_curve=True)
    cfg3["tone_mapping"]["is_enable"] = False
    d2 = tmp_path / "d2"
    process_frame(frame, cfg3, options=ProcessingOptions(
        output_dir=d2, diagnostics=True))
    assert not list(d2.rglob("*.npy")) and not list(d2.rglob("*.json")) \
        and not list(d2.rglob("*.png"))


def test_native_tmo_diagnostics_post_ccm_domain(tmp_path: Path, monkeypatch) -> None:
    import json
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    monkeypatch.chdir(tmp_path)
    cfg = config("hable", is_save=True)
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = False
    frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 4096
    out_dir = tmp_path / "diag"
    result = process_frame(frame, cfg, options=ProcessingOptions(
        output_dir=out_dir, diagnostics=True, capture_stages=("tone_mapping_post",)))
    npys = list(out_dir.rglob("*.npy"))
    jsons = list(out_dir.rglob("*.json"))
    assert len(npys) == 1 and len(jsons) == 1
    meta = json.loads(jsons[0].read_text())
    assert meta["signal_domain"] == "display_linear_rgb"
    assert meta["cfa_pattern"] is None
    saved = np.load(npys[0], allow_pickle=False)
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, result.captures["tone_mapping_post"])


def test_native_diagnostics_and_final_writer_artifacts(tmp_path: Path, monkeypatch) -> None:
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    monkeypatch.chdir(tmp_path)
    cfg = config("hable", is_save=True)
    frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 4096
    out_dir = tmp_path / "both"
    result = process_frame(frame, cfg, options=ProcessingOptions(
        output_dir=out_dir, diagnostics=True, save_outputs=True))
    names = {p.name for p in result.artifacts}
    npys = [p for p in result.artifacts if p.suffix == ".npy"]
    jsons = [p for p in result.artifacts if p.suffix == ".json"]
    finals = [p for p in result.artifacts
              if p.suffix in (".png", ".tiff", ".yuv", ".jpg", ".yaml")]
    assert len(npys) == 1 and len(jsons) >= 2
    assert any(n.endswith("_effective.yaml") for n in names)
    assert any(n.endswith("_migration.json") for n in names)
    assert finals and all(p.exists() for p in result.artifacts)


@pytest.mark.parametrize("selector", ("hable", "hable_integer", "reinhard_integer"))
def test_unsigned_rgb_codes_are_validated_before_luminance_conversion(selector: str) -> None:
    cfg = config(selector)
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = False
    image = np.array([[[65536, 0, 0]]], np.uint32)
    with pytest.raises(ValueError, match="exceed"):
        ToneMapping(image, context(cfg)).execute(False, False)


@pytest.mark.parametrize("selector", ("hable", "hable_integer"))
def test_floating_rgb_headroom_remains_supported(selector: str) -> None:
    cfg = config(selector)
    cfg["tone_mapping"]["tone_mapping_before_demosaic"] = False
    image = np.array([[[65536.0, 0, 0]]], np.float64)
    mapped = ToneMapping(image, context(cfg)).execute(False, False)
    assert mapped.dtype == np.uint16 and mapped.shape == image.shape


@pytest.mark.parametrize("knee,strength,scale", ((1e-200, 1e-200, 1e-200), (1e308, 1e308, 16.0)))
@pytest.mark.parametrize("normalize_output", (False, True))
def test_fixed_ratio_avoids_combined_parameter_underflow_and_overflow(
    knee: float, strength: float, scale: float, normalize_output: bool,
) -> None:
    source = np.array([[0, 1, 1024, 4095]], np.uint16)
    spec = ToneMappingSpec(
        "reinhard", backend="fixed", domain_scale=scale,
        normalize_output=normalize_output,
        parameters=(("knee", knee), ("strength", strength)),
    )
    actual = map_luminance(source, input_max=4095, output_max=65535, config=spec)
    x = np.minimum(source.astype(np.float64) / 4095 * scale, 16)
    white = knee / strength
    expected = x / (x + white)
    if normalize_output:
        reference_x = min(scale, 16)
        expected /= reference_x / (reference_x + white)
    np.testing.assert_allclose(actual, expected * 65535, rtol=0, atol=1)




def test_metadata_only_collision_retries_and_preserves(tmp_path: Path) -> None:
    import json
    (tmp_path / "frame_tone_mapping.json").write_text("foreign")
    path = diagnostics.save_mapped_output(
        np.arange(4, dtype=np.float32),
        platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
        metadata={"k": 1},
    )
    assert path.name.endswith("_0001.npy")
    assert (tmp_path / "frame_tone_mapping.json").read_text() == "foreign"
    assert path.with_suffix(".json").exists()


def test_metadata_dangling_symlink_retries(tmp_path: Path) -> None:
    import os
    link = tmp_path / "frame_tone_mapping.json"
    os.symlink(tmp_path / "nonexistent_target", link)
    path = diagnostics.save_mapped_output(
        np.arange(4, dtype=np.float32),
        platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
        metadata={"k": 1},
    )
    assert path.name.endswith("_0001.npy")
    assert link.is_symlink()


def test_metadata_json_race_between_opens_retries(tmp_path: Path, monkeypatch) -> None:
    real_open = Path.open
    injected = {"done": False}

    def fake_open(self, mode="r", *a, **k):
        if (mode.startswith("x") and self.suffix == ".json"
                and not injected["done"]):
            self.write_text("raced")
            injected["done"] = True
            raise FileExistsError(self.name)
        return real_open(self, mode, *a, **k)

    monkeypatch.setattr(Path, "open", fake_open)
    path = diagnostics.save_mapped_output(
        np.arange(4, dtype=np.float32),
        platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
        metadata={"k": 1},
    )
    assert path.name.endswith("_0001.npy")
    assert (tmp_path / "frame_tone_mapping.json").read_text() == "raced"


def test_failed_npy_save_with_metadata_cleans_own(tmp_path: Path, monkeypatch) -> None:
    foreign = tmp_path / "keep.json"
    foreign.write_text("x")
    monkeypatch.setattr(
        np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("io")))
    with pytest.raises(OSError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
            metadata={"k": 1},
        )
    assert [p.name for p in tmp_path.iterdir()] == ["keep.json"]


def test_metadata_nan_rejected_before_any_io(tmp_path: Path) -> None:
    child = tmp_path / "does_not_exist_yet"
    with pytest.raises(ValueError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(child)}, input_name="frame.raw",
            metadata={"bad": float("nan")},
        )
    assert not child.exists()


def test_json_write_failure_cleans_own_files(tmp_path: Path, monkeypatch) -> None:
    class BoomWriter:
        def write(self, s):
            raise OSError("json io")
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def close(self):
            pass
    real_open = Path.open
    def fake_open(self, mode="r", *a, **k):
        if str(mode).startswith("x") and self.suffix == ".json":
            return cast(Any, BoomWriter())
        return real_open(self, mode, *a, **k)
    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(OSError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
            metadata={"k": 1},
        )
    assert not list(tmp_path.iterdir())


def test_plot_curve_exclusive_and_failed_cleanup(tmp_path: Path, monkeypatch) -> None:
    from modules.tone_mapping.registry import resolve_tone_mapping
    sentinel = tmp_path / "tone_curve_hable_float.png"
    sentinel.write_bytes(b"old")
    spec = resolve_tone_mapping(config("hable", is_plot_curve=True)).spec
    path = diagnostics.plot_curve(spec, {"output_dir": str(tmp_path)})
    assert path is not None and path.name != sentinel.name
    assert sentinel.read_bytes() == b"old"
    assert path.read_bytes()[:4] == b"\x89PNG"
    def boom(*a, **k):
        raise OSError("fig")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig", boom)
    with pytest.raises(OSError):
        diagnostics.plot_curve(spec, {"output_dir": str(tmp_path)})
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert after == before




def _wrap_json_open(monkeypatch, hook):
    """Proxy Path.open for .json exclusives; hook(fh)->fh injected after
    the real exclusive open so the file truly exists on disk."""
    real_open = Path.open

    def fake_open(self, mode="r", *a, **k):
        fh = real_open(self, mode, *a, **k)
        if str(mode).startswith("x") and self.suffix == ".json":
            return hook(fh)
        return fh

    monkeypatch.setattr(Path, "open", fake_open)


def test_json_open_permission_error_rolls_back_npy(tmp_path: Path, monkeypatch) -> None:
    sentinel = tmp_path / "sentinel.npy"
    sentinel.write_bytes(b"keep")
    called = {"save": 0}
    real_save = np.save
    def spy(*a, **k):
        called["save"] += 1
        return real_save(*a, **k)
    monkeypatch.setattr(np, "save", spy)

    real_open = Path.open

    def deny(self, mode="r", *a, **k):
        if str(mode).startswith("x") and self.suffix == ".json":
            raise PermissionError("denied")
        return real_open(self, mode, *a, **k)
    monkeypatch.setattr(Path, "open", deny)

    with pytest.raises(PermissionError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
            metadata={"k": 1},
        )
    assert called["save"] == 0
    assert [p.name for p in tmp_path.iterdir()] == ["sentinel.npy"]
    assert sentinel.read_bytes() == b"keep"


def test_json_close_error_cleans_both_owned(tmp_path: Path, monkeypatch) -> None:
    class CloseBoom:
        def __init__(self, fh):
            self.fh = fh
        def write(self, s):
            return self.fh.write(s)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.fh.close()
            raise OSError("close")
        def close(self):
            self.fh.close()

    _wrap_json_open(monkeypatch, lambda fh: cast(Any, CloseBoom(fh)))
    with pytest.raises(OSError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
            metadata={"k": 1},
        )
    assert not list(tmp_path.iterdir())


def test_npy_close_error_cleans_both_owned(tmp_path: Path, monkeypatch) -> None:
    real_open = Path.open

    class NpyCloseBoom:
        def __init__(self, fh):
            self.fh = fh
        def write(self, *a):
            return self.fh.write(*a)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.fh.close()
            raise OSError("npy close")
        def close(self):
            self.fh.close()

    def fake_open(self, mode="r", *a, **k):
        fh = real_open(self, mode, *a, **k)
        if str(mode).startswith("x") and self.suffix == ".npy":
            return cast(Any, NpyCloseBoom(fh))
        return fh

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(OSError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
            metadata={"k": 1},
        )
    assert not list(tmp_path.iterdir())


def test_write_phase_fileexists_reraises_no_retry(tmp_path: Path, monkeypatch) -> None:
    real_open = Path.open
    opens = {"json": 0}

    class LateBoom:
        def __init__(self, fh):
            self.fh = fh
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.fh.close()
            raise FileExistsError("late race")
        def write(self, s):
            return self.fh.write(s)
        def close(self):
            self.fh.close()

    def fake_open(self, mode="r", *a, **k):
        fh = real_open(self, mode, *a, **k)
        if str(mode).startswith("x") and self.suffix == ".json":
            opens["json"] += 1
            return cast(Any, LateBoom(fh))
        return fh

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(FileExistsError):
        diagnostics.save_mapped_output(
            np.arange(4, dtype=np.float32),
            platform={"output_dir": str(tmp_path)}, input_name="frame.raw",
            metadata={"k": 1},
        )
    assert opens["json"] == 1
    assert not list(tmp_path.iterdir())
