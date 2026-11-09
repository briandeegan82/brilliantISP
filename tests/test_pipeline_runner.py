from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from brilliant_isp import BrilliantISP
from util.pipeline_runner import process_frame
from util.pipeline_types import (
    ProcessingCancelled,
    ProcessingOptions,
    ProcessingState,
)

REPO = Path(__file__).resolve().parents[1]
SVS = REPO / "config" / "SVS_cam.yml"


def _fast_config() -> dict:
    cfg = yaml.safe_load(SVS.read_text())
    cfg["sensor_info"].update(width=64, height=48, hdr_bit_depth=12)
    for sec in (
        "dead_pixel_correction",
        "bayer_noise_reduction",
        "lens_shading_correction",
        "auto_white_balance",
        "auto_exposure",
        "black_level_correction",
        "companding",
        "oecf",
        "ldci",
        "sharpen",
        "2d_noise_reduction",
        "scale",
        "crop",
        "tone_mapping",
        "digital_gain",
        "color_correction_matrix",
    ):
        if isinstance(cfg.get(sec), dict):
            cfg[sec]["is_enable"] = False
    cfg["platform"].update(
        render_3a=False,
        plot_histograms=False,
        debug_enabled=False,
        short_output_names=True,
        disable_progress_bar=True,
    )
    cfg["yuv_conversion_format"]["is_enable"] = False
    return cfg


def _frame() -> np.ndarray:
    return (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) * 16 % 4096).astype(np.uint16)


def test_compute_only_no_files(tmp_path: Path) -> None:
    options = ProcessingOptions(output_dir=tmp_path / "nonexistent_sub")
    before = set(tmp_path.iterdir())
    result = process_frame(_frame(), _fast_config(), options=options)
    assert result.preview_rgb8.shape == (48, 64, 3)
    assert result.preview_rgb8.dtype == np.uint8
    assert len(np.unique(result.preview_rgb8)) > 8
    assert result.native_image.shape == (48, 64, 3)
    assert set(tmp_path.iterdir()) == before
    assert not (tmp_path / "nonexistent_sub").exists()


def test_compute_only_writes_nothing_in_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = process_frame(_frame(), _fast_config())
    assert result.preview_rgb8.size > 0
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftovers == []


def test_save_outputs_requires_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        BrilliantISP("", _fast_config(), outFileName="",
                     options=ProcessingOptions(save_outputs=True))
    with pytest.raises(ValueError, match="output_dir"):
        BrilliantISP("", _fast_config(), outFileName="",
                     options=ProcessingOptions(diagnostics=True))


def test_save_outputs_writes_requested_dir(tmp_path: Path) -> None:
    shared = Path("out_frames")
    before = set(shared.iterdir()) if shared.is_dir() else set()
    out = tmp_path / "req_out"
    options = ProcessingOptions(save_outputs=True, output_dir=out)
    result = process_frame(_frame(), _fast_config(), options=options)
    assert out.is_dir()
    produced = [p for p in out.rglob("*") if p.is_file()]
    assert produced, "expected saved artifacts under requested output_dir"
    after = set(shared.iterdir()) if shared.is_dir() else set()
    assert after == before, "save wrote into the shared out_frames directory"
    assert result.preview_rgb8.size > 0


def test_config_and_source_unchanged() -> None:
    cfg = _fast_config()
    snapshot = copy.deepcopy(cfg)
    frame = _frame()
    frame_snapshot = frame.copy()
    process_frame(frame, cfg)
    assert cfg == snapshot
    np.testing.assert_array_equal(frame, frame_snapshot)


def test_blc_pwc_conflict_fails_before_frame_processing() -> None:
    cfg = _fast_config()
    cfg["black_level_correction"].update(is_enable=True, is_linear=False)
    cfg["companding"].update(is_enable=True, pre_linearization_black_level=16)
    with pytest.raises(ValueError, match="companding.pre_linearization_black_level"):
        process_frame(_frame(), cfg)


def test_array_shape_must_match_declared_geometry() -> None:
    with pytest.raises(ValueError, match="declared"):
        process_frame(np.zeros((10, 10), np.uint16), _fast_config())
    with pytest.raises(ValueError, match="2-D"):
        process_frame(np.zeros((48, 64, 3), np.uint16), _fast_config())


def test_independent_repeats_identical() -> None:
    cfg = _fast_config()
    r1 = process_frame(_frame(), cfg)
    r2 = process_frame(_frame(), cfg)
    np.testing.assert_array_equal(r1.preview_rgb8, r2.preview_rgb8)
    np.testing.assert_array_equal(r1.native_image, r2.native_image)


def test_feedback_does_not_mutate_manual_config() -> None:
    cfg = _fast_config()
    cfg["digital_gain"]["is_enable"] = True
    cfg["digital_gain"]["is_auto"] = False
    cfg["digital_gain"]["current_gain"] = 3
    result = process_frame(_frame(), cfg)
    assert cfg["digital_gain"]["current_gain"] == 3
    assert "digital_gain_index" in result.feedback
    assert "awb_gains" in result.feedback


def test_sequence_state_ignored_for_manual_dg() -> None:
    cfg = _fast_config()
    cfg["digital_gain"]["is_enable"] = True
    cfg["digital_gain"]["is_auto"] = False
    state = ProcessingState(digital_gain_index=0, previous_awb_gains=(5.0, 5.0))
    result = process_frame(_frame(), cfg, state=state)
    assert cfg["digital_gain"]["current_gain"] == 10
    assert state.digital_gain_index == 0


def test_capture_stage_owned_copy() -> None:
    frame = _frame()
    options = ProcessingOptions(capture_stages=("demosaic", "white_balance"))
    result = process_frame(frame, _fast_config(), options=options)
    assert set(result.captures) == {"demosaic", "white_balance"}
    demosaic = result.captures["demosaic"]
    assert demosaic.shape == (48, 64, 3)
    assert not demosaic.flags.writeable
    wb = result.captures["white_balance"]
    assert wb.shape == (48, 64)
    assert not np.shares_memory(demosaic, frame)
    assert not np.shares_memory(demosaic, result.native_image)
    assert not np.shares_memory(wb, frame)


def test_telemetry_order_status_elapsed() -> None:
    events: list = []
    options = ProcessingOptions(progress_callback=events.append)
    result = process_frame(_frame(), _fast_config(), options=options)
    started = [e for e in events if e.status == "started"]
    assert started, "expected started events through callback"
    assert all(e.run_id == options.run_id for e in started)
    stage_ids = [e.stage_id for e in result.timings]
    assert stage_ids.index("crop") < stage_ids.index("demosaic") < stage_ids.index(
        "scale"
    )
    assert all(e.status == "completed" for e in result.timings)
    assert all(e.elapsed_ns >= 0 for e in result.timings)
    assert all(e.pass_index == 1 for e in result.timings)


def test_failed_stage_records_event() -> None:
    cfg = _fast_config()
    cfg["crop"]["is_enable"] = True
    cfg["crop"].update(crop_x_start=0, crop_y_start=0, new_width=10, new_height=10)
    cfg["demosaic"]["algorithm"] = "nonexistent_algorithm"
    events: list = []
    options = ProcessingOptions(progress_callback=events.append)
    with pytest.raises(Exception):
        process_frame(_frame(), cfg, options=options)
    failed = [e for e in events if e.status == "failed"]
    assert failed and failed[0].stage_id == "demosaic"
    assert failed[0].error


class _Token:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set


def test_cancellation_between_stages() -> None:
    token = _Token()
    seen: list = []

    def flip(event) -> None:
        seen.append(event.stage_id)
        if event.stage_id == "crop" and event.status == "completed":
            token._set = True

    options = ProcessingOptions(cancellation_token=token, progress_callback=flip)
    state = ProcessingState()
    with pytest.raises(ProcessingCancelled):
        process_frame(_frame(), _fast_config(), state=state, options=options)
    assert state.ae_history == ()
    assert state.previous_awb_gains is None


def test_no_state_update_on_failure() -> None:
    cfg = _fast_config()
    cfg["crop"]["is_enable"] = True
    cfg["crop"].update(crop_x_start=0, crop_y_start=0, new_width=8, new_height=8)
    cfg["demosaic"]["algorithm"] = "nonexistent"
    state = ProcessingState()
    with pytest.raises(Exception):
        process_frame(_frame(), cfg, state=state)
    assert state.ae_history == ()


def test_result_warnings_include_notices() -> None:
    result = process_frame(_frame(), _fast_config())
    assert any("versionless" in w for w in result.warnings)


def test_metadata_truthful_domains() -> None:
    result = process_frame(_frame(), _fast_config())
    meta = result.native_metadata
    assert meta.signal_domain == "display_encoded_rgb"
    assert meta.transfer == "srgb"
    assert meta.cfa_pattern is None
    assert meta.storage_bit_depth == 8
    assert meta.code_bit_depth == 8
    assert (meta.width, meta.height) == (64, 48)
    assert meta.source_identity.startswith("array:")


def test_file_source_matches_array_source(tmp_path: Path) -> None:
    cfg = _fast_config()
    raw = tmp_path / "tiny.raw"
    _frame().astype("<u2").tofile(raw)
    cfg_file = copy.deepcopy(cfg)
    cfg_file["sensor_info"]["endian_type"] = "ieee-le"
    file_result = process_frame(raw, cfg_file)
    array_result = process_frame(_frame(), cfg)
    np.testing.assert_array_equal(
        file_result.preview_rgb8, array_result.preview_rgb8
    )
    assert file_result.native_metadata.source_identity == str(raw.resolve())
    other = process_frame(_frame(), cfg)
    assert array_result.native_metadata.source_identity != (
        other.native_metadata.source_identity
    )
    assert array_result.native_metadata.source_identity.startswith("array:")


def test_run_pipeline_visualize_false_returns_image(tmp_path: Path) -> None:
    cfg = _fast_config()
    raw_dir = tmp_path
    raw = raw_dir / "tiny.raw"
    _frame().astype("<u2").tofile(raw)
    cfg["sensor_info"]["endian_type"] = "ieee-le"
    isp = BrilliantISP(str(raw_dir), cfg, outFileName="", output_path=str(raw_dir))
    isp.raw = _frame()
    isp.in_file = isp.out_file = "tiny"
    isp.run_pipeline(visualize_output=False)
    assert isp.last_output_rgb is not None
    assert isp.last_output_rgb.shape == (48, 64, 3)
    assert isp.last_output_native is not None


def test_legacy_save_still_writes(tmp_path: Path) -> None:
    cfg = _fast_config()
    cfg["sensor_info"]["endian_type"] = "ieee-le"
    raw = tmp_path / "tiny.raw"
    _frame().astype("<u2").tofile(raw)
    out = tmp_path / "out"
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="", output_path=str(out))
    isp.execute(img_path="tiny.raw")
    assert isp.last_output_rgb is not None
    assert any(out.glob("*.png"))
    assert any(out.glob("*.yaml"))


def test_shared_helpers_spy(tmp_path: Path, monkeypatch) -> None:
    cfg = _fast_config()
    raw = tmp_path / "tiny.raw"
    _frame().astype("<u2").tofile(raw)
    cfg["sensor_info"]["endian_type"] = "ieee-le"
    isp = BrilliantISP(str(tmp_path), cfg, outFileName="", output_path=str(tmp_path))
    isp.raw = _frame()
    calls: list[str] = []
    orig_pre = isp._run_pre_digital_gain
    orig_gain = isp._run_gain_through_white_balance

    def spy_pre():
        calls.append("pre")
        return orig_pre()

    def spy_gain(oecf, **kw):
        calls.append("gain")
        return orig_gain(oecf, **kw)

    monkeypatch.setattr(isp, "_run_pre_digital_gain", spy_pre)
    monkeypatch.setattr(isp, "_run_gain_through_white_balance", spy_gain)
    isp.run_pipeline(visualize_output=False)
    assert calls == ["pre", "gain"]
    calls.clear()
    monkeypatch.setattr(isp, "_run_gain_through_white_balance", spy_gain)
    isp.run_pipeline_up_to_wb()
    assert calls == ["pre", "gain"]
    assert isp.current_frame is not None


def test_up_to_wb_equivalent_to_full_wb() -> None:
    cfg = _fast_config()
    cfg["digital_gain"].update(is_enable=True, is_auto=False, gain_array=[1.5], current_gain=0)
    cfg["white_balance"].update(is_enable=True, is_auto=False, r_gain=2.0, b_gain=0.5)
    isp = BrilliantISP("", cfg, outFileName="",
                       options=ProcessingOptions())
    isp.raw = _frame()
    isp.run_pipeline(visualize_output=False)
    assert isp.current_frame is not None
    full_wb = isp.current_frame.copy()
    isp2 = BrilliantISP("", cfg, outFileName="",
                        options=ProcessingOptions())
    isp2.raw = _frame()
    isp2.run_pipeline_up_to_wb()
    assert isp2.current_frame is not None
    np.testing.assert_array_equal(isp2.current_frame, full_wb)


def test_yuv_packing_write_output_flag(tmp_path: Path) -> None:
    cfg = _fast_config()
    cfg["yuv_conversion_format"]["is_enable"] = True
    cfg["rgb_conversion"]["is_enable"] = False
    out = tmp_path / "packed"
    result = process_frame(
        _frame(),
        cfg,
        options=ProcessingOptions(save_outputs=True, output_dir=out),
    )
    packed = list(out.rglob("*.yuv"))
    assert packed, "expected packed YUV artifact when save_outputs=True"
    result2_dir = tmp_path / "compute"
    shared = Path("out_frames")
    before = set(shared.iterdir()) if shared.is_dir() else set()
    process_frame(
        _frame(),
        cfg,
        options=ProcessingOptions(output_dir=result2_dir),
    )
    assert not result2_dir.exists()
    after = set(shared.iterdir()) if shared.is_dir() else set()
    assert after == before, "compute-only run wrote into shared out_frames"


def test_mapping_config_not_mutated() -> None:
    cfg = _fast_config()
    isp = BrilliantISP("", cfg, outFileName="", options=ProcessingOptions())
    assert isp.config_path == "<mapping>"
    assert "output_dir" not in cfg["platform"]


def test_explicit_save_writes_only_final_pass(tmp_path: Path, monkeypatch) -> None:
    import util.utils as utils_mod

    calls: list = []
    monkeypatch.setattr(
        utils_mod,
        "save_pipeline_output",
        lambda *a, **k: calls.append(a),
    )
    cfg = _fast_config()
    isp = BrilliantISP(
        "",
        cfg,
        outFileName="",
        options=ProcessingOptions(
            save_outputs=True, output_dir=tmp_path
        ),
    )
    isp.raw = _frame()
    isp.in_file = isp.out_file = "tiny"
    import util.output_writer as ow
    monkeypatch.setattr(
        ow, "write_output",
        lambda *a, **k: calls.append(a) or [],
    )
    isp.options = ProcessingOptions(save_outputs=False, output_dir=tmp_path)
    isp.run_pipeline(visualize_output=False)
    assert calls == [], "compute-only wrote output"
    isp.options = ProcessingOptions(save_outputs=True, output_dir=tmp_path)
    isp.run_pipeline(visualize_output=True)
    assert len(calls) == 1, "expected exactly one final-pass save"


def test_diagnostics_without_save_no_final_output(tmp_path: Path, monkeypatch) -> None:
    import util.utils as utils_mod

    calls: list = []
    monkeypatch.setattr(
        utils_mod,
        "save_pipeline_output",
        lambda *a, **k: calls.append(a),
    )
    cfg = _fast_config()
    cfg["platform"]["plot_histograms"] = True
    isp = BrilliantISP(
        "",
        cfg,
        outFileName="",
        options=ProcessingOptions(diagnostics=True, output_dir=tmp_path),
    )
    isp.raw = _frame()
    isp.in_file = isp.out_file = "tiny"
    isp.run_pipeline(visualize_output=True)
    assert calls == [], "diagnostics must not trigger final-output writes"


def test_compute_only_neutralizes_debug_file_logging(tmp_path: Path) -> None:
    canary = tmp_path / "debug_canary.log"
    cfg = _fast_config()
    cfg["platform"]["debug_enabled"] = True
    cfg["platform"]["debug_log_file"] = str(canary)
    process_frame(_frame(), cfg)
    assert not canary.exists()


def test_fractional_geometry_rejected(tmp_path: Path) -> None:
    cfg = _fast_config()
    cfg["sensor_info"]["width"] = 64.5
    with pytest.raises(ValueError, match="sensor_info.width"):
        process_frame(_frame(), cfg)
    cfg2 = _fast_config()
    cfg2["sensor_info"]["width"] = "64"
    with pytest.raises(ValueError, match="sensor_info.width"):
        process_frame(_frame(), cfg2)


def test_unsupported_array_dtype_rejected() -> None:
    for dtype in (np.int16, np.float32, np.float64, object):
        with pytest.raises(ValueError, match="dtype"):
            process_frame(
                np.zeros((48, 64), dtype=dtype), _fast_config()
            )


def test_final_stage_cancellation_no_state_update() -> None:
    events: list = []

    def cancel_at_end(event) -> None:
        events.append(event.stage_id)
        if event.stage_id == "scale" and event.status == "completed":
            token._set = True

    token = _Token()
    state = ProcessingState(previous_awb_gains=(1.0, 1.0))
    options = ProcessingOptions(
        cancellation_token=token, progress_callback=cancel_at_end
    )
    with pytest.raises(ProcessingCancelled):
        process_frame(_frame(), _fast_config(), state=state, options=options)
    assert state.ae_history == ()
    assert state.previous_awb_gains == (1.0, 1.0)


def test_options_snapshot_isolated_from_caller() -> None:
    cfg = _fast_config()
    options = ProcessingOptions()
    original_run_id = options.run_id
    escaped = Path("/nowhere")

    def mutate_during_crop(event) -> None:
        if event.stage_id == "crop" and event.status == "started":
            options.save_outputs = True
            options.output_dir = escaped
            options.capture_stages = ("demosaic",)
            options.run_id = "mutated-run-id"

    options.progress_callback = mutate_during_crop
    result = process_frame(_frame(), cfg, options=options)
    assert result.timings
    assert all(e.run_id == original_run_id for e in result.timings)
    assert result.captures == {}
    assert not escaped.exists()


def test_ycbcr_metadata_when_rgb_disabled(tmp_path: Path) -> None:
    cfg = _fast_config()
    cfg["rgb_conversion"]["is_enable"] = False
    result = process_frame(_frame(), cfg)
    meta = result.native_metadata
    assert meta.cfa_pattern is None
    assert meta.signal_domain == "encoded_ycbcr"

    cfg_packed = _fast_config()
    cfg_packed["rgb_conversion"]["is_enable"] = False
    cfg_packed["yuv_conversion_format"]["is_enable"] = True
    packed = process_frame(_frame(), cfg_packed)
    assert packed.native_metadata.signal_domain == "encoded_ycbcr"
    assert packed.native_metadata.cfa_pattern is None


def test_diagnostics_log_redirected_to_request_dir(tmp_path: Path) -> None:
    external = tmp_path / "external_debug.log"
    cfg = _fast_config()
    cfg["platform"]["debug_log_file"] = str(external)
    cfg["platform"]["debug_enabled"] = True
    out = tmp_path / "req"
    process_frame(
        _frame(),
        cfg,
        options=ProcessingOptions(diagnostics=True, output_dir=out),
    )
    assert not external.exists()
    assert (out / "pipeline.log").exists()
    assert cfg["platform"]["debug_log_file"] == str(external)


def test_ae_history_bounded_and_updated_on_success() -> None:
    cfg = _fast_config()
    cfg["auto_exposure"]["is_enable"] = True
    state = ProcessingState(ae_history=tuple(range(40)))
    result = process_frame(_frame(), cfg, state=state)
    assert len(state.ae_history) == 32
    assert state.ae_history[-1] == result.feedback["ae_feedback"]


def test_state_updates_only_on_success() -> None:
    cfg = _fast_config()
    cfg["auto_exposure"]["is_enable"] = True
    state = ProcessingState()
    result = process_frame(_frame(), cfg, state=state)
    assert len(state.ae_history) == 1
    assert state.ae_history[0] == result.feedback["ae_feedback"]


def test_up_to_wb_repeated_runs_same_result() -> None:
    cfg = _fast_config()
    cfg["crop"]["is_enable"] = True
    cfg["crop"].update(crop_x_start=0, crop_y_start=0, new_width=16, new_height=16)
    isp = BrilliantISP("", cfg, outFileName="", options=ProcessingOptions())
    isp.raw = _frame()
    isp.run_pipeline_up_to_wb()
    assert isp.current_frame is not None
    first = isp.current_frame.copy()
    isp.run_pipeline_up_to_wb()
    assert isp.current_frame is not None
    np.testing.assert_array_equal(isp.current_frame, first)
    assert isp.pass_index == 2
