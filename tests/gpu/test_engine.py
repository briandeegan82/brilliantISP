from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from brilliantisp_gpu.api import process_frame_gpu
from brilliantisp_gpu.types import GpuOptions
from tools.gpu_validation.contracts import (
    DEMOSAIC_ALGORITHMS, FINAL_FLOAT, FINAL_INTEGER, STATISTICS, TONE_PAIRINGS,
)
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.pipeline import _compare_record, observe_gpu_stages
from tools.gpu_validation.reference import jsonable, observe_cpu_stages
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingCancelled, ProcessingOptions, ProcessingState


REPO = Path(__file__).resolve().parents[2]


def _config():
    cfg = yaml.safe_load((REPO / "config" / "SVS_cam.yml").read_text())
    cfg["schema_version"] = 2
    cfg["sensor_info"].update(
        height=8, width=10, bit_depth=12, hdr_bit_depth=24,
        bayer_pattern="rggb", repair_zero_green=False,
    )
    for name in (
        "digital_gain", "lens_shading_correction", "bayer_noise_reduction",
        "black_level_correction", "white_balance", "auto_white_balance",
        "color_correction_matrix", "color_saturation_enhancement", "ldci",
        "sharpen", "2d_noise_reduction", "scale", "companding",
        "dead_pixel_correction", "oecf", "crop", "auto_exposure",
        "gamma_correction", "color_space_conversion", "rgb_conversion",
        "tone_mapping", "yuv_conversion_format",
    ):
        if isinstance(cfg.get(name), dict):
            cfg[name]["is_enable"] = False
    cfg["platform"].update(
        render_3a=False, plot_histograms=False, debug_enabled=False,
        disable_progress_bar=True,
    )
    cfg["demosaic"].update(is_enable=True, algorithm="malvar")
    cfg["output"] = dict(model="rgb", transfer="linear", format="tiff", bit_depth=32)
    return cfg


def _source(cfg):
    sensor = cfg["sensor_info"]
    shape = int(sensor["height"]), int(sensor["width"])
    return np.random.RandomState(11301).randint(0, 4096, size=shape).astype(np.uint16)


def _assert_arrays(reference, candidate, gate, *, data_range=1.0):
    result = compare_arrays(reference, candidate, gate, data_range=data_range)
    assert result["passed"], result


def _compare_engine(tmp_path, device, cfg, source=None, state=None):
    source = _source(cfg) if source is None else source
    original_cfg, original_source = copy.deepcopy(cfg), source.copy()
    cpu_state, gpu_state = copy.deepcopy(state), copy.deepcopy(state)
    cpu_dir, gpu_dir = tmp_path / "cpu", tmp_path / "gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    options = ProcessingOptions(
        run_id="gpu_engine_test",
        capture_stages=("white_balance", "demosaic", "scale", "output"),
        output_dir=tmp_path / "must_not_exist",
    )
    with observe_cpu_stages(cpu_dir) as expected:
        cpu = process_frame(source, cfg, state=cpu_state, options=options)
    with observe_gpu_stages(gpu_dir) as actual:
        gpu = process_frame_gpu(
            source, cfg, state=gpu_state, options=options,
            gpu=GpuOptions(device=str(device)),
        )
    assert cfg == original_cfg
    np.testing.assert_array_equal(source, original_source)
    assert not options.output_dir.exists()
    assert not gpu.artifacts and not cpu.artifacts
    assert gpu.native_metadata == cpu.native_metadata
    assert gpu.effective_config == cpu.effective_config
    assert gpu.migration_report == cpu.migration_report
    assert gpu.warnings == cpu.warnings
    assert gpu.native_image.dtype == cpu.native_image.dtype
    assert gpu.native_image.shape == cpu.native_image.shape
    integer = cpu.native_image.dtype.kind == "u"
    maximum = 2 ** cpu.native_metadata.code_bit_depth - 1 if integer else 1.0
    _assert_arrays(cpu.native_image, gpu.native_image,
                   FINAL_INTEGER if integer else FINAL_FLOAT, data_range=maximum)
    _assert_arrays(cpu.preview_rgb8, gpu.preview_rgb8, FINAL_INTEGER, data_range=255)
    keys = ("stage_id", "pass_index", "ae_pass", "occurrence")
    assert [tuple(e[k] for k in keys) for e in expected] == [
        tuple(e[k] for k in keys) for e in actual
    ]
    for left, right in zip(expected, actual):
        result = _compare_record(
            left, right, cpu_dir, gpu_dir, left["stage_id"],
            cpu.feedback["linear_reference_code_max"],
        )
        assert result["passed"], (left["stage_id"], left["ae_pass"], result)
    assert gpu.feedback.keys() == cpu.feedback.keys()
    for name, value in cpu.feedback.items():
        candidate = gpu.feedback[name]
        if name in ("awb_gains", "ae_meter_average") and value is not None:
            result = compare_arrays(
                np.atleast_1d(value).astype(np.float64),
                np.atleast_1d(candidate).astype(np.float64), STATISTICS,
            )
            assert result["passed"], (name, result)
        else:
            assert jsonable(candidate) == jsonable(value), name
            assert type(candidate) is type(value), name
    if cpu_state is None:
        assert gpu_state is None
    else:
        assert gpu_state.digital_gain_index == cpu_state.digital_gain_index
        assert gpu_state.ae_history == cpu_state.ae_history
        if cpu_state.previous_awb_gains is None:
            assert gpu_state.previous_awb_gains is None
        else:
            result = compare_arrays(
                np.asarray(cpu_state.previous_awb_gains, dtype=np.float64),
                np.asarray(gpu_state.previous_awb_gains, dtype=np.float64),
                STATISTICS,
            )
            assert result["passed"], result
    assert gpu.captures.keys() == cpu.captures.keys()
    for name, capture in gpu.captures.items():
        assert not capture.flags.writeable
        assert capture.shape == cpu.captures[name].shape
        assert capture.dtype == cpu.captures[name].dtype
        assert not np.shares_memory(capture, source)
        assert not np.shares_memory(capture, gpu.native_image)
    assert all(e.status == "completed" and e.elapsed_ns >= 0 for e in gpu.timings)
    assert all(e.pass_index == 1 for e in gpu.timings)
    assert [(e.stage_id, e.pass_index, e.ae_pass, e.shape, e.dtype)
            for e in gpu.timings] == [
        (e.stage_id, e.pass_index, e.ae_pass, e.shape, e.dtype)
        for e in cpu.timings
    ]
    return cpu, gpu, gpu_state


def test_engine_smoke(cuda_device, tmp_path):
    _compare_engine(tmp_path, cuda_device, _config())


@pytest.mark.parametrize("algorithm", DEMOSAIC_ALGORITHMS)
def test_engine_demosaic_variants(cuda_device, tmp_path, algorithm):
    cfg = _config()
    cfg["demosaic"]["algorithm"] = algorithm
    _compare_engine(tmp_path, cuda_device, cfg)


@pytest.mark.parametrize("operator,backend", TONE_PAIRINGS)
@pytest.mark.parametrize("placement", ("post_ccm", "pre_demosaic"))
def test_engine_tone_paths(cuda_device, tmp_path, operator, backend, placement):
    cfg = _config()
    cfg["tone_mapping"] = dict(
        is_enable=True, operator=operator, backend=backend, placement=placement,
        normalize_output=operator != "durand",
    )
    _compare_engine(tmp_path, cuda_device, cfg)


@pytest.mark.parametrize("output", (
    dict(model="rgb", transfer="srgb", format="png", bit_depth=8),
    dict(model="rgb", transfer="power", gamma=2.4, format="png", bit_depth=16),
    dict(model="rgb", transfer="rec709", format="tiff", bit_depth=16),
    dict(model="rgb", transfer="linear", format="tiff", bit_depth=32),
    dict(model="ycbcr", matrix="bt601", bit_depth=8),
    dict(model="ycbcr", matrix="bt709", layout="422", packing="yuyv", bit_depth=8),
    dict(model="ycbcr", matrix="bt601", layout="422", packing="uyvy", bit_depth=16),
))
def test_engine_output_paths(cuda_device, tmp_path, output):
    cfg = _config()
    cfg["output"] = output
    cfg["color_correction_matrix"].update(
        is_enable=True, corrected_red=[-1.0, 0.0, 0.0],
        corrected_green=[0.0, 1.0, 0.0], corrected_blue=[0.0, 0.0, 2.0],
    )
    _compare_engine(tmp_path, cuda_device, cfg)


def test_engine_ae_state_and_manual_config(cuda_device, tmp_path):
    cfg = _config()
    cfg["white_balance"].update(
        is_enable=True, is_auto=False, r_gain=2.0, b_gain=0.5,
    )
    cfg["digital_gain"].update(
        is_enable=True, is_auto=True, gain_array=[1.0, 2.0, 4.0], current_gain=0,
    )
    cfg["auto_exposure"].update(
        is_enable=True, exposure_correction_mode="direct",
        rerun_from_digital_gain=True, target_luminance=0.5,
    )
    state = ProcessingState()
    original = dataclasses.replace(state)
    source = np.full((8, 10), 1024, np.uint16)
    _, gpu, committed = _compare_engine(tmp_path, cuda_device, cfg, source, state)
    assert state == original
    assert gpu.feedback["ae_stop_reason"] == "converged"
    assert gpu.feedback["digital_gain_index"] == 1
    assert gpu.feedback["ae_passes"] == 2
    assert committed.digital_gain_index == 1


@pytest.mark.parametrize("algorithm", ("gray_world", "norm_2", "pca"))
def test_engine_front_half_and_awb(cuda_device, tmp_path, algorithm):
    cfg = _config()
    cfg["crop"].update(
        is_enable=True, crop_x_start=1, crop_y_start=1, new_height=6, new_width=8,
    )
    cfg["dead_pixel_correction"].update(is_enable=True, dp_threshold=64)
    cfg["black_level_correction"].update(
        is_enable=True, is_linear=False,
        r_offset=16, gr_offset=12, gb_offset=8, b_offset=4,
    )
    cfg["sensor_info"]["repair_zero_green"] = True
    cfg["companding"].update(
        is_enable=True, companded_pin=[0, 2048, 4096],
        companded_pout=[0, 65536, 262144],
        pre_linearization_black_level=0, post_linearization_black_level=0,
    )
    cfg["oecf"].update(is_enable=True, r_lut=None)
    cfg["digital_gain"].update(
        is_enable=True, is_auto=False, current_gain=1, gain_array=[1.0, 2.0],
    )
    cfg["lens_shading_correction"].update(
        is_enable=True, r_k1=.1, r_k2=.02, gr_k1=.05, gr_k2=.01,
        gb_k1=.08, gb_k2=.01, b_k1=.12, b_k2=.02,
    )
    cfg["auto_white_balance"].update(
        is_enable=True, algorithm=algorithm, percentage=20,
        underexposed_percentage=.05, overexposed_percentage=.05,
    )
    cfg["white_balance"].update(is_enable=True, is_auto=True)
    source = _source(cfg)
    source[0::2, 1::2] = 0
    source[1::2, 0::2] = 0
    _compare_engine(tmp_path, cuda_device, cfg, source, ProcessingState())


@pytest.mark.parametrize("placement", ("pre_transfer", "post_transfer"))
def test_engine_luma_chain_and_scale(cuda_device, tmp_path, placement):
    cfg = _config()
    cfg["color_saturation_enhancement"].update(is_enable=True, saturation_gain=1.3)
    cfg["ldci"].update(is_enable=True, placement=placement, wind=4, clip_limit=2.0)
    cfg["sharpen"].update(is_enable=True, sharpen_sigma=1.0, sharpen_strength=.5)
    cfg["2d_noise_reduction"].update(
        is_enable=True, window_size=3, patch_size=3, strength=10.0 / 255.0,
    )
    cfg["scale"].update(
        is_enable=True, is_hardware=False, algorithm="Bilinear",
        new_height=5, new_width=7,
    )
    cfg["output"] = dict(model="rgb", transfer="srgb", format="png", bit_depth=8)
    _compare_engine(tmp_path, cuda_device, cfg)


def test_engine_cancellation_does_not_commit_state(cuda_device):
    class Token:
        flag = False

        def is_set(self):
            return self.flag

    token = Token()
    events = []

    def callback(event):
        events.append(event)
        if event.stage_id == "crop" and event.status == "completed":
            token.flag = True

    state = ProcessingState(
        previous_awb_gains=(1.2, 0.8), digital_gain_index=1, ae_history=(0,),
    )
    original = dataclasses.replace(state)
    cfg = _config()
    with pytest.raises(ProcessingCancelled):
        process_frame_gpu(
            _source(cfg), cfg, state=state,
            options=ProcessingOptions(cancellation_token=token,
                                      progress_callback=callback),
            gpu=GpuOptions(device=str(cuda_device)),
        )
    assert state == original
    assert any(e.stage_id == "crop" and e.status == "completed" for e in events)
    assert any(e.status == "cancelled" for e in events)


@pytest.mark.parametrize("algorithm", ("gray_world", "norm_2", "pca"))
def test_awb_rejects_bulk_cpu_transfers(cuda_device, monkeypatch, algorithm):
    import torch
    from brilliantisp_gpu.blocks.auto_white_balance import auto_white_balance
    from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance

    cfg = _config()
    sensor = dict(cfg["sensor_info"], height=16, width=18, hdr_bit_depth=12)
    params = dict(
        cfg["auto_white_balance"], is_enable=True, algorithm=algorithm,
        percentage=20, underexposed_percentage=0.0, overexposed_percentage=0.0,
    )
    white = dict(cfg["white_balance"], is_enable=True, is_auto=True)
    source = np.random.RandomState(11302).randint(
        500, 1501, size=(16, 18),
    ).astype(np.float32)
    oracle = AutoWhiteBalance(source, sensor, params, white)
    expected = np.asarray(oracle.compute_gains(input_gain=1.0), dtype=np.float64)
    image = torch.as_tensor(source[None], device=cuda_device)
    original_cpu, original_to, original_list = (
        torch.Tensor.cpu, torch.Tensor.to, torch.Tensor.tolist,
    )

    def assert_small(tensor):
        assert not tensor.is_cuda or tensor.numel() <= 9, (
            "bulk image/sample transfer to CPU is forbidden", tuple(tensor.shape),
        )

    def audited_cpu(tensor, *args, **kwargs):
        assert_small(tensor)
        return original_cpu(tensor, *args, **kwargs)

    def audited_to(tensor, *args, **kwargs):
        target = kwargs.get("device")
        if target is None and args:
            if isinstance(args[0], (str, torch.device)):
                target = args[0]
            elif isinstance(args[0], torch.Tensor):
                target = args[0].device
        if target is not None and torch.device(target).type == "cpu":
            assert_small(tensor)
        return original_to(tensor, *args, **kwargs)

    def audited_list(tensor, *args, **kwargs):
        assert_small(tensor)
        return original_list(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", audited_cpu)
        patch.setattr(torch.Tensor, "to", audited_to)
        patch.setattr(torch.Tensor, "tolist", audited_list)
        result = auto_white_balance(
            image, sensor, params, white, input_gains=(1.0,),
        )
    assert result.statuses == ("estimated",)
    assert result.sample_counts == (72,)
    comparison = compare_arrays(
        expected, result.gains[0].cpu().numpy(), STATISTICS,
    )
    assert comparison["passed"], comparison


def test_engine_ae_measurement_is_inside_stage(cuda_device, monkeypatch):
    import brilliantisp_gpu.blocks.auto_exposure as block

    original = block.auto_exposure
    active = None
    calls = []

    def callback(event):
        nonlocal active
        active = event.stage_id if event.status == "started" else None

    def measured(*args, **kwargs):
        assert active == "auto_exposure"
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(block, "auto_exposure", measured)
    cfg = _config()
    process_frame_gpu(
        _source(cfg), cfg, options=ProcessingOptions(progress_callback=callback),
        gpu=GpuOptions(device=str(cuda_device)),
    )
    assert calls == [True]


@pytest.mark.parametrize("failure_call", (1, 2))
def test_engine_profile_failure_records_stage(cuda_device, monkeypatch, failure_call):
    import torch
    from brilliantisp_gpu.config import prepare_config
    from brilliantisp_gpu.engine import GpuEngine

    options = ProcessingOptions()
    engine = GpuEngine(
        prepare_config(_config(), options=options), options, None,
        GpuOptions(device=str(cuda_device), profile_stages=True),
    )
    engine.device = cuda_device
    engine.pass_index = 1

    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("profile synchronization failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="profile synchronization failed"):
        engine._run_stage("test_profile", lambda: None)
    assert len(engine.stage_events) == 1
    assert engine.stage_events[0].stage_id == "test_profile"
    assert engine.stage_events[0].status == "failed"
    assert "profile synchronization failed" in engine.stage_events[0].error


@pytest.mark.parametrize("frame_index", (0, 1))
def test_engine_host_result_batch_index(cuda_device, frame_index):
    import torch
    from brilliantisp_gpu.config import prepare_config
    from brilliantisp_gpu.engine import GpuEngine

    options = ProcessingOptions()
    engine = GpuEngine(
        prepare_config(_config(), options=options), options, None,
        GpuOptions(device=str(cuda_device)),
    )
    image = torch.arange(24, dtype=torch.float32, device=cuda_device).reshape(2, 3, 4)
    actual = engine._host_result(image, frame_index=frame_index)
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[frame_index]
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.owndata


@pytest.mark.parametrize("placement", ("post_ccm", "pre_demosaic"))
def test_engine_diagnostics_dtype_and_pixels(cuda_device, tmp_path, placement):
    cfg = _config()
    cfg["tone_mapping"] = dict(
        is_enable=True, operator="reinhard", backend="float",
        placement=placement, normalize_output=True,
        is_save=True, is_plot_curve=False,
    )
    source = _source(cfg)
    cpu = process_frame(
        source, cfg,
        options=ProcessingOptions(diagnostics=True, output_dir=tmp_path / "cpu"),
    )
    gpu = process_frame_gpu(
        source, cfg,
        options=ProcessingOptions(diagnostics=True, output_dir=tmp_path / "gpu"),
        gpu=GpuOptions(device=str(cuda_device)),
    )
    left = [p for p in cpu.artifacts if p.suffix == ".npy"]
    right = [p for p in gpu.artifacts if p.suffix == ".npy"]
    assert len(left) == len(right) == 1
    expected, actual = np.load(left[0]), np.load(right[0])
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    _assert_arrays(expected, actual, FINAL_FLOAT)
