from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from brilliantisp_gpu.config import prepare_config
from brilliantisp_gpu.engine import GpuEngine
from brilliantisp_gpu.microbatch import commit_frame, execute_group
from brilliantisp_gpu.sources import decode_source
from brilliantisp_gpu.types import GpuOptions
from tools.gpu_validation.contracts import (
    CODE_FLOAT_STAGES, DEMOSAIC_ALGORITHMS, FINAL_FLOAT, FINAL_INTEGER,
    INTEGER_STAGES, NORMALIZED_STAGES, STATISTICS, TONE_PAIRINGS, stage_gate,
)
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.reference import jsonable
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingOptions


REPO = Path(__file__).resolve().parents[2]
CAPTURES = tuple(sorted(INTEGER_STAGES | CODE_FLOAT_STAGES | NORMALIZED_STAGES | {"output"}))


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


def _sources(count):
    data = np.random.RandomState(11303).randint(
        0, 4096, size=(count, 8, 10),
    ).astype(np.uint16)
    data[0] = 0
    for i in range(1, count):
        data[i] //= 2 ** (count - i - 1)
    return list(data)


def _assert_same(expected, actual):
    assert expected.native_metadata == actual.native_metadata
    assert expected.effective_config == actual.effective_config
    assert expected.migration_report == actual.migration_report
    assert expected.warnings == actual.warnings
    integer = expected.native_image.dtype.kind == "u"
    maximum = 2 ** expected.native_metadata.code_bit_depth - 1 if integer else 1.0
    result = compare_arrays(
        expected.native_image, actual.native_image,
        FINAL_INTEGER if integer else FINAL_FLOAT, data_range=maximum,
    )
    assert result["passed"], result
    result = compare_arrays(
        expected.preview_rgb8, actual.preview_rgb8, FINAL_INTEGER, data_range=255,
    )
    assert result["passed"], result
    assert actual.feedback.keys() == expected.feedback.keys()
    for name, value in expected.feedback.items():
        candidate = actual.feedback[name]
        if name in ("awb_gains", "ae_meter_average") and value is not None:
            result = compare_arrays(
                np.atleast_1d(value).astype(np.float64),
                np.atleast_1d(candidate).astype(np.float64), STATISTICS,
            )
            assert result["passed"], (name, result)
        else:
            assert type(candidate) is type(value), name
            assert jsonable(candidate) == jsonable(value), name
    assert expected.captures.keys() == actual.captures.keys()
    for name, reference in expected.captures.items():
        candidate = actual.captures[name]
        gate = stage_gate(name, logical_integer=reference.dtype.kind in "ui")
        scale = expected.feedback["linear_reference_code_max"] if name in CODE_FLOAT_STAGES else 1.0
        result = compare_arrays(reference, candidate, gate, scale=scale)
        assert result["passed"], (name, result)
        assert not candidate.flags.writeable
        assert not np.shares_memory(candidate, actual.native_image)
    assert [(e.stage_id, e.pass_index, e.ae_pass, e.shape, e.dtype)
            for e in actual.timings] == [
        (e.stage_id, e.pass_index, e.ae_pass, e.shape, e.dtype)
        for e in expected.timings
    ]
    assert all(e.status == "completed" and e.elapsed_ns >= 0 for e in actual.timings)


def _run(cfg, sources, device, *, options=None):
    before_cfg = copy.deepcopy(cfg)
    before_sources = [v.copy() if isinstance(v, np.ndarray) else v.read_bytes()
                      for v in sources]
    requests = options if options is not None else [
        ProcessingOptions(run_id=f"microbatch_{i}", capture_stages=CAPTURES)
        for i in range(len(sources))
    ]
    expected = [
        process_frame(source, cfg, options=opts)
        for source, opts in zip(sources, requests)
    ]
    engines, decoded = [], []
    for source, opts in zip(sources, requests):
        prepared = prepare_config(cfg, options=opts)
        engines.append(GpuEngine(
            prepared, opts, None, GpuOptions(device=str(device)),
        ))
        decoded.append(decode_source(
            source, sensor_info=prepared.effective_config["sensor_info"],
            platform=prepared.effective_config["platform"], run_id=opts.run_id,
        ))
    pending = execute_group(tuple(engines), tuple(decoded))
    assert len(pending) == len(sources)
    assert all(not frame.result.artifacts for frame in pending)
    actual = [commit_frame(frame) for frame in pending]
    for cpu, gpu in zip(expected, actual):
        _assert_same(cpu, gpu)
    assert cfg == before_cfg
    for source, before in zip(sources, before_sources):
        if isinstance(source, np.ndarray):
            np.testing.assert_array_equal(source, before)
        else:
            assert source.read_bytes() == before
    return actual


@pytest.mark.parametrize("count", (2, 4))
def test_microbatch_has_real_batch_dimension(cuda_device, monkeypatch, count):
    import brilliantisp_gpu.blocks.output as output
    import brilliantisp_gpu.blocks.raw as raw

    raw_sizes, output_sizes = [], []
    original_raw, original_output = raw.dead_pixel_correction, output.quantize_output

    def raw_spy(image, *args, **kwargs):
        raw_sizes.append(image.codes.shape[0])
        return original_raw(image, *args, **kwargs)

    def output_spy(image, *args, **kwargs):
        output_sizes.append(image.shape[0])
        return original_output(image, *args, **kwargs)

    monkeypatch.setattr(raw, "dead_pixel_correction", raw_spy)
    monkeypatch.setattr(output, "quantize_output", output_spy)
    cfg = _config()
    cfg["dead_pixel_correction"].update(is_enable=True, dp_threshold=64)
    _run(cfg, _sources(count), cuda_device)
    assert raw_sizes == [count]
    assert output_sizes == [count]


@pytest.mark.parametrize("operator,backend", TONE_PAIRINGS)
@pytest.mark.parametrize("placement", ("post_ccm", "pre_demosaic"))
def test_microbatch_tone_statistics_are_independent(
    cuda_device, operator, backend, placement,
):
    cfg = _config()
    cfg["tone_mapping"] = dict(
        is_enable=True, operator=operator, backend=backend, placement=placement,
        normalization="per_frame", normalize_output=operator != "durand",
    )
    if operator == "durand":
        cfg["tone_mapping"]["parameters"] = dict(downsample_factor=3)
    _run(cfg, _sources(2), cuda_device)


@pytest.mark.parametrize("algorithm", DEMOSAIC_ALGORITHMS)
def test_microbatch_demosaic_and_pre_render(cuda_device, algorithm):
    cfg = _config()
    cfg["demosaic"]["algorithm"] = algorithm
    cfg["tone_mapping"] = dict(
        is_enable=True, operator="reinhard", backend="float",
        placement="pre_demosaic", normalize_output=True,
    )
    _run(cfg, _sources(2), cuda_device)


def test_microbatch_per_frame_ae_and_green_repair(cuda_device):
    cfg = _config()
    cfg["sensor_info"]["repair_zero_green"] = "auto"
    cfg["crop"].update(
        is_enable=True, crop_x_start=1, crop_y_start=1, new_height=6, new_width=8,
    )
    cfg["digital_gain"].update(
        is_enable=True, is_auto=True, gain_array=[1.0, 2.0, 4.0], current_gain=0,
    )
    cfg["auto_exposure"].update(
        is_enable=True, exposure_correction_mode="direct",
        rerun_from_digital_gain=True, target_luminance=.5,
    )
    cfg["auto_white_balance"].update(
        is_enable=True, algorithm="pca", percentage=20,
        underexposed_percentage=0.0, overexposed_percentage=0.0,
    )
    cfg["white_balance"].update(is_enable=True, is_auto=True)
    sources = _sources(4)
    sources[1][0::2, 1::2] = 0
    sources[1][1::2, 0::2] = 0
    actual = _run(cfg, sources, cuda_device)
    assert actual[0].feedback["ae_stop_reason"] == "no_signal"
    assert actual[0].feedback["awb_status"] == "fallback_ill_conditioned"
    assert actual[0].feedback["ae_passes"] == 1
    assert any(r.feedback["ae_passes"] == 2 for r in actual[1:])


def test_microbatch_file_decode_geometry_once(cuda_device, tmp_path):
    cfg = _config()
    cfg["sensor_info"].update(
        height=9, width=10, source_format="raw", storage_format="uint16",
        endian_type="ieee-be", data_alignment="MSB",
        horizontal_flip=True, vertical_flip=True,
        header_bytes=4, row_stride_bytes=24,
        embedded_rows_top=1, embedded_rows_bottom=0,
    )
    sources = []
    for i in range(2):
        data = np.random.RandomState(11304 + i).randint(
            0, 4096, size=(9, 10),
        ).astype(np.uint16)
        path = tmp_path / ("array_source.raw" if i == 0 else f"frame_{i}.raw")
        path.write_bytes(
            b"HEAD" + b"".join(
                (row << 4).astype(">u2").tobytes() + b"PAD!" for row in data
            )
        )
        sources.append(path)
    actual = _run(cfg, sources, cuda_device)
    assert all(r.native_metadata.height == 8 for r in actual)
    assert all(r.effective_config["sensor_info"]["height"] == 8 for r in actual)


@pytest.mark.parametrize("output", (
    dict(model="rgb", transfer="srgb", format="png", bit_depth=8),
    dict(model="rgb", transfer="srgb", format="png", bit_depth=16),
    dict(model="ycbcr", matrix="bt601", layout="422", packing="uyvy", bit_depth=16),
))
def test_microbatch_integer_and_packed_outputs(cuda_device, output):
    cfg = _config()
    cfg["output"] = output
    _run(cfg, _sources(2), cuda_device)


def test_microbatch_defers_all_writes_until_commit(cuda_device, tmp_path):
    cfg = _config()
    cfg["tone_mapping"] = dict(
        is_enable=True, operator="reinhard", backend="float",
        normalize_output=True, is_save=True, is_plot_curve=False,
    )
    sources = _sources(2)
    engines, decoded, directories = [], [], []
    for i, source in enumerate(sources):
        directory = tmp_path / f"item_{i}"
        directories.append(directory)
        opts = ProcessingOptions(
            run_id=f"deferred_{i}", diagnostics=True, save_outputs=True,
            output_dir=directory,
        )
        prepared = prepare_config(cfg, options=opts)
        engines.append(GpuEngine(
            prepared, opts, None, GpuOptions(device=str(cuda_device)),
        ))
        decoded.append(decode_source(
            source, sensor_info=prepared.effective_config["sensor_info"],
            platform=prepared.effective_config["platform"], run_id=opts.run_id,
        ))
    pending = execute_group(tuple(engines), tuple(decoded))
    assert all(not directory.exists() for directory in directories)
    assert all(not frame.result.artifacts for frame in pending)
    for frame, directory in zip(pending, directories):
        result = commit_frame(frame)
        assert directory.is_dir()
        assert result.artifacts
        assert all(path.is_file() for path in result.artifacts)
        diagnostics = [path for path in result.artifacts if path.suffix == ".npy"]
        assert len(diagnostics) == 1
        assert np.load(diagnostics[0]).dtype == np.float32


@pytest.mark.parametrize("reason,message", (
    ("config", "group-compatible"),
    ("state", "ProcessingState"),
    ("limit", "exceeds max_batch_size"),
))
def test_microbatch_rejects_invalid_groups_before_upload(
    cuda_device, monkeypatch, reason, message,
):
    from brilliantisp_gpu.types import GpuError
    from util.pipeline_types import ProcessingState

    engines, decoded = [], []
    for i, source in enumerate(_sources(2)):
        cfg = _config()
        if reason == "config" and i:
            cfg["demosaic"]["algorithm"] = "bilinear"
        opts = ProcessingOptions(run_id=f"invalid_group_{i}")
        prepared = prepare_config(cfg, options=opts)
        engines.append(GpuEngine(
            prepared, opts, ProcessingState() if reason == "state" else None,
            GpuOptions(device=str(cuda_device), max_batch_size=1 if reason == "limit" else 4),
        ))
        decoded.append(decode_source(
            source, sensor_info=prepared.effective_config["sensor_info"],
            platform=prepared.effective_config["platform"], run_id=opts.run_id,
        ))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid cohort attempted a GPU pixel upload")

    monkeypatch.setattr(GpuEngine, "_upload_source", forbidden)
    with pytest.raises(GpuError, match=message):
        execute_group(tuple(engines), tuple(decoded))


def test_microbatch_cancellation_closes_already_started_contexts(cuda_device):
    from threading import Event
    from brilliantisp_gpu.stages import run_stage
    from util.pipeline_types import ProcessingCancelled

    token = Event()
    engines = []

    def callback(event):
        if event.status == "started":
            token.set()

    for i in range(2):
        opts = ProcessingOptions(
            run_id=f"start_cancel_{i}", cancellation_token=token,
            progress_callback=callback if i == 0 else None,
        )
        engine = GpuEngine(
            prepare_config(_config(), options=opts), opts, None,
            GpuOptions(device=str(cuda_device)),
        )
        engine._prepare_execution()
        engine.pass_index = 1
        engines.append(engine)

    def forbidden():
        pytest.fail("cancelled cohort computed pixels")

    with pytest.raises(ProcessingCancelled):
        run_stage(tuple(engines), "test_cancel", forbidden, split_batch=True)
    for engine in engines:
        assert len(engine.stage_events) == 1
        event = engine.stage_events[0]
        assert event.status == "cancelled"
        assert event.stage_id == "test_cancel"
        assert event.elapsed_ns == 0


def test_microbatch_empty_and_length_validation():
    assert execute_group((), ()) == ()
    with pytest.raises(ValueError, match="length mismatch"):
        execute_group((), (object(),))
    with pytest.raises(ValueError, match="length mismatch"):
        execute_group((object(),), ())


def test_microbatch_stage_rejects_wrong_batch_axis(cuda_device):
    import torch
    from brilliantisp_gpu.stages import run_stage
    from brilliantisp_gpu.types import GpuError

    engines = []
    for i in range(2):
        opts = ProcessingOptions(run_id=f"wrong_batch_{i}")
        engine = GpuEngine(
            prepare_config(_config(), options=opts), opts, None,
            GpuOptions(device=str(cuda_device)),
        )
        engine._prepare_execution()
        engine.pass_index = 1
        engines.append(engine)
    image = torch.zeros((1, 8, 10), device=cuda_device)
    with pytest.raises(GpuError, match="batch"):
        run_stage(tuple(engines), "wrong_batch", lambda: image, split_batch=True)
    assert all(len(engine.stage_events) == 1 for engine in engines)
    assert all(engine.stage_events[0].status == "failed" for engine in engines)
