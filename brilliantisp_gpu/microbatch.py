"""True-batched microbatch: batched front + render, per-frame linear middle.

``execute_group(engines, decoded_sources)`` runs a compatible cohort as one
real batched workload — the front half and render/post chain execute on
leading-B tensors through ``stages.run_stage`` while the bounded
3A/linear middle stays per-frame B=1 (independent qualified controls).
Filesystem IO is deferred to ``commit_frame``; orchestration commits state separately.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np

from util.pipeline_types import PipelineResult

from .grouping import group_key
from .stages import run_stage
from .types import GpuError, RawTensor, require_torch


def _cohort_stage(contexts, stage_id, thunk, *, ae_pass=0):
    """Single-context stages keep the engine seam; cohorts run batched."""
    if len(contexts) == 1:
        return contexts[0]._run_stage(stage_id, thunk, ae_pass=ae_pass)
    return run_stage(contexts, stage_id, thunk, ae_pass=ae_pass,
                     split_batch=True)


class PendingFrame:
    """Completed GPU compute for one frame; IO/state deferred to commit."""

    __slots__ = ("engine", "result", "diagnostics", "_native", "_metadata",
                 "_spec", "_committed", "_failed")

    def __init__(self, engine, result, diagnostics, native, metadata, spec):
        self.engine = engine
        self.result = result
        self.diagnostics = diagnostics
        self._native = native
        self._metadata = metadata
        self._spec = spec
        self._committed = False
        self._failed = False


def _cohort_key(engine):
    sensor = engine.c_yaml["sensor_info"]
    return group_key(
        engine.c_yaml,
        shape=(int(sensor["height"]), int(sensor["width"])),
        cfa=str(sensor["bayer_pattern"]),
        storage_dtype=engine._upload_dtype,
        capture_stages=tuple(engine.options.capture_stages),
        diagnostics=bool(engine.options.diagnostics),
        profile_stages=bool(engine.gpu.profile_stages),
    )


def _front_cohort(contexts, batch: RawTensor) -> RawTensor:
    """Batched crop->DPC->BLC->green->PWC->OECF on full leading B."""
    from .blocks import raw as raw_blocks

    first = contexts[0]
    sensor = first.sensor_info
    platform = first.platform
    skip_disabled = platform.get("skip_disabled_modules", False)

    def _crop():
        if skip_disabled and not first.parm_cro["is_enable"]:
            return batch
        out = raw_blocks.crop(batch, sensor, first.parm_cro, platform=None)
        h, w = int(out.codes.shape[-2]), int(out.codes.shape[-1])
        for ctx in contexts:
            if out is not batch:
                ctx.sensor_info["width"] = w
                ctx.sensor_info["height"] = h
                ctx.c_yaml["sensor_info"]["width"] = w
                ctx.c_yaml["sensor_info"]["height"] = h
                for key in ("bayer_pattern", "orig_size"):
                    if key in sensor:
                        ctx.sensor_info[key] = sensor[key]
                        ctx.c_yaml["sensor_info"][key] = sensor[key]
            raw_blocks._crop_name(ctx.platform, h, w)
        return out
    cropped = _cohort_stage(contexts, "crop", _crop)

    def _dpc():
        if skip_disabled and not first.parm_dpc["is_enable"]:
            return cropped
        return raw_blocks.dead_pixel_correction(cropped, sensor,
                                                first.parm_dpc)
    dpc_raw = _cohort_stage(contexts, "dead_pixel_correction", _dpc)

    def _blc():
        if skip_disabled and not first.parm_blc["is_enable"]:
            return dpc_raw
        return raw_blocks.black_level_correction(dpc_raw, sensor,
                                                 first.parm_blc)
    blc_raw = _cohort_stage(contexts, "black_level_correction", _blc)

    def _green_repair():
        torch = require_torch()
        mode = sensor.get("repair_zero_green", "auto")
        if mode is False or mode == "false":
            return blc_raw
        pattern = str(sensor.get("bayer_pattern", "rggb")).strip().lower()
        frames = blc_raw.codes.shape[0]
        should = []
        for i in range(frames):
            frame = RawTensor(blc_raw.codes[i:i + 1], blc_raw.storage_dtype)
            if int(frame.codes.max()) == 0:
                should.append(False)
                continue
            frac = float(raw_blocks.green_zero_fraction(frame, pattern))
            should.append(
                mode is True or mode == "true"
                or (str(mode).lower() in ("auto", "") and frac > 0.5))
        if not any(should):
            return blc_raw
        repaired = raw_blocks.repair_zero_green_cfa(blc_raw, pattern)[0]
        mask = torch.as_tensor(should, dtype=torch.bool,
                               device=blc_raw.codes.device)[:, None, None]
        codes = torch.where(mask, repaired.codes, blc_raw.codes)
        return RawTensor(codes, blc_raw.storage_dtype)
    grn_raw = _cohort_stage(contexts, "green_repair", _green_repair)

    def _pwc():
        if skip_disabled and not first.parm_cmpd["is_enable"]:
            return RawTensor(grn_raw.codes, "uint32")
        return raw_blocks.companding(grn_raw, sensor, first.parm_cmpd)
    cmpd_raw = _cohort_stage(contexts, "companding", _pwc)

    def _oecf():
        if skip_disabled and not first.parm_oec.get("is_enable", False):
            return cmpd_raw
        input_sensor = dict(sensor)
        input_sensor["hdr_bit_depth"] = (
            sensor.get("hdr_bit_depth", sensor["bit_depth"])
            if first.parm_cmpd.get("is_enable", False)
            else sensor["bit_depth"])
        return raw_blocks.oecf(cmpd_raw, input_sensor, first.parm_oec)
    return _cohort_stage(contexts, "oecf", _oecf)


def _diagnostic_intents(ctx, image, *, stage, resolved, frame_index):
    """Own host snapshot/intent for deferred diagnostics IO."""
    if not resolved.spec.enabled or not ctx.options.diagnostics:
        return None
    if not resolved.is_save and not resolved.is_plot_curve:
        return None
    host = ctx._host_result(image, frame_index=frame_index) \
        if resolved.is_save else None
    return (stage, resolved, host)


def _commit_diagnostics(ctx, stage, resolved, host):
    """Explicit IO at commit only: NPY metadata + optional curve plot."""
    import dataclasses
    from modules.tone_mapping.diagnostics import (
        plot_curve, save_mapped_output)
    artifacts = ()
    module = (f"native_{stage}_{resolved.spec.operator}_"
              f"{resolved.spec.backend}_normalized")
    if resolved.is_save:
        metadata = {
            "diagnostic_schema_version": 1,
            "stage": stage,
            "signal_domain": "tone_mapped_bayer"
            if host.ndim == 2 else "display_linear_rgb",
            "transfer": "linear",
            "value_units": "normalized",
            "reference_white": 1.0,
            "cfa_pattern": (ctx.sensor_info["bayer_pattern"]
                            if host.ndim == 2 else None),
            "shape": list(host.shape),
            "dtype": str(host.dtype),
            "tone_mapping": dataclasses.asdict(resolved.spec),
        }
        npy = save_mapped_output(
            host, platform=ctx.platform,
            input_name=ctx.in_file, module_name=module,
            metadata=metadata)
        artifacts += (npy, npy.with_suffix(".json"))
    if resolved.is_plot_curve:
        curve = plot_curve(resolved.spec, ctx.platform)
        if curve is not None:
            artifacts += (curve,)
    return artifacts


def _render_cohort(contexts, linears, resolved, spec):
    """Batched render: tone/saturation/luma/transfer/scale/output/preview."""
    from .ops.array import checked_float32
    from .ops.color import adjust_saturation, encode_transfer, luminance
    from .ops.luma import replace_luminance
    from .blocks import tone_mapping as tm_block
    from .blocks.color import color_correction_matrix
    from .blocks.demosaic import demosaic
    from .blocks import luma as luma_blocks
    from .blocks import output as output_blocks
    from .blocks.scaling import scale as scale_block

    torch = require_torch()
    first = contexts[0]
    declared_max = float(first.prepared.code_range.maximum)
    p_sensor = dict(first.sensor_info)
    p_sensor["hdr_bit_depth"] = first.code_range.bit_depth
    advanced_pre = (resolved.placement == "pre_demosaic"
                    and resolved.spec.enabled)

    rgb = torch.cat([lr.rgb for lr in linears], dim=0)
    wb = torch.cat([lr.white_balanced for lr in linears], dim=0)
    diagnostics = [[] for _ in contexts]

    if advanced_pre:
        wb_norm = wb.to(torch.float64) / torch.as_tensor(
            declared_max, dtype=torch.float64, device=wb.device)
        mapped_plane = _cohort_stage(
            contexts, "tone_mapping_pre",
            lambda: tm_block.map_native(
                wb_norm, config=resolved.spec, color_model="bayer"))
        for i, ctx in enumerate(contexts):
            intent = _diagnostic_intents(
                ctx, mapped_plane, stage="tone_mapping_pre",
                resolved=resolved, frame_index=i)
            if intent is not None:
                diagnostics[i].append(intent)
        mapped_plane = mapped_plane.to(torch.float64)

        def _render_demosaic():
            pre_dem = demosaic(
                checked_float32(mapped_plane * 65535.0,
                                name="pre_demosaic_input"),
                p_sensor, first.parm_dem)
            pre_rgb = pre_dem.to(torch.float64) / torch.as_tensor(
                65535.0, dtype=torch.float64, device=pre_dem.device)
            return checked_float32(pre_rgb, name="demosaic_render")
        pre_rgb = _cohort_stage(contexts, "demosaic_render",
                                _render_demosaic)
        pre_ccm = _cohort_stage(
            contexts, "color_correction_matrix_render",
            lambda: color_correction_matrix(pre_rgb, p_sensor,
                                            first.parm_ccm))
        mapped = _cohort_stage(contexts, "tone_mapping_post",
                               lambda: pre_ccm).to(torch.float64)
    else:
        mapped = _cohort_stage(
            contexts, "tone_mapping_post",
            lambda: tm_block.map_native(
                rgb, config=resolved.spec, color_model="rgb"))
        for i, ctx in enumerate(contexts):
            intent = _diagnostic_intents(
                ctx, mapped, stage="tone_mapping_post",
                resolved=resolved, frame_index=i)
            if intent is not None:
                diagnostics[i].append(intent)
        mapped = mapped.to(torch.float64)

    def _saturation(m=mapped):
        if first.parm_cse.get("is_enable", False):
            return adjust_saturation(
                checked_float32(m, name="saturation_input"),
                float(first.parm_cse["saturation_gain"]),
                matrix="bt709", dtype="float32")
        return m
    mapped = _cohort_stage(contexts, "saturation",
                           _saturation).to(torch.float64)

    ldci_place = str(first.parm_ldci.get("placement", "post_transfer"))

    def _ldci_pre(m=mapped):
        if first.parm_ldci.get("is_enable", False) \
                and ldci_place == "pre_transfer":
            y = luminance(m, matrix=spec.matrix, dtype="float64")
            out = luma_blocks.clahe_luma(
                y, scale=1.0,
                wind=first.parm_ldci.get("wind", 64),
                clip_limit=first.parm_ldci.get("clip_limit", 2.0))
            return replace_luminance(m, out, matrix=spec.matrix)
        return m
    mapped = _cohort_stage(contexts, "ldci_pre_transfer",
                           _ldci_pre).to(torch.float64)

    def _transfer(m=mapped):
        if spec.floating_linear:
            return checked_float32(
                m, name="linear_float_output").to(torch.float64)
        bounded = m.clamp(0.0, 1.0) if spec.transfer != "linear" else m
        return encode_transfer(bounded, spec.transfer,
                               gamma=getattr(spec, "gamma", 2.2),
                               dtype="float64")
    encoded = _cohort_stage(contexts, "transfer_encode",
                            _transfer).to(torch.float64)

    def _ldci_post(e=encoded):
        if first.parm_ldci.get("is_enable", False) \
                and ldci_place == "post_transfer":
            y = luminance(e, matrix=spec.matrix, dtype="float64")
            out = luma_blocks.clahe_luma(
                y, scale=1.0,
                wind=first.parm_ldci.get("wind", 64),
                clip_limit=first.parm_ldci.get("clip_limit", 2.0))
            return replace_luminance(e, out, matrix=spec.matrix)
        return e
    encoded = _cohort_stage(contexts, "ldci_post_transfer",
                            _ldci_post).to(torch.float64)

    def _sharpen(e=encoded):
        if first.parm_sha.get("is_enable", False):
            y = luminance(e, matrix=spec.matrix, dtype="float64")
            out = luma_blocks.sharpen_luma(
                y, scale=1.0,
                sigma=first.parm_sha.get("sharpen_sigma", 1.0),
                strength=first.parm_sha.get("sharpen_strength", 0.5))
            return replace_luminance(e, out, matrix=spec.matrix)
        return e
    encoded = _cohort_stage(contexts, "sharpen",
                            _sharpen).to(torch.float64)

    def _nlm(e=encoded):
        if first.parm_2dn.get("is_enable", False):
            y = luminance(e, matrix=spec.matrix, dtype="float64")
            out = luma_blocks.non_local_means(
                y, scale=1.0,
                window_size=first.parm_2dn.get("window_size", 3),
                patch_size=first.parm_2dn.get("patch_size", 3),
                strength=first.parm_2dn.get("strength", 10.0 / 255.0))
            return replace_luminance(e, out, matrix=spec.matrix)
        return e
    encoded = _cohort_stage(contexts, "noise_reduction_2d",
                            _nlm).to(torch.float64)

    def _scl(e=encoded):
        if not first.parm_sca.get("is_enable", False):
            return e
        out = scale_block(
            checked_float32(e, name="scale_input"), p_sensor,
            first.parm_sca)
        return out.to(torch.float64)
    encoded = _cohort_stage(contexts, "scale",
                            _scl).to(torch.float64)

    for ctx in contexts:
        ctx._check_cancelled()
    native = _cohort_stage(
        contexts, "output",
        lambda: output_blocks.quantize_output(encoded, spec))
    for ctx in contexts:
        ctx._check_cancelled()
    preview = output_blocks.preview_from_native(native, spec)
    return native, preview, diagnostics


def execute_group(engines, sources):
    """Process a compatible cohort as one real batched GPU workload."""
    if len(engines) != len(sources):
        raise ValueError(
            f"engines/sources length mismatch: {len(engines)} != {len(sources)}")
    if not engines:
        return ()
    torch = require_torch()
    from modules.tone_mapping.registry import resolve_tone_mapping
    from .device import gpu_scope
    from util.native_output import output_metadata

    for engine in engines:
        engine._prepare_execution()
        engine._check_cancelled()
    device = engines[0].device
    if any(e.device != device for e in engines):
        raise GpuError("cohort engines must resolve to the same device")

    for engine, decoded in zip(engines, sources):
        engine._configure_source(decoded)
        dtype = str(decoded.pixels.dtype.name)
        engine._upload_dtype = dtype
    if len(engines) > 1:
        keys = {_cohort_key(e) for e in engines}
        if len(keys) != 1:
            raise GpuError(
                "cohort frames are not group-compatible "
                "(config/geometry/CFA/storage/options differ)")
        if any(e.state is not None for e in engines):
            raise GpuError(
                "cohort processing with a ProcessingState hint is only "
                "supported for single frames")
        limit = min(e.gpu.max_batch_size for e in engines)
        if len(engines) > limit:
            raise GpuError(
                f"cohort size {len(engines)} exceeds max_batch_size {limit}")

    with torch.cuda.device(device), gpu_scope(device):
        parts = [
            engine._upload_source(decoded).codes
            for engine, decoded in zip(engines, sources)
        ]
        codes = torch.cat(parts, dim=0)
        del parts
        batch = RawTensor(codes, engines[0]._upload_dtype)
        contexts = tuple(engines)
        for ctx in contexts:
            ctx.pass_index += 1

        front = _front_cohort(contexts, batch)
        del batch, codes
        linears = [ctx._compute_linear(
            RawTensor(front.codes[i:i + 1], front.storage_dtype))
            for i, ctx in enumerate(contexts)]
        del front

        resolved = resolve_tone_mapping(contexts[0].c_yaml)
        spec = contexts[0].prepared.output_spec
        native, preview, diagnostics = _render_cohort(
            contexts, linears, resolved, spec)
        del linears

        pending = []
        for i, ctx in enumerate(contexts):
            native_np = ctx._host_result(native, frame_index=i)
            preview_np = ctx._host_result(preview, frame_index=i)
            metadata = output_metadata(
                native_np, spec,
                source_identity=ctx.source_identity,
                tone_mapped=bool(resolved.spec.enabled))
            ctx.last_output_native = native_np
            ctx.last_output_rgb = np.array(preview_np, copy=True)
            ctx.last_output_metadata = metadata
            result = PipelineResult(
                native_image=native_np,
                native_metadata=dataclasses.replace(
                    metadata, source_identity=ctx.source_identity),
                preview_rgb8=np.array(preview_np, copy=True),
                effective_config=copy.deepcopy(ctx.c_yaml),
                feedback=dict(ctx.last_feedback),
                migration_report=dict(ctx.prepared.migration_report),
                artifacts=(),
                timings=tuple(ctx.stage_events),
                captures=dict(ctx.captures),
                warnings=tuple(ctx.config_notices),
            )
            pending.append(PendingFrame(
                ctx, result, tuple(diagnostics[i]), native_np, metadata, spec))
    return tuple(pending)


def commit_frame(pending: PendingFrame) -> PipelineResult:
    """Run deferred diagnostics/writer IO for one completed frame."""
    if pending._committed:
        return pending.result
    if pending._failed:
        raise GpuError(
            "frame commit previously failed; not retrying deferred writer")
    ctx = pending.engine
    ctx._check_cancelled()
    try:
        artifacts = []
        for stage, resolved, host in pending.diagnostics:
            artifacts.extend(
                _commit_diagnostics(ctx, stage, resolved, host))
        if ctx.options.save_outputs and ctx.options.output_dir:
            ctx._check_cancelled()
            from util.output_writer import write_output
            artifacts.extend(write_output(
                pending._native, pending._spec, pending._metadata,
                output_dir=Path(ctx.options.output_dir),
                stem=ctx.out_file,
                effective_config=ctx.c_yaml,
                migration_report=ctx.prepared.migration_report))
    except Exception:
        pending._failed = True
        raise
    ctx.last_artifacts += tuple(artifacts)
    pending.result = dataclasses.replace(
        pending.result, artifacts=tuple(ctx.last_artifacts))
    pending._committed = True
    return pending.result
