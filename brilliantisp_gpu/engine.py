"""B=1 GPU native engine; stage-for-stage mirror of the CPU _run_native path."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np

from util.pipeline_types import (
    ProcessingCancelled,
    ProcessingOptions,
    ProcessingState,
    PipelineResult,
    StageEvent,
)

from .config import PreparedConfig
from .types import GpuError, GpuOptions, RawTensor, require_torch


@dataclasses.dataclass(frozen=True)
class LinearResult:
    """Per-frame product of the bounded linear middle (B=1 tensors)."""

    rgb: object
    white_balanced: object
    feedback: dict


def _dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


class GpuEngine:
    """Single-frame GPU native engine (B=1); stage semantics mirror the CPU facade."""

    def __init__(self, prepared: PreparedConfig, options: ProcessingOptions,
                 state: ProcessingState | None, gpu: GpuOptions):
        self.prepared = prepared
        self.options = options
        self.state = state
        self.gpu = gpu
        self.c_yaml = copy.deepcopy(prepared.effective_config)
        self.platform = self.c_yaml["platform"]
        self.sensor_info = self.c_yaml["sensor_info"]
        self.pass_index = 0
        self.stage_events: list[StageEvent] = []
        self.captures: dict = {}
        self.last_feedback: dict = {}
        self.last_artifacts: tuple = ()
        self.last_output_native = None
        self.last_output_rgb = None
        self.last_output_metadata = None
        self.run_id = options.run_id
        self.in_file = self.platform.get("in_file", "native")
        self.out_file = self.platform.get("out_file", "native")
        self.raw = None
        self.awb_gains = (1.0, 1.0)
        self.ae_feedback = None
        self.dga_current_gain = 0
        self.last_awb_status = "disabled"
        self.last_awb_count = 0
        self.last_awb_input_gain = 1.0

    def _emit(self, event: StageEvent) -> None:
        cb = self.options.progress_callback
        if cb is not None:
            cb(event)

    def _check_cancelled(self) -> None:
        token = self.options.cancellation_token
        if token is not None and token.is_set():
            raise ProcessingCancelled("processing cancelled between stages")

    def _stage_image(self, value):
        """Result element carrying image shape/dtype for event metadata."""
        torch = require_torch()
        if isinstance(value, RawTensor):
            return value
        if isinstance(value, tuple) and value:
            first = value[0]
            if isinstance(first, (torch.Tensor, RawTensor)):
                return first
        return value

    def _stage_meta(self, value):
        torch = require_torch()
        v = self._stage_image(value)
        if isinstance(v, RawTensor):
            return tuple(v.codes.shape[1:]), v.storage_dtype
        if isinstance(v, torch.Tensor):
            shape = tuple(v.shape[1:]) if v.ndim >= 3 and v.shape[0] == 1 \
                else tuple(v.shape)
            return shape, _dtype_name(v.dtype)
        return None, None

    def _run_stage(self, stage_id: str, thunk: Callable, ae_pass: int = 0):
        from .stages import run_stage
        return run_stage((self,), stage_id, thunk, ae_pass=ae_pass,
                         split_batch=False)

    def _host_result(self, value, *, frame_index: int = 0):
        """Device results -> owned CPU NumPy/control values (B removed)."""
        torch = require_torch()
        if isinstance(value, RawTensor):
            arr = value.codes[frame_index].detach().cpu().numpy()
            return arr.astype(value.storage_dtype)
        if isinstance(value, torch.Tensor):
            arr = value[frame_index].detach().cpu().numpy()
            return np.array(arr, copy=True)
        if isinstance(value, tuple):
            return tuple(self._host_result(v, frame_index=frame_index)
                         for v in value)
        if isinstance(value, (int, float, type(None), np.integer, np.floating)):
            return value if not isinstance(value, (np.integer, np.floating)) \
                else value.item()
        raise GpuError(f"_host_result cannot map {type(value).__name__}")

    def _prepare_execution(self) -> None:
        """One-time execution setup; idempotent."""
        if getattr(self, "_execution_prepared", False):
            return
        from .device import resolve_device
        self.config_notices = list(self.prepared.warnings)
        self.code_range = self.prepared.code_range
        self.parm_cro = self.c_yaml["crop"]
        self.parm_dpc = self.c_yaml["dead_pixel_correction"]
        self.parm_blc = self.c_yaml["black_level_correction"]
        self.parm_cmpd = self.c_yaml["companding"]
        self.parm_oec = self.c_yaml["oecf"]
        self.parm_dga = self.c_yaml["digital_gain"]
        self.parm_lsc = self.c_yaml["lens_shading_correction"]
        self.parm_bnr = self.c_yaml["bayer_noise_reduction"]
        self.parm_awb = self.c_yaml["auto_white_balance"]
        self.parm_wbc = self.c_yaml["white_balance"]
        self.parm_dem = self.c_yaml["demosaic"]
        self.parm_ccm = self.c_yaml["color_correction_matrix"]
        self.parm_cse = self.c_yaml["color_saturation_enhancement"]
        self.parm_ldci = self.c_yaml["ldci"]
        self.parm_sha = self.c_yaml["sharpen"]
        self.parm_2dn = self.c_yaml["2d_noise_reduction"]
        self.parm_sca = self.c_yaml["scale"]
        self.parm_ae = self.c_yaml["auto_exposure"]
        self.render_3a = bool(self.platform.get("render_3a", False))
        self.device = resolve_device(self.gpu.device)
        self._execution_prepared = True

    def _configure_source(self, decoded) -> None:
        """Apply a DecodedSource to the linked config/context objects."""
        sensor = decoded.sensor_info
        for key in ("width", "height", "bayer_pattern"):
            self.sensor_info[key] = sensor[key]
            self.c_yaml["sensor_info"][key] = sensor[key]
        if not decoded.source_identity.startswith("array:"):
            self.platform["filename"] = Path(
                decoded.source_identity).name
        self.platform["in_file"] = decoded.in_file
        self.platform["out_file"] = decoded.out_file
        self.in_file = decoded.in_file
        self.out_file = decoded.out_file
        self.source_identity = decoded.source_identity
        for warning in decoded.warnings:
            if warning not in self.config_notices:
                self.config_notices.append(warning)

    def _upload_source(self, decoded) -> RawTensor:
        torch = require_torch()
        codes = torch.as_tensor(
            np.ascontiguousarray(decoded.pixels), device=self.device
        ).to(torch.int64)
        return RawTensor(codes[None], decoded.pixels.dtype.name)

    def _run_gain_through_white_balance(self, oecf_raw, *, ae_pass=0,
                                        gain_index=None):
        from .blocks import linear as linear_blocks
        sensor = dict(self.sensor_info)
        sensor["hdr_bit_depth"] = self.code_range.bit_depth
        parm_dga = dict(self.parm_dga)
        parm_dga["ae_feedback"] = None
        if gain_index is not None:
            parm_dga["current_gain"] = int(gain_index)
        elif (self.state is not None and parm_dga.get("is_enable", True)
                and parm_dga.get("is_auto", False)
                and self.state.digital_gain_index is not None):
            parm_dga["current_gain"] = self.state.digital_gain_index

        def _dg():
            f32, indices = linear_blocks.digital_gain(oecf_raw, sensor, parm_dga)
            return f32, int(indices[0])
        dga_raw, gain_index_out = self._run_stage("digital_gain", _dg,
                                                ae_pass=ae_pass)
        self.dga_current_gain = int(gain_index_out)

        lsc_raw = self._run_stage(
            "lens_shading_correction",
            lambda: linear_blocks.lens_shading_correction(dga_raw, sensor, self.parm_lsc),
            ae_pass=ae_pass)

        from .blocks.bayer_noise_reduction import bayer_noise_reduction
        bnr_raw = self._run_stage(
            "bayer_noise_reduction",
            lambda: bayer_noise_reduction(lsc_raw, sensor, self.parm_bnr),
            ae_pass=ae_pass)

        from .blocks.auto_white_balance import auto_white_balance
        hint = self.state.previous_awb_gains if self.state else None
        self.last_awb_input_gain = (
            float(self.parm_dga["gain_array"][self.dga_current_gain])
            if self.parm_dga.get("is_enable", True) else 1.0)

        def _awb():
            measurement = auto_white_balance(
                bnr_raw, sensor, self.parm_awb, self.parm_wbc,
                input_gains=(self.last_awb_input_gain,),
                fallback_gains=(hint,))
            self.last_awb_status = measurement.statuses[0]
            self.last_awb_count = int(measurement.sample_counts[0])
            gains = measurement.gains.detach().cpu().numpy()
            self._awb_gain_tensor = measurement.gains
            return float(gains[0, 0]), float(gains[0, 1])
        self.awb_gains = self._run_stage("auto_white_balance", _awb,
                                         ae_pass=ae_pass)

        wb_raw = self._run_stage(
            "white_balance",
            lambda: linear_blocks.white_balance(
                bnr_raw, sensor, self.parm_wbc, self._awb_gain_tensor),
            ae_pass=ae_pass)
        return wb_raw


    def _compute_linear(self, oecf_raw) -> "LinearResult":
        """Bounded linear middle (B=1): DG->LSC->BNR->AWB->WB->demosaic->CCM
        ->linear tap->AE loop; all tensors on device."""
        from .ops.array import checked_float32
        from .blocks.auto_exposure import auto_exposure
        from .blocks.color import color_correction_matrix
        from .blocks.demosaic import demosaic

        torch = require_torch()
        declared_max = float(self.prepared.code_range.maximum)
        p_sensor = dict(self.sensor_info)
        p_sensor["hdr_bit_depth"] = self.code_range.bit_depth

        dg_auto = bool(self.parm_dga.get("is_enable", True)) and bool(
            self.parm_dga.get("is_auto", False))
        ae_enabled = bool(self.parm_ae.get("is_enable", False))
        ae_mode = str(self.parm_ae.get("exposure_correction_mode", "step"))
        ae_rerun = bool(self.parm_ae.get("rerun_from_digital_gain", True))
        max_passes = (24 if self.render_3a else 2) \
            if ae_enabled and ae_rerun else 1

        gain_idx = int(self.parm_dga.get("current_gain", 0))
        if dg_auto and self.state is not None \
                and self.state.digital_gain_index is not None:
            gain_idx = int(self.state.digital_gain_index)
        next_idx = gain_idx
        ae_stop = "disabled" if not ae_enabled else "no_rerun"
        ae_passes = 0
        ae = None
        suggested = gain_idx
        visited: set = set()
        rgb_linear = None
        wb_code = None
        awb_status = "disabled"
        awb_count = 0

        for ae_pass in range(max_passes):
            ae_passes = ae_pass + 1
            wb_code = self._run_gain_through_white_balance(
                oecf_raw, ae_pass=ae_pass, gain_index=gain_idx)
            awb_status = self.last_awb_status
            awb_count = self.last_awb_count

            def _demosaic(c=wb_code):
                normalized = c.to(torch.float64) / torch.as_tensor(
                    declared_max, dtype=torch.float64, device=c.device)
                bayer65 = checked_float32(normalized * 65535.0,
                                          name="demosaic_input")
                demos = demosaic(bayer65, p_sensor, self.parm_dem)
                out = demos.to(torch.float64) / torch.as_tensor(
                    65535.0, dtype=torch.float64, device=demos.device)
                return checked_float32(out, name="demosaic")
            rgb_linear = self._run_stage(
                "demosaic", _demosaic, ae_pass=ae_pass).to(torch.float64)

            rgb_linear = self._run_stage(
                "color_correction_matrix",
                lambda rgb=rgb_linear: color_correction_matrix(
                    checked_float32(rgb, name="ccm_input"), p_sensor,
                    self.parm_ccm),
                ae_pass=ae_pass)

            ae_rgb = self._run_stage("linear_rgb_pre_tone_mapping",
                                     lambda rgb=rgb_linear: rgb,
                                     ae_pass=ae_pass)
            ae_rgb = ae_rgb[None] if ae_rgb.ndim == 3 else ae_rgb

            def _ae(src=ae_rgb):
                nonlocal ae
                ae = auto_exposure(src, self.parm_ae)
                return ae.feedbacks[0]
            fb = self._run_stage("auto_exposure", _ae, ae_pass=ae_pass)
            self.ae_feedback = fb

            suggested = gain_idx
            meter_avg = ae.meter_averages[0]
            if ae_enabled and meter_avg is not None and dg_auto:
                if ae_mode == "direct":
                    from .blocks.auto_exposure import suggest_direct_gain_index
                    suggested = suggest_direct_gain_index(
                        gain_idx, list(self.parm_dga["gain_array"]),
                        meter_avg, self.parm_ae)
                elif fb == -1:
                    suggested = min(gain_idx + 1,
                                    len(self.parm_dga["gain_array"]) - 1)
                elif fb == 1:
                    suggested = max(gain_idx - 1, 0)
            if not ae_enabled:
                ae_stop = "disabled"
                break
            if meter_avg is None or meter_avg <= 0:
                ae_stop = "no_signal"
                break
            if not dg_auto:
                ae_stop = "manual_gain"
                next_idx = suggested if ae_mode == "direct" else gain_idx
                break
            if not ae_rerun:
                ae_stop = "no_rerun"
                next_idx = suggested
                break
            state_key = (suggested, fb)
            converged = suggested == gain_idx or (
                ae_mode != "direct" and fb == 0)
            if converged:
                if ae_mode == "direct":
                    target = self.parm_ae.get("target_luminance")
                    target = 0.5 if target is None else float(target)
                    wants_more = target > meter_avg
                    wants_less = target < meter_avg
                else:
                    wants_more, wants_less = fb == -1, fb == 1
                gain_limited = (
                    (wants_more and gain_idx == len(
                        self.parm_dga["gain_array"]) - 1)
                    or (wants_less and gain_idx == 0))
                ae_stop = "gain_limit" if gain_limited else "converged"
                next_idx = suggested
                break
            if state_key in visited:
                ae_stop = "repeated_gain"
                next_idx = suggested
                break
            if ae_pass + 1 >= max_passes:
                ae_stop = "pass_limit"
                next_idx = suggested
                break
            visited.add(state_key)
            gain_idx = suggested
            next_idx = suggested
            self._check_cancelled()

        if ae_stop == "disabled" and ae_enabled:
            ae_stop = "no_signal"

        self.last_feedback = {
            "awb_gains": (float(self.awb_gains[0]), float(self.awb_gains[1])),
            "digital_gain_index": int(self.dga_current_gain),
            "ae_feedback": self.ae_feedback,
            "awb_status": awb_status,
            "awb_sample_count": awb_count,
            "awb_input_gain": self.last_awb_input_gain,
            "awb_meter_domain": "gain_compensated_linear_bayer_codes",
            "ae_meter_domain": "scene_linear_rgb",
            "ae_meter_average": getattr(ae, "meter_averages", (None,))[0]
            if ae is not None else None,
            "ae_sample_count": getattr(ae, "sample_counts", (0,))[0]
            if ae is not None else 0,
            "ae_suggested_gain_index": int(suggested),
            "ae_stop_reason": ae_stop,
            "ae_passes": ae_passes,
            "linear_reference_code_max": declared_max,
            "next_digital_gain_index": int(next_idx),
        }
        return LinearResult(rgb=rgb_linear, white_balanced=wb_code,
                            feedback=dict(self.last_feedback))

    def run(self, source) -> PipelineResult:
        """Decode, prepare, and process one frame via execute_group."""
        from .microbatch import commit_frame, execute_group
        from .sources import decode_source

        self._prepare_execution()
        self._check_cancelled()
        decoded = decode_source(
            source, sensor_info=self.sensor_info, platform=self.platform,
            run_id=self.run_id)
        pending = execute_group((self,), (decoded,))
        return commit_frame(pending[0])
