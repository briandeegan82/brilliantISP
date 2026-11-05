"""Pipeline tone-mapping adapter: resolve config, map luminance, emit uint16.

Resolves the effective spec live from the calling context each execute (so
profiling/context mutations are honored), maps the declared-domain
luminance through the shared engine, and recombines the original RGB
ratio. Unsigned Bayer input is passed to integer backends unmodified;
post-CCM fixed backend uses the exact widened weighted numerator
(2126R+7152G+722B, declared range scaled by 10000) per the settled recipe.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from util.isp_types import (
    ToneMappingContext,
)

from modules.tone_mapping.backends import map_luminance
from modules.tone_mapping.normalization import (
    check_code_range,
    code_max_from_bits,
    LUMA_DENOMINATOR,
    recombine,
    rgb_luminance,
    rgb_luminance_numerator,
)
from modules.tone_mapping.registry import (
    ResolvedToneMapping,
    select_legacy_section,
    resolve_tone_mapping,
)


class ToneMapping:
    """Thin pipeline adapter around the shared tone-mapping engine."""

    def __init__(self, img: np.ndarray, pipeline_self: ToneMappingContext) -> None:
        self.img_orig = img
        self.platform = pipeline_self.platform
        self.sensor_info = pipeline_self.sensor_info
        self.context = pipeline_self
        self.logger = pipeline_self.logger
        self.resolved = self._resolve()
        self._apply(self.resolved)
        for notice in self.resolved.notices:
            self.logger.warning(f"tone_mapping: {notice}")
        self._img_out: np.ndarray | None = None

    def _apply(self, r: ResolvedToneMapping) -> None:
        self.resolved_spec = r.spec
        self.placement = r.placement
        self.method = r.spec.operator
        self.enable = r.spec.enabled
        self.is_save = r.is_save
        self.is_plot_curve = r.is_plot_curve
        self.tone_mapping_before_demosaic = r.placement == "pre_demosaic"

    def _resolve(self) -> ResolvedToneMapping:
        cfg: dict[str, Any] = {}
        base = getattr(self.context, "c_yaml", None)
        if isinstance(base, Mapping):
            cfg.update(base)
        live = getattr(self.context, "tone_mapping", None)
        if isinstance(live, Mapping):
            cfg["tone_mapping"] = dict(live)
        tm = cfg.get("tone_mapping", {})
        if not isinstance(tm, Mapping) or "operator" in tm or tm.get("is_enable") is False:
            return resolve_tone_mapping(cfg)
        selector = str(tm.get("tone_mapper", "reinhard_integer"))
        selected_attr = {
            "aces": "param_aces",
            "aces_integer": "param_aces_integer",
            "hable": "param_hable",
            "hable_integer": "param_hable_integer",
            "reinhard_integer": "param_integer_tmo",
            "integer_tmo": "param_integer_tmo",
            "durand": "param_durand",
            "hdr_durand": "param_durand",
        }.get(selector)
        if selected_attr is not None:
            section_name, _ = select_legacy_section(selector, cfg)
            live_section = getattr(self.context, selected_attr, None)
            other_alias = "integer_tmo" if section_name == "reinhard_integer" else "reinhard_integer"
            represents_other_alias = (
                selector in ("integer_tmo", "reinhard_integer")
                and section_name in cfg
                and other_alias in cfg
                and live_section is cfg[other_alias]
            )
            if hasattr(self.context, selected_attr) and not represents_other_alias:
                cfg[section_name] = live_section
        return resolve_tone_mapping(cfg)

    def _input_max(self) -> int:
        if self.placement == "pre_demosaic":
            return code_max_from_bits(self.sensor_info, "hdr_bit_depth", 24, 32)
        return code_max_from_bits(self.sensor_info, "pipeline_rgb_bit_depth", 16, 16)

    def _output_max(self) -> int:
        return code_max_from_bits(self.sensor_info, "pipeline_rgb_bit_depth", 16, 16)

    def _fixed_luminance(self, img: np.ndarray, input_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact unsigned weighted numerator for fixed backend + float Y."""
        if img.ndim == 2:
            y = np.asarray(img, dtype=np.float64)
            if img.dtype.kind == "u":
                return np.asarray(img), y
            return np.asarray(img), y
        if img.dtype.kind != "u" or img.dtype.itemsize not in (1, 2, 4):
            raise ValueError(
                "fixed backend requires unsigned RGB input; choose float/lut "
                "for floating sources"
            )
        if int(img.max()) > input_max:
            raise ValueError("RGB codes exceed declared input_max")
        num = rgb_luminance_numerator(img).astype(np.uint32)
        return num, num.astype(np.float64) / LUMA_DENOMINATOR

    def execute(
        self,
        visualize_output: bool = True,
        save_output: bool = True,
        *,
        plot_curve: bool | None = None,
    ) -> np.ndarray:
        self.resolved = self._resolve()
        self._apply(self.resolved)
        spec = self.resolved_spec
        input_max = float(self._input_max())
        output_max = self._output_max()
        img = self.img_orig
        self._img_out = None

        if not spec.enabled:
            if self.placement == "post_ccm":
                self.img = img
                return img
            scaled = np.clip(
                np.asarray(img, dtype=np.float64) * (output_max / input_max),
                0, float(output_max),
            )
            self.img = scaled.astype(np.uint16)
            return self.img

        if img.ndim == 2:
            luminance = np.asarray(img)
            mapped = map_luminance(
                luminance,
                input_max=input_max,
                output_max=output_max,
                config=spec,
            )
            result = np.clip(np.asarray(mapped, np.float64), 0, output_max).astype(np.uint16)
        else:
            if spec.backend == "fixed":
                src, y_f = self._fixed_luminance(img, int(input_max))
                mapped = map_luminance(
                    src, input_max=input_max * LUMA_DENOMINATOR,
                    output_max=output_max, config=spec,
                )
            else:
                check_code_range(img, input_max, ValueError)
                y_f = rgb_luminance(img)
                mapped = map_luminance(
                    y_f, input_max=input_max, output_max=output_max, config=spec,
                )
            result = recombine(mapped, img, y_f, output_max=output_max)
        self.img = result
        self._img_out = result

        can_save = bool(save_output) and self.resolved.is_save
        can_plot = bool(save_output) and bool(visualize_output) and (
            self.resolved.is_plot_curve
            if plot_curve is None else bool(plot_curve)
        )
        if can_save:
            self.save()
        if can_plot:
            from modules.tone_mapping import diagnostics

            diagnostics.plot_curve(spec, self.platform)
        return result

    def save(self) -> None:
        """Save the actual mapped output as native NPY under output_dir."""
        if self._img_out is None:
            return
        from modules.tone_mapping import diagnostics

        diagnostics.save_mapped_output(
            self._img_out,
            platform=self.platform,
            input_name=str(self.platform.get("in_file", "frame")),
            module_name=f"tone_mapping_{self.method}_{self.resolved_spec.backend}",
        )

    def extract_luminance(self) -> np.ndarray:
        img = np.asarray(self.img_orig)
        if img.ndim == 2:
            return np.asarray(img, dtype=np.float64)
        return rgb_luminance(img)

    def combine_luminance(self, output_luminance: np.ndarray) -> np.ndarray:
        """Legacy chromaticity helper retained for callers."""
        return recombine(
            output_luminance, self.img_orig, self.extract_luminance(),
            output_max=self._output_max(),
        )
