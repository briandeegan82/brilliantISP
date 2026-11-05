"""Shared implementation behind the legacy wrapper classes.

``LegacyToneMapper`` is the single base the retired TMO files subclass:
one constructor, one spec builder, one execute path, consistent flag
handling. Float wrappers use normalized 0..1 input/output; integer
wrappers use an explicit or declared HDR ``input_max`` (never observed
maxima) and emit uint16 at 65535.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from modules.tone_mapping.backends import map_luminance
from modules.tone_mapping.normalization import code_max_from_bits
from modules.tone_mapping.registry import resolve_tone_mapping, select_legacy_section


def hdr_input_max(sensor_info: Mapping[str, Any]) -> int:
    """Declared HDR code maximum (never inferred from image data)."""
    return code_max_from_bits(sensor_info, "hdr_bit_depth", 24, 32)


class LegacyToneMapper:
    """Base for the retired wrapper classes.

    ``float_mode`` True -> normalized float input/output (0..1);
    False -> declared-integer input -> uint16 at 65535.
    """

    legacy_selector = "reinhard_integer"
    float_mode = False

    def __init__(
        self,
        img: np.ndarray,
        platform: Mapping[str, Any],
        sensor_info: Mapping[str, Any],
        params: Mapping[str, Any],
        input_max: int | None = None,
    ) -> None:
        self.img_orig = np.asarray(img)
        self.img = self.img_orig
        section_name, _ = select_legacy_section(self.legacy_selector, {})
        resolved = resolve_tone_mapping({
            "tone_mapping": {"is_enable": True, "tone_mapper": self.legacy_selector},
            section_name: params,
        })
        self.spec = resolved.spec
        self.operator = self.spec.operator
        self.backend = self.spec.backend
        self.is_enable = self.spec.enabled
        self.is_save = resolved.is_save
        self.is_plot_curve = resolved.is_plot_curve
        self.is_debug = params.get("is_debug", False)
        self.input_max = input_max if input_max is not None else (1.0 if self.float_mode else hdr_input_max(sensor_info))
        self.output_max = 1 if self.float_mode else 65535
        self.sensor_info = sensor_info
        self.platform = platform
        self._out: np.ndarray | None = None

    def execute(self) -> np.ndarray:
        out = map_luminance(
            self.img_orig,
            input_max=self.input_max,
            output_max=self.output_max,
            config=self.spec,
        )
        if not self.float_mode:
            out = np.clip(out, 0, 65535).astype(np.uint16)
        self._out = out
        self.img = out
        if not self.is_enable:
            return out
        if self.is_save:
            self.save()
        if self.is_plot_curve:
            self.plot_tone_curve()
        return out

    def save(self) -> None:
        if self._out is None:
            return
        from modules.tone_mapping import diagnostics

        diagnostics.save_mapped_output(
            self._out,
            platform=self.platform,
            input_name=str(self.platform.get("in_file", "frame")),
            module_name=f"{self.operator}_{self.backend}",
        )

    def plot_tone_curve(self) -> None:
        from modules.tone_mapping import diagnostics

        diagnostics.plot_curve(self.spec, self.platform)
