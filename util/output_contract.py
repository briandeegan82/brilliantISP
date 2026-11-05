from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value.lower() not in choices:
        raise ValueError(f"output.{name} must be one of {choices}, got {value!r}")
    return value.lower()


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"output.{name} must be a positive finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"output.{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"output.{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class OutputSpec:
    model: str = "rgb"
    color_space: str = "srgb"
    transfer: str = "srgb"
    gamma: float = 2.2
    bit_depth: int = 8
    format: str = "png"
    matrix: str = "bt709"
    range_mode: str = "full"
    layout: str = "444"
    packing: str = "yuyv"
    is_save: bool = False

    def __post_init__(self) -> None:
        fields = (
            ("model", ("rgb", "ycbcr")),
            ("color_space", ("srgb",)),
            ("transfer", ("srgb", "rec709", "power", "linear")),
            ("format", ("png", "jpeg", "tiff", "yuv")),
            ("matrix", ("bt601", "bt709")),
            ("range_mode", ("full", "limited")),
            ("layout", ("444", "422")),
            ("packing", ("yuyv", "uyvy")),
        )
        for name, choices in fields:
            external_name = "range" if name == "range_mode" else name
            object.__setattr__(self, name, _choice(external_name, getattr(self, name), choices))
        if type(self.bit_depth) is not int or self.bit_depth not in (8, 16, 32):
            raise ValueError("output.bit_depth must be 8, 16, or 32")
        if type(self.is_save) is not bool:
            raise ValueError("output.is_save must be a boolean")
        object.__setattr__(self, "gamma", _positive("gamma", self.gamma) if self.transfer == "power" else 2.2)
        if self.model == "rgb":
            if self.range_mode != "full" or self.layout != "444" or self.packing != "yuyv":
                raise ValueError("RGB output uses full range; YCbCr layout/packing settings require model ycbcr")
            if self.format == "yuv":
                raise ValueError("output.format yuv requires model ycbcr")
            if self.format == "jpeg" and self.bit_depth != 8:
                raise ValueError("JPEG output supports only 8-bit RGB")
            if self.bit_depth == 32 and (self.format != "tiff" or self.transfer != "linear"):
                raise ValueError("32-bit float output requires linear RGB TIFF")
        else:
            if self.format != "yuv" or self.bit_depth not in (8, 16):
                raise ValueError("YCbCr output requires format yuv and bit_depth 8 or 16")
            if self.transfer == "linear":
                raise ValueError("standard Y'CbCr output requires a nonlinear RGB transfer")

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any] | None = None) -> OutputSpec:
        if section is None:
            return cls()
        if not isinstance(section, Mapping):
            raise ValueError("output must be a mapping")
        allowed = {
            "model", "color_space", "transfer", "gamma", "bit_depth", "format",
            "matrix", "range", "layout", "packing", "is_save",
        }
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"unknown output settings: {sorted(unknown, key=str)}")
        model = _choice("model", section.get("model", "rgb"), ("rgb", "ycbcr"))
        values = dict(section)
        values.setdefault("model", model)
        values.setdefault("transfer", "rec709" if model == "ycbcr" else "srgb")
        values.setdefault("format", "yuv" if model == "ycbcr" else "png")
        values.setdefault("range", "limited" if model == "ycbcr" else "full")
        if "range" in values:
            values["range_mode"] = values.pop("range")
        return cls(**values)

    def as_config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "color_space": self.color_space,
            "transfer": self.transfer,
            "bit_depth": self.bit_depth,
            "format": self.format,
            "matrix": self.matrix,
            "is_save": self.is_save,
        }
        if self.transfer == "power":
            result["gamma"] = self.gamma
        if self.model == "ycbcr":
            result.update(range=self.range_mode, layout=self.layout, packing=self.packing)
        return result

    @property
    def floating_linear(self) -> bool:
        return self.model == "rgb" and self.bit_depth == 32

    @property
    def suffix(self) -> str:
        return {"png": ".png", "jpeg": ".jpg", "tiff": ".tiff", "yuv": ".yuv"}[self.format]

    @property
    def nominal_range(self) -> tuple[float, float]:
        return (0.0, 1.0 if self.floating_linear else float(2**self.bit_depth - 1))

    @property
    def component_ranges(self) -> tuple[tuple[float, float], ...]:
        if self.model != "ycbcr" or self.range_mode == "full":
            return (self.nominal_range,) * 3
        shift = 2 ** (self.bit_depth - 8)
        return (
            (16.0 * shift, 235.0 * shift),
            (16.0 * shift, 240.0 * shift),
            (16.0 * shift, 240.0 * shift),
        )
