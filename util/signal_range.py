from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class LinearCodeRange:
    bit_depth: int
    origin: str

    @property
    def maximum(self) -> int:
        return 2**self.bit_depth - 1


def _bits(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an integer in 1..32")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or not 1 <= number <= 32:
        raise ValueError(f"{name} must be an integer in 1..32")
    return int(number)


def _enabled(section: Mapping[str, Any], name: str) -> bool:
    value = section.get("is_enable", False)
    if type(value) is not bool:
        raise ValueError(f"{name}.is_enable must be a boolean")
    return value


def linear_code_range(config: Mapping[str, Any]) -> LinearCodeRange:
    sensor = config.get("sensor_info", {})
    companding = config.get("companding", {})
    oecf = config.get("oecf", {})
    if not all(isinstance(section, Mapping) for section in (sensor, companding, oecf)):
        raise ValueError("sensor_info, companding and oecf must be mappings")
    sensor_bits = _bits(sensor.get("bit_depth"), "sensor_info.bit_depth")
    result = LinearCodeRange(sensor_bits, "sensor")
    if _enabled(companding, "companding"):
        result = LinearCodeRange(_bits(sensor.get("hdr_bit_depth", sensor_bits), "sensor_info.hdr_bit_depth"), "decompanded")
    if _enabled(oecf, "oecf") and oecf.get("r_lut") is not None:
        result = LinearCodeRange(_bits(oecf.get("output_bit_depth", sensor_bits), "oecf.output_bit_depth"), "oecf_lut")
    return result
