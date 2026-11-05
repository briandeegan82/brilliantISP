from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


_IO_KEYS = frozenset((
    "filename", "in_file", "out_file", "output_dir", "final_output_dir",
    "debug_log_file",
))


def _value_key(value):
    if isinstance(value, np.ndarray):
        contents = _value_key(value.tolist()) if value.dtype.hasobject else value.tobytes().hex()
        return ("array", value.dtype.str, value.shape, contents)
    if isinstance(value, np.generic):
        return ("numpy_scalar", value.dtype.str, value.tobytes().hex())
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("configuration mapping keys must be strings")
        return ("mapping", tuple((k, _value_key(value[k])) for k in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_value_key(v) for v in value))
    raise TypeError(f"unsupported configuration value type: {type(value).__name__}")


def group_key(config, *, shape, cfa, storage_dtype, capture_stages=(),
              diagnostics=False, profile_stages=False):
    if len(shape) != 2 or any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape
    ):
        raise ValueError("group geometry must be two positive integers")
    if cfa not in ("rggb", "grbg", "gbrg", "bggr"):
        raise ValueError("group CFA must be a canonical Bayer pattern")
    if storage_dtype not in ("uint8", "uint16", "uint32"):
        raise ValueError("group storage dtype must be unsigned Bayer codes")
    processing = dict(config)
    platform = dict(processing.get("platform", {}))
    processing["platform"] = {k: v for k, v in platform.items() if k not in _IO_KEYS}
    payload = (
        _value_key(processing), tuple(shape), cfa, storage_dtype,
        tuple(capture_stages), bool(diagnostics), bool(profile_stages),
    )
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
