from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from util.raw_decode import DecodedRaw, RawFormat, decode_raw

DEFAULT_CACHE_BYTES = 512 * 1024 * 1024
_WB_SECTIONS = (
    "sensor_info", "crop", "dead_pixel_correction", "black_level_correction",
    "companding", "oecf", "digital_gain", "lens_shading_correction",
    "bayer_noise_reduction", "auto_white_balance", "white_balance",
)
_CCM_SECTIONS = _WB_SECTIONS + ("demosaic", "color_correction_matrix")
_TONE_SECTIONS = (
    "tone_mapping", "aces", "aces_integer", "hable", "hable_integer",
    "reinhard_integer", "integer_tmo", "hdr_durand", "durand",
)


def _json_bytes(value: Any) -> bytes:
    def scalar(v: Any) -> Any:
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(f"unsupported cache-key value {type(v).__name__}")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=scalar,
    ).encode("utf-8")


def stage_cache_key(
    stage_id: str,
    image: np.ndarray,
    *,
    source_identity: str,
    config: Mapping[str, Any],
    sensor_info: Mapping[str, Any],
    awb_gains: Any,
    digital_gain_index: int,
    processing_state: Any = None,
) -> str:
    """Key exact stage inputs and upstream/runtime state, not downstream edits."""
    from util.signal_range import linear_code_range

    if not source_identity:
        raise ValueError("stage caching requires an explicit source identity")
    if stage_id == "white_balance":
        sections = _WB_SECTIONS
        domain = ["linear_bayer_codes", linear_code_range(config).maximum]
    elif stage_id in ("color_correction_matrix", "color_correction_matrix_render"):
        sections = _CCM_SECTIONS
        if stage_id.endswith("_render"):
            sections += _TONE_SECTIONS
        domain = ["post_bayer_mapping_rgb" if stage_id.endswith("_render") else "scene_linear_rgb", 1]
    else:
        raise ValueError(f"stage caching is not defined for {stage_id!r}")
    arr = np.asarray(image)
    if not arr.size or arr.dtype.kind not in "uif":
        raise ValueError("cache keys require nonempty real arrays")
    header = {
        "recipe": "native_cpu_m6_v1", "stage": stage_id,
        "source": source_identity, "dtype": arr.dtype.str, "shape": list(arr.shape),
        "domain": domain, "output_precision": "float32", "backend": "native_cpu",
        "upstream": {key: config.get(key) for key in sections},
        "effective_sensor": dict(sensor_info),
        "awb_gains": awb_gains, "digital_gain_index": digital_gain_index,
        "state": asdict(processing_state) if processing_state is not None else None,
    }
    digest = hashlib.sha256(_json_bytes(header))
    digest.update(b"\0")
    buffer: Any = np.ascontiguousarray(arr)
    digest.update(memoryview(buffer).cast("B"))
    return f"stage:{stage_id}:{digest.hexdigest()}"


class ArrayLru:
    """Byte/count-bounded immutable array storage; not a total-RSS budget."""

    def __init__(self, budget_bytes: int = DEFAULT_CACHE_BYTES, *, max_entries: int = 128) -> None:
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("cache byte budget must be a nonnegative integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("cache entry limit must be a positive integer")
        self.budget_bytes = budget_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[np.ndarray, Any]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> tuple[np.ndarray, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return entry[0].view(), copy.deepcopy(entry[1])

    def get_array(self, key: str) -> np.ndarray | None:
        entry = self.get(key)
        return entry[0] if entry is not None else None

    def put_array(self, key: str, image: np.ndarray, *, metadata: Any = None) -> bool:
        arr = np.asarray(image)
        if not arr.size or arr.dtype.kind not in "uif" or not np.all(np.isfinite(arr)):
            raise ValueError("cache entries must be nonempty finite real arrays")
        if key in self._entries:
            old, _ = self._entries.pop(key)
            self._bytes -= old.nbytes
        if arr.nbytes > self.budget_bytes:
            return False
        while self._entries and (
            self._bytes + arr.nbytes > self.budget_bytes or len(self._entries) >= self.max_entries
        ):
            _, (old, _) = self._entries.popitem(last=False)
            self._bytes -= old.nbytes
            self.evictions += 1
        frozen = np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype).reshape(arr.shape)
        self._entries[key] = frozen, copy.deepcopy(metadata)
        self._bytes += frozen.nbytes
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def stats(self) -> dict[str, int | str]:
        return {
            "budget_bytes": self.budget_bytes, "array_bytes": self._bytes,
            "entries": len(self._entries), "hits": self.hits, "misses": self.misses,
            "evictions": self.evictions,
            "note": "Counts stored array bytes only; not Python overhead, processing working memory, or total RSS.",
        }


class CachedRawDecoder:
    """Bind immutable-source verification to a shared worker array LRU."""

    def __init__(self, store: ArrayLru, *, expected_fingerprint: str) -> None:
        self.store = store
        self.expected_fingerprint = expected_fingerprint

    def decode(self, path: str | Path, fmt: RawFormat) -> DecodedRaw:
        source = Path(path).resolve()
        before = source.stat()
        digest = hashlib.sha256()
        with source.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()
        if self.expected_fingerprint and fingerprint != self.expected_fingerprint:
            raise ValueError("staged source changed; reload the source before processing")
        identity = _json_bytes({"path": str(source), "format": asdict(fmt), "sha256": fingerprint})
        key = "raw:" + hashlib.sha256(identity).hexdigest()
        entry = self.store.get(key)
        if entry is None:
            decoded = decode_raw(source, fmt)
            metadata = {f.name: getattr(decoded, f.name) for f in fields(decoded) if f.name != "pixels"}
            self.store.put_array(key, decoded.pixels, metadata=metadata)
        else:
            pixels, metadata = entry
            decoded = DecodedRaw(pixels=pixels.copy(), **metadata)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("staged source changed during decode")
        return decoded
