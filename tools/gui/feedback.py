from __future__ import annotations

import math
from typing import Any


def feedback_edits(config: dict, feedback: dict, kind: str) -> dict[str, Any]:
    if kind == "awb":
        gains = feedback.get("awb_gains")
        if feedback.get("awb_status") != "estimated" or not isinstance(gains, (list, tuple)) or len(gains) != 2:
            raise ValueError("This snapshot has no estimated white-balance gains.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in gains):
            raise ValueError("Estimated white-balance gains must be finite and positive.")
        return {
            "white_balance.r_gain": gains[0], "white_balance.b_gain": gains[1],
            "white_balance.is_enable": True, "white_balance.is_auto": False,
            "auto_white_balance.is_enable": False,
        }
    if kind == "gain":
        index = feedback.get("digital_gain_index")
        gains = config.get("digital_gain", {}).get("gain_array")
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(gains, list) or not 0 <= index < len(gains):
            raise ValueError("This snapshot has no applicable applied digital-gain index.")
        return {
            "digital_gain.current_gain": index, "digital_gain.is_auto": False,
            "auto_exposure.is_enable": False,
        }
    raise ValueError("Unknown feedback action.")
