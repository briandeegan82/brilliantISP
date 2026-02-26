"""
Pipeline validation utilities for bit depth and linear domain assumptions.
Run with debug_enabled and DEBUG level to verify pipeline consistency.
"""
import numpy as np
import logging

LOG = logging.getLogger("BrilliantISP.PipelineValidation")


def validate_linear_assumption(name: str, img: np.ndarray, stage: str) -> None:
    """
    Log validation that data is in expected linear range.
    Call from pipeline stages when debug validation is enabled.
    """
    if img is None or img.size == 0:
        return
    mn, mx = np.min(img), np.max(img)
    LOG.debug(f"[{stage}] {name}: min={mn}, max={mx} (expected linear, no gamma)")


def validate_bit_depth(
    name: str, img: np.ndarray, expected_bits: int, stage: str
) -> bool:
    """Check that image values fit within expected bit depth."""
    if img is None or img.size == 0:
        return True
    max_val = 2**expected_bits - 1
    actual_max = np.max(img)
    if actual_max > max_val:
        LOG.warning(
            f"[{stage}] {name}: max={actual_max} exceeds {expected_bits}-bit range "
            f"(0-{max_val})"
        )
        return False
    return True
