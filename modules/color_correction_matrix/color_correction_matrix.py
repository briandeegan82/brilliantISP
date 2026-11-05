from util.debug_utils import get_debug_logger

"""
File: color_correction_matrix.py
Description: Applies the 3x3 correction matrix on the image
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""
import time
import numpy as np

from util.isp_types import ColorCorrectionMatrixConfig, Float32Image, PlatformConfig, RGBImage, SensorInfo, UInt16Image
from util.utils import save_output_array
from util.native_array import as_float32


def _matrix_linear(img: np.ndarray, parm_ccm: ColorCorrectionMatrixConfig) -> np.ndarray:
    """Normalized RGB -> float64 matrix application, no clip/rescale.

    Shared by the legacy and optimized wrappers; uses the same
    O*A => A = ccm convention. Coefficients must be finite 3x3 rows.
    """
    f = np.asarray(img, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != 3 or not np.all(np.isfinite(f)):
        raise ValueError(
            f"CCM compute_linear requires a finite HWC3 array, got {f.shape}"
        )
    rows = [
        np.asarray(parm_ccm["corrected_red"], dtype=np.float64),
        np.asarray(parm_ccm["corrected_green"], dtype=np.float64),
        np.asarray(parm_ccm["corrected_blue"], dtype=np.float64),
    ]
    mat = np.stack(rows)
    if mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
        raise ValueError("CCM matrix must be finite 3x3 rows")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        out = np.matmul(f.reshape((-1, 3)), mat.transpose()).reshape(f.shape)
    return out


class ColorCorrectionMatrix:
    "Apply the color correction 3x3 matrix"

    def __init__(
        self,
        img: RGBImage,
        platform: PlatformConfig,
        sensor_info: SensorInfo,
        parm_ccm: ColorCorrectionMatrixConfig,
    ) -> None:
        self.img = img
        self.enable = parm_ccm["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        # CCM operates on the 16-bit linear RGB produced by demosaic, so normalise
        # by pipeline_rgb_bit_depth (typically 16 → 65535), not output_bit_depth (8).
        self.pipeline_bits = sensor_info.get("pipeline_rgb_bit_depth", 16)
        self.output_bit_depth = sensor_info.get("output_bit_depth", 16)
        self.ccm_mat = None
        self.is_save = parm_ccm["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("ColorCorrectionMatrix", config=self.platform)

    def apply_ccm(self) -> UInt16Image:
        """
        Apply CCM Params
        """
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])

        self.ccm_mat = np.array([r_1, r_2, r_3], dtype=np.float32)

        pipeline_max = float(2**self.pipeline_bits - 1)

        # Normalise 16-bit pipeline data to [0, 1]
        self.img = self.img.astype(np.float32) / pipeline_max

        # convert to nx3
        img1 = self.img.reshape(((self.img.shape[0] * self.img.shape[1], 3)))

        # keeping imatest convention of colum sum to 1 mat. O*A => A = ccm
        out = np.matmul(img1, self.ccm_mat.transpose())

        # clipping after ccm is must to eliminate neg values
        out = np.clip(out, 0, 1).astype(np.float32)

        # convert back to 16-bit pipeline range
        out = out.reshape(self.img.shape).astype(self.img.dtype)
        out = (out * pipeline_max).astype(np.uint16)

        return out

    def compute_linear(self) -> Float32Image:
        """Native linear path on normalized RGB: matrix only, no clip/rescale.

        Disabled is an exact identity pass and never reads the matrix
        parameters.
        """
        if not self.enable:
            return as_float32(self.img, name="color_correction_matrix")
        out = _matrix_linear(self.img, self.parm_ccm)
        return as_float32(out, name="color_correction_matrix")

    def save(self) -> None:
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self) -> RGBImage | UInt16Image:
        """Execute ccm if enabled."""
        self.logger.info(f"Color Correction Matrix = {self.enable}")

        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm()
            self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img
