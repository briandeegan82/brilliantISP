from util.debug_utils import get_debug_logger

"""
File: oecf.py
Description: Implements the opto electronic conversion function as a LUT
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class OECF:
    "Optical Electronic Conversion Function - correction"

    def __init__(self, img, platform, sensor_info, parm_oecf):
        self.img = img
        self.enable = parm_oecf["is_enable"]
        self.sensor_info = sensor_info
        self.parm_oecf = parm_oecf
        self.is_save = parm_oecf["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("OECF", config=self.platform)

    def apply_oecf(self):
        """Execute OECF. Direct declared-code LUT lookup; missing r_lut is identity."""
        raw = np.asarray(self.img)
        if raw.ndim != 2 or raw.size == 0 or raw.dtype.kind not in "uif" or not np.all(np.isfinite(raw)):
            raise ValueError("oecf input must be a nonempty finite real code plane")

        # Identity LUT when r_lut not in config (passthrough); the identity
        # preserves the INPUT range (PWC-enabled? hdr : raw bit depth).
        r_lut = self.parm_oecf.get("r_lut")
        if r_lut is None:
            bpp = self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"])
            max_val = 2**bpp - 1
            return np.clip(raw, 0, max_val).astype(np.uint32 if max_val > 65535 else np.uint16, copy=False)

        # get config parm
        bayer = str(self.sensor_info["bayer_pattern"]).lower()
        layouts = {
            "rggb": ("r_lut", "gr_lut", "gb_lut", "bl_lut"),
            "bggr": ("bl_lut", "gb_lut", "gr_lut", "r_lut"),
            "grbg": ("gr_lut", "r_lut", "bl_lut", "gb_lut"),
            "gbrg": ("gb_lut", "bl_lut", "r_lut", "gr_lut"),
        }
        if bayer not in layouts:
            raise ValueError(f"Unsupported oecf Bayer pattern {bayer!r}")
        bits = self.parm_oecf.get("output_bit_depth", self.sensor_info["bit_depth"])
        if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, float, np.integer, np.floating)) \
                or not np.isfinite(bits) or not 1 <= bits <= 32 or int(bits) != bits:
            raise ValueError("oecf.output_bit_depth must be an integer in 1..32")
        out_bits = int(bits)
        out_max = 2**out_bits - 1
        if np.any(raw < 0) or np.any(raw != np.floor(raw)):
            raise ValueError("oecf input codes must be nonnegative integers")

        def _declared_lut(name: str, plane: np.ndarray) -> np.ndarray:
            vals = np.asarray(self.parm_oecf.get(name, r_lut))
            if vals.ndim != 1 or vals.size == 0 or vals.dtype.kind not in "uif":
                raise ValueError(f"oecf {name} must be a nonempty 1-D LUT")
            if not np.all(np.isfinite(vals)) or np.any(vals < 0) \
                    or np.any(vals > out_max) or np.any(vals != np.floor(vals)):
                raise ValueError(
                    f"oecf {name} entries must be finite nonnegative integral <= {out_max}")
            if int(plane.max()) >= vals.size:
                raise ValueError(
                    f"oecf {name} has no entry for declared input code")
            return vals.astype(np.uint64)

        raw_oecf = np.empty(raw.shape, dtype=np.uint32 if out_bits > 16 else np.uint16)
        for site, name in enumerate(layouts[bayer]):
            row, col = divmod(site, 2)
            plane = raw[row::2, col::2]
            if plane.size:
                lut = _declared_lut(name, plane)
                raw_oecf[row::2, col::2] = lut[plane.astype(np.intp)]
        return raw_oecf

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_oecf_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute OECF if enabled."""
        self.logger.info(f"Optical Electronic Conversion Function = {self.enable}")

        if self.enable:
            start = time.time()
            oecf_out = self.apply_oecf()
            execution_time = time.time() - start
            self.logger.info(f"Execution time: {execution_time:.3f}s")
            self.img = oecf_out
        self.save()
        return self.img
