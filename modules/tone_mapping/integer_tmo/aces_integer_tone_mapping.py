"""
ACES integer/LUT tone mapping for production-style ISPs.

Uses precomputed LUTs for the ACES filmic curve and sRGB gamma.
No float ops in the hot path - suitable for hardware implementation.
Based on ACES 1.0 RRT (Knarkowicz 2016 fitted) + sRGB ODT.
"""
import numpy as np
from util.debug_utils import get_debug_logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def _build_aces_rrt_lut(size=65536, hdr_scale=100.0):
    """
    Build ACES RRT (filmic) tone curve LUT.
    Formula: (x * (a*x + b)) / (x * (c*x + d) + e), Knarkowicz 2016.
    LUT index = normalized [0,1] maps to curve input [0, hdr_scale].
    Lower hdr_scale = more highlight compression (darker). Typical: 10-100.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x_norm = np.linspace(0, 1, size, dtype=np.float64)
    x = x_norm * hdr_scale
    out = (x * (a * x + b)) / (np.maximum(x * (c * x + d) + e, 1e-8))
    return np.clip(np.round(out * 65535), 0, 65535).astype(np.uint16)


def _build_srgb_gamma_lut(size=65536, gamma=2.4):
    """
    Build sRGB gamma LUT (linear to display).
    Linear segment: x <= 0.0031308 -> 12.92 * x
    Power segment: x > 0.0031308 -> 1.055 * x^(1/gamma) - 0.055
    """
    x = np.linspace(0, 1, size, dtype=np.float64)
    linear = x <= 0.0031308
    out = np.zeros(size, dtype=np.float64)
    out[linear] = 12.92 * x[linear]
    out[~linear] = 1.055 * np.power(x[~linear], 1.0 / gamma) - 0.055
    return np.clip(np.round(out * 65535), 0, 65535).astype(np.uint16)


class ACESIntegerToneMapping:
    """
    ACES tone mapping via precomputed LUTs.
    Integer input → LUT lookup → integer output.
    """

    # Class-level LUT cache (shared across instances)
    _rrt_lut_cache = {}
    _srgb_lut_cache = {}

    def __init__(self, img, platform, sensor_info, params):
        self.img = np.asarray(img, dtype=np.uint32)
        self.platform = platform
        self.sensor_info = sensor_info
        self.params = params
        self.logger = get_debug_logger("ACESIntegerToneMapping", config=platform)

        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_plot_curve = params.get("is_plot_curve", False)
        self.exposure = params.get("exposure_adjustment", params.get("exposure_adjust", 0.0))
        self.gamma = params.get("gamma", 2.4)
        self.apply_gamma = params.get("apply_odt_gamma", True)
        self.use_normalization = params.get("use_normalization", True)  # Per-image min-max
        self.normalize_output = params.get("normalize_output", False)  # Scale to use full output range
        self.hdr_scale = params.get("hdr_scale", 1.0)  # Lower = more compression

        input_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.size > 0 and self.img.max() <= 65535:
            input_bits = 16
        self.input_max = 2**input_bits - 1
        self.output_max = 65535
        self.lut_size = 65536

        self._build_luts()
        
        # Calculate output normalization factor if enabled
        self.output_scale = 1.0
        if self.normalize_output:
            # Evaluate the LUT at maximum index to find theoretical max
            max_idx = self.lut_size - 1
            theoretical_max_rrt = int(self.rrt_lut[max_idx])
            if self.apply_gamma:
                theoretical_max = int(self.srgb_lut[theoretical_max_rrt])
            else:
                theoretical_max = theoretical_max_rrt
            
            if theoretical_max > 0:
                self.output_scale = self.output_max / theoretical_max
                self.logger.info(f"  ACES normalize output enabled: scaling by {self.output_scale:.3f}x")
                self.logger.info(f"  Theoretical max without normalization: {theoretical_max}")
            else:
                self.output_scale = 1.0

    def _build_luts(self):
        """Build or retrieve cached LUTs."""
        gamma_key = (self.lut_size, self.gamma)
        if gamma_key not in self._srgb_lut_cache:
            self._srgb_lut_cache[gamma_key] = _build_srgb_gamma_lut(
                self.lut_size, self.gamma
            )
        rrt_key = (self.lut_size, self.hdr_scale)
        if rrt_key not in self._rrt_lut_cache:
            self._rrt_lut_cache[rrt_key] = _build_aces_rrt_lut(
                self.lut_size, self.hdr_scale
            )

        self.rrt_lut = self._rrt_lut_cache[(self.lut_size, self.hdr_scale)]
        self.srgb_lut = self._srgb_lut_cache[gamma_key]

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ACES RRT + optional sRGB gamma via LUT lookup.
        Per-image normalization (use_normalization) prevents washed-out output.
        Output normalization (normalize_output) scales to use full range.
        """
        x = np.asarray(x, dtype=np.int64)

        # Exposure: scale by 2^exp (integer)
        if abs(self.exposure) > 1e-6:
            exp_scale = int(round((2.0 ** self.exposure) * 65536))
            x = (x * exp_scale) >> 16
            x = np.clip(x, 0, self.input_max)

        # Map input to LUT index: per-image norm (matches float ACES) or absolute
        if self.use_normalization:
            img_min = int(np.min(x))
            img_max = int(np.max(x))
            range_val = max(1, img_max - img_min)
            
            # Estimate actual dynamic range in stops
            if img_min > 0:
                ratio = img_max / img_min
                actual_dr_stops = np.log2(ratio) if ratio > 1 else 0
                self.logger.info(f"  Image dynamic range: {img_min} to {img_max} (ratio: {ratio:.1f}:1)")
                self.logger.info(f"  Estimated DR: {actual_dr_stops:.2f} stops")
                self.logger.info(f"  Recommended hdr_scale: {actual_dr_stops:.1f} (current: {self.hdr_scale})")
            
            # Normalized [0,1]: (x - min) / range
            idx = ((x - img_min) * (self.lut_size - 1)) // range_val
        else:
            if self.input_max > 0:
                idx = (x * (self.lut_size - 1)) // self.input_max
            else:
                idx = np.zeros_like(x, dtype=np.int64)
        idx = np.clip(idx, 0, self.lut_size - 1)

        out = self.rrt_lut[idx]

        if self.apply_gamma:
            out = self.srgb_lut[out]
        
        # Apply output normalization scaling if enabled
        if self.normalize_output and self.output_scale != 1.0:
            out = (out.astype(np.float64) * self.output_scale).astype(np.int64)
            out = np.clip(out, 0, self.output_max)

        return out.astype(np.uint16)

    def plot_tone_curve(self):
        """Plot and save the ACES integer tone mapping curve."""
        if not self.is_plot_curve:
            return
        
        try:
            # Generate input range - show actual curve without per-image normalization
            x = np.linspace(0, self.input_max, 1000, dtype=np.int64)
            
            # Apply the curve WITHOUT per-image normalization for plotting
            # This shows the true curve behavior in the absolute input space
            y_rrt = np.zeros_like(x, dtype=np.uint16)
            y_full = np.zeros_like(x, dtype=np.uint16)
            
            for i, val in enumerate(x):
                # Map to LUT index using absolute values (no per-image normalization)
                idx = int((val * (self.lut_size - 1)) // max(1, self.input_max))
                idx = np.clip(idx, 0, self.lut_size - 1)
                y_rrt[i] = self.rrt_lut[idx]
                
                # RRT + gamma
                y_full[i] = self.srgb_lut[y_rrt[i]] if self.apply_gamma else y_rrt[i]
            
            # Apply output normalization if enabled (matches actual processing)
            if self.normalize_output and self.output_scale != 1.0:
                y_rrt = np.clip((y_rrt.astype(np.float64) * self.output_scale), 0, self.output_max).astype(np.uint16)
                y_full = np.clip((y_full.astype(np.float64) * self.output_scale), 0, self.output_max).astype(np.uint16)
            
            # Create plot with actual values
            plt.figure(figsize=(10, 7))
            plt.plot(x, y_rrt, 'b-', linewidth=2, label='ACES RRT (LUT, filmic)')
            if self.apply_gamma:
                plt.plot(x, y_full, 'g-', linewidth=2, label='ACES RRT + sRGB gamma (LUT)')
            # Linear reference
            x_linear = np.linspace(0, self.input_max, 100)
            y_linear = x_linear * (self.output_max / self.input_max)
            plt.plot(x_linear, y_linear, 'r--', linewidth=1, alpha=0.5, label='Linear (no tone mapping)')
            plt.grid(True, alpha=0.3)
            plt.xlabel(f'Input (0 to {self.input_max})', fontsize=12)
            plt.ylabel(f'Output (0 to {self.output_max})', fontsize=12)
            title = f'ACES Integer Tone Mapping Curve (LUT-based)\n'
            title += f'(hdr_scale={self.hdr_scale}, exposure={self.exposure:.1f} EV, gamma={self.gamma}'
            if self.use_normalization:
                title += ', per-image norm'
            if self.normalize_output:
                title += f', output_norm: {self.output_scale:.3f}x)'
            else:
                title += ')'
            plt.title(title, fontsize=14)
            
            # Add text showing actual max output value
            actual_max = int(y_full.max() if self.apply_gamma else y_rrt.max())
            plt.text(0.98, 0.02, f'Max output: {actual_max} ({actual_max/self.output_max*100:.1f}%)',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.legend(fontsize=10)
            plt.xlim([0, self.input_max])
            plt.ylim([0, self.output_max])
            
            # Save plot
            output_dir = self.platform.get('output_dir', 'module_output')
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = os.path.join(output_dir, 'tone_curve_aces_integer.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  Tone mapping curve saved to: {plot_filename}")
            self.logger.info(f"  Actual max output: {actual_max} ({actual_max/self.output_max*100:.1f}% of range)")
            
        except Exception as e:
            self.logger.warning(f"  Failed to plot tone curve: {e}")

    def execute(self) -> np.ndarray:
        """Execute ACES integer tone mapping. Returns uint16."""
        if not self.is_enable:
            scale = self.output_max / max(1, self.input_max)
            return (self.img.astype(np.float64) * scale).astype(np.uint16)

        # Plot the curve if debug option is enabled
        self.plot_tone_curve()
        
        return self._apply_curve(self.img)
