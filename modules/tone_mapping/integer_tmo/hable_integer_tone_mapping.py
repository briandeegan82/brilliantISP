"""
Hable (Uncharted 2) integer/LUT tone mapping for production-style ISPs.

Uses precomputed LUT for the Hable filmic curve.
No float ops in the hot path - suitable for hardware implementation.
"""
import numpy as np
from util.debug_utils import get_debug_logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def _hable_partial(x: np.ndarray, A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30) -> np.ndarray:
    """Uncharted 2 partial: ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F"""
    num = x * (A * x + C * B) + D * E
    den = np.maximum(x * (A * x + B) + D * F, 1e-10)
    return num / den - E / F


def _build_hable_lut(size=65536, exposure_bias=2.0, white_point=11.2, hdr_scale=1.0):
    """
    Build Hable filmic tone curve LUT.
    Input [0, hdr_scale] maps to [0, 1]; LUT index = normalized input.
    Lower hdr_scale = more highlight compression.
    """
    x_norm = np.linspace(0, 1, size, dtype=np.float64)
    x = x_norm * hdr_scale
    scaled = x * exposure_bias
    curr = _hable_partial(scaled)
    w_val = _hable_partial(np.array([white_point], dtype=np.float64))[0]
    white_scale = 1.0 / max(w_val, 1e-10)
    out = np.clip(curr * white_scale, 0.0, 1.0)
    return np.clip(np.round(out * 65535), 0, 65535).astype(np.uint16)


class HableIntegerToneMapping:
    """
    Hable tone mapping via precomputed LUT.
    Integer input → LUT lookup → integer output.
    """

    _lut_cache = {}

    def __init__(self, img, platform, sensor_info, params):
        self.img = np.asarray(img, dtype=np.uint32)
        self.platform = platform
        self.sensor_info = sensor_info
        self.params = params
        self.logger = get_debug_logger("HableIntegerToneMapping", config=platform)

        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_plot_curve = params.get("is_plot_curve", False)
        self.exposure_bias = params.get("exposure_bias", params.get("exposure_adjust", 2.0))
        self.white_point = params.get("white_point", 11.2)
        self.use_normalization = params.get("use_normalization", True)  # Per-image min-max
        self.normalize_output = params.get("normalize_output", False)  # Scale to use full output range
        self.hdr_scale = params.get("hdr_scale", 1.0)

        input_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.size > 0 and self.img.max() <= 65535:
            input_bits = 16
        self.input_max = 2**input_bits - 1
        self.output_max = 65535
        self.lut_size = 65536

        self._build_lut()
        
        # Calculate output normalization factor if enabled
        self.output_scale = 1.0
        if self.normalize_output:
            # Evaluate the LUT at maximum index to find theoretical max
            max_idx = self.lut_size - 1
            theoretical_max = int(self.lut[max_idx])
            
            if theoretical_max > 0:
                self.output_scale = self.output_max / theoretical_max
                self.logger.info(f"  Hable normalize output enabled: scaling by {self.output_scale:.3f}x")
                self.logger.info(f"  Theoretical max without normalization: {theoretical_max}")
            else:
                self.output_scale = 1.0

    def _build_lut(self):
        cache_key = (self.lut_size, self.exposure_bias, self.white_point, self.hdr_scale)
        if cache_key not in self._lut_cache:
            self._lut_cache[cache_key] = _build_hable_lut(
                self.lut_size, self.exposure_bias, self.white_point, self.hdr_scale
            )
        self.lut = self._lut_cache[cache_key]

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Hable curve via LUT lookup.
        Per-image normalization (use_normalization) adjusts for input range.
        Output normalization (normalize_output) scales to use full range.
        """
        x = np.asarray(x, dtype=np.int64)

        if self.use_normalization:
            img_min = int(np.min(x))
            img_max = int(np.max(x))
            range_val = max(1, img_max - img_min)
            idx = ((x - img_min) * (self.lut_size - 1)) // range_val
        else:
            idx = (x * (self.lut_size - 1)) // max(1, self.input_max)

        idx = np.clip(idx, 0, self.lut_size - 1)
        out = self.lut[idx]
        
        # Apply output normalization scaling if enabled
        if self.normalize_output and self.output_scale != 1.0:
            out = (out.astype(np.float64) * self.output_scale).astype(np.int64)
            out = np.clip(out, 0, self.output_max)
        
        return out.astype(np.uint16)

    def plot_tone_curve(self):
        """Plot and save the Hable integer tone mapping curve."""
        if not self.is_plot_curve:
            return
        
        try:
            # Generate input range
            x = np.linspace(0, self.input_max, 1000, dtype=np.int64)
            
            # Apply the curve
            y = np.zeros_like(x, dtype=np.uint16)
            for i, val in enumerate(x):
                if self.use_normalization:
                    idx_norm = int(((val - np.min(x)) * (self.lut_size - 1)) // max(1, np.max(x) - np.min(x)))
                else:
                    idx_norm = int((val * (self.lut_size - 1)) // max(1, self.input_max))
                idx_norm = np.clip(idx_norm, 0, self.lut_size - 1)
                y[i] = self.lut[idx_norm]
            
            # Create plot with actual values (not normalized)
            plt.figure(figsize=(10, 7))
            plt.plot(x, y, 'b-', linewidth=2, label='Hable (Uncharted 2) Filmic Curve (LUT)')
            # Linear reference
            x_linear = np.linspace(0, self.input_max, 100)
            y_linear = x_linear * (self.output_max / self.input_max)
            plt.plot(x_linear, y_linear, 'r--', linewidth=1, alpha=0.5, label='Linear (no tone mapping)')
            plt.grid(True, alpha=0.3)
            plt.xlabel(f'Input (0 to {self.input_max})', fontsize=12)
            plt.ylabel(f'Output (0 to {self.output_max})', fontsize=12)
            title = f'Hable Integer Tone Mapping Curve (LUT-based)\n'
            title += f'(exposure_bias={self.exposure_bias}, white_point={self.white_point}, hdr_scale={self.hdr_scale}'
            if self.normalize_output:
                title += f', output_norm: {self.output_scale:.3f}x)'
            else:
                title += ')'
            plt.title(title, fontsize=14)
            
            # Add text showing actual max output value
            actual_max = int(y.max())
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
            plot_filename = os.path.join(output_dir, 'tone_curve_hable_integer.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  Tone mapping curve saved to: {plot_filename}")
            self.logger.info(f"  Actual max output: {actual_max} ({actual_max/self.output_max*100:.1f}% of range)")
            
        except Exception as e:
            self.logger.warning(f"  Failed to plot tone curve: {e}")

    def execute(self) -> np.ndarray:
        """Execute Hable integer tone mapping. Returns uint16."""
        if not self.is_enable:
            scale = self.output_max / max(1, self.input_max)
            return (self.img.astype(np.float64) * scale).astype(np.uint16)

        # Plot the curve if debug option is enabled
        self.plot_tone_curve()
        
        return self._apply_curve(self.img)
