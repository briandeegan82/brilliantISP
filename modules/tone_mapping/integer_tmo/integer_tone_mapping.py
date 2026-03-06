"""
Integer-native Reinhard tone mapping for production-style ISPs.

Implements the Reinhard global operator (2002) in integer arithmetic:
  L_out = L_in / (1 + L_in)
  
Generalized with knee and strength parameters:
  out = (in × out_max) / (in + white_point)
  where: white_point = knee × input_max / strength

Operates directly on integer input (uint32/uint16) without float normalization.
No float ops in the hot path - suitable for fixed-point / hardware implementation.
"""
import numpy as np
from util.debug_utils import get_debug_logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class IntegerReinhardToneMapping:
    """
    Integer-domain Reinhard tone mapping using rational curve.
    
    Based on Reinhard et al. 2002 "Photographic Tone Reproduction for Digital Images"
    Implements the global operator with adjustable white point (knee parameter).
    
    No float conversion - suitable for fixed-point / hardware implementation.
    """

    def __init__(self, img, platform, sensor_info, params):
        self.img = np.asarray(img, dtype=np.uint32)  # Luminance or raw (single channel)
        self.platform = platform
        self.sensor_info = sensor_info
        self.params = params
        self.logger = get_debug_logger("IntegerReinhardToneMapping", config=platform)

        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_plot_curve = params.get("is_plot_curve", False)

        # Reinhard curve parameters
        # knee = white point: controls where highlights start to compress
        # strength = modulates the white point scaling
        self.white_point = params.get("knee", 0.25)  # Reinhard white point (kept as 'knee' for compatibility)
        self.strength = params.get("strength", 1.0)
        self.dark_enh = params.get("dark_enh", 0.0)  # Shadow lift (0-1)
        self.normalize_output = params.get("normalize_output", False)  # Scale to use full output range

        input_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.max() <= 65535:
            input_bits = 16
        self.input_max = 2**input_bits - 1
        self.output_max = 65535  # Pipeline convention
        
        # Calculate normalization factor if enabled
        self.norm_scale = 1.0
        if self.normalize_output:
            # Calculate theoretical max output with current white_point/strength
            # At x=infinity, the Reinhard asymptotic limit is: out_max / (1 + white_point/strength)
            # For practical purposes, evaluate at input_max
            white_point_term = max(1, int(self.white_point * self.input_max / self.strength))
            theoretical_max = (self.input_max * self.output_max) // (self.input_max + white_point_term)
            if theoretical_max > 0:
                self.norm_scale = self.output_max / theoretical_max
                self.logger.info(f"  Reinhard normalize output enabled: scaling by {self.norm_scale:.3f}x")
                self.logger.info(f"  Theoretical max without normalization: {theoretical_max}")
            else:
                self.norm_scale = 1.0

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Reinhard tone curve in integer domain.
        
        Formula: out = (x × out_max) / (x + white_point_term)
        where: white_point_term = white_point × input_max / strength
        
        This is the Reinhard global operator adapted for integer arithmetic.
        With normalization enabled, the output is scaled to use the full range.
        """
        x = np.asarray(x, dtype=np.int64)
        # Avoid div by zero
        white_point_term = max(1, int(self.white_point * self.input_max / self.strength))
        out = (x * self.output_max) // (x + white_point_term)
        
        # Apply normalization scaling if enabled
        if self.normalize_output and self.norm_scale != 1.0:
            out = (out.astype(np.float64) * self.norm_scale).astype(np.int64)

        return np.clip(out, 0, self.output_max).astype(np.uint16)

    def plot_tone_curve(self):
        """Plot and save the Reinhard tone mapping curve."""
        if not self.is_plot_curve:
            return
        
        try:
            # Generate input range
            x = np.linspace(0, self.input_max, 1000)
            y = self._apply_curve(x)
            
            # Create plot with actual values (not normalized)
            plt.figure(figsize=(10, 7))
            plt.plot(x, y, 'b-', linewidth=2, label='Reinhard Tone Curve')
            # Linear reference scaled to output range
            x_linear = np.linspace(0, self.input_max, 100)
            y_linear = x_linear * (self.output_max / self.input_max)
            plt.plot(x_linear, y_linear, 'r--', linewidth=1, alpha=0.5, label='Linear (no tone mapping)')
            plt.grid(True, alpha=0.3)
            plt.xlabel(f'Input Luminance (0 to {self.input_max})', fontsize=12)
            plt.ylabel(f'Output Luminance (0 to {self.output_max})', fontsize=12)
            
            # Add normalization info to title
            title = f'Reinhard Global Operator (Integer)\n(white_point={self.white_point}, strength={self.strength}'
            if self.normalize_output:
                title += f', normalized: {self.norm_scale:.3f}x)'
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
            plot_filename = os.path.join(output_dir, 'tone_curve_reinhard.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  Reinhard tone curve saved to: {plot_filename}")
            self.logger.info(f"  Actual max output: {actual_max} ({actual_max/self.output_max*100:.1f}% of range)")
            
        except Exception as e:
            self.logger.warning(f"  Failed to plot tone curve: {e}")

    def execute(self) -> np.ndarray:
        """Execute Reinhard tone mapping. Returns uint16."""
        if not self.is_enable:
            # Passthrough: scale to output range
            scale = self.output_max / max(1, self.input_max)
            return (self.img.astype(np.float64) * scale).astype(np.uint16)

        # Plot the curve if debug option is enabled
        self.plot_tone_curve()
        
        return self._apply_curve(self.img)


# Alias for backward compatibility
IntegerToneMapping = IntegerReinhardToneMapping
