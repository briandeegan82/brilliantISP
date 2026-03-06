import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from util.utils import save_output_array
import time

from util.debug_utils import get_debug_logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
# Try to import GPU-accelerated version
try:
    from modules.tone_mapping.durand.hdr_durand_fast_gpu import HDRDurandToneMappingGPU
    GPU_VERSION_AVAILABLE = True
except ImportError:
    GPU_VERSION_AVAILABLE = False

class HDRDurandToneMapping:
    """
    HDR Durand Tone Mapping Algorithm Implementation with GPU acceleration
    """
    
    def __init__(self, img, platform, sensor_info, params):
        self.img = img.copy()
        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_debug = params.get("is_debug", False)
        self.is_plot_curve = params.get("is_plot_curve", False)
        self.sigma_space = params.get("sigma_space", 2.0)
        self.sigma_color = params.get("sigma_color", 0.4)
        self.contrast_factor = params.get("contrast_factor", 2.0)
        self.downsample_factor = params.get("downsample_factor", 4)
        self.output_bit_depth = sensor_info.get("output_bit_depth", 8)
        self.sensor_info = sensor_info
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("HDRDurandToneMapping", config=self.platform)
        
        # Check if GPU acceleration should be used
        self.use_gpu = False
        if GPU_VERSION_AVAILABLE:
            try:
                from util.gpu_utils import is_gpu_available, should_use_gpu
                self.use_gpu = (is_gpu_available() and 
                               should_use_gpu((sensor_info.get("height", 1000), sensor_info.get("width", 1000)), 'bilateral_filter'))
            except ImportError:
                self.use_gpu = False
    
    def normalize(self, image):
        """ Normalize image to [0,1] range."""
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def fast_bilateral_filter(self, image):
        """
        Approximate bilateral filtering using a downsampled approach.
        """
        small_img = zoom(image, 1 / self.downsample_factor, order=1)
        small_filtered = self.bilateral_filter(small_img, self.sigma_color, self.sigma_space)
        return zoom(small_filtered, self.downsample_factor, order=1)
    
    def bilateral_filter(self, image, sigma_color, sigma_space):
        """
        Custom bilateral filter using Gaussian filtering approximation.
        """
        spatial_filtered = gaussian_filter(image, sigma=sigma_space)
        intensity_diff = image - spatial_filtered
        range_kernel = np.exp(-0.5 * (intensity_diff / sigma_color) ** 2)
        return spatial_filtered + range_kernel * intensity_diff
    
    def apply_tone_mapping(self):
        """ Durand's tone mapping implementation. """
        # Convert to log domain
        epsilon = 1e-6  # Small value to avoid log(0)
        log_luminance = np.log10(self.img + epsilon)
    
        # Apply bilateral filter to get the base layer
        # For efficiency, we're using OpenCV's bilateral filter
        log_base = self.bilateral_filter(log_luminance.astype(np.float32), 
                                   self.sigma_color, 
                                   self.sigma_space)
    
        # Extract the detail layer
        log_detail = log_luminance - log_base
    
        # Compress the base layer (reduce contrast)
        compressed_log_base = log_base / self.contrast_factor
    
        # Recombine base and detail layers
        log_output = compressed_log_base + log_detail
    
        # Convert back from log domain
        output_luminance = np.power(10, log_output)
    
        # Normalize to [0, 1] range
        output_luminance = (output_luminance - np.min(output_luminance)) / (np.max(output_luminance) - np.min(output_luminance))
        return output_luminance

    def plot_tone_curve(self):
        """Plot and save a representative Durand tone mapping curve.
        
        Note: Durand is a local/adaptive tone mapper, so the actual mapping varies per pixel.
        This plot shows the typical compression effect on the base layer in log domain.
        """
        if not self.is_plot_curve:
            return
        
        try:
            # Create a synthetic HDR range to demonstrate the curve behavior
            # Log domain input (typical HDR range 0.01 to 100)
            log_input = np.linspace(-2, 2, 1000)  # Log10 range
            
            # Simulate base layer compression (what Durand does to the low-freq component)
            log_compressed = log_input / self.contrast_factor
            
            # Convert to linear domain for visualization
            linear_input = np.power(10, log_input)
            linear_output = np.power(10, log_compressed)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Log domain (base layer compression)
            ax1.plot(log_input, log_compressed, 'b-', linewidth=2, label='Compressed base layer')
            ax1.plot(log_input, log_input, 'r--', linewidth=1, alpha=0.5, label='Linear (no compression)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('Log10(Input)', fontsize=12)
            ax1.set_ylabel('Log10(Output)', fontsize=12)
            ax1.set_title(f'Durand Base Layer Compression (Log Domain)\n(contrast_factor={self.contrast_factor})', fontsize=12)
            ax1.legend(fontsize=10)
            
            # Plot 2: Linear domain (overall effect)
            ax2.plot(linear_input, linear_output, 'b-', linewidth=2, label='Durand tone curve')
            ax2.plot([linear_input.min(), linear_input.max()], 
                     [linear_input.min(), linear_input.max()], 
                     'r--', linewidth=1, alpha=0.5, label='Linear (no tone mapping)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('Input (cd/m²)', fontsize=12)
            ax2.set_ylabel('Output (cd/m²)', fontsize=12)
            ax2.set_title('Durand Overall Effect (Linear Domain)', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.set_xlim([linear_input.min(), linear_input.max()])
            ax2.set_ylim([linear_output.min(), linear_output.max()])
            
            plt.tight_layout()
            
            # Add note
            fig.text(0.5, 0.02, 
                    'Note: Durand is a local/adaptive tone mapper. The actual mapping varies per pixel based on local content.\n'
                    'This plot shows the typical compression behavior applied to the base (low-frequency) layer.',
                    ha='center', fontsize=9, style='italic', wrap=True)
            
            # Save plot
            output_dir = self.platform.get('output_dir', 'module_output')
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = os.path.join(output_dir, 'tone_curve_durand.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  Tone mapping curve saved to: {plot_filename}")
            
        except Exception as e:
            self.logger.warning(f"  Failed to plot tone curve: {e}")

    
    def save(self):
        if self.is_save:
            save_output_array(self.platform["in_file"], self.img, "Out_hdr_durand_", 
                              self.platform, self.sensor_info["bit_depth"], self.sensor_info["bayer_pattern"])
   
    def execute(self):
        if self.is_enable is True:
            self.logger.info("Executing HDR Durand Tone Mapping...")
            start = time.time()
            
            # Plot the curve if debug option is enabled
            self.plot_tone_curve()
            
            # Use GPU-accelerated version if available and beneficial
            if self.use_gpu and GPU_VERSION_AVAILABLE:
                try:
                    gpu_hdr = HDRDurandToneMappingGPU(self.img, self.platform, self.sensor_info, {
                        "is_enable": self.is_enable,
                        "is_save": self.is_save,
                        "is_debug": self.is_debug,
                        "sigma_space": self.sigma_space,
                        "sigma_color": self.sigma_color,
                        "contrast_factor": self.contrast_factor,
                        "downsample_factor": self.downsample_factor
                    })
                    self.img = gpu_hdr.execute()
                except Exception as e:
                    self.logger.info(f"  GPU HDR failed, falling back to CPU: {e}")
                    self.img = self.apply_tone_mapping()
            else:
                # Use CPU version
                self.img = self.apply_tone_mapping()
            
            execution_time = time.time() - start
            self.logger.info(f"Execution time: {execution_time:.3f}s")
            
        self.save()
        return self.img
        
