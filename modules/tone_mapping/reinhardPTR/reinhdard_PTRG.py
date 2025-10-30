import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from util.utils import save_output_array
import time

from util.debug_utils import get_debug_logger
# Try to import GPU-accelerated version
try:
    from modules.hdr_durand.hdr_durand_fast_gpu import HDRDurandToneMappingGPU
    GPU_VERSION_AVAILABLE = True
except ImportError:
    GPU_VERSION_AVAILABLE = False

class reinhdard_PTRG:
    """
    HDR Durand Tone Mapping Algorithm Implementation with GPU acceleration
    """
    
    def __init__(self, img, platform, sensor_info, params):
        self.img = img.copy()
        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_debug = params.get("is_debug", False)
        self.sigma_space = params.get("sigma_space", 2.0)
        self.sigma_color = params.get("sigma_color", 0.4)
        self.contrast_factor = params.get("contrast_factor", 2.0)
        self.downsample_factor = params.get("downsample_factor", 4)
        self.output_bit_depth = sensor_info.get("output_bit_depth", 8)
        self.sensor_info = sensor_info
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("reinhdard_PTRG", config=self.platform)
        
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

        #%%
        # # Reinhard TMO   
    # --- Log-average luminance ---
        delta = 1e-4  # to avoid log(0)
        Lbar = np.exp(np.mean(np.log(self.img + delta)))
    
        # --- Compute Lmin and Lmax using percentiles ---
        Lmin = np.percentile(self.img, 1)   # 1st percentile ~ MaxQuart(Lw,0.01)
        Lmax = np.percentile(self.img, 99)  # 99th percentile ~ MaxQuart(Lw,0.99)
    
        # --- Compute alpha ---
        num = 2 * np.log2(Lbar) - np.log2(Lmax) - np.log2(Lmax)
        denom = np.log2(Lmax) - np.log2(Lmin)
        raistopower = num / denom
        alpha = 0.18 * (4 ** raistopower)
    
        # --- Compute Lwhite ---
        Lwhite = 1.5 * 2 ** (np.log2(Lmax) - np.log2(Lmin) - 5)
    
        # --- Scale luminance ---
        L = alpha * (self.img / Lbar)
    
        # --- Reinhard tone mapping ---
        output_luminance = (L * (1 + (L / (Lwhite ** 2)))) / (1 + L)

        return output_luminance
        
        #%%

    
    def save(self):
        if self.is_save:
            save_output_array(self.platform["in_file"], self.img, "Out_hdr_durand_", 
                              self.platform, self.sensor_info["bit_depth"], self.sensor_info["bayer_pattern"])
    
    def execute(self):
        if self.is_enable is True:
            self.logger.info("Executing HDR Durand Tone Mapping...")
            start = time.time()
            
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
        
