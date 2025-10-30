import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from util.utils import save_output_array
import time
from scipy.signal import convolve2d

from util.debug_utils import get_debug_logger
# Try to import GPU-accelerated version
try:
    from modules.hdr_durand.hdr_durand_fast_gpu import HDRDurandToneMappingGPU
    GPU_VERSION_AVAILABLE = True
except ImportError:
    GPU_VERSION_AVAILABLE = False

class reinhdard_PTRLocal:
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
    
    def reinhard_filter_gaussian(luminance_map, alpha, s):
        """
        Exact MATLAB-equivalent of ReinhardFilterGuassian.m
        Uses same convolution logic and zero-padding as MATLAB's conv2.
        """
        # Compute sigma
        sigma = float(alpha * s)
    
        # Compute kernel radius and size
        kernel_radius = int(np.ceil(2 * sigma))
        kernel_size = 2 * kernel_radius + 1
    
        # Create Gaussian kernel (1D)
        x = np.arange(-kernel_radius, kernel_radius + 1)
        gauss_kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss_kernel /= np.sum(gauss_kernel)
    
        # Horizontal convolution (zero padding)
        v1 = convolve2d(luminance_map, gauss_kernel[:, np.newaxis],
                        mode='same', boundary='fill', fillvalue=0)
    
        # Vertical convolution (zero padding)
        v1 = convolve2d(v1, gauss_kernel[np.newaxis, :],
                        mode='same', boundary='fill', fillvalue=0)
    
        return v1
        
    def ReinhardLocalTMO(L, a):
        phi = 8
        epsilon = 0.05
        scale = 11

        r, c = L.shape
        Vi = np.zeros((r, c, scale))
        s = 1.0

        # Step 1: Multi-scale Gaussian filtering
        for i in range(scale):
            Vi[:, :, i] = reinhdard_PTRLocal.reinhard_filter_gaussian(L, a, s)
            s *= 1.6

        # Step 2: Compute local contrast and select detail layer
        s = 1.0
        Ld = Vi[:, :, -1].copy()
        mask = np.zeros((r, c))

        for i in range(scale - 1):
            V1 = Vi[:, :, i]
            V2 = Vi[:, :, i + 1]
            V_denom_first = (2 ** phi * a) / (s ** 2)
            V = np.abs((V1 - V2) / (V_denom_first + V1))

            idx = np.where((V > epsilon) & (mask < 0.5))
            if idx[0].size > 0:
                mask[idx] = i + 1
                Ld[mask == (i + 1)] = V1[mask == (i + 1)]
            s *= 1.6

        return Ld
    
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
        LScaled = alpha * (self.img / Lbar)
        LdLocal=reinhdard_PTRLocal.ReinhardLocalTMO( LScaled,alpha )
            # --- Compute final tone-mapped luminance ---
        Ldnum = LScaled * (1 + (LScaled / (Lwhite ** 2)))/ (1 + LdLocal)
        lumLd = Ldnum / (1 + LdLocal)

        return Ldnum
        
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
        
