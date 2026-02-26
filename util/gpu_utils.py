"""
GPU acceleration utilities for OpenCV operations using UMat and direct CUDA.
Provides helper functions for common image processing operations with GPU acceleration.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Union

def is_gpu_available() -> bool:
    """Check if CUDA-enabled GPU is available for OpenCV operations."""
    return cv2.cuda.getCudaEnabledDeviceCount() > 0

def should_use_gpu(img_size: Tuple[int, int], operation: str) -> bool:
    """
    Determine if GPU acceleration should be used based on image size and operation type.
    Updated based on CUDA benchmark results.
    
    Args:
        img_size: Image dimensions (height, width)
        operation: Type of operation ('resize', 'bilateral_filter', 'filter2d', 'gaussian_blur')
        
    Returns:
        True if GPU should be used, False otherwise
    """
    if not is_gpu_available():
        return False
    
    # Calculate image area
    area = img_size[0] * img_size[1]
    
    # Operation-specific thresholds based on CUDA benchmark results
    thresholds = {
        'gaussian_blur': 100000,  # Always beneficial (7x speedup observed)
        'resize': 4000000,  # Only for large images due to transfer overhead
        'bilateral_filter': 8000000,  # Only for very large images
        'filter2d': 2000000,  # Moderate threshold
    }
    
    return area >= thresholds.get(operation, 2000000)

def to_umat(img: np.ndarray, use_gpu: bool = True) -> Union[cv2.UMat, np.ndarray]:
    """
    Convert numpy array to UMat if GPU is available and requested.
    
    Args:
        img: Input numpy array
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        UMat if GPU available and requested, otherwise original numpy array
    """
    if use_gpu and is_gpu_available():
        return cv2.UMat(img.astype(np.float32))
    return img

def from_umat(umat_or_array: Union[cv2.UMat, np.ndarray]) -> np.ndarray:
    """
    Convert UMat back to numpy array.
    
    Args:
        umat_or_array: UMat or numpy array
        
    Returns:
        Numpy array
    """
    if isinstance(umat_or_array, cv2.UMat):
        return umat_or_array.get()
    return umat_or_array

def gpu_resize(img: np.ndarray, size: Tuple[int, int], 
               interpolation: int = cv2.INTER_LINEAR, 
               use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated image resizing.
    Only beneficial for large images due to transfer overhead.
    
    Args:
        img: Input image
        size: Target size (width, height)
        interpolation: Interpolation method
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Resized image
    """
    if use_gpu and should_use_gpu(img.shape[:2], 'resize'):
        try:
            gpu_img = to_umat(img, use_gpu=True)
            gpu_result = cv2.resize(gpu_img, size, interpolation=interpolation)
            return from_umat(gpu_result)
        except Exception as e:
            logging.getLogger("BrilliantISP.GPU").warning(f"GPU resize failed, falling back to CPU: {e}")
    
    return cv2.resize(img, size, interpolation=interpolation)

def gpu_bilateral_filter(img: np.ndarray, d: int, sigma_color: float, 
                        sigma_space: float, use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated bilateral filtering.
    Only beneficial for very large images due to transfer overhead.
    
    Args:
        img: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Filtered image
    """
    if use_gpu and should_use_gpu(img.shape[:2], 'bilateral_filter'):
        try:
            # Try direct CUDA first (faster than UMat)
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img.astype(np.float32))
            gpu_result = cv2.cuda.bilateralFilter(gpu_img, d, sigma_color, sigma_space)
            return gpu_result.download()
        except Exception as e:
            # Fallback to UMat
            try:
                gpu_img = to_umat(img, use_gpu=True)
                gpu_result = cv2.bilateralFilter(gpu_img, d, sigma_color, sigma_space)
                return from_umat(gpu_result)
            except Exception as e2:
                logging.getLogger("BrilliantISP.GPU").warning(f"GPU bilateral filter failed, falling back to CPU: {e2}")
    
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def gpu_filter2d(img: np.ndarray, kernel: np.ndarray, 
                 use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated 2D filtering.
    
    Args:
        img: Input image
        kernel: Convolution kernel
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Filtered image
    """
    if use_gpu and should_use_gpu(img.shape[:2], 'filter2d'):
        try:
            gpu_img = to_umat(img, use_gpu=True)
            gpu_kernel = to_umat(kernel, use_gpu=True)
            gpu_result = cv2.filter2D(gpu_img, -1, gpu_kernel)
            return from_umat(gpu_result)
        except Exception as e:
            logging.getLogger("BrilliantISP.GPU").warning(f"GPU filter2D failed, falling back to CPU: {e}")
    
    return cv2.filter2D(img, -1, kernel)

def gpu_gaussian_blur(img: np.ndarray, ksize: Tuple[int, int], 
                     sigma_x: float, sigma_y: float = 0, 
                     use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated Gaussian blur.
    Always beneficial when GPU is available (7x speedup observed).
    
    Args:
        img: Input image
        ksize: Kernel size (width, height)
        sigma_x: Gaussian kernel standard deviation in X direction
        sigma_y: Gaussian kernel standard deviation in Y direction
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Blurred image
    """
    # Gaussian blur is always beneficial with GPU
    if use_gpu and is_gpu_available():
        try:
            gpu_img = to_umat(img, use_gpu=True)
            gpu_result = cv2.GaussianBlur(gpu_img, ksize, sigma_x, sigma_y)
            return from_umat(gpu_result)
        except Exception as e:
            logging.getLogger("BrilliantISP.GPU").warning(f"GPU Gaussian blur failed, falling back to CPU: {e}")
    
    return cv2.GaussianBlur(img, ksize, sigma_x, sigma_y)

def gpu_pipeline_optimized(operations: list, img: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Optimized GPU pipeline that keeps data in GPU memory between operations.
    Updated to use direct CUDA where beneficial.
    
    Args:
        operations: List of (function, args, kwargs) tuples for operations
        img: Input image
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Processed image
    """
    if not use_gpu or not is_gpu_available():
        # CPU fallback
        result = img
        for func, args, kwargs in operations:
            result = func(result, *args, **kwargs)
        return result
    
    try:
        # Convert to GPU once
        gpu_img = to_umat(img, use_gpu=True)
        
        # Apply all operations on GPU
        for func, args, kwargs in operations:
            # Handle different function signatures
            if func == cv2.resize:
                gpu_img = func(gpu_img, *args, **kwargs)
            elif func == cv2.bilateralFilter:
                # Try direct CUDA for bilateral filter
                try:
                    cuda_img = cv2.cuda_GpuMat()
                    cuda_img.upload(from_umat(gpu_img))
                    cuda_result = cv2.cuda.bilateralFilter(cuda_img, *args, **kwargs)
                    gpu_img = to_umat(cuda_result.download(), use_gpu=True)
                except:
                    gpu_img = func(gpu_img, *args, **kwargs)
            elif func == cv2.filter2D:
                gpu_img = func(gpu_img, *args, **kwargs)
            elif func == cv2.GaussianBlur:
                gpu_img = func(gpu_img, *args, **kwargs)
            else:
                # For other operations, convert back to CPU
                cpu_img = from_umat(gpu_img)
                cpu_result = func(cpu_img, *args, **kwargs)
                gpu_img = to_umat(cpu_result, use_gpu=True)
        
        # Convert back to CPU at the end
        return from_umat(gpu_img)
        
    except Exception as e:
        logging.getLogger("BrilliantISP.GPU").warning(f"GPU pipeline failed, falling back to CPU: {e}")
        # CPU fallback
        result = img
        for func, args, kwargs in operations:
            result = func(result, *args, **kwargs)
        return result

def benchmark_gpu_vs_cpu(func_gpu, func_cpu, *args, iterations: int = 10):
    """
    Benchmark GPU vs CPU performance for a given function.
    
    Args:
        func_gpu: GPU version of the function
        func_cpu: CPU version of the function
        *args: Arguments to pass to both functions
        iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Warm up
    for _ in range(3):
        func_cpu(*args)
        if is_gpu_available():
            func_gpu(*args)
    
    # CPU timing
    cpu_times = []
    for _ in range(iterations):
        start = time.time()
        func_cpu(*args)
        cpu_times.append(time.time() - start)
    
    # GPU timing
    gpu_times = []
    if is_gpu_available():
        for _ in range(iterations):
            start = time.time()
            func_gpu(*args)
            gpu_times.append(time.time() - start)
    
    return {
        'cpu_avg': np.mean(cpu_times),
        'cpu_std': np.std(cpu_times),
        'gpu_avg': np.mean(gpu_times) if gpu_times else None,
        'gpu_std': np.std(gpu_times) if gpu_times else None,
        'speedup': np.mean(cpu_times) / np.mean(gpu_times) if gpu_times else None
    }
