import cv2
import numpy as np

def durand_tone_mapping(hdr_image, sigma_space=10, sigma_color=0.4, contrast_factor=5, use_gpu=False):
    """
    Apply Durand's tone mapping algorithm to an HDR image.

    Parameters:
        hdr_image (numpy.ndarray): Input HDR image (float32 or float64).
        sigma_space (float): Spatial standard deviation for bilateral filtering.
        sigma_color (float): Color standard deviation for bilateral filtering.
        contrast_factor (float): Scaling factor for compressing the base layer.
        use_gpu (bool): Whether to use GPU acceleration via UMat.

    Returns:
        numpy.ndarray: Tone-mapped LDR image (normalized to [0, 1]).
    """
    # Check if GPU acceleration is available
    gpu_available = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    # Step 1: Convert to logarithmic domain
    log_hdr = np.log1p(hdr_image)

    # Step 2: Apply bilateral filtering to separate base and detail layers
    if gpu_available:
        try:
            # GPU-accelerated bilateral filtering using UMat
            gpu_log_hdr = cv2.UMat(log_hdr.astype(np.float32))
            gpu_base_layer = cv2.bilateralFilter(gpu_log_hdr, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            base_layer = gpu_base_layer.get()
        except Exception as e:
            print(f"GPU bilateral filtering failed, falling back to CPU: {e}")
            base_layer = cv2.bilateralFilter(log_hdr, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    else:
        base_layer = cv2.bilateralFilter(log_hdr, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Step 3: Compress the base layer
    compressed_base = base_layer / contrast_factor

    # Step 4: Recombine the layers
    detail_layer = log_hdr - base_layer
    log_ldr = compressed_base + detail_layer

    # Step 5: Convert back to linear domain
    ldr_image = np.expm1(log_ldr)

    # Normalize the result to the range [0, 1]
    ldr_image = (ldr_image - np.min(ldr_image)) / (np.max(ldr_image) - np.min(ldr_image))

    return ldr_image


def load_hdr_image(file_path):
    """
    Load an HDR image from a file (e.g., OpenEXR format).

    Parameters:
        file_path (str): Path to the HDR image file.

    Returns:
        numpy.ndarray: HDR image as a NumPy array.
    """
    # Load the HDR image using OpenCV
    hdr_image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if hdr_image is None:
        raise ValueError(f"Could not load HDR image from {file_path}")
    return hdr_image


def save_ldr_image(ldr_image, file_path):
    """
    Save an LDR image to a file (e.g., PNG format).

    Parameters:
        ldr_image (numpy.ndarray): LDR image (normalized to [0, 1]).
        file_path (str): Path to save the LDR image.
    """
    # Scale to 8-bit range and save
    ldr_image_8bit = (ldr_image * 255).astype(np.uint8)
    cv2.imwrite(file_path, ldr_image_8bit)


def main():
    # Path to the input HDR image (OpenEXR format)
    hdr_file_path = 'input_hdr.exr'

    # Path to save the output LDR image (PNG format)
    ldr_file_path = 'output_ldr.png'

    # Load the HDR image
    hdr_image = load_hdr_image(hdr_file_path)

    # Apply Durand's tone mapping
    ldr_image = durand_tone_mapping(hdr_image, sigma_space=10, sigma_color=0.4, contrast_factor=5)

    # Save the tone-mapped LDR image
    save_ldr_image(ldr_image, ldr_file_path)

    print(f"Tone-mapped image saved to {ldr_file_path}")


if __name__ == "__main__":
    main()