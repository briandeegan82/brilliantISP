"""
File: malvar_he_cutler_cupy.py
Description: CuPy-accelerated Malvar-He-Cutler algorithm for CFA interpolation
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
from scipy.signal import correlate2d

# Try to import CuPy, fall back to CPU if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using CPU implementation")


class MalvarCuPy:
    """
    CuPy-accelerated CFA interpolation or Demosaicing
    """

    def __init__(self, raw_in, masks):
        self.img = raw_in
        self.masks = masks
        self.use_gpu = CUPY_AVAILABLE and self._should_use_gpu()
        
        if self.use_gpu:
            print("  Using CuPy-accelerated Malvar-He-Cutler demosaicing")
        else:
            print("  Using CPU Malvar-He-Cutler demosaicing")

    def _should_use_gpu(self):
        """Determine if GPU acceleration should be used based on image size."""
        if not CUPY_AVAILABLE:
            return False
        
        # Use GPU for images larger than 1MP
        image_size = self.img.shape[0] * self.img.shape[1]
        return image_size > 1000000  # 1MP threshold

    def _create_filters_cpu(self):
        """Create filter kernels for CPU implementation."""
        # g_channel at r_channel & b_channel location
        g_at_r_and_b = (
            np.float32(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            * 0.125
        )

        # r_channel at green in r_channel row & b_channel column
        r_at_gr_and_b_at_gb = (
            np.float32(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            * 0.125
        )

        r_at_gb_and_b_at_gr = np.transpose(r_at_gr_and_b_at_gb)

        r_at_b_and_b_at_r = (
            np.float32(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            * 0.125
        )

        return g_at_r_and_b, r_at_gr_and_b_at_gb, r_at_gb_and_b_at_gr, r_at_b_and_b_at_r

    def apply_malvar_cpu(self):
        """CPU implementation of Malvar-He-Cutler demosaicing."""
        # 3D masks according to the given bayer
        mask_r, mask_g, mask_b = self.masks
        raw_in = np.float32(self.img)

        # Declaring 3D Demosaiced image
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))

        # Create filter kernels
        g_at_r_and_b, r_at_gr_and_b_at_gb, r_at_gb_and_b_at_gr, r_at_b_and_b_at_r = self._create_filters_cpu()

        # Creating r_channel, g_channel & b_channel channels from raw_in
        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b

        # Creating g_channel channel first after applying g_at_r_and_b filter
        g_channel = np.where(
            np.logical_or(mask_r == 1, mask_b == 1),
            correlate2d(raw_in, g_at_r_and_b, mode="same", boundary="symm"),
            g_channel,
        )

        # Applying other linear filters
        rb_at_g_rbbr = correlate2d(
            raw_in, r_at_gr_and_b_at_gb, mode="same", boundary="symm"
        )
        rb_at_g_brrb = correlate2d(
            raw_in, r_at_gb_and_b_at_gr, mode="same", boundary="symm"
        )
        rb_at_gr_bbrr = correlate2d(
            raw_in, r_at_b_and_b_at_r, mode="same", boundary="symm"
        )

        # Extract row and column masks
        r_rows = np.transpose(np.any(mask_r == 1, axis=1)[np.newaxis]) * np.ones(
            r_channel.shape, dtype=np.float32
        )
        r_col = np.any(mask_r == 1, axis=0)[np.newaxis] * np.ones(
            r_channel.shape, dtype=np.float32
        )
        b_rows = np.transpose(np.any(mask_b == 1, axis=1)[np.newaxis]) * np.ones(
            b_channel.shape, dtype=np.float32
        )
        b_col = np.any(mask_b == 1, axis=0)[np.newaxis] * np.ones(
            b_channel.shape, dtype=np.float32
        )

        # Update R channel
        r_channel = np.where(
            np.logical_and(r_rows == 1, b_col == 1), rb_at_g_rbbr, r_channel
        )
        r_channel = np.where(
            np.logical_and(b_rows == 1, r_col == 1), rb_at_g_brrb, r_channel
        )

        # Update B channel
        b_channel = np.where(
            np.logical_and(b_rows == 1, r_col == 1), rb_at_g_rbbr, b_channel
        )
        b_channel = np.where(
            np.logical_and(r_rows == 1, b_col == 1), rb_at_g_brrb, b_channel
        )

        # Final updates
        r_channel = np.where(
            np.logical_and(b_rows == 1, b_col == 1), rb_at_gr_bbrr, r_channel
        )
        b_channel = np.where(
            np.logical_and(r_rows == 1, r_col == 1), rb_at_gr_bbrr, b_channel
        )

        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel

        return demos_out

    def apply_malvar_gpu(self):
        """Hybrid GPU/CPU implementation of Malvar-He-Cutler demosaicing using CuPy."""
        try:
            # 3D masks according to the given bayer
            mask_r, mask_g, mask_b = self.masks
            raw_in = np.float32(self.img)

            # Create filter kernels
            g_at_r_and_b, r_at_gr_and_b_at_gb, r_at_gb_and_b_at_gr, r_at_b_and_b_at_r = self._create_filters_cpu()

            # Use CPU for convolutions (avoid compilation issues)
            # Creating g_channel channel first after applying g_at_r_and_b filter
            g_filtered = correlate2d(raw_in, g_at_r_and_b, mode="same", boundary="symm")
            
            # Applying other linear filters
            rb_at_g_rbbr = correlate2d(
                raw_in, r_at_gr_and_b_at_gb, mode="same", boundary="symm"
            )
            rb_at_g_brrb = correlate2d(
                raw_in, r_at_gb_and_b_at_gr, mode="same", boundary="symm"
            )
            rb_at_gr_bbrr = correlate2d(
                raw_in, r_at_b_and_b_at_r, mode="same", boundary="symm"
            )

            # Move data to GPU for vectorized operations
            raw_in_gpu = cp.asarray(raw_in)
            mask_r_gpu = cp.asarray(mask_r)
            mask_g_gpu = cp.asarray(mask_g)
            mask_b_gpu = cp.asarray(mask_b)
            
            g_filtered_gpu = cp.asarray(g_filtered)
            rb_at_g_rbbr_gpu = cp.asarray(rb_at_g_rbbr)
            rb_at_g_brrb_gpu = cp.asarray(rb_at_g_brrb)
            rb_at_gr_bbrr_gpu = cp.asarray(rb_at_gr_bbrr)

            # Declaring 3D Demosaiced image on GPU
            demos_out_gpu = cp.empty((raw_in.shape[0], raw_in.shape[1], 3))

            # Creating r_channel, g_channel & b_channel channels from raw_in
            r_channel_gpu = raw_in_gpu * mask_r_gpu
            g_channel_gpu = raw_in_gpu * mask_g_gpu
            b_channel_gpu = raw_in_gpu * mask_b_gpu

            # Creating g_channel channel first after applying g_at_r_and_b filter
            g_channel_gpu = cp.where(
                cp.logical_or(mask_r_gpu == 1, mask_b_gpu == 1),
                g_filtered_gpu,
                g_channel_gpu,
            )

            # Extract row and column masks on GPU
            r_rows_gpu = cp.transpose(cp.any(mask_r_gpu == 1, axis=1)[cp.newaxis]) * cp.ones(
                r_channel_gpu.shape, dtype=cp.float32
            )
            r_col_gpu = cp.any(mask_r_gpu == 1, axis=0)[cp.newaxis] * cp.ones(
                r_channel_gpu.shape, dtype=cp.float32
            )
            b_rows_gpu = cp.transpose(cp.any(mask_b_gpu == 1, axis=1)[cp.newaxis]) * cp.ones(
                b_channel_gpu.shape, dtype=cp.float32
            )
            b_col_gpu = cp.any(mask_b_gpu == 1, axis=0)[cp.newaxis] * cp.ones(
                b_channel_gpu.shape, dtype=cp.float32
            )

            # Update R channel
            r_channel_gpu = cp.where(
                cp.logical_and(r_rows_gpu == 1, b_col_gpu == 1), rb_at_g_rbbr_gpu, r_channel_gpu
            )
            r_channel_gpu = cp.where(
                cp.logical_and(b_rows_gpu == 1, r_col_gpu == 1), rb_at_g_brrb_gpu, r_channel_gpu
            )

            # Update B channel
            b_channel_gpu = cp.where(
                cp.logical_and(b_rows_gpu == 1, r_col_gpu == 1), rb_at_g_rbbr_gpu, b_channel_gpu
            )
            b_channel_gpu = cp.where(
                cp.logical_and(r_rows_gpu == 1, b_col_gpu == 1), rb_at_g_brrb_gpu, b_channel_gpu
            )

            # Final updates
            r_channel_gpu = cp.where(
                cp.logical_and(b_rows_gpu == 1, b_col_gpu == 1), rb_at_gr_bbrr_gpu, r_channel_gpu
            )
            b_channel_gpu = cp.where(
                cp.logical_and(r_rows_gpu == 1, r_col_gpu == 1), rb_at_gr_bbrr_gpu, b_channel_gpu
            )

            demos_out_gpu[:, :, 0] = r_channel_gpu
            demos_out_gpu[:, :, 1] = g_channel_gpu
            demos_out_gpu[:, :, 2] = b_channel_gpu

            # Move result back to CPU
            return cp.asnumpy(demos_out_gpu)

        except Exception as e:
            print(f"  GPU processing failed: {e}, falling back to CPU")
            return self.apply_malvar_cpu()

    def apply_malvar(self):
        """
        Demosaicing the given raw image using Malvar-He-Cutler
        """
        if self.use_gpu:
            return self.apply_malvar_gpu()
        else:
            return self.apply_malvar_cpu()
