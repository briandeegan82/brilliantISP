#!/usr/bin/env python3
"""
Histogram and dynamic range utilities for ISP debugging
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def estimate_dynamic_range(image, percentile_low=0.1, percentile_high=99.9):
    """
    Estimate the dynamic range of an image in stops (EV)
    
    Args:
        image: Input image (numpy array)
        percentile_low: Lower percentile for noise floor estimation (default 0.1%)
        percentile_high: Upper percentile for highlight estimation (default 99.9%)
    
    Returns:
        dict with dynamic range statistics:
            - dynamic_range_ev: Dynamic range in stops (EV)
            - min_val: Minimum non-zero value
            - max_val: Maximum value
            - percentile_min: Lower percentile value
            - percentile_max: Upper percentile value
            - bit_depth_utilized: Effective bit depth used
    """
    # Flatten image if multi-channel
    if image.ndim > 2:
        # For RGB, use luminance
        img_flat = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        img_flat = image.flatten()
    
    # Remove zeros (dead pixels, black level)
    img_nonzero = img_flat[img_flat > 0]
    
    if len(img_nonzero) == 0:
        return {
            'dynamic_range_ev': 0.0,
            'min_val': 0,
            'max_val': 0,
            'percentile_min': 0,
            'percentile_max': 0,
            'bit_depth_utilized': 0
        }
    
    # Get min/max
    min_val = float(np.min(img_nonzero))
    max_val = float(np.max(img_flat))
    
    # Get percentile values (more robust to outliers)
    percentile_min = float(np.percentile(img_nonzero, percentile_low))
    percentile_max = float(np.percentile(img_flat, percentile_high))
    
    # Calculate dynamic range in stops (EV)
    # DR = log2(max/min)
    if percentile_min > 0:
        dynamic_range_ev = np.log2(percentile_max / percentile_min)
    else:
        dynamic_range_ev = 0.0
    
    # Calculate effective bit depth utilized
    bit_depth_utilized = np.log2(max_val + 1) if max_val > 0 else 0
    
    return {
        'dynamic_range_ev': dynamic_range_ev,
        'min_val': min_val,
        'max_val': max_val,
        'percentile_min': percentile_min,
        'percentile_max': percentile_max,
        'bit_depth_utilized': bit_depth_utilized
    }


def plot_histogram_comparison(input_image, output_image, output_dir="module_output", 
                               filename="histogram_comparison.png", input_label="Input (after decompanding)",
                               output_label="Output", show_log=True):
    """
    Plot histograms of input and output images side by side with dynamic range info
    
    Args:
        input_image: Input image (numpy array)
        output_image: Output image (numpy array)
        output_dir: Directory to save the plot
        filename: Output filename
        input_label: Label for input image
        output_label: Label for output image
        show_log: Whether to show log scale histogram as well
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Estimate dynamic range
    input_dr = estimate_dynamic_range(input_image)
    output_dr = estimate_dynamic_range(output_image)
    
    # Prepare images for histogram
    if input_image.ndim > 2:
        # Multi-channel image - compute luminance
        input_luma = 0.299 * input_image[:, :, 0] + 0.587 * input_image[:, :, 1] + 0.114 * input_image[:, :, 2]
    else:
        input_luma = input_image.flatten()
    
    if output_image.ndim > 2:
        # Multi-channel image - compute luminance
        output_luma = 0.299 * output_image[:, :, 0] + 0.587 * output_image[:, :, 1] + 0.114 * output_image[:, :, 2]
    else:
        output_luma = output_image.flatten()
    
    # Create figure
    if show_log:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = [axes]
    
    # Plot input histogram (linear scale)
    ax = axes[0][0] if show_log else axes[0]
    ax.hist(input_luma.flatten(), bins=256, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title(f'{input_label}\nDynamic Range: {input_dr["dynamic_range_ev"]:.2f} EV')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Min: {input_dr["min_val"]:.0f}\n'
                  f'Max: {input_dr["max_val"]:.0f}\n'
                  f'P{0.1}: {input_dr["percentile_min"]:.0f}\n'
                  f'P{99.9}: {input_dr["percentile_max"]:.0f}\n'
                  f'Bit depth used: {input_dr["bit_depth_utilized"]:.1f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')
    
    # Plot output histogram (linear scale)
    ax = axes[0][1] if show_log else axes[1]
    ax.hist(output_luma.flatten(), bins=256, color='green', alpha=0.7, edgecolor='black')
    ax.set_title(f'{output_label}\nDynamic Range: {output_dr["dynamic_range_ev"]:.2f} EV')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Min: {output_dr["min_val"]:.0f}\n'
                  f'Max: {output_dr["max_val"]:.0f}\n'
                  f'P{0.1}: {output_dr["percentile_min"]:.0f}\n'
                  f'P{99.9}: {output_dr["percentile_max"]:.0f}\n'
                  f'Bit depth used: {output_dr["bit_depth_utilized"]:.1f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')
    
    if show_log:
        # Plot input histogram (log scale)
        ax = axes[1][0]
        ax.hist(input_luma.flatten(), bins=256, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f'{input_label} (Log Scale)')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency (log)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot output histogram (log scale)
        ax = axes[1][1]
        ax.hist(output_luma.flatten(), bins=256, color='green', alpha=0.7, edgecolor='black')
        ax.set_title(f'{output_label} (Log Scale)')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency (log)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return input_dr, output_dr


def plot_single_histogram(image, output_dir="module_output", filename="histogram.png",
                          title="Image Histogram", show_log=True, show_channels=False):
    """
    Plot histogram of a single image with dynamic range info
    
    Args:
        image: Input image (numpy array)
        output_dir: Directory to save the plot
        filename: Output filename
        title: Plot title
        show_log: Whether to show log scale histogram as well
        show_channels: For RGB images, show separate R/G/B histograms
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Estimate dynamic range
    dr = estimate_dynamic_range(image)
    
    # Create figure
    if image.ndim > 2 and show_channels:
        if show_log:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes = [[axes[0]], [axes[1]]]
        
        # Plot RGB channels
        colors = ['red', 'green', 'blue']
        channel_names = ['Red', 'Green', 'Blue']
        
        ax = axes[0][0]
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            ax.hist(image[:, :, i].flatten(), bins=256, color=color, alpha=0.5, 
                   label=name, edgecolor='black')
        ax.set_title(f'{title} - RGB Channels\nDynamic Range: {dr["dynamic_range_ev"]:.2f} EV')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot luminance
        luma = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        ax = axes[0][1]
        ax.hist(luma.flatten(), bins=256, color='gray', alpha=0.7, edgecolor='black')
        ax.set_title(f'{title} - Luminance')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        if show_log:
            # RGB log scale
            ax = axes[1][0]
            for i, (color, name) in enumerate(zip(colors, channel_names)):
                ax.hist(image[:, :, i].flatten(), bins=256, color=color, alpha=0.5, 
                       label=name, edgecolor='black')
            ax.set_title(f'{title} - RGB Channels (Log Scale)')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency (log)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Luminance log scale
            ax = axes[1][1]
            ax.hist(luma.flatten(), bins=256, color='gray', alpha=0.7, edgecolor='black')
            ax.set_title(f'{title} - Luminance (Log Scale)')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency (log)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    else:
        # Single channel or luminance only
        if show_log:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            axes = [ax]
        
        if image.ndim > 2:
            # Multi-channel - compute luminance
            img_data = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            img_data = image
        
        # Linear scale
        ax = axes[0]
        ax.hist(img_data.flatten(), bins=256, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f'{title}\nDynamic Range: {dr["dynamic_range_ev"]:.2f} EV')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (f'Min: {dr["min_val"]:.0f}\n'
                      f'Max: {dr["max_val"]:.0f}\n'
                      f'P{0.1}: {dr["percentile_min"]:.0f}\n'
                      f'P{99.9}: {dr["percentile_max"]:.0f}\n'
                      f'Bit depth used: {dr["bit_depth_utilized"]:.1f}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')
        
        if show_log:
            # Log scale
            ax = axes[1]
            ax.hist(img_data.flatten(), bins=256, color='blue', alpha=0.7, edgecolor='black')
            ax.set_title(f'{title} (Log Scale)')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency (log)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return dr
