#!/usr/bin/env python
"""
Test preprocessing on actual stress granule microscopy data
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
from models import StressGranule16bitDataset
import torch

def analyze_real_data():
    """Analyze and test preprocessing on real stress granule data"""
    
    # Get list of image and mask files
    image_paths = sorted(glob('data/images/*.tif'))
    mask_paths = sorted(glob('data/masks/*.tif'))
    
    # Remove .gitkeep files if present
    image_paths = [p for p in image_paths if '.gitkeep' not in p]
    mask_paths = [p for p in mask_paths if '.gitkeep' not in p]
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    # Match images with masks
    matched_pairs = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        # Remove channel info if present
        img_name_clean = img_name.replace('_ch00_t00', '_t00')
        
        # Find corresponding mask
        for mask_path in mask_paths:
            mask_name = os.path.basename(mask_path)
            if img_name_clean in mask_name or mask_name.replace('MASK_', '') == img_name_clean:
                matched_pairs.append((img_path, mask_path))
                break
    
    print(f"Matched {len(matched_pairs)} image-mask pairs")
    
    if len(matched_pairs) == 0:
        print("No matched pairs found!")
        return
    
    # Analyze first few samples
    num_samples = min(3, len(matched_pairs))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        img_path, mask_path = matched_pairs[idx]
        print(f"\nAnalyzing sample {idx+1}:")
        print(f"  Image: {os.path.basename(img_path)}")
        print(f"  Mask: {os.path.basename(mask_path)}")
        
        # Load raw image with unchanged bit depth
        raw_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Print image properties
        print(f"  Image properties:")
        print(f"    - Shape: {raw_image.shape}")
        print(f"    - Data type: {raw_image.dtype}")
        print(f"    - Min value: {raw_image.min()}")
        print(f"    - Max value: {raw_image.max()}")
        print(f"    - Mean value: {raw_image.mean():.2f}")
        print(f"    - Non-zero pixels: {np.count_nonzero(raw_image)} / {raw_image.size}")
        
        print(f"  Mask properties:")
        print(f"    - Shape: {raw_mask.shape}")
        print(f"    - Data type: {raw_mask.dtype}")
        print(f"    - Unique values: {np.unique(raw_mask)}")
        
        # Create dataset with single sample for testing
        dataset = StressGranule16bitDataset(
            [img_path],
            [mask_path],
            target_size=(512, 512),  # Larger size to see more detail
            enhance_contrast=True,
            gaussian_sigma=1.7
        )
        
        # Load processed data
        processed_img, processed_mask = dataset[0]
        
        # Convert to numpy for visualization
        processed_img_np = processed_img.squeeze().numpy()
        processed_mask_np = processed_mask.squeeze().numpy()
        
        # Visualize
        # 1. Raw image (with percentile normalization for display)
        if raw_image.ndim == 3:
            display_raw = raw_image[:,:,0]  # Take first channel if multi-channel
        else:
            display_raw = raw_image
            
        # Use percentile for better visualization of 16-bit data
        p1, p99 = np.percentile(display_raw[display_raw > 0], [1, 99]) if np.any(display_raw > 0) else (0, 1)
        axes[idx, 0].imshow(display_raw, cmap='gray', vmin=p1, vmax=p99)
        axes[idx, 0].set_title(f'Raw Image (16-bit)\nRange: [{raw_image.min()}, {raw_image.max()}]')
        axes[idx, 0].axis('off')
        
        # 2. Raw mask
        if raw_mask.ndim == 3:
            display_mask = raw_mask[:,:,0]
        else:
            display_mask = raw_mask
        axes[idx, 1].imshow(display_mask, cmap='gray')
        axes[idx, 1].set_title(f'Raw Mask\nUnique: {np.unique(raw_mask)}')
        axes[idx, 1].axis('off')
        
        # 3. Processed image
        axes[idx, 2].imshow(processed_img_np, cmap='gray')
        axes[idx, 2].set_title(f'Processed Image\nRange: [{processed_img_np.min():.3f}, {processed_img_np.max():.3f}]')
        axes[idx, 2].axis('off')
        
        # 4. Processed mask
        axes[idx, 3].imshow(processed_mask_np, cmap='gray')
        axes[idx, 3].set_title(f'Processed Mask\nBinary: {np.unique(processed_mask_np)}')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_data_preprocessing_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create histogram analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Analyze intensity distribution of first image
    img_path, mask_path = matched_pairs[0]
    raw_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if raw_image.ndim == 3:
        raw_image = raw_image[:,:,0]
    
    # Histogram of raw values
    axes[0, 0].hist(raw_image.flatten(), bins=100, alpha=0.7, log=True)
    axes[0, 0].set_title('Raw Image Intensity Distribution (log scale)')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Count (log)')
    axes[0, 0].axvline(x=raw_image.mean(), color='r', linestyle='--', label=f'Mean: {raw_image.mean():.1f}')
    axes[0, 0].legend()
    
    # Histogram of non-zero values only
    non_zero_vals = raw_image[raw_image > 0]
    if len(non_zero_vals) > 0:
        axes[0, 1].hist(non_zero_vals, bins=100, alpha=0.7)
        axes[0, 1].set_title('Non-zero Pixel Distribution')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].axvline(x=non_zero_vals.mean(), color='r', linestyle='--', label=f'Mean: {non_zero_vals.mean():.1f}')
        axes[0, 1].legend()
    
    # Test preprocessing with different parameters
    print("\n\nTesting different preprocessing parameters...")
    
    # Test without contrast enhancement
    dataset_no_enhance = StressGranule16bitDataset(
        [img_path],
        [mask_path],
        target_size=(256, 256),
        enhance_contrast=False,
        gaussian_sigma=1.7
    )
    
    img_no_enhance, _ = dataset_no_enhance[0]
    img_no_enhance_np = img_no_enhance.squeeze().numpy()
    
    axes[1, 0].imshow(img_no_enhance_np, cmap='gray')
    axes[1, 0].set_title('Without Contrast Enhancement')
    axes[1, 0].axis('off')
    
    # Test without Gaussian blur
    dataset_no_blur = StressGranule16bitDataset(
        [img_path],
        [mask_path],
        target_size=(256, 256),
        enhance_contrast=True,
        gaussian_sigma=0
    )
    
    img_no_blur, _ = dataset_no_blur[0]
    img_no_blur_np = img_no_blur.squeeze().numpy()
    
    axes[1, 1].imshow(img_no_blur_np, cmap='gray')
    axes[1, 1].set_title('Without Gaussian Blur')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n\nResults saved to:")
    print("  - real_data_preprocessing_test.png")
    print("  - preprocessing_analysis.png")

if __name__ == "__main__":
    analyze_real_data() 