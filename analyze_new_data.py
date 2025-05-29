#!/usr/bin/env python
"""Analyze the new data to check stress granule coverage"""

import cv2
import numpy as np
import os
from train_small_test import match_images_and_masks

def analyze_stress_granule_coverage():
    # Match images and masks
    matched_pairs = match_images_and_masks('data/images', 'data/masks', max_samples=4)
    
    print("Analyzing stress granule coverage in new data:")
    print("=" * 60)
    
    total_pixels = 0
    total_sg_pixels = 0
    
    for img_path, mask_path in matched_pairs:
        # Extract sample name
        img_name = os.path.basename(img_path)
        sample_type = "Noco" if "Noco" in img_name else "Control"
        cell_type = "A549" if "A549" in img_name else "U2OS"
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate coverage
        num_pixels = mask.size
        sg_pixels = np.sum(mask > 127)
        percentage = (sg_pixels / num_pixels) * 100
        
        total_pixels += num_pixels
        total_sg_pixels += sg_pixels
        
        print(f"\n{cell_type} {sample_type}:")
        print(f"  Total pixels: {num_pixels:,}")
        print(f"  Stress granule pixels: {sg_pixels:,}")
        print(f"  Coverage: {percentage:.2f}%")
    
    # Overall statistics
    overall_percentage = (total_sg_pixels / total_pixels) * 100
    print("\n" + "=" * 60)
    print(f"Overall statistics:")
    print(f"  Total pixels across all images: {total_pixels:,}")
    print(f"  Total stress granule pixels: {total_sg_pixels:,}")
    print(f"  Overall coverage: {overall_percentage:.2f}%")
    print(f"  Class imbalance ratio: 1:{int(total_pixels/total_sg_pixels)}")

if __name__ == "__main__":
    analyze_stress_granule_coverage() 