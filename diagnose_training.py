#!/usr/bin/env python
"""
Diagnostic script to check model predictions and data pipeline
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models import UNet, StressGranule16bitDataset
import cv2
import os

def diagnose_model_predictions():
    """Diagnose what the model is actually predicting"""
    
    print("🔍 Diagnostic Analysis")
    print("=" * 60)
    
    # Load the trained model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load('metrics/low_res_test/low_res_test_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a sample image-mask pair
    image_path = 'data/images/A549_Control_TS1_Merged_Processed001_ch00_t00.tif'
    mask_path = 'data/masks/MASK_A549_Control_TS1_Merged_Processed001_t00.tif'
    
    # Create dataset with single sample
    dataset = StressGranule16bitDataset(
        [image_path],
        [mask_path],
        target_size=(512, 512),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    # Get the sample
    image, mask = dataset[0]
    
    # Add batch dimension
    image_batch = image.unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(image_batch)
    
    # Convert to numpy for analysis
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Analyze the data
    print("\n📊 Data Statistics:")
    print(f"Image shape: {image_np.shape}")
    print(f"Image range: [{image_np.min():.3f}, {image_np.max():.3f}]")
    print(f"Mask unique values: {np.unique(mask_np)}")
    print(f"Mask positive pixels: {np.sum(mask_np > 0)} / {mask_np.size} ({100 * np.sum(mask_np > 0) / mask_np.size:.2f}%)")
    
    print(f"\n🤖 Model Output Statistics:")
    print(f"Prediction range: [{pred_np.min():.6f}, {pred_np.max():.6f}]")
    print(f"Prediction mean: {pred_np.mean():.6f}")
    print(f"Prediction std: {pred_np.std():.6f}")
    
    # Check predictions at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("\n📈 Predictions at different thresholds:")
    for thresh in thresholds:
        pred_binary = (pred_np > thresh).astype(float)
        positive_pixels = np.sum(pred_binary)
        percentage = 100 * positive_pixels / pred_binary.size
        print(f"  Threshold {thresh}: {positive_pixels} pixels ({percentage:.2f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Data
    axes[0, 0].imshow(image_np, cmap='gray')
    axes[0, 0].set_title('Input Image (Preprocessed)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_np, cmap='gray')
    axes[0, 1].set_title(f'Ground Truth Mask\n{np.sum(mask_np > 0)} positive pixels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Raw Prediction\n[{pred_np.min():.3f}, {pred_np.max():.3f}]')
    axes[0, 2].axis('off')
    
    # Histogram of predictions
    axes[0, 3].hist(pred_np.flatten(), bins=50, alpha=0.7)
    axes[0, 3].set_title('Prediction Value Distribution')
    axes[0, 3].set_xlabel('Prediction Value')
    axes[0, 3].set_ylabel('Count')
    axes[0, 3].set_yscale('log')
    
    # Row 2: Different thresholds
    for i, thresh in enumerate([0.1, 0.3, 0.5, 0.7]):
        pred_binary = (pred_np > thresh).astype(float)
        axes[1, i].imshow(pred_binary, cmap='gray')
        axes[1, i].set_title(f'Threshold: {thresh}\n{np.sum(pred_binary)} pixels')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnosis_results.png', dpi=150)
    plt.close()
    
    # Also check alignment by overlaying
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show overlay of image and mask
    axes[0].imshow(image_np, cmap='gray', alpha=0.7)
    axes[0].imshow(mask_np, cmap='Reds', alpha=0.3)
    axes[0].set_title('Image + Mask Overlay\n(Check Alignment)')
    axes[0].axis('off')
    
    # Show regions where mask is positive
    mask_regions = image_np.copy()
    mask_regions[mask_np > 0] = image_np.max()
    axes[1].imshow(mask_regions, cmap='gray')
    axes[1].set_title('Stress Granule Regions\n(Highlighted)')
    axes[1].axis('off')
    
    # Zoom in on a region with stress granules
    # Find a region with positive mask pixels
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) > 0:
        # Get center of mass of positive pixels
        cy, cx = int(np.mean(y_indices)), int(np.mean(x_indices))
        # Define zoom window
        window = 64
        y1, y2 = max(0, cy-window), min(mask_np.shape[0], cy+window)
        x1, x2 = max(0, cx-window), min(mask_np.shape[1], cx+window)
        
        # Create zoomed views
        zoom_img = image_np[y1:y2, x1:x2]
        zoom_mask = mask_np[y1:y2, x1:x2]
        zoom_pred = pred_np[y1:y2, x1:x2]
        
        # Show zoomed overlay
        axes[2].imshow(zoom_img, cmap='gray', alpha=0.7)
        axes[2].imshow(zoom_mask, cmap='Reds', alpha=0.3)
        axes[2].contour(zoom_pred > 0.3, colors='yellow', linewidths=1)
        axes[2].set_title(f'Zoomed Region ({window*2}x{window*2})\nRed=Truth, Yellow=Prediction')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No positive mask pixels found!', 
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('alignment_check.png', dpi=150)
    plt.close()
    
    print("\n✅ Diagnostic images saved:")
    print("   - diagnosis_results.png")
    print("   - alignment_check.png")
    
    # Additional check: Look at raw 16-bit data
    print("\n🔬 Checking raw 16-bit data:")
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    print(f"Raw image: shape={raw_image.shape}, dtype={raw_image.dtype}, range=[{raw_image.min()}, {raw_image.max()}]")
    print(f"Raw mask: shape={raw_mask.shape}, dtype={raw_mask.dtype}, unique={np.unique(raw_mask)}")
    
    # Check if dimensions match
    if raw_image.shape != raw_mask.shape:
        print("⚠️  WARNING: Raw image and mask have different shapes!")
    
    return pred_np, mask_np

if __name__ == "__main__":
    pred, mask = diagnose_model_predictions()
    
    # Calculate metrics manually
    print("\n📏 Manual Metric Calculation:")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pred_binary = (pred > thresh).astype(float)
        
        # Calculate metrics
        tp = np.sum((pred_binary == 1) & (mask == 1))
        fp = np.sum((pred_binary == 1) & (mask == 0))
        fn = np.sum((pred_binary == 0) & (mask == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nThreshold {thresh}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}") 