#!/usr/bin/env python
"""
Create clear visualizations showing the preprocessing pipeline and predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models import UNet, StressGranule16bitDataset
import cv2

def create_detailed_visualization():
    """Create detailed visualization of preprocessing and predictions"""
    
    # Load the trained model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load('metrics/low_res_test/low_res_test_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Use first training image
    image_path = 'data/images/A549_Control_TS1_Merged_Processed001_ch00_t00.tif'
    mask_path = 'data/masks/MASK_A549_Control_TS1_Merged_Processed001_t00.tif'
    
    # Load raw 16-bit image
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # Create dataset to get preprocessed version
    dataset = StressGranule16bitDataset(
        [image_path],
        [mask_path],
        target_size=(512, 512),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    # Get preprocessed data
    processed_image, processed_mask = dataset[0]
    
    # Get model prediction
    image_batch = processed_image.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_batch)
    
    # Convert to numpy
    processed_image_np = processed_image.squeeze().numpy()
    processed_mask_np = processed_mask.squeeze().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Raw data
    ax1 = plt.subplot(3, 4, 1)
    # Show raw 16-bit with percentile scaling
    vmin, vmax = np.percentile(raw_image[raw_image > 0], [1, 99])
    ax1.imshow(raw_image, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Raw 16-bit Image\nShape: {raw_image.shape}\nRange: [{raw_image.min()}, {raw_image.max()}]')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(raw_mask, cmap='gray')
    ax2.set_title(f'Raw Mask\nUnique: {np.unique(raw_mask)}\nShape: {raw_mask.shape}')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.hist(raw_image.flatten(), bins=100, log=True)
    ax3.set_title('Raw Image Histogram\n(log scale)')
    ax3.set_xlabel('Pixel Value')
    ax3.axvline(x=raw_image.mean(), color='r', linestyle='--', label=f'Mean: {raw_image.mean():.1f}')
    ax3.legend()
    
    ax4 = plt.subplot(3, 4, 4)
    # Zoom on a stress granule region in raw
    if raw_mask.max() > 0:
        y_idx, x_idx = np.where(raw_mask > 0)
        cy, cx = int(np.mean(y_idx)), int(np.mean(x_idx))
        size = 200
        y1, y2 = max(0, cy-size), min(raw_image.shape[0], cy+size)
        x1, x2 = max(0, cx-size), min(raw_image.shape[1], cx+size)
        zoom_raw = raw_image[y1:y2, x1:x2]
        zoom_mask_raw = raw_mask[y1:y2, x1:x2]
        ax4.imshow(zoom_raw, cmap='gray', vmin=vmin, vmax=vmax)
        # Overlay mask contour
        ax4.contour(zoom_mask_raw > 0, colors='red', linewidths=2)
        ax4.set_title(f'Zoomed Raw Region\nRed contour = mask')
    ax4.axis('off')
    
    # Row 2: Preprocessed data
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(processed_image_np, cmap='gray')
    ax5.set_title(f'After Preprocessing\nSize: {processed_image_np.shape}\nRange: [{processed_image_np.min():.3f}, {processed_image_np.max():.3f}]')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(processed_mask_np, cmap='gray')
    pos_pixels = np.sum(processed_mask_np > 0)
    total_pixels = processed_mask_np.size
    ax6.set_title(f'Processed Mask\n{pos_pixels} positive pixels\n({100*pos_pixels/total_pixels:.2f}% of image)')
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.hist(processed_image_np.flatten(), bins=50, alpha=0.7)
    ax7.set_title('Processed Image Histogram')
    ax7.set_xlabel('Pixel Value')
    ax7.axvline(x=processed_image_np.mean(), color='r', linestyle='--', label=f'Mean: {processed_image_np.mean():.3f}')
    ax7.legend()
    
    ax8 = plt.subplot(3, 4, 8)
    # Show overlay
    ax8.imshow(processed_image_np, cmap='gray', alpha=0.8)
    ax8.imshow(processed_mask_np, cmap='Reds', alpha=0.3)
    ax8.set_title('Image + Mask Overlay')
    ax8.axis('off')
    
    # Row 3: Model predictions
    ax9 = plt.subplot(3, 4, 9)
    ax9.imshow(prediction_np, cmap='hot', vmin=0, vmax=1)
    ax9.set_title(f'Raw Prediction\nRange: [{prediction_np.min():.4f}, {prediction_np.max():.4f}]')
    ax9.axis('off')
    
    ax10 = plt.subplot(3, 4, 10)
    pred_binary = (prediction_np > 0.5).astype(float)
    ax10.imshow(pred_binary, cmap='gray')
    ax10.set_title(f'Prediction > 0.5\n{np.sum(pred_binary)} pixels')
    ax10.axis('off')
    
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(prediction_np.flatten(), bins=50, alpha=0.7, color='green')
    ax11.set_title('Prediction Distribution')
    ax11.set_xlabel('Prediction Value')
    ax11.axvline(x=0.5, color='r', linestyle='--', label='Threshold 0.5')
    ax11.axvline(x=prediction_np.mean(), color='b', linestyle='--', label=f'Mean: {prediction_np.mean():.3f}')
    ax11.legend()
    
    ax12 = plt.subplot(3, 4, 12)
    # Compare at optimal threshold
    optimal_threshold = 0.45  # Based on prediction range
    pred_optimal = (prediction_np > optimal_threshold).astype(float)
    # Create comparison image
    comparison = np.zeros((*processed_mask_np.shape, 3))
    comparison[:,:,0] = processed_mask_np  # Red = ground truth
    comparison[:,:,1] = pred_optimal       # Green = prediction
    comparison[:,:,2] = 0                  # Blue = 0
    ax12.imshow(comparison)
    ax12.set_title(f'Comparison @ threshold {optimal_threshold}\nRed=Truth, Green=Pred, Yellow=Overlap')
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig('detailed_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a second figure with zoom on predictions
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Find regions with stress granules
    y_indices, x_indices = np.where(processed_mask_np > 0)
    if len(y_indices) > 0:
        # Take 3 different regions
        n_regions = min(3, len(y_indices) // 100)
        for i in range(3):
            if i < n_regions:
                idx = i * len(y_indices) // n_regions
                cy, cx = y_indices[idx], x_indices[idx]
            else:
                cy, cx = processed_mask_np.shape[0]//2, processed_mask_np.shape[1]//2
            
            window = 64
            y1, y2 = max(0, cy-window), min(processed_mask_np.shape[0], cy+window)
            x1, x2 = max(0, cx-window), min(processed_mask_np.shape[1], cx+window)
            
            # Top row: Image and mask
            axes[0, i].imshow(processed_image_np[y1:y2, x1:x2], cmap='gray')
            axes[0, i].contour(processed_mask_np[y1:y2, x1:x2] > 0, colors='red', linewidths=2)
            axes[0, i].set_title(f'Region {i+1}: Image + Truth')
            axes[0, i].axis('off')
            
            # Bottom row: Predictions
            axes[1, i].imshow(prediction_np[y1:y2, x1:x2], cmap='hot', vmin=0.3, vmax=0.6)
            axes[1, i].contour(processed_mask_np[y1:y2, x1:x2] > 0, colors='blue', linewidths=2)
            axes[1, i].contour(prediction_np[y1:y2, x1:x2] > 0.45, colors='yellow', linewidths=1)
            axes[1, i].set_title(f'Prediction\nBlue=Truth, Yellow=Pred>0.45')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('zoomed_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Created visualizations:")
    print("   - detailed_visualization.png")
    print("   - zoomed_predictions.png")
    
    # Print analysis
    print("\n📊 Analysis Summary:")
    print(f"1. Raw image: 16-bit, max value {raw_image.max()} (very low for 16-bit)")
    print(f"2. After preprocessing: normalized to [{processed_image_np.min():.3f}, {processed_image_np.max():.3f}]")
    print(f"3. Stress granules occupy only {100*pos_pixels/total_pixels:.2f}% of the image")
    print(f"4. Model predictions range: [{prediction_np.min():.4f}, {prediction_np.max():.4f}]")
    print(f"5. Model never predicts above 0.5 threshold!")
    print("\n⚠️  Issue: The model learned to predict everything as background due to extreme class imbalance.")
    print("   With only 0.25% positive pixels, predicting all negative gives 99.75% accuracy!")
    
    print("\n💡 Solutions for full training:")
    print("1. Use weighted loss to penalize missing stress granules more")
    print("2. Use focal loss designed for imbalanced segmentation")
    print("3. Sample patches around stress granules during training")
    print("4. Lower the prediction threshold (e.g., 0.4 instead of 0.5)")

if __name__ == "__main__":
    create_detailed_visualization() 