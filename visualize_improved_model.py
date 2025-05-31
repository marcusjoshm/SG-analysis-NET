#!/usr/bin/env python
"""
Script to visualize results from the improved stress granule detection model
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from train_small_test import match_images_and_masks
from improved_model import ImprovedUNet

def visualize_improved_model(model_path, image_paths, mask_paths, output_dir, num_samples=3):
    """Load improved model and generate visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cpu')
    model = ImprovedUNet(in_channels=1, out_channels=1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_dice = checkpoint.get('val_dice', 'unknown')
        print(f"✅ Loaded improved model from epoch {epoch} (Dice: {val_dice})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    # Select random images for visualization
    if len(image_paths) > num_samples:
        indices = np.random.choice(len(image_paths), num_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
    
    # Process each image and generate visualization
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4*len(image_paths)))
    
    # Ensure axes is 2D even for single sample
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Load image and mask
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure image is single channel
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get a crop with granules if possible
        crop_size = (256, 256)
        h, w = img.shape
        
        # Try to get a crop with some positive pixels
        sg_coords = np.where(mask > 127)
        if len(sg_coords[0]) > 0:
            idx = np.random.randint(len(sg_coords[0]))
            center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
            
            start_h = max(0, min(h - crop_size[0], center_y - crop_size[0]//2))
            start_w = max(0, min(w - crop_size[1], center_x - crop_size[1]//2))
        else:
            start_h = np.random.randint(0, max(1, h - crop_size[0] + 1))
            start_w = np.random.randint(0, max(1, w - crop_size[1] + 1))
        
        # Extract crops
        img_crop = img[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
        mask_crop = mask[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
        
        # Preprocess image
        img_float = img_crop.astype(np.float32)
        
        # Enhanced contrast using percentile stretching
        p_low = np.percentile(img_float, 0.5)
        p_high = np.percentile(img_float, 99.5)
        img_norm = np.clip(img_float, p_low, p_high)
        img_norm = (img_norm - p_low) / (p_high - p_low)
        
        # Normalize mask to binary
        mask_crop = (mask_crop > 127).astype(np.float32)
        
        # Convert to tensor for model input
        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float()
        
        # Convert tensors to numpy for visualization
        img_np = img_norm
        mask_np = mask_crop
        prob_np = prob.squeeze().numpy()
        pred_np = pred.squeeze().numpy()
        
        # Calculate metrics
        dice = 2 * np.sum(pred_np * mask_np) / (np.sum(pred_np) + np.sum(mask_np) + 1e-6)
        precision = np.sum(pred_np * mask_np) / (np.sum(pred_np) + 1e-6)
        recall = np.sum(pred_np * mask_np) / (np.sum(mask_np) + 1e-6)
        
        # Plot original image
        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title(f'Original Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot probability map
        axes[i, 2].imshow(prob_np, cmap='magma')
        axes[i, 2].set_title(f'Probability Map')
        axes[i, 2].axis('off')
        
        # Plot binary prediction
        axes[i, 3].imshow(pred_np, cmap='gray')
        axes[i, 3].set_title(f'Dice: {dice:.4f}, P: {precision:.2f}, R: {recall:.2f}')
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_predictions.png'))
    plt.close()
    
    print(f"✅ Saved predictions to {os.path.join(output_dir, 'improved_predictions.png')}")
    
    # Also compare at different thresholds
    visualize_thresholds(model, image_paths[0], mask_paths[0], output_dir)

def visualize_thresholds(model, image_path, mask_path, output_dir):
    """Visualize model predictions at different thresholds"""
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure image is single channel
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get a crop with granules if possible
    crop_size = (256, 256)
    h, w = img.shape
    
    # Try to get a crop with some positive pixels
    sg_coords = np.where(mask > 127)
    if len(sg_coords[0]) > 0:
        idx = np.random.randint(len(sg_coords[0]))
        center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
        
        start_h = max(0, min(h - crop_size[0], center_y - crop_size[0]//2))
        start_w = max(0, min(w - crop_size[1], center_x - crop_size[1]//2))
    else:
        start_h = np.random.randint(0, max(1, h - crop_size[0] + 1))
        start_w = np.random.randint(0, max(1, w - crop_size[1] + 1))
    
    # Extract crops
    img_crop = img[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
    mask_crop = mask[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
    
    # Preprocess image
    img_float = img_crop.astype(np.float32)
    
    # Enhanced contrast using percentile stretching
    p_low = np.percentile(img_float, 0.5)
    p_high = np.percentile(img_float, 99.5)
    img_norm = np.clip(img_float, p_low, p_high)
    img_norm = (img_norm - p_low) / (p_high - p_low)
    
    # Normalize mask to binary
    mask_crop = (mask_crop > 127).astype(np.float32)
    
    # Convert to tensor for model input
    img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)
    
    # Generate prediction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).squeeze().numpy()
    
    # Create figure for threshold visualization
    fig, axes = plt.subplots(2, len(thresholds), figsize=(16, 6))
    
    # Plot for different thresholds
    for i, threshold in enumerate(thresholds):
        # Binary prediction at this threshold
        pred = (prob > threshold).astype(np.float32)
        
        # Calculate metrics
        dice = 2 * np.sum(pred * mask_crop) / (np.sum(pred) + np.sum(mask_crop) + 1e-6)
        precision = np.sum(pred * mask_crop) / (np.sum(pred) + 1e-6)
        recall = np.sum(pred * mask_crop) / (np.sum(mask_crop) + 1e-6)
        
        # Plot binary prediction
        axes[0, i].imshow(pred, cmap='gray')
        axes[0, i].set_title(f'Threshold: {threshold:.1f}\nDice: {dice:.4f}')
        axes[0, i].axis('off')
        
        # Create overlay showing TP, FP, FN
        overlay = np.zeros((crop_size[0], crop_size[1], 3), dtype=np.float32)
        overlay[..., 0] = img_norm  # Red channel = original image
        overlay[..., 1] = img_norm  # Green channel = original image
        overlay[..., 2] = img_norm  # Blue channel = original image
        
        # Add prediction as red overlay
        overlay[pred > 0, 0] = 1.0  # Red for prediction
        overlay[pred > 0, 1] = 0.0  # No green for prediction
        overlay[pred > 0, 2] = 0.0  # No blue for prediction
        
        # Add ground truth as green overlay
        overlay[mask_crop > 0, 0] = 0.0   # No red for ground truth
        overlay[mask_crop > 0, 1] = 1.0   # Green for ground truth
        overlay[mask_crop > 0, 2] = 0.0   # No blue for ground truth
        
        # Yellow for overlap (true positives)
        overlap = (pred > 0) & (mask_crop > 0)
        overlay[overlap, 0] = 1.0  # Red + Green = Yellow
        overlay[overlap, 1] = 1.0
        overlay[overlap, 2] = 0.0
        
        # Plot overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'P: {precision:.4f}, R: {recall:.4f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_thresholds.png'))
    plt.close()
    
    print(f"✅ Saved threshold comparison to {os.path.join(output_dir, 'improved_thresholds.png')}")

def compare_original_improved():
    """Compare original and improved models side by side"""
    # Get image paths
    matched_pairs = match_images_and_masks('data/images', 'data/masks')
    image_paths = [p[0] for p in matched_pairs]
    mask_paths = [p[1] for p in matched_pairs]
    
    # Paths to models
    original_model_path = 'metrics/low_res_test/low_res_test_model.pth'
    improved_model_path = 'metrics/improved_training/best_model.pth'
    
    # Output directory
    output_dir = 'metrics/comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize improved model results
    visualize_improved_model(improved_model_path, image_paths, mask_paths, output_dir)
    
    print("\n✅ Visualization complete!")
    print(f"Results saved in: {output_dir}/")

if __name__ == "__main__":
    compare_original_improved()
