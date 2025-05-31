#!/usr/bin/env python
"""
Script to visualize results from stress granule detection model
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import torch.nn as nn
from models import UNet, StressGranule16bitDataset
from train_small_test import match_images_and_masks, analyze_matched_pairs

def load_model_and_visualize(model_path, image_paths, mask_paths, output_dir, num_samples=4):
    """Load model and generate visualizations for sample images"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cpu')
    model = UNet(in_channels=1, out_channels=1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_dice = checkpoint.get('val_dice', 'unknown')
        print(f"✅ Loaded model from epoch {epoch} (Dice: {val_dice})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Try loading just the state dict
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ Loaded model state dict only")
        except:
            print("❌ Failed to load model. Using untrained model for visualization.")
    
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
        
        # Preprocess image
        img_float = img.astype(np.float32)
        
        # Basic percentile normalization
        p_low = np.percentile(img_float, 1)
        p_high = np.percentile(img_float, 99)
        img_norm = np.clip(img_float, p_low, p_high)
        img_norm = (img_norm - p_low) / (p_high - p_low)
        
        # Random crop for visualization
        crop_size = (256, 256)
        h, w = img_norm.shape
        
        # Try to get a crop with some positive pixels if possible
        if mask is not None:
            # Get coordinates of stress granules
            sg_coords = np.where(mask > 127)
            if len(sg_coords[0]) > 0:
                # Randomly select a stress granule pixel
                idx = np.random.randint(len(sg_coords[0]))
                center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                
                # Calculate crop boundaries ensuring we stay within image
                start_h = max(0, min(h - crop_size[0], center_y - crop_size[0]//2))
                start_w = max(0, min(w - crop_size[1], center_x - crop_size[1]//2))
            else:
                # If no stress granules, use random crop
                start_h = np.random.randint(0, max(1, h - crop_size[0] + 1))
                start_w = np.random.randint(0, max(1, w - crop_size[1] + 1))
        else:
            # Random crop
            start_h = np.random.randint(0, max(1, h - crop_size[0] + 1))
            start_w = np.random.randint(0, max(1, w - crop_size[1] + 1))
        
        # Extract crops
        img_crop = img_norm[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
        mask_crop = mask[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]] if mask is not None else None
        
        # Normalize mask to binary
        mask_crop = (mask_crop > 127).astype(np.float32)
        
        # Convert to tensor for model input
        img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Generate prediction
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float()
        
        # Convert tensors to numpy for visualization
        img_np = img_crop
        mask_np = mask_crop
        prob_np = prob.squeeze().numpy()
        pred_np = pred.squeeze().numpy()
        
        # Calculate metrics for this sample
        if mask_np is not None:
            dice = 2 * np.sum(pred_np * mask_np) / (np.sum(pred_np) + np.sum(mask_np) + 1e-6)
            precision = np.sum(pred_np * mask_np) / (np.sum(pred_np) + 1e-6)
            recall = np.sum(pred_np * mask_np) / (np.sum(mask_np) + 1e-6)
        else:
            dice = precision = recall = float('nan')
        
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
        axes[i, 3].set_title(f'Prediction (Dice: {dice:.4f})')
        axes[i, 3].axis('off')
        
        # Add metrics text
        fig.text(0.01, 0.98 - i * (4/len(image_paths) * 0.98), 
                 f"Sample {i+1}: Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}", 
                 fontsize=10, transform=plt.gcf().transFigure)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_predictions.png'))
    plt.close()
    
    print(f"✅ Saved detailed predictions to {os.path.join(output_dir, 'detailed_predictions.png')}")
    
    # Also visualize at different thresholds
    visualize_thresholds(model, image_paths[0], mask_paths[0], output_dir)

def visualize_thresholds(model, image_path, mask_path, output_dir, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Visualize predictions at different thresholds"""
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure image is single channel
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image
    img_float = img.astype(np.float32)
    
    # Basic percentile normalization
    p_low = np.percentile(img_float, 1)
    p_high = np.percentile(img_float, 99)
    img_norm = np.clip(img_float, p_low, p_high)
    img_norm = (img_norm - p_low) / (p_high - p_low)
    
    # Random crop for visualization
    crop_size = (256, 256)
    h, w = img_norm.shape
    
    # Try to get a crop with some positive pixels if possible
    if mask is not None:
        # Get coordinates of stress granules
        sg_coords = np.where(mask > 127)
        if len(sg_coords[0]) > 0:
            # Randomly select a stress granule pixel
            idx = np.random.randint(len(sg_coords[0]))
            center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
            
            # Calculate crop boundaries ensuring we stay within image
            start_h = max(0, min(h - crop_size[0], center_y - crop_size[0]//2))
            start_w = max(0, min(w - crop_size[1], center_x - crop_size[1]//2))
        else:
            # If no stress granules, use random crop
            start_h = np.random.randint(0, max(1, h - crop_size[0] + 1))
            start_w = np.random.randint(0, max(1, w - crop_size[1] + 1))
    else:
        # Random crop
        start_h = np.random.randint(0, max(1, h - crop_size[0] + 1))
        start_w = np.random.randint(0, max(1, w - crop_size[1] + 1))
    
    # Extract crops
    img_crop = img_norm[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
    mask_crop = mask[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]] if mask is not None else None
    
    # Normalize mask to binary
    mask_crop = (mask_crop > 127).astype(np.float32)
    
    # Convert to tensor for model input
    img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Generate prediction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).squeeze().numpy()
    
    # Create figure for threshold visualization
    fig, axes = plt.subplots(2, len(thresholds), figsize=(16, 6))
    
    # Top row: Show binary predictions at different thresholds
    # Bottom row: Show overlay of prediction on original image
    for i, threshold in enumerate(thresholds):
        # Binary prediction at this threshold
        pred = (prob > threshold).astype(np.float32)
        
        # Calculate metrics
        if mask_crop is not None:
            dice = 2 * np.sum(pred * mask_crop) / (np.sum(pred) + np.sum(mask_crop) + 1e-6)
            precision = np.sum(pred * mask_crop) / (np.sum(pred) + 1e-6)
            recall = np.sum(pred * mask_crop) / (np.sum(mask_crop) + 1e-6)
        else:
            dice = precision = recall = float('nan')
        
        # Plot binary prediction
        axes[0, i].imshow(pred, cmap='gray')
        axes[0, i].set_title(f'Threshold: {threshold:.1f}\nDice: {dice:.4f}')
        axes[0, i].axis('off')
        
        # Create RGB overlay
        overlay = np.zeros((crop_size[0], crop_size[1], 3), dtype=np.float32)
        overlay[..., 0] = img_crop  # Red channel = original image
        overlay[..., 1] = img_crop  # Green channel = original image
        overlay[..., 2] = img_crop  # Blue channel = original image
        
        # Add prediction as red overlay
        overlay[pred > 0, 0] = 1.0  # Red for prediction
        overlay[pred > 0, 1] = 0.0  # No green for prediction
        overlay[pred > 0, 2] = 0.0  # No blue for prediction
        
        # Add ground truth as green overlay
        if mask_crop is not None:
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
    plt.savefig(os.path.join(output_dir, 'threshold_comparison.png'))
    plt.close()
    
    print(f"✅ Saved threshold comparison to {os.path.join(output_dir, 'threshold_comparison.png')}")

def display_training_history(history_file):
    """Display training history from saved file"""
    try:
        # Load training history
        history = np.load(history_file, allow_pickle=True).item()
        
        # Extract metrics
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        val_dice_scores = history.get('val_dice', [])
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot Dice score
        plt.subplot(1, 2, 2)
        plt.plot(val_dice_scores, label='Val Dice', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_plot.png')
        plt.close()
        
        print(f"✅ Saved training history plot to training_history_plot.png")
        
        # Also print summary statistics
        best_dice_idx = np.argmax(val_dice_scores)
        best_dice = val_dice_scores[best_dice_idx]
        best_epoch = best_dice_idx + 1
        
        print(f"\n📊 Training Summary:")
        print(f"  Best Dice score: {best_dice:.4f} (epoch {best_epoch})")
        print(f"  Final Dice score: {val_dice_scores[-1]:.4f}")
        print(f"  Initial loss: {train_losses[0]:.4f}")
        print(f"  Final loss: {train_losses[-1]:.4f}")
        print(f"  Improvement: {100 * (train_losses[0] - train_losses[-1]) / train_losses[0]:.1f}%")
        
    except Exception as e:
        print(f"❌ Error loading training history: {e}")

def main():
    """Main function to run visualization"""
    print("=" * 60)
    print("Stress Granule Detection - Result Visualization")
    print("=" * 60)
    
    # Path to saved model
    model_path = 'metrics/low_res_test/best_model.pth'
    output_dir = 'metrics/low_res_test/visualizations'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        
        # Try to find model in metrics directory
        model_paths = glob('metrics/**/*.pth', recursive=True)
        if model_paths:
            model_path = model_paths[0]
            print(f"✅ Using model at {model_path}")
        else:
            print("❌ No model found. Please check the model path.")
            return
    
    # Get the data paths
    matched_pairs = match_images_and_masks('data/images', 'data/masks')
    image_paths = [p[0] for p in matched_pairs]
    mask_paths = [p[1] for p in matched_pairs]
    
    # Load model and visualize
    load_model_and_visualize(model_path, image_paths, mask_paths, output_dir)
    
    # Display training history
    history_file = os.path.join(os.path.dirname(model_path), 'training_history.npy')
    if os.path.exists(history_file):
        display_training_history(history_file)
    else:
        print(f"❌ Training history not found at {history_file}")
    
    print("\n✅ Visualization complete!")
    print(f"Results saved in: {output_dir}/")

if __name__ == "__main__":
    main()
