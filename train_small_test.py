#!/usr/bin/env python
"""
Small-scale training test with 3 images at high resolution
Tests the training pipeline before committing to full dataset
"""

import os
import sys
import numpy as np
from glob import glob
import cv2
import torch
from sklearn.model_selection import train_test_split

# Import from main training script
from main import train_model, visualize_predictions, plot_training_history
from models import UNet, StressGranule16bitDataset
from metrics import MetricsTracker
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def match_images_and_masks(image_dir, mask_dir, max_samples=3):
    """
    Match images with their corresponding masks based on naming convention:
    Image: {core_name}_ch00_t00.tif
    Mask: MASK_{core_name}_t00.tif
    """
    image_paths = sorted(glob(os.path.join(image_dir, '*.tif')))
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.tif')))
    
    # Remove .gitkeep if present
    image_paths = [p for p in image_paths if '.gitkeep' not in p]
    mask_paths = [p for p in mask_paths if '.gitkeep' not in p]
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    matched_pairs = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Extract core name by removing _ch00_t00.tif
        if '_ch00_t00.tif' in img_name:
            core_name = img_name.replace('_ch00_t00.tif', '')
            
            # Expected mask name
            expected_mask_name = f'MASK_{core_name}_t00.tif'
            
            # Find matching mask
            for mask_path in mask_paths:
                mask_name = os.path.basename(mask_path)
                if mask_name == expected_mask_name:
                    matched_pairs.append((img_path, mask_path))
                    print(f"Matched: {img_name} -> {mask_name}")
                    break
    
    # Limit to max_samples
    if len(matched_pairs) > max_samples:
        matched_pairs = matched_pairs[:max_samples]
        print(f"\nLimited to {max_samples} pairs for testing")
    
    return matched_pairs

def analyze_matched_pairs(matched_pairs):
    """Analyze the matched image-mask pairs"""
    print("\nAnalyzing matched pairs:")
    for i, (img_path, mask_path) in enumerate(matched_pairs):
        print(f"\nPair {i+1}:")
        print(f"  Image: {os.path.basename(img_path)}")
        print(f"  Mask: {os.path.basename(mask_path)}")
        
        # Load and check dimensions
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Image range: [{img.min()}, {img.max()}]")
        print(f"  Mask unique values: {np.unique(mask)}")

def main():
    # Configuration for small test
    IMAGE_SIZE = 1024  # High resolution
    BATCH_SIZE = 1     # Small batch size due to high resolution
    NUM_EPOCHS = 20    # Fewer epochs for testing
    LEARNING_RATE = 1e-4
    NUM_SAMPLES = 3    # Only use 3 images
    
    print("=" * 60)
    print("Small-Scale Training Test")
    print(f"Configuration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slow)")
    
    # Data directories
    image_dir = 'data/images'
    mask_dir = 'data/masks'
    
    # Match images and masks
    matched_pairs = match_images_and_masks(image_dir, mask_dir, NUM_SAMPLES)
    
    if len(matched_pairs) == 0:
        print("No matched pairs found! Check your data.")
        return
    
    # Analyze the pairs
    analyze_matched_pairs(matched_pairs)
    
    # Split paths
    image_paths = [pair[0] for pair in matched_pairs]
    mask_paths = [pair[1] for pair in matched_pairs]
    
    # For small dataset, use 2 for training and 1 for validation
    if len(image_paths) >= 3:
        train_images = image_paths[:2]
        train_masks = mask_paths[:2]
        val_images = image_paths[2:3]
        val_masks = mask_paths[2:3]
    else:
        # If less than 3, use all for training and validation
        train_images = image_paths
        train_masks = mask_paths
        val_images = image_paths
        val_masks = mask_paths
    
    print(f"\nTraining set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Reduced rotation for high-res images to save memory
        transforms.RandomRotation(degrees=45),
    ])
    
    # Create datasets
    train_dataset = StressGranule16bitDataset(
        train_images, train_masks,
        transform=train_transform,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    val_dataset = StressGranule16bitDataset(
        val_images, val_masks,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create metrics directory
    metrics_dir = 'metrics/small_test'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir=metrics_dir, experiment_name='small_test')
    
    # Test data loading
    print("\nTesting data loading...")
    try:
        sample_img, sample_mask = train_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample mask shape: {sample_mask.shape}")
        print(f"Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
        print(f"Mask unique values: {torch.unique(sample_mask)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train model
    print("\nStarting training...")
    try:
        train_losses, val_losses, val_dice_scores, metrics_tracker = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            patience=10,
            device=device,
            metrics_tracker=metrics_tracker,
            experiment_name='small_test'
        )
        
        print("\nTraining completed successfully!")
        
        # Plot training history
        plot_training_history(train_losses, val_losses, val_dice_scores)
        
        # Visualize predictions
        print("\nGenerating prediction visualizations...")
        visualize_predictions(
            model, val_dataset, device,
            num_samples=min(3, len(val_dataset)),
            output_file=os.path.join(metrics_dir, 'test_predictions.png')
        )
        
        # Save model specifically for this test
        torch.save({
            'model_state_dict': model.state_dict(),
            'image_size': IMAGE_SIZE,
            'final_dice': val_dice_scores[-1] if val_dice_scores else 0.0
        }, 'small_test_model.pth')
        
        print(f"\nTest model saved as 'small_test_model.pth'")
        print(f"Final validation Dice score: {val_dice_scores[-1] if val_dice_scores else 'N/A':.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 