#!/usr/bin/env python
"""
CPU-optimized test training for stress granule detection
Uses smaller batch size and image crops for efficient CPU training
"""

import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Import from existing modules
from models import UNet, StressGranule16bitDataset, combined_loss, dice_coefficient
from metrics import MetricsTracker
from main import match_images_and_masks_custom, visualize_predictions, plot_training_history
from train_small_test import match_images_and_masks, analyze_matched_pairs

def cpu_test_training():
    """Run optimized training test on CPU"""
    
    # Configuration optimized for CPU
    IMAGE_SIZE = 256
    BATCH_SIZE = 1  # Smaller batch size for CPU
    NUM_EPOCHS = 10  # Reduced epochs for test
    LEARNING_RATE = 1e-4
    NUM_SAMPLES = 6  # Use more of the available data
    CROPS_PER_IMAGE = 2  # Fewer crops per image for faster training
    
    print("=" * 60)
    print("CPU Test Training for Stress Granule Detection")
    print(f"Configuration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Crops per image: {CROPS_PER_IMAGE}")
    print(f"  - Effective training samples: {NUM_SAMPLES * CROPS_PER_IMAGE}")
    print(f"  - Loss: BCE (50%) + Dice (50%)")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Force CPU device
    device = torch.device('cpu')
    print("Using CPU for training")
    
    # Match images and masks
    matched_pairs = match_images_and_masks('data/images', 'data/masks', NUM_SAMPLES)
    
    if len(matched_pairs) < 3:
        print("❌ Need at least 3 image-mask pairs")
        return
    
    # Analyze the pairs
    analyze_matched_pairs(matched_pairs)
    
    # Split: Use more for training
    train_pairs = matched_pairs[:5]  # 5 for training
    val_pairs = matched_pairs[5:6]   # 1 for validation
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_images)} images × {CROPS_PER_IMAGE} crops = {len(train_images) * CROPS_PER_IMAGE} samples")
    print(f"  Validation: {len(val_images)} images × {CROPS_PER_IMAGE} crops = {len(val_images) * CROPS_PER_IMAGE} samples")
    
    # Light but effective augmentation
    import torchvision.transforms as transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Add rotation augmentation
    ])
    
    # Create datasets with enhanced contrast and preprocessing
    train_dataset = StressGranule16bitDataset(
        train_images, train_masks,
        transform=train_transform,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7,
        random_crop=True,
        crops_per_image=CROPS_PER_IMAGE
    )
    
    val_dataset = StressGranule16bitDataset(
        val_images, val_masks,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7,
        random_crop=True,  # Use random crops for validation too
        crops_per_image=CROPS_PER_IMAGE
    )
    
    # Create data loaders optimized for CPU (fewer workers)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No extra workers for CPU
        pin_memory=False  # Disable pin_memory for CPU
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model with single input channel
    model = UNet(in_channels=1, out_channels=1)
    print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create metrics directory
    metrics_dir = 'metrics/cpu_test'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir=metrics_dir, experiment_name='cpu_test')
    
    # Test data loading
    print("\n🔍 Testing data loading...")
    try:
        sample_img, sample_mask = train_dataset[0]
        print(f"✅ Sample image shape: {sample_img.shape}")
        print(f"✅ Sample mask shape: {sample_mask.shape}")
        print(f"✅ Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
        
        # Quick visualization of a sample
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample_img.squeeze().numpy(), cmap='gray')
        axes[0].set_title('Sample Image (Preprocessed)')
        axes[0].axis('off')
        
        axes[1].imshow(sample_mask.squeeze().numpy(), cmap='gray')
        axes[1].set_title('Sample Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'sample_data.png'))
        plt.close()
        print(f"✅ Sample visualization saved")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Train model with optimized settings for CPU
    print("\n🚀 Starting CPU test training...")
    start_time = time.time()
    
    # Define optimizer and learning rate scheduler 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    
    try:
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            
            # Use tqdm for progress bar
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
            
            for batch_idx, (data, target) in enumerate(train_loop):
                # Forward pass
                output = model(data)
                loss = combined_loss(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
                
                for data, target in val_loop:
                    # Forward pass
                    output = model(data)
                    
                    # Calculate loss and Dice coefficient
                    loss = combined_loss(output, target)
                    dice = dice_coefficient(output, target)
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_loop.set_postfix(loss=loss.item(), dice=dice.item())
            
            # Calculate average validation loss and Dice score
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)
            val_losses.append(avg_val_loss)
            val_dice_scores.append(avg_val_dice)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Dice: {avg_val_dice:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_dice = avg_val_dice
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                    'val_dice': best_val_dice
                }, os.path.join(metrics_dir, 'best_model.pth'))
                
                print(f"  ✅ New best model saved!")
            else:
                patience_counter += 1
                print(f"  ⚠️ No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= 5:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
            
        # Training completed
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f} seconds")
        print(f"Best validation Dice score: {best_val_dice:.4f} (epoch {best_epoch+1})")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot Dice scores
        plt.subplot(1, 2, 2)
        plt.plot(val_dice_scores, label='Val Dice', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.title('Validation Dice Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'training_history.png'))
        
        # Load best model for visualization
        checkpoint = torch.load(os.path.join(metrics_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Visualize predictions
        print("\n🎨 Generating prediction visualizations...")
        visualize_predictions(
            model, val_dataset, device,
            num_samples=2,
            output_file=os.path.join(metrics_dir, 'predictions.png')
        )
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

if __name__ == "__main__":
    success = cpu_test_training()
    
    if success:
        print("\n✅ CPU test training completed successfully!")
        print("Next steps:")
        print("  1. Review metrics and visualizations in 'metrics/cpu_test/'")
        print("  2. Adjust hyperparameters if needed")
        print("  3. Prepare for GPU training with larger dataset")
    else:
        print("\n❌ CPU test training failed.")
        print("Please check the error messages above.")
