#!/usr/bin/env python
"""
GPU test training script for stress granule detection
Uses the same parameters as the successful improved model training
but leverages GPU acceleration
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import time
import platform
from datetime import datetime

# Import from existing modules
from improved_model import ImprovedUNet, improved_combined_loss, dice_coefficient
from improved_training import ImprovedDataset, create_enhanced_augmentations
from train_small_test import match_images_and_masks, analyze_matched_pairs

def gpu_train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
                patience=15, device='cuda', metrics_dir='metrics/gpu_test'):
    """
    GPU-optimized training function with the same parameters as the successful improved model
    """
    model = model.to(device)
    
    # Timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{metrics_dir}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/10)
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_precisions = []
    val_recalls = []
    best_val_dice = 0
    patience_counter = 0
    
    # Training start time
    start_time = time.time()
    
    print(f"\n🚀 Starting GPU training with {num_epochs} epochs")
    print(f"📁 Metrics will be saved to {run_dir}")
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        # Progress bar for training
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
        for images, masks in train_progress:
            # Move data to GPU
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = improved_combined_loss(outputs, masks, device, alpha=0.4, pos_weight=5.0)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Calculate average training loss
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_dice_score = 0
        epoch_precision = 0
        epoch_recall = 0
        num_val_batches = 0
        
        # Progress bar for validation
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, masks in val_progress:
                # Move data to GPU
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = improved_combined_loss(outputs, masks, device, alpha=0.4, pos_weight=5.0)
                epoch_val_loss += loss.item()
                
                # Calculate Dice coefficient
                dice = dice_coefficient(outputs, masks)
                epoch_dice_score += dice.item()
                
                # Calculate precision and recall
                pred = (torch.sigmoid(outputs) > 0.5).float()
                true_positives = (pred * masks).sum().item()
                pred_positives = pred.sum().item()
                actual_positives = masks.sum().item()
                
                # Avoid division by zero
                precision = true_positives / max(pred_positives, 1e-6)
                recall = true_positives / max(actual_positives, 1e-6)
                
                epoch_precision += precision
                epoch_recall += recall
                num_val_batches += 1
                
                val_progress.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"})
        
        # Calculate average validation metrics
        epoch_val_loss /= max(num_val_batches, 1)
        epoch_dice_score /= max(num_val_batches, 1)
        epoch_precision /= max(num_val_batches, 1)
        epoch_recall /= max(num_val_batches, 1)
        
        val_losses.append(epoch_val_loss)
        val_dice_scores.append(epoch_dice_score)
        val_precisions.append(epoch_precision)
        val_recalls.append(epoch_recall)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Dice: {epoch_dice_score:.4f}, "
              f"Precision: {epoch_precision:.4f}, "
              f"Recall: {epoch_recall:.4f}, "
              f"Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
        
        # Save metrics to CSV
        with open(f"{run_dir}/metrics.csv", "a") as f:
            if epoch == 1:
                f.write("epoch,train_loss,val_loss,val_dice,precision,recall,lr\n")
            f.write(f"{epoch},{epoch_train_loss:.6f},{epoch_val_loss:.6f},"
                   f"{epoch_dice_score:.6f},{epoch_precision:.6f},{epoch_recall:.6f},"
                   f"{scheduler.get_last_lr()[0]}\n")
        
        # Early stopping with checkpoint saving
        if epoch_dice_score > best_val_dice:
            best_val_dice = epoch_dice_score
            patience_counter = 0
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_dice': epoch_dice_score,
                'train_loss': epoch_train_loss
            }, f"{run_dir}/best_model.pth")
            
            print(f"✅ New best model saved! (Dice: {epoch_dice_score:.4f})")
        else:
            patience_counter += 1
            
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break
    
    # Final elapsed time
    total_time = time.time() - start_time
    print(f"\n⏱️ Total training time: {total_time//60:.0f}m {total_time%60:.0f}s")
    
    # Save final metrics as JSON
    import json
    best_metrics = {
        'best_dice': best_val_dice,
        'best_epoch': val_dice_scores.index(best_val_dice) + 1,
        'best_loss': val_losses[val_dice_scores.index(best_val_dice)]
    }
    
    with open(f"{run_dir}/best_metrics.json", 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    # Load best model for returning
    best_model = ImprovedUNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(f"{run_dir}/best_model.pth")
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    return best_val_dice, best_model


# Main function to run GPU test training
def main():
    """Main function to run GPU test training with the same parameters as the successful model"""
    # Configuration - same as in improved_training.py
    IMAGE_SIZE = 256
    BATCH_SIZE = 4  # Increased for GPU
    NUM_EPOCHS = 75
    LEARNING_RATE = 8e-5
    PATIENCE = 15
    NUM_SAMPLES = 8
    CROPS_PER_IMAGE = 3
    
    print("=" * 60)
    print("GPU Test Training for Stress Granule Detection")
    print(f"Configuration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Crops per image: {CROPS_PER_IMAGE}")
    print(f"  - Effective training samples: {NUM_SAMPLES * CROPS_PER_IMAGE}")
    print(f"  - Loss: BCE (40%) + Dice (60%) with positive weight")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Check for GPU - specific handling for Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using Apple Silicon GPU (MPS device): {device}")
        print("Apple Silicon GPU acceleration is enabled")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {device}")
        
        # Print GPU info if NVIDIA
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU: {device}")
        print("⚠️ Warning: GPU not available, using CPU instead")
        print("Training will be much slower on CPU")
    
    # Match images and masks
    matched_pairs = match_images_and_masks('data/images', 'data/masks', NUM_SAMPLES)
    
    if len(matched_pairs) < 4:
        print("❌ Need at least 4 image-mask pairs")
        return
    
    # Split: Use same 70/30 split
    train_size = int(0.7 * len(matched_pairs))
    train_pairs = matched_pairs[:train_size]
    val_pairs = matched_pairs[train_size:]
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_images)} images × {CROPS_PER_IMAGE} crops = {len(train_images) * CROPS_PER_IMAGE} samples")
    print(f"  Validation: {len(val_images)} images × {CROPS_PER_IMAGE} crops = {len(val_images) * CROPS_PER_IMAGE} samples")
    
    # Create enhanced augmentations
    train_transform = create_enhanced_augmentations()
    
    # Create datasets
    train_dataset = ImprovedDataset(
        train_images, train_masks,
        transform=train_transform,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        random_crop=True,
        crops_per_image=CROPS_PER_IMAGE
    )
    
    val_dataset = ImprovedDataset(
        val_images, val_masks,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        random_crop=True,
        crops_per_image=CROPS_PER_IMAGE
    )
    
    # Create data loaders with GPU optimizations (conditional pin_memory)
    use_pin_memory = device.type == 'cuda'  # Only use pin_memory with CUDA, not with MPS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced for MPS compatibility
        pin_memory=use_pin_memory  # Only for CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,  # Reduced for MPS compatibility
        pin_memory=use_pin_memory  # Only for CUDA
    )
    
    # Initialize model
    model = ImprovedUNet(in_channels=1, out_channels=1)
    print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test data loading
    print("\n🔍 Testing data loading...")
    try:
        sample_img, sample_mask = train_dataset[0]
        print(f"✅ Sample image shape: {sample_img.shape}")
        print(f"✅ Sample mask shape: {sample_mask.shape}")
        print(f"✅ Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
        
        # Test device transfer
        print("Testing GPU transfer...")
        test_tensor = torch.ones((1, 256, 256)).to(device)
        print(f"✅ Successfully transferred tensor to {device}")
        del test_tensor  # Clean up test tensor
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run GPU training
    metrics_dir = 'metrics/gpu_test'
    os.makedirs(metrics_dir, exist_ok=True)
    
    best_dice, model = gpu_train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        device=device,
        metrics_dir=metrics_dir
    )
    
    print(f"\n🎉 GPU Training complete! Best Dice score: {best_dice:.4f}")
    print(f"Compare this with the previous best Dice score of >0.9 from the CPU model")


if __name__ == "__main__":
    main()
