#!/usr/bin/env python
"""
Improved training script for stress granule detection with GPU acceleration
Optimized for Apple Silicon using Metal Performance Shaders (MPS)
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from glob import glob
import time
import platform
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Import improved model and loss functions
from improved_model import ImprovedUNet, improved_combined_loss, dice_coefficient
from train_small_test import match_images_and_masks, analyze_matched_pairs
from models import StressGranule16bitDataset


class ImprovedDataset(Dataset):
    """Enhanced dataset with better preprocessing for stress granule detection"""
    
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256), 
                 random_crop=True, crops_per_image=3):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.random_crop = random_crop
        self.crops_per_image = crops_per_image if random_crop else 1
        
        # Validate paths exist
        for img_path, mask_path in zip(image_paths, mask_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image path does not exist: {img_path}")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask path does not exist: {mask_path}")
    
    def __len__(self):
        return len(self.image_paths) * self.crops_per_image
    
    def _enhance_contrast(self, image, percentile_low=0.5, percentile_high=99.5):
        """Enhanced contrast using tighter percentile stretching"""
        # Calculate percentiles for non-zero pixels only
        non_zero_pixels = image[image > 0]
        if len(non_zero_pixels) > 0:
            p_low = np.percentile(non_zero_pixels, percentile_low)
            p_high = np.percentile(non_zero_pixels, percentile_high)
        else:
            # Fallback if all pixels are zero
            p_low = 0
            p_high = 1
        
        # Clip and rescale
        if p_high > p_low:
            image_clipped = np.clip(image, p_low, p_high)
            image_rescaled = (image_clipped - p_low) / (p_high - p_low)
        else:
            # Handle edge case where all values are the same
            image_rescaled = np.zeros_like(image, dtype=np.float32)
        
        return image_rescaled
    
    def _apply_gaussian_blur(self, image, sigma=1.5):
        """Apply Gaussian blur with adjusted sigma"""
        # Calculate kernel size based on sigma (6*sigma + 1, must be odd)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def _smart_crop(self, image, mask, crop_size):
        """Extract a smart crop with bias toward stress granules but some background too"""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        # Ensure crop size isn't larger than image
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # Find stress granule locations
        if mask is not None:
            # Get coordinates of stress granules
            sg_coords = np.where(mask > 127)
            if len(sg_coords[0]) > 0:
                # 70% probability to focus on a stress granule
                if np.random.random() < 0.7:
                    # Randomly select a stress granule pixel
                    idx = np.random.randint(len(sg_coords[0]))
                    center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                    
                    # Add some random offset to avoid always centering exactly on granule
                    offset_y = np.random.randint(-crop_h//4, crop_h//4)
                    offset_x = np.random.randint(-crop_w//4, crop_w//4)
                    
                    center_y += offset_y
                    center_x += offset_x
                    
                    # Calculate crop boundaries ensuring we stay within image
                    start_h = max(0, min(h - crop_h, center_y - crop_h//2))
                    start_w = max(0, min(w - crop_w, center_x - crop_w//2))
                else:
                    # Random crop without focusing on stress granules
                    start_h = np.random.randint(0, max(1, h - crop_h + 1))
                    start_w = np.random.randint(0, max(1, w - crop_w + 1))
            else:
                # If no stress granules, use random crop
                start_h = np.random.randint(0, max(1, h - crop_h + 1))
                start_w = np.random.randint(0, max(1, w - crop_w + 1))
        else:
            # For validation or if mask is None, use random crop
            start_h = np.random.randint(0, max(1, h - crop_h + 1))
            start_w = np.random.randint(0, max(1, w - crop_w + 1))
        
        # Extract crops
        image_crop = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
        mask_crop = mask[start_h:start_h + crop_h, start_w:start_w + crop_w] if mask is not None else None
        
        return image_crop, mask_crop
    
    def __getitem__(self, idx):
        # Map the expanded index back to the original image index
        img_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image
        
        try:
            # Load 16-bit image
            image = cv2.imread(self.image_paths[img_idx], cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[img_idx]}")
            
            # Load 8-bit mask
            mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_paths[img_idx]}")
            
            # Ensure image is single channel
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply enhanced preprocessing pipeline
            # 1. Convert to float32 for processing
            image = image.astype(np.float32)
            
            # 2. Enhanced contrast using tighter percentile stretching
            image = self._enhance_contrast(image, percentile_low=0.5, percentile_high=99.5)
            
            # 3. Apply Gaussian blur with adjusted sigma
            image = self._apply_gaussian_blur(image, sigma=1.5)
            
            # 4. Extract smart crop with some randomness
            if self.random_crop:
                image, mask = self._smart_crop(image, mask, self.target_size)
            else:
                # Resize directly if not using random crops
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # 5. Normalize mask to binary (0 or 1)
            mask = (mask > 127).astype(np.float32)
            
            # Convert to tensors
            image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
            mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension
            
            # Apply same transforms to both image and mask
            if self.transform:
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                image = self.transform(image)
                torch.manual_seed(seed)
                mask = self.transform(mask)
                
                # Re-binarize mask after transforms
                mask = (mask > 0.5).float()
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading image/mask at index {img_idx}: {e}")
            # Return a default small image in case of error
            image = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            mask = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return image, mask


def improved_train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
                patience=15, device='mps', metrics_dir='metrics/improved_gpu'):
    """
    Improved training function with better learning rate scheduling and regularization
    """
    model = model.to(device)
    
    # Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create a learning rate scheduler that works with epoch number
    class WarmupLRScheduler:
        def __init__(self, optimizer, warmup_epochs=5, initial_lr=learning_rate):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.initial_lr = initial_lr
            self.current_epoch = 0
        
        def step(self):
            # Calculate the new learning rate
            if self.current_epoch < self.warmup_epochs:
                # Linear warm-up
                lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
            else:
                # Step decay after warm-up
                lr = self.initial_lr * (0.8 ** ((self.current_epoch - self.warmup_epochs) // 10))
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Increment epoch counter
            self.current_epoch += 1
            
            return lr
    
    # Initialize custom scheduler
    scheduler = WarmupLRScheduler(optimizer)
    
    # Training tracking variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_precisions = []
    val_recalls = []
    
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Create metrics directory
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Start training
    print("\n🚀 Starting improved training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loop):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate improved loss
            loss = improved_combined_loss(output, target, device, alpha=0.4, pos_weight=5.0)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        # For precision/recall calculation
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = improved_combined_loss(output, target, device, alpha=0.4, pos_weight=5.0)
                
                # Calculate Dice coefficient
                dice = dice_coefficient(output, target)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()
                
                # Store predictions and targets for precision/recall
                pred_binary = (torch.sigmoid(output) > 0.5).float()
                all_predictions.append(pred_binary.cpu())
                all_targets.append(target.cpu())
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)
        
        # Calculate precision and recall
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        
        # True positives, false positives, false negatives
        tp = (all_pred * all_targ).sum().item()
        fp = (all_pred * (1 - all_targ)).sum().item()
        fn = ((1 - all_pred) * all_targ).sum().item()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        val_precisions.append(precision)
        val_recalls.append(recall)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {avg_val_dice:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check if we have a new best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'precision': precision,
                'recall': recall
            }, os.path.join(metrics_dir, 'best_model.pth'))
            
            print(f"  ✅ New best model saved! (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement for {patience_counter} epochs")
        
        # Save visualization of current predictions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            visualize_predictions(model, val_loader, device, os.path.join(metrics_dir, f'predictions_epoch_{epoch+1}.png'))
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.2f} seconds")
    print(f"Best validation Dice score: {best_val_dice:.4f} (epoch {best_epoch+1})")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_dice_scores, val_precisions, val_recalls, metrics_dir)
    
    return best_val_dice, model


def visualize_predictions(model, val_loader, device, output_file):
    """Generate visualization of model predictions"""
    model.eval()
    
    # Get a batch of validation data
    data_iter = iter(val_loader)
    images, masks = next(data_iter)
    
    # Limit to 4 samples for visualization
    images = images[:min(4, len(images))]
    masks = masks[:min(4, len(masks))]
    
    # Generate predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
    
    # Move to CPU for visualization
    images = images.cpu()
    masks = masks.cpu()
    probs = probs.cpu()
    preds = preds.cpu()
    
    # Create figure
    n_samples = images.shape[0]
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    # Ensure axes is 2D even for single sample
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Probability map
        axes[i, 2].imshow(probs[i, 0], cmap='magma')
        axes[i, 2].set_title('Probability Map')
        axes[i, 2].axis('off')
        
        # Binary prediction
        axes[i, 3].imshow(preds[i, 0], cmap='gray')
        # Calculate Dice score for this sample
        dice = dice_coefficient(outputs[i:i+1], masks[i:i+1].to(device)).item()
        axes[i, 3].set_title(f'Prediction (Dice: {dice:.4f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_training_history(train_losses, val_losses, val_dice_scores, val_precisions, val_recalls, metrics_dir):
    """Plot and save training history metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot Dice scores
    plt.subplot(2, 2, 2)
    plt.plot(val_dice_scores, label='Val Dice', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Validation Dice Score')
    
    # Plot Precision
    plt.subplot(2, 2, 3)
    plt.plot(val_precisions, label='Precision', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Validation Precision')
    
    # Plot Recall
    plt.subplot(2, 2, 4)
    plt.plot(val_recalls, label='Recall', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Validation Recall')
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'training_history.png'))
    plt.close()


def create_enhanced_augmentations():
    """Create enhanced data augmentation pipeline"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])


def main():
    """Main function to run improved training"""
    # Configuration
    IMAGE_SIZE = 256
    BATCH_SIZE = 2
    NUM_EPOCHS = 75
    LEARNING_RATE = 8e-5  # Slightly higher learning rate
    PATIENCE = 15
    NUM_SAMPLES = 8  # Use all available data
    CROPS_PER_IMAGE = 3
    
    print("=" * 60)
    print("GPU-Accelerated Training for Stress Granule Detection")
    print(f"Configuration:")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - System: {platform.system()} {platform.machine()}")
    print("=" * 60)
    
    # Check for GPU - specific handling for Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using Apple Silicon GPU (MPS device): {device}")
        print("Apple Silicon GPU acceleration is enabled")
        
        # Print MPS details
        is_built = torch.backends.mps.is_built()
        is_available = torch.backends.mps.is_available()
        print(f"MPS built: {is_built}, available: {is_available}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {device}")
        
        # Print GPU info if NVIDIA
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU: {device}")
        print(" Warning: GPU not available, using CPU instead")
    
    # Match images and masks
    matched_pairs = match_images_and_masks('data/images', 'data/masks', NUM_SAMPLES)
    
    if len(matched_pairs) < 4:
        print("Need at least 4 image-mask pairs")
        return
    
    # Analyze the pairs
    analyze_matched_pairs(matched_pairs)
    
    # Split: Use more for training (70/30 split)
    train_size = int(0.7 * len(matched_pairs))
    train_pairs = matched_pairs[:train_size]
    val_pairs = matched_pairs[train_size:]
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_images)} images × {CROPS_PER_IMAGE} crops = {len(train_images) * CROPS_PER_IMAGE} samples")
    print(f"  Validation: {len(val_images)} images × {CROPS_PER_IMAGE} crops = {len(val_images) * CROPS_PER_IMAGE} samples")
    
    # Create enhanced augmentations
    train_transform = create_enhanced_augmentations()
    
    # Create datasets with improved preprocessing
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
        num_workers=2,  # Optimized for GPU workflows
        pin_memory=use_pin_memory  # Only use with CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory
    )
    
    # Initialize model
    model = ImprovedUNet(in_channels=1, out_channels=1)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test device transfer
    print("\nTesting GPU transfer...")
    try:
        test_tensor = torch.ones((1, 256, 256)).to(device)
        print(f"Successfully transferred tensor to {device}")
        del test_tensor  # Clean up test tensor
    except Exception as e:
        print(f"Error transferring to GPU: {e}")
        print("Falling back to CPU")
        device = torch.device('cpu')
    
    # Set up metrics directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f'metrics/improved_gpu_{timestamp}'
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Metrics will be saved to {metrics_dir}")
    
    # Test data loading
    print("\nTesting data loading...")
    try:
        sample_img, sample_mask = train_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample mask shape: {sample_mask.shape}")
        print(f"Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
        
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
        print("Sample visualization saved")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train model with GPU acceleration
    print(f"\nStarting GPU accelerated training with {NUM_EPOCHS} epochs")
    start_time = time.time()
    
    best_dice, model = improved_train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        device=device,
        metrics_dir=metrics_dir
    )
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nGPU Training complete! Best Dice score: {best_dice:.4f}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save summary
    with open(f"{metrics_dir}/training_summary.txt", "w") as f:
        f.write("GPU Training Summary\n")
        f.write("===================\n")
        f.write(f"Device: {device}\n")
        f.write(f"Best Dice Score: {best_dice:.4f}\n")
        f.write(f"Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        f.write(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Patience: {PATIENCE}\n")
        f.write(f"Crops Per Image: {CROPS_PER_IMAGE}\n")
    
    print(f"Training summary saved to {metrics_dir}/training_summary.txt")
    print(f"Results saved in: {metrics_dir}/")


if __name__ == "__main__":
    main()
