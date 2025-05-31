#!/usr/bin/env python
"""
Simplified training script to recover previous performance (~0.5 Dice score)
Focuses on proven techniques for stress granule detection
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from glob import glob
import time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Import just what we need from our modules
from models import dice_coefficient, dice_loss
from train_small_test import match_images_and_masks, analyze_matched_pairs


class OptimizedStressGranuleDataset(Dataset):
    """Optimized dataset for stress granule detection with proven preprocessing"""
    
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
    
    def _enhance_contrast(self, image, percentile_low=1, percentile_high=99):
        """Enhance contrast using percentile stretching (proven approach)"""
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
    
    def _apply_gaussian_blur(self, image, sigma=1.7):
        """Apply Gaussian blur to reduce noise"""
        # Calculate kernel size based on sigma (6*sigma + 1, must be odd)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def _focus_crop(self, image, mask, crop_size):
        """Extract a crop with focus on stress granules when possible"""
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
                # With 80% probability, focus on a stress granule
                if np.random.random() < 0.8:
                    # Randomly select a stress granule pixel
                    idx = np.random.randint(len(sg_coords[0]))
                    center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                    
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
            
            # Store original shapes for debugging
            original_shape = image.shape
            
            # Apply proven preprocessing pipeline
            # 1. Convert to float32 for processing
            image = image.astype(np.float32)
            
            # 2. Enhance contrast using percentile stretching
            image = self._enhance_contrast(image, percentile_low=1, percentile_high=99)
            
            # 3. Apply Gaussian blur to reduce noise
            image = self._apply_gaussian_blur(image, sigma=1.7)
            
            # 4. Extract crop (random or focused on stress granules)
            if self.random_crop:
                image, mask = self._focus_crop(image, mask, self.target_size)
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


# Simple U-Net with proven configuration
class SimpleUNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Downsampling path
        self.down_conv1 = self._double_conv(in_channels, 64)
        self.down_conv2 = self._double_conv(64, 128)
        self.down_conv3 = self._double_conv(128, 256)
        self.down_conv4 = self._double_conv(256, 512)
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Upsampling path
        self.up_trans1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = self._double_conv(1024, 512)
        
        self.up_trans2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = self._double_conv(512, 256)
        
        self.up_trans3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = self._double_conv(256, 128)
        
        self.up_trans4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = self._double_conv(128, 64)
        
        # Final convolution
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _double_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Downsampling path with skip connections
        conv1 = self.down_conv1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.down_conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down_conv3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.down_conv4(x)
        x = self.maxpool(conv4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling path with skip connections
        x = self.up_trans1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv1(x)
        
        x = self.up_trans2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv2(x)
        
        x = self.up_trans3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv3(x)
        
        x = self.up_trans4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv4(x)
        
        # Final convolution
        return self.final_conv(x)


# Custom loss function with proven weights
def combined_bce_dice_loss(pred, target, bce_weight=0.3, dice_weight=0.7, smooth=1e-6):
    """Combined BCE and Dice loss with proven weights (30% BCE, 70% Dice)"""
    # Binary cross entropy
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    
    # Dice loss
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = 1 - ((2. * intersection + smooth) / (union + smooth)).mean()
    
    # Combined loss
    return bce_weight * bce + dice_weight * dice


def train_and_evaluate(max_epochs=75, batch_size=2, image_size=256, crops_per_image=3, learning_rate=5e-5):
    """Train and evaluate the model with proven configuration"""
    print(f"\n{'='*60}")
    print(f"Stress Granule Detection - Recovery Training")
    print(f"Configuration:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Crops per image: {crops_per_image}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"{'='*60}")
    
    # Set device to CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Match images and masks
    matched_pairs = match_images_and_masks('data/images', 'data/masks')
    
    if len(matched_pairs) < 4:
        print(f"❌ Need at least 4 image-mask pairs")
        return
    
    # Analyze matched pairs
    analyze_matched_pairs(matched_pairs[:4])
    
    # Use 3 for training, 1 for validation (proven split)
    train_pairs = matched_pairs[:3]
    val_pairs = matched_pairs[3:4]
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_images)} images × {crops_per_image} crops = {len(train_images) * crops_per_image} samples")
    print(f"  Validation: {len(val_images)} images × {crops_per_image} crops = {len(val_images) * crops_per_image} samples")
    
    # Create transforms (proven augmentations)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    # Create datasets
    train_dataset = OptimizedStressGranuleDataset(
        train_images, train_masks,
        transform=train_transform,
        target_size=(image_size, image_size),
        random_crop=True,
        crops_per_image=crops_per_image
    )
    
    val_dataset = OptimizedStressGranuleDataset(
        val_images, val_masks,
        target_size=(image_size, image_size),
        random_crop=True,
        crops_per_image=crops_per_image
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = SimpleUNet(in_channels=1, out_channels=1)
    model = model.to(device)
    print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create metrics directory
    metrics_dir = 'metrics/recovery_training'
    os.makedirs(metrics_dir, exist_ok=True)
    
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
        print(f"✅ Sample visualization saved to {metrics_dir}/sample_data.png")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Optimizer with proven learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (proven approach)
    scheduler = torch.optim.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training tracking variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Start training
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch_idx, (data, target) in enumerate(train_loop):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate loss (proven 30% BCE, 70% Dice)
            loss = combined_bce_dice_loss(output, target, bce_weight=0.3, dice_weight=0.7)
            
            # Backward pass
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
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]")
            
            for data, target in val_loop:
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = combined_bce_dice_loss(output, target, bce_weight=0.3, dice_weight=0.7)
                
                # Calculate Dice coefficient
                dice = dice_coefficient(output, target)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()
                
                val_loop.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{max_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {avg_val_dice:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
                'val_dice': avg_val_dice
            }, os.path.join(metrics_dir, 'best_model.pth'))
            
            print(f"  ✅ New best model saved! (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement for {patience_counter} epochs")
        
        # Early stopping with patience 10
        if patience_counter >= 10:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.2f} seconds")
    print(f"Best validation Dice score: {best_val_dice:.4f} (epoch {best_epoch+1})")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot Dice scores
    plt.subplot(1, 3, 2)
    plt.plot(val_dice_scores, label='Val Dice', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Validation Dice Score')
    
    # Plot learning rate
    learning_rates = []
    for i in range(len(train_losses)):
        if i == 0:
            learning_rates.append(learning_rate)
        else:
            # Approximate LR based on scheduler behavior
            prev_lr = learning_rates[-1]
            if i > 5 and val_losses[i] >= val_losses[i-1]:
                learning_rates.append(prev_lr * 0.5)  # Assuming factor=0.5 in scheduler
            else:
                learning_rates.append(prev_lr)
    
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'training_history.png'))
    plt.close()
    
    # Load best model for visualization
    checkpoint = torch.load(os.path.join(metrics_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate visualization with best model
    print("\n🎨 Generating prediction visualizations...")
    model.eval()
    
    # Choose a few samples for visualization
    num_samples = min(3, len(val_dataset))
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and mask
            image, mask = val_dataset[idx]
            
            # Move to device
            image = image.unsqueeze(0).to(device)
            
            # Generate prediction
            output = model(image)
            output = torch.sigmoid(output)
            
            # Move tensors to CPU for visualization
            image = image.squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()
            pred = output.squeeze().cpu().numpy()
            
            # Convert prediction to binary
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Calculate Dice for this sample
            dice = np.sum(2 * pred_binary * mask) / (np.sum(pred_binary) + np.sum(mask) + 1e-6)
            
            # Plot original image
            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # Plot ground truth mask
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Plot prediction
            axes[i, 2].imshow(pred_binary, cmap='gray')
            axes[i, 2].set_title(f'Prediction (Dice: {dice:.4f})')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'predictions.png'))
    plt.close()
    
    print(f"✅ Results saved in {metrics_dir}/")
    return best_val_dice, model


if __name__ == "__main__":
    # Run with proven parameters that achieved ~0.5 Dice score
    best_dice, model = train_and_evaluate(
        max_epochs=75,
        batch_size=2,
        image_size=256,  # Smaller crop size that works better
        crops_per_image=3,  # Multiple crops per image
        learning_rate=5e-5  # Proven learning rate
    )
    
    print(f"\n🎉 Training complete! Best Dice score: {best_dice:.4f}")
    print("Next steps:")
    print("  1. Review metrics and visualizations in 'metrics/recovery_training/'")
    print("  2. If Dice score is still below ~0.5, try adjusting parameters")
    print("  3. When performance is recovered, move to GPU training with larger dataset")
