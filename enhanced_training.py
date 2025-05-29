import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
from tqdm import tqdm
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# Import from existing modules
from models import UNet, dice_coefficient, dice_loss, combined_loss
from metrics import MetricsTracker, evaluate_thresholds, plot_threshold_analysis
from main import match_images_and_masks_custom, save_checkpoint, load_checkpoint

class EnhancedStressGranuleDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256), 
                 channels=3, is_training=True, contrast_enhancement=True, gaussian_blur=True, blur_sigma=1.7):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.channels = channels
        self.is_training = is_training
        self.contrast_enhancement = contrast_enhancement
        self.gaussian_blur = gaussian_blur
        self.blur_sigma = blur_sigma
        
        # Validate paths exist
        self._validate_paths()
        
    def _validate_paths(self):
        """Check if all image and mask paths exist."""
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image path does not exist: {img_path}")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask path does not exist: {mask_path}")
        
    def _apply_gaussian_blur(self, image):
        """Apply Gaussian blur to reduce noise and improve segmentation"""
        if self.gaussian_blur:
            # Calculate kernel size based on sigma (rule of thumb: 6*sigma + 1, must be odd)
            kernel_size = int(6 * self.blur_sigma + 1)
            if kernel_size % 2 == 0:  # Ensure odd kernel size
                kernel_size += 1
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), self.blur_sigma)
            return blurred
        else:
            return image
        
    def _enhance_contrast(self, image):
        """Enhanced contrast adjustment for 16-bit images with low dynamic range"""
        if image.dtype == np.uint16:
            # Convert to float for processing
            img_float = image.astype(np.float32)
            
            # Get current range
            img_min = img_float.min()
            img_max = img_float.max()
            
            if img_max > img_min:
                # Stretch to better utilize dynamic range
                # Map current range to a larger portion of 0-255 range
                stretched = (img_float - img_min) / (img_max - img_min)
                
                if self.is_training:
                    # Add random brightness/contrast adjustments during training
                    brightness_factor = np.random.uniform(0.7, 1.3)  # ±30% brightness
                    contrast_factor = np.random.uniform(0.8, 1.5)   # ±50% contrast
                    
                    # Apply brightness and contrast
                    stretched = stretched * contrast_factor + (brightness_factor - 1.0)
                    stretched = np.clip(stretched, 0, 1)
                else:
                    # Fixed enhancement for validation
                    contrast_factor = 1.2  # 20% contrast boost
                    stretched = stretched * contrast_factor
                    stretched = np.clip(stretched, 0, 1)
                
                # Convert to 8-bit range
                enhanced = (stretched * 255).astype(np.uint8)
            else:
                enhanced = np.zeros_like(image, dtype=np.uint8)
                
            return enhanced
        else:
            # Already 8-bit, apply smaller adjustments
            if self.is_training:
                brightness = np.random.uniform(0.8, 1.2)
                contrast = np.random.uniform(0.9, 1.3)
                adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness*20)
                return adjusted
            else:
                return image
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        try:
            # Load image with explicit 16-bit support
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
                
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_paths[idx]}")
            
            # Apply contrast enhancement first
            if self.contrast_enhancement:
                image = self._enhance_contrast(image)
            
            # Apply Gaussian blur
            image = self._apply_gaussian_blur(image)
            
            # Convert BGR to RGB for image (if color)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            print(f"Error loading images: {e}")
            # Return a default small image in case of error
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
        
        # Convert to grayscale if needed
        if self.channels == 1 and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1] - now working with enhanced contrast
        image = image.astype(np.float32) / 255.0
        
        # Convert mask to binary
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors
        if self.channels == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC to CHW
        else:
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        # Apply geometric transforms
        if self.transform and self.is_training:
            # Apply same transform to both image and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            
            # Re-binarize mask after transforms (fixes interpolation issues)
            mask = (mask > 0.5).float()
        
        return image, mask

def create_enhanced_augmentations():
    """Create enhanced data augmentation pipeline"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),  # Increased rotation
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ], p=0.3),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),
    ])

def enhanced_train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
                        patience=25, device='cuda', resume_from=None, metrics_tracker=None,
                        experiment_name=None):
    """Enhanced training function with better learning rate scheduling"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    
    # Initialize variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    best_dice = 0.0
    counter = 0
    start_epoch = 0
    
    if metrics_tracker is None:
        metrics_tracker = MetricsTracker(experiment_name=experiment_name)
    
    # Try to load checkpoint if resume_from is provided
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch, best_dice = load_checkpoint(model, optimizer, resume_from, device)
        print(f"Resuming from epoch {start_epoch+1} with best dice score: {best_dice:.4f}")
    
    # Enhanced learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=8, factor=0.7, min_lr=1e-7
    )
    
    print(f"Starting enhanced training with {len(train_loader.dataset)} training samples...")
    
    for epoch in range(start_epoch, num_epochs):
        metrics_tracker.start_epoch()
        
        # Training phase
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(loop):
            if data.nelement() == 0 or target.nelement() == 0:
                continue
                
            try:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                if output.shape != target.shape:
                    output = F.resize(output, size=target.shape[2:], 
                                     interpolation=transforms.InterpolationMode.NEAREST)
                
                loss = combined_loss(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print(f"Runtime error in training batch {batch_idx}: {e}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_samples = 0
        
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    if data.nelement() == 0 or target.nelement() == 0:
                        continue
                        
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    if output.shape != target.shape:
                        output = F.resize(output, size=target.shape[2:], 
                                         interpolation=transforms.InterpolationMode.NEAREST)
                    
                    val_loss += combined_loss(output, target).item()
                    val_dice += dice_coefficient(output, target).item()
                    val_samples += 1
                    
                    all_val_preds.append(output.detach().cpu())
                    all_val_targets.append(target.detach().cpu())
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
        
        # Calculate averages
        train_loss /= max(1, len(train_loader))
        if val_samples > 0:
            val_loss /= val_samples
            val_dice /= val_samples
        else:
            val_loss = float('inf')
            val_dice = 0.0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        
        # Update learning rate scheduler (based on dice score now)
        scheduler.step(val_dice)
        
        # Update metrics tracker
        all_preds_tensor = torch.cat(all_val_preds, dim=0) if all_val_preds else torch.tensor([])
        all_targets_tensor = torch.cat(all_val_targets, dim=0) if all_val_targets else torch.tensor([])
        
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_tracker.end_epoch(epoch, train_loss, val_loss, all_preds_tensor, all_targets_tensor, current_lr)
        
        metrics_summary = metrics_tracker.get_current_metrics_summary()
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, best_dice, 'best_enhanced_model.pth')
            counter = 0
            print(f"✅ New best model! Dice: {best_dice:.4f}")
        else:
            counter += 1
        
        # Regular checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, best_dice, 'enhanced_checkpoint.pth')
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Dice: {val_dice:.4f} (Best: {best_dice:.4f})')
        print(f'  Precision: {metrics_summary["precision"]:.4f}')
        print(f'  Recall: {metrics_summary["recall"]:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Plot metrics every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            metrics_tracker.plot_metrics()
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final processing
    metrics_tracker.plot_metrics()
    
    if len(all_val_preds) > 0:
        try:
            threshold_results = evaluate_thresholds(model, val_loader, device)
            plot_threshold_analysis(threshold_results, 
                                   save_path=os.path.join(metrics_tracker.save_dir, 'enhanced_threshold_analysis.png'))
        except Exception as e:
            print(f"Error during threshold analysis: {e}")
    
    return train_losses, val_losses, val_dice_scores, metrics_tracker

def main():
    parser = argparse.ArgumentParser(description='Enhanced training for stress granule segmentation')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory for data (default: data)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=75,
                        help='Number of epochs to train (default: 75)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience (default: 25)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (default: 256)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--no_contrast_enhancement', action='store_true',
                        help='Disable contrast enhancement')
    parser.add_argument('--no_gaussian_blur', action='store_true',
                        help='Disable Gaussian blur preprocessing')
    parser.add_argument('--blur_sigma', type=float, default=1.7,
                        help='Sigma value for Gaussian blur (default: 1.7)')
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        args.experiment_name = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load and match data
    image_paths = sorted(glob(os.path.join(args.data_dir, 'images', '*.tif')))
    mask_paths = sorted(glob(os.path.join(args.data_dir, 'masks', '*.tif')))
    
    matched_images, matched_masks = match_images_and_masks_custom(image_paths, mask_paths)
    
    print(f"Found {len(matched_images)} matched image-mask pairs")
    
    if len(matched_images) == 0:
        print("No data found!")
        return
    
    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        matched_images, matched_masks, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Check image properties
    sample_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    print(f"Sample image shape: {sample_img.shape}, dtype: {sample_img.dtype}")
    print(f"Sample image range: {sample_img.min()} - {sample_img.max()}")
    
    input_channels = 3 if len(sample_img.shape) == 3 else 1
    
    # Create enhanced datasets
    train_transform = create_enhanced_augmentations()
    
    train_dataset = EnhancedStressGranuleDataset(
        train_images, train_masks, 
        transform=train_transform,
        channels=input_channels,
        target_size=(args.image_size, args.image_size),
        is_training=True,
        contrast_enhancement=not args.no_contrast_enhancement,
        gaussian_blur=not args.no_gaussian_blur,
        blur_sigma=args.blur_sigma
    )
    
    val_dataset = EnhancedStressGranuleDataset(
        val_images, val_masks,
        channels=input_channels,
        target_size=(args.image_size, args.image_size),
        is_training=False,
        contrast_enhancement=not args.no_contrast_enhancement,
        gaussian_blur=not args.no_gaussian_blur,
        blur_sigma=args.blur_sigma
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # Initialize model
    model = UNet(in_channels=input_channels, out_channels=1)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create metrics tracker
    metrics_dir = os.path.join('metrics', args.experiment_name)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_tracker = MetricsTracker(save_dir=metrics_dir, experiment_name=args.experiment_name)
    
    # Start training
    print("🚀 Starting enhanced training with contrast improvements...")
    print(f"📋 Preprocessing pipeline:")
    print(f"   - Contrast enhancement: {'✅ Enabled' if not args.no_contrast_enhancement else '❌ Disabled'}")
    print(f"   - Gaussian blur: {'✅ Enabled' if not args.no_gaussian_blur else '❌ Disabled'} (σ={args.blur_sigma})")
    start_time = time.time()
    
    try:
        train_losses, val_losses, val_dice_scores, metrics_tracker = enhanced_train_model(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            patience=args.patience, 
            device=device,
            resume_from='enhanced_checkpoint.pth' if args.resume else None,
            metrics_tracker=metrics_tracker,
            experiment_name=args.experiment_name
        )
        
        training_time = time.time() - start_time
        print(f"🎉 Enhanced training completed in {training_time/60:.2f} minutes")
        
        print("\n📊 Creating enhanced visualizations...")
        # Import and run visualization
        from enhanced_visualization import create_enhanced_visualizations, create_performance_summary
        
        create_enhanced_visualizations('best_enhanced_model.pth', save_dir=f'enhanced_visualizations_{args.experiment_name}')
        create_performance_summary(f'{metrics_dir}/{args.experiment_name}_metrics.csv', 
                                  save_dir=f'enhanced_visualizations_{args.experiment_name}')
        
        print(f"✅ All outputs saved!")
        print(f"Best model: best_enhanced_model.pth")
        print(f"Metrics: {metrics_dir}")
        print(f"Visualizations: enhanced_visualizations_{args.experiment_name}")
        
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 'interrupted'
        }, 'enhanced_interrupted_model.pth')
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 