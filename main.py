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

# Import model classes and functions
from models import UNet, StressGranuleDataset, StressGranule16bitDataset, dice_coefficient, dice_loss, combined_loss

# Import metrics system
from metrics import MetricsTracker, evaluate_thresholds, plot_threshold_analysis

# Training Functions
def save_checkpoint(model, optimizer, epoch, best_dice, filename):
    """Save checkpoint with all important information to resume training"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_dice': best_dice
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename, device):
    """Load checkpoint to resume training"""
    try:
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_dice = checkpoint['best_dice']
        
        # Update optimizer device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
                    
        return model, optimizer, epoch, best_dice
    except FileNotFoundError:
        print(f"Checkpoint file {filename} not found. Starting from scratch.")
        return model, optimizer, 0, 0.0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model, optimizer, 0, 0.0

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
                patience=15, device='cuda', resume_from=None, metrics_tracker=None,
                experiment_name=None):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    best_dice = 0.0
    counter = 0  # For early stopping
    start_epoch = 0
    
    # Create metrics tracker if not provided
    if metrics_tracker is None:
        metrics_tracker = MetricsTracker(experiment_name=experiment_name)
    
    # Try to load checkpoint if resume_from is provided
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch, best_dice = load_checkpoint(model, optimizer, resume_from, device)
        print(f"Resuming from epoch {start_epoch+1} with best dice score: {best_dice:.4f}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    for epoch in range(start_epoch, num_epochs):
        # Start timing for this epoch
        metrics_tracker.start_epoch()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(loop):
            # Handle empty batch (rare case due to errors)
            if data.nelement() == 0 or target.nelement() == 0:
                print(f"Warning: Empty batch encountered (batch_idx={batch_idx})")
                continue
                
            try:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Handle shape mismatch between output and target
                if output.shape != target.shape:
                    output = F.resize(output, size=target.shape[2:], 
                                     interpolation=transforms.InterpolationMode.NEAREST)
                
                loss = combined_loss(output, target)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Update progress bar
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
        
        # For tracking all predictions and targets for metrics calculation
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    # Handle empty batch
                    if data.nelement() == 0 or target.nelement() == 0:
                        continue
                        
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    # Handle shape mismatch
                    if output.shape != target.shape:
                        output = F.resize(output, size=target.shape[2:], 
                                         interpolation=transforms.InterpolationMode.NEAREST)
                    
                    val_loss += combined_loss(output, target).item()
                    val_dice += dice_coefficient(output, target).item()
                    val_samples += 1
                    
                    # Collect predictions and targets for metrics
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
            print("Warning: No valid samples in validation set!")
            val_loss = float('inf')
            val_dice = 0.0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Concatenate all validation predictions and targets
        all_preds_tensor = torch.cat(all_val_preds, dim=0) if all_val_preds else torch.tensor([])
        all_targets_tensor = torch.cat(all_val_targets, dim=0) if all_val_targets else torch.tensor([])
        
        # Update metrics tracker
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_tracker.end_epoch(epoch, train_loss, val_loss, all_preds_tensor, all_targets_tensor, current_lr)
        
        # Get current metrics summary
        metrics_summary = metrics_tracker.get_current_metrics_summary()
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, best_dice, 'best_stress_granule_model.pth')
            counter = 0  # Reset counter when improvement found
        else:
            counter += 1
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, best_dice, 'checkpoint.pth')
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Dice: {val_dice:.4f}')
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
    
    # Final metrics plots
    metrics_tracker.plot_metrics()
    
    # Find optimal threshold on validation set
    if len(all_val_preds) > 0:
        try:
            threshold_results = evaluate_thresholds(model, val_loader, device)
            plot_threshold_analysis(threshold_results, save_path=os.path.join(metrics_tracker.save_dir, 'threshold_analysis.png'))
        except Exception as e:
            print(f"Error during threshold analysis: {e}")
    
    return train_losses, val_losses, val_dice_scores, metrics_tracker

def plot_training_history(train_losses, val_losses, val_dice_scores):
    """Plot training metrics"""
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot dice scores
        ax2.plot(val_dice_scores, label='Val Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Validation Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Error plotting training history: {e}")

def visualize_predictions(model, dataset, device='cuda', num_samples=4, output_file='predictions_visualization.png'):
    """Visualize model predictions"""
    model.eval()
    
    # Determine actual number of samples (min of requested and available)
    num_samples = min(num_samples, len(dataset))
    
    if num_samples == 0:
        print("No samples available for visualization.")
        return
    
    try:
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        # Handle case of single sample
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for i in range(num_samples):
                # Get random sample
                idx = np.random.randint(0, len(dataset))
                image, mask = dataset[idx]
                
                # Handle single image case
                if image.dim() == 2:
                    image = image.unsqueeze(0)  # Add channel dimension
                    
                # Predict
                image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
                pred = model(image_batch).cpu().squeeze()
                
                # Convert to numpy for visualization
                if image.shape[0] == 3:  # RGB
                    image_np = image.permute(1, 2, 0).numpy()
                else:  # Grayscale
                    image_np = image.squeeze().numpy()
                    
                mask_np = mask.squeeze().numpy()
                
                # Threshold prediction to binary
                pred_np = (pred > 0.5).float().numpy()
                
                # Handle single channel images for display
                if image.shape[0] == 1:  # Handle grayscale images
                    axes[i, 0].imshow(image_np, cmap='gray')
                else:
                    # Clip to [0, 1] range for display
                    image_np = np.clip(image_np, 0, 1)
                    axes[i, 0].imshow(image_np)
                
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask_np, cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred_np, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Predictions visualization saved as '{output_file}'")
    except Exception as e:
        print(f"Error visualizing predictions: {e}")

def match_images_and_masks_custom(image_paths, mask_paths):
    """
    Custom function to match images and masks based on the naming convention:
    - Images: {name}_ch00_t00.tif
    - Masks: MASK_{name}_t00.tif
    """
    matched_images = []
    matched_masks = []
    
    # Extract core names from image paths
    image_core_names = {}
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # Remove '_ch00_t00' suffix to get core name
        if basename.endswith('_ch00_t00'):
            core_name = basename[:-9]  # Remove last 9 characters '_ch00_t00'
            image_core_names[core_name] = img_path
        else:
            # Fallback: use full basename
            image_core_names[basename] = img_path
    
    # Extract core names from mask paths and match
    for mask_path in mask_paths:
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        # Remove 'MASK_' prefix and '_t00' suffix to get core name
        if basename.startswith('MASK_') and basename.endswith('_t00'):
            core_name = basename[5:-4]  # Remove 'MASK_' (5 chars) and '_t00' (4 chars)
            if core_name in image_core_names:
                matched_images.append(image_core_names[core_name])
                matched_masks.append(mask_path)
    
    return matched_images, matched_masks

def find_best_image_format(image_dir):
    """Find the most common image format in the directory"""
    extensions = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
    counts = {}
    
    for ext in extensions:
        count = len(glob(os.path.join(image_dir, ext)))
        if count > 0:
            counts[ext] = count
    
    if not counts:
        return extensions  # Return all if none found
        
    # Sort by count (most common first)
    sorted_exts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [ext for ext, _ in sorted_exts]

# Main execution function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a U-Net model for stress granule segmentation')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory for data (default: data)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images (default: data/images)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing masks (default: data/masks)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (default: 256)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (default: timestamp)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint if available')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='Checkpoint file to resume from (default: checkpoint.pth)')
    
    args = parser.parse_args()
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set image and mask directories
    if args.image_dir is None:
        args.image_dir = os.path.join(args.data_dir, 'images')
    if args.mask_dir is None:
        args.mask_dir = os.path.join(args.data_dir, 'masks')
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Print available memory
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Data paths
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    
    # Check if directories exist, create if not
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Find most common file extensions in the directory
    image_extensions = find_best_image_format(image_dir)
    mask_extensions = find_best_image_format(mask_dir)
    
    # Get file paths
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(sorted(glob(os.path.join(image_dir, ext))))
    
    mask_paths = []
    for ext in mask_extensions:
        mask_paths.extend(sorted(glob(os.path.join(mask_dir, ext))))
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    if len(image_paths) == 0 or len(mask_paths) == 0:
        print(f"No images or masks found in {image_dir} and {mask_dir}")
        print("Please add your data files before running training")
        return
    
    # Always use custom matching to properly pair images and masks based on naming convention
    print("Matching images and masks based on naming convention...")
    matched_images, matched_masks = match_images_and_masks_custom(image_paths, mask_paths)
    
    print(f"Matched {len(matched_images)} image-mask pairs")
    
    if len(matched_images) == 0:
        print("Could not match any images with masks. Please check your data naming convention:")
        print("Expected: Images ending with '_ch00_t00.tif' and masks starting with 'MASK_' and ending with '_t00.tif'")
        return
    
    # Use the matched pairs
    image_paths = matched_images
    mask_paths = matched_masks
    
    # Check if images are grayscale or RGB
    try:
        sample_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)  # Read unchanged to preserve bit depth
        if sample_img is None:
            raise ValueError(f"Failed to load sample image: {image_paths[0]}")
            
        # Check bit depth and channels
        is_16bit = sample_img.dtype == np.uint16
        is_grayscale = len(sample_img.shape) == 2 or sample_img.shape[2] == 1
        
        # For your stress granule data, we expect single channel 16-bit images
        input_channels = 1  # Force single channel for stress granule analysis
        
        print(f"Detected image properties:")
        print(f"  - Bit depth: {'16-bit' if is_16bit else '8-bit'}")
        print(f"  - Channels: {input_channels} (grayscale)")
        print(f"  - Max value in sample: {sample_img.max()}")
        
    except Exception as e:
        print(f"Error detecting input channels: {e}")
        print("Defaulting to 1 channel (grayscale)")
        input_channels = 1
        is_16bit = True  # Assume 16-bit for safety
    
    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Create metrics directory
    metrics_dir = os.path.join('metrics', args.experiment_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir=metrics_dir, experiment_name=args.experiment_name)
    
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
    ])
    
    # Create datasets using the 16-bit dataset class for stress granule data
    train_dataset = StressGranule16bitDataset(
        train_images, train_masks, 
        transform=train_transform, 
        target_size=(args.image_size, args.image_size),
        enhance_contrast=True,  # Enable contrast enhancement for low intensity values
        gaussian_sigma=1.7      # Apply Gaussian blur as requested
    )
    
    val_dataset = StressGranule16bitDataset(
        val_images, val_masks, 
        target_size=(args.image_size, args.image_size),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    # Determine batch size based on GPU memory
    if torch.cuda.is_available():
        # Auto-tune batch size based on GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if gpu_mem > 10:  # High-end GPU
            batch_size = min(16, args.batch_size)
        elif gpu_mem > 6:  # Mid-range GPU
            batch_size = min(8, args.batch_size)
        else:  # Low-end GPU
            batch_size = min(4, args.batch_size)
    else:
        batch_size = args.batch_size
    
    print(f"Using batch size: {batch_size}")
    
    # Adjust num_workers based on available CPU cores
    num_workers = min(4, os.cpu_count() or 1)
    
    # Set up data loaders with error handling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster data transfer to GPU
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # Initialize model
    model = UNet(in_channels=input_channels, out_channels=1)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Check for existing checkpoint if resuming
    checkpoint_path = args.checkpoint if args.resume else None
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    try:
        train_losses, val_losses, val_dice_scores, metrics_tracker = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            patience=args.patience, 
            device=device,
            resume_from=checkpoint_path,
            metrics_tracker=metrics_tracker,
            experiment_name=args.experiment_name
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")
        
        # Load best model and visualize predictions
        if os.path.exists('best_stress_granule_model.pth'):
            # Load best model for visualization
            checkpoint = torch.load('best_stress_granule_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with dice score: {checkpoint['best_dice']:.4f}")
            
            # Visualize predictions
            visualize_predictions(
                model, val_dataset, device, 
                output_file=os.path.join(metrics_dir, 'predictions_visualization.png')
            )
            
            # Get confusion matrix
            val_preds = []
            val_targets = []
            
            model.eval()
            with torch.no_grad():
                for data, target in tqdm(val_loader, desc="Computing confusion matrix"):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_preds.append(output.cpu())
                    val_targets.append(target.cpu())
            
            all_preds = torch.cat(val_preds, dim=0)
            all_targets = torch.cat(val_targets, dim=0)
            
            # Plot confusion matrix
            metrics_tracker.plot_confusion_matrix(all_preds, all_targets)
        else:
            print("Warning: Best model file not found. Using last model state.")
            visualize_predictions(model, val_dataset, device)
        
        print("Training completed! Best model saved as 'best_stress_granule_model.pth'")
        print(f"Metrics and visualizations saved in {metrics_dir}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model state...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 'interrupted'
        }, 'interrupted_model.pth')
        print("Model state saved as 'interrupted_model.pth'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()