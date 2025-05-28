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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# Custom Dataset Class
class StressGranuleDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256), channels=3):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.channels = channels  # Allow specifying number of channels
        
        # Validate paths exist
        self._validate_paths()
        
    def _validate_paths(self):
        """Check if all image and mask paths exist."""
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image path does not exist: {img_path}")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask path does not exist: {mask_path}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
                
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_paths[idx]}")
                
            # Convert BGR to RGB for image
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
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)  # Use NEAREST for masks
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert mask to binary (adjust threshold if needed)
        mask = (mask > 127).astype(np.float32)  # Binary threshold
        
        # Convert to tensors
        if self.channels == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC to CHW
        else:
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension for grayscale
            
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            # Apply same transform to both image and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        
        return image, mask

# U-Net Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def pad_to_power_of_two(self, x):
        """Pad input to ensure dimensions are powers of 2 for U-Net."""
        _, _, h, w = x.size()
        
        # Calculate next power of 2
        new_h = 2 ** (h-1).bit_length()
        new_w = 2 ** (w-1).bit_length()
        
        # Calculate padding
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        
        # Apply padding symmetrically
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        padded_x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        
        return padded_x, (pad_left, pad_right, pad_top, pad_bottom)

    def remove_padding(self, x, padding):
        """Remove padding added earlier."""
        pad_left, pad_right, pad_top, pad_bottom = padding
        _, _, h, w = x.size()
        
        # Calculate original dimensions
        orig_h = h - pad_top - pad_bottom
        orig_w = w - pad_left - pad_right
        
        # Extract original region
        return x[:, :, pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]

    def forward(self, x):
        # Optional padding for arbitrary input sizes
        padded_x, padding = self.pad_to_power_of_two(x)
        x = padded_x
        
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Handle potential shape mismatch properly
            if x.shape[2:] != skip_connection.shape[2:]:
                try:
                    # Try using PyTorch's interpolation
                    x = F.resize(x, size=skip_connection.shape[2:], interpolation=transforms.InterpolationMode.NEAREST)
                except (AttributeError, TypeError):
                    # Fallback for older PyTorch versions
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='nearest')

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        output = torch.sigmoid(self.final_conv(x))
        
        # Remove padding if added
        if padding != (0, 0, 0, 0):
            output = self.remove_padding(output, padding)
            
        return output

# Training Functions
def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for evaluation"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    # Add smooth factor to avoid division by zero
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss function"""
    return 1 - dice_coefficient(pred, target, smooth)

def combined_loss(pred, target, alpha=0.5):
    """Combination of BCE and Dice loss"""
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice

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
                patience=15, device='cuda', resume_from=None):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize variables
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    best_dice = 0.0
    counter = 0  # For early stopping
    start_epoch = 0
    
    # Try to load checkpoint if resume_from is provided
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch, best_dice = load_checkpoint(model, optimizer, resume_from, device)
        print(f"Resuming from epoch {start_epoch+1} with best dice score: {best_dice:.4f}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    for epoch in range(start_epoch, num_epochs):
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
        
        scheduler.step(val_loss)
        
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
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, val_dice_scores

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
    image_dir = "data/images"
    mask_dir = "data/masks"
    
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
    
    # Verify that there are matching numbers of images and masks
    if len(image_paths) != len(mask_paths):
        print(f"Warning: Unequal number of images ({len(image_paths)}) and masks ({len(mask_paths)})")
        # Trying to match by filename
        matched_images = []
        matched_masks = []
        
        image_basenames = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
        mask_basenames = [os.path.splitext(os.path.basename(p))[0] for p in mask_paths]
        
        for i, img_name in enumerate(image_basenames):
            if img_name in mask_basenames:
                mask_idx = mask_basenames.index(img_name)
                matched_images.append(image_paths[i])
                matched_masks.append(mask_paths[mask_idx])
        
        print(f"Matched {len(matched_images)} image-mask pairs by filename")
        
        if len(matched_images) == 0:
            print("Could not match any images with masks. Please check your data.")
            return
            
        image_paths = matched_images
        mask_paths = matched_masks
    
    # Check if images are grayscale or RGB
    try:
        sample_img = cv2.imread(image_paths[0])
        if sample_img is None:
            raise ValueError(f"Failed to load sample image: {image_paths[0]}")
            
        # Check if grayscale or color
        if len(sample_img.shape) == 2 or sample_img.shape[2] == 1:
            input_channels = 1
        else:
            input_channels = 3
            
        print(f"Detected {input_channels} input channels")
    except Exception as e:
        print(f"Error detecting input channels: {e}")
        print("Defaulting to 3 channels (RGB)")
        input_channels = 3
    
    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
    ])
    
    # Create datasets
    train_dataset = StressGranuleDataset(train_images, train_masks, transform=train_transform, channels=input_channels)
    val_dataset = StressGranuleDataset(val_images, val_masks, channels=input_channels)
    
    # Determine batch size based on GPU memory
    if torch.cuda.is_available():
        # Auto-tune batch size based on GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if gpu_mem > 10:  # High-end GPU
            batch_size = 16
        elif gpu_mem > 6:  # Mid-range GPU
            batch_size = 8
        else:  # Low-end GPU
            batch_size = 4
    else:
        batch_size = 4  # Default for CPU
    
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
    
    # Check for existing checkpoint
    checkpoint_path = 'checkpoint.pth'
    resume_training = os.path.exists(checkpoint_path)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    try:
        train_losses, val_losses, val_dice_scores = train_model(
            model, train_loader, val_loader, 
            num_epochs=100, learning_rate=1e-4, patience=15, device=device,
            resume_from=checkpoint_path if resume_training else None
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")
        
        # Plot training history
        plot_training_history(train_losses, val_losses, val_dice_scores)
        
        # Load best model and visualize predictions
        if os.path.exists('best_stress_granule_model.pth'):
            # Load best model for visualization
            checkpoint = torch.load('best_stress_granule_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with dice score: {checkpoint['best_dice']:.4f}")
            
            visualize_predictions(model, val_dataset, device)
        else:
            print("Warning: Best model file not found. Using last model state.")
            visualize_predictions(model, val_dataset, device)
        
        print("Training completed! Best model saved as 'best_stress_granule_model.pth'")
        
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