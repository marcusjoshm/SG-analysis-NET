import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
from tqdm import tqdm

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
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        try:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
                
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_paths[idx]}")
                
        except Exception as e:
            print(f"Error loading images: {e}")
            # Return a default small image in case of error
            image = np.zeros((64, 64), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
        
        # Convert to RGB if required
        if self.channels == 3 and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
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

    def forward(self, x):
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
                x = F.resize(x, size=skip_connection.shape[2:], interpolation=transforms.InterpolationMode.NEAREST)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

# Training Functions
def dice_coefficient(pred, target, smooth=1):
    """Calculate Dice coefficient for evaluation"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def dice_loss(pred, target, smooth=1):
    """Dice loss function"""
    return 1 - dice_coefficient(pred, target, smooth)

def combined_loss(pred, target, alpha=0.5):
    """Combination of BCE and Dice loss"""
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
                patience=15, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    best_dice = 0.0
    counter = 0  # For early stopping
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = combined_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += combined_loss(output, target).item()
                val_dice += dice_coefficient(output, target).item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_stress_granule_model.pth')
            counter = 0  # Reset counter when improvement found
        else:
            counter += 1
        
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_dice_scores, label='Val Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, dataset, device='cuda', num_samples=4):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(dataset))
            image, mask = dataset[idx]
            
            # Predict
            image_batch = image.unsqueeze(0).to(device)
            pred = model(image_batch).cpu().squeeze()
            
            # Convert to numpy for visualization
            if image.shape[0] == 3:  # RGB
                image_np = image.permute(1, 2, 0).numpy()
            else:  # Grayscale
                image_np = image.squeeze().numpy()
                
            mask_np = mask.squeeze().numpy()
            pred_np = (pred > 0.5).float().numpy()
            
            # Plot
            if image.shape[0] == 1:  # Handle grayscale images
                axes[i, 0].imshow(image_np, cmap='gray')
            else:
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
    plt.savefig('predictions_visualization.png')
    plt.show()

# Main execution function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    image_dir = "data/images"
    mask_dir = "data/masks"
    
    # Check if directories exist, create if not
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Support multiple file extensions
    extensions = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
    
    # Get file paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(sorted(glob(os.path.join(image_dir, ext))))
    
    mask_paths = []
    for ext in extensions:
        mask_paths.extend(sorted(glob(os.path.join(mask_dir, ext))))
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    if len(image_paths) == 0 or len(mask_paths) == 0:
        print(f"No images or masks found in {image_dir} and {mask_dir}")
        print("Please add your data files before running training")
        return
    
    # Verify data alignment
    assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
    
    # Check if images are grayscale or RGB
    sample_img = cv2.imread(image_paths[0])
    input_channels = 3 if sample_img.ndim == 3 else 1
    print(f"Detected {input_channels} input channels")
    
    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
    ])
    
    # Create datasets
    train_dataset = StressGranuleDataset(train_images, train_masks, transform=train_transform, channels=input_channels)
    val_dataset = StressGranuleDataset(val_images, val_masks, channels=input_channels)
    
    # Create data loaders
    batch_size = 8
    # Adjust num_workers based on available CPU cores
    num_workers = min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize model
    model = UNet(in_channels=input_channels, out_channels=1)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, val_dice_scores = train_model(
        model, train_loader, val_loader, 
        num_epochs=100, learning_rate=1e-4, patience=15, device=device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_dice_scores)
    
    # Load best model and visualize predictions
    model.load_state_dict(torch.load('best_stress_granule_model.pth'))
    visualize_predictions(model, val_dataset, device)
    
    print("Training completed! Best model saved as 'best_stress_granule_model.pth'")

if __name__ == "__main__":
    main()