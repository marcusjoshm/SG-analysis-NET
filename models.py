import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# Custom Dataset Class
class StressGranuleDataset(torch.utils.data.Dataset):
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

# Enhanced Dataset Class for 16-bit images with preprocessing
class StressGranule16bitDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256), 
                 enhance_contrast=True, gaussian_sigma=1.7, random_crop=False,
                 crops_per_image=3):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast
        self.gaussian_sigma = gaussian_sigma
        self.random_crop = random_crop
        self.crops_per_image = crops_per_image if random_crop else 1
        
        # Validate paths exist
        self._validate_paths()
    
    def __len__(self):
        return len(self.image_paths) * self.crops_per_image
    
    def _validate_paths(self):
        """Check if all image and mask paths exist."""
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image path does not exist: {img_path}")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask path does not exist: {mask_path}")
        
    def _enhance_contrast(self, image, percentile_low=1, percentile_high=99):
        """
        Enhance contrast of 16-bit image using percentile stretching.
        This is especially useful for images with low intensity values (200-300 max).
        """
        # Calculate percentiles for non-zero pixels to avoid background
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
    
    def _center_crop(self, image, crop_size):
        """Extract a center crop of given size."""
        h, w = image.shape[:2]
        start_h = (h - crop_size[0]) // 2
        start_w = (w - crop_size[1]) // 2
        return image[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]
    
    def _random_crop(self, image, mask, crop_size):
        """Extract a random crop of given size that contains stress granules."""
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
                # Randomly select a stress granule pixel
                idx = np.random.randint(len(sg_coords[0]))
                center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                
                # Calculate crop boundaries ensuring we stay within image
                start_h = max(0, min(h - crop_h, center_y - crop_h//2))
                start_w = max(0, min(w - crop_w, center_x - crop_w//2))
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
        
        try:
            # Load 16-bit image using cv2.IMREAD_UNCHANGED to preserve bit depth
            image = cv2.imread(self.image_paths[img_idx], cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[img_idx]}")
            
            # Load 8-bit mask as grayscale
            mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_paths[img_idx]}")
            
            # Ensure image is single channel
            if len(image.shape) > 2:
                # If multi-channel, convert to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert to float32 for processing
            image = image.astype(np.float32)
            
            # Apply Gaussian blur before contrast enhancement
            if self.gaussian_sigma > 0:
                image = cv2.GaussianBlur(image, (0, 0), self.gaussian_sigma)
            
            # Normalize based on actual bit depth
            if image.max() > 255:  # 16-bit image
                # For 16-bit images with low values (200-300), direct normalization would be too dark
                if self.enhance_contrast:
                    image = self._enhance_contrast(image)
                else:
                    # Simple normalization for 16-bit
                    image = image / 65535.0
            else:
                # 8-bit image
                image = image / 255.0
            
            # Either random crop or resize
            if self.random_crop:
                image, mask = self._random_crop(image, mask, self.target_size)
            else:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Ensure mask is binary (0 or 1)
            mask = (mask > 127).astype(np.float32)
            
            # Convert to tensors
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            mask = torch.from_numpy(mask).unsqueeze(0)    # Add channel dimension
            
            if self.transform:
                # Apply same transform to both image and mask
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                image = self.transform(image)
                torch.manual_seed(seed)
                mask = self.transform(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading images at index {idx}: {e}")
            # Return a default image in case of error
            image = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            mask = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return image, mask

# U-Net Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        # Add residual connection if input and output channels match
        self.use_residual = in_channels == out_channels
        
    def forward(self, x):
        if self.use_residual:
            return self.conv(x) + x
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
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
                nn.Sequential(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(feature, momentum=0.1),
                    nn.ReLU(inplace=True)
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Final convolution with additional feature reduction
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]//2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, kernel_size=1)
        )
        
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

        # Encoder pathway
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder pathway
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:], 
                           interpolation=transforms.InterpolationMode.NEAREST)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = torch.sigmoid(self.final_conv(x))
        
        # Remove padding if added
        if padding != (0, 0, 0, 0):
            x = self.remove_padding(x, padding)
            
        return x

# Loss Functions
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

def bce_loss(pred, target):
    """Standard BCE loss without weighting"""
    # Clip predictions to avoid log(0)
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    
    # Calculate BCE loss
    loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    return loss.mean()

def combined_loss(pred, target, alpha=0.5):
    """
    Equal combination of BCE and Dice loss
    Args:
        pred: Model predictions
        target: Ground truth masks
        alpha: Weight for BCE loss (1-alpha is weight for Dice loss)
               Set to 0.5 for equal weighting
    """
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice 