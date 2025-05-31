"""
Improved U-Net model architecture for stress granule detection
Includes spatial dropout and better normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvWithDropout(nn.Module):
    """
    Double convolution block with spatial dropout and residual connection
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvWithDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.05),  # Lower momentum for more stable training
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Add spatial dropout to prevent overfitting
        )
        
        # Add residual connection if input and output channels match
        self.use_residual = in_channels == out_channels
        
        # Projection shortcut if channels don't match
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.05)
            )
        
    def forward(self, x):
        conv_out = self.conv(x)
        
        # Apply residual connection if applicable
        if self.use_residual:
            return conv_out + x
        else:
            return conv_out + self.shortcut(x)


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(ImprovedUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET with improved blocks
        for feature in features:
            self.downs.append(DoubleConvWithDropout(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET with improved blocks
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(feature, momentum=0.05),
                    nn.ReLU(inplace=True)
                )
            )
            self.ups.append(DoubleConvWithDropout(feature*2, feature))
        
        # Bottleneck with dropout
        self.bottleneck = nn.Sequential(
            DoubleConvWithDropout(features[-1], features[-1]*2),
            nn.Dropout2d(0.2)  # Higher dropout in bottleneck
        )
        
        # Final convolution with additional feature reduction
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]//2, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, kernel_size=1)
        )
        
    def pad_to_power_of_two(self, x):
        """Pad input to ensure dimensions are powers of 2 for U-Net."""
        h, w = x.shape[2], x.shape[3]
        
        # Find the next power of 2
        target_h = 2 ** (h - 1).bit_length()
        target_w = 2 ** (w - 1).bit_length()
        
        # Calculate padding
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        # Apply padding
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        padded_x = F.pad(x, padding, mode='reflect')
        
        return padded_x, padding
    
    def remove_padding(self, x, padding):
        """Remove padding added earlier."""
        if padding[3] > 0 or padding[1] > 0:
            h, w = x.shape[2], x.shape[3]
            return x[:, :, :h-padding[3], :w-padding[1]]
        return x
    
    def forward(self, x):
        # Apply padding to ensure dimensions are powers of 2
        original_h, original_w = x.shape[2], x.shape[3]
        x, padding = self.pad_to_power_of_two(x)
        
        # Store skip connections
        skip_connections = []
        
        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Up path with skip connections
        skip_connections = skip_connections[::-1]  # Reverse the list
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transposed conv
            skip_connection = skip_connections[idx//2]
            
            # Handle case where the dimensions don't match exactly
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
                
            # Concatenate with skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Double conv
        
        # Final convolution
        x = self.final_conv(x)
        
        # Remove padding to match original dimensions
        x = self.remove_padding(x, padding)
        
        return x


# Improved loss functions
def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for evaluation"""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_binary * target).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss function"""
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def improved_bce_loss(pred, target, device, pos_weight=5.0):
    """
    Improved BCE loss with class balancing to address imbalance
    Uses pos_weight to penalize false negatives more
    """
    weight_tensor = torch.tensor([pos_weight]).to(device)
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight_tensor)

def improved_combined_loss(pred, target, device, alpha=0.4, pos_weight=5.0, smooth=1e-6):
    """
    Improved combined loss with adjusted BCE weight and class balancing
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        device: Current device
        alpha: Weight for BCE loss (1-alpha is weight for Dice loss)
               Increased from 0.3 to 0.4 to penalize false positives more
        pos_weight: Positive class weight for BCE loss
    """
    bce = improved_bce_loss(pred, target, device, pos_weight)
    dice = dice_loss(pred, target, smooth)
    
    return alpha * bce + (1 - alpha) * dice
