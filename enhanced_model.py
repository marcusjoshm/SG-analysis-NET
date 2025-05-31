"""
Enhanced U-Net model architecture for stress granule detection
Includes spatial attention modules and deeper feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Spatial attention module to help the model focus on stress granule regions
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Generate attention map
        attention = self.conv(x)
        # Apply attention
        return x * attention


class DoubleConvWithAttention(nn.Module):
    """
    Double convolution block with residual connection and spatial attention
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvWithAttention, self).__init__()
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
        
        # Add spatial attention module
        self.attention = SpatialAttention(out_channels)
        
    def forward(self, x):
        conv_out = self.conv(x)
        attention_out = self.attention(conv_out)
        
        # Apply residual connection if applicable
        if self.use_residual:
            return attention_out + x
        return attention_out


class DilatedBottleneck(nn.Module):
    """
    Bottleneck with dilated convolutions to increase receptive field
    """
    def __init__(self, in_channels, out_channels):
        super(DilatedBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            # First dilated conv with dilation=2
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            # Second dilated conv with dilation=4
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        
        # Add spatial attention
        self.attention = SpatialAttention(out_channels)
        
    def forward(self, x):
        x = self.bottleneck(x)
        return self.attention(x)


class EnhancedUNet(nn.Module):
    """
    Enhanced U-Net architecture with:
    - Spatial attention modules
    - Deeper feature extraction
    - Dilated convolutions in the bottleneck
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024, 2048]):
        super(EnhancedUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of U-Net with attention
        for feature in features:
            self.downs.append(DoubleConvWithAttention(in_channels, feature))
            in_channels = feature
        
        # Bottleneck with dilated convolutions
        self.bottleneck = DilatedBottleneck(features[-1], features[-1]*2)
        
        # Up part of U-Net with attention
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(feature, momentum=0.1),
                    nn.ReLU(inplace=True)
                )
            )
            self.ups.append(DoubleConvWithAttention(feature*2, feature))
        
        # Final convolution with additional feature reduction
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]//2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, kernel_size=1)
        )
        
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


# Custom loss functions for stress granule segmentation
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

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    Focuses more on hard-to-classify examples
    """
    pred_sigmoid = torch.sigmoid(pred)
    
    # Calculate binary cross entropy
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Calculate focal weights
    p_t = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply weights to BCE loss
    focal = focal_weight * bce
    
    return focal.mean()

def boundary_loss(pred, target, weight=1.0):
    """
    Boundary-aware loss to improve edge detection
    Uses Sobel filters to detect edges in target mask
    """
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    
    # Apply Sobel filters to detect edges in target
    target_expanded = target.repeat(1, 3, 1, 1) if target.size(1) == 1 else target
    
    # Convert to grayscale if multichannel
    if target_expanded.size(1) > 1:
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1).to(pred.device)
        target_gray = (target_expanded * weights).sum(dim=1, keepdim=True)
    else:
        target_gray = target_expanded
    
    # Detect edges
    edges_x = F.conv2d(target_gray, sobel_x, padding=1)
    edges_y = F.conv2d(target_gray, sobel_y, padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    # Create boundary mask
    boundary_mask = (edges > 0.1).float()
    
    # Calculate boundary-aware BCE loss
    boundary_bce = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none'
    )
    
    # Apply higher weight to boundary regions
    weighted_bce = boundary_bce * (1 + weight * boundary_mask)
    
    return weighted_bce.mean()

def combined_enhanced_loss(pred, target, alpha_bce=0.3, alpha_dice=0.5, alpha_focal=0.1, alpha_boundary=0.1):
    """
    Enhanced combined loss function with boundary awareness and focal loss
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        alpha_bce: Weight for BCE loss
        alpha_dice: Weight for Dice loss
        alpha_focal: Weight for Focal loss
        alpha_boundary: Weight for Boundary loss
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    boundary = boundary_loss(pred, target)
    
    return (alpha_bce * bce + 
            alpha_dice * dice + 
            alpha_focal * focal + 
            alpha_boundary * boundary)
