"""
Enhanced preprocessing module for stress granule detection
Includes CLAHE preprocessing and model architecture improvements
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CLAHEPreprocessor:
    """
    Adaptive histogram equalization (CLAHE) for 16-bit microscopy images
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def process_16bit(self, image):
        """
        Apply CLAHE to 16-bit images by scaling, processing, and rescaling
        
        Args:
            image: 16-bit numpy array
        
        Returns:
            Enhanced image as float32 (0-1 range)
        """
        # Convert to float32 for processing
        img_float = image.astype(np.float32)
        
        # Get non-zero values to avoid background influence
        non_zero = img_float[img_float > 0]
        if len(non_zero) == 0:
            # No foreground pixels, return normalized image
            return img_float / np.max(img_float) if np.max(img_float) > 0 else img_float
            
        # Calculate percentiles for better contrast
        p_low = np.percentile(non_zero, 0.5)
        p_high = np.percentile(non_zero, 99.5)
        
        # Initial contrast stretch
        img_stretched = np.clip((img_float - p_low) / (p_high - p_low), 0, 1)
        
        # Convert to 8-bit for CLAHE
        img_8bit = (img_stretched * 255).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        # Apply CLAHE
        img_clahe = clahe.apply(img_8bit)
        
        # Convert back to float32 (0-1 range)
        img_enhanced = img_clahe.astype(np.float32) / 255.0
        
        return img_enhanced

class EnhancedStressGranuleDataset(Dataset):
    """
    Enhanced dataset class with CLAHE preprocessing
    """
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(512, 512), 
                 random_crop=True, crops_per_image=3, clahe_clip_limit=2.0,
                 clahe_grid_size=(8, 8), gaussian_sigma=1.7):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.random_crop = random_crop
        self.crops_per_image = crops_per_image if random_crop else 1
        self.gaussian_sigma = gaussian_sigma
        
        # Initialize CLAHE preprocessor
        self.clahe = CLAHEPreprocessor(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_grid_size
        )
        
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
        return len(self.image_paths) * self.crops_per_image
    
    def _apply_gaussian_blur(self, image):
        """Apply Gaussian blur to reduce noise"""
        # Calculate kernel size based on sigma (6*sigma + 1, must be odd)
        kernel_size = int(6 * self.gaussian_sigma + 1)
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        
        # Apply Gaussian blur
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), self.gaussian_sigma)
    
    def _random_crop(self, image, mask, crop_size):
        """Extract a random crop that contains stress granules if possible"""
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
                # With 70% probability, crop around a stress granule
                if np.random.random() < 0.7:
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
            
            # Apply CLAHE preprocessing
            image = self.clahe.process_16bit(image)
            
            # Apply Gaussian blur
            image = self._apply_gaussian_blur(image)
            
            # Random or center crop
            if self.random_crop:
                image, mask = self._random_crop(image, mask, self.target_size)
            else:
                # Resize directly if not using random crops
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize mask to binary
            mask = (mask > 127).astype(np.float32)
            
            # Convert to tensors
            image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
            mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension
            
            # Apply same transform to both image and mask
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

# Enhanced augmentations for stress granule detection
def create_advanced_augmentations():
    """Create advanced data augmentation pipeline"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ], p=0.3),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),
    ])
