#!/usr/bin/env python
"""
Robust training script for stress granule detection with GPU acceleration
Prioritizes model robustness and dataset diversity over training speed
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import random

# Import improved model and loss functions
from improved_model import ImprovedUNet, improved_combined_loss, dice_coefficient
from train_small_test import match_images_and_masks, analyze_matched_pairs
from models import StressGranule16bitDataset


class RobustDataset(Dataset):
    """Enhanced dataset with robust preprocessing and diverse augmentation strategies"""
    
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256), 
                 random_crop=True, crops_per_image=5, mode='train'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.random_crop = random_crop
        self.crops_per_image = crops_per_image if random_crop else 1
        self.mode = mode
        
        # Validate paths exist
        self.valid_indices = []
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_indices.append(i)
            else:
                print(f"Warning: Missing files at index {i}")
        
        print(f"Dataset initialized with {len(self.valid_indices)} valid image-mask pairs")
        
        # Calculate sample weights for balanced sampling
        self.sample_weights = self._calculate_sample_weights()
    
    def _calculate_sample_weights(self):
        """Calculate weights for each sample based on stress granule content"""
        weights = []
        for idx in self.valid_indices:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Higher weight for images with more stress granules
                sg_ratio = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
                # Use inverse frequency weighting with smoothing
                weight = 1.0 / (sg_ratio + 0.01)  # Add small constant to avoid division by zero
                weights.append(weight)
            else:
                weights.append(1.0)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        return weights
    
    def __len__(self):
        return len(self.valid_indices) * self.crops_per_image
    
    def _multi_scale_enhancement(self, image):
        """Apply multi-scale contrast enhancement"""
        # Apply CLAHE at multiple scales
        enhanced_scales = []
        
        for clip_limit in [2.0, 4.0, 8.0]:
            for grid_size in [(4, 4), (8, 8), (16, 16)]:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
                # Convert to uint16 for CLAHE
                img_uint16 = (image * 65535).astype(np.uint16)
                enhanced = clahe.apply(img_uint16)
                enhanced = enhanced.astype(np.float32) / 65535.0
                enhanced_scales.append(enhanced)
        
        # Combine multi-scale enhancements
        enhanced_combined = np.mean(enhanced_scales, axis=0)
        
        # Blend with original
        alpha = 0.7  # Weight for enhanced image
        result = alpha * enhanced_combined + (1 - alpha) * image
        
        return result
    
    def _adaptive_preprocessing(self, image):
        """Adaptive preprocessing based on image statistics"""
        # Calculate image statistics
        mean_val = np.mean(image[image > 0])
        std_val = np.std(image[image > 0]) if np.sum(image > 0) > 0 else 1.0
        
        # Adaptive normalization
        if std_val < 0.1:  # Low contrast image
            # Apply stronger enhancement
            image = self._enhance_contrast(image, percentile_low=0.1, percentile_high=99.9)
        else:
            # Standard enhancement
            image = self._enhance_contrast(image, percentile_low=1.0, percentile_high=99.0)
        
        # Apply multi-scale enhancement
        image = self._multi_scale_enhancement(image)
        
        # Edge enhancement for better boundary detection
        if self.mode == 'train' and np.random.random() < 0.3:
            # Randomly apply edge enhancement during training
            laplacian = cv2.Laplacian(image, cv2.CV_32F)
            image = image - 0.1 * laplacian
            image = np.clip(image, 0, 1)
        
        return image
    
    def _enhance_contrast(self, image, percentile_low=1.0, percentile_high=99.0):
        """Enhanced contrast with adaptive percentile stretching"""
        non_zero_pixels = image[image > 0]
        if len(non_zero_pixels) > 0:
            p_low = np.percentile(non_zero_pixels, percentile_low)
            p_high = np.percentile(non_zero_pixels, percentile_high)
        else:
            p_low = 0
            p_high = 1
        
        if p_high > p_low:
            image_clipped = np.clip(image, p_low, p_high)
            image_rescaled = (image_clipped - p_low) / (p_high - p_low)
        else:
            image_rescaled = np.zeros_like(image, dtype=np.float32)
        
        return image_rescaled
    
    def _diverse_crop(self, image, mask, crop_size):
        """Extract diverse crops with various strategies"""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        # Ensure crop size isn't larger than image
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # Strategy selection based on mode and randomness
        strategies = ['center_sg', 'edge_sg', 'background', 'mixed', 'random']
        
        if self.mode == 'train':
            # Training: Use all strategies with probabilities
            strategy_probs = [0.3, 0.2, 0.15, 0.2, 0.15]
            strategy = np.random.choice(strategies, p=strategy_probs)
        else:
            # Validation: More deterministic
            strategy = 'mixed'
        
        if mask is not None and strategy != 'random':
            sg_coords = np.where(mask > 127)
            
            if len(sg_coords[0]) > 0 and strategy in ['center_sg', 'edge_sg', 'mixed']:
                if strategy == 'center_sg':
                    # Focus on stress granule centers
                    idx = np.random.randint(len(sg_coords[0]))
                    center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                    
                elif strategy == 'edge_sg':
                    # Focus on stress granule edges
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate((mask > 127).astype(np.uint8), kernel, iterations=2)
                    eroded = cv2.erode((mask > 127).astype(np.uint8), kernel, iterations=2)
                    edges = dilated - eroded
                    edge_coords = np.where(edges > 0)
                    
                    if len(edge_coords[0]) > 0:
                        idx = np.random.randint(len(edge_coords[0]))
                        center_y, center_x = edge_coords[0][idx], edge_coords[1][idx]
                    else:
                        # Fallback to center
                        idx = np.random.randint(len(sg_coords[0]))
                        center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                
                elif strategy == 'mixed':
                    # 50-50 mix of stress granule and background
                    if np.random.random() < 0.5:
                        idx = np.random.randint(len(sg_coords[0]))
                        center_y, center_x = sg_coords[0][idx], sg_coords[1][idx]
                    else:
                        # Random background location
                        bg_mask = mask <= 127
                        bg_coords = np.where(bg_mask)
                        if len(bg_coords[0]) > 0:
                            idx = np.random.randint(len(bg_coords[0]))
                            center_y, center_x = bg_coords[0][idx], bg_coords[1][idx]
                        else:
                            # Fallback to random
                            center_y = np.random.randint(crop_h//2, h - crop_h//2)
                            center_x = np.random.randint(crop_w//2, w - crop_w//2)
                
                # Add random offset
                offset_range = crop_h // 3
                offset_y = np.random.randint(-offset_range, offset_range)
                offset_x = np.random.randint(-offset_range, offset_range)
                center_y = np.clip(center_y + offset_y, crop_h//2, h - crop_h//2)
                center_x = np.clip(center_x + offset_x, crop_w//2, w - crop_w//2)
                
                start_h = max(0, min(h - crop_h, center_y - crop_h//2))
                start_w = max(0, min(w - crop_w, center_x - crop_w//2))
            
            elif strategy == 'background':
                # Focus on background regions
                bg_mask = mask <= 127
                bg_coords = np.where(bg_mask)
                if len(bg_coords[0]) > 0:
                    idx = np.random.randint(len(bg_coords[0]))
                    center_y, center_x = bg_coords[0][idx], bg_coords[1][idx]
                    start_h = max(0, min(h - crop_h, center_y - crop_h//2))
                    start_w = max(0, min(w - crop_w, center_x - crop_w//2))
                else:
                    # Fallback to random
                    start_h = np.random.randint(0, max(1, h - crop_h + 1))
                    start_w = np.random.randint(0, max(1, w - crop_w + 1))
            else:
                # Random crop
                start_h = np.random.randint(0, max(1, h - crop_h + 1))
                start_w = np.random.randint(0, max(1, w - crop_w + 1))
        else:
            # Random crop for validation or if mask is None
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
        
        # Use valid indices
        actual_idx = self.valid_indices[img_idx % len(self.valid_indices)]
        
        try:
            # Load 16-bit image
            image = cv2.imread(self.image_paths[actual_idx], cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[actual_idx]}")
            
            # Load 8-bit mask
            mask = cv2.imread(self.mask_paths[actual_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_paths[actual_idx]}")
            
            # Ensure image is single channel
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32) / 65535.0
            
            # Apply adaptive preprocessing
            image = self._adaptive_preprocessing(image)
            
            # Extract diverse crop
            if self.random_crop:
                image, mask = self._diverse_crop(image, mask, self.target_size)
            else:
                # Resize directly if not using random crops
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Apply data augmentation using albumentations for more diverse transforms
            if self.transform and self.mode == 'train':
                # Convert to uint8 for albumentations
                image_uint8 = (image * 255).astype(np.uint8)
                
                # Apply transformations
                transformed = self.transform(image=image_uint8, mask=mask)
                image = transformed['image'].float() / 255.0
                mask = transformed['mask'].float()
                
                # Ensure correct dimensions
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                elif len(mask.shape) == 3 and mask.shape[0] > 1:
                    # If mask has multiple channels, take only the first
                    mask = mask[0:1]
            else:
                # Simple conversion to tensor for validation
                image = torch.from_numpy(image).float().unsqueeze(0)
                mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            
            # Ensure mask is binary
            mask = (mask > 0.5).float()
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading image/mask at index {actual_idx}: {e}")
            # Return a default image in case of error
            image = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            mask = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return image, mask


def create_robust_augmentations():
    """Create comprehensive data augmentation pipeline using albumentations"""
    return A.Compose([
        # Spatial transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        
        # Elastic and grid distortions for robustness
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.1, p=1.0),
        ], p=0.3),
        
        # Intensity transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        
        # Noise for robustness
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),
        
        # Blur for robustness to different image qualities
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        
        # Cutout/Coarse dropout for robustness to occlusions
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                       min_holes=1, min_height=8, min_width=8,
                       fill_value=0, mask_fill_value=0, p=0.3),
        
        # Convert to tensor
        ToTensorV2()
    ])


def focal_tversky_loss(outputs, targets, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Focal Tversky Loss - handles class imbalance better than standard losses
    alpha: controls false negatives
    beta: controls false positives
    gamma: focal parameter
    """
    smooth = 1e-6
    
    # Apply sigmoid to outputs
    outputs = torch.sigmoid(outputs)
    
    # Flatten the tensors
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    
    # True positives, false positives, false negatives
    tp = (outputs * targets).sum()
    fp = ((1 - targets) * outputs).sum()
    fn = (targets * (1 - outputs)).sum()
    
    # Tversky index
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    
    # Focal Tversky loss
    focal_tversky = (1 - tversky_index) ** gamma
    
    return focal_tversky


def robust_combined_loss(outputs, targets, device, alpha=0.3, beta=0.3, gamma=0.4):
    """
    Robust combined loss using multiple loss functions
    alpha: weight for BCE
    beta: weight for Focal Tversky
    gamma: weight for Dice
    """
    # Weighted BCE loss
    pos_weight = torch.tensor([10.0]).to(device)  # Higher weight for positive class
    bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, pos_weight=pos_weight)
    
    # Focal Tversky loss
    ftl = focal_tversky_loss(outputs, targets, alpha=0.7, beta=0.3, gamma=0.75)
    
    # Dice loss
    dice_loss = 1 - dice_coefficient(outputs, targets)
    
    # Combine losses
    total_loss = alpha * bce_loss + beta * ftl + gamma * dice_loss
    
    return total_loss


def robust_train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-4,
                      patience=30, device='mps', metrics_dir='metrics/robust_gpu',
                      checkpoint_interval=10):
    """
    Robust training function with extensive monitoring and checkpointing
    """
    model = model.to(device)
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training for efficiency
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Training tracking variables
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    best_val_dice = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Create metrics directory
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'patience': patience,
        'device': str(device),
        'checkpoint_interval': checkpoint_interval,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts',
        'loss_function': 'robust_combined_loss'
    }
    
    with open(os.path.join(metrics_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Start training
    print("\n🚀 Starting robust training...")
    print(f"Training for {num_epochs} epochs with patience {patience}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loop):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Mixed precision training
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                # Forward pass
                output = model(data)
                
                # Calculate robust loss
                loss = robust_combined_loss(output, target, device)
            
            # Backward pass
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            with torch.no_grad():
                dice = dice_coefficient(output, target)
                train_dice += dice.item()
            
            train_loop.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check if we have a new best model (using F1 score as primary metric)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(metrics_dir, 'best_model.pth'))
            
            print(f"  ✅ New best model saved! (F1: {best_val_f1:.4f}, Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement for {patience_counter} epochs")
        
        # Regular checkpoints
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(metrics_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Save visualizations periodically
        if (epoch + 1) % 20 == 0 or epoch == 0:
            visualize_predictions(model, val_loader, device, 
                                os.path.join(metrics_dir, f'predictions_epoch_{epoch+1}.png'))
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n✅ Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation F1 score: {best_val_f1:.4f} (Dice: {best_val_dice:.4f}) at epoch {best_epoch+1}")
    
    # Save final history
    with open(os.path.join(metrics_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot comprehensive training history
    plot_robust_training_history(history, metrics_dir)
    
    # Generate final evaluation report
    generate_evaluation_report(model, val_loader, device, metrics_dir)
    
    return best_val_dice, best_val_f1, model


def evaluate_model(model, data_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    
    total_loss = 0.0
    total_dice = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = robust_combined_loss(output, target, device)
            
            # Calculate Dice coefficient
            dice = dice_coefficient(output, target)
            
            # Update metrics
            total_loss += loss.item()
            total_dice += dice.item()
            
            # Store predictions and targets
            pred_binary = (torch.sigmoid(output) > 0.5).float()
            all_predictions.append(pred_binary.cpu())
            all_targets.append(target.cpu())
    
    # Calculate average metrics
    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)
    
    # Calculate precision, recall, and F1
    all_pred = torch.cat(all_predictions, dim=0)
    all_targ = torch.cat(all_targets, dim=0)
    
    tp = (all_pred * all_targ).sum().item()
    fp = (all_pred * (1 - all_targ)).sum().item()
    fn = ((1 - all_pred) * all_targ).sum().item()
    tn = ((1 - all_pred) * (1 - all_targ)).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    
    metrics = {
        'loss': avg_loss,
        'dice': avg_dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity
    }
    
    return metrics


def plot_robust_training_history(history, metrics_dir):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot losses
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Dice scores
    axes[0, 1].plot(history['val_dice'], label='Val Dice', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].set_title('Validation Dice Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Precision and Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', color='orange')
    axes[1, 0].plot(history['val_recall'], label='Recall', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].set_title('Validation Precision and Recall')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot F1 Score
    axes[1, 1].plot(history['val_f1'], label='F1 Score', color='brown')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot Learning Rate
    axes[2, 0].plot(history['learning_rates'], label='Learning Rate', color='magenta')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Learning Rate')
    axes[2, 0].set_yscale('log')
    axes[2, 0].legend()
    axes[2, 0].set_title('Learning Rate Schedule')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot all metrics together (normalized)
    axes[2, 1].plot(np.array(history['val_dice']), label='Dice', color='green')
    axes[2, 1].plot(np.array(history['val_precision']), label='Precision', color='orange')
    axes[2, 1].plot(np.array(history['val_recall']), label='Recall', color='purple')
    axes[2, 1].plot(np.array(history['val_f1']), label='F1', color='brown')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Score')
    axes[2, 1].legend()
    axes[2, 1].set_title('All Validation Metrics')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'training_history_comprehensive.png'), dpi=150)
    plt.close()


def generate_evaluation_report(model, val_loader, device, metrics_dir):
    """Generate comprehensive evaluation report with visualizations"""
    print("\n📊 Generating comprehensive evaluation report...")
    
    # Get detailed metrics
    metrics = evaluate_model(model, val_loader, device)
    
    # Create evaluation report
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics,
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Save report as JSON
    with open(os.path.join(metrics_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate detailed visualizations
    visualize_detailed_predictions(model, val_loader, device, metrics_dir)
    
    # Print summary
    print("\n📈 Final Evaluation Metrics:")
    print(f"  Dice Score: {metrics['dice']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")


def visualize_predictions(model, val_loader, device, output_file):
    """Generate visualization of model predictions"""
    model.eval()
    
    # Get a batch of validation data
    data_iter = iter(val_loader)
    images, masks = next(data_iter)
    
    # Limit to 6 samples for visualization
    images = images[:min(6, len(images))]
    masks = masks[:min(6, len(masks))]
    
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
        im = axes[i, 2].imshow(probs[i, 0], cmap='magma', vmin=0, vmax=1)
        axes[i, 2].set_title('Probability Map')
        axes[i, 2].axis('off')
        
        # Binary prediction with overlay
        axes[i, 3].imshow(images[i, 0], cmap='gray', alpha=0.7)
        axes[i, 3].imshow(preds[i, 0], cmap='Reds', alpha=0.3)
        
        # Calculate metrics for this sample
        dice = dice_coefficient(outputs[i:i+1], masks[i:i+1].to(device)).item()
        axes[i, 3].set_title(f'Prediction (Dice: {dice:.3f})')
        axes[i, 3].axis('off')
    
    # Use subplots_adjust instead of tight_layout when adding colorbar
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
    
    # Add colorbar in the adjusted space
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_detailed_predictions(model, val_loader, device, metrics_dir):
    """Generate detailed visualizations for evaluation"""
    model.eval()
    
    # Collect predictions for multiple batches
    all_images = []
    all_masks = []
    all_preds = []
    all_probs = []
    
    num_batches = min(5, len(val_loader))  # Limit to 5 batches
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
    
    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)[:20]  # Limit to 20 samples
    all_masks = torch.cat(all_masks, dim=0)[:20]
    all_probs = torch.cat(all_probs, dim=0)[:20]
    all_preds = torch.cat(all_preds, dim=0)[:20]
    
    # Create detailed visualization
    n_samples = min(20, len(all_images))
    n_cols = 5
    n_rows = n_samples // n_cols + (1 if n_samples % n_cols > 0 else 0)
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(20, 8 * n_rows))
    
    for i in range(n_samples):
        row = (i // n_cols) * 2
        col = i % n_cols
        
        # Input image with ground truth overlay
        axes[row, col].imshow(all_images[i, 0], cmap='gray')
        axes[row, col].imshow(all_masks[i, 0], cmap='Greens', alpha=0.3)
        axes[row, col].set_title(f'Input + GT #{i+1}')
        axes[row, col].axis('off')
        
        # Prediction with probability overlay
        axes[row + 1, col].imshow(all_images[i, 0], cmap='gray')
        axes[row + 1, col].imshow(all_probs[i, 0], cmap='hot', alpha=0.5, vmin=0, vmax=1)
        
        # Calculate dice for this sample
        dice = dice_coefficient(
            torch.sigmoid(model(all_images[i:i+1].to(device))),
            all_masks[i:i+1].to(device)
        ).item()
        
        axes[row + 1, col].set_title(f'Pred (D: {dice:.3f})')
        axes[row + 1, col].axis('off')
    
    # Remove empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = (i // n_cols) * 2
        col = i % n_cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'detailed_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run robust training"""
    # Configuration
    IMAGE_SIZE = 512  # Increased from 256 to capture more detail
    BATCH_SIZE = 2  # Reduced from 4 due to larger image size
    NUM_EPOCHS = 200  # More epochs for robustness
    LEARNING_RATE = 1e-4
    PATIENCE = 30  # More patience for robust training
    CROPS_PER_IMAGE = 5  # More crops per image for diversity
    CHECKPOINT_INTERVAL = 10
    
    print("=" * 80)
    print("🔧 ROBUST GPU Training for Stress Granule Detection")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Patience: {PATIENCE}")
    print(f"  - Crops per image: {CROPS_PER_IMAGE}")
    print(f"  - System: {platform.system()} {platform.machine()}")
    print("=" * 80)
    
    # Check for GPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using Apple Silicon GPU (MPS): {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using NVIDIA GPU: {device}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"⚠️ Using CPU: {device}")
        print("   Warning: Training will be significantly slower without GPU")
    
    # Match all available images and masks
    print("\n📁 Loading dataset...")
    matched_pairs = match_images_and_masks('data/images', 'data/masks', max_samples=1000)  # Use large number to get all available
    
    if len(matched_pairs) < 4:
        print("❌ Error: Need at least 4 image-mask pairs")
        return
    
    # Analyze the pairs
    analyze_matched_pairs(matched_pairs)
    
    # Use all data but with smart train/val split
    # Shuffle pairs first for randomness
    random.shuffle(matched_pairs)
    
    # Split: 80/20 for robust training
    train_size = int(0.8 * len(matched_pairs))
    train_pairs = matched_pairs[:train_size]
    val_pairs = matched_pairs[train_size:]
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_images)} images × {CROPS_PER_IMAGE} crops = {len(train_images) * CROPS_PER_IMAGE} samples")
    print(f"  Validation: {len(val_images)} images × {CROPS_PER_IMAGE} crops = {len(val_images) * CROPS_PER_IMAGE} samples")
    
    # Create robust augmentations
    train_transform = create_robust_augmentations()
    
    # Create datasets with robust preprocessing
    print("\n🔄 Creating datasets with robust preprocessing...")
    train_dataset = RobustDataset(
        train_images, train_masks,
        transform=train_transform,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        random_crop=True,
        crops_per_image=CROPS_PER_IMAGE,
        mode='train'
    )
    
    val_dataset = RobustDataset(
        val_images, val_masks,
        transform=None,  # No augmentation for validation
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        random_crop=True,
        crops_per_image=CROPS_PER_IMAGE,
        mode='val'
    )
    
    # Create data loaders with balanced sampling
    use_pin_memory = device.type == 'cuda'
    
    # Create weighted sampler for balanced training
    train_sampler = WeightedRandomSampler(
        weights=np.repeat(train_dataset.sample_weights, CROPS_PER_IMAGE),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,  # More workers for data loading
        pin_memory=use_pin_memory,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
        persistent_workers=True
    )
    
    # Initialize model
    print("\n🏗️ Initializing model...")
    model = ImprovedUNet(in_channels=1, out_channels=1)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Set up metrics directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f'metrics/robust_gpu_{timestamp}'
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"📁 Metrics will be saved to: {metrics_dir}")
    
    # Test data loading and preprocessing
    print("\n🧪 Testing data pipeline...")
    try:
        sample_batch = next(iter(train_loader))
        sample_imgs, sample_masks = sample_batch
        print(f"  ✅ Batch loaded successfully")
        print(f"     Image batch shape: {sample_imgs.shape}")
        print(f"     Mask batch shape: {sample_masks.shape}")
        print(f"     Image range: [{sample_imgs.min():.3f}, {sample_imgs.max():.3f}]")
        
        # Visualize preprocessing effects
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(min(4, len(sample_imgs))):
            axes[0, i].imshow(sample_imgs[i, 0].numpy(), cmap='gray')
            axes[0, i].set_title(f'Preprocessed Image {i+1}')
            axes[0, i].axis('off')
            
            # Handle mask shape - it might not have channel dimension
            if len(sample_masks.shape) == 3:  # [batch, height, width]
                axes[1, i].imshow(sample_masks[i].numpy(), cmap='gray')
            else:  # [batch, channels, height, width]
                axes[1, i].imshow(sample_masks[i, 0].numpy(), cmap='gray')
            axes[1, i].set_title(f'Mask {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'preprocessing_samples.png'))
        plt.close()
        print("  ✅ Preprocessing visualization saved")
        
    except Exception as e:
        print(f"❌ Error in data pipeline: {e}")
        return
    
    # Train model with robust settings
    print(f"\n🚀 Starting robust GPU training with {NUM_EPOCHS} epochs")
    print("   This training prioritizes robustness over speed...")
    start_time = time.time()
    
    best_dice, best_f1, model = robust_train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        device=device,
        metrics_dir=metrics_dir,
        checkpoint_interval=CHECKPOINT_INTERVAL
    )
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n✅ Robust training complete!")
    print(f"   Best Dice score: {best_dice:.4f}")
    print(f"   Best F1 score: {best_f1:.4f}")
    print(f"   Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save final summary
    summary = {
        "training_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "best_dice_score": float(best_dice),
        "best_f1_score": float(best_f1),
        "total_training_time": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "configuration": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "crops_per_image": CROPS_PER_IMAGE,
            "train_samples": len(train_images),
            "val_samples": len(val_images)
        }
    }
    
    with open(os.path.join(metrics_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n📊 All results saved in: {metrics_dir}/")
    print("   - best_model.pth: Best model checkpoint")
    print("   - training_history.json: Complete training history")
    print("   - evaluation_report.json: Detailed evaluation metrics")
    print("   - Various visualizations and plots")
    
    # Copy best model to root directory for easy access
    import shutil
    shutil.copy(
        os.path.join(metrics_dir, 'best_model.pth'),
        'robust_stress_granule_model.pth'
    )
    print("\n✅ Best model also saved as: robust_stress_granule_model.pth")


if __name__ == "__main__":
    main() 