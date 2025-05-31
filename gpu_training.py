#!/usr/bin/env python
"""
GPU-optimized training script for stress granule detection
Incorporates all improvements:
1. CLAHE preprocessing for better contrast in 16-bit images
2. Enhanced U-Net with spatial attention modules
3. Advanced loss functions (focal + boundary-aware)
4. GPU-optimized training strategy
5. K-fold cross-validation
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
import time
import argparse
from datetime import datetime
import pandas as pd

# Import our enhanced modules
from enhanced_preprocessing import EnhancedStressGranuleDataset, create_advanced_augmentations
from enhanced_model import EnhancedUNet, combined_enhanced_loss, dice_coefficient
from train_small_test import match_images_and_masks, analyze_matched_pairs

# Cross-validation and metrics tracking
class KFoldTrainer:
    """K-fold cross-validation trainer for stress granule detection"""
    
    def __init__(self, 
                 image_paths, 
                 mask_paths, 
                 n_splits=4,
                 num_epochs=50,
                 batch_size=4,
                 learning_rate=1e-4,
                 image_size=512,
                 crops_per_image=4,
                 patience=10,
                 metrics_dir='metrics/gpu_training',
                 experiment_name='stress_granule_cv',
                 device=None):
        """
        Initialize K-fold cross-validation trainer
        
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks
            n_splits: Number of folds for cross-validation
            num_epochs: Maximum number of epochs to train each fold
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            image_size: Image size for training (square)
            crops_per_image: Number of random crops per image
            patience: Early stopping patience
            metrics_dir: Directory to save metrics
            experiment_name: Name of the experiment
            device: Device to use for training (cuda or cpu)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.crops_per_image = crops_per_image
        self.patience = patience
        self.metrics_dir = metrics_dir
        self.experiment_name = experiment_name
        
        # Ensure metrics directory exists
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize K-fold cross-validation
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize results tracking
        self.fold_results = []
    
    def create_datasets(self, train_indices, val_indices):
        """Create training and validation datasets for a fold"""
        
        # Create data augmentation transforms
        train_transform = create_advanced_augmentations()
        
        # Select image and mask paths for this fold
        train_imgs = [self.image_paths[i] for i in train_indices]
        train_masks = [self.mask_paths[i] for i in train_indices]
        val_imgs = [self.image_paths[i] for i in val_indices]
        val_masks = [self.mask_paths[i] for i in val_indices]
        
        # Create datasets
        train_dataset = EnhancedStressGranuleDataset(
            train_imgs, train_masks,
            transform=train_transform,
            target_size=(self.image_size, self.image_size),
            random_crop=True,
            crops_per_image=self.crops_per_image,
            clahe_clip_limit=2.0,
            clahe_grid_size=(8, 8),
            gaussian_sigma=1.7
        )
        
        val_dataset = EnhancedStressGranuleDataset(
            val_imgs, val_masks,
            transform=None,  # No augmentation for validation
            target_size=(self.image_size, self.image_size),
            random_crop=True,  # Still use random crops for validation
            crops_per_image=2,  # Fewer crops for validation
            clahe_clip_limit=2.0,
            clahe_grid_size=(8, 8),
            gaussian_sigma=1.7
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,  # Adjust based on CPU cores
            pin_memory=True  # Speeds up CPU to GPU transfers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    def train_fold(self, fold, train_loader, val_loader):
        """Train a single fold"""
        
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}/{self.n_splits}")
        print(f"{'='*50}")
        
        # Initialize model
        model = EnhancedUNet(in_channels=1, out_channels=1)
        model = model.to(self.device)
        
        # Initialize optimizer with weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler with warm-up
        def get_lr(epoch):
            # Linear warm-up for first 5 epochs
            if epoch < 5:
                return self.learning_rate * (epoch + 1) / 5
            # After warm-up, use cosine annealing
            return self.learning_rate * 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (self.num_epochs - 5)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
        
        # Initialize tracking variables
        train_losses = []
        val_losses = []
        val_dice_scores = []
        best_val_loss = float('inf')
        best_val_dice = 0
        best_epoch = 0
        patience_counter = 0
        
        # Create fold directory for metrics
        fold_dir = os.path.join(self.metrics_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            for batch_idx, (data, target) in enumerate(train_loop):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                
                # Calculate loss
                loss = combined_enhanced_loss(
                    output, target,
                    alpha_bce=0.3,  # 30% BCE
                    alpha_dice=0.5,  # 50% Dice
                    alpha_focal=0.1,  # 10% Focal
                    alpha_boundary=0.1  # 10% Boundary
                )
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_dice = 0
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
                
                all_val_dices = []  # Store dice scores for each batch
                
                for data, target in val_loop:
                    # Move data to device
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    output = model(data)
                    
                    # Calculate loss
                    loss = combined_enhanced_loss(
                        output, target,
                        alpha_bce=0.3,
                        alpha_dice=0.5,
                        alpha_focal=0.1,
                        alpha_boundary=0.1
                    )
                    
                    # Calculate Dice coefficient
                    dice = dice_coefficient(output, target)
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_dice += dice.item()
                    all_val_dices.append(dice.item())
                    
                    val_loop.set_postfix(loss=loss.item(), dice=dice.item())
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)
            val_losses.append(avg_val_loss)
            val_dice_scores.append(avg_val_dice)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Dice: {avg_val_dice:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save metrics for this epoch
            metrics_df = pd.DataFrame({
                'epoch': [epoch+1],
                'train_loss': [avg_train_loss],
                'val_loss': [avg_val_loss],
                'val_dice': [avg_val_dice],
                'lr': [current_lr]
            })
            
            metrics_path = os.path.join(fold_dir, 'metrics.csv')
            if epoch == 0:
                metrics_df.to_csv(metrics_path, index=False)
            else:
                metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
            
            # Check if we have a new best model
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_dice': avg_val_dice
                }, os.path.join(fold_dir, 'best_model.pth'))
                
                print(f"  ✅ New best model saved! (Dice: {best_val_dice:.4f})")
            else:
                patience_counter += 1
                print(f"  ⚠️ No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed for this fold
        total_time = time.time() - start_time
        print(f"\n✅ Fold {fold+1} completed in {total_time:.2f} seconds")
        print(f"Best validation Dice score: {best_val_dice:.4f} (epoch {best_epoch+1})")
        
        # Plot training history for this fold
        self.plot_fold_history(fold, train_losses, val_losses, val_dice_scores, fold_dir)
        
        # Save fold results
        fold_result = {
            'fold': fold+1,
            'best_val_dice': best_val_dice,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch+1,
            'total_epochs': len(train_losses),
            'training_time': total_time
        }
        
        self.fold_results.append(fold_result)
        
        # Return best model
        return model, best_val_dice, fold_dir
    
    def plot_fold_history(self, fold, train_losses, val_losses, val_dice_scores, fold_dir):
        """Plot training history for a fold"""
        
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} - Training and Validation Loss')
        
        # Plot Dice scores
        plt.subplot(1, 2, 2)
        plt.plot(val_dice_scores, label='Val Dice', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.title(f'Fold {fold+1} - Validation Dice Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, 'training_history.png'))
        plt.close()
    
    def run_cross_validation(self):
        """Run K-fold cross-validation"""
        
        print(f"\n{'='*60}")
        print(f"Running {self.n_splits}-fold Cross-Validation")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Crops per image: {self.crops_per_image}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        # Generate fold indices
        indices = list(range(len(self.image_paths)))
        
        # Track best model across all folds
        best_overall_dice = 0
        best_fold = -1
        best_model_path = None
        
        # Summarize dataset
        print(f"\nDataset:")
        print(f"  Total images: {len(self.image_paths)}")
        print(f"  Folds: {self.n_splits}")
        print(f"  Images per fold (validation): ~{len(self.image_paths) // self.n_splits}")
        
        # Run training for each fold
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
            # Create datasets and loaders for this fold
            train_loader, val_loader, train_dataset, val_dataset = self.create_datasets(train_idx, val_idx)
            
            print(f"\nFold {fold+1} dataset:")
            print(f"  Training images: {len(train_idx)}")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation images: {len(val_idx)}")
            print(f"  Validation samples: {len(val_dataset)}")
            
            # Train this fold
            model, best_dice, fold_dir = self.train_fold(fold, train_loader, val_loader)
            
            # Generate visualizations for this fold
            self.generate_fold_visualizations(fold, model, val_dataset, fold_dir)
            
            # Check if this is the best fold
            if best_dice > best_overall_dice:
                best_overall_dice = best_dice
                best_fold = fold
                best_model_path = os.path.join(fold_dir, 'best_model.pth')
        
        # Summarize cross-validation results
        self.summarize_cv_results(best_fold, best_overall_dice, best_model_path)
        
        return best_model_path
    
    def generate_fold_visualizations(self, fold, model, val_dataset, fold_dir):
        """Generate prediction visualizations for a fold"""
        
        print(f"\nGenerating visualizations for fold {fold+1}...")
        
        # Move model to evaluation mode
        model.eval()
        
        # Choose a few samples for visualization
        num_samples = min(4, len(val_dataset))
        indices = np.random.choice(len(val_dataset), num_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                # Get image and mask
                image, mask = val_dataset[idx]
                
                # Move to device
                image = image.unsqueeze(0).to(self.device)
                
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
        plt.savefig(os.path.join(fold_dir, 'predictions.png'))
        plt.close()
    
    def summarize_cv_results(self, best_fold, best_overall_dice, best_model_path):
        """Summarize cross-validation results"""
        
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results Summary")
        print(f"{'='*60}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.fold_results)
        
        # Calculate average metrics
        avg_dice = results_df['best_val_dice'].mean()
        std_dice = results_df['best_val_dice'].std()
        
        # Print summary
        print(f"\nResults per fold:")
        for fold in range(self.n_splits):
            fold_result = results_df[results_df['fold'] == fold+1].iloc[0]
            print(f"  Fold {fold+1}: Dice = {fold_result['best_val_dice']:.4f}, "
                  f"Epoch {fold_result['best_epoch']}/{fold_result['total_epochs']}")
        
        print(f"\nOverall results:")
        print(f"  Mean Dice: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"  Best Dice: {best_overall_dice:.4f} (Fold {best_fold+1})")
        print(f"  Best model saved at: {best_model_path}")
        
        # Save summary to CSV
        results_df.to_csv(os.path.join(self.metrics_dir, 'cv_summary.csv'), index=False)
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        plt.bar(results_df['fold'], results_df['best_val_dice'], color='skyblue')
        plt.axhline(y=avg_dice, color='red', linestyle='-', label=f'Mean Dice: {avg_dice:.4f}')
        plt.axhline(y=avg_dice+std_dice, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=avg_dice-std_dice, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Fold')
        plt.ylabel('Best Dice Coefficient')
        plt.title('Cross-Validation Results')
        plt.ylim(0, 1)
        plt.xticks(results_df['fold'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'cv_summary.png'))
        plt.close()


def main():
    """Main function to run GPU training with cross-validation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GPU-optimized Stress Granule Detection Training')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--folds', type=int, default=4, help='Number of cross-validation folds')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--crops', type=int, default=4, help='Number of crops per image')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    args = parser.parse_args()
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
        print("Forced CPU training")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU available)")
    
    # Match images and masks
    print("\nLoading dataset...")
    image_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'masks')
    
    matched_pairs = match_images_and_masks(image_dir, mask_dir)
    
    if len(matched_pairs) < args.folds:
        print(f"❌ Need at least {args.folds} image-mask pairs for {args.folds}-fold cross-validation")
        print(f"Found only {len(matched_pairs)} pairs")
        return
    
    # Analyze matched pairs
    analyze_matched_pairs(matched_pairs)
    
    # Extract image and mask paths
    image_paths = [p[0] for p in matched_pairs]
    mask_paths = [p[1] for p in matched_pairs]
    
    # Create metrics directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f'metrics/gpu_training_{timestamp}'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize K-fold trainer
    trainer = KFoldTrainer(
        image_paths=image_paths,
        mask_paths=mask_paths,
        n_splits=args.folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        crops_per_image=args.crops,
        patience=args.patience,
        metrics_dir=metrics_dir,
        experiment_name='stress_granule_cv',
        device=device
    )
    
    # Run cross-validation
    best_model_path = trainer.run_cross_validation()
    
    print(f"\n✅ Training complete!")
    print(f"Results saved in: {metrics_dir}")
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
