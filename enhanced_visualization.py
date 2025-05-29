import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from main import UNet, StressGranuleDataset
from sklearn.model_selection import train_test_split
from glob import glob

def create_enhanced_visualizations(model_path, data_dir='data', num_samples=6, save_dir='enhanced_visualizations'):
    """
    Create enhanced visualizations showing model performance on actual images
    with overlays and detailed analysis
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model with best Dice: {checkpoint.get('best_dice', 'Unknown'):.4f}")
    else:
        print("No trained model found. Using untrained model for demonstration.")
    
    model = model.to(device)
    
    # Load data (same logic as main.py)
    from main import match_images_and_masks_custom
    
    image_paths = sorted(glob(os.path.join(data_dir, 'images', '*.tif')))
    mask_paths = sorted(glob(os.path.join(data_dir, 'masks', '*.tif')))
    
    matched_images, matched_masks = match_images_and_masks_custom(image_paths, mask_paths)
    
    # Split data the same way as training
    train_images, val_images, train_masks, val_masks = train_test_split(
        matched_images, matched_masks, test_size=0.2, random_state=42
    )
    
    # Create dataset
    val_dataset = StressGranuleDataset(val_images, val_masks, channels=3, target_size=(256, 256))
    
    # Select samples for visualization
    num_samples = min(num_samples, len(val_dataset))
    sample_indices = np.linspace(0, len(val_dataset)-1, num_samples, dtype=int)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample
            image, mask = val_dataset[idx]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            pred = model(image_batch).cpu().squeeze()
            
            # Convert to numpy
            if image.shape[0] == 3:  # RGB
                image_np = image.permute(1, 2, 0).numpy()
            else:  # Grayscale
                image_np = image.squeeze().numpy()
            
            mask_np = mask.squeeze().numpy()
            pred_np = pred.numpy()
            binary_pred = (pred_np > 0.5).astype(np.float32)
            
            # Original image
            axes[i, 0].imshow(np.clip(image_np, 0, 1))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Raw prediction (probability map)
            im = axes[i, 2].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction (Raw)')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Binary prediction
            axes[i, 3].imshow(binary_pred, cmap='gray')
            axes[i, 3].set_title('Prediction (Binary)')
            axes[i, 3].axis('off')
            
            # Overlay comparison
            overlay = image_np.copy()
            if len(overlay.shape) == 2:  # Convert grayscale to RGB for overlay
                overlay = np.stack([overlay, overlay, overlay], axis=2)
            
            # Add ground truth in green, predictions in red
            overlay_mask = np.zeros_like(overlay)
            overlay_mask[mask_np > 0.5] = [0, 1, 0]  # Green for ground truth
            overlay_mask[binary_pred > 0.5] = [1, 0, 0]  # Red for predictions
            
            # Where both overlap, make yellow
            overlap = (mask_np > 0.5) & (binary_pred > 0.5)
            overlay_mask[overlap] = [1, 1, 0]  # Yellow for overlap
            
            # Blend with original image
            alpha = 0.4
            result = overlay * (1 - alpha) + overlay_mask * alpha
            
            axes[i, 4].imshow(np.clip(result, 0, 1))
            axes[i, 4].set_title('Overlay (GT=Green, Pred=Red, Overlap=Yellow)')
            axes[i, 4].axis('off')
            
            # Calculate metrics for this sample
            intersection = (binary_pred * mask_np).sum()
            union = binary_pred.sum() + mask_np.sum() - intersection
            dice = 2 * intersection / (binary_pred.sum() + mask_np.sum() + 1e-6)
            iou = intersection / (union + 1e-6)
            
            # Add metrics text
            metrics_text = f'Dice: {dice:.3f}, IoU: {iou:.3f}'
            fig.text(0.02, 0.95 - i * (0.95/num_samples), f'Sample {i+1}: {metrics_text}', 
                    fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhanced_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create individual high-quality images for each sample
    for i, idx in enumerate(sample_indices):
        image, mask = val_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(image_batch).cpu().squeeze()
        
        # Save individual high-res comparison
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        if image.shape[0] == 3:
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = image.squeeze().numpy()
        
        mask_np = mask.squeeze().numpy()
        pred_np = pred.numpy()
        
        axes[0].imshow(np.clip(image_np, 0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        im = axes[2].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Model Prediction')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        axes[3].imshow((pred_np > 0.5).astype(float), cmap='gray')
        axes[3].set_title('Binary Prediction')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}_detailed.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Enhanced visualizations saved to {save_dir}/")
    print(f"- enhanced_predictions.png: Overview of all samples")
    print(f"- sample_X_detailed.png: Individual high-resolution comparisons")

def create_performance_summary(metrics_csv, save_dir='enhanced_visualizations'):
    """Create a performance summary visualization"""
    import pandas as pd
    
    if not os.path.exists(metrics_csv):
        print(f"Metrics file {metrics_csv} not found")
        return
    
    df = pd.read_csv(metrics_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice score
    axes[0, 1].plot(df['epoch'], df['dice_score'], color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score Progress')
    axes[0, 1].grid(True)
    
    # Precision and Recall
    axes[1, 0].plot(df['epoch'], df['precision'], label='Precision')
    axes[1, 0].plot(df['epoch'], df['recall'], label='Recall')
    axes[1, 0].plot(df['epoch'], df['f1_score'], label='F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall, F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Time per epoch
    axes[1, 1].plot(df['epoch'], df['time_per_epoch'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Training Time per Epoch')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n=== TRAINING SUMMARY ===")
    print(f"Total epochs: {len(df)}")
    print(f"Best Dice score: {df['dice_score'].max():.4f} (Epoch {df.loc[df['dice_score'].idxmax(), 'epoch']})")
    print(f"Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final val loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Average time per epoch: {df['time_per_epoch'].mean():.1f} seconds")

if __name__ == "__main__":
    # Run enhanced visualizations
    model_path = 'best_stress_granule_model.pth'
    
    print("Creating enhanced visualizations...")
    create_enhanced_visualizations(model_path)
    
    print("\nCreating performance summary...")
    metrics_file = 'metrics/stress_granule_full_training/stress_granule_full_training_metrics.csv'
    create_performance_summary(metrics_file)
    
    print("\n✅ All visualizations complete!") 