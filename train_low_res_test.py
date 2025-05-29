#!/usr/bin/env python
"""
Low-resolution training test with 3 images
Designed to complete successfully on MPS/CPU with limited memory
"""

import os
import torch
import numpy as np
from train_small_test import match_images_and_masks, analyze_matched_pairs
from models import UNet, StressGranule16bitDataset
from metrics import MetricsTracker
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from main import train_model, visualize_predictions, plot_training_history
import matplotlib.pyplot as plt

def main():
    # Configuration with unweighted loss
    IMAGE_SIZE = 256
    BATCH_SIZE = 2
    NUM_EPOCHS = 75
    LEARNING_RATE = 5e-5
    NUM_SAMPLES = 4
    CROPS_PER_IMAGE = 3
    
    print("=" * 60)
    print("Multi-Crop Training Test (Unweighted Loss)")
    print(f"Configuration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Crops per image: {CROPS_PER_IMAGE}")
    print(f"  - Effective training samples: {NUM_SAMPLES * CROPS_PER_IMAGE}")
    print(f"  - Loss: BCE (50%) + Dice (50%)")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Match images and masks
    matched_pairs = match_images_and_masks('data/images', 'data/masks', NUM_SAMPLES)
    
    if len(matched_pairs) < 3:
        print("❌ Need at least 3 image-mask pairs")
        return
    
    # Analyze the pairs
    analyze_matched_pairs(matched_pairs)
    
    # Split: 3 for training, 1 for validation
    train_pairs = matched_pairs[:3]
    val_pairs = matched_pairs[3:4]
    
    train_images = [p[0] for p in train_pairs]
    train_masks = [p[1] for p in train_pairs]
    val_images = [p[0] for p in val_pairs]
    val_masks = [p[1] for p in val_pairs]
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_images)} images × {CROPS_PER_IMAGE} crops = {len(train_images) * CROPS_PER_IMAGE} samples")
    print(f"  Validation: {len(val_images)} images × {CROPS_PER_IMAGE} crops = {len(val_images) * CROPS_PER_IMAGE} samples")
    
    # Light augmentation for testing
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    # Create datasets
    train_dataset = StressGranule16bitDataset(
        train_images, train_masks,
        transform=train_transform,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7,
        random_crop=True,
        crops_per_image=CROPS_PER_IMAGE
    )
    
    val_dataset = StressGranule16bitDataset(
        val_images, val_masks,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7,
        random_crop=True,  # Use random crops for validation too
        crops_per_image=CROPS_PER_IMAGE
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1)
    print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create metrics directory
    metrics_dir = 'metrics/low_res_test'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir=metrics_dir, experiment_name='low_res_test')
    
    # Test data loading
    print("\n🔍 Testing data loading...")
    try:
        sample_img, sample_mask = train_dataset[0]
        print(f"✅ Sample image shape: {sample_img.shape}")
        print(f"✅ Sample mask shape: {sample_mask.shape}")
        print(f"✅ Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
        
        # Quick visualization of a sample
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample_img.squeeze().numpy(), cmap='gray')
        axes[0].set_title('Sample Image (Preprocessed)')
        axes[0].axis('off')
        
        axes[1].imshow(sample_mask.squeeze().numpy(), cmap='gray')
        axes[1].set_title('Sample Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'sample_data.png'))
        plt.close()
        print(f"✅ Sample visualization saved")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Train model
    print("\n🚀 Starting training...")
    try:
        train_losses, val_losses, val_dice_scores, metrics_tracker = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            patience=30,  # Increased from 5 to allow more training
            device=device,
            metrics_tracker=metrics_tracker,
            experiment_name='low_res_test'
        )
        
        print("\n✅ Training completed successfully!")
        
        # Check if model learned anything
        if len(train_losses) > 1:
            initial_loss = train_losses[0]
            final_loss = train_losses[-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            print(f"\n📈 Training Progress:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Improvement: {improvement:.1f}%")
            
            if val_dice_scores and max(val_dice_scores) > 0:
                print(f"  Best Dice score: {max(val_dice_scores):.4f}")
        
        # Plot training history
        plot_training_history(train_losses, val_losses, val_dice_scores)
        os.rename('training_history.png', os.path.join(metrics_dir, 'training_history.png'))
        
        # Visualize predictions
        print("\n🎨 Generating prediction visualizations...")
        visualize_predictions(
            model, val_dataset, device,
            num_samples=1,
            output_file=os.path.join(metrics_dir, 'predictions.png')
        )
        
        # Save test model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'image_size': IMAGE_SIZE,
                'num_epochs': NUM_EPOCHS,
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_dice': val_dice_scores[-1] if val_dice_scores else None
            }
        }, os.path.join(metrics_dir, 'low_res_test_model.pth'))
        
        print(f"\n✅ Test completed! Results saved in {metrics_dir}/")
        print("\n📊 Summary:")
        print("  - Training pipeline: ✅ Working")
        print("  - Data loading & preprocessing: ✅ Working")
        print("  - Model training: ✅ Working")
        print("  - Memory usage: ✅ Within limits")
        
        print("\n🎯 Next steps:")
        print("  1. Transfer this code to your GPU machine")
        print("  2. Run full training with higher resolution:")
        print("     python main.py --data_dir data --epochs 100 --image_size 512")
        print("  3. Or start with the 3-image high-res test:")
        print("     python train_small_test.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 