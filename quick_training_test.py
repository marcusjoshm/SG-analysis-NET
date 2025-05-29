#!/usr/bin/env python
"""
Quick training test - runs 1-2 epochs with smaller images to verify setup
This is meant to run quickly even on CPU/MPS to ensure everything works
"""

import os
import torch
import numpy as np
from train_small_test import match_images_and_masks
from models import UNet, StressGranule16bitDataset
from torch.utils.data import DataLoader
import time

def quick_test():
    """Run a very quick training test"""
    
    print("=" * 60)
    print("Quick Training Test - Verifying Setup")
    print("=" * 60)
    
    # Ultra small configuration for quick testing
    IMAGE_SIZE = 256  # Small size for quick test
    BATCH_SIZE = 1
    NUM_EPOCHS = 2    # Just 2 epochs
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (MPS not available)")
    
    # Match 2 images
    matched_pairs = match_images_and_masks('data/images', 'data/masks', max_samples=2)
    
    if len(matched_pairs) < 2:
        print("❌ Need at least 2 image-mask pairs for this test")
        return False
    
    # Use first for training, second for validation
    train_images = [matched_pairs[0][0]]
    train_masks = [matched_pairs[0][1]]
    val_images = [matched_pairs[1][0]]
    val_masks = [matched_pairs[1][1]]
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_images)} image")
    print(f"  Validation: {len(val_images)} image")
    
    # Create minimal datasets
    train_dataset = StressGranule16bitDataset(
        train_images, train_masks,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    val_dataset = StressGranule16bitDataset(
        val_images, val_masks,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        enhance_contrast=True,
        gaussian_sigma=1.7
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()
    
    print(f"\n🧠 Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    print("\n🔍 Testing forward pass...")
    try:
        sample_img, sample_mask = train_dataset[0]
        sample_img = sample_img.unsqueeze(0).to(device)
        sample_mask = sample_mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(sample_img)
        
        print(f"✅ Forward pass successful!")
        print(f"  Input shape: {sample_img.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Quick training loop
    print(f"\n🚀 Running {NUM_EPOCHS} training epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Time: {epoch_time:.2f}s")
    
    print("\n✅ Training test completed successfully!")
    print("\n📝 Summary:")
    print("  - Data loading: ✅")
    print("  - Model forward pass: ✅")
    print("  - Training loop: ✅")
    print("  - Device support: ✅")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 All tests passed! You can now run the full training with:")
        print("   python train_small_test.py")
        print("\nOr on a GPU machine:")
        print("   python main.py --data_dir data --epochs 100")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 