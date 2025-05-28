import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tqdm import tqdm
import time
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# Import model architecture from main.py
from main import UNet, StressGranuleDataset

class InferenceDataset(Dataset):
    """Dataset for inference only (no masks required)"""
    def __init__(self, image_paths, target_size=(256, 256), channels=3):
        self.image_paths = image_paths
        self.target_size = target_size
        self.channels = channels
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            print(f"Error loading image: {e}")
            # Return a default small image in case of error
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Convert to grayscale if needed
        if self.channels == 1 and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Store original size for reference
        original_size = image.shape[:2]
        
        # Resize to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        if self.channels == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC to CHW
        else:
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension for grayscale
        
        return image, original_size, self.image_paths[idx]

def load_model(model_path, input_channels=3, device='cpu'):
    """Load a trained model from checkpoint"""
    model = UNet(in_channels=input_channels, out_channels=1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both state_dict and full checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_dice' in checkpoint:
                print(f"Loaded model with validation Dice score: {checkpoint['best_dice']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict successfully")
            
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def run_inference(model, dataset, batch_size=4, device='cpu', threshold=0.5, save_dir='predictions'):
    """Run inference on dataset and save predictions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            images, original_sizes, file_paths = batch
            images = images.to(device)
            
            # Run inference
            outputs = model(images)
            
            # Process each image in the batch
            for i in range(len(images)):
                # Get prediction
                pred = outputs[i].cpu().squeeze().numpy()
                
                # Apply threshold
                binary_pred = (pred > threshold).astype(np.uint8) * 255
                
                # Get original size and filepath
                orig_h, orig_w = original_sizes[0][i].item(), original_sizes[1][i].item()
                file_path = file_paths[i]
                
                # Resize back to original size if needed
                if binary_pred.shape != (orig_h, orig_w):
                    binary_pred = cv2.resize(binary_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # Create output filename
                filename = os.path.basename(file_path)
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(save_dir, f"{base_name}_pred.png")
                
                # Save prediction
                cv2.imwrite(output_path, binary_pred)
                
                # Optional: create overlay visualization
                original_img = cv2.imread(file_path)
                if original_img is not None:
                    overlay = original_img.copy()
                    # Create colored mask (red)
                    mask_colored = np.zeros_like(original_img)
                    mask_colored[binary_pred > 0] = [0, 0, 255]  # Red in BGR
                    # Blend
                    alpha = 0.4
                    overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
                    # Save overlay
                    overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
                    cv2.imwrite(overlay_path, overlay)
                
                # Store result info
                results.append({
                    'filename': filename,
                    'prediction_path': output_path,
                    'overlay_path': overlay_path if original_img is not None else None,
                    'pixels_segmented': np.sum(binary_pred > 0),
                    'total_pixels': orig_h * orig_w,
                    'percent_segmented': (np.sum(binary_pred > 0) / (orig_h * orig_w)) * 100
                })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'inference_results.csv'), index=False)
    
    print(f"Inference complete. Processed {len(dataset)} images.")
    print(f"Results saved to {save_dir}")
    
    return results_df

def create_montage(results_df, save_dir, max_images=16):
    """Create a montage of original images and their predictions"""
    n_images = min(len(results_df), max_images)
    if n_images == 0:
        print("No images to create montage")
        return
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4))
    axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        if i < n_images:
            # Load overlay image
            overlay_path = results_df.iloc[i]['overlay_path']
            if overlay_path and os.path.exists(overlay_path):
                img = cv2.imread(overlay_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(os.path.basename(results_df.iloc[i]['filename']))
            else:
                axes[i].text(0.5, 0.5, "Image not available", 
                            horizontalalignment='center', verticalalignment='center')
        
        # Turn off axis
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_montage.png'))
    plt.close()
    print(f"Montage saved to {os.path.join(save_dir, 'inference_montage.png')}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained stress granule segmentation model')
    
    parser.add_argument('--model', type=str, default='best_stress_granule_model.pth',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images for model input')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--extensions', type=str, default='.tif,.tiff,.png,.jpg,.jpeg',
                        help='Comma-separated list of image extensions to process')
    parser.add_argument('--create_montage', action='store_true',
                        help='Create a montage of predictions')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return
    
    # Get image file paths
    extensions = args.extensions.split(',')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(args.input_dir, f"*{ext}")))
    
    if not image_paths:
        print(f"No images found in '{args.input_dir}' with extensions {extensions}")
        return
    
    print(f"Found {len(image_paths)} images for processing")
    
    # Create dataset
    dataset = InferenceDataset(
        image_paths=image_paths,
        target_size=(args.image_size, args.image_size),
        channels=args.channels
    )
    
    # Load model
    model = load_model(args.model, input_channels=args.channels, device=device)
    
    # Run inference
    start_time = time.time()
    results = run_inference(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        threshold=args.threshold,
        save_dir=args.output_dir
    )
    elapsed_time = time.time() - start_time
    
    print(f"Inference completed in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(dataset):.4f} seconds")
    
    # Create montage if requested
    if args.create_montage:
        create_montage(results, args.output_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 