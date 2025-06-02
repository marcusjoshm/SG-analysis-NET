#!/usr/bin/env python
"""
Stress Granule Segmentation Framework
Uses trained models to segment stress granules from experimental microscopy images
"""

import os
import sys
import torch
import numpy as np
import cv2
import time
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from skimage import measure
from skimage.measure import regionprops
import pandas as pd

# Import model architecture
from improved_model import ImprovedUNet, dice_coefficient


class StressGranuleSegmenter:
    """Framework for segmenting stress granules from microscopy images"""
    
    def __init__(self, model_path, output_dir=None, device=None, tile_size=1024, 
                 overlap=256, threshold=0.5, use_gpu=True):
        """
        Initialize the segmentation framework
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            output_dir: Directory to save results (default: results_{timestamp})
            device: Torch device to use (default: auto-detect)
            tile_size: Size of tiles for processing large images (default: 1024)
            overlap: Overlap between tiles to avoid boundary artifacts (default: 256)
            threshold: Probability threshold for binary segmentation (default: 0.5)
            use_gpu: Whether to use GPU acceleration if available (default: True)
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold
        
        # Set up device
        if device is None:
            if use_gpu and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print(f"Using Apple Silicon GPU (MPS)")
            elif use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("Using CPU for inference")
        else:
            self.device = device
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"results_{timestamp}"
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "overlays"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Set up results tracking
        self.results = {
            "filename": [],
            "processing_time": [],
            "image_size": [],
            "num_granules": [],
            "total_area": [],
            "mean_granule_size": [],
            "max_granule_size": []
        }
    
    def _load_model(self):
        """Load the trained model from checkpoint"""
        try:
            # Initialize model architecture
            model = ImprovedUNet(in_channels=1, out_channels=1)
            
            # Load weights
            if self.device.type == 'cpu':
                checkpoint = torch.load(self.model_path, map_location='cpu')
            else:
                checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    def _enhance_contrast(self, image, percentile_low=0.5, percentile_high=99.5):
        """Enhanced contrast using CLAHE"""
        # Calculate percentiles for non-zero pixels only
        non_zero_pixels = image[image > 0]
        if len(non_zero_pixels) > 0:
            p_low = np.percentile(non_zero_pixels, percentile_low)
            p_high = np.percentile(non_zero_pixels, percentile_high)
            
            # Apply contrast stretching
            image_norm = np.zeros_like(image, dtype=np.float32)
            image_norm = np.clip((image - p_low) / (p_high - p_low + 1e-6), 0, 1)
        else:
            image_norm = np.zeros_like(image, dtype=np.float32)
        
        # Apply CLAHE for additional contrast enhancement
        # Convert to 8-bit for CLAHE
        image_8bit = (image_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_8bit)
        
        # Convert back to float32 [0,1]
        return image_clahe.astype(np.float32) / 255.0
    
    def _apply_gaussian_blur(self, image, sigma=1.5):
        """Apply Gaussian blur to reduce noise"""
        return cv2.GaussianBlur(image, (0, 0), sigma)
    
    def _preprocess_image(self, image):
        """Apply all preprocessing steps to an image"""
        # Normalize to [0,1] range
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Apply contrast enhancement
        image = self._enhance_contrast(image)
        
        # Apply Gaussian blur
        image = self._apply_gaussian_blur(image)
        
        return image
    
    def _create_tiles(self, image):
        """Split large image into overlapping tiles"""
        h, w = image.shape
        tiles = []
        positions = []
        
        # Calculate effective stride (tile_size - overlap)
        stride = self.tile_size - self.overlap
        
        # Generate tiles with positions
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Adjust to not exceed image dimensions
                end_y = min(y + self.tile_size, h)
                end_x = min(x + self.tile_size, w)
                
                # Adjust start position for last tiles to maintain tile size
                start_y = max(0, end_y - self.tile_size)
                start_x = max(0, end_x - self.tile_size)
                
                # Extract tile
                tile = image[start_y:end_y, start_x:end_x]
                
                # Only add if tile is full-sized or at the image edge
                if tile.shape[0] > 0 and tile.shape[1] > 0:
                    tiles.append(tile)
                    positions.append((start_y, start_x, end_y, end_x))
        
        return tiles, positions
    
    def _reconstruct_from_tiles(self, tiles, positions, original_shape):
        """Reconstruct full image from tiles and their positions"""
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        for tile, (start_y, start_x, end_y, end_x) in zip(tiles, positions):
            # Create weight mask (higher weight in the center, lower at edges)
            y_weights = np.ones(end_y - start_y)
            x_weights = np.ones(end_x - start_x)
            
            # Apply weight only for overlapping tiles
            if end_y - start_y == self.tile_size:
                y_weights[:self.overlap//2] = np.linspace(0.1, 1, self.overlap//2)
                y_weights[-self.overlap//2:] = np.linspace(1, 0.1, self.overlap//2)
            
            if end_x - start_x == self.tile_size:
                x_weights[:self.overlap//2] = np.linspace(0.1, 1, self.overlap//2)
                x_weights[-self.overlap//2:] = np.linspace(1, 0.1, self.overlap//2)
            
            # Create 2D weight mask
            weight_mask = np.outer(y_weights, x_weights)
            
            # Add tile to reconstruction with weight
            reconstructed[start_y:end_y, start_x:end_x] += tile * weight_mask
            weights[start_y:end_y, start_x:end_x] += weight_mask
        
        # Normalize by weights to get final reconstruction
        # Avoid division by zero
        weights = np.maximum(weights, 1e-6)
        reconstructed = reconstructed / weights
        
        return reconstructed
    
    def segment_image(self, image_path):
        """
        Segment stress granules in a microscopy image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            binary_mask: Binary segmentation mask
            probability_mask: Probability mask (before thresholding)
            processing_time: Time taken to process the image (seconds)
        """
        start_time = time.time()
        
        # Load image
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            # For RGB images, convert to grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            original_shape = image.shape
            print(f"Processing image: {os.path.basename(image_path)} ({original_shape[0]}×{original_shape[1]})")
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None, None, 0
        
        # Preprocess image
        preprocessed = self._preprocess_image(image)
        
        # Split into tiles for large images
        tiles, positions = self._create_tiles(preprocessed)
        print(f"Split into {len(tiles)} tiles of size {self.tile_size}×{self.tile_size} with {self.overlap}px overlap")
        
        # Process each tile
        processed_tiles = []
        with torch.no_grad():
            for i, tile in enumerate(tqdm(tiles, desc="Processing tiles")):
                # Prepare for model (add batch and channel dimensions)
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Get prediction
                prediction = self.model(tile_tensor)
                
                # Convert to probability map
                probability = torch.sigmoid(prediction).squeeze().cpu().numpy()
                
                # Add to processed tiles
                processed_tiles.append(probability)
        
        # Reconstruct full probability mask
        probability_mask = self._reconstruct_from_tiles(processed_tiles, positions, original_shape)
        
        # Apply threshold to get binary mask
        binary_mask = (probability_mask > self.threshold).astype(np.uint8) * 255
        
        processing_time = time.time() - start_time
        
        return binary_mask, probability_mask, processing_time
    
    def analyze_mask(self, binary_mask, min_size=10):
        """
        Analyze the segmentation mask to extract properties of stress granules
        
        Args:
            binary_mask: Binary segmentation mask
            min_size: Minimum size (in pixels) for a region to be considered a granule
            
        Returns:
            properties: Dictionary of properties
            labeled_mask: Labeled mask with unique IDs for each granule
        """
        # Ensure mask is binary (0 or 1)
        binary = (binary_mask > 0).astype(np.uint8)
        
        # Label connected components
        labeled_mask = measure.label(binary)
        props = regionprops(labeled_mask)
        
        # Filter small regions
        filtered_props = [p for p in props if p.area >= min_size]
        
        # Get properties
        areas = [p.area for p in filtered_props]
        
        # Calculate statistics
        if areas:
            total_area = sum(areas)
            mean_area = total_area / len(areas)
            max_area = max(areas)
        else:
            total_area = mean_area = max_area = 0
        
        properties = {
            "num_granules": len(filtered_props),
            "total_area": total_area,
            "mean_granule_size": mean_area,
            "max_granule_size": max_area,
            "granule_areas": areas
        }
        
        return properties, labeled_mask
    
    def create_overlay(self, image, mask, alpha=0.5):
        """Create an overlay of the segmentation on the original image"""
        # Ensure image is RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # Normalize to 0-255 if needed
        if image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            else:
                image_rgb = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Create color mask (red)
        mask_rgb = np.zeros_like(image_rgb)
        mask_rgb[:,:,2] = mask  # Red channel
        
        # Create overlay
        overlay = cv2.addWeighted(image_rgb, 1, mask_rgb, alpha, 0)
        
        return overlay
    
    def process_image(self, image_path, save_results=True, visualize=True):
        """
        Process a single image through the entire pipeline
        
        Args:
            image_path: Path to the image file
            save_results: Whether to save the results to disk
            visualize: Whether to generate and save visualizations
            
        Returns:
            results: Dictionary with segmentation results and properties
        """
        # Extract filename
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Segment image
        binary_mask, probability_mask, processing_time = self.segment_image(image_path)
        
        if binary_mask is None:
            return None
        
        # Load original image for visualization
        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Analyze mask
        properties, labeled_mask = self.analyze_mask(binary_mask)
        
        # Create results dictionary
        result = {
            "filename": filename,
            "processing_time": processing_time,
            "image_size": original_image.shape,
            "num_granules": properties["num_granules"],
            "total_area": properties["total_area"],
            "mean_granule_size": properties["mean_granule_size"],
            "max_granule_size": properties["max_granule_size"]
        }
        
        # Update results tracking
        for key in self.results:
            if key in result:
                self.results[key].append(result[key])
        
        # Save results if requested
        if save_results:
            # Save binary mask
            mask_path = os.path.join(self.output_dir, "masks", f"{name_without_ext}_mask.png")
            cv2.imwrite(mask_path, binary_mask)
            
            # Save granule properties
            props_path = os.path.join(self.output_dir, "analysis", f"{name_without_ext}_properties.csv")
            pd.DataFrame({
                "granule_id": range(1, len(properties["granule_areas"]) + 1),
                "area": properties["granule_areas"]
            }).to_csv(props_path, index=False)
            
            # Save visualizations if requested
            if visualize:
                # Create and save overlay
                overlay = self.create_overlay(original_image, binary_mask)
                overlay_path = os.path.join(self.output_dir, "overlays", f"{name_without_ext}_overlay.png")
                cv2.imwrite(overlay_path, overlay)
                
                # Create visualization figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                if original_image.dtype == np.uint16:
                    # Normalize 16-bit to 0-1 for display
                    display_img = original_image.astype(np.float32) / 65535.0
                else:
                    display_img = original_image.astype(np.float32) / 255.0
                
                axes[0].imshow(display_img, cmap='gray')
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # Probability map
                im = axes[1].imshow(probability_mask, cmap='viridis')
                axes[1].set_title("Probability Map")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                # Binary mask
                axes[2].imshow(binary_mask, cmap='gray')
                axes[2].set_title(f"Segmentation ({properties['num_granules']} granules)")
                axes[2].axis('off')
                
                plt.tight_layout()
                fig_path = os.path.join(self.output_dir, "analysis", f"{name_without_ext}_visualization.png")
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"✅ Processed {filename} in {processing_time:.2f}s - Found {properties['num_granules']} granules")
        return result
    
    def process_directory(self, input_dir, pattern="*.tif"):
        """
        Process all matching images in a directory
        
        Args:
            input_dir: Directory containing images
            pattern: Glob pattern for image files (default: *.tif)
        """
        # Find all matching image files
        image_paths = glob(os.path.join(input_dir, pattern))
        
        if not image_paths:
            print(f"No images found matching pattern '{pattern}' in {input_dir}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process each image
        for image_path in image_paths:
            self.process_image(image_path)
        
        # Save summary results
        self.save_summary()
    
    def save_summary(self):
        """Save summary results to CSV and generate summary visualizations"""
        # Save results to CSV
        summary_path = os.path.join(self.output_dir, "analysis", "summary_results.csv")
        pd.DataFrame(self.results).to_csv(summary_path, index=False)
        
        # Create summary visualizations
        if len(self.results["filename"]) > 0:
            # Create granule count and size bar charts
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Number of granules by image
            ax1.bar(self.results["filename"], self.results["num_granules"])
            ax1.set_title("Number of Stress Granules by Image")
            ax1.set_xlabel("Image")
            ax1.set_ylabel("Number of Granules")
            ax1.tick_params(axis='x', rotation=45)
            
            # Mean granule size by image
            ax2.bar(self.results["filename"], self.results["mean_granule_size"])
            ax2.set_title("Mean Granule Size by Image")
            ax2.set_xlabel("Image")
            ax2.set_ylabel("Mean Size (pixels)")
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "analysis", "summary_granules.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create processing time comparison
            plt.figure(figsize=(10, 5))
            plt.bar(self.results["filename"], self.results["processing_time"])
            plt.title("Processing Time by Image")
            plt.xlabel("Image")
            plt.ylabel("Processing Time (seconds)")
            plt.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "analysis", "summary_timing.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Summary results saved to {summary_path}")


def main():
    """Main function for command-line use"""
    parser = argparse.ArgumentParser(description="Segment stress granules in microscopy images")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint (.pth file)")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a single image file")
    input_group.add_argument("--dir", help="Path to directory containing images")
    
    # Optional arguments
    parser.add_argument("--output", help="Directory to save results (default: results_{timestamp})")
    parser.add_argument("--pattern", default="*.tif", help="Pattern for matching image files (default: *.tif)")
    parser.add_argument("--tile-size", type=int, default=1024, help="Size of tiles for processing large images (default: 1024)")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap between tiles (default: 256)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation (default: 0.5)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization generation")
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = StressGranuleSegmenter(
        model_path=args.model,
        output_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
        use_gpu=not args.no_gpu
    )
    
    # Process input
    if args.image:
        # Process single image
        segmenter.process_image(args.image, visualize=not args.no_vis)
        segmenter.save_summary()
    else:
        # Process directory
        segmenter.process_directory(args.dir, pattern=args.pattern)


if __name__ == "__main__":
    main()
