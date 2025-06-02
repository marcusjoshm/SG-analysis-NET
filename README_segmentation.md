# Stress Granule Segmentation Framework

This framework provides a complete pipeline for segmenting stress granules from experimental microscopy images using your trained neural network model.

## Features

- GPU-accelerated inference using Apple Silicon MPS or NVIDIA CUDA
- Handles large microscopy images via tiled processing
- Applies the same preprocessing as during training (CLAHE, Gaussian blur)
- Generates binary masks, probability maps, and overlays
- Provides detailed analysis of stress granule properties
- Batch processes entire directories of images
- Generates summary statistics and visualizations

## Usage

### Basic Usage

```bash
# Process a single image
python segment_stress_granules.py --model metrics/improved_gpu_20250531_193306/best_model.pth --image data/test_images/example.tif

# Process a directory of images
python segment_stress_granules.py --model metrics/improved_gpu_20250531_193306/best_model.pth --dir data/test_images
```

### All Options

```
usage: segment_stress_granules.py [-h] --model MODEL (--image IMAGE | --dir DIR)
                                  [--output OUTPUT] [--pattern PATTERN]
                                  [--tile-size TILE_SIZE] [--overlap OVERLAP]
                                  [--threshold THRESHOLD] [--no-gpu] [--no-vis]

Segment stress granules in microscopy images

required arguments:
  --model MODEL         Path to the trained model checkpoint (.pth file)
  --image IMAGE         Path to a single image file
  --dir DIR             Path to directory containing images

optional arguments:
  --output OUTPUT       Directory to save results (default: results_{timestamp})
  --pattern PATTERN     Pattern for matching image files (default: *.tif)
  --tile-size TILE_SIZE Size of tiles for processing large images (default: 1024)
  --overlap OVERLAP     Overlap between tiles (default: 256)
  --threshold THRESHOLD Threshold for binary segmentation (default: 0.5)
  --no-gpu              Disable GPU acceleration
  --no-vis              Disable visualization generation
```

## Output Structure

The framework creates the following directory structure:

```
results_{timestamp}/
├── masks/                  # Binary segmentation masks
│   └── image_name_mask.png
├── overlays/               # Original images with segmentation overlay
│   └── image_name_overlay.png
└── analysis/               # Analysis results and visualizations
    ├── image_name_properties.csv       # Per-granule properties
    ├── image_name_visualization.png    # Multi-panel visualization
    ├── summary_results.csv             # Summary of all processed images
    ├── summary_granules.png            # Bar charts of granule counts/sizes
    └── summary_timing.png              # Processing time comparison
```

## Integration with Existing Pipeline

The segmentation framework is designed to work seamlessly with your existing stress granule analysis pipeline:

1. Train models using `improved_training_gpu.py`
2. Use the best model checkpoint with `segment_stress_granules.py`
3. Analyze results using the generated CSV files and visualizations

## Technical Details

### Large Image Handling

The framework processes large microscopy images by:
1. Splitting them into overlapping tiles
2. Processing each tile through the model
3. Reconstructing the full-size prediction with smooth blending

This approach allows handling images of any size while maintaining GPU memory efficiency.

### Granule Analysis

For each detected stress granule, the framework calculates:
- Area (in pixels)
- Centroid location
- Shape properties

Summary statistics include:
- Total number of granules
- Total granule area
- Mean granule size
- Maximum granule size

## Example Workflow

```bash
# Step 1: Train your model with GPU acceleration
python improved_training_gpu.py

# Step 2: Segment experimental images
python segment_stress_granules.py --model metrics/improved_gpu_20250531_193306/best_model.pth --dir experimental_data

# Step 3: Review results in the output directory
open results_20250531_210000/analysis/summary_results.csv
```

## Performance Considerations

- GPU acceleration provides 5-10x faster processing compared to CPU
- Tile size can be adjusted based on GPU memory (larger = faster, but more memory)
- For extremely large images, increase overlap to reduce boundary artifacts
