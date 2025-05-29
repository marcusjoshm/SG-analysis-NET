# Stress Granule Analysis with Neural Networks

This project uses deep learning to analyze stress granules from microscopy data. It implements a U-Net architecture to segment stress granules in microscopy images with comprehensive metrics tracking, enhanced preprocessing, and inference capabilities.

## Features

- **U-Net implementation** for semantic segmentation of stress granules
- **Enhanced preprocessing pipeline** with 16-bit image support and automatic contrast enhancement
- **Gaussian blur preprocessing** for noise reduction and improved segmentation (σ=1.7)
- **High-resolution training** support (up to 1024×1024 or higher)
- **Modular code structure** with separate files for models, metrics, and inference
- **Comprehensive metrics tracking** including Dice, IoU, precision, recall, F1-score
- **Command-line interface** for flexible training and inference
- **Automatic checkpoint saving** and resuming capabilities
- **GPU acceleration support** with automatic batch size adjustment
- **Threshold analysis** to find optimal segmentation parameters
- **Enhanced inference script** with proper 16-bit support and visualization
- **Advanced visualization tools** for model predictions and training progress
- **Setup verification script** to check dependencies and project structure

## Project Structure

```
SG-analysis-NET/
├── main.py                    # Original training script (256×256)
├── enhanced_training.py       # Enhanced training with contrast & Gaussian blur
├── models.py                  # U-Net model architecture and dataset classes
├── metrics.py                 # Metrics tracking and evaluation utilities
├── inference.py               # Enhanced inference with 16-bit support
├── enhanced_visualization.py  # Advanced visualization tools
├── check_setup.py            # Setup verification and dependency check
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT license
├── data/
│   ├── images/              # Training images (.tif, .png, .jpg)
│   └── masks/               # Corresponding binary masks
└── metrics/                 # Generated training metrics and plots (excluded from git)
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- pandas
- tqdm

## Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/marcusjoshm/SG-analysis-NET.git
   cd SG-analysis-NET
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup**
   ```bash
   python check_setup.py
   ```

## Data Organization

Organize your data as follows:
```
data/
├── images/
│   ├── img001.tif
│   ├── img002.tif
│   └── ...
└── masks/
    ├── mask001.tif  # Should match corresponding image names
    ├── mask002.tif
    └── ...
```

**Important**: Image and mask files should have matching names (excluding extensions).

## Usage

### Enhanced Training (Recommended)

For best results with 16-bit microscopy images, use the enhanced training script:

```bash
# Basic enhanced training with contrast enhancement and Gaussian blur
python enhanced_training.py --epochs 75 --batch_size 4 --experiment_name "enhanced_sg_training"

# High-resolution training for better detail (requires more memory)
python enhanced_training.py --epochs 50 --batch_size 2 --image_size 1024 --experiment_name "high_res_training"

# Custom Gaussian blur settings
python enhanced_training.py --blur_sigma 2.0 --experiment_name "custom_blur"

# Disable specific enhancements if needed
python enhanced_training.py --no_gaussian_blur --no_contrast_enhancement
```

### Original Training

Basic training with the original script (256×256 resolution):
```bash
python main.py
```

Advanced training with custom parameters:
```bash
python main.py --data_dir path/to/data \
               --epochs 100 \
               --batch_size 8 \
               --learning_rate 0.0001 \
               --experiment_name my_experiment \
               --image_size 512
```

Resume training from checkpoint:
```bash
python main.py --resume --checkpoint checkpoint.pth
```

### Enhanced Inference

Apply a trained model to new images with proper 16-bit support:
```bash
# Basic inference with enhanced visualization
python inference.py --model best_enhanced_model.pth \
                   --input_dir path/to/new/images \
                   --output_dir predictions \
                   --create_montage \
                   --channels 1

# High-resolution inference
python inference.py --model best_enhanced_model.pth \
                   --input_dir path/to/images \
                   --output_dir predictions \
                   --image_size 1024 \
                   --channels 1
```

## Enhanced Training Features

### 16-bit Image Support

The enhanced training pipeline automatically handles 16-bit TIFF images common in microscopy:
- **Automatic detection** of 16-bit vs 8-bit images
- **Dynamic range stretching** from low-contrast (e.g., 0-202) to full 0-255 range
- **Proper bit-depth handling** throughout the pipeline

### Contrast Enhancement

Enhanced preprocessing dramatically improves training on low-contrast microscopy images:
- **Training**: Random brightness (±30%) and contrast (±50%) augmentation
- **Validation**: Fixed contrast enhancement for consistent evaluation
- **Inference**: Same enhancement applied for proper model input

### Gaussian Blur Preprocessing

Noise reduction using Gaussian blur (default σ=1.7):
- **Based on research**: Optimal sigma for microscopy image segmentation
- **Configurable**: Adjust `--blur_sigma` or disable with `--no_gaussian_blur`
- **Applied before**: Resizing and normalization for best results

### High-Resolution Training

Support for training at higher resolutions:
- **Standard**: 256×256 (fast, good for initial experiments)
- **High-res**: 1024×1024 (4x better detail, reduces pixelation)
- **Custom**: Any resolution supported
- **Memory scaling**: Automatically adjusts for available resources

### Advanced Data Augmentation

Enhanced geometric augmentations:
- **Random rotations**: Up to 45 degrees
- **Affine transformations**: Translation and scaling
- **Gaussian blur augmentation**: Additional noise reduction
- **Mask-preserving**: Proper handling of binary masks during transforms

### Command Line Options

For detailed help on available options:
```bash
python enhanced_training.py --help
python inference.py --help
python main.py --help
```

Common enhanced training options:
- `--image_size`: Training resolution (256, 512, 1024, etc.)
- `--blur_sigma`: Gaussian blur sigma (default: 1.7)
- `--no_gaussian_blur`: Disable Gaussian blur
- `--no_contrast_enhancement`: Disable contrast enhancement
- `--batch_size`: Adjust based on GPU memory
- `--patience`: Early stopping patience (default: 25)

## Model Architecture

The project uses a U-Net architecture with the following structure:
- **Encoder**: 4 downsampling blocks with double convolution
- **Bottleneck**: Double convolution layer
- **Decoder**: 4 upsampling blocks with skip connections
- **Automatic padding**: Handles arbitrary input sizes
- **Combined loss**: BCE + Dice loss for better segmentation

## Evaluation Metrics

The system tracks multiple metrics during training:
- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Percentage of correctly classified pixels

## Output Files

After training, the following files are generated:
- `best_stress_granule_model.pth`: Best model checkpoint
- `metrics/experiment_name/`: Directory containing:
  - Training curves and metrics plots
  - Confusion matrices
  - Threshold analysis
  - CSV files with detailed metrics
  - Prediction visualizations

## Troubleshooting

### General Issues
1. **GPU Memory Issues**: Reduce batch size using `--batch_size 4` or `--batch_size 2`
2. **File Not Found**: Ensure data is in the correct directory structure
3. **Dependencies**: Run `python check_setup.py` to verify installation
4. **Training Stalled**: Check that images and masks are properly aligned

### Enhanced Training Issues
1. **16-bit Images Loading as Black**: Use `enhanced_training.py` instead of `main.py`
2. **High Memory Usage**: Reduce `--image_size` from 1024 to 512 or 256
3. **Poor Performance on Low-Contrast Images**: Ensure contrast enhancement is enabled (default)
4. **Runtime Errors with Transforms**: Update to latest PyTorch version
5. **Channel Mismatch in Inference**: Match `--channels` parameter to model training

### Performance Tips
1. **For 16-bit TIFF files**: Always use `enhanced_training.py` and `enhanced_inference.py`
2. **For high-resolution training**: Start with smaller datasets (3-5 images) to test
3. **For best results**: Use image_size 1024 with batch_size 1-2
4. **For faster training**: Use image_size 256-512 with larger batch sizes

## Advanced Usage

### Custom Data Augmentation

Modify the data augmentation in `main.py`:
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    # Add more transforms here
])
```

### Experiment Comparison

The metrics system supports comparing multiple experiments:
```python
from metrics import MetricsTracker
tracker = MetricsTracker()
tracker.compare_experiments(['exp1_metrics.csv', 'exp2_metrics.csv'])
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.