# Stress Granule Analysis with Neural Networks

This project uses deep learning to analyze stress granules from microscopy data. It implements a U-Net architecture to segment stress granules in microscopy images with comprehensive metrics tracking and inference capabilities.

## Features

- **U-Net implementation** for semantic segmentation of stress granules
- **Modular code structure** with separate files for models, metrics, and inference
- **Comprehensive metrics tracking** including Dice, IoU, precision, recall, F1-score
- **Command-line interface** for flexible training and inference
- **Automatic checkpoint saving** and resuming capabilities
- **GPU acceleration support** with automatic batch size adjustment
- **Threshold analysis** to find optimal segmentation parameters
- **Inference script** for applying trained models to new data
- **Visualization tools** for model predictions and training progress
- **Setup verification script** to check dependencies and project structure

## Project Structure

```
SG-analysis-NET/
├── main.py              # Main training script
├── models.py            # U-Net model architecture and dataset classes
├── metrics.py           # Metrics tracking and evaluation utilities
├── inference.py         # Inference script for applying trained models
├── check_setup.py       # Setup verification and dependency check
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT license
├── data/
│   ├── images/         # Training images (.tif, .png, .jpg)
│   └── masks/          # Corresponding binary masks
└── metrics/            # Generated training metrics and plots
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

### Training a Model

Basic training:
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

### Running Inference

Apply a trained model to new images:
```bash
python inference.py --model best_stress_granule_model.pth \
                   --input_dir path/to/new/images \
                   --output_dir predictions \
                   --create_montage
```

### Command Line Options

For detailed help on available options:
```bash
python main.py --help
python inference.py --help
```

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

1. **GPU Memory Issues**: Reduce batch size using `--batch_size 4` or `--batch_size 2`
2. **File Not Found**: Ensure data is in the correct directory structure
3. **Dependencies**: Run `python check_setup.py` to verify installation
4. **Training Stalled**: Check that images and masks are properly aligned

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

```
MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.