# Robust GPU Training for Stress Granule Detection

This document describes the robust training script (`robust_training_gpu.py`) designed to train a more resilient and accurate stress granule detection model using GPU acceleration.

## Key Features

### 1. **Enhanced Data Diversity**
- **Multi-scale contrast enhancement**: Applies CLAHE at multiple scales (2.0, 4.0, 8.0) with different grid sizes
- **Diverse cropping strategies**: 
  - Center on stress granules (30%)
  - Focus on stress granule edges (20%)
  - Background regions (15%)
  - Mixed regions (20%)
  - Random crops (15%)
- **5 crops per image** for increased data diversity

### 2. **Advanced Data Augmentation**
Using albumentations library for sophisticated transformations:
- **Spatial transforms**: Random rotations, flips, shifts, scaling
- **Elastic deformations**: ElasticTransform, GridDistortion, OpticalDistortion
- **Intensity variations**: Brightness/contrast, gamma correction, CLAHE
- **Noise injection**: Gaussian and multiplicative noise
- **Blur effects**: Motion, median, and Gaussian blur
- **Occlusion simulation**: CoarseDropout for robustness to partial occlusions

### 3. **Robust Loss Function**
Combines three complementary loss functions:
- **Weighted BCE**: With pos_weight=10 for class imbalance
- **Focal Tversky Loss**: Better handling of false positives/negatives
- **Dice Loss**: For overlap optimization

### 4. **Advanced Training Strategy**
- **AdamW optimizer**: With weight decay for regularization
- **Cosine Annealing with Warm Restarts**: Dynamic learning rate scheduling
- **Weighted sampling**: Balances training based on stress granule content
- **Extended training**: 200 epochs with patience of 30
- **Regular checkpointing**: Every 10 epochs

### 5. **Comprehensive Evaluation**
- **Multiple metrics**: Dice, Precision, Recall, F1, Accuracy, Specificity
- **Detailed visualizations**: 
  - Training history plots
  - Prediction visualizations
  - Probability maps
  - Overlay comparisons
- **JSON reports**: All metrics and configurations saved

## Usage

```bash
python robust_training_gpu.py
```

## Configuration

Default settings (can be modified in the script):
- **Image size**: 256×256 pixels
- **Batch size**: 4
- **Learning rate**: 1e-4
- **Epochs**: 200
- **Patience**: 30
- **Crops per image**: 5

## Output

The script creates a timestamped directory in `metrics/robust_gpu_YYYYMMDD_HHMMSS/` containing:
- `best_model.pth`: Best model checkpoint
- `checkpoint_epoch_N.pth`: Regular checkpoints every 10 epochs
- `training_history.json`: Complete training metrics
- `evaluation_report.json`: Final evaluation metrics
- `config.json`: Training configuration
- Various visualization PNG files

The best model is also copied to the root directory as `robust_stress_granule_model.pth`.

## Requirements

- PyTorch with GPU support (CUDA or MPS)
- albumentations
- scikit-learn
- OpenCV
- matplotlib
- numpy

## Differences from improved_training_gpu.py

1. **Data Preprocessing**:
   - Multi-scale enhancement vs single-scale
   - Adaptive preprocessing based on image statistics
   - Edge enhancement for boundary detection

2. **Data Augmentation**:
   - Albumentations vs torchvision transforms
   - More diverse augmentation strategies
   - Elastic deformations and occlusion simulation

3. **Training Strategy**:
   - Weighted sampling for balanced training
   - Focal Tversky loss for better class imbalance handling
   - F1 score as primary metric vs Dice score
   - More extensive evaluation metrics

4. **Training Duration**:
   - 200 epochs vs 75 epochs
   - Patience 30 vs 15
   - More crops per image (5 vs 3)

This script prioritizes model robustness and accuracy over training speed, making it ideal when you have time for thorough training and want the best possible model performance. 