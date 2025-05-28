# Stress Granule Analysis with Neural Networks

This project uses deep learning to analyze stress granules from microscopy data. It implements a U-Net architecture to segment stress granules in microscopy images.

## Features

- U-Net implementation for semantic segmentation of stress granules
- Custom dataset class for handling microscopy images and masks
- Training pipeline with Dice coefficient and BCE loss
- Visualization tools for model predictions
- Model checkpoint saving and loading

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- Matplotlib

## Setup

1. Clone this repository
2. Create a virtual environment (recommended): `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Data Organization

Organize your data as follows:
```
project/
├── data/
│   ├── images/
│   │   ├── img1.tif
│   │   ├── img2.tif
│   │   └── ...
│   └── masks/
│       ├── mask1.tif
│       ├── mask2.tif
│       └── ...
```

## Usage

1. Update the image and mask directories in `main.py`
2. Run the training script: `python main.py`

## Model Architecture

The project uses a U-Net architecture with the following structure:
- Encoder: 4 downsampling blocks with double convolution
- Bottleneck: Double convolution
- Decoder: 4 upsampling blocks with skip connections

## Evaluation

The model is evaluated using the Dice coefficient, which measures the overlap between the predicted segmentation and ground truth.

## License

[MIT License](LICENSE) 