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