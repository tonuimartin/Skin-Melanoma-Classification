# Skin-Melanoma-Classification
# Melanoma Classification using Deep Learning

A deep learning model for binary classification of skin lesions as either benign or malignant melanoma using TensorFlow and ResNet50V2 architecture.

## Features

- Binary classification of skin lesions (benign/malignant)
- Built on ResNet50V2 architecture with transfer learning
- Two-phase training process with fine-tuning
- Data augmentation for improved model robustness
- Comprehensive evaluation metrics including ROC curves and confusion matrices
- Support for both single image and batch predictions
- TFLite model conversion for mobile deployment

## Project Structure

```
├── config.py           # Configuration parameters
├── data_loader.py      # Data loading and augmentation utilities
├── model.py            # Model architecture definition
├── train.py           # Training script with two-phase training
├── evaluate.py        # Model evaluation utilities
├── predict.py         # Prediction utilities for deployed model
└── run_prediction.py  # Script to run various types of predictions
```

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- scikit-learn
- seaborn
- matplotlib
- pandas

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install tensorflow numpy pillow scikit-learn seaborn matplotlib pandas
```

## Dataset Organization

The dataset should be organized in the following structure:
```
data/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## Model Architecture

- Base Model: ResNet50V2 (pretrained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Batch Normalization
  - Dense layers (512 and 256 units) with ReLU activation
  - Dropout layers (0.3)
  - Final Dense layer with sigmoid activation

## Training Process

The model uses a two-phase training approach:

1. First Phase:
   - Base model layers are frozen
   - Only top layers are trained
   - Initial learning rate: 1e-3

2. Fine-tuning Phase:
   - Base model is unfrozen
   - Full model fine-tuning
   - Lower learning rate: 1e-5

## Usage

### Training

To train the model:
```bash
python train.py
```

To resume training from a checkpoint:
```bash
python train.py --resume
```

### Evaluation

To evaluate the trained model:
```bash
python evaluate.py
```

This will generate:
- ROC curve plot
- Confusion matrix
- Detailed classification metrics

### Making Predictions

1. Single image prediction:
```bash
python predict.py path/to/image.jpg
```

2. Batch prediction:
```bash
python predict.py path/to/directory --batch
```

3. Full test directory evaluation:
```bash
python predict.py path/to/test_directory --evaluate
```

## Model Performance

The model is evaluated using multiple metrics:
- Accuracy
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- ROC curve and AUC score
- Confusion matrix

## Mobile Deployment

The model is automatically converted to TFLite format with float16 quantization for mobile deployment. The converted model is saved as 'melanoma_classifier.tflite'.

## Configuration

Key parameters can be modified in `config.py`:
- Image dimensions: 299x299
- Batch size: 16
- Training epochs: 150
- Learning rates: 1e-3 (initial), 1e-5 (fine-tuning)
- Data augmentation parameters

