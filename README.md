# CIFAR-10 Image Classification

A convolutional neural network trained to classify 32x32 RGB images into one of ten categories using the CIFAR-10 benchmark dataset. Implemented in TensorFlow/Keras with data augmentation, batch normalisation, and dropout regularisation. A second experiment applies transfer learning via DenseNet121 pre-trained on ImageNet.

Implemented in late 2024 as part of structured self-study in deep learning. Repository published in 2026 for portfolio use.


## What the project does

The project trains a custom three-block CNN to recognise images from the CIFAR-10 dataset, which contains 60,000 colour photographs across ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The training pipeline covers data loading, normalisation, on-the-fly augmentation, model training with callbacks, and full evaluation including a confusion matrix and per-class classification report.

The project includes a Jupyter notebook for interactive exploration and a standalone command-line script (`demo.py`) that reproduces the same training and evaluation steps without requiring a notebook environment.


## Dataset

CIFAR-10 (Canadian Institute for Advanced Research) is a standard computer vision benchmark. It contains:

- 50,000 training images and 10,000 test images
- Resolution: 32x32 pixels, 3 RGB channels
- 10 balanced classes with 6,000 images each

The dataset is downloaded automatically by Keras on first run (approximately 170 MB). No manual download is required.


## Model architecture

The custom CNN uses three convolutional blocks followed by a fully connected classifier:

```
Input: (32, 32, 3)

Block 1
  Conv2D(32, 3x3, relu, same)
  BatchNormalization
  Conv2D(32, 3x3, relu, same)
  BatchNormalization
  MaxPool2D(2x2)
  Dropout(0.25)

Block 2
  Conv2D(64, 3x3, relu, same)
  BatchNormalization
  Conv2D(64, 3x3, relu, same)
  BatchNormalization
  MaxPool2D(2x2)
  Dropout(0.25)

Block 3
  Conv2D(128, 3x3, relu, same)
  BatchNormalization
  Conv2D(128, 3x3, relu, same)
  BatchNormalization
  MaxPool2D(2x2)
  Dropout(0.25)

Classifier
  Flatten
  Dense(128, relu)
  Dropout(0.25)
  Dense(10, softmax)
```

Total trainable parameters: approximately 551,000.

The model is compiled with the Adam optimiser, categorical cross-entropy loss, and three tracked metrics: accuracy, precision, and recall.

A secondary experiment fine-tunes DenseNet121 (pre-trained on ImageNet) by freezing the convolutional base and training only a Dense(10) softmax head. This serves as a comparison point for transfer learning at low resolution.


## Training procedure

Data augmentation is applied on-the-fly using `ImageDataGenerator`:

- Horizontal flip (probability 0.5)
- Width shift up to 10% of image width
- Height shift up to 10% of image height

Augmentation is applied only to training images. Test images are evaluated without modification.

Two callbacks regulate training:

- EarlyStopping monitors validation loss and halts training if it does not improve for five consecutive epochs, restoring the best weights found.
- ReduceLROnPlateau halves the learning rate when validation loss stagnates for three consecutive epochs, down to a minimum of 1e-6.


## Features

- Full end-to-end pipeline from raw data download to evaluation
- Three-block CNN with batch normalisation at every convolutional layer
- On-the-fly data augmentation via ImageDataGenerator
- Training callbacks: early stopping and learning rate scheduling
- Evaluation: loss/accuracy curves, confusion matrix, per-class classification report
- Inference demo: single-image prediction and a 40-image prediction grid with confidence bar charts
- Transfer learning experiment: DenseNet121 fine-tuning
- Command-line demo script with configurable epochs, batch size, and augmentation toggle
- All outputs (plots, classification report, saved model) written to a configurable output directory


## Expected results

| Model | Test accuracy |
|---|---|
| Custom 3-block CNN (50 epochs, augmented) | 75-85% |
| DenseNet121 head-only (20 epochs) | 60-75% |

Exact values vary with random seed and hardware. The custom CNN is consistent with published baselines for CNNs of this depth on CIFAR-10 without advanced techniques such as residual connections, label smoothing, or cosine annealing.

Common misclassifications occur between visually similar pairs: cat/dog, automobile/truck, and bird/airplane.


## Requirements

- Python 3.9 or later
- TensorFlow 2.13 or later (includes Keras)
- NumPy 1.24 or later
- Matplotlib 3.7 or later
- scikit-learn 1.3 or later
- Pillow 9.0 or later
- pandas 2.0 or later (used in the notebook)

For the complete list with minimum version pins, see `requirements.txt`.


## Installation

```bash
git clone https://github.com/LAKSHAY-ATREJA/CIFAR-10.git
cd CIFAR-10

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```


## Running the notebook

```bash
jupyter notebook CIFAR.ipynb
```

Run all cells from top to bottom. The notebook downloads CIFAR-10 automatically on first run. Training 50 epochs takes roughly 15-30 minutes on a CPU and 3-5 minutes on a GPU.

The notebook is structured as follows:

| Section | Description |
|---|---|
| 1. Install dependencies | Installs required packages via subprocess |
| 2. Imports | Loads all libraries and sets random seeds |
| 3. Load and inspect data | Downloads CIFAR-10, prints shapes |
| 4. Exploratory analysis | 10x10 image grid, class distribution bar chart |
| 5. Preprocessing | Pixel normalisation and one-hot encoding |
| 6. Model definition | Three-block CNN built with the Keras functional API |
| 7. Training | Up to 50 epochs with augmentation and callbacks |
| 8. Training curves | Loss, accuracy, precision, and recall plots |
| 9. Evaluation | Confusion matrix and classification report |
| 10. Inference demo | Single-image and 40-image prediction grids |
| 11. Transfer learning | DenseNet121 fine-tuning experiment |
| 12. Summary | Result table and notes on further improvements |


## Running the demo script

The demo script trains the model and writes all plots and evaluation outputs to a directory without requiring Jupyter.

```bash
python demo.py
```

Optional arguments:

```
--epochs INT        Maximum training epochs (default: 30)
--batch-size INT    Mini-batch size (default: 32)
--no-augment        Disable data augmentation
--patience INT      Early stopping patience in epochs (default: 5)
--output-dir PATH   Directory for saved plots and model (default: outputs)
--seed INT          Random seed (default: 42)
```

Example: a quick run with 10 epochs and no augmentation:

```bash
python demo.py --epochs 10 --no-augment --output-dir quick_run
```

Outputs written to the output directory:

- `training_curves.png` - loss, accuracy, precision, recall over epochs
- `confusion_matrix.png` - 10x10 confusion matrix on the test set
- `sample_predictions.png` - 25 random test images with predicted and true labels
- `classification_report.txt` - per-class precision, recall, and F1-score
- `cifar10_cnn.keras` - the trained model in Keras format


## Repository structure

```
CIFAR-10/
  CIFAR.ipynb          Jupyter notebook with full pipeline and explanations
  demo.py              Command-line training and evaluation script
  requirements.txt     Package dependencies with minimum version pins
  README.md            This file
```


## License

MIT. Built by Lakshay Atreja.
