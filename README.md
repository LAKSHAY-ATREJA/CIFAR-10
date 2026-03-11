# CIFAR-10 Image Classification

A convolutional neural network implementation for classifying 32x32 RGB images into one of ten object categories using the CIFAR-10 benchmark dataset.

Implemented in late 2024 as part of structured self-learning in deep learning. Repository published in 2026 for portfolio use.

## Dataset

CIFAR-10 (Canadian Institute for Advanced Research) contains 60,000 colour images across ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is split into 50,000 training and 10,000 test images. It is downloaded automatically by Keras on first run.

## Architecture

The custom CNN uses three convolutional blocks followed by a fully connected classifier:

- Block 1: two Conv2D(32) layers with BatchNorm and 0.25 Dropout
- Block 2: two Conv2D(64) layers with BatchNorm and 0.25 Dropout
- Block 3: two Conv2D(128) layers with BatchNorm and 0.25 Dropout
- Dense(128) with ReLU and 0.25 Dropout
- Dense(10) with Softmax output

The model is trained with the Adam optimiser, categorical cross-entropy loss, and data augmentation (horizontal flips and small shifts) via ImageDataGenerator. A secondary experiment with DenseNet121 (pretrained on ImageNet) is also included in the notebook.

## Results

The custom CNN achieves approximately 75-82% test accuracy after 50 epochs, depending on random initialisation. Results are consistent with standard CNN baselines on CIFAR-10 without advanced techniques such as residual connections or label smoothing.

## Requirements

- Python 3.9 or later
- See requirements.txt for package versions

## Running the notebook

```bash
git clone https://github.com/LAKSHAY-ATREJA/CIFAR-10.git
cd CIFAR-10

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook CIFAR.ipynb
```

The notebook downloads the CIFAR-10 dataset automatically on first run (approximately 170 MB). Training 50 epochs takes roughly 15-30 minutes on a CPU and 3-5 minutes with a GPU.

## Notebook structure

| Section                   | Description                                            |
|---------------------------|--------------------------------------------------------|
| Data loading              | Download CIFAR-10, inspect shapes                      |
| Exploratory visualisation | 10x10 image grid, class distribution bar chart         |
| Preprocessing             | Pixel normalisation to [0,1], one-hot encoding         |
| Model definition          | Custom 3-block CNN with BatchNorm and Dropout          |
| Training                  | 50 epochs with data augmentation                       |
| Evaluation                | Loss/accuracy curves, confusion matrix, classification report |
| Inference demo            | Per-image prediction with confidence bar charts        |
| Transfer learning         | DenseNet121 fine-tuning experiment                     |

## Known bugs fixed

- Corrected `X_train[index, 1:]` to `X_train[index]` in the image grid cells. The original slice dropped the first pixel row of every displayed image without affecting model training.
- Replaced bare `pip install` magic command with `subprocess.check_call` for compatibility across notebook environments.

## License

MIT. Built by Lakshay Atreja.
