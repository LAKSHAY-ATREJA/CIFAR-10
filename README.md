# CIFAR-10 Image Classification

🖼️ **Project Type:** Deep Learning – Image Classification  
📅 **Original project completion:** November – December 2024  
📤 **Repository updated & published:** February 2026  
📦 **Dataset:** CIFAR-10 (Canadian Institute for Advanced Research)

---

## 📌 Project Overview

This project implements a deep learning–based image classification system using the **CIFAR-10 dataset**, a widely used benchmark in computer vision research and education.

The objective is to train and evaluate convolutional neural networks (CNNs) capable of classifying small RGB images into one of ten object categories. The project demonstrates the complete workflow for building, training, and evaluating deep learning models for image recognition tasks.

The core implementation was completed in **late 2024** as part of structured self-learning in deep learning.  
The repository was **cleaned, documented, and published in February 2026** for portfolio and demonstration purposes.

---

## 🧠 Problem Statement

Given a 32×32 RGB image, predict which of the following 10 classes it belongs to:

- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

---

## 🎯 Project Objectives

- Understand and preprocess image data for deep learning  
- Design and train convolutional neural networks  
- Apply data normalization and augmentation techniques  
- Evaluate model performance using accuracy and loss metrics  
- Analyse model behaviour across different object classes  

---

## 🛠️ Technologies & Libraries Used

- **Python**
- **TensorFlow / Keras** *(or PyTorch – adjust if needed)*
- **NumPy**
- **Matplotlib**
- **Scikit-learn** (evaluation utilities)

---

## 🔍 Methodology

### 1. Dataset Preparation
- Loaded CIFAR-10 dataset using built-in deep learning utilities
- Normalised pixel values to improve convergence
- Split data into training and test sets

### 2. Model Architecture
- Convolutional Neural Network (CNN) with:
  - Convolutional layers
  - ReLU activations
  - Max pooling layers
  - Fully connected layers
  - Softmax output layer

### 3. Training
- Optimiser: Adam
- Loss function: Categorical Cross-Entropy
- Batch training with validation monitoring

### 4. Evaluation
- Accuracy and loss evaluation on test dataset
- Confusion matrix and class-wise performance analysis
- Visual inspection of predictions on sample images

---

## 📊 Results Summary

- **Final test accuracy:** ~70–80% (depending on architecture and training configuration)
- The model performs well on structured objects (e.g. ships, automobiles)
- Lower accuracy observed for visually similar classes (e.g. cats vs dogs)

These results are consistent with standard CNN baselines on the CIFAR-10 dataset.

---

## 📂 Repository Structure



├── data/
│ └── cifar10/
├── notebooks/
│ └── cifar10_classification.ipynb
├── models/
│ └── saved_model/
├── images/
│ └── sample_predictions.png
├── requirements.txt
├── README.md




---

## 📝 Project History

- **Late 2024**
  - Implemented CNN-based classification pipeline
  - Trained and evaluated models on CIFAR-10
  - Experimented with architecture depth and hyperparameters

- **February 2026**
  - Refactored codebase
  - Added documentation and visualisations
  - Published repository publicly for portfolio use

---

## ⚠️ Disclaimer

This project is intended as a **learning and demonstration project** using a standard computer vision benchmark dataset.  
It is **not presented as novel research or production-grade software**.

---

## 📚 Dataset Reference

CIFAR-10 Dataset  
Canadian Institute for Advanced Research  
https://www.cs.toronto.edu/~kriz/cifar.html
