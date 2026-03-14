"""
demo.py - CIFAR-10 image classification demo script.

Trains the three-block CNN on CIFAR-10 and shows predictions on a sample of
test images. Produces the same results as the CIFAR.ipynb notebook, but can
be run from the command line without Jupyter.

Usage:
    python demo.py                      # train and evaluate (default)
    python demo.py --epochs 10          # quick run with fewer epochs
    python demo.py --no-augment         # disable data augmentation

Requirements: see requirements.txt
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # headless backend so the script works on servers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate a CNN on CIFAR-10.'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Maximum number of training epochs (default: 30).'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Mini-batch size for training (default: 32).'
    )
    parser.add_argument(
        '--no-augment', action='store_true',
        help='Disable data augmentation (horizontal flip and shifts).'
    )
    parser.add_argument(
        '--patience', type=int, default=5,
        help='Early stopping patience in epochs (default: 5).'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Directory to save plots and the trained model (default: outputs).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_data():
    """Load CIFAR-10, normalise pixels, and one-hot encode labels."""
    import tensorflow as tf
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    print("Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples:     {X_test.shape[0]:,}")
    print(f"  Image shape:      {X_train.shape[1:]}  (height x width x channels)")

    # Normalise to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    y_cat_train = to_categorical(y_train, num_classes=10)
    y_cat_test  = to_categorical(y_test,  num_classes=10)

    return X_train, X_test, y_train, y_test, y_cat_train, y_cat_test


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model():
    """
    Build the three-block CNN:
      Block 1: Conv2D(32) x2 + BatchNorm + MaxPool + Dropout(0.25)
      Block 2: Conv2D(64) x2 + BatchNorm + MaxPool + Dropout(0.25)
      Block 3: Conv2D(128) x2 + BatchNorm + MaxPool + Dropout(0.25)
      Head:    Flatten -> Dense(128, ReLU) -> Dropout(0.25) -> Dense(10, Softmax)

    Total trainable parameters: ~551,000
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, Conv2D, BatchNormalization, MaxPool2D, Dropout,
        Flatten, Dense
    )

    model = Sequential([
        Input(shape=(32, 32, 3)),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classifier head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(10, activation='softmax'),
    ], name='cifar10_cnn')

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_cat_train, X_test, y_cat_test, args):
    """Train the model with optional data augmentation and early stopping."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    steps_per_epoch = X_train.shape[0] // args.batch_size

    if not args.no_augment:
        print("Data augmentation enabled (horizontal flip, width/height shift 10%).")
        data_gen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        train_data = data_gen.flow(X_train, y_cat_train, batch_size=args.batch_size)
    else:
        print("Data augmentation disabled.")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        data_gen = ImageDataGenerator()
        train_data = data_gen.flow(X_train, y_cat_train, batch_size=args.batch_size)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print(f"\nTraining for up to {args.epochs} epochs "
          f"({steps_per_epoch} steps/epoch, batch size {args.batch_size})...")
    print("Early stopping enabled with patience =", args.patience)
    print()

    history = model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_cat_test),
        callbacks=callbacks,
        verbose=1
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation and plotting
# ---------------------------------------------------------------------------

def evaluate_and_plot(model, history, X_test, y_test, y_cat_test, output_dir):
    """Evaluate the model and save plots to output_dir."""
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    os.makedirs(output_dir, exist_ok=True)

    # ---- Overall metrics ----
    results = model.evaluate(X_test, y_cat_test, verbose=0)
    metric_names = model.metrics_names
    print("\nTest set results:")
    for name, value in zip(metric_names, results):
        print(f"  {name:12s}: {value:.4f}")

    # ---- Predictions ----
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    # ---- Training curves ----
    hist = history.history
    epochs_ran = range(1, len(hist['loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for (train_key, val_key, title), ax in zip(
        [
            ('loss',      'val_loss',      'Loss'),
            ('accuracy',  'val_accuracy',  'Accuracy'),
            ('precision', 'val_precision', 'Precision'),
            ('recall',    'val_recall',    'Recall'),
        ],
        axes.flatten()
    ):
        if train_key in hist:
            ax.plot(epochs_ran, hist[train_key], label=f'Train {title}', linewidth=2)
        if val_key in hist:
            ax.plot(epochs_ran, hist[val_key], label=f'Val {title}',
                    linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} over epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nTraining curves saved to: {curves_path}")

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(xticks_rotation='vertical', ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix on Test Set')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # ---- Classification report ----
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        for name, value in zip(metric_names, results):
            f.write(f"{name:12s}: {value:.4f}\n")
        f.write("\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    # ---- Sample predictions grid ----
    rng = np.random.default_rng(seed=7)
    sample_indices = rng.integers(0, len(X_test), 25)

    fig, axes = plt.subplots(5, 5, figsize=(13, 13))
    axes = axes.ravel()

    for plot_idx, img_idx in enumerate(sample_indices):
        axes[plot_idx].imshow(X_test[img_idx])
        pred_label = CLASS_NAMES[int(y_pred[img_idx])]
        true_label = CLASS_NAMES[int(y_test[img_idx])]
        correct = y_pred[img_idx] == int(y_test[img_idx])
        color = 'green' if correct else 'red'
        axes[plot_idx].set_title(
            f"Pred: {pred_label}\nTrue: {true_label}",
            fontsize=8, color=color
        )
        axes[plot_idx].axis('off')

    plt.suptitle('Sample test predictions (green=correct, red=incorrect)', fontsize=12)
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    preds_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(preds_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions grid saved to: {preds_path}")

    return results[1]   # return test accuracy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Import TF here so the script can display help without TF loading
    import tensorflow as tf

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    print(f"TensorFlow version: {tf.__version__}")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Training on CPU (this will be slow for many epochs).")

    print()

    # Load data
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = load_data()

    # Build model
    print("\nBuilding CNN model...")
    model = build_model()
    model.summary()

    # Train
    history = train_model(model, X_train, y_cat_train, X_test, y_cat_test, args)

    # Evaluate and save plots
    test_accuracy = evaluate_and_plot(
        model, history, X_test, y_test, y_cat_test, args.output_dir
    )

    # Save model
    model_path = os.path.join(args.output_dir, 'cifar10_cnn.keras')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    print(f"\nFinal test accuracy: {test_accuracy * 100:.2f}%")
    print(f"All outputs written to: {args.output_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()
