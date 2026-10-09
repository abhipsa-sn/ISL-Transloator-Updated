"""
Training script for dynamic (sequence-based) gesture recognition using LSTM.
Uses sequences of hand landmark data to classify dynamic ISL signs.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    from src import config
except ImportError:
    try:
        import config
    except ImportError:
        class _FallbackConfig:
            PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
            MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            EPOCHS = 50
            BATCH_SIZE = 32
            TEST_SIZE = 0.2
            RANDOM_STATE = 42
            SEQUENCE_LENGTH = 30
            NUM_FEATURES_BOTH_HANDS = 126
        config = _FallbackConfig()


def build_lstm_model(sequence_length, num_features, num_classes):
    """
    Build and compile an LSTM model for dynamic gesture recognition.

    Architecture:
        LSTM(128, return_sequences=True) → Dropout(0.3)
        LSTM(64) → Dropout(0.3)
        Dense(64, relu) → BatchNormalization
        Dense(32, relu)
        Dense(num_classes, softmax)
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def train_model(epochs=None, batch_size=None):
    """
    Train the dynamic gesture recognition model.
    Loads preprocessed dynamic data, trains LSTM, saves model and plots.
    """
    if epochs is None:
        epochs = getattr(config, 'EPOCHS', 50)
    if batch_size is None:
        batch_size = getattr(config, 'BATCH_SIZE', 32)

    proc_path = getattr(config, 'PROCESSED_DATA_PATH',
                        str(getattr(config, 'DATA_PROCESSED_DIR', 'data/processed')))
    models_path = getattr(config, 'MODELS_PATH',
                          str(getattr(config, 'MODELS_DIR', 'models')))
    os.makedirs(models_path, exist_ok=True)

    print(f"Loading dynamic data from {proc_path}...")
    try:
        X = np.load(os.path.join(proc_path, 'X_dynamic_train.npy'))
        X_test = np.load(os.path.join(proc_path, 'X_dynamic_test.npy'))
        y = np.load(os.path.join(proc_path, 'y_dynamic_train.npy'))
        y_test = np.load(os.path.join(proc_path, 'y_dynamic_test.npy'))
        classes = np.load(os.path.join(proc_path, 'label_encoder_dynamic.npy'), allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run preprocessing for dynamic data first.")
        print("Run: python src/preprocess.py --data-type dynamic")
        return None, None

    num_classes = len(classes)
    sequence_length = X.shape[1]
    num_features = X.shape[2]

    print(f"Dataset shape: X_train={X.shape}, X_test={X_test.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Sequence length: {sequence_length}, Features: {num_features}")

    # One-hot encode labels
    y_cat = to_categorical(y, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Build model
    model = build_lstm_model(sequence_length, num_features, num_classes)

    # Callbacks
    model_path = os.path.join(models_path, 'dynamic_model.h5')
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss', verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, monitor='val_loss', verbose=1)
    ]

    # Train
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        X, y_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Evaluate
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=classes))

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    history_path = os.path.join(models_path, 'dynamic_training_history.png')
    plt.savefig(history_path)
    print(f"Training history plot saved to {history_path}")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Dynamic Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(models_path, 'dynamic_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    plt.close()

    print(f"\nModel successfully saved to {model_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dynamic gesture recognition model (LSTM)")
    parser.add_argument("--epochs", type=int, default=getattr(config, 'EPOCHS', 50), help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=getattr(config, 'BATCH_SIZE', 32), help="Batch size")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size)
