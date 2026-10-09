import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

try:
    from src import config
except ImportError:
    try:
        import config
    except ImportError:
        # Fallback config
        class Config:
            PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
            MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            EPOCHS = 50
            BATCH_SIZE = 32
        config = Config()

def build_model(num_features, num_classes):
    """
    Builds a Dense Neural Network for static gesture recognition.
    """
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
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
    Loads data, trains the model, plots curves, and saves artifacts.
    """
    if epochs is None:
        epochs = getattr(config, 'EPOCHS', 50)
    if batch_size is None:
        batch_size = getattr(config, 'BATCH_SIZE', 32)
        
    proc_path = config.PROCESSED_DATA_PATH
    models_path = config.MODELS_PATH
    os.makedirs(models_path, exist_ok=True)
    
    print(f"Loading data from {proc_path}...")
    try:
        X_train = np.load(os.path.join(proc_path, 'X_train.npy'))
        X_test = np.load(os.path.join(proc_path, 'X_test.npy'))
        y_train = np.load(os.path.join(proc_path, 'y_train.npy'))
        y_test = np.load(os.path.join(proc_path, 'y_test.npy'))
        classes = np.load(os.path.join(proc_path, 'label_encoder.npy'), allow_pickle=True)
    except Exception as e:
        print(f"Error loading processed data: {e}. Please run preprocess.py first.")
        return None, None

    num_classes = len(classes)
    num_features = X_train.shape[1]
    
    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Number of classes: {num_classes}")
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    model = build_model(num_features, num_classes)
    
    model_save_path = os.path.join(models_path, 'static_model.h5')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    history_path = os.path.join(models_path, 'training_history.png')
    plt.savefig(history_path)
    print(f"Training history plot saved to {history_path}")
    plt.close()
    
    # Evaluate
    print("Evaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(models_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    plt.close()
    
    print(f"Model successfully trained and saved to {model_save_path}")
    return model, history

def evaluate_model(model_path):
    """
    Loads saved model and test data, prints accuracy and confusion matrix.
    """
    print(f"Evaluating saved model at {model_path}...")
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
        
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    
    proc_path = config.PROCESSED_DATA_PATH
    try:
        X_test = np.load(os.path.join(proc_path, 'X_test.npy'))
        y_test = np.load(os.path.join(proc_path, 'y_test.npy'))
        classes = np.load(os.path.join(proc_path, 'label_encoder.npy'), allow_pickle=True)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
        
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Evaluation)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate static gesture recognition model.")
    parser.add_argument('--epochs', type=int, default=getattr(config, 'EPOCHS', 50), help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=getattr(config, 'BATCH_SIZE', 32), help='Batch size')
    parser.add_argument('--no-augment', action='store_true', help='Skip data augmentation (handled in preprocess)')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate existing model')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(os.path.join(getattr(config, 'MODELS_PATH', 'models'), 'static_model.h5'))
    else:
        # Note: --no-augment flag is provided in args but preprocessing is typically run separately. 
        # If we wanted to run preprocessing here based on that flag we could, but typically it's run via preprocess.py
        if not args.no_augment:
            print("Note: Augmentation is handled during preprocessing step.")
        train_model(epochs=args.epochs, batch_size=args.batch_size)
