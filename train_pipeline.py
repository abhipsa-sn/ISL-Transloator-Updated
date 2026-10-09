"""
Quick pipeline script: Convert dynamic data → static format, preprocess, and train.
"""
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import config

RAW_PATH = config.RAW_DATA_PATH
PROC_PATH = config.PROCESSED_DATA_PATH
os.makedirs(PROC_PATH, exist_ok=True)

# ===== STEP 1: Load dynamic data and extract individual frames =====
print("=" * 50)
print("STEP 1: Loading collected data...")
print("=" * 50)

X_all = []
y_all = []

classes = sorted(os.listdir(RAW_PATH))
for cls in classes:
    cls_dir = os.path.join(RAW_PATH, cls)
    if not os.path.isdir(cls_dir):
        continue

    dynamic_file = os.path.join(cls_dir, "dynamic_data.npy")
    static_file = os.path.join(cls_dir, "static_data.npy")

    if os.path.exists(dynamic_file):
        data = np.load(dynamic_file)
        # Shape: (num_sequences, sequence_length, 126)
        # Flatten to individual frames: (num_sequences * sequence_length, 126)
        num_seq, seq_len, features = data.shape
        frames = data.reshape(-1, features)
        # Remove zero frames (where no hands were detected)
        non_zero = np.any(frames != 0, axis=1)
        frames = frames[non_zero]
        X_all.append(frames)
        y_all.extend([cls] * len(frames))
        print(f"  {cls}: {num_seq} sequences -> {len(frames)} usable frames")
    elif os.path.exists(static_file):
        data = np.load(static_file)
        X_all.append(data)
        y_all.extend([cls] * len(data))
        print(f"  {cls}: {len(data)} static frames")

X = np.vstack(X_all)
y = np.array(y_all)
print(f"\nTotal: {len(X)} samples across {len(set(y))} classes")

# ===== STEP 2: Normalize landmarks =====
print("\n" + "=" * 50)
print("STEP 2: Normalizing landmarks...")
print("=" * 50)

def normalize_landmarks(X):
    """Normalize each sample's landmarks relative to wrist position."""
    X_norm = np.zeros_like(X)
    for idx in range(len(X)):
        landmarks = X[idx]
        for i in range(0, 126, 63):
            hand = landmarks[i:i + 63]
            if np.all(hand == 0):
                continue
            wrist = hand[0:3].copy()
            shifted = hand.copy()
            for j in range(0, 63, 3):
                shifted[j] -= wrist[0]
                shifted[j + 1] -= wrist[1]
                shifted[j + 2] -= wrist[2]
            distances = [np.sqrt(shifted[j]**2 + shifted[j+1]**2 + shifted[j+2]**2)
                         for j in range(0, 63, 3)]
            max_dist = max(distances) if distances else 0
            if max_dist > 0:
                shifted /= max_dist
            X_norm[idx, i:i + 63] = shifted
    return X_norm

X = normalize_landmarks(X)
print(f"  Normalized {len(X)} samples")

# ===== STEP 3: Augment data =====
print("\n" + "=" * 50)
print("STEP 3: Augmenting data...")
print("=" * 50)

augmentation_factor = 3
X_aug = [X]
y_aug = [y]
for _ in range(augmentation_factor - 1):
    noise = np.random.normal(0, 0.01, X.shape)
    scale = np.random.uniform(0.9, 1.1, X.shape)
    translation = np.random.normal(0, 0.02, X.shape)
    X_new = (X * scale) + noise + translation
    X_aug.append(X_new)
    y_aug.append(y)

X = np.vstack(X_aug)
y = np.concatenate(y_aug)
print(f"  After {augmentation_factor}x augmentation: {len(X)} samples")

# ===== STEP 4: Encode labels and split =====
print("\n" + "=" * 50)
print("STEP 4: Encoding labels and splitting data...")
print("=" * 50)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Save processed data (6-class specific)
np.save(os.path.join(PROC_PATH, 'X_train_6class.npy'), X_train)
np.save(os.path.join(PROC_PATH, 'X_test_6class.npy'), X_test)
np.save(os.path.join(PROC_PATH, 'y_train_6class.npy'), y_train)
np.save(os.path.join(PROC_PATH, 'y_test_6class.npy'), y_test)
np.save(os.path.join(PROC_PATH, 'label_encoder_6class.npy'), le.classes_)

print(f"  Classes: {list(le.classes_)}")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")
print(f"  Saved to {PROC_PATH}")

# ===== STEP 5: Train model =====
print("\n" + "=" * 50)
print("STEP 5: Training model...")
print("=" * 50)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

num_classes = len(le.classes_)
num_features = X_train.shape[1]

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

models_path = config.MODELS_PATH
os.makedirs(models_path, exist_ok=True)
model_save_path = os.path.join(models_path, 'static_model_6class.h5')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# ===== STEP 6: Evaluate =====
print("\n" + "=" * 50)
print("STEP 6: Evaluation Results")
print("=" * 50)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save training plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(models_path, 'training_history_6class.png'))
print(f"Training plot saved to {models_path}/training_history_6class.png")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(models_path, 'confusion_matrix_6class.png'))
print(f"Confusion matrix saved to {models_path}/confusion_matrix_6class.png")

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\n{'='*50}")
print(f"  FINAL TEST ACCURACY: {test_acc*100:.2f}%")
print(f"  Model saved to: {model_save_path}")
print(f"{'='*50}")
print("\nYou can now run inference:")
print("  python src/inference.py --mode static")
