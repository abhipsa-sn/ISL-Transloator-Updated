"""
Pipeline script to preprocess dynamic sequences and train an LSTM model.
"""
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import config

RAW_PATH = config.RAW_DATA_PATH
PROC_PATH = config.PROCESSED_DATA_PATH
MODELS_PATH = config.MODELS_PATH

os.makedirs(PROC_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# ===== STEP 1: Load dynamic sequences =====
print("=" * 50)
print("STEP 1: Loading dynamic sequences...")
print("=" * 50)

X_all = []
y_all = []

# Target dynamic classes: explicit selection from config (excludes static-only classes like 'I' and 'You')
dynamic_classes = getattr(config, 'DYNAMIC_CLASSES', None)
if dynamic_classes:
    classes = sorted(dynamic_classes)
else:
    classes = ['Bye', 'Food', 'Hello', 'Help', 'Namaste', 'Sorry', 'Thank You', 'Want', 'Water']
for cls in classes:
    cls_dir = os.path.join(RAW_PATH, cls)
    if not os.path.isdir(cls_dir):
        continue

    dynamic_file = os.path.join(cls_dir, "dynamic_data.npy")
    if os.path.exists(dynamic_file):
        data = np.load(dynamic_file)
        X_all.append(data)
        y_all.extend([cls] * len(data))
        print(f"  {cls}: {data.shape[0]} sequences, sequence_length={data.shape[1]}")
    else:
        print(f"  {cls}: No dynamic data found!")

if not X_all:
    print("Error: No dynamic data found in data/raw/ directories!")
    sys.exit(1)

X = np.vstack(X_all)
y = np.array(y_all)
print(f"\nTotal Dataset Shape: X={X.shape}, y={y.shape}")

# ===== STEP 2: Normalize landmarks frame-by-frame =====
print("\n" + "=" * 50)
print("STEP 2: Normalizing sequence landmarks...")
print("=" * 50)

def normalize_landmarks_sequence(X):
    num_sequences, sequence_length, num_features = X.shape
    X_norm = np.zeros_like(X)
    
    for seq_idx in range(num_sequences):
        for frame_idx in range(sequence_length):
            landmarks = X[seq_idx, frame_idx]
            if np.all(landmarks == 0):
                continue
            
            for i in range(0, 126, 63):
                hand = landmarks[i:i + 63]
                if np.all(hand == 0):
                    continue
                wrist = hand[0:3].copy()
                shifted = hand.copy()
                for j in range(0, 63, 3):
                    shifted[j] -= wrist[0]
                    shifted[j+1] -= wrist[1]
                    shifted[j+2] -= wrist[2]
                distances = [np.sqrt(shifted[j]**2 + shifted[j+1]**2 + shifted[j+2]**2)
                             for j in range(0, 63, 3)]
                max_dist = max(distances) if distances else 0
                if max_dist > 0:
                    shifted /= max_dist
                X_norm[seq_idx, frame_idx, i:i + 63] = shifted
    return X_norm

X = normalize_landmarks_sequence(X)
print("  Landmarks normalized successfully.")

# ===== STEP 3: Encode labels and split train/test =====
print("\n" + "=" * 50)
print("STEP 3: Encoding labels and splitting dynamic dataset...")
print("=" * 50)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

np.save(os.path.join(PROC_PATH, 'X_dynamic_train.npy'), X_train)
np.save(os.path.join(PROC_PATH, 'X_dynamic_test.npy'), X_test)
np.save(os.path.join(PROC_PATH, 'y_dynamic_train.npy'), y_train)
np.save(os.path.join(PROC_PATH, 'y_dynamic_test.npy'), y_test)
np.save(os.path.join(PROC_PATH, 'label_encoder_dynamic.npy'), le.classes_)

print(f"  Classes: {list(le.classes_)}")
print(f"  Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"  Test shape:  X={X_test.shape}, y={y_test.shape}")
print(f"  Saved to {PROC_PATH}")

# ===== STEP 4: Build LSTM Model =====
print("\n" + "=" * 50)
print("STEP 4: Building LSTM neural network...")
print("=" * 50)

num_classes = len(le.classes_)
sequence_length = X_train.shape[1]
num_features = X_train.shape[2]

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===== STEP 5: Train LSTM Model =====
print("\n" + "=" * 50)
print("STEP 5: Training dynamic LSTM model...")
print("=" * 50)

model_save_path = os.path.join(MODELS_PATH, 'dynamic_model.h5')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)

# ===== STEP 6: Evaluation Results =====
print("\n" + "=" * 50)
print("STEP 6: Evaluating dynamic LSTM model...")
print("=" * 50)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report (LSTM):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Dynamic Model Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Dynamic Model Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'dynamic_training_history.png'))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Dynamic Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'dynamic_confusion_matrix.png'))

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\n{'='*50}")
print(f"  FINAL DYNAMIC (LSTM) TEST ACCURACY: {test_acc*100:.2f}%")
print(f"  Dynamic model saved to: {model_save_path}")
print(f"{'='*50}")
