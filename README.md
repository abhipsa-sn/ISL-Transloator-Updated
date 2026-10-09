# 🤟 Indian Sign Language (ISL) Translator

A real-time Indian Sign Language detection and translation system that uses a webcam to recognize hand gestures, builds sentences, and converts them into fluent English using an LLM — with text-to-speech output.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Hand Landmark Detection** | MediaPipe HandLandmarker — 21 landmarks per hand (126 features for two hands) |
| **Static Gesture Recognition** | Dense Neural Network — recognizes held/static signs (6-class model) |
| **Dynamic Gesture Recognition** | LSTM — recognizes motion-based signs using 30-frame sequences |
| **Sentence Builder** | Accumulates detected signs into a word list with gesture debouncing |
| **LLM Grammar Correction** | Gemini 3.6 Flash API converts ISL word sequences (SOV order) into fluent English sentences |
| **Text-to-Speech** | Windows SAPI (SpVoice via win32com) — offline, zero-latency, non-blocking |
| **Desktop GUI** | CustomTkinter dark-mode UI with live webcam feed, confidence bar, history log |

---

## 🗂️ Project Structure

```
ISL-Transloator-updated/
├── app.py                    # Desktop GUI application (CustomTkinter)
├── run_app.bat               # One-click Windows launcher
├── .env                      # API keys (GEMINI_API_KEY)
├── requirements.txt
│
├── src/
│   ├── config.py             # Central config — paths, hyperparameters, class lists
│   ├── data_collection.py    # Interactive webcam + MediaPipe data capture tool
│   ├── preprocess.py         # Normalization, augmentation, train/test split
│   ├── train_static.py       # Train Dense NN for static gestures
│   ├── train_dynamic.py      # Train LSTM for dynamic gestures
│   ├── inference.py          # Real-time CLI inference with TTS
│   ├── llm_translator.py     # Gemini API grammar correction (ISL-aware prompt)
│   └── tts.py                # Windows SAPI text-to-speech engine
│
├── models/
│   ├── hand_landmarker.task  # MediaPipe hand landmark model
│   ├── static_model_6class.h5
│   ├── dynamic_model.h5
│   └── *.png                 # Training history & confusion matrix charts
│
└── data/
    ├── raw/                  # Raw landmark arrays per gesture class
    └── processed/            # Merged, normalized, split datasets + label encoders
```

---

## 🚀 Quick Start

### 1. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file in the project root (or edit the existing one):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey). The app still works without it — grammar correction falls back to joining the raw detected words.

### 3. Launch the Application

- **Double-click** `run_app.bat` (Windows)
- **Or** run from terminal:
  ```bash
  python app.py
  ```

---

## 🖥️ Desktop GUI — How to Use

1. **Start Camera** — Click the Start button to begin real-time recognition
2. **Sign gestures** in front of the webcam — detected words accumulate in the Sentence box
3. **✨ Translate** — Sends the accumulated words to Gemini to produce a grammatically correct English sentence (displayed in the box)
4. **🔊 Speak** — Runs grammar correction and speaks the corrected sentence aloud via TTS
5. **🗑 Clear** — Clears the sentence buffer and resumes live word detection
6. **Mode toggle** — Switch between `Static` (held signs) and `Dynamic` (motion signs) mode

> The Sentence box shows raw detected words during signing, and locks to the corrected translation after clicking Translate or Speak. It resets on Clear.

---

## 🧠 Gesture Classes

### Static Model (6 classes)
`Hello` · `Namaste` · `I` · `You` · `Bye` · `Thank You` · `Sorry`


### Dynamic Model (9 classes)
`Bye` · `Food` · `Hello` · `Namaste` · `Sorry` · `Thank You` · `Want` · `Water`

---

## 🔧 Full Pipeline (Train From Scratch)

### Step 1 — Collect Training Data
```bash
python src/data_collection.py
```
Interactive menu lets you record gestures class-by-class. Saves raw landmark `.npy` arrays.
- **Static:** 500 samples per class
- **Dynamic:** 50 sequences × 30 frames per class

### Step 2 — Preprocess
```bash
python src/preprocess.py
```
Normalizes landmarks (wrist-relative, scale-invariant), augments data, and splits into train/test sets.

### Step 3 — Train Models
```bash
# Static gesture model (Dense NN)
python src/train_static.py

# Dynamic gesture model (LSTM)
python src/train_dynamic.py
```
Training artifacts (`.h5` models, confusion matrices, history plots) are saved to `models/`.

### Step 4 — CLI Inference (optional, no GUI)
```bash
python src/inference.py --mode static
# or
python src/inference.py --mode dynamic
```

#### CLI Controls
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Clear sentence |
| `Space` | Grammar-correct & speak sentence |
| `M` | Toggle static / dynamic mode |

---

## 🏗️ Architecture

```
Webcam Frame
     │
     ▼
MediaPipe HandLandmarker
(126 landmarks: x, y, z × 21 × 2 hands)
     │
     ├─── Static mode ──► Dense NN ──► Gesture class + confidence
     │
     └─── Dynamic mode ─► LSTM (30-frame sequence) ──► Gesture class + confidence
                                │
                         Gesture Debouncer
                    (prevents duplicate commits)
                                │
                         Sentence Buffer
                        [Hello, I, food]
                                │
                    Gemini 3.6 Flash API
                  (ISL SOV → fluent English)
                                │
                   "Hello! I would like food."
                                │
                     Windows SAPI TTS
```

---

## ⚙️ Configuration (`src/config.py`)

| Parameter | Value | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | Frames per dynamic gesture sequence |
| `SAMPLES_PER_CLASS` | 500 | Static training samples per class |
| `SEQUENCES_PER_CLASS` | 50 | Dynamic training sequences per class |
| `MIN_DETECTION_CONFIDENCE` | 0.7 | MediaPipe hand detection threshold |
| `PREDICTION_THRESHOLD` | 0.6 | Minimum confidence to accept a prediction |
| `STABILITY_FRAMES` | 5 | Consecutive frames needed to confirm a gesture |
| `GESTURE_RESET_FRAMES` | 8 | Neutral frames before a new gesture stroke can begin |
| `REST_POSITION_Y_THRESHOLD` | 0.85 | Wrist Y > 0.85 = hand at rest (no detection) |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **OpenCV** | Webcam capture |
| **MediaPipe Tasks** | Hand landmark extraction (VIDEO mode) |
| **TensorFlow / Keras** | Dense NN + LSTM training & inference |
| **CustomTkinter** | Desktop GUI (dark theme) |
| **Pillow** | Frame rendering in GUI |
| **Google Gemini API** | LLM grammar correction (ISL-aware, SOV-aware prompt) |
| **win32com / pywin32** | Windows SAPI text-to-speech |
| **python-dotenv** | API key management via `.env` |
| **scikit-learn** | Data splitting & evaluation |
| **numpy / pandas** | Data processing |
| **matplotlib / seaborn** | Training visualization |

---

## 📋 Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.30
tensorflow>=2.15.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
pyttsx3>=2.90
customtkinter>=5.2.0
Pillow>=10.0.0
pywin32>=306
requests>=2.28.0
python-dotenv>=1.0.0
```

> **Note:** `pywin32` is required for Windows SAPI TTS. This project currently targets **Windows only**.

---

## 📌 Notes

- **Gesture debouncing** — A sign must be stable for `STABILITY_FRAMES` consecutive frames before being committed, and the same sign won't be added twice without a neutral gap between strokes.
- **ISL word order** — ISL uses Subject-Object-Verb (SOV) order (e.g., *"I food eat"* means *"I eat food"*). The Gemini prompt is explicitly tuned for this so grammar correction handles reordering automatically.
- **Graceful degradation** — If no API key is set or the Gemini call fails, the app falls back to returning the raw detected words joined by spaces.
- **Windows only** — TTS uses Windows SAPI directly via `win32com`. The rest of the pipeline is cross-platform.
