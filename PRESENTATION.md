# Presentation: Indian Sign Language Detection & Translation System

---

## Page 1: Front Page

*(Intentionally left blank)*

---

## Page 2: Outline

### Presentation Agenda

1. **Introduction & Problem Statement**
   - Background of Indian Sign Language (ISL)
   - Motivation and Key Challenges
   - Project Objectives

2. **Literature Study**
   - Comparative Analysis of Existing ISL Recognition Frameworks
   - Summary Table (Journals, Methodologies, Findings & Limitations)

3. **Technical Details**
   - **Page 5:** Data Pipeline, 3D Hand Landmark Extraction & Normalization
   - **Page 6:** Dual Neural Network Architectures (Dense NN & LSTM)
   - **Page 7:** Real-Time Inference, Prediction Stabilization & Windows SAPI TTS Integration

4. **Results and Discussion**
   - Performance Metrics & Confusion Matrices
   - Comparative Model Evaluation (Static 99.75% vs Dynamic 100.00%)
   - Real-Time Latency & Stability Analysis

5. **Conclusion & Future Scope**
   - Key Achievements
   - Limitations & Future Roadmap

6. **References**
   - Key Academic Literature & Standard Citations

---

## Page 3: Introduction

### Background & Motivation
- **Demographics:** According to Census 2011, approximately **1.8 million individuals in India** are deaf or hard of hearing.
- **ISL Characteristics:** Unlike American Sign Language (ASL), Indian Sign Language (ISL) is predominantly **two-handed**, possessing distinct grammar, structural rules, and spatial gestures.
- **The Barrier:** A severe shortage of certified ISL interpreters creates communication barriers in education, healthcare, and employment.

### Problem Statement
Existing sign language systems often rely on bulky sensor gloves or expensive RGB-D depth cameras, limiting their practical deployment. There is a need for a **lightweight, software-only, vision-based real-time ISL recognition system** that runs locally on standard desktop hardware with a webcam.

### Project Objectives
- Extract 3D hand landmarks using **MediaPipe HandLandmarker** without custom hardware.
- Implement a **dual-model deep learning pipeline**:
  - **Dense Neural Network (DNN)** for single-frame static signs (e.g., *Namaste*, *OK*).
  - **Long Short-Term Memory (LSTM)** network for dynamic motion sequences (e.g., *Hello*, *Thank You*).
- Integrate non-blocking, native **Windows SAPI 5 Text-to-Speech (TTS)** for instant spoken output.
- Deliver a modern desktop GUI using **CustomTkinter** for user interaction.

---

> 🖼️ **IMAGE PROMPT:**
> ```text
> A modern, sleek banner illustration showing a person performing two-handed Indian Sign Language gestures in front of a webcam, with glowing 3D hand skeleton landmark points overlaid on the hands, connecting to a neural network diagram and outputting text and speech soundwaves. Dark theme, vibrant blue and neon green accents, minimalist high-tech style.
> ```

---

## Page 4: Literature Study

| Sl. No. | Journal, Publisher & Year | Title & Author | Methodology | Findings | Limitations |
|---|---|---|---|---|---|
| **1** | *Multimedia Tools and Applications* (Springer, 2024) | "Automatic Indian Sign Language Recognition using MediaPipe Holistic and LSTM Network" <br>*Khartheesvar et al.* | MediaPipe Holistic landmark extraction + 2-layer LSTM sequence classifier. | Achieved high accuracy on isolated dynamic ISL signs using temporal sequences. | High computational overhead due to full holistic mesh extraction; real-time lag on CPU. |
| **2** | *Electronics* (MDPI, 2024) | "Isolated Video-Based Sign Language Recognition Using Hybrid CNN-LSTM with Attention" <br>*Kumari & Anand* | 3D CNN feature extractor coupled with an Attention-based LSTM network. | Attention mechanism effectively weighted key spatial-temporal frames in gesture sequences. | Requires raw RGB video frames leading to high memory footprint and slow frame rates. |
| **3** | *Electronics* (MDPI, 2022) | "DeepSign: Sign Language Detection and Recognition Using Deep Learning" <br>*Kothadiya et al.* | Keypoint extraction using OpenPose + Dense Neural Network classification. | Effective classification of static hand poses with low parameter count. | OpenPose keypoint extraction is prone to failure under hand self-occlusion; poor dynamic sign handling. |
| **4** | *Information* (MDPI, 2023) | "LiST: A Lightweight Framework for Continuous ISL Translation" <br>*Al-Qurishi et al.* | Lightweight 2D CNN feature map extraction + GRU sequence decoder. | Reduced model latency suitable for mobile and edge computing environments. | Restricted vocabulary size; sensitive to complex background clutter and lighting shifts. |
| **5** | *Array* (Elsevier, 2023) | "Indian Sign Language Recognition Using SURF with SVM and CNN" <br>*Sahoo et al.* | Hand segmentation via SURF descriptor + hybrid SVM & CNN classifier. | Robust static hand posture classification without deep keypoint estimation. | Hand segmentation relies heavily on skin-color thresholding, failing under varying skin tones or shadows. |

---

## Page 5: Technical Details — Data Pipeline & Feature Extraction

### 1. MediaPipe 3D Hand Landmark Extraction
- Tracks **21 3D joint landmarks** $(x, y, z)$ per hand using Google MediaPipe Tasks API.
- Dual-hand capture: 21 landmarks $\times$ 3 coordinates $\times$ 2 hands = **126 features per frame**.
- Zero-padding scheme applied when only a single hand is detected.

```
Hand Landmark Map (21 points per hand):
  0: Wrist
  1-4: Thumb (CMC → Tip)
  5-8: Index Finger (MCP → Tip)
  9-12: Middle Finger (MCP → Tip)
  13-16: Ring Finger (MCP → Tip)
  17-20: Pinky (MCP → Tip)
```

### 2. Landmark Normalization
To achieve position and hand-size invariance:
$$\hat{L}_i = \frac{L_i - L_0}{\max_{j} \| L_j - L_0 \|}$$
1. **Translation Invariance:** Subtract wrist position ($L_0$) from all 21 landmark coordinates.
2. **Scale Invariance:** Divide by the maximum Euclidean distance from the wrist to any landmark.

### 3. Dataset Collection & Augmentation
- **Static Dataset:** 500 frames collected per sign class.
- **Dynamic Dataset:** 50 sequences $\times$ 30 frames per sign class (~1 second motion window at 30 FPS).
- **Augmentation (3$\times$ factor):** Gaussian noise ($\sigma=0.01$), random scaling ($0.9-1.1$), and translation ($\sigma=0.02$).

---

> 🖼️ **IMAGE PROMPT:**
> ```text
> A detailed technical diagram illustrating the data pipeline: Webcam video frame input -> 21 3D hand landmark mesh nodes mapped onto both hands -> 126-dimensional coordinate vector extraction -> Wrist-relative translation and scale normalization formula box -> Processed feature vector array. Clean blue and dark gray technical chart style.
> ```

---

## Page 6: Technical Details — Dual Model Architecture

### Why Dual Architectures?
- **Static signs** (e.g., *Namaste*, *OK*) possess distinct single-frame hand poses.
- **Dynamic signs** (e.g., *Hello*, *Thank You*, *Bye*) involve hand motion patterns over time.

```
               ┌──────────────────────┐
               │  126-D Landmark Vector│
               └──────────┬───────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
  ┌───────────────────┐       ┌───────────────────┐
  │   Static Model    │       │   Dynamic Model   │
  │     (Dense NN)    │       │      (LSTM)       │
  │ Input: (126,)     │       │ Input: (30, 126)  │
  └─────────┬─────────┘       └─────────┬─────────┘
            │                           │
            ▼                           ▼
  Single-Frame Sign          30-Frame Sequence Sign
```

### 1. Static Model Architecture (Dense NN)
- **Input Layer:** (126,) normalized features
- **Dense Layer 1:** 256 units, ReLU activation $\rightarrow$ BatchNormalization $\rightarrow$ Dropout (0.3)
- **Dense Layer 2:** 128 units, ReLU activation $\rightarrow$ BatchNormalization $\rightarrow$ Dropout (0.3)
- **Dense Layer 3:** 64 units, ReLU activation $\rightarrow$ BatchNormalization
- **Output Layer:** Softmax classifier ($N$ classes)
- **Total Parameters:** 75,781 (~296 KB)

### 2. Dynamic Model Architecture (LSTM)
- **Input Shape:** (30, 126) — 30 consecutive frames $\times$ 126 features
- **LSTM Layer 1:** 128 units (`return_sequences=True`) $\rightarrow$ Dropout (0.3)
- **LSTM Layer 2:** 64 units (`return_sequences=False`) $\rightarrow$ Dropout (0.3)
- **Dense Layer 1:** 64 units, ReLU $\rightarrow$ BatchNormalization
- **Dense Layer 2:** 32 units, ReLU
- **Output Layer:** Softmax classifier ($N$ classes)

---

> 🖼️ **IMAGE PROMPT:**
> ```text
> Side-by-side architectural diagram comparing a Dense Neural Network layer block and an LSTM Recurrent Neural Network unrolled over 30 time steps. Include layer shapes, dropout blocks, batch normalization blocks, and softmax output nodes. Futuristic neural network design with purple, teal, and dark slate color scheme.
> ```

---

## Page 7: Technical Details — Real-Time Inference & System Pipeline

### 1. Prediction Stabilization Engine
To prevent prediction flickering between adjacent frames:
- **Rolling Buffer:** Maintains a sliding window of the last **5 frames**.
- **Threshold Gate:** Accepts predictions only if confidence exceeds $0.6$ / $0.7$.
- **Consistency Check:** Triggers output only when all 5 frames yield the identical predicted class.

### 2. Sentence Builder & Cooldown Control
- **Cooldown Period:** 1-second delay imposed after speech output to prevent rapid-fire repeated triggering.
- **Sentence Accumulation:** Confirmed words are automatically appended into an ongoing sentence string.

### 3. Asynchronous Text-to-Speech (TTS) Engine
- Uses Windows native **SAPI 5 (`SpVoice`)** via `win32com` COM dispatch.
- Bypasses blocking `pyttsx3` loops using `SVSFlagsAsync` (flag = 1) for non-blocking speech playback during active video inference.

### 4. CustomTkinter Desktop Application
- **Multi-threaded Architecture:** 
  - Worker thread handles webcam capture + MediaPipe + Keras model evaluation.
  - Main thread renders CustomTkinter GUI at 30 FPS using thread-safe `queue.Queue` polling.

---

> 🖼️ **IMAGE PROMPT:**
> ```text
> High-resolution screenshot mockup of a modern dark-mode desktop GUI application window titled 'ISL Translator'. Left side shows a live webcam feed of a person doing sign language with green skeletal hand tracking lines. Right side shows a large bold prediction title 'NAMASTE', a glowing green confidence progress bar at 99.8%, a sentence history textbox, mode toggle buttons, and voice setting sliders. Professional UI mockup.
> ```

---

## Page 8: Results and Discussion

### Performance Summary

| Metric | Static Model (Dense NN) | Dynamic Model (LSTM) |
|---|---|---|
| **Architecture** | 4-Layer Dense NN | 2-Layer LSTM + Dense |
| **Input Shape** | (126,) | (30, 126) |
| **Training Epochs** | 16 (Early stopped at epoch 6) | 100 (Best weights at epoch 93) |
| **Test Accuracy** | **99.75%** | **100.00%** |
| **Inference Latency** | ~3 ms / frame | ~12 ms / sequence |
| **Model File Size** | 963 KB | 2.3 MB |

### Classification Report (5 Core Vocabulary Classes)

```
Static Model (Dense NN):
  Class       Precision    Recall    F1-Score
  Bye           0.99        1.00       0.99
  Hello         1.00        0.99       0.99
  Namaste       1.00        1.00       1.00
  Sorry         1.00        1.00       1.00
  Thank You     1.00        1.00       1.00
  Overall       1.00        1.00       0.9975

Dynamic Model (LSTM):
  Class       Precision    Recall    F1-Score
  Bye           1.00        1.00       1.00
  Hello         1.00        1.00       1.00
  Namaste       1.00        1.00       1.00
  Sorry         1.00        1.00       1.00
  Thank You     1.00        1.00       1.00
  Overall       1.00        1.00       1.0000
```

### Discussion
- Normalization effectively eliminated variation caused by distance from the webcam and hand sizes.
- The 5-frame stabilization buffer eliminated false positives during transitional hand movements between signs.

---

> 🖼️ **IMAGE PROMPT:**
> ```text
> Dual visualization chart panel: Left side showing training and validation accuracy and loss curves over epochs for an LSTM model; Right side showing a 5x5 confusion matrix heatmap with high diagonal intensity values labeled with class names (Hello, Namaste, Bye, Thank You, Sorry). Professional scientific plot design with dark background, crisp green and blue gradient heatmaps.
> ```

---

## Page 9: Conclusion & Future Scope

### Conclusion
- Developed a complete, real-time vision-based **Indian Sign Language detection and translation system**.
- Achieved **99.75% accuracy on static gestures** and **100.00% accuracy on dynamic gesture sequences**.
- Built a lightweight feature representation (126-D vector) eliminating the need for GPU acceleration or depth sensors.
- Integrated non-blocking **Windows SAPI TTS** and built a responsive **CustomTkinter desktop application**.

### Limitations
- Current initial dataset covers 5 core dynamic/static phrases (*Hello*, *Namaste*, *Bye*, *Thank You*, *Sorry*).
- Requires clear hand visibility within webcam field of view without extreme lighting dropouts.

### Future Scope
1. **Vocabulary Expansion:** Extend to full ISL Alphabets (A–Z), Numbers (0–9), and 50+ extended phrases.
2. **Continuous Sign Recognition:** Implement Connectionist Temporal Classification (CTC) loss for unsegmented continuous sign streams.
3. **NLP Grammar Mapping:** Add an NLP Transformer layer to map raw gloss sequences into grammatically correct English sentences.

---

> 🖼️ **IMAGE PROMPT:**
> ```text
> An infographic roadmap showing three future milestones: Step 1 (Alphabet & Digit Expansion with 26 letters), Step 2 (Continuous Sign Language Recognition with CTC Loss), and Step 3 (NLP Grammar Translation into full sentences). Modern icon-based flowchart on dark background with connecting light trails.
> ```

---

## Page 10: References

1. **Khartheesvar, G., et al.** (2024). *"Automatic Indian Sign Language Recognition using MediaPipe Holistic and LSTM Network"*. *Multimedia Tools and Applications*, Springer (SCI). [DOI: 10.1007/s11042-023-17361-y](https://doi.org/10.1007/s11042-023-17361-y)
2. **Al-Qurishi, M., et al.** (2023). *"LiST: A Lightweight Framework for Continuous ISL Translation"*. *Information*, MDPI (Scopus). [DOI: 10.3390/info14020079](https://doi.org/10.3390/info14020079)
3. **Kumari, D., & Anand, R. S.** (2024). *"Isolated Video-Based Sign Language Recognition Using Hybrid CNN-LSTM with Attention"*. *Electronics*, MDPI (SCI). [DOI: 10.3390/electronics13071229](https://doi.org/10.3390/electronics13071229)
4. **Kothadiya, D., et al.** (2022). *"DeepSign: Sign Language Detection and Recognition Using Deep Learning"*. *Electronics*, MDPI (SCI). [DOI: 10.3390/electronics11111780](https://doi.org/10.3390/electronics11111780)
5. **IEEE Access.** (2025). *"Real-Time Sign Language-to-Speech Translation with AI-Powered Wearable Technology"*. *IEEE Access*, (SCI). [DOI: 10.1109/ACCESS.2025.3602794](https://doi.org/10.1109/ACCESS.2025.3602794)
6. **JSIR.** (2022). *"Real-time Static and Dynamic Sign Language Recognition using Deep Learning"*. *Journal of Scientific and Industrial Research*, (SCI). [DOI: 10.56042/jsir.v81i11.60865](https://doi.org/10.56042/jsir.v81i11.60865)
7. **IEEE MysuruCon.** (2021). *"Evaluation of Machine Learning Models for Real-Time Sign Recognition"*. *IEEE MysuruCon 2021*. [DOI: 10.1109/MysuruCon52639.2021.9641518](https://doi.org/10.1109/MysuruCon52639.2021.9641518)
8. **Sahoo, S., et al.** (2023). *"Indian Sign Language Recognition Using SURF with SVM and CNN"*. *Array*, Elsevier (Scopus). [DOI: 10.1016/j.array.2023.100257](https://doi.org/10.1016/j.array.2023.100257)
