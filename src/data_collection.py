"""
Data Collection Script for Indian Sign Language (ISL) Detection.
Captures hand landmarks from webcam using MediaPipe Tasks API
and saves them as numpy arrays for model training.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import threading

# Try importing config
try:
    from src.config import (
        STATIC_CLASSES, DYNAMIC_CLASSES, SEQUENCE_LENGTH,
        SAMPLES_PER_CLASS, SEQUENCES_PER_CLASS,
        RAW_DATA_PATH, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
        MODELS_PATH
    )
    NUM_SAMPLES_STATIC = SAMPLES_PER_CLASS
    NUM_SEQUENCES = SEQUENCES_PER_CLASS
except ImportError:
    try:
        from config import (
            STATIC_CLASSES, DYNAMIC_CLASSES, SEQUENCE_LENGTH,
            SAMPLES_PER_CLASS, SEQUENCES_PER_CLASS,
            RAW_DATA_PATH, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
            MODELS_PATH
        )
        NUM_SAMPLES_STATIC = SAMPLES_PER_CLASS
        NUM_SEQUENCES = SEQUENCES_PER_CLASS
    except ImportError:
        # Fallback config
        STATIC_CLASSES = ['Hello', 'Namaste', 'India', 'Language', 'Bye',
                          'Thank You', 'Welcome', 'Please', 'Sorry', 'Practice']
        DYNAMIC_CLASSES = []
        SEQUENCE_LENGTH = 30
        NUM_SAMPLES_STATIC = 500
        NUM_SEQUENCES = 50
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
        MODELS_PATH = os.path.join(BASE_DIR, "models")
        MIN_DETECTION_CONFIDENCE = 0.7
        MIN_TRACKING_CONFIDENCE = 0.5


# ---------- MediaPipe Tasks API Setup ----------

# Resolve hand_landmarker.task model path
_MODEL_PATH = os.path.join(MODELS_PATH, 'hand_landmarker.task')
if not os.path.exists(_MODEL_PATH):
    # Try looking relative to this file
    _alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'hand_landmarker.task')
    if os.path.exists(_alt):
        _MODEL_PATH = _alt
    else:
        print(f"ERROR: hand_landmarker.task not found at {_MODEL_PATH}")
        print("Download it from: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print(f"Place it in: {MODELS_PATH}")
        exit(1)


# Hand landmark connections for drawing (replicating mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]


def draw_hand_landmarks(image, hand_landmarks, image_width, image_height,
                        landmark_color=(0, 255, 0), connection_color=(255, 255, 255)):
    """Draw hand landmarks and connections on an image using OpenCV."""
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append((x, y))
        cv2.circle(image, (x, y), 4, landmark_color, -1)
        cv2.circle(image, (x, y), 6, landmark_color, 1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], connection_color, 2)


def extract_hand_landmarks(detection_result):
    """
    Extracts landmarks for both hands (left and right) from detection result.
    Returns a numpy array of shape (126,) — 63 features per hand.
    If only one hand is detected, fills the other with zeros.
    If no hands are detected, returns None.
    """
    if not detection_result.hand_landmarks:
        return None

    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        # Get handedness
        handedness = detection_result.handedness[hand_idx][0].category_name

        # Flatten landmarks to 1D array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()

        if handedness == 'Left':
            left_hand = landmarks
        elif handedness == 'Right':
            right_hand = landmarks

    return np.concatenate([left_hand, right_hand])


def draw_visual_feedback(image, class_name, current_count, total_count, fps,
                         recording=False, is_dynamic=False, sequence=None,
                         total_sequences=None, hand_detected=None):
    """Draws visual feedback and instructions on the webcam feed."""
    h, w, _ = image.shape

    # Draw border (Green if recording and hand detected, Red if not)
    if recording:
        border_color = (0, 255, 0) if (hand_detected is not False) else (0, 0, 255)
    else:
        border_color = (0, 0, 255)
    cv2.rectangle(image, (0, 0), (w, h), border_color, 4)

    # Top info background
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)

    # Display FPS
    cv2.putText(image, f"FPS: {fps}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display Current Class Name
    cv2.putText(image, f"Class: {class_name}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Hand detection status indicator during recording
    if recording and hand_detected is not None:
        status_text = "Hand Detected" if hand_detected else "NO HAND DETECTED"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(image, status_text, (w // 2 - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if recording:
        # Recording indicator (Red dot)
        cv2.circle(image, (w - 30, h - 30), 10, (0, 0, 255), -1)

        # Bottom info background
        cv2.rectangle(image, (0, h - 40), (w, h), (0, 0, 0), -1)

        if is_dynamic:
            info_text = (f"Sequence: {sequence}/{total_sequences} | "
                         f"Frames: {current_count}/{total_count}")
        else:
            info_text = f"Collected: {current_count}/{total_count}"

        cv2.putText(image, info_text, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(image, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(image, "Press 'S' to Start Recording | 'Q' to Quit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    return image


def show_countdown(cap):
    """Shows a 3-2-1 countdown on the video feed."""
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cv2.putText(frame, str(i), (w // 2 - 20, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 165, 255), 4)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1000)


def create_landmarker():
    """Create a HandLandmarker instance using the VIDEO running mode."""
    base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    return vision.HandLandmarker.create_from_options(options)


def collect_static_data(class_name, num_samples=NUM_SAMPLES_STATIC):
    """
    Collects static hand sign data (single frames) for a specific class.
    Saves the data as a numpy array in the raw data directory.
    """
    save_dir = os.path.join(RAW_DATA_PATH, class_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    data = []
    recording = False

    landmarker = create_landmarker()

    print(f"\n--- Collecting static data for '{class_name}' ---")
    print("Show the sign. Press 'S' to start recording, 'Q' to quit.")

    p_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to MediaPipe Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks
        frame_count += 1
        timestamp_ms = int(frame_count * 1000 / 30)  # Approximate 30fps timestamps
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks on frame
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, w, h)

        # Calculate FPS
        c_time = time.time()
        fps = int(1 / (c_time - p_time)) if (c_time - p_time) > 0 else 0
        p_time = c_time

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Data collection cancelled.")
            break

        if key == ord('s') and not recording:
            show_countdown(cap)
            recording = True
            p_time = time.time()

        if recording:
            landmarks = extract_hand_landmarks(detection_result)
            if landmarks is not None:
                data.append(landmarks)

            if len(data) >= num_samples:
                print(f"Successfully collected {num_samples} samples for {class_name}.")
                break

        frame = draw_visual_feedback(frame, class_name, len(data), num_samples,
                                     fps, recording)
        cv2.imshow("Data Collection", frame)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data
    if len(data) > 0:
        np_data = np.array(data)
        save_path = os.path.join(save_dir, "static_data.npy")
        np.save(save_path, np_data)
        print(f"Data saved to {save_path} with shape {np_data.shape}")


def collect_dynamic_data(class_name, num_sequences=NUM_SEQUENCES,
                         sequence_length=SEQUENCE_LENGTH):
    """
    Collects dynamic hand sign data (sequences of frames) for a specific class.
    Saves the data as a 3D numpy array.
    """
    save_dir = os.path.join(RAW_DATA_PATH, class_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    data = []

    landmarker = create_landmarker()

    print(f"\n--- Collecting dynamic data for '{class_name}' ---")
    print("Press 'S' to start recording sequences, 'Q' to quit.")

    frame_count = 0

    # Wait for user to start
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Class: {class_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'S' to Start, 'Q' to Quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            landmarker.close()
            cap.release()
            cv2.destroyAllWindows()
            return

    p_time = 0

    for sequence in range(1, num_sequences + 1):
        sequence_data = []

        # Brief pause between sequences
        for i in range(2, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Sequence {sequence} in {i}...",
                        (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)

        frame_num = 0
        while frame_num < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            frame_count += 1
            timestamp_ms = int(frame_count * 1000 / 30)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw landmarks
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks, w, h)

            # Calculate FPS
            c_time = time.time()
            fps = int(1 / (c_time - p_time)) if (c_time - p_time) > 0 else 0
            p_time = c_time

            landmarks = extract_hand_landmarks(detection_result)
            hand_present = landmarks is not None

            if hand_present:
                sequence_data.append(landmarks)
                frame_num += 1

            frame = draw_visual_feedback(
                frame, class_name, frame_num, sequence_length, fps,
                recording=True, is_dynamic=True, sequence=sequence,
                total_sequences=num_sequences, hand_detected=hand_present
            )
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Data collection cancelled.")
                landmarker.close()
                cap.release()
                cv2.destroyAllWindows()
                return

        data.append(sequence_data)
        print(f"Collected sequence {sequence}/{num_sequences} for {class_name}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data — append to existing file if present
    if len(data) == 0:
        print("No sequences were collected. Nothing saved.")
        return

    np_new = np.array(data)
    save_path = os.path.join(save_dir, "dynamic_data.npy")

    if os.path.exists(save_path):
        existing = np.load(save_path)
        backup_path = save_path.replace(".npy", "_backup.npy")
        np.save(backup_path, existing)
        print(f"Backup saved: {backup_path}  shape={existing.shape}")
        combined = np.concatenate([existing, np_new], axis=0)
        np.save(save_path, combined)
        print(f"Appended {len(np_new)} new sequences to {len(existing)} existing => {combined.shape}")
    else:
        np.save(save_path, np_new)
        print(f"Data saved: {save_path}  shape={np_new.shape}")


def show_summary():
    """Displays a summary of all collected data."""
    print("\n" + "=" * 40)
    print("DATA COLLECTION SUMMARY")
    print("=" * 40)

    if not os.path.exists(RAW_DATA_PATH):
        print(f"Raw data directory '{RAW_DATA_PATH}' does not exist.")
        return

    classes = os.listdir(RAW_DATA_PATH)
    if not classes:
        print("No data collected yet.")
        return

    for cls in classes:
        cls_dir = os.path.join(RAW_DATA_PATH, cls)
        if not os.path.isdir(cls_dir):
            continue

        print(f"\nClass: {cls}")
        static_file = os.path.join(cls_dir, "static_data.npy")
        dynamic_file = os.path.join(cls_dir, "dynamic_data.npy")

        if os.path.exists(static_file):
            try:
                data = np.load(static_file)
                print(f"  - Static data: {data.shape[0]} samples, shape: {data.shape}")
            except Exception as e:
                print(f"  - Static data: Error loading ({e})")
        else:
            print("  - Static data: None")

        if os.path.exists(dynamic_file):
            try:
                data = np.load(dynamic_file)
                print(f"  - Dynamic data: {data.shape[0]} sequences, "
                      f"length {data.shape[1]}, shape: {data.shape}")
            except Exception as e:
                print(f"  - Dynamic data: Error loading ({e})")
        else:
            print("  - Dynamic data: None")

    print("=" * 40 + "\n")


def main():
    while True:
        print("\n--- ISL Data Collection Menu ---")
        print(f"Static classes: {STATIC_CLASSES}")
        print(f"Dynamic classes: {DYNAMIC_CLASSES}")
        print()
        print("1. Collect static data for a specific class")
        print("2. Collect static data for ALL static classes")
        print("3. Collect dynamic data for a specific class")
        print("4. Collect dynamic data for ALL dynamic classes")
        print("5. View collected data summary")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        if choice == '1':
            print("\nAvailable static classes:", STATIC_CLASSES)
            cls = input("Enter class name: ")
            if cls:
                collect_static_data(cls)
        elif choice == '2':
            for cls in STATIC_CLASSES:
                collect_static_data(cls)
                print(f"Finished {cls}. Preparing for next...")
                time.sleep(2)
        elif choice == '3':
            print("\nAvailable dynamic classes:", DYNAMIC_CLASSES)
            cls = input("Enter class name: ")
            if cls:
                # Show existing count so the user knows what is already on disk
                existing_path = os.path.join(RAW_DATA_PATH, cls, "dynamic_data.npy")
                if os.path.exists(existing_path):
                    existing_shape = np.load(existing_path).shape
                    print(f"  Existing data: {existing_shape}  ({existing_shape[0]} sequences already saved)")
                else:
                    print("  No existing data for this class.")
                n_str = input(f"How many NEW sequences to collect? [default={NUM_SEQUENCES}]: ").strip()
                n = int(n_str) if n_str.isdigit() and int(n_str) > 0 else NUM_SEQUENCES
                collect_dynamic_data(cls, num_sequences=n)
        elif choice == '4':
            for cls in DYNAMIC_CLASSES:
                collect_dynamic_data(cls)
                print(f"Finished {cls}. Preparing for next...")
                time.sleep(2)
        elif choice == '5':
            show_summary()
        elif choice == '0':
            print("Exiting Data Collection...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
