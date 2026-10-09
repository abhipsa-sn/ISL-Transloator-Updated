"""
Real-time Indian Sign Language recognition and translation.
Captures webcam feed, extracts hand landmarks using MediaPipe Tasks API,
classifies gestures, displays predictions, and speaks detected signs using TTS.
"""

import os
import sys
from unittest.mock import MagicMock

# Mock matplotlib to prevent DLL load failure under Windows Application Control policy
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.ft2font'] = MagicMock()

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from collections import deque

try:
    from src import config
    from src.tts import TTSEngine
    from src import llm_translator
except ImportError:
    try:
        import config
        from tts import TTSEngine
        import llm_translator
    except ImportError:
        print("Error: Could not import config, tts, or llm_translator modules.")
        print("Please run from the project root or src directory.")
        exit(1)

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_landmarks(image, hand_landmarks, image_width, image_height,
                        landmark_color=(0, 255, 0), connection_color=(255, 255, 255)):
    """Draw hand landmarks and connections on an image."""
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


class ISLRecognizer:
    """
    Real-time Indian Sign Language recognizer.
    Supports static (single-frame) and dynamic (sequence-based) modes.
    """

    def __init__(self, mode='static'):
        self.mode = mode
        self.sequence_length = getattr(config, 'SEQUENCE_LENGTH', 30)

        # Resolve model path for MediaPipe
        models_path = getattr(config, 'MODELS_PATH',
                              str(getattr(config, 'MODELS_DIR', 'models')))
        self.mp_model_path = os.path.join(models_path, 'hand_landmarker.task')
        if not os.path.exists(self.mp_model_path):
            print(f"ERROR: hand_landmarker.task not found at {self.mp_model_path}")
            print("Download from: https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            exit(1)

        # Initialize MediaPipe HandLandmarker (VIDEO mode)
        base_options = python.BaseOptions(model_asset_path=self.mp_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.7),
            min_hand_presence_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.7),
            min_tracking_confidence=getattr(config, 'MIN_TRACKING_CONFIDENCE', 0.5)
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Initialize TTS
        self.tts = TTSEngine()

        # Load ML Models and Encoders
        self.static_model = None
        self.dynamic_model = None
        self.static_classes = None
        self.dynamic_classes = None
        self.load_models()

        # Prediction stabilization
        stability_frames = getattr(config, 'STABILITY_FRAMES', 10)
        self.prediction_buffer = deque(maxlen=stability_frames)
        self.sequence_buffer = deque(maxlen=self.sequence_length)

        # Sentence builder state
        self.current_sentence = []
        self.last_spoken_time = 0
        self.cooldown_period = 1.0
        self.last_word = ""

        # Phase 1: Gesture lifecycle & debounce state
        # Tracks gesture strokes: prevents duplicate additions while holding,
        # and resets state only when hands rest or gesture concludes.
        self.active_gesture = None          # Current active sign class name (e.g. 'Hello')
        self.gesture_committed = False      # True once current stroke has been added to current_sentence
        self.neutral_frame_count = 0        # Counter for consecutive neutral/unconfirmed frames
        self.reset_frames = getattr(config, 'GESTURE_RESET_FRAMES', 8)
        self.rest_y_threshold = getattr(config, 'REST_POSITION_Y_THRESHOLD', 0.85)

        # Frame counter for timestamps
        self.frame_count = 0

        # Phase 2: Passive motion telemetry (measurement only, no routing)
        self.telemetry_history_len = 15
        self.raw_wrist_history = deque(maxlen=self.telemetry_history_len)
        self.prev_raw_landmarks = None
        self.telemetry_log_interval = 15
        self.telemetry_frame_counter = 0
        self.last_wrist_motion = 0.0
        self.last_art_motion = 0.0
        self.last_composite_motion = 0.0

        # Phase 3: Three-Layer AUTO Router Architecture
        # Layer 1: Rolling raw landmark history (un-normalized 126-D coordinates)
        self.auto_history_maxlen = 45
        self.recent_landmark_history = deque(maxlen=self.auto_history_maxlen)

        # Layer 3: AUTO-specific dynamic sequence buffer (genuine 30 frames for LSTM)
        # Note: self.sequence_buffer remains dedicated strictly to manual 'dynamic' mode
        self.dynamic_sequence_buffer = deque(maxlen=self.sequence_length)

        # Layer 2: AUTO Router state & temporal tracking
        self.router_state = "uncertain"   # "uncertain", "static", "dynamic", "dynamic_pause"
        self.consecutive_static_candidate = 0
        self.consecutive_dynamic_candidate = 0
        self.dynamic_pause_count = 0
        self.dynamic_onset_frame_id = None
        self.in_dynamic_gesture = False

        # AUTO-local parameters and thresholds (isolated and easy to tune)
        self.auto_art_weight = 1.8               # Multiplier for relative landmark articulation
        self.auto_static_motion_ceiling = 0.007   # Composite motion <= 0.007 indicates stationary pose
        self.auto_dynamic_motion_floor = 0.010    # Composite motion >= 0.010 indicates active movement
        self.auto_min_history_for_decision = 4    # Minimum valid frames before initial decision
        self.auto_static_confirm_frames = 4       # Consecutive low-motion frames to confirm STATIC
        self.auto_dynamic_confirm_frames = 3      # Consecutive frames to confirm DYNAMIC (rejects 1-2 frame spikes)
        self.auto_pre_roll_frames = 3             # Pre-roll frames included before detected dynamic onset
        self.auto_dynamic_pause_max = 12          # Low-motion frames tolerated before ending DYNAMIC
        self.auto_log_interval = 10               # Periodic diagnostic logging interval
        self.auto_frame_counter = 0

        # AUTO-specific dynamic stabilization & re-arming
        self.auto_dynamic_stability_frames = 2    # Short deliberate confirmation (2 consecutive matching predictions)
        self.auto_dynamic_buffer = deque(maxlen=self.auto_dynamic_stability_frames)
        self.auto_dynamic_last_committed = None   # Suppresses duplicate commits within the same dynamic stroke

        # AUTO post-dynamic terminal posture guard (Part A)
        self.auto_terminal_guard_active = False       # True when guarding against false static from held terminal pose
        self.auto_terminal_landmarks = None           # Snapshot of raw landmarks when dynamic pause timed out
        self.auto_posture_change_confirm = 0          # Frames confirming posture changed from terminal pose
        self.auto_dynamic_episode_committed = False   # True if current dynamic episode committed a sign

    def load_models(self):
        """Load trained ML models and label encoders based on mode."""
        proc_path = getattr(config, 'PROCESSED_DATA_PATH',
                            str(getattr(config, 'DATA_PROCESSED_DIR', 'data/processed')))
        models_path = getattr(config, 'MODELS_PATH',
                              str(getattr(config, 'MODELS_DIR', 'models')))

        try:
            if self.mode in ['static', 'both', 'auto']:
                static_model_path = os.path.join(models_path, 'static_model_6class.h5')
                static_le_path = os.path.join(proc_path, 'label_encoder_6class.npy')

                if not os.path.exists(static_model_path):
                    raise FileNotFoundError(
                        f"Static model not found at {static_model_path}. "
                        "Please train the model first: python src/train_static.py"
                    )
                if not os.path.exists(static_le_path):
                    raise FileNotFoundError(
                        f"Label encoder not found at {static_le_path}. "
                        "Please run preprocessing first: python src/preprocess.py"
                    )

                self.static_model = tf.keras.models.load_model(static_model_path)
                self.static_classes = np.load(static_le_path, allow_pickle=True)
                print(f"  Static model loaded ({len(self.static_classes)} classes)")

            if self.mode in ['dynamic', 'both', 'auto']:
                dynamic_model_path = os.path.join(models_path, 'dynamic_model.h5')
                dynamic_le_path = os.path.join(proc_path, 'label_encoder_dynamic.npy')

                if not os.path.exists(dynamic_model_path):
                    raise FileNotFoundError(
                        f"Dynamic model not found at {dynamic_model_path}. "
                        "Please train the model first: python src/train_dynamic.py"
                    )
                if not os.path.exists(dynamic_le_path):
                    raise FileNotFoundError(
                        f"Dynamic label encoder not found at {dynamic_le_path}."
                    )

                self.dynamic_model = tf.keras.models.load_model(dynamic_model_path)
                self.dynamic_classes = np.load(dynamic_le_path, allow_pickle=True)
                print(f"  Dynamic model loaded ({len(self.dynamic_classes)} classes)")

        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nPlease follow these steps:")
            print("  1. Collect data:    python src/data_collection.py")
            print("  2. Preprocess:      python src/preprocess.py")
            print("  3. Train model:     python src/train_static.py")
            print("  4. Run inference:   python src/inference.py")
            exit(1)
        except Exception as e:
            print(f"\nUnexpected error loading models: {e}")
            exit(1)

    def extract_landmarks(self, frame):
        """
        Process a frame with MediaPipe Tasks API and extract hand landmarks.

        Returns:
            landmarks: numpy array of shape (126,)
            frame: annotated frame with hand skeleton drawn
            hands_detected: boolean
        """
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self.frame_count += 1
        timestamp_ms = int(self.frame_count * 1000 / 30)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = np.zeros(126)
        hands_detected = bool(detection_result.hand_landmarks)

        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Draw on frame
                draw_hand_landmarks(frame, hand_landmarks, w, h)

                # Determine hand offset
                handedness = detection_result.handedness[idx][0].category_name
                offset = 63 if handedness == 'Right' else 0

                # Extract landmark coordinates
                for i, lm in enumerate(hand_landmarks):
                    landmarks[offset + i * 3] = lm.x
                    landmarks[offset + i * 3 + 1] = lm.y
                    landmarks[offset + i * 3 + 2] = lm.z

        return landmarks, frame, hands_detected

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position for each hand."""
        normalized = np.zeros_like(landmarks)
        for i in range(0, 126, 63):
            hand = landmarks[i:i + 63]
            if np.all(hand == 0):
                continue
            wrist = hand[0:3]
            shifted = hand.copy()
            for j in range(0, 63, 3):
                shifted[j] -= wrist[0]
                shifted[j + 1] -= wrist[1]
                shifted[j + 2] -= wrist[2]
            distances = [np.sqrt(shifted[j]**2 + shifted[j+1]**2 + shifted[j+2]**2)
                         for j in range(0, 63, 3)]
            max_dist = max(distances) if distances else 0
            if max_dist > 0:
                shifted = shifted / max_dist
            normalized[i:i + 63] = shifted
        return normalized

    def predict_static(self, landmarks):
        """Predict sign from single-frame landmarks."""
        normalized = self.normalize_landmarks(landmarks)
        pred = self.static_model.predict(np.array([normalized]), verbose=0)[0]
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        class_name = self.static_classes[class_idx]
        return class_name, confidence

    def predict_dynamic(self, sequence):
        """Predict sign from a sequence of landmark frames."""
        seq_array = np.array(sequence)
        # Normalize each frame in the sequence
        normalized_seq = np.array([self.normalize_landmarks(frame) for frame in seq_array])
        pred = self.dynamic_model.predict(np.array([normalized_seq]), verbose=0)[0]
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        class_name = self.dynamic_classes[class_idx]
        return class_name, confidence

    def stabilize_prediction(self, prediction, confidence):
        """Only confirm if same prediction appears consistently with high confidence."""
        threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.7)
        if confidence > threshold:
            self.prediction_buffer.append(prediction)
        else:
            self.prediction_buffer.append(None)

        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            if (len(set(self.prediction_buffer)) == 1
                    and self.prediction_buffer[0] is not None):
                return self.prediction_buffer[0]
        return None

    def stabilize_auto_dynamic(self, prediction, confidence):
        """
        AUTO-specific dynamic stabilization.
        Requires 2 consecutive matching high-confidence predictions (short deliberate confirmation)
        and suppresses duplicate commits for the same dynamic gesture episode.
        """
        threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.6)
        if confidence > threshold and prediction is not None:
            self.auto_dynamic_buffer.append(prediction)
        else:
            self.auto_dynamic_buffer.append(None)

        if len(self.auto_dynamic_buffer) == self.auto_dynamic_buffer.maxlen:
            if (len(set(self.auto_dynamic_buffer)) == 1
                    and self.auto_dynamic_buffer[0] is not None):
                candidate = self.auto_dynamic_buffer[0]
                if candidate != self.auto_dynamic_last_committed:
                    self.auto_dynamic_last_committed = candidate
                    self.auto_dynamic_episode_committed = True
                    return candidate
        return None

    def _is_posture_changed(self, landmarks, terminal_landmarks):
        """
        Check if current hand configuration differs meaningfully from terminal posture.
        Evaluates wrist-relative normalized articulation landmarks (excluding the wrist point)
        using both overall mean displacement and maximum per-landmark displacement.
        Wrist translation alone does not indicate a posture change.

        Returns True if:
        - Hand presence changed (1-hand vs 2-hand transition)
        - Mean normalized articulation displacement exceeds 0.08
        - Maximum localized landmark displacement exceeds 0.15
        """
        if terminal_landmarks is None:
            return True

        curr_left = bool(np.any(landmarks[0:63] != 0))
        curr_right = bool(np.any(landmarks[63:126] != 0))
        term_left = bool(np.any(terminal_landmarks[0:63] != 0))
        term_right = bool(np.any(terminal_landmarks[63:126] != 0))

        # Different hand count or handedness presence -> structural posture change
        if (curr_left != term_left) or (curr_right != term_right):
            return True

        # Normalized hand landmarks (wrist translated to (0,0,0), scale normalized to max_dist=1.0)
        curr_norm = self.normalize_landmarks(landmarks)
        term_norm = self.normalize_landmarks(terminal_landmarks)

        hand_shape_changed = False
        hands_evaluated = False

        for offset, is_curr, is_term in [(0, curr_left, term_left), (63, curr_right, term_right)]:
            if is_curr and is_term:
                hands_evaluated = True
                # Exclude wrist landmark at index 0 (always (0,0,0) in normalized space)
                # and compare the 20 finger articulation landmarks (indices 1 to 20)
                pts_curr = curr_norm[offset:offset + 63].reshape(21, 3)[1:]
                pts_term = term_norm[offset:offset + 63].reshape(21, 3)[1:]

                per_lm_dists = np.linalg.norm(pts_curr - pts_term, axis=1)
                mean_dist = float(np.mean(per_lm_dists))
                max_dist = float(np.max(per_lm_dists))

                # Posture change detected if overall shape shifts OR any single finger digit shifts
                # Threshold units: Euclidean distance in wrist-centered, scale-normalized coordinate space
                if mean_dist > 0.08 or max_dist > 0.15:
                    hand_shape_changed = True
                    break

        if not hands_evaluated:
            return True

        return hand_shape_changed

    def is_hand_at_rest(self, landmarks):
        """
        Check if all detected hands are resting near the bottom of the camera frame.
        Uses normalized MediaPipe landmark coordinates where y=0 is top and y=1 is bottom.
        """
        wrist_ys = []
        # Left hand wrist (offset 0, landmark index 0 -> y coordinate at index 1)
        if np.any(landmarks[0:63] != 0):
            wrist_ys.append(landmarks[1])
        # Right hand wrist (offset 63, landmark index 0 -> y coordinate at index 64)
        if np.any(landmarks[63:126] != 0):
            wrist_ys.append(landmarks[64])

        if not wrist_ys:
            return True
        return all(y > self.rest_y_threshold for y in wrist_ys)

    def _handle_gesture_commit(self, stable_pred, current_conf=0.0):
        """
        Commit a stable prediction as a single gesture stroke.
        Prevents duplicate additions while holding the same sign.
        Allows immediate transition if the sign changes.
        """
        if stable_pred is not None:
            self.neutral_frame_count = 0
            # Commit if no gesture is currently active, or if sign transitioned to a new class
            if not self.gesture_committed or stable_pred != self.active_gesture:
                self.tts.speak(stable_pred)
                self.current_sentence.append(stable_pred)
                self.last_word = stable_pred
                self.active_gesture = stable_pred
                self.gesture_committed = True
                self.last_spoken_time = time.time()
                return True
        return False

    def _handle_gesture_reset(self):
        """
        Increment neutral frame counter and reset the gesture stroke state
        once hands rest, leave the frame, or confidence drops for consecutive frames.
        """
        self.neutral_frame_count += 1
        if self.neutral_frame_count >= self.reset_frames:
            self.active_gesture = None
            self.gesture_committed = False

    def clear_sentence(self):
        """Clear accumulated sentence and reset gesture lifecycle state."""
        self.current_sentence = []
        self.last_word = ""
        self.active_gesture = None
        self.gesture_committed = False
        self.neutral_frame_count = 0
        self.auto_dynamic_last_committed = None
        self.auto_terminal_guard_active = False
        self.auto_terminal_landmarks = None
        self.auto_posture_change_confirm = 0
        self.auto_dynamic_episode_committed = False

    def _calculate_motion(self, landmarks, prev_landmarks):
        """
        Calculate wrist displacement and relative landmark articulation displacement
        between two raw 126-D landmark frames.
        Returns (wrist_motion, art_motion, composite_motion).
        """
        if prev_landmarks is None:
            return 0.0, 0.0, 0.0

        has_left_curr = bool(np.any(landmarks[0:63] != 0))
        has_left_prev = bool(np.any(prev_landmarks[0:63] != 0))
        has_right_curr = bool(np.any(landmarks[63:126] != 0))
        has_right_prev = bool(np.any(prev_landmarks[63:126] != 0))

        wrist_deltas = []
        art_deltas = []

        # Left hand (indices 0:63)
        if has_left_curr and has_left_prev:
            w_curr = landmarks[0:3]
            w_prev = prev_landmarks[0:3]
            dw_left = float(np.linalg.norm(w_curr - w_prev))
            wrist_deltas.append(dw_left)

            # 20 hand landmarks relative to wrist
            pts_curr = landmarks[0:63].reshape(21, 3) - w_curr
            pts_prev = prev_landmarks[0:63].reshape(21, 3) - w_prev
            dart_left = float(np.mean(np.linalg.norm(pts_curr[1:] - pts_prev[1:], axis=1)))
            art_deltas.append(dart_left)

        # Right hand (indices 63:126)
        if has_right_curr and has_right_prev:
            w_curr = landmarks[63:66]
            w_prev = prev_landmarks[63:66]
            dw_right = float(np.linalg.norm(w_curr - w_prev))
            wrist_deltas.append(dw_right)

            pts_curr = landmarks[63:126].reshape(21, 3) - w_curr
            pts_prev = prev_landmarks[63:126].reshape(21, 3) - w_prev
            dart_right = float(np.mean(np.linalg.norm(pts_curr[1:] - pts_prev[1:], axis=1)))
            art_deltas.append(dart_right)

        wrist_motion = max(wrist_deltas) if wrist_deltas else 0.0
        art_motion = max(art_deltas) if art_deltas else 0.0
        composite_motion = max(wrist_motion, self.auto_art_weight * art_motion)

        return wrist_motion, art_motion, composite_motion

    def _update_motion_telemetry(self, landmarks, hands_detected):
        """
        Passive motion telemetry tracker (Phase 2: measurement only).
        Calculates inter-frame displacement of RAW landmarks before normalization.
        Updates self.last_wrist_motion, self.last_art_motion, self.last_composite_motion.
        """
        self.telemetry_frame_counter += 1
        should_log = (self.telemetry_frame_counter % self.telemetry_log_interval == 0)

        if not hands_detected:
            self.prev_raw_landmarks = None
            self.raw_wrist_history.clear()
            self.last_wrist_motion = 0.0
            self.last_art_motion = 0.0
            self.last_composite_motion = 0.0
            if should_log and self.mode != 'auto':
                print(f"[MOTION] mode={self.mode} hands=0 motion=unavailable")
            return

        has_left = bool(np.any(landmarks[0:63] != 0))
        has_right = bool(np.any(landmarks[63:126] != 0))
        num_hands = (1 if has_left else 0) + (1 if has_right else 0)

        wrist_motion, art_motion, composite_motion = self._calculate_motion(landmarks, self.prev_raw_landmarks)
        self.last_wrist_motion = wrist_motion
        self.last_art_motion = art_motion
        self.last_composite_motion = composite_motion

        # Store current raw landmarks for consecutive-frame delta
        self.prev_raw_landmarks = landmarks.copy()

        self.raw_wrist_history.append(wrist_motion)

        # Rolling statistics
        rolling_mean = float(np.mean(self.raw_wrist_history)) if len(self.raw_wrist_history) > 0 else None
        rolling_max = float(np.max(self.raw_wrist_history)) if len(self.raw_wrist_history) > 0 else None

        if should_log and self.mode != 'auto':
            w_str = f"{wrist_motion:.4f}"
            art_str = f"{art_motion:.4f}"
            comp_str = f"{composite_motion:.4f}"
            mean_str = f"{rolling_mean:.4f}" if rolling_mean is not None else "N/A"
            max_str = f"{rolling_max:.4f}" if rolling_max is not None else "N/A"
            print(f"[MOTION] mode={self.mode} hands={num_hands} "
                  f"wrist={w_str} art={art_str} comp={comp_str} "
                  f"rolling_mean={mean_str} rolling_max={max_str}")

    def _reset_auto_router(self):
        """Reset the internal temporal candidate state and buffers of the AUTO router."""
        self.router_state = "uncertain"
        self.consecutive_static_candidate = 0
        self.consecutive_dynamic_candidate = 0
        self.dynamic_pause_count = 0
        self.dynamic_onset_frame_id = None
        self.in_dynamic_gesture = False
        self.recent_landmark_history.clear()
        self.dynamic_sequence_buffer.clear()
        self.raw_wrist_history.clear()
        self.prev_raw_landmarks = None
        self.auto_dynamic_buffer.clear()
        self.auto_dynamic_last_committed = None
        self.auto_terminal_guard_active = False
        self.auto_terminal_landmarks = None
        self.auto_posture_change_confirm = 0
        self.auto_dynamic_episode_committed = False

    def _detect_dynamic_onset_and_seed(self):
        """
        Layer 3 Dynamic Onset & Window Construction:
        When AUTO enters DYNAMIC, locate where meaningful motion began in
        recent_landmark_history, add a small pre-roll (3 frames),
        and initialize dynamic_sequence_buffer without artificial padding.
        """
        H = list(self.recent_landmark_history)
        N = len(H)
        if N == 0:
            return

        # Scan backwards to locate the start of meaningful motion.
        # Look for a sustained hold of 4 consecutive still frames; if found,
        # onset is immediately after that hold. Otherwise onset is at start of history.
        onset_idx = 0
        for j in range(N - 1, 3, -1):
            if all(H[j - k][4] < self.auto_static_motion_ceiling for k in range(4)):
                onset_idx = j
                break

        onset_frame_id = H[onset_idx][0]
        self.dynamic_onset_frame_id = onset_frame_id

        # Small pre-roll before onset (3 frames)
        start_idx = max(0, onset_idx - self.auto_pre_roll_frames)

        # Seed dynamic_sequence_buffer with genuine raw frames
        self.dynamic_sequence_buffer.clear()
        self.auto_dynamic_buffer.clear()
        self.auto_dynamic_episode_committed = False
        for j in range(start_idx, N):
            self.dynamic_sequence_buffer.append(H[j][1])

    def _evaluate_auto_router(self, landmarks, hands_detected):
        """
        Layer 2: AUTO Motion/Routing Analysis and State Machine.
        Operates across Layer 1 (recent_landmark_history) to choose between
        'uncertain' (WAIT), 'static', 'dynamic', and 'dynamic_pause'.
        Calls predict_static() or predict_dynamic() appropriately.
        Returns (effective_mode, raw_pred, conf).
        """
        if not hands_detected:
            self._reset_auto_router()
            return "uncertain", None, 0.0

        wrist_motion = self.last_wrist_motion
        art_motion = self.last_art_motion
        comp_motion = self.last_composite_motion

        self.auto_frame_counter += 1
        current_frame_id = self.frame_count if self.frame_count > 0 else self.auto_frame_counter

        # Layer 1: Append raw landmark frame to rolling history (no normalization, no fake frames)
        self.recent_landmark_history.append(
            (current_frame_id, landmarks.copy(), wrist_motion, art_motion, comp_motion)
        )

        n_hist = len(self.recent_landmark_history)

        # Insufficient temporal context -> remain UNCERTAIN / WAIT
        if n_hist < self.auto_min_history_for_decision and not self.in_dynamic_gesture:
            self.router_state = "uncertain"
            if self.auto_frame_counter % self.auto_log_interval == 0:
                print(f"[AUTO] frame={current_frame_id} state=UNCERTAIN wrist={wrist_motion:.4f} "
                      f"art={art_motion:.4f} motion={comp_motion:.4f} history={n_hist} "
                      f"dynamic_seq={len(self.dynamic_sequence_buffer)} onset={self.dynamic_onset_frame_id}")
            return "uncertain", None, 0.0

        # Recent motion window (last 4 frames)
        recent = list(self.recent_landmark_history)[-4:]
        recent_comps = [h[4] for h in recent]
        elevated_count = sum(c >= self.auto_dynamic_motion_floor for c in recent_comps)
        # Sustained dynamic signal requires multiple frames of elevated motion, NOT a single isolated spike
        has_dynamic_signal = (elevated_count >= 2)

        raw_pred = None
        conf = 0.0

        # --- STATE MACHINE ---
        if not self.in_dynamic_gesture:
            if self.router_state == "static":
                # --- ALREADY IN STATIC: STRONG HYSTERESIS ---
                # Once in STATIC, isolated spikes or brief MediaPipe jitter do NOT eject to UNCERTAIN.
                # Only sustained dynamic evidence (active motion on current frame accumulating confirm frames)
                # triggers transition to DYNAMIC.
                if comp_motion >= self.auto_dynamic_motion_floor:
                    self.consecutive_dynamic_candidate += 1
                    if self.consecutive_dynamic_candidate >= self.auto_dynamic_confirm_frames:
                        self.in_dynamic_gesture = True
                        self.router_state = "dynamic"
                        self.dynamic_pause_count = 0
                        self.consecutive_dynamic_candidate = 0
                        self.auto_terminal_guard_active = False
                        self.auto_terminal_landmarks = None
                        self.auto_posture_change_confirm = 0
                        self._detect_dynamic_onset_and_seed()
                else:
                    self.consecutive_dynamic_candidate = 0

            else:
                # --- IN UNCERTAIN / ACQUIRING ---
                if has_dynamic_signal:
                    self.consecutive_dynamic_candidate += 1
                    self.consecutive_static_candidate = 0

                    if self.consecutive_dynamic_candidate >= self.auto_dynamic_confirm_frames:
                        # Sustained dynamic activity confirmed
                        self.in_dynamic_gesture = True
                        self.router_state = "dynamic"
                        self.dynamic_pause_count = 0
                        self.consecutive_dynamic_candidate = 0
                        self.auto_terminal_guard_active = False
                        self.auto_terminal_landmarks = None
                        self.auto_posture_change_confirm = 0
                        self._detect_dynamic_onset_and_seed()
                elif comp_motion <= self.auto_static_motion_ceiling:
                    # Low motion candidate
                    if self.auto_terminal_guard_active:
                        # Post-dynamic guard active: check whether user deliberately changed posture
                        if self._is_posture_changed(landmarks, self.auto_terminal_landmarks):
                            self.auto_posture_change_confirm += 1
                            if self.auto_posture_change_confirm >= 2:
                                # Confirmed deliberate posture change to new static sign
                                self.auto_terminal_guard_active = False
                                self.auto_terminal_landmarks = None
                                self.auto_posture_change_confirm = 0
                                self.consecutive_static_candidate = 1
                                self.consecutive_dynamic_candidate = 0
                        else:
                            # Holding terminal posture: suppress static candidate accumulation
                            self.auto_posture_change_confirm = 0
                            self.consecutive_static_candidate = 0
                            self.consecutive_dynamic_candidate = 0
                    else:
                        self.consecutive_static_candidate += 1
                        self.consecutive_dynamic_candidate = 0

                        if self.consecutive_static_candidate >= self.auto_static_confirm_frames:
                            self.router_state = "static"
                else:
                    # Ambiguous motion: decay counters gradually
                    self.consecutive_dynamic_candidate = max(0, self.consecutive_dynamic_candidate - 1)
                    self.consecutive_static_candidate = max(0, self.consecutive_static_candidate - 1)
                    self.auto_posture_change_confirm = 0

        else:
            # Currently in dynamic gesture (self.in_dynamic_gesture is True)
            if comp_motion >= self.auto_dynamic_motion_floor * 0.7:
                # Active movement resuming or continuing
                self.router_state = "dynamic"
                self.dynamic_pause_count = 0
            else:
                # Temporary low motion (pause / direction reversal)
                self.dynamic_pause_count += 1
                if self.dynamic_pause_count <= self.auto_dynamic_pause_max:
                    self.router_state = "dynamic_pause"
                else:
                    # Sustained inactivity: gesture has ended!
                    if self.auto_dynamic_episode_committed or self.auto_dynamic_last_committed is not None:
                        self.auto_terminal_guard_active = True
                        self.auto_terminal_landmarks = landmarks.copy()
                        self.auto_posture_change_confirm = 0
                    self.in_dynamic_gesture = False
                    self.router_state = "uncertain"
                    self.dynamic_pause_count = 0
                    self.dynamic_sequence_buffer.clear()
                    self.dynamic_onset_frame_id = None
                    self.auto_dynamic_buffer.clear()
                    self.auto_dynamic_last_committed = None
                    self.auto_dynamic_episode_committed = False

        # --- EXECUTE PREDICTION BASED ON ROUTER STATE ---
        if self.router_state == "static" and self.static_model is not None:
            raw_pred, conf = self.predict_static(landmarks)
            if self.auto_frame_counter % self.auto_log_interval == 0:
                print(f"[AUTO] frame={current_frame_id} state=STATIC wrist={wrist_motion:.4f} "
                      f"art={art_motion:.4f} motion={comp_motion:.4f} history={n_hist} "
                      f"dynamic_seq={len(self.dynamic_sequence_buffer)} onset={self.dynamic_onset_frame_id}")

        elif self.in_dynamic_gesture and self.dynamic_model is not None:
            # Append genuine frame to dynamic_sequence_buffer
            if len(self.dynamic_sequence_buffer) == 0 or not np.array_equal(self.dynamic_sequence_buffer[-1], landmarks):
                self.dynamic_sequence_buffer.append(landmarks)

            if len(self.dynamic_sequence_buffer) == self.sequence_length:
                raw_pred, conf = self.predict_dynamic(list(self.dynamic_sequence_buffer))
                print(f"[AUTO][DYNAMIC_PREDICT] frame={current_frame_id} sequence_len=30 "
                      f"prediction={raw_pred} confidence={conf*100:.1f}%")
            else:
                if self.auto_frame_counter % self.auto_log_interval == 0:
                    print(f"[AUTO] frame={current_frame_id} state={self.router_state.upper()} "
                          f"wrist={wrist_motion:.4f} art={art_motion:.4f} motion={comp_motion:.4f} "
                          f"history={n_hist} dynamic_seq={len(self.dynamic_sequence_buffer)} "
                          f"onset={self.dynamic_onset_frame_id}")
        else:
            if self.auto_frame_counter % self.auto_log_interval == 0:
                print(f"[AUTO] frame={current_frame_id} state=UNCERTAIN wrist={wrist_motion:.4f} "
                      f"art={art_motion:.4f} motion={comp_motion:.4f} history={n_hist} "
                      f"dynamic_seq={len(self.dynamic_sequence_buffer)} onset={self.dynamic_onset_frame_id}")

        return self.router_state, raw_pred, conf

    def process_frame(self, frame):
        """
        Process a single frame for GUI integration.
        Returns (annotated_frame, prediction, confidence, hands_detected).
        The old run() method is preserved for CLI usage.
        """
        frame = cv2.flip(frame, 1)
        landmarks, annotated_frame, hands_detected = self.extract_landmarks(frame)

        # Passive motion telemetry (Phase 2: measurement only)
        self._update_motion_telemetry(landmarks, hands_detected)

        current_pred = None
        current_conf = 0.0
        stable_pred = None

        is_rest = not hands_detected or self.is_hand_at_rest(landmarks)

        if hands_detected and not is_rest:
            if self.mode == 'static':
                if self.static_model is not None:
                    raw_pred, conf = self.predict_static(landmarks)
                    current_pred = raw_pred
                    current_conf = conf
                stable_pred = self.stabilize_prediction(current_pred, current_conf)
            elif self.mode == 'dynamic':
                if self.dynamic_model is not None:
                    self.sequence_buffer.append(landmarks)
                    if len(self.sequence_buffer) == self.sequence_length:
                        raw_pred, conf = self.predict_dynamic(list(self.sequence_buffer))
                        current_pred = raw_pred
                        current_conf = conf
                stable_pred = self.stabilize_prediction(current_pred, current_conf)
            elif self.mode in ['auto', 'both']:
                router_state, raw_pred, conf = self._evaluate_auto_router(landmarks, hands_detected)
                current_pred = raw_pred
                current_conf = conf
                if router_state in ['dynamic', 'dynamic_pause']:
                    stable_pred = self.stabilize_auto_dynamic(current_pred, current_conf)
                    self.prediction_buffer.append(None)
                elif router_state == 'static':
                    self.auto_dynamic_buffer.append(None)
                    stable_pred = self.stabilize_prediction(current_pred, current_conf)
                else:
                    self.prediction_buffer.append(None)
                    self.auto_dynamic_buffer.append(None)
                    stable_pred = None

            if stable_pred:
                self._handle_gesture_commit(stable_pred, current_conf)
            else:
                self._handle_gesture_reset()
        else:
            if self.mode in ['auto', 'both']:
                self._reset_auto_router()
            if self.mode == 'dynamic' and len(self.sequence_buffer) > 0:
                self.sequence_buffer.popleft()
            self.prediction_buffer.append(None)
            self.auto_dynamic_buffer.append(None)
            self._handle_gesture_reset()

        return annotated_frame, current_pred, current_conf, hands_detected

    def set_mode(self, new_mode):
        """Switch recognition mode at runtime. Used by the GUI app."""
        if new_mode == self.mode:
            return
        self.mode = new_mode
        self.prediction_buffer.clear()
        self.sequence_buffer.clear()
        self.active_gesture = None
        self.gesture_committed = False
        self.neutral_frame_count = 0
        self.raw_wrist_history.clear()
        self.prev_raw_landmarks = None
        self._reset_auto_router()

    def draw_ui(self, frame, prediction, confidence, fps):
        """Draw polished UI overlay on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent panels
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Prediction
        pred_text = prediction if prediction else "--"
        cv2.putText(frame, f"Sign: {pred_text}", (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if confidence and confidence > 0:
            bar_x = 350
            bar_w = int(200 * confidence)
            bar_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_w, 50), bar_color, -1)
            cv2.rectangle(frame, (bar_x, 30), (bar_x + 200, 50), (255, 255, 255), 1)
            cv2.putText(frame, f"{confidence * 100:.0f}%", (bar_x + 210, 48),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mode and FPS
        mode_display = f"AUTO ({self.router_state.upper()})" if self.mode == 'auto' else self.mode.upper()
        cv2.putText(frame, f"Mode: {mode_display}", (w - 240, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (w - 220, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Sentence and controls
        sentence_str = " ".join(self.current_sentence) if self.current_sentence else "(empty)"
        cv2.putText(frame, f"Sentence: {sentence_str}", (20, h - 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        controls = "C: Clear | Space: Speak | M: Mode | Q: Quit"
        cv2.putText(frame, controls, (20, h - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame

    def run(self):
        """Main inference loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        pTime = 0

        print(f"\n{'='*50}")
        print(f"  ISL Recognition System - {self.mode.upper()} mode")
        print(f"{'='*50}")
        print("Controls: Q=Quit  C=Clear  Space=Speak  M=Mode")
        print(f"{'='*50}\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame, hands_detected = self.extract_landmarks(frame)

            # Passive motion telemetry (Phase 2: measurement only)
            self._update_motion_telemetry(landmarks, hands_detected)

            current_pred = None
            current_conf = 0.0

            is_rest = not hands_detected or self.is_hand_at_rest(landmarks)

            if hands_detected and not is_rest:
                if self.mode == 'static':
                    if self.static_model is not None:
                        raw_pred, conf = self.predict_static(landmarks)
                        current_pred = raw_pred
                        current_conf = conf
                    stable_pred = self.stabilize_prediction(current_pred, current_conf)
                elif self.mode == 'dynamic':
                    if self.dynamic_model is not None:
                        self.sequence_buffer.append(landmarks)
                        if len(self.sequence_buffer) == self.sequence_length:
                            raw_pred, conf = self.predict_dynamic(list(self.sequence_buffer))
                            current_pred = raw_pred
                            current_conf = conf
                    stable_pred = self.stabilize_prediction(current_pred, current_conf)
                elif self.mode in ['auto', 'both']:
                    router_state, raw_pred, conf = self._evaluate_auto_router(landmarks, hands_detected)
                    current_pred = raw_pred
                    current_conf = conf
                    if router_state in ['dynamic', 'dynamic_pause']:
                        stable_pred = self.stabilize_auto_dynamic(current_pred, current_conf)
                        self.prediction_buffer.append(None)
                    elif router_state == 'static':
                        self.auto_dynamic_buffer.append(None)
                        stable_pred = self.stabilize_prediction(current_pred, current_conf)
                    else:
                        self.prediction_buffer.append(None)
                        self.auto_dynamic_buffer.append(None)
                        stable_pred = None

                if stable_pred:
                    if self._handle_gesture_commit(stable_pred, current_conf):
                        print(f"  Detected: {stable_pred} ({current_conf*100:.0f}%)")
                else:
                    self._handle_gesture_reset()
            else:
                if self.mode in ['auto', 'both']:
                    self._reset_auto_router()
                if self.mode == 'dynamic' and len(self.sequence_buffer) > 0:
                    self.sequence_buffer.popleft()
                self.prediction_buffer.append(None)
                self.auto_dynamic_buffer.append(None)
                self._handle_gesture_reset()

            cTime = time.time()
            fps = int(1 / (cTime - pTime)) if pTime > 0 else 0
            pTime = cTime

            output_frame = self.draw_ui(annotated_frame, current_pred, current_conf, fps)
            cv2.imshow('ISL Recognition System', output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_sentence()
                print("  Sentence cleared.")
            elif key == ord(' '):
                if self.current_sentence:
                    sentence = llm_translator.correct_grammar(self.current_sentence)
                    print(f"  Speaking: \"{sentence}\"")
                    self.tts.speak(sentence)
            elif key == ord('m'):
                if self.static_model and self.dynamic_model:
                    mode_cycle = {'auto': 'static', 'static': 'dynamic', 'dynamic': 'auto'}
                    new_mode = mode_cycle.get(self.mode, 'auto')
                    self.set_mode(new_mode)
                    print(f"  Switched to {self.mode.upper()} mode")
                else:
                    print("  Cannot toggle: both models must be loaded (use --mode both)")

        self.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nISL Recognition System closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ISL Recognition & Translation")
    parser.add_argument("--mode", type=str, default="static",
                        choices=['static', 'dynamic', 'both', 'auto'],
                        help="Recognition mode: static, dynamic, both, or auto")
    args = parser.parse_args()

    recognizer = ISLRecognizer(mode=args.mode)
    recognizer.run()
