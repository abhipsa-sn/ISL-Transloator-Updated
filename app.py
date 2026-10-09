"""
ISL Desktop Application — Indian Sign Language Detection & Translation
A modern desktop GUI built with CustomTkinter for real-time sign language recognition.
"""

import os
import sys
from unittest.mock import MagicMock

# Mock matplotlib before any imports to bypass Windows Application Control DLL block
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.ft2font'] = MagicMock()

import cv2
import time
import queue
import threading
import numpy as np
from PIL import Image, ImageTk

import customtkinter as ctk

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import ISLRecognizer
from src import config
from src import llm_translator


# ═══════════════════════════════════════════════════════════════════════════════
# Theme & Color Palette
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":       "#0D0D0D",
    "bg_panel":      "#1A1A2E",
    "bg_card":       "#16213E",
    "bg_input":      "#0F3460",
    "accent":        "#E94560",
    "accent_hover":  "#FF6B6B",
    "accent_green":  "#00E676",
    "accent_amber":  "#FFB300",
    "accent_blue":   "#448AFF",
    "text_primary":  "#F5F5F5",
    "text_secondary":"#B0BEC5",
    "text_muted":    "#607D8B",
    "border":        "#2A2A4A",
    "confidence_high":"#00E676",
    "confidence_mid": "#FFB300",
    "confidence_low": "#E94560",
    "success":       "#00C853",
    "warning":       "#FF6D00",
}

FONT_FAMILY = "Segoe UI"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class ISLDesktopApp(ctk.CTk):
    """Main desktop application for Indian Sign Language Detection & Translation."""

    def __init__(self):
        super().__init__()

        # ── Window Setup ──────────────────────────────────────────────────────
        self.title("ISL Translator — Indian Sign Language Detection")
        self.geometry("1280x720")
        self.minsize(1024, 600)
        self.configure(fg_color=COLORS["bg_dark"])

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── State Variables ───────────────────────────────────────────────────
        self.recognizer = None
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.worker_thread = None
        self.fps = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.hands_detected = False
        self.init_mode = "auto"     # Default mode: AUTO (conservative temporal router)
        self._translation_pinned = False  # True while translated text is shown in the textbox

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_header()
        self._build_main_content()
        self._build_footer()

        # ── Initialize Recognizer ─────────────────────────────────────────────
        self._init_recognizer()

        # ── Protocol ──────────────────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_header(self):
        """Build the top header bar."""
        header = ctk.CTkFrame(self, height=60, fg_color=COLORS["bg_panel"],
                              corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # App title
        title_label = ctk.CTkLabel(
            header, text="✋  ISL Translator",
            font=(FONT_FAMILY, 20, "bold"),
            text_color=COLORS["text_primary"]
        )
        title_label.pack(side="left", padx=20)

        # Subtitle
        subtitle = ctk.CTkLabel(
            header, text="Indian Sign Language Detection & Translation",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"]
        )
        subtitle.pack(side="left", padx=10)

        # Status indicator
        self.status_dot = ctk.CTkLabel(
            header, text="●",
            font=(FONT_FAMILY, 16),
            text_color=COLORS["text_muted"]
        )
        self.status_dot.pack(side="right", padx=5)

        self.status_label = ctk.CTkLabel(
            header, text="Initializing...",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"]
        )
        self.status_label.pack(side="right", padx=(0, 5))

        # FPS Counter
        self.fps_label = ctk.CTkLabel(
            header, text="FPS: --",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["accent_amber"]
        )
        self.fps_label.pack(side="right", padx=15)

    def _build_main_content(self):
        """Build the main 2-column layout: video + controls."""
        main_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"])
        main_frame.pack(fill="both", expand=True, padx=10, pady=(5, 0))
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # ── Left: Video Feed ──────────────────────────────────────────────────
        self._build_video_panel(main_frame)

        # ── Right: Controls & Info ────────────────────────────────────────────
        self._build_control_panel(main_frame)

    def _build_video_panel(self, parent):
        """Build the video feed display panel."""
        video_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                                   corner_radius=12, border_width=1,
                                   border_color=COLORS["border"])
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        # Video canvas label
        self.video_label = ctk.CTkLabel(
            video_frame, text="📷  Camera feed will appear here\nClick 'Start Camera' to begin",
            font=(FONT_FAMILY, 16),
            text_color=COLORS["text_muted"]
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def _build_control_panel(self, parent):
        """Build the right-side control panel."""
        panel = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                             corner_radius=12, border_width=1,
                             border_color=COLORS["border"], width=340)
        panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        panel.grid_propagate(False)

        # Scrollable content
        scroll = ctk.CTkScrollableFrame(panel, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # ── Detection Display ─────────────────────────────────────────────────
        det_section = self._section_label(scroll, "🎯  Detection")

        # Prediction display
        self.pred_label = ctk.CTkLabel(
            scroll, text="--",
            font=(FONT_FAMILY, 36, "bold"),
            text_color=COLORS["accent_green"]
        )
        self.pred_label.pack(pady=(5, 2))

        # Confidence bar
        conf_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        conf_frame.pack(fill="x", padx=15, pady=(0, 5))

        self.conf_label = ctk.CTkLabel(
            conf_frame, text="Confidence: 0%",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"]
        )
        self.conf_label.pack(anchor="w")

        self.conf_bar = ctk.CTkProgressBar(
            conf_frame, height=10,
            progress_color=COLORS["accent_green"],
            fg_color=COLORS["bg_card"],
            corner_radius=5
        )
        self.conf_bar.pack(fill="x", pady=(2, 0))
        self.conf_bar.set(0)

        # Hands status
        self.hands_label = ctk.CTkLabel(
            scroll, text="✋ No hands detected",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"]
        )
        self.hands_label.pack(pady=(0, 10))

        # ── Separator ─────────────────────────────────────────────────────────
        sep1 = ctk.CTkFrame(scroll, height=1, fg_color=COLORS["border"])
        sep1.pack(fill="x", padx=10, pady=5)

        # ── Sentence Builder ──────────────────────────────────────────────────
        self._section_label(scroll, "📝  Sentence")

        self.sentence_text = ctk.CTkTextbox(
            scroll, height=80,
            font=(FONT_FAMILY, 14),
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_primary"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=8,
            wrap="word"
        )
        self.sentence_text.pack(fill="x", padx=10, pady=5)
        self.sentence_text.insert("1.0", "(empty)")
        self.sentence_text.configure(state="disabled")

        # Sentence buttons
        btn_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(0, 5))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        self.speak_btn = ctk.CTkButton(
            btn_frame, text="🔊 Speak",
            font=(FONT_FAMILY, 13, "bold"),
            fg_color=COLORS["accent_blue"],
            hover_color="#5C9AFF",
            corner_radius=8, height=36,
            command=self._speak_sentence
        )
        self.speak_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.translate_btn = ctk.CTkButton(
            btn_frame, text="✨ Translate",
            font=(FONT_FAMILY, 13, "bold"),
            fg_color=COLORS["accent_green"],
            hover_color="#00C853",
            text_color="#0D0D0D",
            corner_radius=8, height=36,
            command=self._translate_sentence
        )
        self.translate_btn.grid(row=0, column=1, sticky="ew", padx=(3, 3))

        self.clear_btn = ctk.CTkButton(
            btn_frame, text="🗑 Clear",
            font=(FONT_FAMILY, 13, "bold"),
            fg_color=COLORS["bg_input"],
            hover_color="#1A5276",
            corner_radius=8, height=36,
            command=self._clear_sentence
        )
        self.clear_btn.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        # ── Separator ─────────────────────────────────────────────────────────
        sep2 = ctk.CTkFrame(scroll, height=1, fg_color=COLORS["border"])
        sep2.pack(fill="x", padx=10, pady=5)

        # ── Mode Selector ─────────────────────────────────────────────────────
        self._section_label(scroll, "⚙️  Mode")

        self.mode_var = ctk.StringVar(value=self.init_mode)
        self.mode_selector = ctk.CTkSegmentedButton(
            scroll, values=["auto", "static", "dynamic"],
            variable=self.mode_var,
            font=(FONT_FAMILY, 12, "bold"),
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent_hover"],
            unselected_color=COLORS["bg_card"],
            unselected_hover_color=COLORS["bg_input"],
            corner_radius=8,
            command=self._on_mode_change
        )
        self.mode_selector.pack(fill="x", padx=10, pady=5)

        # ── Separator ─────────────────────────────────────────────────────────
        sep3 = ctk.CTkFrame(scroll, height=1, fg_color=COLORS["border"])
        sep3.pack(fill="x", padx=10, pady=5)

        # ── Camera Controls ───────────────────────────────────────────────────
        self._section_label(scroll, "📷  Camera")

        self.camera_btn = ctk.CTkButton(
            scroll, text="▶  Start Camera",
            font=(FONT_FAMILY, 14, "bold"),
            fg_color=COLORS["success"],
            hover_color="#00E676",
            text_color=COLORS["bg_dark"],
            corner_radius=10, height=44,
            command=self._toggle_camera
        )
        self.camera_btn.pack(fill="x", padx=10, pady=5)

        # ── Separator ─────────────────────────────────────────────────────────
        sep4 = ctk.CTkFrame(scroll, height=1, fg_color=COLORS["border"])
        sep4.pack(fill="x", padx=10, pady=5)

        # ── TTS Settings ──────────────────────────────────────────────────────
        self._section_label(scroll, "🔊  Voice Settings")

        # Volume slider
        vol_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        vol_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(vol_frame, text="Volume",
                     font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        self.vol_slider = ctk.CTkSlider(
            vol_frame, from_=0, to=100,
            number_of_steps=20,
            progress_color=COLORS["accent_blue"],
            fg_color=COLORS["bg_card"],
            button_color=COLORS["accent_blue"],
            button_hover_color="#5C9AFF",
            command=self._on_volume_change
        )
        self.vol_slider.set(100)
        self.vol_slider.pack(fill="x", pady=2)

        # Rate slider
        rate_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        rate_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(rate_frame, text="Speed",
                     font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        self.rate_slider = ctk.CTkSlider(
            rate_frame, from_=-5, to=5,
            number_of_steps=10,
            progress_color=COLORS["accent_blue"],
            fg_color=COLORS["bg_card"],
            button_color=COLORS["accent_blue"],
            button_hover_color="#5C9AFF",
            command=self._on_rate_change
        )
        self.rate_slider.set(0)
        self.rate_slider.pack(fill="x", pady=2)

        # ── Detection History ─────────────────────────────────────────────────
        sep5 = ctk.CTkFrame(scroll, height=1, fg_color=COLORS["border"])
        sep5.pack(fill="x", padx=10, pady=5)

        self._section_label(scroll, "📋  History")

        self.history_text = ctk.CTkTextbox(
            scroll, height=100,
            font=(FONT_FAMILY, 11),
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_secondary"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=8,
            wrap="word"
        )
        self.history_text.pack(fill="x", padx=10, pady=5)
        self.history_text.configure(state="disabled")

    def _build_footer(self):
        """Build the bottom status bar."""
        footer = ctk.CTkFrame(self, height=30, fg_color=COLORS["bg_panel"],
                              corner_radius=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        info = ctk.CTkLabel(
            footer,
            text="  ISL Translator v1.0  •  MediaPipe + TensorFlow + SAPI TTS  •  Press Ctrl+Q to quit",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"]
        )
        info.pack(side="left", padx=10)

        self.model_info_label = ctk.CTkLabel(
            footer, text="",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"]
        )
        self.model_info_label.pack(side="right", padx=10)

    def _section_label(self, parent, text):
        """Create a section header label."""
        lbl = ctk.CTkLabel(
            parent, text=text,
            font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
            anchor="w"
        )
        lbl.pack(fill="x", padx=10, pady=(10, 2))
        return lbl

    # ═══════════════════════════════════════════════════════════════════════════
    # Recognizer Initialization
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_recognizer(self):
        """Initialize the ISL recognizer in a background thread to avoid blocking UI."""
        self._set_status("Loading models...", COLORS["accent_amber"])

        def init_worker():
            try:
                # Load BOTH models so the GUI mode-toggle works without restart.
                self.recognizer = ISLRecognizer(mode='both')
                self.recognizer.set_mode(self.init_mode)
                model_info = f"Mode: {self.init_mode.upper()}"
                if self.recognizer.static_model:
                    n = len(self.recognizer.static_classes)
                    model_info += f"  |  Static: {n} classes"
                if self.recognizer.dynamic_model:
                    n = len(self.recognizer.dynamic_classes)
                    model_info += f"  |  Dynamic: {n} classes"

                self.after(0, lambda: self.model_info_label.configure(text=model_info))
                self.after(0, lambda: self._set_status("Ready — Click 'Start Camera'",
                                                        COLORS["accent_green"]))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}", COLORS["accent"]))

        t = threading.Thread(target=init_worker, daemon=True)
        t.start()

    # ═══════════════════════════════════════════════════════════════════════════
    # Camera & Inference
    # ═══════════════════════════════════════════════════════════════════════════

    def _toggle_camera(self):
        """Start or stop the camera feed."""
        if self.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        """Open camera and start the inference worker thread."""
        if self.recognizer is None:
            self._set_status("Models not loaded yet, please wait...", COLORS["accent_amber"])
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("Error: Cannot open webcam!", COLORS["accent"])
            return

        self.is_running = True
        self.camera_btn.configure(
            text="⏹  Stop Camera",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text_primary"]
        )
        self._set_status("Running — Detecting signs...", COLORS["accent_green"])

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

        # Start GUI update loop
        self._update_gui()

    def _stop_camera(self):
        """Stop the camera and inference."""
        self.is_running = False

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        self.camera_btn.configure(
            text="▶  Start Camera",
            fg_color=COLORS["success"],
            hover_color="#00E676",
            text_color=COLORS["bg_dark"]
        )
        self._set_status("Stopped — Click 'Start Camera' to resume", COLORS["accent_amber"])

        # Reset video display
        self.video_label.configure(
            image=None,
            text="📷  Camera stopped\nClick 'Start Camera' to resume"
        )

    def _inference_loop(self):
        """Background worker thread: capture frames, run inference, push to queue."""
        prev_time = time.time()

        while self.is_running and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            # Process frame through recognizer
            annotated, pred, conf, hands = self.recognizer.process_frame(frame)

            # Calculate FPS
            curr_time = time.time()
            self.fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Push to queue (drop old frames if queue is full)
            try:
                self.frame_queue.put_nowait((frame_rgb, pred, conf, hands))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame_rgb, pred, conf, hands))
                except queue.Empty:
                    pass

    def _update_gui(self):
        """Poll the frame queue and update GUI widgets (runs on main thread)."""
        if not self.is_running:
            return

        try:
            frame_rgb, pred, conf, hands = self.frame_queue.get_nowait()

            # Resize frame to fit the video label
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w > 10 and label_h > 10:
                # Maintain aspect ratio
                fh, fw = frame_rgb.shape[:2]
                scale = min(label_w / fw, label_h / fh)
                new_w = int(fw * scale)
                new_h = int(fh * scale)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            else:
                frame_resized = frame_rgb

            # Convert to CTkImage
            img = Image.fromarray(frame_resized)
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img,
                                      size=(frame_resized.shape[1], frame_resized.shape[0]))
            self.video_label.configure(image=ctk_image, text="")
            self.video_label._current_image = ctk_image  # prevent garbage collection

            # Update prediction display
            self._update_prediction(pred, conf, hands)

            # Update FPS
            self.fps_label.configure(text=f"FPS: {self.fps}")

            # Update sentence display
            self._update_sentence_display()

        except queue.Empty:
            pass

        # Schedule next update (~30 FPS)
        if self.is_running:
            self.after(33, self._update_gui)

    def _update_prediction(self, pred, conf, hands):
        """Update prediction label and confidence bar."""
        if pred and conf > 0:
            self.pred_label.configure(text=pred)

            # Color based on confidence
            if conf > 0.8:
                color = COLORS["confidence_high"]
            elif conf > 0.5:
                color = COLORS["confidence_mid"]
            else:
                color = COLORS["confidence_low"]

            self.pred_label.configure(text_color=color)
            self.conf_bar.set(conf)
            self.conf_bar.configure(progress_color=color)
            self.conf_label.configure(text=f"Confidence: {conf * 100:.1f}%")
        else:
            self.pred_label.configure(text="--", text_color=COLORS["text_muted"])
            self.conf_bar.set(0)
            self.conf_label.configure(text="Confidence: 0%")

        # Hands status
        if hands:
            self.hands_label.configure(
                text="✋ Hands detected",
                text_color=COLORS["accent_green"]
            )
        else:
            self.hands_label.configure(
                text="✋ No hands detected",
                text_color=COLORS["text_muted"]
            )

    def _update_sentence_display(self):
        """Update the sentence text widget from recognizer state."""
        if self.recognizer is None:
            return

        # Don't overwrite a pinned translation result
        if self._translation_pinned:
            return

        sentence = self.recognizer.current_sentence
        sentence_str = " ".join(sentence) if sentence else "(empty)"

        self.sentence_text.configure(state="normal")
        self.sentence_text.delete("1.0", "end")
        self.sentence_text.insert("1.0", sentence_str)
        self.sentence_text.configure(state="disabled")

        # Update history with latest detections
        if sentence:
            self.history_text.configure(state="normal")
            history_str = " → ".join(sentence[-15:])  # Show last 15 words
            self.history_text.delete("1.0", "end")
            self.history_text.insert("1.0", history_str)
            self.history_text.configure(state="disabled")


    # ═══════════════════════════════════════════════════════════════════════════
    # Controls & Actions
    # ═══════════════════════════════════════════════════════════════════════════

    def _translate_sentence(self):
        """Run LLM grammar correction and show the result in the sentence textbox."""
        if not self.recognizer or not self.recognizer.current_sentence:
            return
        raw_words = self.recognizer.current_sentence[:]
        self.translate_btn.configure(state="disabled", text="⏳ Translating…")
        self.update()

        def _do_translate():
            corrected = llm_translator.correct_grammar(raw_words)
            self.after(0, lambda: self._apply_translation(corrected))

        threading.Thread(target=_do_translate, daemon=True).start()

    def _apply_translation(self, corrected: str):
        """Apply the corrected sentence to the UI textbox (called on main thread)."""
        self._translation_pinned = True  # Prevent GUI loop from overwriting
        self.sentence_text.configure(state="normal")
        self.sentence_text.delete("1.0", "end")
        self.sentence_text.insert("1.0", corrected)
        self.sentence_text.configure(state="disabled")
        self.translate_btn.configure(state="normal", text="✨ Translate")
        self._log_history(f"✨ Translated: \"{corrected}\"")

    def _speak_sentence(self):
        """Speak the accumulated sentence after grammar correction."""
        if self.recognizer and self.recognizer.current_sentence:
            raw_words = self.recognizer.current_sentence[:]
            self.speak_btn.configure(state="disabled", text="⏳ Speaking…")
            self.update()

            def _do_speak():
                sentence = llm_translator.correct_grammar(raw_words)
                self.after(0, lambda: self._apply_translation(sentence))
                self.recognizer.tts.speak(sentence)
                self.after(0, lambda: self.speak_btn.configure(state="normal", text="🔊 Speak"))

            threading.Thread(target=_do_speak, daemon=True).start()

    def _clear_sentence(self):
        """Clear the sentence buffer and reset gesture state."""
        if self.recognizer:
            self._translation_pinned = False  # Unpin so display reflects raw words again
            self.recognizer.clear_sentence()
            self._update_sentence_display()
            self._log_history("🗑 Sentence cleared")


    def _on_mode_change(self, new_mode):
        """Handle mode switch."""
        if self.recognizer:
            self.recognizer.set_mode(new_mode)
            self._set_status(f"Mode: {new_mode.upper()}", COLORS["accent_green"])
            self._log_history(f"⚙️ Switched to {new_mode.upper()} mode")

            # Update model info in footer
            model_info = f"Mode: {new_mode.upper()}"
            if self.recognizer.static_model:
                n = len(self.recognizer.static_classes)
                model_info += f"  |  Static: {n} classes"
            if self.recognizer.dynamic_model:
                n = len(self.recognizer.dynamic_classes)
                model_info += f"  |  Dynamic: {n} classes"
            self.model_info_label.configure(text=model_info)

    def _on_volume_change(self, value):
        """Handle volume slider change."""
        if self.recognizer and self.recognizer.tts:
            self.recognizer.tts.set_volume(int(value))

    def _on_rate_change(self, value):
        """Handle rate slider change."""
        if self.recognizer and self.recognizer.tts:
            self.recognizer.tts.set_rate(int(value))

    def _log_history(self, text):
        """Append a log entry to the history panel."""
        self.history_text.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.history_text.insert("end", f"\n[{timestamp}] {text}")
        self.history_text.see("end")
        self.history_text.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def _set_status(self, text, color):
        """Update the header status indicator."""
        self.status_label.configure(text=text, text_color=color)
        self.status_dot.configure(text_color=color)

    def _on_close(self):
        """Clean shutdown."""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.recognizer:
            try:
                self.recognizer.landmarker.close()
            except Exception:
                pass
            try:
                self.recognizer.tts.stop()
            except Exception:
                pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ISLDesktopApp()
    app.mainloop()
