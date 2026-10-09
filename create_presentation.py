import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    
    # 16:9 Widescreen aspect ratio
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    blank_layout = prs.slide_layouts[6] # Blank slide layout

    # Theme Colors
    COLOR_BG = RGBColor(15, 23, 42)          # Slate 900
    COLOR_CARD = RGBColor(30, 41, 59)        # Slate 800
    COLOR_TITLE = RGBColor(248, 250, 252)     # White
    COLOR_ACCENT = RGBColor(56, 189, 248)     # Sky Blue / Cyan
    COLOR_GREEN = RGBColor(74, 222, 128)      # Emerald Green
    COLOR_TEXT = RGBColor(226, 232, 240)      # Slate 200
    COLOR_MUTED = RGBColor(148, 163, 184)     # Slate 400
    COLOR_PROMPT_BG = RGBColor(15, 23, 42)    # Dark background for prompt
    COLOR_BORDER = RGBColor(51, 65, 85)       # Border slate

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.fill.background() # No line

    def add_header(slide, title_text, subtitle_text=None):
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Segoe UI"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = COLOR_TITLE

        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.name = "Segoe UI"
            p2.font.size = Pt(12)
            p2.font.color.rgb = COLOR_MUTED
            p2.space_before = Pt(4)

    def add_image_prompt_box(slide, left, top, width, height, prompt_text):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = COLOR_CARD
        box.line.color.rgb = COLOR_ACCENT
        box.line.width = Pt(1.5)

        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.2)
        tf.margin_bottom = Inches(0.2)

        p = tf.paragraphs[0]
        p.text = "🖼️ IMAGE PLACEHOLDER & PROMPT"
        p.font.name = "Segoe UI"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        p.space_after = Pt(8)

        p2 = tf.add_paragraph()
        p2.text = f"Prompt: {prompt_text}"
        p2.font.name = "Consolas"
        p2.font.size = Pt(10)
        p2.font.color.rgb = COLOR_GREEN

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1: Front Page (Blank as requested)
    # ═══════════════════════════════════════════════════════════════════════════
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)
    # (Left completely blank as per request)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2: Outline
    # ═══════════════════════════════════════════════════════════════════════════
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Presentation Agenda & Outline", "Overview of topics covered in this presentation")

    agenda_box = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.3))
    agenda_box.fill.solid()
    agenda_box.fill.fore_color.rgb = COLOR_CARD
    agenda_box.line.color.rgb = COLOR_BORDER

    tf2 = agenda_box.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = Inches(0.4)

    items = [
        ("01. Introduction & Motivation", "Background of ISL, problem statement, key challenges & objectives"),
        ("02. Literature Study", "Comparative survey of existing ISL models, methodologies, and limitations"),
        ("03. Technical Details", "Data pipeline, 3D hand landmark extraction, landmark normalization, dual models (Dense NN + LSTM), prediction stabilization, and Windows SAPI TTS"),
        ("04. Results & Discussion", "Model performance evaluation, 99.75% static & 100% dynamic accuracy, classification reports, and latency analysis"),
        ("05. Conclusion & Future Scope", "Summary of achievements, current constraints, and future expansion roadmap"),
        ("06. References", "Academic citations and research literature")
    ]

    for idx, (title, desc) in enumerate(items):
        p = tf2.paragraphs[0] if idx == 0 else tf2.add_paragraph()
        p.text = title
        p.font.name = "Segoe UI"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(12)

        p_desc = tf2.add_paragraph()
        p_desc.text = f"      {desc}"
        p_desc.font.name = "Segoe UI"
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = COLOR_TEXT

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3: Introduction
    # ═══════════════════════════════════════════════════════════════════════════
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "1. Introduction & Motivation", "Indian Sign Language (ISL) Recognition System")

    # Left content box
    content3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    content3.fill.solid()
    content3.fill.fore_color.rgb = COLOR_CARD
    content3.line.color.rgb = COLOR_BORDER
    tf3 = content3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_top = tf3.margin_right = tf3.margin_bottom = Inches(0.3)

    intro_bullets = [
        ("Background & Motivation", "• ~1.8 Million deaf individuals in India (Census 2011)\n• ISL is a predominantly 2-handed sign language with distinct grammar\n• Severe shortage of human interpreters creates communication barriers"),
        ("Problem Statement", "• Existing solutions require expensive depth cameras or sensor gloves\n• Need for a lightweight, software-only system operating via webcam"),
        ("Core Objectives", "• MediaPipe 3D HandLandmarker feature extraction (126 features)\n• Dual model pipeline: Dense NN (Static) + LSTM (Dynamic)\n• Asynchronous Windows SAPI Text-to-Speech translation output\n• Responsive CustomTkinter desktop GUI")
    ]

    for idx, (heading, text) in enumerate(intro_bullets):
        p = tf3.paragraphs[0] if idx == 0 else tf3.add_paragraph()
        p.text = heading
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(10)

        p_t = tf3.add_paragraph()
        p_t.text = text
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT
        p_t.space_before = Pt(2)

    # Right Image Prompt Box
    add_image_prompt_box(
        slide3, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.3),
        "A modern banner illustration showing a person performing two-handed Indian Sign Language gestures in front of a webcam, with glowing 3D hand skeleton landmark points overlaid on the hands, connecting to a neural network diagram and outputting text and speech soundwaves. Dark theme, blue & neon green accents."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 4: Literature Study Table
    # ═══════════════════════════════════════════════════════════════════════════
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "2. Literature Study", "Comparative summary of existing ISL recognition research")

    rows, cols = 6, 6
    left, top, width, height = Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.4)
    table_shape = slide4.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(0.6)   # Sl. No.
    table.columns[1].width = Inches(2.1)   # Journal, Publisher & Year
    table.columns[2].width = Inches(2.2)   # Title & Author
    table.columns[3].width = Inches(2.2)   # Methodology
    table.columns[4].width = Inches(2.3)   # Findings
    table.columns[5].width = Inches(2.333) # Limitations

    headers = ["Sl. No.", "Journal, Publisher & Year", "Title & Author", "Methodology", "Findings", "Limitations"]
    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_CARD
        for p in cell.text_frame.paragraphs:
            p.font.name = "Segoe UI"
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = COLOR_ACCENT

    lit_data = [
        ("1", "Multimedia Tools & Applications\n(Springer, 2024)", "Automatic ISL Recognition using MediaPipe & LSTM\n(Khartheesvar et al.)", "MediaPipe Holistic + 2-layer LSTM sequence model", "High accuracy on isolated dynamic ISL signs", "High computational cost due to full holistic mesh extraction"),
        ("2", "Electronics\n(MDPI, 2024)", "Isolated ISL Recognition via Hybrid CNN-LSTM\n(Kumari & Anand)", "3D CNN feature extractor + Attention LSTM", "Effective spatial-temporal frame weighting", "High RAM & processing overhead from raw RGB frames"),
        ("3", "Electronics\n(MDPI, 2022)", "DeepSign: Sign Language Detection\n(Kothadiya et al.)", "OpenPose keypoints + Dense Neural Network", "Low parameter count for static poses", "Fails under hand self-occlusion; poor dynamic gesture handling"),
        ("4", "Information\n(MDPI, 2023)", "LiST: Lightweight Framework for ISL\n(Al-Qurishi et al.)", "Lightweight 2D CNN + GRU sequence decoder", "Reduced model latency for mobile devices", "Limited vocabulary; sensitive to lighting variations"),
        ("5", "Array\n(Elsevier, 2023)", "ISL Recognition using SURF with SVM & CNN\n(Sahoo et al.)", "SURF descriptor + hybrid SVM & CNN classifier", "Robust static hand posture classification", "Segmentation fails under varying skin tones or shadows")
    ]

    for row_idx, row_values in enumerate(lit_data, start=1):
        for col_idx, val in enumerate(row_values):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(22, 33, 56) if row_idx % 2 == 1 else COLOR_CARD
            for p in cell.text_frame.paragraphs:
                p.font.name = "Segoe UI"
                p.font.size = Pt(8.5)
                p.font.color.rgb = COLOR_TEXT

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 5: Technical Details — Data Pipeline
    # ═══════════════════════════════════════════════════════════════════════════
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "3. Technical Details — Feature Pipeline & Normalization", "System Architecture (Part 1)")

    box5 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    box5.fill.solid()
    box5.fill.fore_color.rgb = COLOR_CARD
    box5.line.color.rgb = COLOR_BORDER
    tf5 = box5.text_frame
    tf5.word_wrap = True
    tf5.margin_left = tf5.margin_top = tf5.margin_right = tf5.margin_bottom = Inches(0.3)

    pipe_points = [
        ("MediaPipe 3D Hand Landmarks", "• 21 3D landmarks (x, y, z) extracted per hand\n• 2 hands x 21 landmarks x 3 coordinates = 126 features/frame\n• Single-hand detections are zero-padded to maintain 126-D shape"),
        ("Wrist-Relative Normalization", "• Translation Invariance: Subtract wrist position (L₀) from all 21 points\n• Scale Invariance: Divide by max Euclidean distance from wrist\n• Math: L̂ᵢ = (Lᵢ - L₀) / maxⱼ ||Lⱼ - L₀||"),
        ("Data Collection & Augmentation", "• Static dataset: 500 frames collected per sign class\n• Dynamic dataset: 50 sequences x 30 frames per sign class (~1 sec)\n• 3x Data Augmentation: Gaussian noise (σ=0.01), scaling, translation")
    ]

    for idx, (heading, text) in enumerate(pipe_points):
        p = tf5.paragraphs[0] if idx == 0 else tf5.add_paragraph()
        p.text = heading
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(10)

        p_t = tf5.add_paragraph()
        p_t.text = text
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT
        p_t.space_before = Pt(2)

    add_image_prompt_box(
        slide5, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.3),
        "A detailed technical diagram illustrating the data pipeline: Webcam video frame input -> 21 3D hand landmark mesh nodes mapped onto both hands -> 126-dimensional coordinate vector extraction -> Wrist-relative translation and scale normalization formula box -> Processed feature vector array."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 6: Technical Details — Dual Model Architecture
    # ═══════════════════════════════════════════════════════════════════════════
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "3. Technical Details — Dual Model Architectures", "System Architecture (Part 2)")

    box6 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    box6.fill.solid()
    box6.fill.fore_color.rgb = COLOR_CARD
    box6.line.color.rgb = COLOR_BORDER
    tf6 = box6.text_frame
    tf6.word_wrap = True
    tf6.margin_left = tf6.margin_top = tf6.margin_right = tf6.margin_bottom = Inches(0.3)

    arch_points = [
        ("Why Dual Architectures?", "• Static Signs (Namaste, OK): Classified from single frames\n• Dynamic Signs (Hello, Thank You, Bye): Classified over temporal motion"),
        ("Static Model — Dense Neural Network", "• Input: (126,) normalized feature vector\n• Architecture: Dense(256) → BatchNorm → Dropout(0.3) → Dense(128) → BatchNorm → Dropout(0.3) → Dense(64) → Softmax(N)\n• Size: ~75,781 parameters (296 KB)"),
        ("Dynamic Model — LSTM Network", "• Input: (30, 126) sequence over 30 consecutive frames\n• Architecture: LSTM(128, return_seq=True) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(64) → Dense(32) → Softmax(N)\n• Captures temporal velocity and direction of hand motion")
    ]

    for idx, (heading, text) in enumerate(arch_points):
        p = tf6.paragraphs[0] if idx == 0 else tf6.add_paragraph()
        p.text = heading
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(10)

        p_t = tf6.add_paragraph()
        p_t.text = text
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT
        p_t.space_before = Pt(2)

    add_image_prompt_box(
        slide6, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.3),
        "Side-by-side architectural diagram comparing a Dense Neural Network layer block and an LSTM Recurrent Neural Network unrolled over 30 time steps. Include layer shapes, dropout blocks, batch normalization blocks, and softmax output nodes. Purple, teal, and dark slate color scheme."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 7: Technical Details — Inference & TTS
    # ═══════════════════════════════════════════════════════════════════════════
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "3. Technical Details — Real-Time Inference & TTS", "System Architecture (Part 3)")

    box7 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    box7.fill.solid()
    box7.fill.fore_color.rgb = COLOR_CARD
    box7.line.color.rgb = COLOR_BORDER
    tf7 = box7.text_frame
    tf7.word_wrap = True
    tf7.margin_left = tf7.margin_top = tf7.margin_right = tf7.margin_bottom = Inches(0.3)

    infer_points = [
        ("Prediction Stabilization Engine", "• 5-Frame Rolling Buffer: Requires identical predictions over 5 frames\n• Confidence Threshold: Confirms sign only if confidence > 0.6 / 0.7\n• Prevents flickering during hand transition movements"),
        ("Asynchronous Windows SAPI Speech", "• Bypasses pyttsx3 freezing by using Windows native SAPI (SpVoice)\n• Uses SVSFlagsAsync for non-blocking voice output during video capture\n• Cooldown period of 1.0s prevents repetitive word playback"),
        ("CustomTkinter Desktop Application", "• Threaded architecture: Background thread handles camera & inference\n• Main thread renders CustomTkinter GUI smoothly at ~30 FPS\n• Native controls for sentence clearing, mode toggle, volume & rate sliders")
    ]

    for idx, (heading, text) in enumerate(infer_points):
        p = tf7.paragraphs[0] if idx == 0 else tf7.add_paragraph()
        p.text = heading
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(10)

        p_t = tf7.add_paragraph()
        p_t.text = text
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT
        p_t.space_before = Pt(2)

    add_image_prompt_box(
        slide7, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.3),
        "High-resolution screenshot mockup of a modern dark-mode desktop GUI application window titled 'ISL Translator'. Left side shows a live webcam feed of a person doing sign language with green skeletal hand tracking lines. Right side shows bold prediction 'NAMASTE', green confidence progress bar, sentence box, mode buttons, and sliders."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 8: Results and Discussion
    # ═══════════════════════════════════════════════════════════════════════════
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "4. Results & Discussion", "Model performance, accuracy, and real-time latency evaluation")

    box8 = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    box8.fill.solid()
    box8.fill.fore_color.rgb = COLOR_CARD
    box8.line.color.rgb = COLOR_BORDER
    tf8 = box8.text_frame
    tf8.word_wrap = True
    tf8.margin_left = tf8.margin_top = tf8.margin_right = tf8.margin_bottom = Inches(0.3)

    res_points = [
        ("Experimental Performance Summary", "• Static Model (Dense NN): 99.75% Test Accuracy (Early stopped at epoch 6)\n• Dynamic Model (LSTM): 100.00% Test Accuracy (Best weights at epoch 93)\n• Inference Latency: ~3ms per frame (Static), ~12ms per sequence (LSTM)"),
        ("Classification Report (Precision / Recall / F1)", "• Hello: Precision 1.00 | Recall 1.00 | F1-Score 1.00\n• Namaste: Precision 1.00 | Recall 1.00 | F1-Score 1.00\n• Bye: Precision 1.00 | Recall 1.00 | F1-Score 1.00\n• Thank You: Precision 1.00 | Recall 1.00 | F1-Score 1.00\n• Sorry: Precision 1.00 | Recall 1.00 | F1-Score 1.00"),
        ("Key Key Discussion Insights", "• Wrist normalization successfully eliminated distance/scale variance\n• Zero false-positive triggers observed in dynamic stability buffer")
    ]

    for idx, (heading, text) in enumerate(res_points):
        p = tf8.paragraphs[0] if idx == 0 else tf8.add_paragraph()
        p.text = heading
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(10)

        p_t = tf8.add_paragraph()
        p_t.text = text
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT
        p_t.space_before = Pt(2)

    add_image_prompt_box(
        slide8, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.3),
        "Dual visualization chart panel: Left side showing training and validation accuracy and loss curves over epochs for an LSTM model; Right side showing a 5x5 confusion matrix heatmap with high diagonal intensity values labeled with class names (Hello, Namaste, Bye, Thank You, Sorry). Dark background with green & blue heatmaps."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 9: Conclusion & Future Scope
    # ═══════════════════════════════════════════════════════════════════════════
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "5. Conclusion & Future Scope", "Project contributions and future development roadmap")

    box9 = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.3))
    box9.fill.solid()
    box9.fill.fore_color.rgb = COLOR_CARD
    box9.line.color.rgb = COLOR_BORDER
    tf9 = box9.text_frame
    tf9.word_wrap = True
    tf9.margin_left = tf9.margin_top = tf9.margin_right = tf9.margin_bottom = Inches(0.3)

    concl_points = [
        ("Project Summary & Contributions", "• Built a complete, real-time vision-based ISL translation system\n• Achieved 99.75% static accuracy and 100.00% dynamic sequence accuracy\n• Lightweight 126-D feature pipeline runs smoothly on standard CPUs\n• Delivered a responsive CustomTkinter GUI with Windows SAPI TTS integration"),
        ("Current Limitations", "• Vocabulary currently covers 5 core dynamic/static phrases\n• Requires visible hands in webcam view without heavy motion blur"),
        ("Future Expansion Roadmap", "1. Vocabulary Expansion: Extend to full ISL Alphabets (A-Z) & Numbers (0-9)\n2. Continuous Sign Recognition: Integrate Connectionist Temporal Classification (CTC)\n3. NLP Translation Layer: Add Transformer for ISL gloss to English grammar translation")
    ]

    for idx, (heading, text) in enumerate(concl_points):
        p = tf9.paragraphs[0] if idx == 0 else tf9.add_paragraph()
        p.text = heading
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        if idx > 0:
            p.space_before = Pt(10)

        p_t = tf9.add_paragraph()
        p_t.text = text
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT
        p_t.space_before = Pt(2)

    add_image_prompt_box(
        slide9, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.3),
        "An infographic roadmap showing three future milestones: Step 1 (Alphabet & Digit Expansion with 26 letters), Step 2 (Continuous Sign Language Recognition with CTC Loss), and Step 3 (NLP Grammar Translation into full sentences). Modern icon-based flowchart on dark background with connecting light trails."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 10: References
    # ═══════════════════════════════════════════════════════════════════════════
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "6. References", "Academic research publications and literature citations")

    box10 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.3))
    box10.fill.solid()
    box10.fill.fore_color.rgb = COLOR_CARD
    box10.line.color.rgb = COLOR_BORDER
    tf10 = box10.text_frame
    tf10.word_wrap = True
    tf10.margin_left = tf10.margin_top = tf10.margin_right = tf10.margin_bottom = Inches(0.3)

    refs = [
        "1. Khartheesvar, G., et al. (2024). \"Automatic Indian Sign Language Recognition using MediaPipe Holistic and LSTM Network\". Multimedia Tools and Applications, Springer (SCI). DOI: 10.1007/s11042-023-17361-y",
        "2. Al-Qurishi, M., et al. (2023). \"LiST: A Lightweight Framework for Continuous ISL Translation\". Information, MDPI (Scopus). DOI: 10.3390/info14020079",
        "3. Kumari, D., & Anand, R. S. (2024). \"Isolated Video-Based Sign Language Recognition Using Hybrid CNN-LSTM with Attention\". Electronics, MDPI (SCI). DOI: 10.3390/electronics13071229",
        "4. Kothadiya, D., et al. (2022). \"DeepSign: Sign Language Detection and Recognition Using Deep Learning\". Electronics, MDPI (SCI). DOI: 10.3390/electronics11111780",
        "5. IEEE Access. (2025). \"Real-Time Sign Language-to-Speech Translation with AI-Powered Wearable Technology\". IEEE Access, (SCI). DOI: 10.1109/ACCESS.2025.3602794",
        "6. JSIR. (2022). \"Real-time Static and Dynamic Sign Language Recognition using Deep Learning\". Journal of Scientific and Industrial Research, (SCI). DOI: 10.56042/jsir.v81i11.60865",
        "7. IEEE MysuruCon. (2021). \"Evaluation of Machine Learning Models for Real-Time Sign Recognition\". IEEE MysuruCon 2021. DOI: 10.1109/MysuruCon52639.2021.9641518",
        "8. Sahoo, S., et al. (2023). \"Indian Sign Language Recognition Using SURF with SVM and CNN\". Array, Elsevier (Scopus). DOI: 10.1016/j.array.2023.100257"
    ]

    for idx, ref in enumerate(refs):
        p = tf10.paragraphs[0] if idx == 0 else tf10.add_paragraph()
        p.text = ref
        p.font.name = "Segoe UI"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT
        if idx > 0:
            p.space_before = Pt(8)

    output_path = r"c:\Pratyush\Project\MajorProject\ISL_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")

if __name__ == "__main__":
    create_deck()
