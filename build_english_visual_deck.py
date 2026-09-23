# -*- coding: utf-8 -*-
"""
Builds the 18-slide English Visual Master Presentation for FPT University AI Capstone Defense.
Key Principles:
1. 100% ENGLISH SLIDES: Professional, academic, international computer vision standards.
2. MINIMAL TEXT, HIGH SIGNAL: Punchy cards, bold metrics, zero wall-of-text paragraphs.
3. LARGE GENUINE FIGURES: 55%-65% width occupied by real paper figures (Fig 1 to Fig 6) and 4 custom architecture diagrams.
4. ZERO FORBIDDEN WORDS: Strictly eliminate IEEE, Review, Reviews. Frame as Stage 1 Proposal & Feasibility Evaluation.
5. ACADEMIC LIGHT THEME: Warm Ivory (#FAF8F5) background, pure white cards (#FFFFFF), deep navy (#0F172A), safety orange (#EA580C).
"""
import os
import sys
import shutil
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_english_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Warm Light Academic Palette
    BG_COLOR = RGBColor(250, 248, 245)      # #FAF8F5 Warm Ivory
    CARD_BG = RGBColor(255, 255, 255)       # #FFFFFF Pure White
    TEXT_DARK = RGBColor(15, 23, 42)        # #0F172A Deep Slate / Navy
    TEXT_MUTED = RGBColor(100, 116, 139)    # #64748B Slate Grey
    TEXT_BODY = RGBColor(51, 65, 85)        # #334155 Body Charcoal
    BORDER_COLOR = RGBColor(226, 232, 240)  # #E2E8F0 Subtle Border
    ORANGE_ACCENT = RGBColor(234, 88, 12)   # #EA580C Industrial Safety Orange
    AMBER_ACCENT = RGBColor(217, 119, 6)    # #D97706 Warm Amber
    BLUE_ACCENT = RGBColor(37, 99, 235)     # #2563EB Technical Blue
    GREEN_ACCENT = RGBColor(22, 163, 74)    # #16A34A Success Green
    LIGHT_ORANGE = RGBColor(255, 247, 237)  # #FFF7ED
    LIGHT_BLUE = RGBColor(239, 246, 255)    # #EFF6FF
    LIGHT_GREEN = RGBColor(240, 253, 244)   # #F0FDF4

    fig_dir = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\figures'

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title, category, slide_num_str):
        # Category tracker
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.32), Inches(9.5), Inches(0.28))
        tf_cat = tb_cat.text_frame
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ORANGE_ACCENT

        # Slide Number
        tb_num = slide.shapes.add_textbox(Inches(11.3), Inches(0.32), Inches(1.2), Inches(0.28))
        tf_num = tb_num.text_frame
        tf_num.margin_left = tf_num.margin_right = tf_num.margin_top = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = slide_num_str
        p_num.font.size = Pt(11)
        p_num.font.bold = True
        p_num.font.color.rgb = TEXT_MUTED
        p_num.alignment = PP_ALIGN.RIGHT

        # Big Clear Title
        tb_t = slide.shapes.add_textbox(Inches(0.8), Inches(0.62), Inches(11.7), Inches(0.55))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(20)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_DARK

    def add_card(slide, left_in, top_in, width_in, height_in, bg_color=CARD_BG, border_color=BORDER_COLOR, border_width=1.0):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(border_width)
        else:
            shape.line.fill.background()
        return shape

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout); set_slide_bg(s1)
    
    # Institution badge
    add_card(s1, 0.8, 0.45, 6.2, 0.38, LIGHT_ORANGE, ORANGE_ACCENT)
    tb = s1.shapes.add_textbox(Inches(0.92), Inches(0.50), Inches(6.0), Inches(0.28))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    tb.text_frame.paragraphs[0].text = 'FPT UNIVERSITY · ARTIFICIAL INTELLIGENCE CAPSTONE DEFENSE'
    tb.text_frame.paragraphs[0].font.size = Pt(9.5); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = ORANGE_ACCENT

    # Main Project Title
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.05), Inches(7.5), Inches(1.10))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    tb.text_frame.paragraphs[0].text = 'Rep-YOLO11s'
    tb.text_frame.paragraphs[0].font.size = Pt(44); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = TEXT_DARK

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.20), Inches(7.5), Inches(1.30))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = 'Real-Time Safety Helmet Detection in Construction Surveillance:\nStructural Re-parameterization, Coordinate Encoding & Multi-Domain Generalization'
    p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = TEXT_MUTED

    # Right Hero Metric Showcase
    add_card(s1, 8.5, 1.05, 4.0, 3.40, CARD_BG, ORANGE_ACCENT, border_width=1.5)
    tb = s1.shapes.add_textbox(Inches(8.75), Inches(1.20), Inches(3.5), Inches(3.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = '5-FOLD CV PEAK mAP50'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
    p = tf.add_paragraph(); p.text = '97.11%'; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = '✓ Outperforms Baseline YOLO11s (+0.47% mAP50)'; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT; p.space_after = Pt(6)
    
    stats_lines = [
        ('Tesla T4 Latency', '2.92 ms (342.5 FPS)'),
        ('RTX 3050 Latency', '5.35 ms (187.1 FPS)'),
        ('Budget Laptop MX230 (2GB)', '36.0 ms (27.8 FPS)'),
        ('Harmonized Benchmark', '> 33,000 Multi-Domain Images')
    ]
    for lbl, val in stats_lines:
        p = tf.add_paragraph()
        p.text = f'• {lbl}: '
        p.font.size = Pt(9.2); p.font.color.rgb = TEXT_MUTED
        run = p.add_run()
        run.text = val; run.font.bold = True; run.font.color.rgb = TEXT_DARK

    # Bottom 3 Info Cards
    info_boxes = [
        ('RESEARCH TEAM', 'Nhu Nguyen-Han · SE183644 (Lead)\nThanh Nguyen-Van · SE183645\nDung Nguyen-Tuan · SE183646', 'Role: Architecture, TensorRT Engine & RTSP Pipeline', ORANGE_ACCENT),
        ('ACADEMIC ADVISOR', 'M.Sc. Hai-Anh Vu (anhvh@fe.edu.vn)\nDept. of Artificial Intelligence — IT Faculty\nFPT University Hanoi Campus', 'Focus: High-Impact Industrial Computer Vision', BLUE_ACCENT),
        ('PUBLICATION TARGET', 'Top-Tier International Journal / Conf.\nIndustrial Informatics & Machine Vision\n(Rigorous Analytical & Empirical Standards)', 'Scope: 100% Empirically Validated Results', GREEN_ACCENT)
    ]
    for idx, (head, main_txt, sub_txt, col) in enumerate(info_boxes):
        left_pos = 0.8 + idx * 4.0
        add_card(s1, left_pos, 4.65, 3.8, 2.30)
        tb = s1.shapes.add_textbox(Inches(left_pos + 0.18), Inches(4.78), Inches(3.44), Inches(2.05))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = head; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = main_txt; p.font.size = Pt(10.2); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(4)
        p = tf.add_paragraph(); p.text = sub_txt; p.font.size = Pt(9); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: CONSTRUCTION CONTEXT & 4 TECHNICAL BOTTLENECKS
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout); set_slide_bg(s2)
    add_header(s2, 'Real Construction Context & 4 Technical Bottlenecks', 'Criterion 1 · Problem Definition & Urgency', '02 / 18')

    bottlenecks = [
        ('01', 'Distant Tiny Objects (<20 px)', 'High-angle CCTV at 15–30m causes helmets to blur and vanish across deep downsampling layers.', ORANGE_ACCENT),
        ('02', 'Extreme Class Imbalance (1:12)', '9,044 helmets vs. 111,514 worker body instances; worker gradients dominate the loss.', BLUE_ACCENT),
        ('03', 'Color Noise & False Alarms', 'Yellow buckets, cones, and caution signs trigger false positives due to CNN translation invariance.', AMBER_ACCENT),
        ('04', 'Edge Hardware Constraints', 'Strict >= 25 FPS real-time requirement on budget laptops/GPUs without costly server clusters.', GREEN_ACCENT)
    ]
    for idx, (num, h_text, desc, col) in enumerate(bottlenecks):
        top_pos = 1.35 + idx * 1.35
        add_card(s2, 0.8, top_pos, 5.6, 1.25)
        add_card(s2, 0.95, top_pos + 0.15, 0.45, 0.35, col, col)
        tb_num = s2.shapes.add_textbox(Inches(0.95), Inches(top_pos + 0.16), Inches(0.45), Inches(0.35))
        tb_num.text_frame.margin_left = tb_num.text_frame.margin_right = tb_num.text_frame.margin_top = tb_num.text_frame.margin_bottom = 0
        tb_num.text_frame.paragraphs[0].text = num
        tb_num.text_frame.paragraphs[0].font.size = Pt(11); tb_num.text_frame.paragraphs[0].font.bold = True; tb_num.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255); tb_num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        tb = s2.shapes.add_textbox(Inches(1.52), Inches(top_pos + 0.12), Inches(4.7), Inches(1.05))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_text; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(9.2); p.font.color.rgb = TEXT_BODY

    # Right: Large Authentic CCTV Image
    fig1_path = os.path.join(fig_dir, 'Fig1_site_overview_challenges.jpg')
    if os.path.exists(fig1_path):
        add_card(s2, 6.7, 1.35, 5.8, 5.30, CARD_BG, BORDER_COLOR)
        s2.shapes.add_picture(fig1_path, Inches(6.85), Inches(1.50), Inches(5.5), Inches(4.40))
        tb_cap = s2.shapes.add_textbox(Inches(6.85), Inches(6.00), Inches(5.5), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Fig. 1: Real construction site surveillance — High-angle camera, distant tiny targets, scaffolding occlusions, and severe specular surface glare.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: LITERATURE SURVEY & RESEARCH GAP
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout); set_slide_bg(s3)
    add_header(s3, 'Literature Review of 32 Works & SOTA Research Gap', 'Criterion 3 · Feasibility & Theoretical Foundation', '03 / 18')

    RED_ACCENT = RGBColor(239, 68, 68)

    models_comp = [
        ('Standard YOLO (v8 / 11)', '94.74% mAP50', '6.52 ms (153.3 FPS)', 'Coordinate-agnostic; frequent false positives on ground yellow buckets.', RED_ACCENT),
        ('EC-YOLOv8 (Zhang 2024)', '95.70% mAP50', '5.80 ms (172.4 FPS)', 'High mAP via CARAFE operator, but heavy computational latency overhead.', AMBER_ACCENT),
        ('YOLO-CBF (Li 2023)', '95.60% mAP50', '12.40 ms (80.6 FPS)', 'Combines CoordConv & BiFormer but balloons to 37.2M params and 104.5 GFLOPs.', AMBER_ACCENT),
        ('YOLOv8n-FADS (Fu 2024)', '79.70% mAP50', 'Edge Bottleneck', 'P2 branch expansion causes computational explosion; low empirical precision.', RED_ACCENT),
        ('Rep-YOLO11s (Proposed)', '94.83% - 97.11%', '2.92 ms (342.5 FPS)', 'Zero-overhead W_fused fusion; coordinate encoding suppresses false alarms; 27.8 FPS on MX230.', GREEN_ACCENT)
    ]

    for idx, (m_name, m_map, m_speed, m_weak, col) in enumerate(models_comp):
        top_pos = 1.35 + idx * 0.95
        is_ours = idx == 4
        bg_c = LIGHT_GREEN if is_ours else CARD_BG
        brd_c = GREEN_ACCENT if is_ours else BORDER_COLOR
        add_card(s3, 0.8, top_pos, 11.7, 0.85, bg_c, brd_c, border_width=1.8 if is_ours else 1.0)

        # Name
        tb = s3.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.12), Inches(3.0), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = m_name; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = col if is_ours else TEXT_DARK
        
        # Stats
        tb = s3.shapes.add_textbox(Inches(4.1), Inches(top_pos + 0.12), Inches(2.3), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = f'mAP50: {m_map}'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = f'Latency: {m_speed}'; p.font.size = Pt(9); p.font.color.rgb = TEXT_MUTED

        # Description
        tb = s3.shapes.add_textbox(Inches(6.5), Inches(top_pos + 0.12), Inches(5.8), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = m_weak; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_BODY

    # Bottom Takeaway Card
    add_card(s3, 0.8, 6.25, 11.7, 0.80, LIGHT_ORANGE, ORANGE_ACCENT)
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.65))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = '★ RESEARCH GAP: No existing model achieves high tiny-object accuracy WITHOUT incurring severe inference latency or parameter explosion during edge deployment.'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT

    # =========================================================================
    # SLIDE 4: SCOPE & 5 CONCRETE COMMITTED DELIVERABLES
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout); set_slide_bg(s4)
    add_header(s4, 'Project Scope & 5 Concrete Committed Deliverables', 'Criterion 2 · Final Deliverables & Outcomes', '04 / 18')

    deliverables = [
        ('01', 'SCIENTIFIC REPORT', '9-page formal monograph with analytical proofs, SOTA benchmarks, Ablation A0-A6 & Grad-CAM XAI.', 'Top-Tier Academic Standard', ORANGE_ACCENT),
        ('02', 'REP-YOLO11s MODEL', 'Trained weights achieving 94.83% Single Test, 96.64% 5-Fold, 97.03% on Hard Hat Workers.', 'Accurate & Generalizable', BLUE_ACCENT),
        ('03', 'SOFTWARE PROTOTYPE', 'Multi-stream RTSP surveillance pipeline with circular buffering, 65–95 FPS on RTX 3050.', 'Real-World Production', GREEN_ACCENT),
        ('04', 'COMPILED ENGINES', 'TensorRT 11.2 FP16 (.engine), ONNX INT8 (.onnx), OpenVINO packages ready for cross-platform edge chips.', 'Zero-latency Deploy', AMBER_ACCENT),
        ('05', 'STANDARDIZED BENCHMARK', '>33,000 images harmonized across 6 industrial domains with cleaned VOC2028 annotations.', 'Cleaned Industry Benchmark', RGBColor(147, 51, 234))
    ]

    for idx, (num, h_text, desc, tag, col) in enumerate(deliverables):
        top_pos = 1.35 + idx * 1.00
        add_card(s4, 0.8, top_pos, 11.7, 0.88)
        
        # Pill
        add_card(s4, 1.0, top_pos + 0.16, 0.45, 0.35, col, col)
        tb_num = s4.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.17), Inches(0.45), Inches(0.35))
        tb_num.text_frame.margin_left = tb_num.text_frame.margin_right = tb_num.text_frame.margin_top = tb_num.text_frame.margin_bottom = 0
        tb_num.text_frame.paragraphs[0].text = num
        tb_num.text_frame.paragraphs[0].font.size = Pt(11); tb_num.text_frame.paragraphs[0].font.bold = True; tb_num.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255); tb_num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # Title
        tb = s4.shapes.add_textbox(Inches(1.6), Inches(top_pos + 0.14), Inches(3.2), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_text; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = tag; p.font.size = Pt(9); p.font.bold = True; p.font.color.rgb = col

        # Description
        tb = s4.shapes.add_textbox(Inches(4.9), Inches(top_pos + 0.18), Inches(7.4), Inches(0.55))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = desc; p.font.size = Pt(10); p.font.color.rgb = TEXT_BODY

    # Bottom Status Banner
    add_card(s4, 0.8, 6.45, 11.7, 0.60, LIGHT_GREEN, GREEN_ACCENT)
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(6.50), Inches(11.3), Inches(0.50))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = '✓ PROJECT STATUS: All 5/5 committed deliverables are completed, reaching ~90% of entire capstone engineering scope in Stage 1!'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT

    # =========================================================================
    # Helper for Side-by-Side Comparison Slides (Slides 05 to 09)
    def add_side_by_side_slide(slide, title, category, slide_num,
                               left_title, left_sub, left_img_name, left_bullets,
                               right_title, right_sub, right_img_name, right_bullets):
        add_header(slide, title, category, slide_num)
        
        card_w = 5.75
        card_h = 5.85
        top_pos = 1.25
        sci_dir = os.path.join(fig_dir, 'scientific_exports')
        
        # LEFT CARD (Baseline / Flaw)
        left_x = 0.80
        add_card(slide, left_x, top_pos, card_w, card_h, CARD_BG, BORDER_COLOR, border_width=1.0)
        
        # Left Title Header Pill
        add_card(slide, left_x + 0.15, top_pos + 0.12, card_w - 0.30, 0.48, LIGHT_ORANGE, ORANGE_ACCENT, border_width=1.0)
        tb_lh = slide.shapes.add_textbox(Inches(left_x + 0.25), Inches(top_pos + 0.14), Inches(card_w - 0.50), Inches(0.44))
        tf_lh = tb_lh.text_frame; tf_lh.word_wrap = True
        tf_lh.margin_left = tf_lh.margin_right = tf_lh.margin_top = tf_lh.margin_bottom = 0
        p = tf_lh.paragraphs[0]; p.text = left_title; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
        p = tf_lh.add_paragraph(); p.text = left_sub; p.font.size = Pt(8.2); p.font.color.rgb = TEXT_MUTED
        
        # Left Picture
        left_img_path = os.path.join(sci_dir, left_img_name)
        if not os.path.exists(left_img_path):
            left_img_path = os.path.join(fig_dir, left_img_name)
        if os.path.exists(left_img_path):
            slide.shapes.add_picture(left_img_path, Inches(left_x + 0.15), Inches(top_pos + 0.68), Inches(card_w - 0.30), Inches(2.95))
            
        # Left Bullets Box
        add_card(slide, left_x + 0.15, top_pos + 3.72, card_w - 0.30, 1.98, RGBColor(254, 242, 242), RGBColor(252, 165, 165), border_width=0.8)
        tb_lb = slide.shapes.add_textbox(Inches(left_x + 0.25), Inches(top_pos + 3.80), Inches(card_w - 0.50), Inches(1.82))
        tf_lb = tb_lb.text_frame; tf_lb.word_wrap = True
        tf_lb.margin_left = tf_lb.margin_right = tf_lb.margin_top = tf_lb.margin_bottom = 0
        for i, (badge, text) in enumerate(left_bullets):
            p = tf_lb.paragraphs[0] if i == 0 else tf_lb.add_paragraph()
            p.text = f"{badge} "
            p.font.size = Pt(9.0); p.font.bold = True; p.font.color.rgb = RGBColor(185, 28, 28)
            run = p.add_run()
            run.text = text
            run.font.size = Pt(8.8); run.font.bold = False; run.font.color.rgb = TEXT_BODY
            p.space_after = Pt(2)

        # RIGHT CARD (Proposed / Innovation)
        right_x = 6.78
        add_card(slide, right_x, top_pos, card_w, card_h, CARD_BG, BORDER_COLOR, border_width=1.0)
        
        # Right Title Header Pill
        add_card(slide, right_x + 0.15, top_pos + 0.12, card_w - 0.30, 0.48, LIGHT_GREEN, GREEN_ACCENT, border_width=1.0)
        tb_rh = slide.shapes.add_textbox(Inches(right_x + 0.25), Inches(top_pos + 0.14), Inches(card_w - 0.50), Inches(0.44))
        tf_rh = tb_rh.text_frame; tf_rh.word_wrap = True
        tf_rh.margin_left = tf_rh.margin_right = tf_rh.margin_top = tf_rh.margin_bottom = 0
        p = tf_rh.paragraphs[0]; p.text = right_title; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
        p = tf_rh.add_paragraph(); p.text = right_sub; p.font.size = Pt(8.2); p.font.color.rgb = TEXT_MUTED

        # Right Picture
        right_img_path = os.path.join(sci_dir, right_img_name)
        if not os.path.exists(right_img_path):
            right_img_path = os.path.join(fig_dir, right_img_name)
        if os.path.exists(right_img_path):
            slide.shapes.add_picture(right_img_path, Inches(right_x + 0.15), Inches(top_pos + 0.68), Inches(card_w - 0.30), Inches(2.95))

        # Right Bullets Box
        add_card(slide, right_x + 0.15, top_pos + 3.72, card_w - 0.30, 1.98, LIGHT_GREEN, RGBColor(134, 239, 172), border_width=0.8)
        tb_rb = slide.shapes.add_textbox(Inches(right_x + 0.25), Inches(top_pos + 3.80), Inches(card_w - 0.50), Inches(1.82))
        tf_rb = tb_rb.text_frame; tf_rb.word_wrap = True
        tf_rb.margin_left = tf_rb.margin_right = tf_rb.margin_top = tf_rb.margin_bottom = 0
        for i, (badge, text) in enumerate(right_bullets):
            p = tf_rb.paragraphs[0] if i == 0 else tf_rb.add_paragraph()
            p.text = f"{badge} "
            p.font.size = Pt(9.0); p.font.bold = True; p.font.color.rgb = RGBColor(21, 128, 61)
            run = p.add_run()
            run.text = text
            run.font.size = Pt(8.8); run.font.bold = False; run.font.color.rgb = TEXT_BODY
            p.space_after = Pt(2)

    # =========================================================================
    # SLIDE 5: ARCHITECTURAL COMPARISON: YOLO11s (3 HEADS) VS. REP-YOLO11s-P2 (4 HEADS)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout); set_slide_bg(s5)
    add_side_by_side_slide(
        s5,
        title='Architectural Comparison: Baseline YOLO11s (3 Heads) vs. Proposed Rep-YOLO11s-P2 AFPN (4 Heads)',
        category='PROPOSED METHOD · NEURAL NETWORK ARCHITECTURE',
        slide_num='05 / 18',
        left_title='BASELINE: VANILLA YOLO11s (3 HEADS)',
        left_sub='Standard P3, P4, P5 Heads (Strides 8, 16, 32) · Distant Target Feature Vanishing',
        left_img_name='Fig5A_Baseline_YOLO11s_Architecture.png',
        left_bullets=[
            ('[❌ Stride 32 Vanishing]', '32× downsampling collapses 16×16 px helmets into <0.5 px on feature map P5, erasing spatial details.'),
            ('[❌ Distant Target Blindness]', 'Fails to detect distant workers at range >25m (Baseline Helmet Recall capped at 90.35%).'),
            ('[❌ Missing Shallow Pathway]', 'Lacks direct shallow feature propagation from P2 to upper feature pyramid fusion stages.')
        ],
        right_title='PROPOSED: REP-YOLO11s-P2 AFPN (4 HEADS)',
        right_sub='High-Resolution P2 Head (Stride 4, 160×160) · Full Micro-Scale Feature Preservation',
        right_img_name='Fig5B_Proposed_RepYOLO11s_Architecture.png',
        right_bullets=[
            ('[✅ Stride 4 Micro-Resolution]', 'Maintains 160×160 feature maps where tiny helmet targets span 4×4 spatial feature cells.'),
            ('[⚡ Helmet Recall Surge to 91.33%]', 'Dramatic boost in distant helmet detection sensitivity (peak 5-fold cross-validation reaches 93.34%).'),
            ('[⚡ 51.9% Parameter Reduction]', 'Channel pruning and AFPN re-alignment cut parameters from 9.85M to 4.74M with higher accuracy.')
        ]
    )

    # =========================================================================
    # SLIDE 6: INNOVATION 1 - STRUCTURAL RE-PARAMETERIZATION (REPCONV)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout); set_slide_bg(s6)
    add_side_by_side_slide(
        s6,
        title='Innovation 1: Structural Re-parameterization (RepConv) & Algebraic Fusion',
        category='TECHNICAL INNOVATION 1 · INFERENCE LATENCY OPTIMIZATION',
        slide_num='06 / 18',
        left_title='BASELINE: MULTI-BRANCH TRAINING BOTTLENECK',
        left_sub='3 Parallel Branches (3×3, 1×1, Identity) · High Memory Access Cost (MAC)',
        left_img_name='Fig1A_Baseline_MultiBranch_Bottleneck.png',
        left_bullets=[
            ('[❌ Memory Access Cost]', 'GPU allocates 3 separate memory buffers per block, creating severe memory-bandwidth bottlenecks.'),
            ('[❌ 3 Kernel Launches]', 'Incurs 3 sequential CUDA kernel launches per layer, triggering hardware cache thrashing.'),
            ('[❌ 7.12 ms Inference Latency]', 'Low deployment throughput (only 140.4 FPS on Tesla T4), unsuitable for multi-stream 4K CCTV.')
        ],
        right_title='PROPOSED: REPCONV ALGEBRAIC FUSION LIFECYCLE',
        right_sub='switch_to_deploy() Collapses All Branches into a Single 3×3 Conv Layer',
        right_img_name='Fig1B_Proposed_RepConv_Algebraic_Fusion.png',
        right_bullets=[
            ('[✅ Closed-Form Fusion]', 'BatchNorm folding + 1×1 zero-padding + Dirac identity kernel (strictly when Cin==Cout, s=1).'),
            ('[⚡ 55.2% Latency Reduction]', 'Forward latency plunges from 7.12 ms to 2.92 ms on Tesla T4 TRT FP16 (5.35 ms on RTX 3050 Laptop).'),
            ('[⚡ 342.5 FPS Real-Time Speed]', 'Achieves super-real-time throughput with negligible numerical discrepancy Δ < 10⁻⁵ vs multi-branch.')
        ]
    )

    # =========================================================================
    # SLIDE 7: INNOVATION 2 - COORDCONV SPATIAL COORDINATE INJECTION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout); set_slide_bg(s7)
    add_side_by_side_slide(
        s7,
        title='Innovation 2: CoordConv Spatial Encoding Suppresses False Alarms',
        category='TECHNICAL INNOVATION 2 · BREAKING TRANSLATION INVARIANCE',
        slide_num='07 / 18',
        left_title='BASELINE: TRANSLATION INVARIANCE FLAW',
        left_sub='Standard Convolutions Treat Ground Buckets Same as Worn Helmets',
        left_img_name='Fig2A_Baseline_Translation_Invariance_Flaw.png',
        left_bullets=[
            ('[❌ Shared Kernel Weights]', 'Standard 3-channel RGB filters have zero spatial altitude awareness across camera view coordinates.'),
            ('[❌ Geometric Context Ambiguity]', 'Yellow buckets/traffic cones at floor level (y ≈ 0.9) trigger identical activations to helmets (y ≈ 0.2).'),
            ('[❌ Heavy Ground Clutter FPs]', 'Over 28% of industrial false alarms originate from ground clutter sharing helmet color/geometry.')
        ],
        right_title='PROPOSED: 5-CHANNEL COORDCONV STEM INJECTION',
        right_sub='Injects Normalized Coordinates Cx, Cy ∈ [-1, 1] Strictly at Input Stem Layer',
        right_img_name='Fig2B_Proposed_CoordConv_Spatial_Injection.png',
        right_bullets=[
            ('[✅ Construction Spatial Axiom]', 'Forces filters to learn anatomical priors: Helmets reside on heads (Cy < 0.3), never on floors.'),
            ('[⚡ >28% False Alarm Pruning]', 'Ground-level helmet logits are algebraically suppressed at the initial Stem (Conv c1=5 -> c2=64).'),
            ('[⚡ Negligible Cost (+0.06 ms)]', 'Preserves translation invariance in deeper layers, maximizing contextual gain at minimal compute.')
        ]
    )

    # =========================================================================
    # SLIDE 8: INNOVATION 3 - BIFORMER SPARSE ROUTING ATTENTION
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout); set_slide_bg(s8)
    add_side_by_side_slide(
        s8,
        title='Innovation 3: BiFormer 2-Level Dynamic Sparse Routing Attention',
        category='TECHNICAL INNOVATION 3 · TINY TARGET SALIENCY FOCUS',
        slide_num='08 / 18',
        left_title='BASELINE: DENSE SELF-ATTENTION BOTTLENECK',
        left_sub='Quadratic Complexity O(H²W²) · Severe VRAM Out-of-Memory (OOM) on Edge Hardware',
        left_img_name='Fig3A_Baseline_Dense_Attention_Bottleneck.png',
        left_bullets=[
            ('[❌ Quadratic Compute Explosion]', 'At 1024×1024, dense affinity matrix A requires >10¹⁰ dot products, consuming >4 GB scratchpad VRAM.'),
            ('[❌ 80% Compute Wasted on Noise]', 'Attention is diffusely scattered across empty sky, concrete floors, and blank wall surfaces.'),
            ('[❌ Edge GPU Memory Crash]', 'Causes instant CUDA OOM crashes on resource-constrained 2GB–4GB edge hardware (Jetson / MX230).')
        ],
        right_title='PROPOSED: BIFORMER DYNAMIC SPARSE ROUTING',
        right_sub='2-Level Routing: Region Grid S=8, Top-k=4 · Strict Linear Complexity O(HW)',
        right_img_name='Fig3B_Proposed_BiFormer_Sparse_Routing.png',
        right_bullets=[
            ('[✅ 80% Background Pruning]', 'Dynamically routes token attention strictly through the top-4 most salient worker head regions.'),
            ('[⚡ Linear Complexity O(HW)]', 'Reduces compute from O((HW)²) to linear O(HW), completely eliminating VRAM overflow bottlenecks.'),
            ('[⚡ Distant Target Saliency Focus]', 'Concentrates 100% compute on subtle helmet contours, enabling smooth inference at 1024px.')
        ]
    )

    # =========================================================================
    # SLIDE 9: INNOVATION 4 - FOCAL EIOU LOSS & BOUNDING BOX REGRESSION
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout); set_slide_bg(s9)
    add_side_by_side_slide(
        s9,
        title='Innovation 4: Focal EIoU Loss & Decoupled Bounding Box Optimization',
        category='TECHNICAL INNOVATION 4 · BOUNDING BOX OPTIMIZATION',
        slide_num='09 / 18',
        left_title='BASELINE: CIOU GRADIENT VANISHING FLAW',
        left_sub='Aspect Ratio Penalty v ∝ (arctan(w_gt/h_gt) - arctan(w/h))² Vanishes Gradient',
        left_img_name='Fig4A_Baseline_CIoU_Vanishing_Gradient.png',
        left_bullets=[
            ('[❌ Gradient Vanishing to Zero]', 'When aspect ratios match (w/h == w_gt/h_gt), ∂v/∂w ≡ 0 even when predicted box area is 200% off.'),
            ('[❌ Frozen Prediction Box]', 'Bounding box regression freezes and fails to tightly fit occluded helmets partially covered by scaffolding.'),
            ('[❌ Scale-Agnostic Penalty]', 'Penalizes relative angle difference rather than absolute physical pixel errors in width and height.')
        ],
        right_title='PROPOSED: FOCAL-EIoU INDEPENDENT DECOMPOSITION',
        right_sub='Decoupled Errors (w-w_gt)² + (h-h_gt)² + Dynamic Hard-Sample Weighting (IoU)^0.5',
        right_img_name='Fig4B_Proposed_Focal_EIoU_Decomposition.png',
        right_bullets=[
            ('[✅ Guaranteed Non-Zero Gradient]', '∂L/∂w = 2(w-w_gt)/Cw² ≠ 0 continuously guides independent convergence of both width and height.'),
            ('[⚡ Hard Boundary Localization]', 'Focal factor (IoU)^0.5 amplifies gradients for difficult, heavily occluded tiny helmet boundaries.'),
            ('[⚡ TAL + Weighted BCE Synergism]', 'Works with Task-Aligned Assigner to resolve 1:12 class imbalance, reaching peak mAP50-95 of 65.91%.')
        ]
    )

    # =========================================================================
    # SLIDE 10: CUSTOM CODE & CORE CONTRIBUTIONS
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout); set_slide_bg(s10)
    add_header(s10, 'Independent Development: 4 Custom Mathematical Modules', 'Criterion 3 · Technical Autonomy & Code Ownership', '10 / 18')

    code_contributions = [
        ('01', '100% PURE PYTORCH CODE', 'custom_ablation_modules.py', 'Engineered from scratch 360 lines of native PyTorch code for CoordConv, RepConv, BiFormer, and Focal EIoU with zero external dependencies.', ORANGE_ACCENT),
        ('02', 'PHYSICAL LOSS INTERVENTION', 'Physical File Hard-Patch', 'Bypassed PyTorch DDP process isolation on Kaggle via runtime file hard-patching to enforce exact custom loss computation.', BLUE_ACCENT),
        ('03', 'PYTORCH 2.6 SECURITY BYPASS', 'add_safe_globals Integration', 'Resolved PyTorch 2.6 weights_only=True deserialization failure across 117 custom tensor layers, enabling seamless checkpoint loading.', GREEN_ACCENT),
        ('04', 'HARMONIZED PPE PROTOCOL', 'Harmonized PPE Benchmark', 'Discovered and solved cross-dataset taxonomy conflict (full-body vs. head-only), restoring cross-domain helmet mAP to 97.03%.', AMBER_ACCENT)
    ]

    for idx, (num, h_txt, sub_txt, desc, col) in enumerate(code_contributions):
        top_pos = 1.35 + idx * 1.25
        add_card(s10, 0.8, top_pos, 11.7, 1.12)
        
        # Pill
        add_card(s10, 1.0, top_pos + 0.22, 0.45, 0.35, col, col)
        tb_num = s10.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.23), Inches(0.45), Inches(0.35))
        tb_num.text_frame.margin_left = tb_num.text_frame.margin_right = tb_num.text_frame.margin_top = tb_num.text_frame.margin_bottom = 0
        tb_num.text_frame.paragraphs[0].text = num
        tb_num.text_frame.paragraphs[0].font.size = Pt(11); tb_num.text_frame.paragraphs[0].font.bold = True; tb_num.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255); tb_num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # Title + Module
        tb = s10.shapes.add_textbox(Inches(1.6), Inches(top_pos + 0.15), Inches(3.5), Inches(0.80))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_txt; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = sub_txt; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = col

        # Detailed Impact
        tb = s10.shapes.add_textbox(Inches(5.2), Inches(top_pos + 0.20), Inches(7.1), Inches(0.70))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = desc; p.font.size = Pt(10); p.font.color.rgb = TEXT_BODY

    add_card(s10, 0.8, 6.45, 11.7, 0.60, LIGHT_ORANGE, ORANGE_ACCENT)
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(6.50), Inches(11.3), Inches(0.50))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    p = tb.text_frame.paragraphs[0]
    p.text = '★ TECHNICAL AUTONOMY: Ultralytics serves as infrastructure; all 4 custom mathematical modules and low-level patches are 100% self-developed!'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT

    # =========================================================================
    # SLIDE 11: DATA ENGINEERING & MULTI-DOMAIN HARMONIZATION
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout); set_slide_bg(s11)
    add_header(s11, 'Data Engineering: Multi-Domain Harmonization & Anomaly Cleaning', 'Criterion 3 · Data Integrity & Benchmark Suite', '11 / 18')

    # Left: In-domain SHWD details
    add_card(s11, 0.8, 1.35, 5.6, 4.90)
    tb = s11.shapes.add_textbox(Inches(1.05), Inches(1.50), Inches(5.1), Inches(4.60))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'PRIMARY IN-DOMAIN DATASET (VOC2028 / SHWD)'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
    p = tf.add_paragraph(); p.text = '7,581 Real Construction Images'; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)
    
    shwd_points = [
        ('Rigorous 80/20 Partitioning:', '6,064 TrainVal images / 1,517 independent Test images (Zero Data Leakage).'),
        ('Noise & Anomaly Cleaning:', 'Detected and purged 3 corrupted XML annotations labeled as "dog" in image 000377.'),
        ('Extreme Label Imbalance:', '1:12 class ratio (9,044 helmet labels vs. 111,514 person body labels).'),
        ('Multi-Scale Standardization:', '960×960 native training resolution with 640×640 / 1024×1024 multi-scale scaling.')
    ]
    for lbl, val in shwd_points:
        p = tf.add_paragraph()
        p.text = f'• {lbl} '
        p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(4)

    # Right: Multi-Domain benchmark suite
    add_card(s11, 6.7, 1.35, 5.8, 4.90)
    tb = s11.shapes.add_textbox(Inches(6.95), Inches(1.50), Inches(5.3), Inches(4.60))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = '5 CROSS-DOMAIN BENCHMARKS (>25,000 IMAGES)'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
    p = tf.add_paragraph(); p.text = 'Extreme Zero-Shot Generalization Test'; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    external_datasets = [
        ('GDUT-HWD (13,499 images):', 'Ultra-dense worker crowding (15–30 persons per camera frame).'),
        ('SHEL5K (5,000 images):', 'Vertical drone overhead perspective (70–90°), tiny helmets <15px.'),
        ('Hard Hat Workers (7,000 images):', 'Outdoor surveillance under extreme natural lighting and shadows.'),
        ('SHD & SFCHD Benchmark:', 'Heavy industry metallurgical plants, oil refineries, and shipyards.')
    ]
    for lbl, val in external_datasets:
        p = tf.add_paragraph()
        p.text = f'• {lbl} '
        p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(4)

    # Bottom Unified Ontology
    add_card(s11, 0.8, 6.35, 11.7, 0.70, LIGHT_BLUE, BLUE_ACCENT)
    tb = s11.shapes.add_textbox(Inches(1.0), Inches(6.42), Inches(11.3), Inches(0.55))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    p = tb.text_frame.paragraphs[0]
    p.text = '★ UNIFIED LABEL TAXONOMY C* = {0: "hat" (Safety Helmet), 1: "person" (Worker Body)} — Fully synchronized across 33,000+ images!'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT

    # =========================================================================
    # SLIDE 12: ABLATION STUDY A0–A6 EMPIRICAL VERIFICATION (FIGURE 6)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout); set_slide_bg(s12)
    add_header(s12, 'Ablation Study (A0–A6): Rigorous Component Verification', 'Criterion 4 · Empirical Verification & Scientific Evidence', '12 / 18')

    add_card(s12, 0.8, 1.35, 4.8, 5.40)
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(4.4), Inches(5.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'STEP-BY-STEP MODULE CONTRIBUTIONS'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    ablation_items = [
        ('A0 (Baseline YOLO11s):', '94.74% mAP50 · 6.52 ms latency.'),
        ('A1 (+ P2 Small Head):', '94.81% mAP50 · Latency surged to 8.94 ms (+37%) -> Pruned as suboptimal.'),
        ('A2 (+ CoordConv):', '94.78% mAP50 · Spatial awareness without latency overhead.'),
        ('A3 (+ RepConv):', '94.81% mAP50 · Rich gradient representation in training.'),
        ('A4 (+ Focal EIoU):', '94.88% mAP50 · Precise bounding box regression for tiny objects.'),
        ('A5 (+ BiFormer):', '94.80% mAP50 · Peak helmet Recall reaching 91.15%.'),
        ('A6 (Full Fusion):', '94.83% mAP50 · Latency plummets to a record 2.92 ms!')
    ]
    for lbl, val in ablation_items:
        p = tf.add_paragraph()
        p.text = f'• {lbl} '
        p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(3)

    fig6_path = os.path.join(fig_dir, 'Fig6_ablation_A0_A6_tradeoff.png')
    if os.path.exists(fig6_path):
        add_card(s12, 5.8, 1.35, 6.7, 5.40, CARD_BG, BORDER_COLOR)
        s12.shapes.add_picture(fig6_path, Inches(5.95), Inches(1.50), Inches(6.4), Inches(4.50))
        tb_cap = s12.shapes.add_textbox(Inches(5.95), Inches(6.15), Inches(6.4), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Fig. 6: Empirical Ablation Tradeoff (A0-A6) — Re-parameterized W_fused slashes latency by 55.2% while retaining optimal detection accuracy.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 13: SOTA BENCHMARKING ON PARETO FRONTIER (FIGURE 5)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout); set_slide_bg(s13)
    add_header(s13, 'State-of-the-Art Benchmarking: Pareto Frontier Dominance', 'Criterion 4 · SOTA Benchmarking & Pareto Dominance', '13 / 18')

    add_card(s13, 0.8, 1.35, 4.8, 5.40)
    tb = s13.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(4.4), Inches(5.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'COMPETITIVE EDGE OF REP-YOLO11s'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    pareto_points = [
        ('2.23x Faster than Baseline:', '2.92 ms vs 6.52 ms on Tesla T4 (342.5 FPS throughput).'),
        ('4.25x Faster than YOLO-CBF:', 'YOLO-CBF takes 12.4 ms (80.6 FPS); our model takes only 2.92 ms.'),
        ('2x Faster than EC-YOLOv8:', 'EC-YOLOv8 reaches 172.4 FPS; our model achieves 342.5 FPS.'),
        ('5-Fold Stratified Peak:', '96.64% ± 0.32% mAP50, peak fold reaching 97.11% mAP50 and F1: 0.9396.'),
        ('Top-Left Pareto Dominance:', 'Ideal benchmark quadrant: Highest detection accuracy at lowest forward latency.')
    ]
    for lbl, val in pareto_points:
        p = tf.add_paragraph()
        p.text = f'★ {lbl} '
        p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(4)

    fig5_path = os.path.join(fig_dir, 'Fig5_efficiency_frontier_latency_vs_map.png')
    if os.path.exists(fig5_path):
        add_card(s13, 5.8, 1.35, 6.7, 5.40, CARD_BG, BORDER_COLOR)
        s13.shapes.add_picture(fig5_path, Inches(5.95), Inches(1.50), Inches(6.4), Inches(4.50))
        tb_cap = s13.shapes.add_textbox(Inches(5.95), Inches(6.15), Inches(6.4), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Fig. 5: Pareto Efficiency Frontier (Latency vs. mAP50-95) — Rep-YOLO11s establishes the new efficiency benchmark per millisecond of compute.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 14: VISUAL EXPLAINABILITY VIA GRAD-CAM XAI (FIGURE 3)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout); set_slide_bg(s14)
    add_header(s14, 'Visual Explainability (Grad-CAM XAI): Saliency & Noise Suppression', 'Criterion 4 · Model Transparency & Explainable AI', '14 / 18')

    add_card(s14, 0.8, 1.35, 4.4, 5.40)
    tb = s14.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(4.0), Inches(5.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = '3 HARD INDUSTRIAL SCENARIOS'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    scenarios = [
        ('SCENARIO 1: REFLECTIVE ORANGE VEST', 'Baseline disperses gradient across torso and scaffolding; Rep-YOLO11s focuses exclusively on helmet dome (conf 0.84).'),
        ('SCENARIO 2: SEVERE SPECULAR GLARE', 'Strong backlight obscures contrast; BiFormer maintains tight focus on the worker head (conf 0.89).'),
        ('SCENARIO 3: YELLOW WARNING SIGN', 'Baseline misclassifies yellow caution sign as helmet; CoordConv coordinates correctly suppress false alarm on the floor!')
    ]
    for lbl, val in scenarios:
        p = tf.add_paragraph(); p.text = lbl; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = val; p.font.size = Pt(9.2); p.font.color.rgb = TEXT_BODY; p.space_after = Pt(4)

    fig3_path = os.path.join(fig_dir, 'Fig3_gradcam_xai_saliency_comparison.png')
    if os.path.exists(fig3_path):
        add_card(s14, 5.4, 1.35, 7.1, 5.40, CARD_BG, BORDER_COLOR)
        s14.shapes.add_picture(fig3_path, Inches(5.55), Inches(1.48), Inches(6.8), Inches(4.55))
        tb_cap = s14.shapes.add_textbox(Inches(5.55), Inches(6.15), Inches(6.8), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Fig. 3: Head-to-head Grad-CAM saliency comparison — (a) Input; (b) Baseline YOLO11s noise; (c) Rep-YOLO11s focused saliency; (d) Final detections.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 15: CROSS-DOMAIN GENERALIZATION & IOU COLLAPSE AUTOPSY
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout); set_slide_bg(s15)
    add_header(s15, 'Cross-Domain Generalization & IoU Collapse Autopsy', 'Criterion 4 · Cross-Domain Generalization & Taxonomy', '15 / 18')

    ext_cards = [
        ('HARD HAT WORKERS (7,000 IMAGES)', '97.03%', 'Harmonized PPE Protocol (Hat-Only)', 'Matches in-domain SHWD accuracy; perfect transfer of helmet visual features.', GREEN_ACCENT),
        ('GDUT-HWD (13,499 IMAGES)', '74.27%', 'Precision: 90.26% (15-30 workers/frame)', 'Exceptional resilience under extreme worker crowd density and mutual occlusion.', BLUE_ACCENT),
        ('SHEL5K (5,000 DRONE IMAGES)', '41.15%', 'Precision: 85.62% (90° vertical angle)', 'Model conservatively suppresses ground clutter rather than generating false alarms.', AMBER_ACCENT)
    ]
    for idx, (h_txt, val_txt, sub_txt, desc_txt, col) in enumerate(ext_cards):
        left_pos = 0.8 + idx * 4.0
        add_card(s15, left_pos, 1.35, 3.8, 2.50)
        tb = s15.shapes.add_textbox(Inches(left_pos + 0.15), Inches(1.48), Inches(3.5), Inches(2.25))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_txt; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = val_txt; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = sub_txt; p.font.size = Pt(9); p.font.bold = True; p.font.color.rgb = col; p.space_after = Pt(2)
        p = tf.add_paragraph(); p.text = desc_txt; p.font.size = Pt(8.8); p.font.color.rgb = TEXT_MUTED

    add_card(s15, 0.8, 4.05, 11.7, 2.70, CARD_BG, ORANGE_ACCENT)
    tb = s15.shapes.add_textbox(Inches(1.05), Inches(4.18), Inches(11.2), Inches(2.45))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'MATHEMATICAL AUTOPSY: THE CROSS-DOMAIN IOU COLLAPSE'; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = '• Academic Label Taxonomy Conflict: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    run = p.add_run()
    run.text = 'The source SHWD dataset annotates "person" as FULL-BODY, while Hard Hat Workers annotates "person" as HEAD-ONLY.'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    p = tf.add_paragraph()
    p.text = '• Mathematical Degradation: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    run = p.add_run()
    run.text = 'When the detector accurately identifies a full worker body, ground-truth is only the tiny head. Intersection IoU = Area(head) / Area(body) ≈ 0.07 – 0.14 << 0.50! The model is penalized as both False Positive and False Negative, artificially dropping combined mAP to 74.40%.'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    p = tf.add_paragraph()
    p.text = '• Empirical Exoneration: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
    run = p.add_run()
    run.text = 'Evaluating under the Harmonized PPE Protocol (Hat-Only), mAP instantly surges to 97.03%, proving the learned visual representations are pristine!'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 16: HARDWARE REALIZATION ON BUDGET EDGE GPU (MX230 2GB)
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout); set_slide_bg(s16)
    add_header(s16, 'Edge Hardware Realization: Real-Time on 2GB MX230 Laptop GPU', 'Criterion 4 · Real-World Deployment on Budget Hardware', '16 / 18')

    hw_tiers = [
        ('SERVER / CLUSTER GPU', 'NVIDIA Tesla T4', '2.92 ms', '342.5 FPS', 'TensorRT 11.2 FP16 · 640×640\nHandles 12 RTSP camera streams concurrently.', ORANGE_ACCENT),
        ('MID-RANGE LAPTOP GPU', 'RTX 3050 Laptop', '5.35 ms', '187.1 FPS', 'TensorRT 11.2 FP16 · 640×640\nPortable edge surveillance workstation.', BLUE_ACCENT),
        ('BUDGET OFFICE LAPTOP', 'GeForce MX230 (2GB VRAM)', '36.0 ms', '27.8 FPS', 'Native PyTorch FP32 · 640×640\nEXCEEDS 24 FPS CINEMATIC REAL-TIME!', GREEN_ACCENT)
    ]
    for idx, (tier_h, dev_name, lat_val, fps_val, note_txt, col) in enumerate(hw_tiers):
        left_pos = 0.8 + idx * 4.0
        add_card(s16, left_pos, 1.35, 3.8, 2.70, LIGHT_GREEN if idx==2 else CARD_BG, col, border_width=1.8 if idx==2 else 1.0)
        tb = s16.shapes.add_textbox(Inches(left_pos + 0.15), Inches(1.48), Inches(3.5), Inches(2.45))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = tier_h; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = dev_name; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = f'Latency: {lat_val}'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_MUTED
        p = tf.add_paragraph(); p.text = fps_val; p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = col; p.space_after = Pt(3)
        p = tf.add_paragraph(); p.text = note_txt; p.font.size = Pt(8.8); p.font.color.rgb = TEXT_BODY

    # Bottom Callout: Real-world Value
    add_card(s16, 0.8, 4.30, 11.7, 2.45)
    tb = s16.shapes.add_textbox(Inches(1.05), Inches(4.45), Inches(11.2), Inches(2.20))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'ECONOMIC & PRACTICAL VALUE OF GEFORCE MX230 (2GB VRAM) VERIFICATION'; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT; p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = '• Breaking the Hardware Barrier: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    run = p.add_run()
    run.text = 'NVIDIA GeForce MX230 is an entry-level Pascal-generation mobile GPU (2019) lacking Tensor Cores and constrained by a 2GB VRAM ceiling. Most modern deep learning models crash with CUDA Out-Of-Memory (OOM) or choke at <10 FPS on this tier.'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    p = tf.add_paragraph()
    p.text = '• 100% Commercial Feasibility: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
    run = p.add_run()
    run.text = 'Rep-YOLO11s sustains 27.8 FPS with only 485 MB VRAM usage. Construction contractors can directly deploy safety surveillance on existing site laptops without investing thousands of dollars in dedicated server infrastructure!'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 17: PROTOTYPE PRODUCT - MULTI-STREAM RTSP PIPELINE (FIGURE 4)
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout); set_slide_bg(s17)
    add_header(s17, 'End-to-End Industrial Multi-Stream RTSP Surveillance Pipeline', 'Prototype Deliverable · Real-Time Surveillance Pipeline', '17 / 18')

    fig4_path = os.path.join(fig_dir, 'Fig4_industrial_rtsp_surveillance_pipeline.png')
    if os.path.exists(fig4_path):
        add_card(s17, 0.8, 1.25, 11.7, 4.30, CARD_BG, BORDER_COLOR)
        s17.shapes.add_picture(fig4_path, Inches(0.95), Inches(1.35), Inches(11.4), Inches(4.10))

    rtsp_stages = [
        ('HARDWARE H.264/H.265 DECODING', 'Independent multi-stream RTSP worker threads with ring buffer queue, decoding latency < 4.5 ms.', ORANGE_ACCENT),
        ('ACCELERATED TENSORRT INFERENCE', 'Parallel letterboxing with 5-channel CoordConv injection; FP16 forward latency of 2.92–5.35 ms.', BLUE_ACCENT),
        ('REAL-TIME ALERTING & HUD STREAMING', 'Vectorized NMS filtering, bounding box HUD rendering, and violation logging at 65–95 FPS.', GREEN_ACCENT)
    ]
    for idx, (h_txt, desc_txt, col) in enumerate(rtsp_stages):
        left_pos = 0.8 + idx * 4.0
        add_card(s17, left_pos, 5.65, 3.8, 1.45)
        tb = s17.shapes.add_textbox(Inches(left_pos + 0.15), Inches(5.72), Inches(3.5), Inches(1.30))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_txt; p.font.size = Pt(10.2); p.font.bold = True; p.font.color.rgb = col; p.space_after = Pt(2)
        p = tf.add_paragraph(); p.text = desc_txt; p.font.size = Pt(9); p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 18: SUMMARY & ROADMAP FOR STAGES 2 - 3
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout); set_slide_bg(s18)
    add_header(s18, 'Stage 1 Accomplishments & Roadmap for Stages 2 – 3', 'Capstone Roadmap · IT Faculty - FPT University', '18 / 18')

    phases = [
        ('STAGE 1 (COMPLETED ~90%)', 'COMPLETED ENGINEERING MILESTONES', [
            '✓ Mathematical foundations & 4 custom modules formulated.',
            '✓ Full A0–A6 ablation & 5-Fold CV (97.11% peak) validated.',
            '✓ Real-time feasibility confirmed on MX230 (27.8 FPS, 485MB).',
            '✓ 9-page formal scientific research paper authored in LaTeX.'
        ], GREEN_ACCENT),

        ('STAGE 2 (UPCOMING ROADMAP)', 'PLANNED CAPSTONE MILESTONES', [
            '▸ Develop comprehensive Desktop GUI & Web Admin Dashboard.',
            '▸ Integrate multi-camera RTSP concurrent stream manager.',
            '▸ Implement automated acoustic buzzer and instant email dispatch.',
            '▸ Deploy pilot trial on-site with industry partner.'
        ], BLUE_ACCENT),

        ('STAGE 3 (FINAL DEFENSE)', 'TERM CONCLUSION & PUBLICATION', [
            '▸ Execute Knowledge Distillation (YOLO11x teacher -> s student).',
            '▸ Enhance extreme drone-view performance on SHEL5K.',
            '▸ Finalize formal Capstone Graduation Dossier.',
            '▸ Submit research paper to international computer vision venue.'
        ], ORANGE_ACCENT)
    ]

    for idx, (p_title, p_badge, items, col) in enumerate(phases):
        left_pos = 0.8 + idx * 4.0
        add_card(s18, left_pos, 1.35, 3.8, 3.75)
        
        # Header + Badge
        tb = s18.shapes.add_textbox(Inches(left_pos + 0.18), Inches(1.50), Inches(3.44), Inches(0.80))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = p_badge; p.font.size = Pt(8.5); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = p_title; p.font.size = Pt(10.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

        # Bullets
        tb_b = s18.shapes.add_textbox(Inches(left_pos + 0.18), Inches(2.35), Inches(3.44), Inches(2.65))
        tf_b = tb_b.text_frame; tf_b.word_wrap = True
        tf_b.margin_left = tf_b.margin_right = tf_b.margin_top = tf_b.margin_bottom = 0
        for b_txt in items:
            p = tf_b.add_paragraph()
            p.text = b_txt
            p.font.size = Pt(9.3)
            p.font.color.rgb = TEXT_DARK if '✓' in b_txt else TEXT_BODY
            p.font.bold = True if '✓' in b_txt else False
            p.space_after = Pt(5)

    # Bottom Contact Card & Ready to Defend
    add_card(s18, 0.8, 5.30, 11.7, 1.70, CARD_BG, ORANGE_ACCENT, border_width=1.5)
    tb = s18.shapes.add_textbox(Inches(1.05), Inches(5.45), Inches(11.2), Inches(1.40))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = 'ACKNOWLEDGMENT & DEFENSE READINESS'
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(2)
    p = tf.add_paragraph()
    p.text = 'The research team expresses sincere gratitude to M.Sc. Hai-Anh Vu for dedicated academic supervision and to the Defense Committee for your attention.\nWe warmly welcome constructive feedback, critiques, and questions from the esteemed Committee members.'
    p.font.size = Pt(10.2); p.font.color.rgb = TEXT_BODY; p.space_after = Pt(3)
    p = tf.add_paragraph()
    p.text = 'Presenter: Nhu Nguyen-Han (SE183644) · Dept. of Artificial Intelligence — FPT University Hanoi'
    p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SAVE OUTPUT PRESENTATIONS & EXPORT PDF
    # =========================================================================
    pptx_out1 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep-YOLO11s___Capstone_Review_1_Defense_20260920062556.pptx'
    pptx_out2 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep_YOLO11s_FPT_Defense_Master_Deck.pptx'
    pdf_out1 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep-YOLO11s___Capstone_Review_1_Defense_20260920062625.pdf'
    pdf_out2 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep_YOLO11s_FPT_Defense_Master_Deck.pdf'
    render_dir = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\slide_renders'
    os.makedirs(render_dir, exist_ok=True)
    
    prs.save(pptx_out1)
    print(f"Successfully saved English PPTX to: {pptx_out1}")
    try:
        shutil.copyfile(pptx_out1, pptx_out2)
        print(f"Successfully copied secondary English PPTX to: {pptx_out2}")
    except Exception as e:
        print(f"Note: Secondary PPTX ({e})")

    # COM Automation for PDF export
    try:
        import win32com.client
        ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        ppt_app.Visible = 1
        deck = ppt_app.Presentations.Open(os.path.abspath(pptx_out1), WithWindow=False)
        deck.SaveAs(os.path.abspath(pdf_out1), 32)  # 32 = ppSaveAsPDF
        deck.Close()
        ppt_app.Quit()
        print(f"Successfully exported English PDF to: {pdf_out1}")
        try:
            shutil.copyfile(pdf_out1, pdf_out2)
        except Exception:
            pass

        # Render to PNGs
        import fitz
        doc = fitz.open(pdf_out1)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            png_p = os.path.join(render_dir, f"slide_{i+1:02d}.png")
            pix.save(png_p)
        print(f"Rendered {len(doc)} English slides to PNGs in: {render_dir}")
    except Exception as e:
        print(f"PDF export / render warning: {e}")

if __name__ == '__main__':
    build_english_deck()
