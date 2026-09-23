"""
Script to generate PowerPoint slides for Sections 8, 9, 10, 11, and 12
of the Capstone Review 1 presentation.

Based on: review_1_main/HUONG_DAN_TRA_LOI_12_MUC_REVIEW_1.md
Matching the design style of: review_1_main/Capstone_Review_1.pptx
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_pptx_deck(output_pptx_path, existing_prs=None):
    if existing_prs is not None:
        prs = existing_prs
    else:
        prs = pptx.Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Colors
    c_bg = RGBColor(0xFA, 0xF8, 0xF5)
    c_white = RGBColor(0xFF, 0xFF, 0xFF)
    c_border = RGBColor(0xE2, 0xE8, 0xF0)
    c_orange = RGBColor(0xEA, 0x58, 0x0C)
    c_blue = RGBColor(0x25, 0x63, 0xEB)
    c_amber = RGBColor(0xD9, 0x77, 0x06)
    c_green = RGBColor(0x16, 0xA3, 0x4A)
    c_purple = RGBColor(0x7C, 0x3A, 0xED)
    c_slate = RGBColor(0x47, 0x55, 0x69)
    c_red = RGBColor(0xDC, 0x26, 0x26)
    c_title = RGBColor(0x0F, 0x17, 0x2A)
    c_body = RGBColor(0x33, 0x41, 0x55)
    c_muted = RGBColor(0x64, 0x74, 0x8B)

    def setup_slide_header(slide, criterion_tag, slide_num, title_text):
        # Background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = c_bg
        bg.line.fill.background()

        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(9.5), Inches(0.28))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = criterion_tag.upper()
        p_cat.font.name = 'Calibri'
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = c_orange

        # Slide number
        num_box = slide.shapes.add_textbox(Inches(11.0), Inches(0.35), Inches(1.5), Inches(0.28))
        tf_num = num_box.text_frame
        tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = slide_num
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.name = 'Calibri'
        p_num.font.size = Pt(11)
        p_num.font.color.rgb = c_muted

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.55))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = 'Calibri'
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = c_title

    def add_bottom_banner(slide, text):
        box_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.6), Inches(11.733), Inches(0.52))
        box_shape.fill.solid()
        box_shape.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
        box_shape.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
        box_shape.line.width = Pt(1)

        # Orange Takeaway pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.95), Inches(6.68), Inches(1.2), Inches(0.35))
        pill.fill.solid()
        pill.fill.fore_color.rgb = c_orange
        pill.line.fill.background()
        tf_p = pill.text_frame
        tf_p.margin_left = tf_p.margin_top = tf_p.margin_right = tf_p.margin_bottom = 0
        p_pl = tf_p.paragraphs[0]
        p_pl.text = "TAKEAWAY"
        p_pl.alignment = PP_ALIGN.CENTER
        p_pl.font.name = 'Calibri'
        p_pl.font.size = Pt(9.5)
        p_pl.font.bold = True
        p_pl.font.color.rgb = c_white

        t_box = slide.shapes.add_textbox(Inches(2.25), Inches(6.62), Inches(10.1), Inches(0.48))
        tf = t_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = text
        p.font.name = 'Calibri'
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = c_title

    # -------------------------------------------------------------
    # SLIDE 8: CRITERION 4 · EVALUATION STRATEGY
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s8, "CRITERION 4 · EVALUATION STRATEGY", "08 / 18",
                       "Rigorous Multi-Metric Evaluation & Life-Critical Recall Strategy")

    # Card 1: Core Detection Metrics (Top-Left)
    c1 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(2.45))
    c1.fill.solid(); c1.fill.fore_color.rgb = c_white; c1.line.color.rgb = c_border; c1.line.width = Pt(1)
    b1 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b1.fill.solid(); b1.fill.fore_color.rgb = c_blue; b1.line.fill.background()
    b1.text_frame.text = "01"; b1.text_frame.paragraphs[0].font.name = 'Calibri'; b1.text_frame.paragraphs[0].font.size = Pt(11); b1.text_frame.paragraphs[0].font.bold = True; b1.text_frame.paragraphs[0].font.color.rgb = c_white
    
    t1 = s8.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(2.2))
    tf1 = t1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "Core Accuracy & Localization Metrics"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_s8_1 = [
        "mAP50 (Gold Standard): Evaluates primary detection precision at standard IoU 0.50.",
        "mAP50-95 (Boundary Tightness): Strict average across 10 IoU steps [0.50:0.05:0.95]; heavily penalizes loose or misaligned bounding boxes.",
        "Precision & F1-Score: Measures trade-off between false detections and completeness.",
        "Real-Time Constraint: Enforces strict inference latency suitable for 25+ FPS streams."
    ]
    for itm in items_s8_1:
        p = tf1.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    # Card 2: Life-Critical Recall (Top-Right)
    c2 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(2.45))
    c2.fill.solid(); c2.fill.fore_color.rgb = c_white; c2.line.color.rgb = c_border; c2.line.width = Pt(1)
    b2 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b2.fill.solid(); b2.fill.fore_color.rgb = c_orange; b2.line.fill.background()
    b2.text_frame.text = "02"; b2.text_frame.paragraphs[0].font.name = 'Calibri'; b2.text_frame.paragraphs[0].font.size = Pt(11); b2.text_frame.paragraphs[0].font.bold = True; b2.text_frame.paragraphs[0].font.color.rgb = c_white

    t2 = s8.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(2.2))
    tf2 = t2.text_frame; tf2.word_wrap = True
    p = tf2.paragraphs[0]; p.text = "Life-Critical Priority: Recall Over Precision"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_s8_2 = [
        "Zero-Tolerance for False Negatives (FN): In safety monitoring, a missed violation allows an unprotected worker into a hazard zone -> fatal risk.",
        "Asymmetric Impact: A False Alarm (FP) causes minor verification overhead; a Missed Violation (FN) leads to irreversible human tragedy.",
        "Target Recallhat > 90%: Architecture specifically refined to maximize helmet recall.",
        "Gradient Countermeasure: Focal BCE and Task-Aligned Assigner prevent helmet loss collapse."
    ]
    for itm in items_s8_2:
        p = tf2.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    # Card 3: 5-Fold Stratified CV (Bottom-Left)
    c3 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.95), Inches(5.7), Inches(2.45))
    c3.fill.solid(); c3.fill.fore_color.rgb = c_white; c3.line.color.rgb = c_border; c3.line.width = Pt(1)
    b3 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.1), Inches(0.42), Inches(0.32))
    b3.fill.solid(); b3.fill.fore_color.rgb = c_amber; b3.line.fill.background()
    b3.text_frame.text = "03"; b3.text_frame.paragraphs[0].font.name = 'Calibri'; b3.text_frame.paragraphs[0].font.size = Pt(11); b3.text_frame.paragraphs[0].font.bold = True; b3.text_frame.paragraphs[0].font.color.rgb = c_white

    t3 = s8.shapes.add_textbox(Inches(1.55), Inches(4.08), Inches(4.8), Inches(2.2))
    tf3 = t3.text_frame; tf3.word_wrap = True
    p = tf3.paragraphs[0]; p.text = "5-Fold Stratified Cross-Validation Protocol"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_s8_3 = [
        "Rigorous Stratification: All 5 folds strictly preserve the exact 1:12.33 class ratio.",
        "Statistical Confidence (μ ± σ): Reports 96.64% ± 0.32% mAP50, proving performance is not a fluke of a lucky test split.",
        "Cross-Fold Stability: Validates consistency across varying illumination and crowd densities.",
        "Ablation Reliability: Ensures all module comparisons are statistically significant."
    ]
    for itm in items_s8_3:
        p = tf3.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    # Card 4: Grad-CAM Explainable AI (Bottom-Right)
    c4 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.95), Inches(5.7), Inches(2.45))
    c4.fill.solid(); c4.fill.fore_color.rgb = c_white; c4.line.color.rgb = c_border; c4.line.width = Pt(1)
    b4 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(4.1), Inches(0.42), Inches(0.32))
    b4.fill.solid(); b4.fill.fore_color.rgb = c_green; b4.line.fill.background()
    b4.text_frame.text = "04"; b4.text_frame.paragraphs[0].font.name = 'Calibri'; b4.text_frame.paragraphs[0].font.size = Pt(11); b4.text_frame.paragraphs[0].font.bold = True; b4.text_frame.paragraphs[0].font.color.rgb = c_white

    t4 = s8.shapes.add_textbox(Inches(7.55), Inches(4.08), Inches(4.8), Inches(2.2))
    tf4 = t4.text_frame; tf4.word_wrap = True
    p = tf4.paragraphs[0]; p.text = "Model Transparency: Grad-CAM XAI Audit"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_s8_4 = [
        "Feature Saliency Validation: Grad-CAM heatmaps verify network attention is sharply localized on head-worn helmets.",
        "Suppression of Vest & Floor Clutter: Proves CoordConv successfully damps out ground-level yellow buckets and reflective vests.",
        "Long-Range Attention Routing: Confirms BiFormer routes attention directly to small distant heads (<20px).",
        "Industrial Audit Readiness: Provides visual accountability required by HSE site managers."
    ]
    for itm in items_s8_4:
        p = tf4.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    add_bottom_banner(s8, "Multi-metric evaluation prioritizes Life-Critical Recall while eliminating split bias via 5-Fold Stratified CV and Grad-CAM interpretability.")

    # -------------------------------------------------------------
    # SLIDE 9: CRITERION 5 · PROJECT PLAN & GANTT MILESTONES
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s9, "CRITERION 5 · PROJECT PLAN & GANTT MILESTONES", "09 / 18",
                       "Project Roadmap: 5 Phases & Milestone Tracking (W1–W15)")

    phases = [
        ("Phase 1 (W1–W5)", "Foundation & Review 1", "30-paper survey, RQ/Gap, SHWD data cleaning, Baseline setup.", "Review 1 Deck & Cleaned Data", c_orange),
        ("Phase 2 (W6–W8)", "Core Engineering & Review 2", "Module coding (RepConv, CoordConv, BiFormer, EIoU), Ablation A0–A6.", "Review 2 Report & Code", c_blue),
        ("Phase 3 (W9–W11)", "Validation & Review 3", "5-Fold CV, Cross-domain testing (25k+ imgs), Grad-CAM XAI.", "Review 3 Data & Weights", c_amber),
        ("Phase 4 (W12–W14)", "Deployment & Docs", "RTSP camera pipeline, Docker packaging, Scientific paper polish.", "RTSP Demo & Paper Draft", c_green),
        ("Phase 5 (W15)", "Final Defense", "Thesis Book submission, final defense presentation rehearsal.", "Thesis Book & Defense", c_purple),
    ]

    p_w = Inches(2.23)
    p_gap = Inches(0.14)
    p_left_start = Inches(0.8)

    for i, (p_title, p_sub, p_desc, p_deliv, p_col) in enumerate(phases):
        cur_left = p_left_start + i * (p_w + p_gap)
        c_p = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_left, Inches(1.35), p_w, Inches(2.45))
        c_p.fill.solid(); c_p.fill.fore_color.rgb = c_white; c_p.line.color.rgb = p_col; c_p.line.width = Pt(1.5)

        # Header tag box
        tag_b = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_left + Inches(0.08), Inches(1.45), p_w - Inches(0.16), Inches(0.32))
        tag_b.fill.solid(); tag_b.fill.fore_color.rgb = p_col; tag_b.line.fill.background()
        tag_b.text_frame.text = p_title
        tag_b.text_frame.paragraphs[0].font.name = 'Calibri'; tag_b.text_frame.paragraphs[0].font.size = Pt(10); tag_b.text_frame.paragraphs[0].font.bold = True; tag_b.text_frame.paragraphs[0].font.color.rgb = c_white
        tag_b.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        tf_p = s9.shapes.add_textbox(cur_left + Inches(0.08), Inches(1.82), p_w - Inches(0.16), Inches(1.9)).text_frame
        tf_p.word_wrap = True; tf_p.margin_left = tf_p.margin_right = 0
        p = tf_p.paragraphs[0]; p.text = p_sub; p.font.bold = True; p.font.size = Pt(10); p.font.color.rgb = c_title
        p = tf_p.add_paragraph(); p.text = p_desc; p.font.size = Pt(9); p.font.color.rgb = c_body; p.space_before = Pt(3)
        p = tf_p.add_paragraph(); p.text = "Target: " + p_deliv; p.font.size = Pt(8.5); p.font.bold = True; p.font.color.rgb = p_col; p.space_before = Pt(4)

    # 2 Bottom Cards
    c_agile = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.95), Inches(5.7), Inches(2.45))
    c_agile.fill.solid(); c_agile.fill.fore_color.rgb = c_white; c_agile.line.color.rgb = c_border; c_agile.line.width = Pt(1)
    b_ag = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.1), Inches(0.42), Inches(0.32))
    b_ag.fill.solid(); b_ag.fill.fore_color.rgb = c_blue; b_ag.line.fill.background()
    b_ag.text_frame.text = "01"; b_ag.text_frame.paragraphs[0].font.name = 'Calibri'; b_ag.text_frame.paragraphs[0].font.size = Pt(11); b_ag.text_frame.paragraphs[0].font.bold = True; b_ag.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_ag = s9.shapes.add_textbox(Inches(1.55), Inches(4.08), Inches(4.8), Inches(2.2)).text_frame
    tf_ag.word_wrap = True
    p = tf_ag.paragraphs[0]; p.text = "Agile Sprint Execution & Contingency Buffers"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_ag = [
        "2-Week Sprint Cadence: Rapid iterative cycles with bi-weekly progress synchronization with Academic Advisor.",
        "1-Week Risk Buffer per Phase: Dedicated contingency time for cloud GPU re-training and parameter exploration.",
        "Review Gating Criteria: Strict quality gates and benchmark hurdles before advancing to the next Review.",
        "Transparent Version Control: Daily code, configuration, and model checkpoint commits synchronized to GitHub."
    ]
    for itm in items_ag:
        p = tf_ag.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    c_role = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.95), Inches(5.7), Inches(2.45))
    c_role.fill.solid(); c_role.fill.fore_color.rgb = c_white; c_role.line.color.rgb = c_border; c_role.line.width = Pt(1)
    b_ro = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(4.1), Inches(0.42), Inches(0.32))
    b_ro.fill.solid(); b_ro.fill.fore_color.rgb = c_orange; b_ro.line.fill.background()
    b_ro.text_frame.text = "02"; b_ro.text_frame.paragraphs[0].font.name = 'Calibri'; b_ro.text_frame.paragraphs[0].font.size = Pt(11); b_ro.text_frame.paragraphs[0].font.bold = True; b_ro.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_ro = s9.shapes.add_textbox(Inches(7.55), Inches(4.08), Inches(4.8), Inches(2.2)).text_frame
    tf_ro.word_wrap = True
    p = tf_ro.paragraphs[0]; p.text = "Strategic Team Role Allocation"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_ro = [
        "Nguyễn Hàn Như (Lead · SE183644): Overall network architecture, custom PyTorch modules, and Scientific paper writing.",
        "Nguyễn Văn Thành (SE180387): Data engineering, label harmonization (C*), SHWD cleaning, 5-Fold Stratified CV.",
        "Trần Phạm Tuấn Dũng (SE183674): Real-time RTSP video streaming pipeline, hardware acceleration, and documentation.",
        "High-Efficiency Collaboration: Clear responsibility boundaries with unified shared Git repo and issue tracking."
    ]
    for itm in items_ro:
        p = tf_ro.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    add_bottom_banner(s9, "Structured 15-week Hybrid Agile roadmap strictly aligned with 3 Review milestones and final Capstone Defense.")

    # -------------------------------------------------------------
    # SLIDE 10: CRITERION 6 · SCIENTIFIC RESEARCH & PUBLICATION PLAN
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s10, "CRITERION 6 · SCIENTIFIC RESEARCH & PUBLICATION PLAN", "10 / 18",
                       "Scientific Research Manuscript: Publication-Ready Paper Plan")

    # Hero Card: Target Academic Standards & Publication Readiness
    c_hero = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(2.2))
    c_hero.fill.solid(); c_hero.fill.fore_color.rgb = c_white; c_hero.line.color.rgb = c_blue; c_hero.line.width = Pt(1.5)

    b_h = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.48), Inches(3.6), Inches(0.32))
    b_h.fill.solid(); b_h.fill.fore_color.rgb = c_blue; b_h.line.fill.background()
    b_h.text_frame.text = "SCIENTIFIC MANUSCRIPT & ACADEMIC STANDARDS"
    b_h.text_frame.paragraphs[0].font.name = 'Calibri'; b_h.text_frame.paragraphs[0].font.size = Pt(10); b_h.text_frame.paragraphs[0].font.bold = True; b_h.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_h = s10.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(11.3), Inches(1.6)).text_frame
    tf_h.word_wrap = True
    p = tf_h.paragraphs[0]; p.text = "Publication-Ready Master Manuscript (Targeting Peer-Reviewed Venues)"; p.font.bold = True; p.font.size = Pt(15); p.font.color.rgb = c_title
    p = tf_h.add_paragraph(); p.text = "Formatted in Standard IEEE Double-Column · Ready for Academic Committee Review & Journal Submission"; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = c_orange; p.space_before = Pt(2)
    p = tf_h.add_paragraph(); p.text = 'Title: "Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance"'; p.font.size = Pt(10.5); p.font.italic = True; p.font.color.rgb = c_body; p.space_before = Pt(3)
    p = tf_h.add_paragraph(); p.text = "• Manuscript Status: Complete 10-page master technical paper fully drafted in IEEE format, ready for Academic Council evaluation and journal submission."; p.font.size = Pt(10.5); p.font.color.rgb = c_title; p.space_before = Pt(2)

    # Bottom Left Card: 4 Core Scientific Contributions
    c_cont = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.7), Inches(5.7), Inches(2.7))
    c_cont.fill.solid(); c_cont.fill.fore_color.rgb = c_white; c_cont.line.color.rgb = c_border; c_cont.line.width = Pt(1)
    b_ct = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.85), Inches(0.42), Inches(0.32))
    b_ct.fill.solid(); b_ct.fill.fore_color.rgb = c_orange; b_ct.line.fill.background()
    b_ct.text_frame.text = "01"; b_ct.text_frame.paragraphs[0].font.name = 'Calibri'; b_ct.text_frame.paragraphs[0].font.size = Pt(11); b_ct.text_frame.paragraphs[0].font.bold = True; b_ct.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_ct = s10.shapes.add_textbox(Inches(1.55), Inches(3.83), Inches(4.8), Inches(2.45)).text_frame
    tf_ct.word_wrap = True
    p = tf_ct.paragraphs[0]; p.text = "4 Key Scientific Contributions"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_ct = [
        "1. Novel Architecture: Rep-YOLO11s integrating spatial coordinates, attention routing, and edge-decoupled loss.",
        "2. Closed-Form Mathematical Proof: Rigorous algebraic proof of lossless 3-branch RepConv re-parameterization (error < 10^-5).",
        "3. Explainable Ground Invariance: Empirical Grad-CAM proof that CoordConv eliminates ground-level color false alarms.",
        "4. Multi-Domain Taxonomy Autopsy: Comprehensive benchmark across >33,000 images resolving the IoU Collapse phenomenon."
    ]
    for itm in items_ct:
        p = tf_ct.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    # Bottom Right Card: Publication Roadmap
    c_pub = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.7), Inches(5.7), Inches(2.7))
    c_pub.fill.solid(); c_pub.fill.fore_color.rgb = c_white; c_pub.line.color.rgb = c_border; c_pub.line.width = Pt(1)
    b_pb = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(3.85), Inches(0.42), Inches(0.32))
    b_pb.fill.solid(); b_pb.fill.fore_color.rgb = c_green; b_pb.line.fill.background()
    b_pb.text_frame.text = "02"; b_pb.text_frame.paragraphs[0].font.name = 'Calibri'; b_pb.text_frame.paragraphs[0].font.size = Pt(11); b_pb.text_frame.paragraphs[0].font.bold = True; b_pb.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_pb = s10.shapes.add_textbox(Inches(7.55), Inches(3.83), Inches(4.8), Inches(2.45)).text_frame
    tf_pb.word_wrap = True
    p = tf_pb.paragraphs[0]; p.text = "Publication Roadmap & Scientific Integrity"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    items_pb = [
        "Phase 1–3 (Empirical Evidence): Solidify 5-Fold Stratified CV, multi-dataset zero-shot tests, and Grad-CAM saliency maps.",
        "Phase 4 (Writing & Academic Review): Rigorous academic polishing, IEEE LaTeX double-column typesetting, and mentor co-author review.",
        "Phase 5 & Defense: Present manuscript to Academic Council; finalize submission to reputable journals/conferences under advisor mentorship.",
        "Reproducibility Standards: Full open-source release with deterministic seed control and complete training scripts."
    ]
    for itm in items_pb:
        p = tf_pb.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(10); p.font.color.rgb = c_body; p.space_before = Pt(3)

    add_bottom_banner(s10, "Publication-ready scientific manuscript prepared to international IEEE standards, fully verifiable by the Academic Council and ready for peer-reviewed venues.")

    # -------------------------------------------------------------
    # SLIDE 11: CRITERION 7 · RISK ASSESSMENT & MITIGATION MATRIX
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s11, "CRITERION 7 · RISK ASSESSMENT & MITIGATION MATRIX", "11 / 18",
                       "Proactive Risk Analysis & Engineering Mitigation Matrix")

    risks = [
        ("1. Extreme Class Imbalance (1:12)", "HIGH RISK", c_orange,
         "111,514 body instances overpower 9,044 helmet instances, causing helmet gradient starvation.",
         "Task-Aligned Assigner (TAL) top-10 anchor alignment + Focal BCE Loss with dynamic alpha-balance to prioritize helmet learning."),
        ("2. Ground False Positives", "HIGH RISK", c_amber,
         "Yellow buckets, traffic cones, and caution signs on the floor trigger high false alarm rates (>28%).",
         "CoordConv Stem injects vertical coordinate Cy; provides anatomical prior suppressing ground logits (Cy > 0.4) to <0.02."),
        ("3. Distant Tiny Helmets (<20px)", "HIGH RISK", c_purple,
         "Small targets under high-angle CCTV blur and vanish through standard convolutional downsampling.",
         "BiFormer Bi-Level Routing Attention filters out 93.75% of background and preserves small target features with O(HW) cost."),
        ("4. CIoU Gradient Vanishing", "MED RISK", c_blue,
         "When aspect ratio w/h matches ground truth, CIoU penalty gradient vanishes, stalling boundary regression.",
         "Focal-EIoU Loss decouples width and height errors independently; guarantees dL/dw != 0 whenever width is not aligned."),
        ("5. Video Sequence Data Leakage", "HIGH RISK", c_red,
         "Consecutive frames from identical camera scenes appearing in both train and test artificially inflate accuracy.",
         "Video sequence hashing groups continuous frames strictly into the same partition, guaranteeing 100% leak-free evaluation."),
        ("6. GPU Memory & Compute Limits", "MED RISK", c_slate,
         "Training high-resolution multi-module models risks CUDA Out-Of-Memory (OOM) and long training times.",
         "Dual Kaggle Tesla T4 GPUs (32 GB VRAM) using PyTorch DDP; FP16 mixed precision halves memory footprint without accuracy loss.")
    ]

    rw = Inches(3.72)
    rh = Inches(2.45)
    rx_gap = Inches(0.28)
    ry_gap = Inches(0.18)

    for idx, (rtitle, rlevel, rcol, rdesc, rmit) in enumerate(risks):
        row = idx // 3
        col = idx % 3
        c_x = Inches(0.8) + col * (rw + rx_gap)
        c_y = Inches(1.35) + row * (rh + ry_gap)

        rc = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_x, c_y, rw, rh)
        rc.fill.solid(); rc.fill.fore_color.rgb = c_white; rc.line.color.rgb = c_border; rc.line.width = Pt(1)

        # Badge
        rb = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_x + Inches(0.12), c_y + Inches(0.12), Inches(0.95), Inches(0.24))
        rb.fill.solid(); rb.fill.fore_color.rgb = rcol; rb.line.fill.background()
        rb.text_frame.text = rlevel
        rb.text_frame.paragraphs[0].font.name = 'Calibri'; rb.text_frame.paragraphs[0].font.size = Pt(8.5); rb.text_frame.paragraphs[0].font.bold = True; rb.text_frame.paragraphs[0].font.color.rgb = c_white
        rb.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        tf_r = s11.shapes.add_textbox(c_x + Inches(0.12), c_y + Inches(0.38), rw - Inches(0.24), rh - Inches(0.42)).text_frame
        tf_r.word_wrap = True; tf_r.margin_left = tf_r.margin_right = 0
        p = tf_r.paragraphs[0]; p.text = rtitle; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_title
        p = tf_r.add_paragraph(); p.text = "Risk Trigger: " + rdesc; p.font.size = Pt(8.8); p.font.color.rgb = RGBColor(0x99, 0x1B, 0x1B); p.space_before = Pt(2)
        p = tf_r.add_paragraph(); p.text = "Engineering Mitigation: " + rmit; p.font.size = Pt(8.8); p.font.bold = True; p.font.color.rgb = RGBColor(0x16, 0x65, 0x34); p.space_before = Pt(3)

    add_bottom_banner(s11, "All 6 core engineering risks are proactively resolved through mathematically sound and empirically verified architectural solutions.")

    # -------------------------------------------------------------
    # SLIDE 12: CRITERION 8 · EXPECTED OUTCOMES & CONCLUSION
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s12, "CRITERION 8 · EXPECTED OUTCOMES & CONCLUSION", "12 / 18",
                       "Final Deliverables & 8-Point Project Synthesis")

    # Left Column: 5 Deliverables
    c_del = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_del.fill.solid(); c_del.fill.fore_color.rgb = c_white; c_del.line.color.rgb = c_border; c_del.line.width = Pt(1)
    b_dl = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_dl.fill.solid(); b_dl.fill.fore_color.rgb = c_blue; b_dl.line.fill.background()
    b_dl.text_frame.text = "01"; b_dl.text_frame.paragraphs[0].font.name = 'Calibri'; b_dl.text_frame.paragraphs[0].font.size = Pt(11); b_dl.text_frame.paragraphs[0].font.bold = True; b_dl.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_dl = s12.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame
    tf_dl.word_wrap = True
    p = tf_dl.paragraphs[0]; p.text = "5 Tangible Project Deliverables"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    deliv_items = [
        ("1. High-Accuracy AI Weights (.pt)", "Rep-YOLO11s trained weights achieving >96% 5-Fold mAP50 with optimized small helmet recall."),
        ("2. Standardized Dataset Pipeline", "Cleaned SHWD (7,581 images) with zero label corruption, plus unified C* taxonomy across >33,000 industrial images."),
        ("3. Open-Source Code Repository", "Clean GitHub repository with PyTorch modules, switch_to_deploy() scripts, and multi-GPU DDP training configs."),
        ("4. Real-Time RTSP Software Demo", "Surveillance desktop application running on live CCTV streams with automated violation alerts and bounding boxes."),
        ("5. Scientific Research Manuscript", "Publication-ready master paper in IEEE format, ready for Academic Council evaluation and journal submission.")
    ]
    for d_title, d_desc in deliv_items:
        p = tf_dl.add_paragraph(); p.text = d_title; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_blue; p.space_before = Pt(5)
        p = tf_dl.add_paragraph(); p.text = d_desc; p.font.size = Pt(9.5); p.font.color.rgb = c_body; p.space_before = Pt(1)

    # Right Column: 8-Point Synthesis (Conclusion)
    c_con = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_con.fill.solid(); c_con.fill.fore_color.rgb = c_white; c_con.line.color.rgb = c_border; c_con.line.width = Pt(1)
    b_cn = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_cn.fill.solid(); b_cn.fill.fore_color.rgb = c_orange; b_cn.line.fill.background()
    b_cn.text_frame.text = "02"; b_cn.text_frame.paragraphs[0].font.name = 'Calibri'; b_cn.text_frame.paragraphs[0].font.size = Pt(11); b_cn.text_frame.paragraphs[0].font.bold = True; b_cn.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_cn = s12.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame
    tf_cn.word_wrap = True
    p = tf_cn.paragraphs[0]; p.text = "8-Point Project Synthesis (Conclusion)"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title
    points_syn = [
        ("• Problem:", "Industrial CCTV challenges (tiny targets, 1:12 class imbalance, ground false alarms)."),
        ("• Research Gap:", "Heavy architectures lack real-time viability; standard models lack spatial height priors."),
        ("• SOTA Baselines:", "Directly benchmarked against YOLO11s (latest SOTA) and YOLOv8s (industry standard)."),
        ("• Proposed Method:", "Rep-YOLO11s integrating RepConv, CoordConv, BiFormer, and Focal-EIoU."),
        ("• Data Engineering:", "7,581 SHWD images with zero leakage + 5 cross-domain benchmarks (>25,000 images)."),
        ("• Evaluation:", "Life-critical Recall prioritization, 5-Fold Stratified CV, and Grad-CAM explainability."),
        ("• Feasibility:", "High-efficiency single-path architecture verified on accessible GPU hardware."),
        ("• Practical Value:", "Seamlessly bridges top-tier academic rigor with live worker safety protection.")
    ]
    for p_title, p_desc in points_syn:
        p = tf_cn.add_paragraph(); p.text = f"{p_title} {p_desc}"; p.font.size = Pt(9.5); p.font.color.rgb = c_body; p.space_before = Pt(3)

    add_bottom_banner(s12, "A comprehensive, publication-grade AI capstone delivering high-accuracy helmet surveillance, robust software, and verifiable scientific contributions.")

    # Ensure all text paragraphs use Calibri
    for slide in [s8, s9, s10, s11, s12]:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    p.font.name = 'Calibri'

    prs.save(output_pptx_path)
    print(f"Successfully generated PowerPoint deck: {output_pptx_path}")

def build_presentation(base_pptx=None, output_path=None):
    if base_pptx and os.path.exists(base_pptx):
        prs = pptx.Presentation(base_pptx)
    else:
        prs = pptx.Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
    
    # Generate slides
    create_pptx_deck(output_path, existing_prs=prs)

if __name__ == '__main__':
    out_dir = os.path.join(os.getcwd(), 'review_1_main')
    pptx_path = os.path.join(out_dir, 'Slides_Muc_8_9_10_11_12.pptx')
    create_pptx_deck(pptx_path)

    # Also generate merged deck for user convenience
    base_file = os.path.join(out_dir, 'Capstone_Review_1.pptx')
    if os.path.exists(base_file):
        merged_path = os.path.join(out_dir, 'Capstone_Review_1_Merged_Muc_8_to_12.pptx')
        merged_prs = pptx.Presentation(base_file)
        create_pptx_deck(merged_path, existing_prs=merged_prs)
