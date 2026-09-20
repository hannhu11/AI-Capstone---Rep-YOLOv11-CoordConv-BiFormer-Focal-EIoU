# -*- coding: utf-8 -*-
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Palette (Warm Ivory / Light Academic Theme matching Figure 2 / Page 6)
BG_COLOR = RGBColor(250, 248, 245)      # #FAF8F5
CARD_BG = RGBColor(255, 255, 255)       # #FFFFFF
TEXT_DARK = RGBColor(15, 23, 42)        # #0F172A
TEXT_MUTED = RGBColor(71, 85, 105)      # #475569
BORDER_COLOR = RGBColor(226, 232, 240)  # #E2E8F0
ORANGE_ACCENT = RGBColor(234, 88, 12)   # #EA580C
AMBER_ACCENT = RGBColor(217, 119, 6)    # #D97706
BLUE_ACCENT = RGBColor(37, 99, 235)     # #2563EB
GREEN_ACCENT = RGBColor(22, 163, 74)    # #16A34A
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
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(9.0), Inches(0.3))
    tf_cat = tb_cat.text_frame
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ORANGE_ACCENT

    tb_num = slide.shapes.add_textbox(Inches(11.5), Inches(0.35), Inches(1.0), Inches(0.3))
    tf_num = tb_num.text_frame
    p_num = tf_num.paragraphs[0]
    p_num.text = slide_num_str
    p_num.font.size = Pt(11)
    p_num.font.bold = True
    p_num.font.color.rgb = TEXT_MUTED
    p_num.alignment = PP_ALIGN.RIGHT

    tb_t = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_DARK

def add_card(slide, left, top, width, height, bg_col=CARD_BG, border_col=BORDER_COLOR):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = bg_col
    card.line.color.rgb = border_col
    card.line.width = Pt(1.2)
    return card

# ----------------- SLIDE 1: TITLE -----------------
s1 = prs.slides.add_slide(blank_layout); set_slide_bg(s1)
tb = s1.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.7), Inches(0.4))
tb.text_frame.paragraphs[0].text = 'FPT UNIVERSITY · DEPARTMENT OF ARTIFICIAL INTELLIGENCE · CAPSTONE REVIEW 1'
tb.text_frame.paragraphs[0].font.size = Pt(10); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = ORANGE_ACCENT

add_card(s1, 0.8, 1.15, 3.2, 0.45, LIGHT_ORANGE, ORANGE_ACCENT)
tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.15), Inches(3.2), Inches(0.45))
tb.text_frame.paragraphs[0].text = 'REAL-TIME INDUSTRIAL SURVEILLANCE'
tb.text_frame.paragraphs[0].font.size = Pt(10); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = ORANGE_ACCENT

tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(7.5), Inches(1.1))
tb.text_frame.paragraphs[0].text = 'Rep-YOLO11s'
tb.text_frame.paragraphs[0].font.size = Pt(46); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = TEXT_DARK

tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.85), Inches(7.6), Inches(1.3))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = 'Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Construction Surveillance'
p.font.size = Pt(16); p.font.color.rgb = TEXT_MUTED

add_card(s1, 8.8, 1.5, 3.7, 2.7, CARD_BG, ORANGE_ACCENT)
tb = s1.shapes.add_textbox(Inches(9.0), Inches(1.7), Inches(3.3), Inches(2.3))
tf = tb.text_frame
tf.paragraphs[0].text = 'PEAK FOLD-3 mAP50'
tf.paragraphs[0].font.size = Pt(11); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '97.11%'; p.font.size = Pt(42); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = '5-Fold Stratified Partition · Tesla T4\nSingle Split mAP50: 94.83% (Unbiased)'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

add_card(s1, 0.8, 4.8, 3.7, 1.9)
tb = s1.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(3.3), Inches(1.7))
tf = tb.text_frame
tf.paragraphs[0].text = 'AUTHORS'
tf.paragraphs[0].font.size = Pt(10); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = TEXT_MUTED
p = tf.add_paragraph(); p.text = 'Nhu Han · SE183644 (Lead Researcher)\nVan-Thanh Nguyen\nTuan-Dung Nguyen'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_DARK

add_card(s1, 4.8, 4.8, 3.7, 1.9)
tb = s1.shapes.add_textbox(Inches(5.0), Inches(4.9), Inches(3.3), Inches(1.7))
tf = tb.text_frame
tf.paragraphs[0].text = 'SUPERVISOR & VENUE'
tf.paragraphs[0].font.size = Pt(10); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = TEXT_MUTED
p = tf.add_paragraph(); p.text = 'MSc. Vu Hai-Anh (FPT University)\nTarget Venue: IEEE TII / IEEE TPAMI\nSession: Capstone Review 1 (Autumn 2026)'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_DARK

add_card(s1, 8.8, 4.8, 3.7, 1.9)
tb = s1.shapes.add_textbox(Inches(9.0), Inches(4.9), Inches(3.3), Inches(1.7))
tf = tb.text_frame
tf.paragraphs[0].text = 'VERIFIED THROUGHPUT'
tf.paragraphs[0].font.size = Pt(10); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = TEXT_MUTED
p = tf.add_paragraph(); p.text = '342.5 FPS · Tesla T4 (TensorRT FP16)\n187.1 FPS · RTX 3050 Laptop\n27.8 FPS · Low-End MX230 (2GB VRAM)'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_DARK

# ----------------- SLIDE 2: PROBLEM BACKGROUND -----------------
s2 = prs.slides.add_slide(blank_layout); set_slide_bg(s2)
add_header(s2, 'Construction Surveillance is Broken. Manual Oversight Cannot Scale.', 'Why This Problem Exists', '02 / 16')

cards_data = [
    ('01', 'Fatal Head Injury Imperative', 'Head injuries from falling debris are the leading cause of fatal and permanently disabling accidents on industrial construction sites.'),
    ('02', 'Human Supervision Bottleneck', 'Human site safety officers are intermittent, occluded, labor-intensive — and physically cannot monitor dozens of high-elevation camera feeds at once.'),
    ('03', 'Automated Non-Intrusive Solution', 'Existing on-site CCTV / RTSP streams already cover active zones — the critical missing layer is automated, real-time PPE compliance auditing.')
]
for i, (num, h, b) in enumerate(cards_data):
    top_pos = 1.45 + i * 1.55
    add_card(s2, 0.8, top_pos, 5.5, 1.4)
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.1), Inches(5.1), Inches(1.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f'{num}  {h}'
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT if i==0 else TEXT_DARK
    p = tf.add_paragraph()
    p.text = b
    p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

# Right side: REAL IMAGE
fig1_path = os.path.join(fig_dir, 'Fig1_site_overview_challenges.jpg')
if os.path.exists(fig1_path):
    add_card(s2, 6.6, 1.45, 5.9, 4.5, CARD_BG, BORDER_COLOR)
    s2.shapes.add_picture(fig1_path, Inches(6.75), Inches(1.6), Inches(5.6), Inches(3.7))
    tb_c = s2.shapes.add_textbox(Inches(6.75), Inches(5.35), Inches(5.6), Inches(0.55))
    tf_c = tb_c.text_frame; tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = 'Fig. 1: Real-world construction CCTV conditions: high camera elevation (15-30m), tiny worker silhouettes, and heavy equipment clutter.'
    p.font.size = Pt(9.5); p.font.italic = True; p.font.color.rgb = TEXT_MUTED

add_card(s2, 0.8, 6.25, 11.7, 0.65, LIGHT_BLUE, BLUE_ACCENT)
tb = s2.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = 'Rep-YOLO11s turns each existing CCTV feed into an automated, continuous, non-intrusive auditor for personal protective equipment.'
p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT

# ----------------- SLIDE 3: 4 BOTTLENECKS -----------------
s3 = prs.slides.add_slide(blank_layout); set_slide_bg(s3)
add_header(s3, 'Four Core Bottlenecks of Standard CNNs on Construction Sites', 'Mathematical & Physical Challenges', '03 / 16')

b_data = [
    ('01', 'DISTANT TINY TARGETS', 'Helmets collapse under 20x20 px — vanishing through stride-16/32 backbones.', '15-30m high-angle CCTV compresses helmets to <15x15 px. Standard downsampling erases spatial signals before the detection head sees them.', ORANGE_ACCENT),
    ('02', 'ACUTE CLASS IMBALANCE', '9,044 helmets vs 111,514 persons — an extreme 1:12 starvation ratio.', 'Loss terms get dominated by easy, abundant worker body boxes. Rare small helmets receive almost zero gradient updates by construction.', AMBER_ACCENT),
    ('03', 'SPATIAL & COLOR CLUTTER', 'Yellow machinery, buckets, warning signs mimic helmet geometry.', 'Translation-invariant convolutions cannot distinguish yellow on top of a head from yellow on the floor. False alarms fill the safety report.', BLUE_ACCENT),
    ('04', 'EDGE-AI LATENCY BOUND', 'Surveillance demands >=60 FPS — multi-camera on on-premise laptops.', 'Heavy Transformer detectors crush memory-access cost (MAC). They collapse on the entry-level laptop GPU running the guard house.', GREEN_ACCENT)
]
for idx, (num, tag, headline, body, col) in enumerate(b_data):
    r = idx // 2; c = idx % 2
    l_pos = 0.8 + c * 6.0; t_pos = 1.5 + r * 2.5
    add_card(s3, l_pos, t_pos, 5.7, 2.25)
    tb = s3.shapes.add_textbox(Inches(l_pos + 0.2), Inches(t_pos + 0.15), Inches(5.3), Inches(1.95))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = f'{num}  ·  {tag}'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = col
    p = tf.add_paragraph(); p.text = headline; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = body; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

tb_foot = s3.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(11.7), Inches(0.4))
p = tb_foot.text_frame.paragraphs[0]
p.text = 'Choosing the right structural priors and attention mechanisms four ways — without paying an inference latency tax.'
p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 4: LITERATURE SURVEY -----------------
s4 = prs.slides.add_slide(blank_layout); set_slide_bg(s4)
add_header(s4, 'State-of-the-Art Falls Short in Four Known Directions', 'Survey of 32 Benchmark Papers (2019 - 2026)', '04 / 16')

sota_cards = [
    ('YOLOv8 / v10 / 11', 'General-Purpose Baselines', 'Fast general accuracy, but lack spatial coordinate priors for tiny targets, producing high false-positive rates amid yellow construction clutter.'),
    ('EC-YOLOv8 · 2024', '95.70% mAP — 172 FPS', 'CARAFE upsampling raises mAP50 to 95.70%, but inflates computation and drops desktop-GPU throughput to 172 FPS — insufficient for multi-camera feeds.'),
    ('YOLO-CBF · 2023', '37.2 M params · 80.6 FPS', 'Adds CoordConv and BiFormer, but at 37.2M parameters and 104.5 GFLOPs. Forward latency drops to 80.6 FPS — heavy by construction.'),
    ('YOLOv8n-FADS · 2024', 'P2 Head Expansion', 'P2 head expansion helped coal-mine detections but explodes feature-map resolution and becomes a severe latency bottleneck on high-resolution CCTV.')
]
for i, (tag, h, b) in enumerate(sota_cards):
    t_pos = 1.45 + i * 1.18
    add_card(s4, 0.8, t_pos, 7.5, 1.05)
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(t_pos + 0.08), Inches(7.1), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = f'{tag}  —  {h}'; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = b; p.font.size = Pt(10); p.font.color.rgb = TEXT_MUTED

add_card(s4, 8.6, 1.45, 3.9, 4.8, LIGHT_ORANGE, ORANGE_ACCENT)
tb = s4.shapes.add_textbox(Inches(8.8), Inches(1.65), Inches(3.5), Inches(4.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'THE RESEARCH GAP'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = 'No prior model achieves small-object accuracy without an inference latency penalty.'; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Each prior direction forces a severe tradeoff between precision and real-time speed.\n\nOur core thesis: Training-time richness and deployment-time parsimony can co-exist when algebraic re-parameterization fusion is provably correct.'; p.font.size = Pt(11.5); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 5: SCOPE & 5 DELIVERABLES -----------------
s5 = prs.slides.add_slide(blank_layout); set_slide_bg(s5)
add_header(s5, 'Five Concrete Deliverables — Reviewed, Committed, and Dated', 'Project Scope & Deliverables', '05 / 16')

delivs = [
    ('01', 'IEEE Transactions Paper', '9-page manuscript with complete math proofs, SOTA benchmarks, A0-A6 ablation, Grad-CAM XAI, and 32-paper BibTeX.'),
    ('02', 'Rep-YOLO11s Weights', 'Novel model weights reaching 94.83% mAP50 (single split), 96.64% +/- 0.32% (5-fold CV), 97.03% on Hard Hat Workers.'),
    ('03', 'End-to-End RTSP Pipeline', 'H.264/H.265 hardware decoding, CoordConv prep, TensorRT inference, NMS, alert UI — 65 to 95 FPS on RTX 3050 Laptop.'),
    ('04', 'Compiled Deployment Engines', 'TensorRT 11.2 FP16 (.engine), ONNX Runtime INT8 (.onnx), and OpenVINO IR for zero-overhead multi-platform deployment.'),
    ('05', 'Standardized Dataset Suite', '33,000+ canonicalized images across VOC2028, GDUT-HWD, SHEL5K, Hard Hat Workers, SHD, SFCHD with harmonized PPE protocol.')
]
for i, (num, h, b) in enumerate(delivs):
    t_pos = 1.45 + i * 0.95
    add_card(s5, 0.8, t_pos, 7.5, 0.85)
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(t_pos + 0.05), Inches(7.1), Inches(0.75))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = f'{num}  {h}'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT if i==0 else TEXT_DARK
    p = tf.add_paragraph(); p.text = b; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_MUTED

add_card(s5, 8.6, 1.45, 3.9, 4.8, LIGHT_BLUE, BLUE_ACCENT)
tb = s5.shapes.add_textbox(Inches(8.8), Inches(1.65), Inches(3.5), Inches(4.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'TRAINING & EDGE TESTBED'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = '2 x Tesla T4 Nodes'; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Kaggle Dual-GPU DDP clusters for 100-epoch full convergence.\n\nEdge Hardware Audit:\n- Server GPU: NVIDIA Tesla T4 (2.92 ms)\n- Edge Laptop: RTX 3050 Laptop (5.35 ms)\n- Budget Consumer: GeForce MX230 2GB (36.0 ms, 27.8 FPS)\n- Embedded CPU: Intel 4-Core INT8 (28.56 ms)\n\nEvery reported millisecond is physically measured with CUDA synchronization.'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 6: NEURAL ARCHITECTURE -----------------
s6 = prs.slides.add_slide(blank_layout); set_slide_bg(s6)
add_header(s6, 'Overall Neural Network Architecture of Rep-YOLO11s', 'Proposed Neural Network Architecture', '06 / 16')

fig2_path = os.path.join(fig_dir, 'Fig2_rep_yolo11s_neural_architecture.png')
if os.path.exists(fig2_path):
    add_card(s6, 0.8, 1.4, 11.7, 3.8, CARD_BG, BORDER_COLOR)
    s6.shapes.add_picture(fig2_path, Inches(0.9), Inches(1.45), Inches(11.5), Inches(3.7))

subsys = [
    ('01 · BACKBONE', 'CoordConv + RepConv Blocks', 'Input stem concatenated with normalized coordinates (Cx, Cy). Multi-branch RepConv captures diverse features during training, fusing to single 3x3 at deploy.'),
    ('02 · NECK', 'Bi-Level Routing Attention (BiFormer)', 'Top-k region selection dynamically routes attention strictly to helmet candidate tokens, dropping quadratic cost toward linear O(S^2 + k*HW/S^2).'),
    ('03 · HEAD', 'Decoupled Head + Focal EIoU', 'Decoupled classification and regression branches. Focal EIoU explicitly optimizes width/height errors and weights hard occluded boxes.')
]
for i, (tag, h, b) in enumerate(subsys):
    l_pos = 0.8 + i * 4.0
    add_card(s6, l_pos, 5.35, 3.7, 1.65)
    tb = s6.shapes.add_textbox(Inches(l_pos + 0.15), Inches(5.45), Inches(3.4), Inches(1.45))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = tag; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT if i==0 else (BLUE_ACCENT if i==1 else GREEN_ACCENT)
    p = tf.add_paragraph(); p.text = h; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = b; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 7: INNOVATION 1 (REPCONV) -----------------
s7 = prs.slides.add_slide(blank_layout); set_slide_bg(s7)
add_header(s7, 'Train Rich, Deploy Parsimonious: Structural Re-Parameterization', 'Innovation 1 of 3 · Structural Re-Parameterization', '07 / 16')

add_card(s7, 0.8, 1.45, 5.6, 4.8)
tb = s7.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = '01 · TRAINING PHASE (MULTI-BRANCH)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = 'y = BN(W_3x3 * x) + BN(W_1x1 * x) + BN(x)'; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Three parallel branches learn complementary edge, texture, and identity representations. Gradient diversity avoids local minima during backpropagation.'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED
p = tf.add_paragraph(); p.text = '\n02 · DEPLOYMENT PHASE (SWITCH_TO_DEPLOY)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = 'y = W_fused * x + b_fused (Single 3x3 Conv)'; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'One single operator replaces the entire multi-branch subgraph. Zero parameter overhead, zero branch switching delay, bit-exact equivalent under BN fusion.'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

add_card(s7, 6.7, 1.45, 5.8, 2.6)
tb = s7.shapes.add_textbox(Inches(6.9), Inches(1.6), Inches(5.4), Inches(2.3))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'ALGEBRAIC FUSION — CLOSED FORM, NO APPROXIMATION'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_MUTED
p = tf.add_paragraph(); p.text = 'W_fused = W\'_3x3 + Pad(W\'_1x1) + W\'_id\nb_fused = b\'_3x3 + b\'_1x1 + b\'_id'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = 'where W\'_i = (gamma_i / sqrt(sigma_i^2 + eps)) * W_i\nand b\'_i = beta_i - (gamma_i * mu_i / sqrt(sigma_i^2 + eps))'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

add_card(s7, 6.7, 4.25, 5.8, 2.0, LIGHT_ORANGE, ORANGE_ACCENT)
tb = s7.shapes.add_textbox(Inches(6.9), Inches(4.4), Inches(5.4), Inches(1.7))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'EMPIRICAL INFERENCE GAIN · TESLA T4'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '7.12 ms  →  2.92 ms  (342.5 FPS)'; p.font.size = Pt(24); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Forward latency collapsed by 59.0%. Accuracy delta: exactly 0.00%. mAP50 holds firmly at 94.83%.'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 8: INNOVATION 2 (COORDCONV & BIFORMER) -----------------
s8 = prs.slides.add_slide(blank_layout); set_slide_bg(s8)
add_header(s8, 'Two Spatial Mechanisms: Coordinate Injection & Dynamic Routing', 'Innovation 2 of 3 · Spatial Priors & Dynamic Routing', '08 / 16')

add_card(s8, 0.8, 1.45, 5.6, 4.8)
tb = s8.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(4.3))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'COORDCONV: BREAKING TRANSLATION INVARIANCE'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = 'Input Tensor: [RGB; Cx; Cy]  in  R^(5 x H x W)'; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Cx(i, j) = 2j/(W-1) - 1,   Cy(i, j) = 2i/(H-1) - 1'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = '\nAnatomical Prior:\n- Safety helmets strictly appear above human torsos/heads.\n- Yellow construction buckets, warning cones, and ground clutter reside at floor level (Cy -> +1).\n- Cost: exactly +2 channels on 640x640 stem (0.078% parameter delta). Hardcoded geometric prior with zero runtime latency tax!'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

add_card(s8, 6.7, 1.45, 5.8, 4.8)
tb = s8.shapes.add_textbox(Inches(6.9), Inches(1.65), Inches(5.4), Inches(4.3))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'BIFORMER: BI-LEVEL ROUTING SPARSE ATTENTION'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = 'Attention Complexity: O(S^2 + k * HW/S^2) << O(H^2 W^2)'; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Step 1: Region Partition (S x S) & Regional Affinity A^r = Q^r (K^r)^T\nStep 2: Top-k Routing Index Pruning (gating out 85%+ background)\nStep 3: Fine-grained token attention across gathered regions.'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED
p = tf.add_paragraph(); p.text = '\nOutcome: 100% of attention compute focuses on tiny, occluded helmet regions without quadratic self-attention blowup.'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT

# ----------------- SLIDE 9: INNOVATION 3 (FOCAL EIOU) -----------------
s9 = prs.slides.add_slide(blank_layout); set_slide_bg(s9)
add_header(s9, 'Tight Boxes on Tiny, Occluded Helmets: Focal EIoU Loss', 'Innovation 3 of 3 · Focal EIoU Regression Loss', '09 / 16')

add_card(s9, 0.8, 1.45, 3.7, 4.2)
tb = s9.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(3.3), Inches(3.8))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'LIMITATION OF CIOU'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = 'Aspect Ratio Ambiguity'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'CIoU uses relative aspect ratio:\nv proportional to (arctan(w_gt/h_gt) - arctan(w/h))^2\n\nWhen objects are partially occluded by scaffolding, aspect ratio saturates even when width and height are completely mismatched. Gradients vanish for tiny boxes.'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

add_card(s9, 4.8, 1.45, 3.7, 4.2)
tb = s9.shapes.add_textbox(Inches(5.0), Inches(1.65), Inches(3.3), Inches(3.8))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'THE FIX: EIOU DECOMPOSITION'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = 'Independent Side Lengths'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'L_EIoU = L_IoU + L_dis + L_asp\n= (1 - IoU) + rho^2(b, b_gt)/c^2 + rho^2(w, w_gt)/C_w^2 + rho^2(h, h_gt)/C_h^2\n\nDirectly minimizes real width/height errors independently, preventing box drift.'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

add_card(s9, 8.8, 1.45, 3.7, 4.2, LIGHT_GREEN, GREEN_ACCENT)
tb = s9.shapes.add_textbox(Inches(9.0), Inches(1.65), Inches(3.3), Inches(3.8))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'FOCAL RE-WEIGHTING'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
p = tf.add_paragraph(); p.text = 'IoU^gamma Hard Weighting'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'L_Focal-EIoU = IoU^0.5 * L_EIoU\n\nEasy background boxes receive suppressed gradients, while hard, occluded helmet targets with low IoU receive boosted supervision.'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

add_card(s9, 0.8, 5.85, 11.7, 0.95, CARD_BG, BORDER_COLOR)
tb = s9.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.3), Inches(0.75))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'Empirical Localization Consequence: Razor-sharp localization where helmets meet scaffolding — mAP50 rises +0.14% and mAP50-95 reaches 62.54% with zero extra inference overhead.'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = TEXT_DARK

# ----------------- SLIDE 10: DATA ENGINEERING -----------------
s10 = prs.slides.add_slide(blank_layout); set_slide_bg(s10)
add_header(s10, 'One Source, Six External Sites, One Canonical Schema', 'Data Engineering & Harmonized Canonical Space', '10 / 16')

add_card(s10, 0.8, 1.45, 5.6, 4.0)
tb = s10.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(3.6))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'SOURCE DOMAIN · SHWD / VOC2028'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '7,581 Authentic Images (80/20 Fixed Split)'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = '· 6,064 Trainval images / 1,517 Test images (Strict 100% leak-free split)\n· Purged 3 erroneous "dog" XML labels from VOC2028 originals\n· Normalized coordinates: [x_c, y_c, w, h] in [0, 1]\n· Severe class imbalance: 9,044 helmets vs 111,514 bodies (1:12 ratio)'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

add_card(s10, 6.7, 1.45, 5.8, 4.0)
tb = s10.shapes.add_textbox(Inches(6.9), Inches(1.65), Inches(5.4), Inches(3.6))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'EXTERNAL MULTI-DOMAIN SUITE (33,000+ IMAGES)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
p = tf.add_paragraph(); p.text = 'Five Heterogeneous External Testbeds'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = '· GDUT-HWD: 13,499 images (Dense crowds, 15-30 workers/frame)\n· SHEL5K: 5,000 images (Vertical drone nadir views, 70-90 degrees)\n· Hard Hat Workers: 7,000 images (Outdoor construction surveillance)\n· Safety Helmet Detection (SHD) & SFCHD: Industrial plant CCTV feeds'; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

add_card(s10, 0.8, 5.65, 11.7, 1.2, LIGHT_ORANGE, ORANGE_ACCENT)
tb = s10.shapes.add_textbox(Inches(1.0), Inches(5.75), Inches(11.3), Inches(1.0))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'CANONICAL CLASS SPACE C* = {0: "hat" (Helmet), 1: "person" (Bare Head / Worker)}'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = 'Harmonized PPE Protocol: Eliminates annotation discrepancies across datasets (Head-only vs Full-body) to deliver true scientific zero-shot domain transfer.'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK

# ----------------- SLIDE 11: SOTA BENCHMARK -----------------
s11 = prs.slides.add_slide(blank_layout); set_slide_bg(s11)
add_header(s11, 'Outperforming SOTA Baselines in Precision and Throughput', 'State-of-the-Art Comparison · Table I', '11 / 16')

add_card(s11, 0.8, 1.45, 5.8, 4.8)
tb = s11.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.4), Inches(4.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'QUANTITATIVE BENCHMARK (TABLE I IN PAPER)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '· YOLOv8s: 94.89% mAP50 | 62.21% mAP50-95 | 6.10 ms (163.9 FPS)\n· YOLOv10s: 94.39% mAP50 | 62.19% mAP50-95 | 6.23 ms (160.5 FPS)\n· YOLO11s Baseline: 94.74% mAP50 | 62.54% mAP50-95 | 6.52 ms (153.3 FPS)\n· EC-YOLOv8 (2024): 95.70% mAP50 | 5.80 ms (172.4 FPS)\n· YOLO-CBF (2023): 95.60% mAP50 | 12.40 ms (80.6 FPS)\n-----------------------------------------------------\n· Rep-YOLO11s (Ours Single): 94.83% mAP50 | 62.54% mAP50-95 | 2.92 ms (342.5 FPS)\n· Rep-YOLO11s (5-Fold Mean): 96.64 +/- 0.32% mAP50 | 65.91% mAP50-95 (Peak: 97.11%)'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = '\nTakeaway: Half the forward latency of baseline YOLO11s with equal or higher mAP!'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT

fig5_path = os.path.join(fig_dir, 'Fig5_efficiency_frontier_latency_vs_map.png')
if os.path.exists(fig5_path):
    add_card(s11, 6.9, 1.45, 5.6, 4.8)
    s11.shapes.add_picture(fig5_path, Inches(7.05), Inches(1.6), Inches(5.3), Inches(3.8))
    tb_c = s11.shapes.add_textbox(Inches(7.05), Inches(5.5), Inches(5.3), Inches(0.6))
    tf_c = tb_c.text_frame; tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = 'Fig. 5: Efficiency Frontier: Latency (ms) vs mAP50-95 (%). Rep-YOLO11s establishes the dominant Pareto frontier.'
    p.font.size = Pt(9.5); p.font.italic = True; p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 12: ABLATION STUDY -----------------
s12 = prs.slides.add_slide(blank_layout); set_slide_bg(s12)
add_header(s12, 'Systematic Ablation Analysis: Each Component Earns Its Slot', 'Systematic Ablation Study · A0 to A6', '12 / 16')

add_card(s12, 0.8, 1.45, 5.8, 4.8)
tb = s12.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.4), Inches(4.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'PROGRESSIVE COMPONENT BREAKDOWN (TABLE II)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '· A0 (Baseline YOLO11s): 94.74% mAP50 | 62.34% mAP50-95 | 6.52 ms\n· A1 (+ P2 Small Head): 94.81% (+0.07%), but Latency surges to 8.94 ms [REJECTED]\n· A2 (+ CoordConv): 94.78% mAP50 | 62.45% mAP50-95 | 6.58 ms (Zero cost)\n· A3 (+ RepConv): 94.81% mAP50 | 62.48% mAP50-95 | 6.64 ms (Pre-deploy)\n· A4 (+ Focal EIoU): 94.88% mAP50 | 62.51% mAP50-95 (Tight boundaries)\n· A5 (+ BiFormer): 94.80% mAP50 | Recall reaches 91.15%\n-----------------------------------------------------\n· A6 (Full Fusion Champion): 94.83% mAP50 | 62.54% mAP50-95\n  Latency collapses from 7.12 ms -> 2.92 ms after switch_to_deploy!'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK

fig6_path = os.path.join(fig_dir, 'Fig6_map_ablation_comparison.png')
if os.path.exists(fig6_path):
    add_card(s12, 6.9, 1.45, 5.6, 4.8)
    s12.shapes.add_picture(fig6_path, Inches(7.05), Inches(1.6), Inches(5.3), Inches(3.8))
    tb_c = s12.shapes.add_textbox(Inches(7.05), Inches(5.5), Inches(5.3), Inches(0.6))
    tf_c = tb_c.text_frame; tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = 'Fig. 6: Quantitative mAP progression across A0-A6 showing consistent accuracy gains.'
    p.font.size = Pt(9.5); p.font.italic = True; p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 13: GRAD-CAM XAI (THE CRUCIAL SLIDE) -----------------
s13 = prs.slides.add_slide(blank_layout); set_slide_bg(s13)
add_header(s13, 'Visual Saliency Proof: Heatmaps Land on Helmets, Not Clutter', 'Explainable AI · Grad-CAM Visual Sanity Check', '13 / 16')

fig3_path = os.path.join(fig_dir, 'Fig3_gradcam_xai_saliency_comparison.png')
if os.path.exists(fig3_path):
    add_card(s13, 0.8, 1.4, 11.7, 4.2)
    s13.shapes.add_picture(fig3_path, Inches(0.9), Inches(1.45), Inches(11.5), Inches(4.1))

scenarios = [
    ('Scenario 1: Vests & Scaffolding', 'Baseline YOLO11s leaks gradient onto orange vests and timber poles. Rep-YOLO11s concentrates centroids strictly on 4 helmets (conf 0.84).'),
    ('Scenario 2: Window Glare & Backlight', 'Baseline gradients drown in window illumination. BiFormer dynamic routing pierces harsh glare to lock onto tiny helmets (conf 0.89).'),
    ('Scenario 3: Yellow Warning Signs', 'Baseline activates heavily on triangular yellow hazard signs (false alarms). CoordConv spatial prior completely rejects signs, locking onto heads only.')
]
for i, (h, b) in enumerate(scenarios):
    l_pos = 0.8 + i * 4.0
    add_card(s13, l_pos, 5.75, 3.7, 1.35)
    tb = s13.shapes.add_textbox(Inches(l_pos + 0.15), Inches(5.8), Inches(3.4), Inches(1.25))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = h; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT if i==2 else TEXT_DARK
    p = tf.add_paragraph(); p.text = b; p.font.size = Pt(9); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 14: CROSS DOMAIN & IOU COLLAPSE -----------------
s14 = prs.slides.add_slide(blank_layout); set_slide_bg(s14)
add_header(s14, 'Zero-Shot Generalization & Mathematical IoU Collapse Autopsy', 'Cross-Domain Transfer & Label Harmonization', '14 / 16')

add_card(s14, 0.8, 1.45, 6.0, 4.8)
tb = s14.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.6), Inches(4.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'CROSS-DATASET BENCHMARK RESULTS (TABLE III)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '· Hard Hat Workers (AndrewMVD, 7,000 imgs): 97.03% mAP50 (Harmonized PPE) — perfectly matches source domain fidelity!\n· GDUT-HWD (13,499 imgs): 74.27% mAP50, 90.26% Precision in dense crowds (15-30 workers/frame).\n· Safety Helmet Detection (SHD): 76.85% mAP50\n· SFCHD Benchmark: 64.80% mAP50\n· SHEL5K (5,000 imgs): 41.15% mAP50 (Drone nadir 90 deg view)'; p.font.size = Pt(11); p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = '\nTakeaway: The model transfers robustly across ground-level sites; precision remains high (85-94%) across all external datasets.'; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT

add_card(s14, 7.1, 1.45, 5.4, 4.8, LIGHT_ORANGE, ORANGE_ACCENT)
tb = s14.shapes.add_textbox(Inches(7.3), Inches(1.65), Inches(5.0), Inches(4.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'IOU COLLAPSE AUTOPSY: WHY JOINT DROPPED TO 74.40%'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = 'IoU = A_head / A_body approx 0.07 - 0.14 << 0.50'; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_DARK
p = tf.add_paragraph(); p.text = 'Label Schema Conflict:\n- Source SHWD defines "person" as Full-Body.\n- Hard Hat Workers defines "person" as Head-Only.\n\nWhen Rep-YOLO11s predicts the full body, the ground-truth head box is nested inside. Because IoU << 0.50, the model is penalized as BOTH a False Positive and a False Negative!\n\nHarmonized PPE Resolution:\nEvaluating on the helmet class alone (where definitions are identical) yields 97.03% mAP50, proving zero feature degradation.'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 15: HARDWARE DEPLOYMENT & RTSP -----------------
s15 = prs.slides.add_slide(blank_layout); set_slide_bg(s15)
add_header(s15, 'From Tesla T4 Servers to a 2-GB Edge Laptop: All Real-Time', 'Hardware Deployment & End-to-End RTSP Pipeline', '15 / 16')

add_card(s15, 0.8, 1.45, 5.6, 4.8)
tb = s15.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'MULTI-PLATFORM HARDWARE AUDIT (TABLE IV)'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
p = tf.add_paragraph(); p.text = '· Tesla T4 · TensorRT FP16: 2.92 ms  |  342.5 FPS (Ultra Real-Time)\n· RTX 3050 Laptop · TRT FP16: 5.35 ms  |  187.1 FPS (Edge AI)\n· RTX 3050 Laptop · Native FP32: 12.43 ms  |  80.4 FPS\n· GeForce MX230 (2GB VRAM) · FP32: 36.00 ms  |  27.8 FPS [BUDGET]\n· Edge CPU (4-Cores) · INT8: 28.56 ms  |  35.0 FPS\n-----------------------------------------------------\nOn a budget Pascal MX230 with only 2GB VRAM and NO Tensor Cores, Rep-YOLO11s still delivers 27.8 FPS — clearing the 24 FPS real-time floor!'; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_DARK

fig4_path = os.path.join(fig_dir, 'Fig4_industrial_rtsp_surveillance_pipeline.png')
if os.path.exists(fig4_path):
    add_card(s15, 6.7, 1.45, 5.8, 4.8)
    s15.shapes.add_picture(fig4_path, Inches(6.85), Inches(1.55), Inches(5.5), Inches(2.6))
    tb_c = s15.shapes.add_textbox(Inches(6.85), Inches(4.25), Inches(5.5), Inches(1.9))
    tf_c = tb_c.text_frame; tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = 'END-TO-END RTSP VIDEO PIPELINE BREAKDOWN:'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
    p = tf_c.add_paragraph()
    p.text = 'T_total = T_dec (3.5-5.0ms) + T_prep (1.2-2.0ms) + T_gpu (2.92-5.35ms) + T_nms (1.5-2.8ms) + T_ui (2.2-3.4ms) = 10.54 - 15.38 ms\n\nYields 65 to 95 FPS live camera stream throughput on RTX 3050 Laptop!'
    p.font.size = Pt(10); p.font.color.rgb = TEXT_MUTED

# ----------------- SLIDE 16: REVIEW 1 SUMMARY & ROADMAP -----------------
s16 = prs.slides.add_slide(blank_layout); set_slide_bg(s16)
add_header(s16, 'Review 1 is Complete: What Shipped & Roadmap for Review 2 & 3', 'Summary & Future Milestones', '16 / 16')

cols_data = [
    ('REVIEW 1 · COMPLETED', 'Problem & Feasibility Proven', '· All 4 Review-1 criteria answered with measurable evidence.\n· 9-page IEEE paper, A0-A6 ablation study, Grad-CAM XAI.\n· Verified physical deployment on T4 (342.5 FPS), RTX 3050 (187.1 FPS), and MX230 2GB (27.8 FPS).\n· 94.83% mAP50 single split, 96.64% on 5-fold CV.', ORANGE_ACCENT),
    ('REVIEW 2 · MID-TERM PLAN', 'Software Dashboard & Dispatch', '· Industrial Desktop GUI / Web Monitoring Dashboard.\n· Automated violation alerting & audit log replay.\n· Dynamic multi-camera routing & hyperparameter sweep.\n· Edge cloud integration: MQTT + WebRTC stream push.', BLUE_ACCENT),
    ('REVIEW 3 · FINAL DEFENSE', 'Distillation & Official Defense', '· Knowledge distillation (YOLO11x -> Rep-YOLO11s) to improve steep drone angles (SHEL5K).\n· Submit IEEE Transactions manuscript under LaTeX Overleaf.\n· Finalize thesis documentation and committee defense binder.\n· Open-source standardized benchmark dataset.', GREEN_ACCENT)
]
for i, (tag, h, b, col) in enumerate(cols_data):
    l_pos = 0.8 + i * 4.0
    add_card(s16, l_pos, 1.45, 3.7, 4.3)
    tb = s16.shapes.add_textbox(Inches(l_pos + 0.15), Inches(1.65), Inches(3.4), Inches(3.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = tag; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = col
    p = tf.add_paragraph(); p.text = h; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = b; p.font.size = Pt(10.5); p.font.color.rgb = TEXT_MUTED

add_card(s16, 0.8, 5.95, 11.7, 0.9, LIGHT_ORANGE, ORANGE_ACCENT)
tb = s16.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.3), Inches(0.7))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = 'Rep-YOLO11s — Technical Feasibility, Scientific Novelty, and Industrial Applicability: Scoped, Built, and Empirically Measured.'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT

out_path = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep_YOLO11s_Capstone_Review1_Deck.pptx'
prs.save(out_path)
print(f'SUCCESS! Wrote complete 16-slide PowerPoint to: {out_path}')
