"""
Script to render high-resolution (300 DPI) publication-grade PNG images
for Sections 8, 9, 10, 11, and 12 of the Capstone Review 1 presentation.

Outputs saved to: review_1_main/slide_renders_8_to_12/
Uses exact Calibri font family with robust text wrapping and clean vector badges.
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure Matplotlib fonts to strictly use Calibri
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Segoe UI', 'Arial', 'DejaVu Sans']

# Ensure output directory exists
out_dir = os.path.join(os.getcwd(), 'review_1_main', 'slide_renders_8_to_12')
os.makedirs(out_dir, exist_ok=True)

# Color Palette
C_BG = '#FAF8F5'
C_CARD_BG = '#FFFFFF'
C_BORDER = '#E2E8F0'
C_ORANGE = '#EA580C'
C_BLUE = '#2563EB'
C_AMBER = '#D97706'
C_GREEN = '#16A34A'
C_PURPLE = '#7C3AED'
C_RED = '#DC2626'
C_SLATE = '#475569'
C_TITLE = '#0F172A'
C_BODY = '#334155'
C_MUTED = '#64748B'
C_BANNER_BG = '#F8FAFC'
C_BANNER_BORDER = '#CBD5E1'

def wrap_str(text, width=65):
    return textwrap.fill(text, width=width)

def setup_base_slide(criterion_tag, slide_num, title_text):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Category Tag
    ax.text(0.8, 8.4, criterion_tag.upper(), fontsize=12, fontweight='bold',
            color=C_ORANGE, va='center')

    # Slide Number
    ax.text(15.2, 8.4, slide_num, fontsize=13, fontweight='normal',
            color=C_MUTED, va='center', ha='right')

    # Main Title
    ax.text(0.8, 7.85, title_text, fontsize=21, fontweight='bold',
            color=C_TITLE, va='center')

    return fig, ax

def add_banner(ax, text):
    banner = patches.FancyBboxPatch(
        (0.8, 0.45), 14.4, 0.65,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=C_BANNER_BG, edgecolor=C_BANNER_BORDER, linewidth=1.2
    )
    ax.add_patch(banner)
    
    # Left pill badge
    badge = patches.FancyBboxPatch(
        (1.0, 0.58), 1.5, 0.38,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=C_ORANGE, edgecolor=C_ORANGE
    )
    ax.add_patch(badge)
    ax.text(1.75, 0.77, "TAKEAWAY", fontsize=9.5, fontweight='bold',
            color='#FFFFFF', va='center', ha='center')

    ax.text(2.65, 0.77, text, fontsize=11, fontweight='bold',
            color=C_TITLE, va='center')

def draw_card(ax, x, y, w, h, border_color=C_BORDER, bg_color=C_CARD_BG, lw=1.2):
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=bg_color, edgecolor=border_color, linewidth=lw
    )
    ax.add_patch(card)

def draw_badge(ax, x, y, w, h, text, bg_color, fontsize=11):
    badge = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor=bg_color, edgecolor=bg_color, linewidth=1
    )
    ax.add_patch(badge)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize, fontweight='bold',
            color='#FFFFFF', va='center', ha='center')

# =====================================================================
# SLIDE 8: EVALUATION STRATEGY
# =====================================================================
def render_slide_8():
    fig, ax = setup_base_slide(
        "CRITERION 4 · EVALUATION STRATEGY", "08 / 18",
        "Rigorous Multi-Metric Evaluation & Life-Critical Recall Strategy"
    )

    # Card 1: Detection Metrics (Top-Left)
    draw_card(ax, 0.8, 4.3, 7.0, 3.1)
    draw_badge(ax, 1.05, 6.85, 0.6, 0.35, "01", C_BLUE)
    ax.text(1.8, 7.02, "Core Accuracy & Localization Metrics", fontsize=14, fontweight='bold', color=C_TITLE, va='center')
    
    lines_1 = [
        ("• mAP50 (Gold Standard):", "Primary benchmark at IoU 0.50 measuring holistic detection accuracy."),
        ("• mAP50-95 (Boundary Tightness):", "Strict average across 10 IoU steps [0.50:0.05:0.95]; penalizes loose bounding boxes."),
        ("• Precision & F1-Score:", "Monitors trade-off between false detections (ground objects) and completeness."),
        ("• Real-Time Constraint:", "Enforces strict inference latency suitable for 25+ FPS real-time surveillance streams.")
    ]
    y_pos = 6.45
    for title, desc in lines_1:
        ax.text(1.1, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(1.1, y_pos - 0.25, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.52

    # Card 2: Life-Critical Recall (Top-Right)
    draw_card(ax, 8.2, 4.3, 7.0, 3.1)
    draw_badge(ax, 8.45, 6.85, 0.6, 0.35, "02", C_ORANGE)
    ax.text(9.2, 7.02, "Life-Critical Priority: Recall Over Precision", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_2 = [
        ("• Zero-Tolerance for False Negatives (FN):", "A missed violation allows an unprotected worker into a hazard zone -> fatal risk."),
        ("• Asymmetric Risk Impact:", "A False Alarm (FP) causes minor verification overhead; a Missed Violation (FN) causes loss of life."),
        ("• Target Recallhat > 90%:", "Model loss weights and label assigner are heavily biased to prioritize helmet sensitivity."),
        ("• Gradient Starvation Defense:", "Focal BCE and Task-Aligned Assigner prevent 1:12 worker body labels from collapsing helmet loss.")
    ]
    y_pos = 6.45
    for title, desc in lines_2:
        ax.text(8.5, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(8.5, y_pos - 0.25, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.52

    # Card 3: 5-Fold Stratified CV (Bottom-Left)
    draw_card(ax, 0.8, 1.35, 7.0, 2.75)
    draw_badge(ax, 1.05, 3.55, 0.6, 0.35, "03", C_AMBER)
    ax.text(1.8, 3.72, "5-Fold Stratified Cross-Validation Protocol", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_3 = [
        ("• Strict Stratification:", "5 splits maintain identical 1:12.33 class ratio, preventing partition imbalance."),
        ("• Statistical Significance (μ ± σ):", "Reports 96.64% ± 0.32% mAP50, proving results are not due to an easy split."),
        ("• Cross-Fold Stability:", "Validates robust generalization across varying weather, lighting, and worker density."),
        ("• Ablation Integrity:", "Ensures every architectural improvement is statistically meaningful and reproducible.")
    ]
    y_pos = 3.15
    for title, desc in lines_3:
        ax.text(1.1, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(1.1, y_pos - 0.23, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.48

    # Card 4: Grad-CAM XAI (Bottom-Right)
    draw_card(ax, 8.2, 1.35, 7.0, 2.75)
    draw_badge(ax, 8.45, 3.55, 0.6, 0.35, "04", C_GREEN)
    ax.text(9.2, 3.72, "Model Transparency: Grad-CAM XAI Audit", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_4 = [
        ("• Feature Saliency Verification:", "Grad-CAM heatmaps confirm gradient activations focus squarely on head-worn helmets."),
        ("• Floor Clutter Suppression:", "Directly proves CoordConv cancels out false alarms from ground buckets & orange vests."),
        ("• Long-Range Attention Focus:", "Verifies BiFormer directs fine-grained attention to small distant targets (<20px)."),
        ("• Industrial Safety Compliance:", "Meets HSE safety auditor requirements for transparent, explainable decision boundaries.")
    ]
    y_pos = 3.15
    for title, desc in lines_4:
        ax.text(8.5, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(8.5, y_pos - 0.23, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.48

    add_banner(ax, "Multi-metric evaluation prioritizes Life-Critical Recall while eliminating split bias via 5-Fold Stratified CV and Grad-CAM interpretability.")
    fig.savefig(os.path.join(out_dir, 'Slide_08_Evaluation_Strategy.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Rendered: Slide_08_Evaluation_Strategy.png")

# =====================================================================
# SLIDE 9: PROJECT PLAN & GANTT CHART
# =====================================================================
def render_slide_9():
    fig, ax = setup_base_slide(
        "CRITERION 5 · PROJECT PLAN & GANTT MILESTONES", "09 / 18",
        "Project Roadmap: 5 Phases & Milestone Tracking (W1–W15)"
    )

    phases = [
        ("Phase 1 (W1–W5)", "Foundation & Review 1", "30-paper survey, RQ/Gap, SHWD data cleaning, Baseline setup.", "Review 1 Deck & Cleaned Data", C_ORANGE),
        ("Phase 2 (W6–W8)", "Core Engineering & Review 2", "Module coding (RepConv, CoordConv, BiFormer, EIoU), Ablation A0–A6.", "Review 2 Report & Code", C_BLUE),
        ("Phase 3 (W9–W11)", "Validation & Review 3", "5-Fold CV, Cross-domain testing (25k+ imgs), Grad-CAM XAI.", "Review 3 Data & Weights", C_AMBER),
        ("Phase 4 (W12–W14)", "Deployment & Docs", "RTSP camera pipeline, Docker packaging, Scientific paper polish.", "RTSP Demo & Paper Draft", C_GREEN),
        ("Phase 5 (W15)", "Final Defense", "Thesis Book submission, final defense presentation rehearsal.", "Thesis Book & Defense", C_PURPLE),
    ]

    # Draw 5 Timeline cards
    w_card = 2.72
    gap = 0.2
    for i, (p_title, p_sub, p_desc, p_deliv, p_col) in enumerate(phases):
        x = 0.8 + i * (w_card + gap)
        draw_card(ax, x, 4.3, w_card, 3.1, border_color=p_col, lw=1.6)

        # Header tag
        tag = patches.FancyBboxPatch((x + 0.1, 6.85), w_card - 0.2, 0.4,
                                     boxstyle="round,pad=0.03,rounding_size=0.08",
                                     facecolor=p_col, edgecolor=p_col)
        ax.add_patch(tag)
        ax.text(x + w_card / 2, 7.05, p_title, fontsize=10.5, fontweight='bold', color='#FFFFFF', ha='center', va='center')

        # Subtitle
        ax.text(x + 0.15, 6.5, p_sub, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        
        # Description (wrapped tightly)
        desc_wrapped = wrap_str(p_desc, 27)
        ax.text(x + 0.15, 6.25, desc_wrapped, fontsize=9.2, fontweight='normal', color=C_BODY, va='top')

        # Deliverable badge
        d_box = patches.FancyBboxPatch((x + 0.1, 4.45), w_card - 0.2, 0.48,
                                       boxstyle="round,pad=0.03,rounding_size=0.08",
                                       facecolor='#F8FAFC', edgecolor=p_col, linewidth=1)
        ax.add_patch(d_box)
        deliv_wrapped = wrap_str("Target: " + p_deliv, 26)
        ax.text(x + w_card / 2, 4.69, deliv_wrapped, fontsize=8.2, fontweight='bold', color=p_col, ha='center', va='center')

    # Card A: Agile Execution (Bottom-Left)
    draw_card(ax, 0.8, 1.35, 7.0, 2.75)
    draw_badge(ax, 1.05, 3.55, 0.6, 0.35, "01", C_BLUE)
    ax.text(1.8, 3.72, "Agile Sprint Execution & Contingency Buffers", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_a = [
        ("• 2-Week Sprint Cadence:", "Rapid iterative cycles with bi-weekly progress synchronization with Academic Advisor."),
        ("• 1-Week Risk Buffer per Phase:", "Dedicated contingency time for cloud GPU re-training and parameter exploration."),
        ("• Review Gating Criteria:", "Strict quality gates and benchmark hurdles before advancing to the next Review."),
        ("• Transparent Version Control:", "Daily code, configuration, and model checkpoint commits synchronized to GitHub.")
    ]
    y_pos = 3.15
    for title, desc in lines_a:
        ax.text(1.1, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(1.1, y_pos - 0.23, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.48

    # Card B: Role Allocation (Bottom-Right)
    draw_card(ax, 8.2, 1.35, 7.0, 2.75)
    draw_badge(ax, 8.45, 3.55, 0.6, 0.35, "02", C_ORANGE)
    ax.text(9.2, 3.72, "Strategic Team Role Allocation", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_b = [
        ("• Nguyễn Hàn Như (Lead · SE183644):", "Overall network architecture, custom PyTorch modules, and Scientific paper writing."),
        ("• Nguyễn Văn Thành (SE180387):", "Data engineering, label harmonization (C*), SHWD cleaning, 5-Fold Stratified CV."),
        ("• Trần Phạm Tuấn Dũng (SE183674):", "Real-time RTSP video streaming pipeline, hardware acceleration, and documentation."),
        ("• High-Efficiency Collaboration:", "Clear responsibility boundaries with unified shared Git repo and issue tracking.")
    ]
    y_pos = 3.15
    for title, desc in lines_b:
        ax.text(8.5, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(8.5, y_pos - 0.23, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.48

    add_banner(ax, "Structured 15-week Hybrid Agile roadmap strictly aligned with 3 Review milestones and final Capstone Defense.")
    fig.savefig(os.path.join(out_dir, 'Slide_09_Project_Plan_Gantt.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Rendered: Slide_09_Project_Plan_Gantt.png")

# =====================================================================
# SLIDE 10: RESEARCH PAPER PLAN
# =====================================================================
def render_slide_10():
    fig, ax = setup_base_slide(
        "CRITERION 6 · SCIENTIFIC RESEARCH & PUBLICATION PLAN", "10 / 18",
        "Scientific Research Manuscript: Publication-Ready Paper Plan"
    )

    # Hero Card: Target Academic Standards & Publication Readiness (Top)
    draw_card(ax, 0.8, 4.75, 14.4, 2.65, border_color=C_BLUE, lw=1.8)
    draw_badge(ax, 1.1, 6.9, 4.3, 0.35, "SCIENTIFIC MANUSCRIPT & ACADEMIC STANDARDS", C_BLUE, fontsize=9.2)

    ax.text(1.1, 6.45, "Publication-Ready Master Manuscript (Targeting Peer-Reviewed Venues)", fontsize=15, fontweight='bold', color=C_TITLE, va='center')
    ax.text(1.1, 6.05, "Formatted in Standard IEEE Double-Column · Ready for Academic Committee Review & Journal Submission", fontsize=11, fontweight='bold', color=C_ORANGE, va='center')
    paper_title = 'Paper Title: "Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance"'
    ax.text(1.1, 5.65, wrap_str(paper_title, 110), fontsize=10.8, fontstyle='italic', color=C_BODY, va='center')
    ax.text(1.1, 5.15, "• Manuscript Status: Complete 10-page master technical paper fully drafted in IEEE format, ready for Academic Council evaluation and journal submission.", fontsize=10.5, fontweight='normal', color=C_TITLE, va='center')

    # Card 1: 4 Scientific Contributions (Bottom-Left)
    draw_card(ax, 0.8, 1.35, 7.0, 3.2)
    draw_badge(ax, 1.05, 4.05, 0.6, 0.35, "01", C_ORANGE)
    ax.text(1.8, 4.22, "4 Key Scientific Contributions", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_c = [
        ("1. Novel Architecture:", "Rep-YOLO11s harmonizing high detection recall, minimal latency, and low memory footprint."),
        ("2. Mathematical Proof:", "Exact closed-form algebraic fusion of 3-branch RepConv with zero precision degradation (error < 10^-5)."),
        ("3. Spatial Ground Invariance:", "CoordConv vertical gradient proves effective in silencing ground false alarms from floor objects."),
        ("4. Cross-Domain Taxonomy Autopsy:", "Comprehensive empirical evaluation across >33,000 images resolving the IoU Collapse phenomenon.")
    ]
    y_pos = 3.65
    for title, desc in lines_c:
        ax.text(1.1, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(1.1, y_pos - 0.24, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.54

    # Card 2: Publication Roadmap (Bottom-Right)
    draw_card(ax, 8.2, 1.35, 7.0, 3.2)
    draw_badge(ax, 8.45, 4.05, 0.6, 0.35, "02", C_GREEN)
    ax.text(9.2, 4.22, "Publication Roadmap & Scientific Integrity", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    lines_d = [
        ("• Phase 1–3 (Empirical Evidence):", "Solidify 5-Fold Stratified CV, multi-dataset zero-shot tests, and Grad-CAM saliency maps."),
        ("• Phase 4 (Writing & Academic Review):", "Rigorous academic polishing, IEEE LaTeX double-column typesetting, and mentor co-author review."),
        ("• Phase 5 & Defense:", "Present manuscript to Academic Council; finalize submission to reputable journals/conferences under advisor mentorship."),
        ("• Reproducibility Standards:", "Full open-source release with deterministic seed control and complete training scripts.")
    ]
    y_pos = 3.65
    for title, desc in lines_d:
        ax.text(8.5, y_pos, title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(8.5, y_pos - 0.24, wrap_str(desc, 68), fontsize=10, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.54

    add_banner(ax, "Publication-ready scientific manuscript prepared to international IEEE standards, fully verifiable by the Academic Council and ready for peer-reviewed venues.")
    fig.savefig(os.path.join(out_dir, 'Slide_10_Research_Paper_Plan.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Rendered: Slide_10_Research_Paper_Plan.png")

# =====================================================================
# SLIDE 11: RISKS & MITIGATION
# =====================================================================
def render_slide_11():
    fig, ax = setup_base_slide(
        "CRITERION 7 · RISK ASSESSMENT & MITIGATION MATRIX", "11 / 18",
        "Proactive Risk Analysis & Engineering Mitigation Matrix"
    )

    risks = [
        ("1. Extreme Class Imbalance (1:12)", "HIGH RISK", C_ORANGE,
         "111,514 body instances overpower 9,044 helmet instances, causing helmet gradient starvation.",
         "Task-Aligned Assigner (TAL) top-10 anchor alignment + Focal BCE Loss with dynamic alpha-balance to prioritize helmet learning."),
        ("2. Ground False Positives", "HIGH RISK", C_AMBER,
         "Yellow buckets, traffic cones, and caution signs on the floor trigger high false alarm rates (>28%).",
         "CoordConv Stem injects vertical coordinate Cy; provides anatomical prior suppressing ground logits (Cy > 0.4) to <0.02."),
        ("3. Distant Tiny Helmets (<20px)", "HIGH RISK", C_PURPLE,
         "Small targets under high-angle CCTV blur and vanish through standard convolutional downsampling.",
         "BiFormer Bi-Level Routing Attention filters out 93.75% of background and preserves small target features with O(HW) cost."),
        ("4. CIoU Gradient Vanishing", "MED RISK", C_BLUE,
         "When aspect ratio w/h matches ground truth, CIoU penalty gradient vanishes, stalling boundary regression.",
         "Focal-EIoU Loss decouples width and height errors independently; guarantees dL/dw != 0 whenever width is not aligned."),
        ("5. Video Sequence Data Leakage", "HIGH RISK", C_RED,
         "Consecutive frames from identical camera scenes appearing in both train and test artificially inflate accuracy.",
         "Video sequence hashing groups continuous frames strictly into the same partition, guaranteeing 100% leak-free evaluation."),
        ("6. GPU Memory & Compute Limits", "MED RISK", C_SLATE,
         "Training high-resolution multi-module models risks CUDA Out-Of-Memory (OOM) and long training times.",
         "Dual Kaggle Tesla T4 GPUs (32 GB VRAM) using PyTorch DDP; FP16 mixed precision halves memory footprint without accuracy loss.")
    ]

    rw = 4.6
    rh = 2.85
    rx_gap = 0.3
    ry_gap = 0.25

    for idx, (rtitle, rlevel, rcol, rdesc, rmit) in enumerate(risks):
        row = idx // 3
        col = idx % 3
        c_x = 0.8 + col * (rw + rx_gap)
        c_y = 4.45 - row * (rh + ry_gap)

        draw_card(ax, c_x, c_y, rw, rh)
        draw_badge(ax, c_x + 0.15, c_y + rh - 0.42, 1.25, 0.3, rlevel, rcol)
        ax.text(c_x + 1.5, c_y + rh - 0.27, rtitle, fontsize=11, fontweight='bold', color=C_TITLE, va='center')

        # Risk description
        ax.text(c_x + 0.18, c_y + rh - 0.68, "[Risk Trigger]:", fontsize=9.5, fontweight='bold', color='#991B1B', va='center')
        ax.text(c_x + 0.18, c_y + rh - 0.90, wrap_str(rdesc, 44), fontsize=9.0, fontweight='normal', color=C_BODY, va='top')

        # Mitigation description
        ax.text(c_x + 0.18, c_y + 1.28, "[Engineering Mitigation]:", fontsize=9.5, fontweight='bold', color='#166534', va='center')
        ax.text(c_x + 0.18, c_y + 1.06, wrap_str(rmit, 44), fontsize=9.0, fontweight='normal', color=C_TITLE, va='top')

    add_banner(ax, "All 6 core engineering risks are proactively resolved through mathematically sound and empirically verified architectural solutions.")
    fig.savefig(os.path.join(out_dir, 'Slide_11_Risk_Assessment_Mitigation.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Rendered: Slide_11_Risk_Assessment_Mitigation.png")

# =====================================================================
# SLIDE 12: EXPECTED OUTCOMES & CONCLUSION
# =====================================================================
def render_slide_12():
    fig, ax = setup_base_slide(
        "CRITERION 8 · EXPECTED OUTCOMES & CONCLUSION", "12 / 18",
        "Final Deliverables & 8-Point Project Synthesis"
    )

    # Left Column: 5 Deliverables
    draw_card(ax, 0.8, 1.35, 7.0, 6.05)
    draw_badge(ax, 1.05, 6.9, 0.6, 0.35, "01", C_BLUE)
    ax.text(1.8, 7.07, "5 Tangible Project Deliverables", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

    deliv_items = [
        ("1. High-Accuracy AI Weights (.pt)", "Rep-YOLO11s trained weights achieving >96% 5-Fold mAP50 with optimized small helmet recall."),
        ("2. Standardized Dataset Pipeline", "Cleaned SHWD (7,581 images) with zero label corruption, plus unified C* taxonomy across >33,000 industrial images."),
        ("3. Open-Source Code Repository", "Clean GitHub repository with PyTorch modules, switch_to_deploy() scripts, and multi-GPU DDP training configs."),
        ("4. Real-Time RTSP Software Demo", "Surveillance desktop application running on live CCTV streams with automated violation alerts and bounding boxes."),
        ("5. Scientific Research Manuscript", "Publication-ready master paper in IEEE format, ready for Academic Council evaluation and journal submission.")
    ]
    y_pos = 6.45
    for d_title, d_desc in deliv_items:
        ax.text(1.1, y_pos, d_title, fontsize=11, fontweight='bold', color=C_BLUE, va='center')
        ax.text(1.1, y_pos - 0.30, wrap_str(d_desc, 68), fontsize=9.6, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.98

    # Right Column: 8-Point Synthesis (Conclusion)
    draw_card(ax, 8.2, 1.35, 7.0, 6.05)
    draw_badge(ax, 8.45, 6.9, 0.6, 0.35, "02", C_ORANGE)
    ax.text(9.2, 7.07, "8-Point Project Synthesis (Conclusion)", fontsize=14, fontweight='bold', color=C_TITLE, va='center')

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
    y_pos = 6.45
    for p_title, p_desc in points_syn:
        ax.text(8.5, y_pos, p_title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')
        ax.text(8.5, y_pos - 0.24, wrap_str(p_desc, 68), fontsize=9.6, fontweight='normal', color=C_BODY, va='center')
        y_pos -= 0.62

    add_banner(ax, "A comprehensive, publication-grade AI capstone delivering high-accuracy helmet surveillance, robust software, and verifiable scientific contributions.")
    fig.savefig(os.path.join(out_dir, 'Slide_12_Expected_Outcomes_Conclusion.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Rendered: Slide_12_Expected_Outcomes_Conclusion.png")

if __name__ == '__main__':
    render_slide_8()
    render_slide_9()
    render_slide_10()
    render_slide_11()
    render_slide_12()
    print("All slides successfully rendered at 300 DPI!")
