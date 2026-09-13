"""
Generate Pixel-Perfect IEEE Q1 Publication-Quality Model Architecture Diagram for Rep-YOLO11s.
Fixes all text and arrow overlapping issues with generous spacing, crisp alignment,
and verified empirical benchmark numbers (no fake INT8/KD).
Outputs both 300 DPI PNG and Vector PDF.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_rep_yolo11_architecture_clean():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    # Extra canvas width & height for clean spacing
    fig, ax = plt.subplots(figsize=(18.0, 10.0), dpi=300)
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 10.0)
    ax.axis('off')

    # Color Palette - Professional IEEE Journal Theme
    c_bg = '#FFFFFF'
    c_dark_banner = '#0F172A'
    c_card_bg = '#F8FAFC'
    c_primary = '#1E3A8A'      # Deep Navy Blue
    c_accent = '#0284C7'       # Sky Blue
    c_neck = '#EFF6FF'         # Blue tint
    c_head = '#FEF2F2'         # Red tint
    c_innov = '#ECFDF5'        # Emerald Mint (Novelties)
    c_innov_border = '#10B981' # Emerald Green
    c_reparam = '#FFFBEB'      # Amber tint
    c_reparam_border = '#F59E0B'
    c_border = '#CBD5E1'       # Slate 300
    c_text_dark = '#0F172A'

    fig.patch.set_facecolor(c_bg)

    # 1. Main Title Banner
    title_box = patches.FancyBboxPatch(
        (0.40, 9.05), 17.20, 0.75,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=c_dark_banner, edgecolor='none'
    )
    ax.add_patch(title_box)
    ax.text(9.00, 9.50, "PROPOSED Rep-YOLO11s NEURAL NETWORK ARCHITECTURE & RE-PARAMETERIZATION",
            ha='center', va='center', color='#FFFFFF', fontsize=12.2, fontweight='bold')
    ax.text(9.00, 9.20, "CoordConv Spatial Priors • Multi-Scale CSPDarknet Backbone • BiFormer Dynamic Routing • Structural RepConv • Focal-EIoU Head",
            ha='center', va='center', color='#94A3B8', fontsize=8.5, fontweight='normal')

    # ==========================================
    # SECTION 1: INPUT & COORDCONV ENCODING (0.40 -> 3.10, w=2.70)
    # ==========================================
    sec1_box = patches.FancyBboxPatch((0.40, 4.45), 2.70, 4.40, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=c_card_bg, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(sec1_box)
    ax.text(1.75, 8.60, "1. INPUT & COORDCONV", ha='center', va='center', color=c_primary, fontsize=8.8, fontweight='bold')

    # Input RGB Image Box
    b_rgb = patches.FancyBboxPatch((0.55, 7.50), 2.40, 0.80, boxstyle="round,pad=0.04,rounding_size=0.08",
                                  facecolor='#FFFFFF', edgecolor='#94A3B8', linewidth=0.8)
    ax.add_patch(b_rgb)
    ax.text(1.75, 8.00, "Input Construction Image", ha='center', va='center', color=c_text_dark, fontsize=7.8, fontweight='bold')
    ax.text(1.75, 7.70, "Shape: (B, 3, 640, 640)", ha='center', va='center', color='#64748B', fontsize=7.2)

    # CoordConv Module Box (Innovation)
    b_cc = patches.FancyBboxPatch((0.55, 5.85), 2.40, 1.30, boxstyle="round,pad=0.04,rounding_size=0.08",
                                 facecolor=c_innov, edgecolor=c_innov_border, linewidth=1.2)
    ax.add_patch(b_cc)
    ax.text(1.75, 6.85, "[+] CoordConv Generator", ha='center', va='center', color='#065F46', fontsize=8.0, fontweight='bold')
    ax.text(1.75, 6.50, "Injects Normalized Grid:", ha='center', va='center', color='#047857', fontsize=7.2)
    ax.text(1.75, 6.22, "h_i in [-1, 1], w_j in [-1, 1]", ha='center', va='center', color='#065F46', fontsize=7.0, fontweight='bold')
    ax.text(1.75, 5.98, "-> Breaks Translation Invariance", ha='center', va='center', color='#047857', fontsize=6.8, fontstyle='italic')

    # Output 5-Channel Tensor
    b_cc_out = patches.FancyBboxPatch((0.55, 4.65), 2.40, 0.80, boxstyle="round,pad=0.04,rounding_size=0.06",
                                      facecolor='#1E293B', edgecolor='none')
    ax.add_patch(b_cc_out)
    ax.text(1.75, 5.15, "Spatial Augmented Tensor", ha='center', va='center', color='#38BDF8', fontsize=7.5, fontweight='bold')
    ax.text(1.75, 4.88, "Shape: (B, 5, 640, 640)", ha='center', va='center', color='#E0F2FE', fontsize=7.0)

    # Connecting Arrows inside Sec 1 (pointing DOWN from Input to CoordConv to Output)
    ax.annotate('', xy=(1.75, 7.15), xytext=(1.75, 7.50), arrowprops=dict(arrowstyle="-|>", color=c_primary, lw=1.5))
    ax.annotate('', xy=(1.75, 5.45), xytext=(1.75, 5.85), arrowprops=dict(arrowstyle="-|>", color=c_primary, lw=1.5))

    # Stepped architectural data flow from Spatial Augmented Tensor (Sec 1) into Stem Conv 3x3 / P1 (Sec 2)
    x_mid = 3.20
    y_src = 5.05
    y_dst = 8.00
    ax.plot([2.95, x_mid, x_mid, 3.45], [y_src, y_src, y_dst, y_dst], color=c_primary, lw=1.8, solid_capstyle='round')
    ax.annotate('', xy=(3.45, y_dst), xytext=(3.30, y_dst),
                arrowprops=dict(arrowstyle="-|>", color=c_primary, lw=1.8, mutation_scale=12))

    # ==========================================
    # SECTION 2: BACKBONE (CSPDarknet + SPPF + C2PSA) (3.30 -> 6.40, w=3.10)
    # ==========================================
    sec2_box = patches.FancyBboxPatch((3.30, 4.45), 3.10, 4.40, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=c_card_bg, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(sec2_box)
    ax.text(4.85, 8.60, "2. MULTI-SCALE BACKBONE", ha='center', va='center', color=c_primary, fontsize=8.8, fontweight='bold')

    bb_stages = [
        ("Stem Conv 3x3 / P1", "Shape: (B, 32, 320, 320)", 8.00, False),
        ("C3k2 Block 1 / P2", "Shape: (B, 64, 160, 160) - High-Res", 7.25, False),
        ("C3k2 Block 2 / P3", "Shape: (B, 128, 80, 80) -> To Neck", 6.50, True),
        ("C3k2 Block 3 / P4", "Shape: (B, 256, 40, 40) -> To Neck", 5.75, True),
        ("SPPF + C2PSA / P5", "Shape: (B, 512, 20, 20) -> To Neck", 5.00, True),
    ]
    for name, shape, y_pos, has_out in bb_stages:
        box = patches.FancyBboxPatch((3.45, y_pos - 0.20), 2.80, 0.58, boxstyle="round,pad=0.02,rounding_size=0.06",
                                     facecolor='#EFF6FF' if has_out else '#FFFFFF',
                                     edgecolor='#3B82F6' if has_out else '#CBD5E1', linewidth=0.8)
        ax.add_patch(box)
        ax.text(3.55, y_pos + 0.12, name, ha='left', va='center', color=c_text_dark, fontsize=7.4, fontweight='bold')
        ax.text(3.55, y_pos - 0.08, shape, ha='left', va='center', color='#2563EB' if has_out else '#64748B', fontsize=6.6)

    # Arrows between Backbone stages
    for y_ar in [7.62, 6.87, 6.12, 5.37]:
        ax.annotate('', xy=(4.85, y_ar - 0.15), xytext=(4.85, y_ar + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))

    # ==========================================
    # SECTION 3: NECK (BiFormer Attention + RepConv) (6.60 -> 11.80, w=5.20)
    # ==========================================
    sec3_box = patches.FancyBboxPatch((6.60, 4.45), 5.20, 4.40, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=c_neck, edgecolor='#2563EB', linewidth=1.5)
    ax.add_patch(sec3_box)
    ax.text(9.20, 8.60, "3. NECK: BIFORMER ATTENTION & REPCONV FUSION", ha='center', va='center', color='#1D4ED8', fontsize=8.8, fontweight='bold')

    # BiFormer Attention Sub-card (Clean bounds: 6.75 -> 11.65, w=4.90)
    bif_box = patches.FancyBboxPatch((6.75, 6.70), 4.90, 1.70, boxstyle="round,pad=0.04,rounding_size=0.08",
                                     facecolor=c_innov, edgecolor=c_innov_border, linewidth=1.2)
    ax.add_patch(bif_box)
    ax.text(9.20, 8.15, "[+] BiFormer: Bi-Level Routing Dynamic Attention", ha='center', va='center', color='#065F46', fontsize=8.0, fontweight='bold')
    ax.text(6.90, 7.80, "• Step 1: Regional Feature Partition & Routing Affinity Map", ha='left', va='center', color='#047857', fontsize=6.8)
    ax.text(6.90, 7.50, "• Step 2: Dynamic Top-k Routing (Prunes Background Context)", ha='left', va='center', color='#065F46', fontsize=6.8, fontweight='bold')
    ax.text(6.90, 7.20, "• Step 3: Fine Token-to-Token Attention on Salient Contours", ha='left', va='center', color='#047857', fontsize=6.8)
    ax.text(6.90, 6.90, "-> Dynamic Sparse Routing: O(N) vs Dense O(N^2) Complexity", ha='left', va='center', color='#B45309', fontsize=6.7, fontstyle='italic')

    # Structural RepConv Sub-card (Clean bounds: 6.75 -> 11.65, w=4.90)
    rep_box = patches.FancyBboxPatch((6.75, 4.65), 4.90, 1.85, boxstyle="round,pad=0.04,rounding_size=0.08",
                                     facecolor=c_reparam, edgecolor=c_reparam_border, linewidth=1.2)
    ax.add_patch(rep_box)
    ax.text(9.20, 6.25, "[+] Structural RepConv Feature Fusion Block", ha='center', va='center', color='#92400E', fontsize=8.0, fontweight='bold')
    ax.text(6.90, 5.95, "• Train: Multi-branch (Conv 3x3 + Conv 1x1 + Identity Residual)", ha='left', va='center', color='#78350F', fontsize=6.8)
    ax.text(6.90, 5.68, "• Deploy: Algebraic Re-parameterization via switch_to_deploy()", ha='left', va='center', color='#B45309', fontsize=6.8, fontweight='bold')
    ax.text(6.90, 5.41, "  W_fused = W_3x3 + pad(W_1x1) + diag(I) -> Plain 3x3 Conv Layer", ha='left', va='center', color='#92400E', fontsize=6.9, fontweight='bold')
    ax.text(6.90, 5.14, "-> Pure GPU Latency: 7.12 ms -> 2.92 ms (2.44x Speedup on Tesla T4)", ha='left', va='center', color='#065F46', fontsize=6.8, fontweight='bold')
    ax.text(6.90, 4.87, "• Multi-Scale Neck Features: P3 (80x80), P4 (40x40), P5 (20x20)", ha='left', va='center', color='#1E3A8A', fontsize=6.7)

    # Clean non-overlapping routing arrows from Backbone to Neck
    ax.annotate('', xy=(6.60, 6.50), xytext=(6.40, 6.50), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8, mutation_scale=12))
    ax.annotate('', xy=(6.60, 5.75), xytext=(6.40, 5.75), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8, mutation_scale=12))
    ax.annotate('', xy=(6.60, 5.00), xytext=(6.40, 5.00), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8, mutation_scale=12))

    # ==========================================
    # SECTION 4: DECOUPLED HEAD & SUPERVISION LOSSES (12.00 -> 17.60, w=5.60)
    # ==========================================
    sec4_box = patches.FancyBboxPatch((12.00, 4.45), 5.60, 4.40, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=c_head, edgecolor='#EF4444', linewidth=1.5)
    ax.add_patch(sec4_box)
    ax.text(14.80, 8.60, "4. DECOUPLED HEAD & SUPERVISION LOSSES", ha='center', va='center', color='#B91C1C', fontsize=8.8, fontweight='bold')

    # Multi-scale decoupled heads (12.15 -> 17.45, w=5.30)
    h_box1 = patches.FancyBboxPatch((12.15, 7.35), 5.30, 1.05, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor='#FFFFFF', edgecolor='#FCA5A5', linewidth=0.8)
    ax.add_patch(h_box1)
    ax.text(12.30, 8.12, "Decoupled Detection Heads (Multi-Scale P3, P4, P5):", ha='left', va='center', color=c_text_dark, fontsize=7.4, fontweight='bold')
    ax.text(12.30, 7.82, "• Classification Head (cv3): 2 Classes ('hat', 'person') via BCE Loss", ha='left', va='center', color='#475569', fontsize=6.8)
    ax.text(12.30, 7.55, "• Regression Head (cv2): 4 Bbox Coordinates (x_c, y_c, w, h) & DFL", ha='left', va='center', color='#475569', fontsize=6.8)

    # Focal EIoU Box Regression (12.15 -> 17.45, w=5.30)
    h_box2 = patches.FancyBboxPatch((12.15, 5.95), 5.30, 1.25, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor=c_innov, edgecolor=c_innov_border, linewidth=1.0)
    ax.add_patch(h_box2)
    ax.text(12.30, 6.93, "[+] Focal-EIoU Loss (Bounding Box Refinement):", ha='left', va='center', color='#065F46', fontsize=7.4, fontweight='bold')
    ax.text(12.30, 6.63, "• L_EIoU = L_IoU + L_dis + L_asp (Decoupled Dimensions Penalty)", ha='left', va='center', color='#047857', fontsize=6.7)
    ax.text(12.30, 6.36, "• L_Focal-EIoU = (IoU)^gamma * L_EIoU (gamma=0.5, Hard-Sample Mining)", ha='left', va='center', color='#065F46', fontsize=6.7, fontweight='bold')
    ax.text(12.30, 6.10, "-> Resolves severe 1:12 class imbalance & extreme crouching occlusion", ha='left', va='center', color='#047857', fontsize=6.6, fontstyle='italic')

    # Multi-Task Loss Supervision & Task-Aligned Assigner (12.15 -> 17.45, w=5.30)
    h_box3 = patches.FancyBboxPatch((12.15, 4.65), 5.30, 1.15, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor='#FFF7ED', edgecolor='#FB923C', linewidth=1.0)
    ax.add_patch(h_box3)
    ax.text(12.30, 5.53, "[+] Multi-Task Training Objective & Task Alignment:", ha='left', va='center', color='#C2410C', fontsize=7.4, fontweight='bold')
    ax.text(12.30, 5.25, "• L_total = lambda_1 * L_BCE + lambda_2 * L_FEIoU + lambda_3 * L_DFL", ha='left', va='center', color='#9A3412', fontsize=6.7, fontweight='bold')
    ax.text(12.30, 4.98, "• Task-Aligned Assigner (TAL): Dynamically balances cls & reg quality", ha='left', va='center', color='#7C2D12', fontsize=6.7)
    ax.text(12.30, 4.75, "-> End-to-end multi-task convergence without deployment overhead", ha='left', va='center', color='#7C2D12', fontsize=6.5)

    # Neck Internal Flow: BiFormer Dynamic Routing into RepConv Multi-Scale Fusion
    ax.annotate('', xy=(9.20, 6.50), xytext=(9.20, 6.70), arrowprops=dict(arrowstyle="-|>", color='#1D4ED8', lw=1.5))

    # Arrow from Neck to Decoupled Detection Heads (cv2 & cv3)
    ax.annotate('', xy=(12.00, 7.875), xytext=(11.80, 7.875), arrowprops=dict(arrowstyle="-|>", color='#DC2626', lw=2.0, mutation_scale=12))

    # Head Internal Supervisory Flow: Heads -> Focal-EIoU Regression -> Multi-Task Objective & TAL
    ax.annotate('', xy=(14.80, 7.20), xytext=(14.80, 7.35), arrowprops=dict(arrowstyle="-|>", color='#DC2626', lw=1.3))
    ax.annotate('', xy=(14.80, 5.80), xytext=(14.80, 5.95), arrowprops=dict(arrowstyle="-|>", color='#DC2626', lw=1.3))

    # ==========================================
    # BOTTOM SECTION: RE-PARAM MECHANISM & EMPIRICAL TENSORRT FP16 BENCHMARK
    # ==========================================
    bot_box = patches.FancyBboxPatch((0.40, 0.85), 17.20, 3.40, boxstyle="round,pad=0.06,rounding_size=0.12",
                                     facecolor='#F8FAFC', edgecolor='#64748B', linewidth=1.2)
    ax.add_patch(bot_box)
    ax.text(9.00, 4.00, "5. STRUCTURAL RE-PARAMETERIZATION LIFECYCLE & TENSORRT FP16 ACCELERATION BENCHMARK",
            ha='center', va='center', color=c_dark_banner, fontsize=9.5, fontweight='bold')

    # Left: Training Multi-Branch Diagram (0.60 -> 5.40, w=4.80)
    train_box = patches.FancyBboxPatch((0.60, 1.05), 4.80, 2.65, boxstyle="round,pad=0.04,rounding_size=0.08",
                                       facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=0.8)
    ax.add_patch(train_box)
    ax.text(3.00, 3.45, "Training Phase (Multi-Branch Topology)", ha='center', va='center', color='#1E3A8A', fontsize=8.0, fontweight='bold')
    
    # Input X badge
    in_b = patches.FancyBboxPatch((0.75, 2.15), 0.70, 0.40, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#F1F5F9', edgecolor='#94A3B8')
    ax.add_patch(in_b)
    ax.text(1.10, 2.35, "Input X", ha='center', va='center', color=c_text_dark, fontsize=7.2, fontweight='bold')

    # 3 Parallel Branches
    b1 = patches.FancyBboxPatch((1.75, 2.70), 1.70, 0.38, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#DBEAFE', edgecolor='#2563EB')
    b2 = patches.FancyBboxPatch((1.75, 2.15), 1.70, 0.38, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#DCFCE7', edgecolor='#16A34A')
    b3 = patches.FancyBboxPatch((1.75, 1.60), 1.70, 0.38, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#FEF3C7', edgecolor='#D97706')
    ax.add_patch(b1); ax.add_patch(b2); ax.add_patch(b3)
    ax.text(2.60, 2.89, "Conv 3x3 + BN", ha='center', va='center', color='#1E40AF', fontsize=7.2, fontweight='bold')
    ax.text(2.60, 2.34, "Conv 1x1 + BN", ha='center', va='center', color='#15803D', fontsize=7.2, fontweight='bold')
    ax.text(2.60, 1.79, "Identity + BN", ha='center', va='center', color='#B45309', fontsize=7.2, fontweight='bold')
    
    # Branch split arrows
    ax.annotate('', xy=(1.75, 2.89), xytext=(1.45, 2.35), arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))
    ax.annotate('', xy=(1.75, 2.35), xytext=(1.45, 2.35), arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))
    ax.annotate('', xy=(1.75, 1.79), xytext=(1.45, 2.35), arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))

    # Summation node
    circle_sum = patches.Circle((3.95, 2.35), 0.22, facecolor='#1E293B', edgecolor='none')
    ax.add_patch(circle_sum)
    ax.text(3.95, 2.35, "+", ha='center', va='center', color='#FFFFFF', fontsize=9.0, fontweight='bold')

    # Branch join arrows
    ax.annotate('', xy=(3.73, 2.45), xytext=(3.45, 2.89), arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))
    ax.annotate('', xy=(3.73, 2.35), xytext=(3.45, 2.35), arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))
    ax.annotate('', xy=(3.73, 2.25), xytext=(3.45, 1.79), arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2))

    # Out Y badge
    out_b = patches.FancyBboxPatch((4.45, 2.15), 0.70, 0.40, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#F1F5F9', edgecolor='#94A3B8')
    ax.add_patch(out_b)
    ax.text(4.80, 2.35, "Out Y", ha='center', va='center', color=c_text_dark, fontsize=7.2, fontweight='bold')
    ax.annotate('', xy=(4.45, 2.35), xytext=(4.17, 2.35), arrowprops=dict(arrowstyle="-|>", color='#1E293B', lw=1.2))
    
    ax.text(3.00, 1.25, "Rich Gradient Representation during Training", ha='center', va='center', color='#64748B', fontsize=6.8, fontstyle='italic')

    # Middle Transition: Clean Algebraic Re-parameterization (5.60 -> 7.55, w=1.95)
    trans_box = patches.FancyBboxPatch((5.60, 1.65), 1.95, 1.45, boxstyle="round,pad=0.02,rounding_size=0.06",
                                       facecolor='#FFFBEB', edgecolor='#F59E0B', linewidth=1.0)
    ax.add_patch(trans_box)
    ax.text(6.57, 2.70, "ALGEBRAIC FUSION", ha='center', va='center', color='#B45309', fontsize=7.2, fontweight='bold')
    ax.text(6.57, 2.42, "switch_to_deploy()", ha='center', va='center', color='#92400E', fontsize=7.0, fontfamily='monospace', fontweight='bold')
    ax.text(6.57, 2.14, "Merge Conv & BN", ha='center', va='center', color='#78350F', fontsize=6.6)
    ax.text(6.57, 1.90, "Zero-Pad 1x1 & Id", ha='center', va='center', color='#78350F', fontsize=6.6)

    ax.annotate('', xy=(5.60, 2.35), xytext=(5.15, 2.35), arrowprops=dict(arrowstyle="-|>", color='#D97706', lw=1.8, mutation_scale=12))
    ax.annotate('', xy=(7.75, 2.35), xytext=(7.55, 2.35), arrowprops=dict(arrowstyle="-|>", color='#D97706', lw=1.8, mutation_scale=12))

    # Right: Inference Single-Path Conv 3x3 (7.75 -> 11.60, w=3.85)
    infer_box = patches.FancyBboxPatch((7.75, 1.05), 3.85, 2.65, boxstyle="round,pad=0.04,rounding_size=0.08",
                                       facecolor='#ECFDF5', edgecolor='#10B981', linewidth=1.0)
    ax.add_patch(infer_box)
    ax.text(9.67, 3.45, "Inference Phase (Single 3x3 Conv)", ha='center', va='center', color='#065F46', fontsize=8.0, fontweight='bold')
    
    in_infer = patches.FancyBboxPatch((7.90, 2.15), 0.65, 0.40, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#FFFFFF', edgecolor='#94A3B8')
    ax.add_patch(in_infer)
    ax.text(8.22, 2.35, "Input X", ha='center', va='center', color=c_text_dark, fontsize=7.0, fontweight='bold')

    single_conv = patches.FancyBboxPatch((8.80, 2.05), 1.75, 0.60, boxstyle="round,pad=0.02,rounding_size=0.06",
                                         facecolor='#059669', edgecolor='#047857')
    ax.add_patch(single_conv)
    ax.text(9.67, 2.45, "Fused Conv 3x3", ha='center', va='center', color='#FFFFFF', fontsize=7.5, fontweight='bold')
    ax.text(9.67, 2.22, "W_fused + b_fused", ha='center', va='center', color='#D1FAE5', fontsize=6.8)
    
    out_infer = patches.FancyBboxPatch((10.80, 2.15), 0.65, 0.40, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor='#FFFFFF', edgecolor='#94A3B8')
    ax.add_patch(out_infer)
    ax.text(11.12, 2.35, "Out Y", ha='center', va='center', color=c_text_dark, fontsize=7.0, fontweight='bold')

    ax.annotate('', xy=(8.80, 2.35), xytext=(8.55, 2.35), arrowprops=dict(arrowstyle="-|>", color='#059669', lw=1.2))
    ax.annotate('', xy=(10.80, 2.35), xytext=(10.55, 2.35), arrowprops=dict(arrowstyle="-|>", color='#059669', lw=1.2))
    ax.text(9.67, 1.25, "Zero Latency Overhead • Max Tensor Core Efficiency", ha='center', va='center', color='#047857', fontsize=6.8, fontweight='bold')

    # Far Right: Quantization & Speed Summary Card (11.80 -> 17.45, w=5.65)
    quant_box = patches.FancyBboxPatch((11.80, 1.05), 5.65, 2.65, boxstyle="round,pad=0.04,rounding_size=0.08",
                                       facecolor='#1E293B', edgecolor='none')
    ax.add_patch(quant_box)
    ax.text(14.62, 3.45, "TENSORRT FP16 BENCHMARK SUMMARY", ha='center', va='center', color='#38BDF8', fontsize=8.2, fontweight='bold')
    
    ax.text(12.00, 3.05, "• Tesla T4 GPU (FP16 Engine):", ha='left', va='center', color='#94A3B8', fontsize=7.3)
    ax.text(17.25, 3.05, "2.92 ms (342.5 FPS)", ha='right', va='center', color='#FFFFFF', fontsize=7.3, fontweight='bold')

    ax.text(12.00, 2.65, "• RTX 3050 Laptop (Edge FP16):", ha='left', va='center', color='#34D399', fontsize=7.3, fontweight='bold')
    ax.text(17.25, 2.65, "5.35 ms (187.1 FPS)", ha='right', va='center', color='#34D399', fontsize=7.5, fontweight='bold')

    ax.text(12.00, 2.25, "• RTX 3050 Native PyTorch:", ha='left', va='center', color='#E2E8F0', fontsize=7.3)
    ax.text(17.25, 2.25, "12.43 ms (80.4 FPS)", ha='right', va='center', color='#E2E8F0', fontsize=7.3, fontweight='bold')

    ax.text(12.00, 1.85, "• High-Res Batch-1 (960x960, T4):", ha='left', va='center', color='#E2E8F0', fontsize=7.3)
    ax.text(17.25, 1.85, "20.53 ms (48.7 FPS)", ha='right', va='center', color='#38BDF8', fontsize=7.3, fontweight='bold')

    ax.text(12.00, 1.45, "• In-Domain SHWD Benchmark:", ha='left', va='center', color='#FBBF24', fontsize=7.3)
    ax.text(17.25, 1.45, "94.83% mAP50 (5-Fold: 96.64%)", ha='right', va='center', color='#FBBF24', fontsize=7.3, fontweight='bold')

    ax.text(14.62, 1.15, "Real Synchronized CUDA Event Profiling (torch.cuda.synchronize)", ha='center', va='center', color='#A7F3D0', fontsize=6.8, fontstyle='italic')

    # Footer Caption
    ax.text(9.00, 0.48, "Figure 2: Complete Neural Network Architecture of Rep-YOLO11s displaying CoordConv coordinate generation, multi-scale CSPDarknet backbone,",
            ha='center', va='center', color='#475569', fontsize=7.5, style='italic')
    ax.text(9.00, 0.25, "BiFormer sparse attention neck, structural re-parameterization lifecycle (training multi-branch to single-path inference), and decoupled heads with TensorRT FP16 acceleration.",
            ha='center', va='center', color='#475569', fontsize=7.5, style='italic')

    plt.tight_layout()

    out_pdf = Path("paper_overleaf/figures/shwd_architecture_pipeline.pdf")
    out_png = Path("paper_overleaf/figures/shwd_architecture_pipeline.png")
    root_pdf = Path("shwd_architecture_pipeline_v2.pdf")
    root_png = Path("shwd_architecture_pipeline_v2.png")

    try:
        fig.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')
        fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"[Notice on overleaf folder]: {e}")

    try:
        fig.savefig(root_pdf, format='pdf', dpi=300, bbox_inches='tight')
        fig.savefig(root_png, format='png', dpi=300, bbox_inches='tight')
        fig.savefig("shwd_architecture_pipeline_ieee.png", format='png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"[Notice on root folder]: {e}")

    plt.close()

    print(f"✅ Clean IEEE Architecture Diagram successfully generated at:")
    print(f"   - {out_pdf.resolve()}")
    print(f"   - {out_png.resolve()}")
    print(f"   - {root_png.resolve()}")

if __name__ == "__main__":
    draw_rep_yolo11_architecture_clean()
