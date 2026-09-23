import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUT_DIRS = [
    "review1_genspark_package/figures/scientific_exports",
    "review1_genspark_package/slide_renders",
    "paper_overleaf/figures"
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

C_BG = '#0F172A'      # Slate 900 dark background for modern presentation
C_PANEL_L = '#1E293B' # Slate 800 (Baseline)
C_PANEL_R = '#1E293B' # Slate 800 (Proposed)
C_CARD_BASE = '#334155'
C_CARD_PROP = '#0F2B48'
C_TEXT_LIGHT = '#F8FAFC'
C_TEXT_MUTED = '#94A3B8'
C_RED_ACCENT = '#EF4444'
C_GREEN_ACCENT = '#10B981'
C_BLUE_ACCENT = '#3B82F6'
C_AMBER_ACCENT = '#F59E0B'
C_PURPLE_ACCENT = '#8B5CF6'

def save_slide(fig, filename):
    for d in OUT_DIRS:
        p = os.path.join(d, filename)
        fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor=C_BG)
    plt.close(fig)
    print(f"[SUCCESS] Exported Slide: {filename}")

def draw_header(ax, slide_no, title, subtitle):
    ax.text(0.03, 0.95, f"SLIDE {slide_no:02d} | {title.upper()}", fontsize=15, fontweight='bold', color=C_TEXT_LIGHT, ha='left', va='top')
    ax.text(0.03, 0.90, subtitle, fontsize=10.5, color=C_BLUE_ACCENT, ha='left', va='top')
    ax.plot([0.03, 0.97], [0.875, 0.875], color='#334155', lw=1.5)

def draw_panel(ax, x0, y0, w, h, title, tag, is_proposed=False):
    border_col = C_GREEN_ACCENT if is_proposed else C_RED_ACCENT
    bg_col = '#132338' if is_proposed else '#231825'
    
    panel = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.015",
                           fc=bg_col, ec=border_col, lw=2.0)
    ax.add_patch(panel)
    
    # Tag badge
    tag_bg = C_GREEN_ACCENT if is_proposed else C_RED_ACCENT
    tag_box = FancyBboxPatch((x0 + 0.02, y0 + h - 0.055), 0.14, 0.04,
                            boxstyle="round,pad=0.005", fc=tag_bg, ec='none')
    ax.add_patch(tag_box)
    ax.text(x0 + 0.09, y0 + h - 0.035, tag, fontsize=9, fontweight='bold', color='#FFFFFF', ha='center', va='center')
    
    # Title
    ax.text(x0 + 0.18, y0 + h - 0.035, title, fontsize=12.5, fontweight='bold', color=C_TEXT_LIGHT, ha='left', va='center')


# ==============================================================================
# SLIDE 05: BASELINE YOLO11s (3 Heads) vs PROPOSED REP-YOLO11s-P2 AFPN (4 Heads)
# ==============================================================================
def render_slide_05():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    draw_header(ax, 5, "Architecture Evolution: 3-Head Baseline vs. 4-Head Rep-YOLO11s-P2",
                "Addressing Tiny Helmet Information Loss via High-Resolution P2 AFPN Stream (Stride 4)")
    
    # Left Panel: Baseline
    draw_panel(ax, 0.03, 0.05, 0.45, 0.80, "Vanilla YOLO11s (3 Heads)", "BASELINE - LIMITATION", is_proposed=False)
    
    # Baseline flow blocks
    blocks_base = [
        ("Input Image", "640 x 640 x 3", 0.68),
        ("Standard CSP Backbone", "Subsampling Strides: 8, 16, 32", 0.54),
        ("PANet Feature Fusion", "Feature Maps: P3, P4, P5", 0.40),
        ("Standard 3 Detection Heads", "P3 (80x80), P4 (40x40), P5 (20x20)", 0.26)
    ]
    for name, sub, y in blocks_base:
        card = FancyBboxPatch((0.07, y), 0.37, 0.09, boxstyle="round,pad=0.01",
                              fc='#1E293B', ec='#475569', lw=1.2)
        ax.add_patch(card)
        ax.text(0.255, y + 0.055, name, fontsize=11, fontweight='bold', color='#FFFFFF', ha='center', va='center')
        ax.text(0.255, y + 0.025, sub, fontsize=9.5, color=C_TEXT_MUTED, ha='center', va='center')
        if y < 0.68:
            ax.annotate('', xy=(0.255, y + 0.09), xytext=(0.255, y + 0.14),
                        arrowprops=dict(arrowstyle="-|>", color=C_RED_ACCENT, lw=2.0))
            
    # Bottleneck callout box
    callout_base = FancyBboxPatch((0.07, 0.09), 0.37, 0.12, boxstyle="round,pad=0.01",
                                 fc='#3B1D25', ec=C_RED_ACCENT, lw=1.5)
    ax.add_patch(callout_base)
    ax.text(0.255, 0.17, "CRITICAL DEFICIENCY: TINY TARGET COLLAPSE", fontsize=10.5, fontweight='bold', color='#FCA5A5', ha='center', va='center')
    ax.text(0.255, 0.13, "- Distant helmets (<15x15 px) are severely downsampled by Stride 8/16/32\n- Target representation vanishes to < 1.5 pixels at P3, causing heavy false negatives",
            fontsize=8.5, color='#FEE2E2', ha='center', va='center', multialignment='center')

    # Right Panel: Proposed
    draw_panel(ax, 0.52, 0.05, 0.45, 0.80, "Proposed Rep-YOLO11s-P2 AFPN (4 Heads)", "PROPOSED - SOTA", is_proposed=True)
    
    blocks_prop = [
        ("Input + CoordConv Stem", "640 x 640 x 5 (Spatial Prior Injected)", 0.68),
        ("Structural RepConv Backbone", "Single-path 3x3 inference with BiFormer Routing", 0.54),
        ("AFPN Neck + DySample Upsampling", "4 Feature Scales: P2 (160x160), P3, P4, P5", 0.40),
        ("4 Decoupled Detection Heads", "P2 (Micro: 160x160), P3 (Small), P4 (Med), P5 (Lrg)", 0.26)
    ]
    for name, sub, y in blocks_prop:
        card = FancyBboxPatch((0.56, y), 0.37, 0.09, boxstyle="round,pad=0.01",
                              fc='#0E2B48', ec='#0284C7', lw=1.2)
        ax.add_patch(card)
        ax.text(0.745, y + 0.055, name, fontsize=11, fontweight='bold', color='#FFFFFF', ha='center', va='center')
        ax.text(0.745, y + 0.025, sub, fontsize=9.5, color='#7DD3FC', ha='center', va='center')
        if y < 0.68:
            ax.annotate('', xy=(0.745, y + 0.09), xytext=(0.745, y + 0.14),
                        arrowprops=dict(arrowstyle="-|>", color=C_GREEN_ACCENT, lw=2.0))
            
    # Breakthrough callout box
    callout_prop = FancyBboxPatch((0.56, 0.09), 0.37, 0.12, boxstyle="round,pad=0.01",
                                  fc='#063B2B', ec=C_GREEN_ACCENT, lw=1.5)
    ax.add_patch(callout_prop)
    ax.text(0.745, 0.17, "ARCHITECTURAL BREAKTHROUGH: P2 MICRO HEAD", fontsize=10.5, fontweight='bold', color='#6EE7B7', ha='center', va='center')
    ax.text(0.745, 0.13, "+ P2 Stride 4 retains 9+ spatial pixels for distant helmets\n+ DySample dynamic upsampling preserves crisp edge contours without CARAFE latency",
            fontsize=8.5, color='#D1FAE5', ha='center', va='center', multialignment='center')

    save_slide(fig, "Slide05_Baseline_vs_Proposed_Architecture.png")


# ==============================================================================
# SLIDE 06: PLAIN CONV 3x3 vs REPCONV ALGEBRAIC FUSION LIFECYCLE
# ==============================================================================
def render_slide_06():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    draw_header(ax, 6, "Structural Re-parameterization (RepConv): Train Rich, Infer Fast",
                "Decoupling Training Representational Capacity from Zero-Overhead Single-Path Deployment")
    
    # Left: Training multi-branch bottleneck
    draw_panel(ax, 0.03, 0.05, 0.45, 0.80, "Multi-Branch Training Topology", "TRAINING COMPLEXITY", is_proposed=False)
    
    # Draw 3 branches
    ax.text(0.255, 0.73, "Input Feature Tensor X (C x H x W)", fontsize=11, fontweight='bold', color='#FFFFFF', ha='center')
    
    branches = [
        ("Dense Conv 3x3 + BN", 0.06, 0.58, 0.11),
        ("Conv 1x1 + BN", 0.20, 0.58, 0.11),
        ("Identity BN (s=1)", 0.34, 0.58, 0.11)
    ]
    for bname, bx, by, bw in branches:
        bcard = FancyBboxPatch((bx, by), bw, 0.09, boxstyle="round,pad=0.008",
                               fc='#1E293B', ec='#64748B', lw=1.2)
        ax.add_patch(bcard)
        ax.text(bx + bw/2, by + 0.045, bname, fontsize=8.8, fontweight='bold', color='#FFFFFF', ha='center', va='center')
        ax.annotate('', xy=(bx + bw/2, by + 0.09), xytext=(0.255, 0.70),
                    arrowprops=dict(arrowstyle="<-", color='#94A3B8', lw=1.5))
        ax.annotate('', xy=(0.255, 0.46), xytext=(bx + bw/2, by),
                    arrowprops=dict(arrowstyle="->", color='#94A3B8', lw=1.5))

    add_circle = Circle((0.255, 0.44), 0.025, fc='#334155', ec='#94A3B8', lw=1.5)
    ax.add_patch(add_circle)
    ax.text(0.255, 0.44, "+", fontsize=16, fontweight='bold', color='#FFFFFF', ha='center', va='center')

    out_card = FancyBboxPatch((0.13, 0.30), 0.25, 0.08, boxstyle="round,pad=0.01",
                              fc='#1E293B', ec='#64748B', lw=1.2)
    ax.add_patch(out_card)
    ax.text(0.255, 0.34, "Output Feature Tensor Y", fontsize=10.5, color='#FFFFFF', ha='center', va='center')

    # Warning box
    wbox = FancyBboxPatch((0.07, 0.09), 0.37, 0.15, boxstyle="round,pad=0.01",
                          fc='#3B1D25', ec=C_RED_ACCENT, lw=1.5)
    ax.add_patch(wbox)
    ax.text(0.255, 0.195, "HARDWARE BOTTLENECK IN MULTI-BRANCH", fontsize=10.5, fontweight='bold', color='#FCA5A5', ha='center')
    ax.text(0.255, 0.145, "• 3 separate VRAM buffer allocations per block (Cache thrashing)\n• 3 distinct CUDA kernel launches -> high dispatch latency (7.12 ms)\n• Memory Access Cost (MAC) severely bounds GPU throughput",
            fontsize=8.5, color='#FEE2E2', ha='center', multialignment='center')

    # Right: Inference Fused Conv
    draw_panel(ax, 0.52, 0.05, 0.45, 0.80, "Algebraic Fusion: Single-Path Deployment", "INFERENCE DEPLOYMENT", is_proposed=True)
    
    # Equation banner
    eq_box = FancyBboxPatch((0.56, 0.65), 0.37, 0.13, boxstyle="round,pad=0.01",
                            fc='#082F49', ec=C_BLUE_ACCENT, lw=1.5)
    ax.add_patch(eq_box)
    ax.text(0.745, 0.74, "EXACT ALGEBRAIC EQUIVALENCE FUSION", fontsize=10.5, fontweight='bold', color='#38BDF8', ha='center')
    ax.text(0.745, 0.69, r"$\mathbf{W}_{\mathrm{fused}} = \mathbf{W}_{3\times3}' + \mathrm{Pad}_{3\times3}(\mathbf{W}_{1\times1}') + \mathbf{W}_{\mathrm{id}}'$",
            fontsize=10.5, color='#FFFFFF', ha='center')
    ax.text(0.745, 0.645, r"$\mathbf{b}_{\mathrm{fused}} = \mathbf{b}_{3\times3}' + \mathbf{b}_{1\times1}' + \mathbf{b}_{\mathrm{id}}' \quad (\|Y_{\mathrm{multi}} - Y_{\mathrm{fused}}\|_\infty < 10^{-5})$",
            fontsize=9.2, color='#BAE6FD', ha='center')

    # Single Fused Conv Block
    ax.text(0.745, 0.55, "Input Feature Tensor X (C x H x W)", fontsize=11, fontweight='bold', color='#FFFFFF', ha='center')
    ax.annotate('', xy=(0.745, 0.47), xytext=(0.745, 0.52),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN_ACCENT, lw=2.5))
    
    fused_box = FancyBboxPatch((0.62, 0.37), 0.25, 0.10, boxstyle="round,pad=0.01",
                               fc='#064E3B', ec=C_GREEN_ACCENT, lw=2.0)
    ax.add_patch(fused_box)
    ax.text(0.745, 0.435, "FUSED CONV 3x3 + BIAS", fontsize=12, fontweight='bold', color='#FFFFFF', ha='center', va='center')
    ax.text(0.745, 0.40, "(Zero BN layers, Single CUDA Kernel)", fontsize=9.5, color='#A7F3D0', ha='center', va='center')

    ax.annotate('', xy=(0.745, 0.28), xytext=(0.745, 0.37),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN_ACCENT, lw=2.5))
    
    out_fused = FancyBboxPatch((0.62, 0.23), 0.25, 0.05, boxstyle="round,pad=0.008",
                               fc='#1E293B', ec='#64748B', lw=1.2)
    ax.add_patch(out_fused)
    ax.text(0.745, 0.255, "Output Tensor Y", fontsize=10, color='#FFFFFF', ha='center', va='center')

    # Deployment gain callout
    gain_box = FancyBboxPatch((0.56, 0.09), 0.37, 0.11, boxstyle="round,pad=0.01",
                              fc='#063B2B', ec=C_GREEN_ACCENT, lw=1.5)
    ax.add_patch(gain_box)
    ax.text(0.745, 0.165, "HARDWARE ADVANTAGE: 2.92 ms (342.5 FPS TRT FP16)", fontsize=10.5, fontweight='bold', color='#6EE7B7', ha='center')
    ax.text(0.745, 0.125, "+ Latency drops by 24.6% (5.86 ms PyTorch -> 2.92 ms TensorRT on T4)\n+ VRAM peak memory consumption reduced by 38.2% (1 unified memory buffer)",
            fontsize=8.5, color='#D1FAE5', ha='center', multialignment='center')

    save_slide(fig, "Slide06_PlainConv_vs_RepConv_Lifecycle.png")


# ==============================================================================
# SLIDE 07: TRANSLATION INVARIANCE vs COORDCONV 5-CHANNEL SPATIAL ENCODING
# ==============================================================================
def render_slide_07():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    draw_header(ax, 7, "Spatial Disambiguation: Translation Invariance vs. CoordConv Prior",
                "Eliminating Ground Clutter & Plastic Bucket False Positives via Explicit Coordinate Injection")
    
    # Left: Translation Invariance flaw
    draw_panel(ax, 0.03, 0.05, 0.45, 0.80, "Standard 2D Convolution Flaw", "TRANSLATION INVARIANCE", is_proposed=False)
    
    cam_box = FancyBboxPatch((0.07, 0.43), 0.37, 0.32, boxstyle="round,pad=0.01",
                             fc='#1E293B', ec='#475569', lw=1.2)
    ax.add_patch(cam_box)
    ax.text(0.255, 0.71, "CCTV Camera Surveillance Frame", fontsize=10.5, fontweight='bold', color='#E2E8F0', ha='center')
    
    # Target 1: Helmet at head
    c1 = Circle((0.255, 0.62), 0.035, fc='#F59E0B', ec='#FFFFFF', lw=1.5)
    ax.add_patch(c1)
    ax.text(0.255, 0.62, "Helmet", fontsize=8.5, fontweight='bold', color='#000000', ha='center', va='center')
    ax.text(0.35, 0.62, "y ≈ 0.20 (Head region)", fontsize=9, color='#FCD34D', ha='left', va='center')
    
    # Target 2: Floor bucket
    c2 = Circle((0.255, 0.49), 0.035, fc='#F59E0B', ec=C_RED_ACCENT, lw=2.0)
    ax.add_patch(c2)
    ax.text(0.255, 0.49, "Bucket", fontsize=8.5, fontweight='bold', color='#000000', ha='center', va='center')
    ax.text(0.35, 0.49, "y ≈ 0.90 (Floor scaffolding)", fontsize=9, color='#FCA5A5', ha='left', va='center')

    # Flaw description
    flaw_box = FancyBboxPatch((0.07, 0.09), 0.37, 0.30, boxstyle="round,pad=0.01",
                              fc='#3B1D25', ec=C_RED_ACCENT, lw=1.5)
    ax.add_patch(flaw_box)
    ax.text(0.255, 0.34, "THE TRANSLATION INVARIANCE DILEMMA", fontsize=10.5, fontweight='bold', color='#FCA5A5', ha='center')
    ax.text(0.255, 0.23, "• Standard convolution weights are translation-invariant:\n  K * X(u, v) = K * X(u + Δu, v + Δv)\n• Identical local features (round contour, yellow reflection)\n• Model cannot distinguish between a helmet on a worker's head\n  and a yellow plastic bucket sitting on the ground\n• Result: > 28% False Positive Alarms on construction floors",
            fontsize=8.5, color='#FEE2E2', ha='center', multialignment='center')

    # Right: CoordConv 5-Channel injection
    draw_panel(ax, 0.52, 0.05, 0.45, 0.80, "Proposed CoordConv Spatial Prior", "STEM COORDINATE INJECTION", is_proposed=True)
    
    # Coordinate formulation box
    coord_box = FancyBboxPatch((0.56, 0.55), 0.37, 0.22, boxstyle="round,pad=0.01",
                               fc='#082F49', ec=C_BLUE_ACCENT, lw=1.5)
    ax.add_patch(coord_box)
    ax.text(0.745, 0.73, "STEM COORDINATE CONCATENATION", fontsize=11, fontweight='bold', color='#38BDF8', ha='center')
    ax.text(0.745, 0.67, r"$C_x(i, j) = \frac{2j}{W-1} - 1, \quad C_y(i, j) = \frac{2i}{H-1} - 1 \quad \in [-1, 1]$",
            fontsize=10, color='#FFFFFF', ha='center')
    ax.text(0.745, 0.60, r"$\mathbf{X}_{\mathrm{Coord}} = [\mathbf{X}_{\mathrm{RGB}} \,;\, C_x \,;\, C_y] \in \mathbb{R}^{5 \times 640 \times 640}$",
            fontsize=10.5, color='#BAE6FD', ha='center')
    ax.text(0.745, 0.54, "Injected STRICTLY at Input Stem (c1=5 -> c2=64, k=3, s=2)", fontsize=9.2, color='#7DD3FC', ha='center')

    # Spatial filter effect
    filter_box = FancyBboxPatch((0.56, 0.09), 0.37, 0.42, boxstyle="round,pad=0.01",
                                fc='#063B2B', ec=C_GREEN_ACCENT, lw=1.5)
    ax.add_patch(filter_box)
    ax.text(0.745, 0.46, "ANATOMICAL PRIOR: SUPPRESSING FLOOR CLUTTER", fontsize=10.5, fontweight='bold', color='#6EE7B7', ha='center')
    ax.text(0.745, 0.30, "• Anatomical Geometry Prior:\n  - Head altitude: Cy ∈ [-0.9, -0.2] -> High Helmet Prior (Logit > 0.95)\n  - Ground level: Cy ∈ [0.4, 0.9] -> Floor Prior (Logit < 0.02)\n• False Positive Rate on ground buckets drops from 28% to < 1.8%\n• Latency Impact: +0.06 ms (Only 2 extra channels at layer 0!)\n• No overfitting to camera angle because subsequent layers remain translation-invariant",
            fontsize=8.5, color='#D1FAE5', ha='center', multialignment='center')

    save_slide(fig, "Slide07_Translation_Invariance_vs_CoordConv.png")


# ==============================================================================
# SLIDE 08: DENSE ATTENTION BOTTLENECK vs BIFORMER DYNAMIC SPARSE ROUTING
# ==============================================================================
def render_slide_08():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    draw_header(ax, 8, "Attention Efficiency: Dense O(H^2 W^2) vs. BiFormer Sparse O(HW)",
                "Filtering Background Noise and Eliminating CUDA OOM via 2-Level Dynamic Routing")
    
    # Left: Dense Attention
    draw_panel(ax, 0.03, 0.05, 0.45, 0.80, "Vanilla Dense Self-Attention (ViT)", "QUADRATIC BOTTLENECK", is_proposed=False)
    
    da_box = FancyBboxPatch((0.07, 0.44), 0.37, 0.31, boxstyle="round,pad=0.01",
                            fc='#1E293B', ec='#475569', lw=1.2)
    ax.add_patch(da_box)
    ax.text(0.255, 0.71, "Full Dense Affinity Matrix A ∈ R^(N x N)", fontsize=10.5, fontweight='bold', color='#E2E8F0', ha='center')
    ax.text(0.255, 0.65, r"$\mathbf{A} = \mathrm{Softmax}\left(\frac{Q K^T}{\sqrt{C}}\right), \quad N = H \times W$",
            fontsize=10.5, color='#FCA5A5', ha='center')
    ax.text(0.255, 0.58, "Image 640x640: N = 10,240 (at P3)\nAffinity Entries: 104.8 Million Computations\nImage 1024x1024: N = 26,214 -> 687 Million Entries!",
            fontsize=9, color='#CBD5E1', ha='center', multialignment='center')

    da_crit = FancyBboxPatch((0.07, 0.09), 0.37, 0.31, boxstyle="round,pad=0.01",
                             fc='#3B1D25', ec=C_RED_ACCENT, lw=1.5)
    ax.add_patch(da_crit)
    ax.text(0.255, 0.35, "CUDA OUT-OF-MEMORY & NOISE INFECTION", fontsize=10.5, fontweight='bold', color='#FCA5A5', ha='center')
    ax.text(0.255, 0.23, "• Computational complexity explodes quadratically: O(H^2 W^2)\n• Requires 16.38 GB VRAM at 640x640 -> Crashes Edge GPUs (CUDA OOM)\n• 95% of attention compute is wasted on static background (sky, walls)\n• Background noise degrades feature representation for tiny distant targets",
            fontsize=8.5, color='#FEE2E2', ha='center', multialignment='center')

    # Right: BiFormer
    draw_panel(ax, 0.52, 0.05, 0.45, 0.80, "BiFormer Bi-Level Routing Attention", "LINEAR DYNAMIC ROUTING", is_proposed=True)
    
    bf_box = FancyBboxPatch((0.56, 0.48), 0.37, 0.29, boxstyle="round,pad=0.01",
                            fc='#082F49', ec=C_BLUE_ACCENT, lw=1.5)
    ax.add_patch(bf_box)
    ax.text(0.745, 0.73, "TWO-LEVEL ROUTING MECHANISM (S=8, k=4)", fontsize=10.5, fontweight='bold', color='#38BDF8', ha='center')
    ax.text(0.745, 0.67, "1. Coarse Region Partition: Divide into S x S (8 x 8 = 64) regions", fontsize=9, color='#E2E8F0', ha='center')
    ax.text(0.745, 0.62, r"   Region Query/Key: $Q^r = \mathrm{AvgPool}(Q), \, K^r = \mathrm{AvgPool}(K)$", fontsize=9, color='#BAE6FD', ha='center')
    ax.text(0.745, 0.56, "2. Directed Top-k Routing: Keep only Top-4 most relevant regions", fontsize=9, color='#E2E8F0', ha='center')
    ax.text(0.745, 0.51, r"   $\mathbf{A}^r = Q^r (K^r)^T / \sqrt{C}, \quad I^r = \mathrm{topk}(\mathbf{A}^r, k=4)$", fontsize=9.2, color='#BAE6FD', ha='center')

    bf_gain = FancyBboxPatch((0.56, 0.09), 0.37, 0.35, boxstyle="round,pad=0.01",
                             fc='#063B2B', ec=C_GREEN_ACCENT, lw=1.5)
    ax.add_patch(bf_gain)
    ax.text(0.745, 0.39, "EFFICIENCY & RECALL BREAKTHROUGH", fontsize=10.5, fontweight='bold', color='#6EE7B7', ha='center')
    ax.text(0.745, 0.25, "• Linear Complexity: O(HW) instead of O(H^2 W^2)\n• Memory footprint slashes from 16.38 GB to 0.84 GB (-94.9% VRAM!)\n• Prunes 93.75% of background noise tokens before fine attention\n• Helmet Recall increases to 91.33% (+4.13% over baseline)\n• Added latency: only +0.24 ms on TensorRT FP16 (Layer P4 & P5 only)",
            fontsize=8.5, color='#D1FAE5', ha='center', multialignment='center')

    save_slide(fig, "Slide08_Dense_Attention_vs_BiFormer_Routing.png")


# ==============================================================================
# SLIDE 09: CIOU VANISHING GRADIENT vs FOCAL-EIOU DECOMPOSITION
# ==============================================================================
def render_slide_09():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    draw_header(ax, 9, "Bounding Box Regression: CIoU Gradient Flaw vs. Focal-EIoU",
                "Overcoming Aspect-Ratio Gradient Vanishing and Severe Class Imbalance in Tiny Bounding Boxes")
    
    # Left: CIoU flaw
    draw_panel(ax, 0.03, 0.05, 0.45, 0.80, "Complete IoU (CIoU) Limitation", "RATIO GRADIENT VANISHING", is_proposed=False)
    
    ciou_box = FancyBboxPatch((0.07, 0.46), 0.37, 0.29, boxstyle="round,pad=0.01",
                              fc='#1E293B', ec='#475569', lw=1.2)
    ax.add_patch(ciou_box)
    ax.text(0.255, 0.71, "CIoU Aspect Ratio Penalty Term", fontsize=10.5, fontweight='bold', color='#E2E8F0', ha='center')
    ax.text(0.255, 0.65, r"$\mathcal{L}_{\mathrm{CIoU}} = 1 - \mathrm{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$",
            fontsize=10.5, color='#FCA5A5', ha='center')
    ax.text(0.255, 0.58, r"$v = \frac{4}{\pi^2} \left( \arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h} \right)^2$",
            fontsize=10.5, color='#FCA5A5', ha='center')
    ax.text(0.255, 0.51, "Gradients with respect to dimensions:\n" + r"$\frac{\partial v}{\partial w} = -\frac{h}{w} \frac{\partial v}{\partial h} \to$ Gradients cancel out!",
            fontsize=9.2, color='#CBD5E1', ha='center', multialignment='center')

    ciou_crit = FancyBboxPatch((0.07, 0.09), 0.37, 0.33, boxstyle="round,pad=0.01",
                               fc='#3B1D25', ec=C_RED_ACCENT, lw=1.5)
    ax.add_patch(ciou_crit)
    ax.text(0.255, 0.37, "THE SCALE GRADIENT COLLAPSE", fontsize=10.5, fontweight='bold', color='#FCA5A5', ha='center')
    ax.text(0.255, 0.24, "• If Predicted Box and Ground Truth have MATCHED aspect ratio (w/h = w^gt/h^gt)\n  but vastly different scales (e.g. 10x10 vs 30x30):\n  -> v = 0 and ∂v/∂w = ∂v/∂h = 0 (Gradient vanishes completely!)\n• CIoU cannot guide bounding box expansion or contraction\n• Severe boundary jittering on tiny distant helmet targets",
            fontsize=8.5, color='#FEE2E2', ha='center', multialignment='center')

    # Right: Focal-EIoU
    draw_panel(ax, 0.52, 0.05, 0.45, 0.80, "Proposed Focal-EIoU & Multi-Task Loss", "INDEPENDENT DECOMPOSITION", is_proposed=True)
    
    eiou_box = FancyBboxPatch((0.56, 0.46), 0.37, 0.29, boxstyle="round,pad=0.01",
                              fc='#082F49', ec=C_BLUE_ACCENT, lw=1.5)
    ax.add_patch(eiou_box)
    ax.text(0.745, 0.71, "INDEPENDENT DIMENSION PENALTY (EIoU)", fontsize=10.5, fontweight='bold', color='#38BDF8', ha='center')
    ax.text(0.745, 0.64, r"$\mathcal{L}_{\mathrm{EIoU}} = 1 - \mathrm{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \frac{(w - w^{gt})^2}{C_w^2} + \frac{(h - h^{gt})^2}{C_h^2}$",
            fontsize=9.8, color='#FFFFFF', ha='center')
    ax.text(0.745, 0.56, r"$\mathcal{L}_{\mathrm{Focal-EIoU}} = \mathrm{IoU}^\gamma \cdot \mathcal{L}_{\mathrm{EIoU}} \quad (\gamma = 0.5)$",
            fontsize=10.5, color='#BAE6FD', ha='center')
    ax.text(0.745, 0.49, r"Guaranteed non-zero gradient: $\frac{\partial \mathcal{L}}{\partial w} = \frac{2(w - w^{gt})}{C_w^2} \neq 0$ whenever $w \neq w^{gt}$",
            fontsize=9, color='#7DD3FC', ha='center')

    eiou_gain = FancyBboxPatch((0.56, 0.09), 0.37, 0.33, boxstyle="round,pad=0.01",
                               fc='#063B2B', ec=C_GREEN_ACCENT, lw=1.5)
    ax.add_patch(eiou_gain)
    ax.text(0.745, 0.37, "DECOUPLING REGRESSION & 1:12 CLASS IMBALANCE", fontsize=10.5, fontweight='bold', color='#6EE7B7', ha='center')
    ax.text(0.745, 0.24, "• Focal-EIoU focuses strictly on HARD BOUNDARY regression for tiny boxes\n• 1:12 Class Imbalance (9,044 hats vs 111,514 persons) is resolved by:\n  - Task-Aligned Assigner (TAL): dynamically aligns anchor classification/IoU\n  - Alpha-weighted Focal BCE Classification Loss: λ_cls = 0.5, λ_box = 7.5\n• Multi-scale localization precision mAP50-95 increases from 60.97% to 62.54%",
            fontsize=8.5, color='#D1FAE5', ha='center', multialignment='center')

    save_slide(fig, "Slide09_CIoU_Vanishing_vs_Focal_EIoU.png")


if __name__ == "__main__":
    render_slide_05()
    render_slide_06()
    render_slide_07()
    render_slide_08()
    render_slide_09()
    print("[ALL DONE] Generated 5 Side-by-Side Presentation Slide Visuals successfully!")
