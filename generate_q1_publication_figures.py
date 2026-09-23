import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# IEEE Q1 Publication Typography & Standards
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUT_DIRS = [
    "review1_genspark_package/figures/scientific_exports",
    "paper_overleaf/figures"
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

# Elegant Scientific Palette (IEEE Q1 Style)
C_BG            = '#FFFFFF'
C_DARK          = '#0F172A' # Slate 900
C_TEXT          = '#1E293B' # Slate 800
C_MUTED         = '#64748B' # Slate 500
C_LINE          = '#94A3B8' # Slate 400
C_BORDER        = '#CBD5E1' # Slate 300

# Color Families: Fill, Edge, Text
BLUE_FILL   = '#EFF6FF'; BLUE_EDGE   = '#2563EB'; BLUE_TEXT   = '#1E40AF'
TEAL_FILL   = '#F0FDFA'; TEAL_EDGE   = '#0D9488'; TEAL_TEXT   = '#115E59'
GREEN_FILL  = '#ECFDF5'; GREEN_EDGE  = '#059669'; GREEN_TEXT  = '#065F46'
AMBER_FILL  = '#FFFBEB'; AMBER_EDGE  = '#D97706'; AMBER_TEXT  = '#92400E'
PURPLE_FILL = '#FAF5FF'; PURPLE_EDGE = '#7C3AED'; PURPLE_TEXT = '#5B21B6'
RED_FILL    = '#FEF2F2'; RED_EDGE    = '#DC2626'; RED_TEXT    = '#991B1B'
GRAY_FILL   = '#F8FAFC'; GRAY_EDGE   = '#64748B'; GRAY_TEXT   = '#334155'


def save_fig(fig, filename):
    for d in OUT_DIRS:
        p = os.path.join(d, filename)
        fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor=C_BG)
    plt.close(fig)
    print(f"[SUCCESS] Exported: {filename}")


# -------------------------------------------------------------
# GRAPHICAL PRIMITIVES
# -------------------------------------------------------------

def add_header(ax, fig_code, title, subtitle):
    """Clean IEEE conference header banner with robust length bounding."""
    ax.text(0.025, 0.965, f"({fig_code}) {title}", fontsize=10.8, fontweight='bold',
            color=C_DARK, ha='left', va='top')
    if subtitle:
        ax.text(0.025, 0.925, subtitle, fontsize=8.0, color=C_MUTED, ha='left', va='top')
    ax.plot([0.025, 0.975], [0.905, 0.905], color=C_BORDER, lw=0.9)


def draw_panel(ax, xy, w, h, title="", fill=C_BG, edge=C_BORDER, lw=1.0, title_col=C_DARK):
    """Draws a clean boundary panel for modular sub-sections."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                         facecolor=fill, edgecolor=edge, lw=lw, zorder=1)
    ax.add_patch(box)
    if title:
        ax.text(x + 0.015, y + h - 0.022, title, fontsize=8.8, fontweight='bold',
                color=title_col, ha='left', va='top', zorder=2)


def draw_cube(ax, x, y, dx, dy, dz, fill=BLUE_FILL, edge=BLUE_EDGE, label="", dims="", alpha=0.9, zorder=3, dims_fs=6.5):
    """Draws an isometric 3D feature tensor cube."""
    front = patches.Polygon([
        (x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)
    ], closed=True, facecolor=fill, edgecolor=edge, lw=1.1, alpha=alpha, zorder=zorder)
    ax.add_patch(front)
    top = patches.Polygon([
        (x, y + dy), (x + dx, y + dy), 
        (x + dx + dz*0.4, y + dy + dz*0.4), (x + dz*0.4, y + dy + dz*0.4)
    ], closed=True, facecolor=fill, edgecolor=edge, lw=1.1, alpha=alpha*0.85, zorder=zorder)
    ax.add_patch(top)
    right = patches.Polygon([
        (x + dx, y), (x + dx + dz*0.4, y + dz*0.4), 
        (x + dx + dz*0.4, y + dy + dz*0.4), (x + dx, y + dy)
    ], closed=True, facecolor=fill, edgecolor=edge, lw=1.1, alpha=alpha*0.7, zorder=zorder)
    ax.add_patch(right)
    
    if label:
        ax.text(x + dx/2, y + dy/2, label, fontsize=8, fontweight='bold',
                color=edge, ha='center', va='center', zorder=zorder+1)
    if dims:
        ax.text(x + dx/2, y - 0.025, dims, fontsize=dims_fs, color=C_TEXT,
                ha='center', va='top', zorder=zorder+1)


def draw_block(ax, xy, w, h, title, subtitle="", tag="", 
               fill=BLUE_FILL, edge=BLUE_EDGE, text_col=BLUE_TEXT, sub_col=None,
               fontsize=8.2, lw=1.1, rad=0.012, zorder=4):
    """Draws a rounded neural network layer block with robust multi-line layout."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                         facecolor=fill, edgecolor=edge, lw=lw, zorder=zorder)
    ax.add_patch(box)
    
    if tag:
        tw = len(tag) * 0.0065 + 0.015
        th = 0.022
        tbox = FancyBboxPatch((x + w - tw - 0.005, y + h - th*0.75), tw, th,
                              boxstyle="round,pad=0,rounding_size=0.006",
                              facecolor=edge, edgecolor='none', zorder=zorder+2)
        ax.add_patch(tbox)
        ax.text(x + w - tw/2 - 0.005, y + h - th*0.25, tag, fontsize=6.0, fontweight='bold',
                color='#FFFFFF', ha='center', va='center', zorder=zorder+3)
        
    s_col = sub_col if sub_col is not None else C_MUTED
    if subtitle:
        sub_lines = subtitle.split('\n')
        if len(sub_lines) > 1:
            ax.text(x + w/2, y + h*0.76, title, fontsize=fontsize, fontweight='bold',
                    color=text_col, ha='center', va='center', zorder=zorder+1)
            ax.text(x + w/2, y + h*0.30, subtitle, fontsize=fontsize*0.70,
                    color=s_col, ha='center', va='center', zorder=zorder+1)
        else:
            if h <= 0.065:
                ax.text(x + w/2, y + h*0.69, title, fontsize=fontsize, fontweight='bold',
                        color=text_col, ha='center', va='center', zorder=zorder+1)
                ax.text(x + w/2, y + h*0.27, subtitle, fontsize=fontsize*0.72,
                        color=s_col, ha='center', va='center', zorder=zorder+1)
            else:
                ax.text(x + w/2, y + h*0.64, title, fontsize=fontsize, fontweight='bold',
                        color=text_col, ha='center', va='center', zorder=zorder+1)
                ax.text(x + w/2, y + h*0.29, subtitle, fontsize=fontsize*0.76,
                        color=s_col, ha='center', va='center', zorder=zorder+1)
    else:
        ax.text(x + w/2, y + h/2, title, fontsize=fontsize, fontweight='bold',
                color=text_col, ha='center', va='center', zorder=zorder+1)


def draw_arrow(ax, p1, p2, color=C_LINE, lw=1.2, label="", label_pos=0.5, 
               text_offset=(0, 0.015), style='-|>', rad=0.0, zorder=4):
    """Draws a clean directed connection arrow."""
    if rad == 0.0:
        arr = FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                              lw=lw, mutation_scale=10, zorder=zorder)
    else:
        arr = FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                              color=color, lw=lw, mutation_scale=10, zorder=zorder)
    ax.add_patch(arr)
    
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + text_offset[0]
        my = p1[1] + (p2[1] - p1[1]) * label_pos + text_offset[1]
        ax.text(mx, my, label, fontsize=7.2, fontweight='bold', color=C_DARK,
                ha='center', va='center', zorder=zorder+1,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.9))


def draw_op(ax, xy, symbol="+", r=0.016, fill=GRAY_FILL, edge=C_DARK, zorder=6):
    """Draws an operation circle: +, (c), (x)."""
    x, y = xy
    c = Circle((x, y), r, facecolor=fill, edgecolor=edge, lw=1.2, zorder=zorder)
    ax.add_patch(c)
    ax.text(x, y, symbol, fontsize=9.5, fontweight='bold', color=edge,
            ha='center', va='center', zorder=zorder+1)


def draw_badge(ax, xy, w, h, line1, line2="", fill=GREEN_FILL, edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=7.2):
    """Draws a clean two-line bottom summary badge that never overflows."""
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                       facecolor=fill, edgecolor=edge, lw=1.1, zorder=3)
    ax.add_patch(b)
    if line2:
        ax.text(x + w/2, y + h*0.62, line1, fontsize=fontsize, fontweight='bold',
                color=text_col, ha='center', va='center', zorder=4)
        ax.text(x + w/2, y + h*0.32, line2, fontsize=fontsize*0.95,
                color=text_col, ha='center', va='center', zorder=4)
    else:
        ax.text(x + w/2, y + h/2, line1, fontsize=fontsize, fontweight='bold',
                color=text_col, ha='center', va='center', zorder=4)


# =====================================================================
# FIGURE 1A: BASELINE MULTI-BRANCH BOTTLENECK
# =====================================================================
def generate_fig1a():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 1A", "Baseline: Conventional Plain Conv vs. Multi-Branch Inference Bottleneck",
               "Single-path gradient representation deficit vs. GPU Memory Access Cost (MAC) explosion when deployed directly")

    # Panel 1: Plain Conv (Vanilla YOLO11s)
    draw_panel(ax, (0.035, 0.16), 0.44, 0.71, "1. Standard Single-Path Conv (Vanilla YOLO11s)", fill=GRAY_FILL, edge=C_BORDER)
    
    # Input Tensor
    draw_cube(ax, 0.060, 0.44, 0.035, 0.18, 0.040, fill=BLUE_FILL, edge=BLUE_EDGE, label="X", dims=r"$C_{\mathrm{in}} \times H \times W$")
    draw_arrow(ax, (0.120, 0.53), (0.145, 0.53))
    
    # Linear Pipeline
    draw_block(ax, (0.145, 0.48), 0.075, 0.09, "Conv 3×3", "k=3, s=1", fill=GRAY_FILL, edge=GRAY_EDGE, text_col=GRAY_TEXT, fontsize=7.8)
    draw_arrow(ax, (0.220, 0.53), (0.245, 0.53))
    draw_block(ax, (0.245, 0.48), 0.075, 0.09, "BatchNorm", r"$\gamma, \beta$", fill=BLUE_FILL, edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.8)
    draw_arrow(ax, (0.320, 0.53), (0.345, 0.53))
    draw_block(ax, (0.345, 0.48), 0.040, 0.09, "SiLU", "Act", fill=AMBER_FILL, edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.6)
    draw_arrow(ax, (0.385, 0.53), (0.405, 0.53))

    # Output Tensor
    draw_cube(ax, 0.405, 0.44, 0.030, 0.18, 0.035, fill=GREEN_FILL, edge=GREEN_EDGE, label="Y", dims=r"$C_{\mathrm{out}} \times H \times W$")

    # Summary box
    p1_box = FancyBboxPatch((0.050, 0.195), 0.415, 0.16, boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor='#FFFFFF', edgecolor=C_BORDER, lw=0.9, zorder=2)
    ax.add_patch(p1_box)
    ax.text(0.065, 0.300, "Single-Path Execution: Fast linear contiguous memory access.", fontsize=6.7, fontweight='bold', color=C_TEXT, zorder=3)
    ax.text(0.065, 0.255, "Training Deficit: Constrained gradient representation space", fontsize=6.7, fontweight='bold', color=RED_TEXT, zorder=3)
    ax.text(0.065, 0.220, "during backpropagation; vulnerable to early saturation.", fontsize=6.5, color=C_MUTED, zorder=3)

    # Panel 2: Naive Multi-Branch in Deployment
    draw_panel(ax, (0.515, 0.16), 0.45, 0.71, "2. Naive Multi-Branch in Deployment", fill=RED_FILL, edge=RED_EDGE, title_col=RED_TEXT)

    # Input Tensor
    draw_cube(ax, 0.538, 0.44, 0.030, 0.18, 0.035, fill=BLUE_FILL, edge=BLUE_EDGE, label="X", dims=r"$C_1 \times H \times W$")
    
    # Branch arrows
    draw_arrow(ax, (0.590, 0.55), (0.615, 0.68), color=RED_EDGE)
    draw_arrow(ax, (0.590, 0.53), (0.615, 0.53), color=AMBER_EDGE)
    draw_arrow(ax, (0.590, 0.51), (0.615, 0.38), color=BLUE_EDGE)

    # 3 Branches
    draw_block(ax, (0.615, 0.64), 0.145, 0.075, "Conv 3×3 + BN", r"$W_{3\times 3} \in \mathbb{R}^{C_2 \times C_1 \times 3 \times 3}$", tag="Path 1", fill='#FFFFFF', edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.3)
    draw_block(ax, (0.615, 0.49), 0.145, 0.075, "Conv 1×1 + BN", r"$W_{1\times 1} \in \mathbb{R}^{C_2 \times C_1 \times 1 \times 1}$", tag="Path 2", fill='#FFFFFF', edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.3)
    draw_block(ax, (0.615, 0.34), 0.145, 0.075, "Identity + BN", r"$\text{BatchNorm2d}(C_1)$", tag="Path 3", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.3)

    # Merge arrows
    draw_arrow(ax, (0.760, 0.68), (0.790, 0.55), color=RED_EDGE)
    draw_arrow(ax, (0.760, 0.53), (0.775, 0.53), color=AMBER_EDGE)
    draw_arrow(ax, (0.760, 0.38), (0.790, 0.51), color=BLUE_EDGE)

    # Sum & Activation
    draw_op(ax, (0.795, 0.53), "+", r=0.015, fill='#FFFFFF', edge=C_DARK)
    draw_arrow(ax, (0.810, 0.53), (0.835, 0.53))
    draw_block(ax, (0.835, 0.495), 0.045, 0.07, "SiLU", "", fill=AMBER_FILL, edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.3)
    draw_arrow(ax, (0.880, 0.53), (0.900, 0.53))
    draw_cube(ax, 0.900, 0.44, 0.025, 0.18, 0.030, fill=GREEN_FILL, edge=GREEN_EDGE, label="Y", dims="")

    # Hardware breakdown
    p2_box = FancyBboxPatch((0.525, 0.195), 0.420, 0.16, boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor='#FFFFFF', edgecolor=RED_EDGE, lw=0.9, zorder=2)
    ax.add_patch(p2_box)
    ax.text(0.540, 0.300, "3× Kernel Launches: Frequent DRAM ↔ SRAM cache thrashing.", fontsize=6.7, fontweight='bold', color=RED_TEXT, zorder=3)
    ax.text(0.540, 0.255, "Latency Penalty: Surges to 7.12 ms (140.4 FPS) on Tesla T4.", fontsize=6.7, fontweight='bold', color=RED_TEXT, zorder=3)
    ax.text(0.540, 0.220, "Completely impractical for real-time 4K CCTV surveillance streams.", fontsize=6.5, color=C_MUTED, zorder=3)

    # Bottom Badge
    draw_badge(ax, (0.035, 0.04), 0.93, 0.09,
               "Architectural Dilemma: Multi-branch enriches backpropagation gradients but destroys deployment throughput.",
               "Core Research Question: How to capture multi-branch diversity during training without incurring any inference penalty?",
               fill=RED_FILL, edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.2)

    save_fig(fig, "Fig1A_Baseline_MultiBranch_Bottleneck.png")


# =====================================================================
# FIGURE 1B: PROPOSED REPCONV ALGEBRAIC FUSION
# =====================================================================
def generate_fig1b():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 1B", "Proposed: Structural Re-parameterization Lifecycle & Algebraic Fusion (RepConv)",
               r"Multi-branch topology during training folded into a single $3\times 3$ Conv2d for zero-overhead inference via switch_to_deploy()")

    # Stage 1: Training Phase (Multi-Branch)
    draw_panel(ax, (0.025, 0.16), 0.305, 0.71, "1. Training Phase (Multi-Branch)", fill=BLUE_FILL, edge=BLUE_EDGE, title_col=BLUE_TEXT)
    
    draw_cube(ax, 0.046, 0.44, 0.030, 0.18, 0.035, fill=BLUE_FILL, edge=BLUE_EDGE, label="X", dims=r"$C_1 \times H \times W$")
    draw_arrow(ax, (0.096, 0.55), (0.118, 0.68), color=RED_EDGE)
    draw_arrow(ax, (0.096, 0.53), (0.118, 0.53), color=AMBER_EDGE)
    draw_arrow(ax, (0.096, 0.51), (0.118, 0.38), color=BLUE_EDGE)

    # 3 Branches
    draw_block(ax, (0.118, 0.64), 0.127, 0.075, "Conv 3×3 + BN", r"$W_3 \in \mathbb{R}^{C_2 \times C_1 \times 3 \times 3}$", fill='#FFFFFF', edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.3)
    draw_block(ax, (0.118, 0.49), 0.127, 0.075, "Conv 1×1 + BN", r"$W_1 \in \mathbb{R}^{C_2 \times C_1 \times 1 \times 1}$", fill='#FFFFFF', edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.3)
    draw_block(ax, (0.118, 0.34), 0.127, 0.075, "Identity + BN", r"$\text{BN}(C_1) \text{ if } C_1=C_2$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.3)

    # Merge
    draw_arrow(ax, (0.245, 0.68), (0.270, 0.55), color=RED_EDGE)
    draw_arrow(ax, (0.245, 0.53), (0.255, 0.53), color=AMBER_EDGE)
    draw_arrow(ax, (0.245, 0.38), (0.270, 0.51), color=BLUE_EDGE)
    draw_op(ax, (0.270, 0.53), "+", r=0.014, fill='#FFFFFF', edge=C_DARK)
    draw_arrow(ax, (0.284, 0.53), (0.310, 0.53))

    ax.text(0.175, 0.23, "Rich Representation Space\nMulti-scale gradient exploration", fontsize=6.8, fontweight='bold', color=BLUE_TEXT, ha='center')

    # Stage 2: Closed-Form Algebraic Fusion
    draw_panel(ax, (0.345, 0.16), 0.365, 0.71, "2. Closed-Form Algebraic Fusion", fill=AMBER_FILL, edge=AMBER_EDGE, title_col=AMBER_TEXT)

    eq_box = FancyBboxPatch((0.360, 0.195), 0.335, 0.58, boxstyle="round,pad=0,rounding_size=0.01",
                            facecolor='#FFFFFF', edgecolor=AMBER_EDGE, lw=0.9, zorder=2)
    ax.add_patch(eq_box)

    ax.text(0.375, 0.730, "Step 1: BatchNorm Folding", fontsize=7.4, fontweight='bold', color=AMBER_TEXT)
    ax.text(0.375, 0.690, r"$W' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W, \quad b' = \beta - \frac{\gamma \mu}{\sqrt{\sigma^2 + \epsilon}}$", fontsize=7.2, color=C_TEXT)

    ax.text(0.375, 0.630, "Step 2: Zero-Padding 1×1 → 3×3", fontsize=7.4, fontweight='bold', color=AMBER_TEXT)
    ax.text(0.375, 0.590, r"$W'_{1\times 1 \to 3\times 3} = \mathrm{pad}(W'_{1\times 1}, [1, 1, 1, 1])$", fontsize=7.2, color=C_TEXT)

    ax.text(0.375, 0.530, "Step 3: Identity as Dirac Delta", fontsize=7.4, fontweight='bold', color=AMBER_TEXT)
    ax.text(0.375, 0.490, r"$W'_{\mathrm{id}}[i, i, 1, 1] = \frac{\gamma_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad b'_{\mathrm{id}} = \beta_i - \frac{\gamma_i \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$", fontsize=6.8, color=C_TEXT)

    ax.text(0.375, 0.430, "Step 4: Linear Summation", fontsize=7.4, fontweight='bold', color=AMBER_TEXT)
    ax.text(0.375, 0.390, r"$W_{\mathrm{fused}} = W'_3 + W'_{1\to 3} + W'_{\mathrm{id}}, \quad b_{\mathrm{fused}} = \sum b'$", fontsize=7.2, color=C_TEXT)

    ax.text(0.375, 0.330, "Step 5: switch_to_deploy() Verification", fontsize=7.4, fontweight='bold', color=GREEN_TEXT)
    ax.text(0.375, 0.290, r"$\| Y_{\mathrm{multi}} - Y_{\mathrm{fused}} \|_{\infty} < 10^{-5}$ (Strict Equivalence)", fontsize=6.8, color=GREEN_TEXT)
    ax.text(0.375, 0.230, "Grounded in custom_ablation_modules.py (Lines 71-163)", fontsize=6.3, color=C_MUTED)

    # Transition arrows
    draw_arrow(ax, (0.330, 0.53), (0.345, 0.53), lw=1.5, color=AMBER_EDGE)
    draw_arrow(ax, (0.710, 0.53), (0.720, 0.53), lw=1.5, color=GREEN_EDGE)

    # Stage 3: Inference Phase (Single-Path Deploy)
    draw_panel(ax, (0.720, 0.16), 0.255, 0.71, "3. Inference (Single-Path)", fill=GREEN_FILL, edge=GREEN_EDGE, title_col=GREEN_TEXT)

    draw_cube(ax, 0.747, 0.44, 0.020, 0.16, 0.022, fill=BLUE_FILL, edge=BLUE_EDGE, label="X", dims=r"$C_1 \times H \times W$")
    draw_arrow(ax, (0.771, 0.52), (0.782, 0.52))

    draw_block(ax, (0.782, 0.45), 0.096, 0.14, "Fused Conv 3×3", r"$(W_{\mathrm{fused}}, b_{\mathrm{fused}})$" + "\n" + r"$s=1, p=1$", tag="Single", fill='#FFFFFF', edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=7.0)
    draw_arrow(ax, (0.878, 0.52), (0.892, 0.52))

    draw_block(ax, (0.892, 0.48), 0.028, 0.08, "SiLU", "Act", fill=AMBER_FILL, edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=6.5)
    draw_arrow(ax, (0.920, 0.52), (0.931, 0.52))

    draw_cube(ax, 0.931, 0.44, 0.016, 0.16, 0.018, fill=GREEN_FILL, edge=GREEN_EDGE, label="Y", dims=r"$C_2 \times H \times W$")

    p3_box = FancyBboxPatch((0.735, 0.195), 0.225, 0.18, boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor='#FFFFFF', edgecolor=GREEN_EDGE, lw=0.9, zorder=2)
    ax.add_patch(p3_box)
    ax.text(0.847, 0.325, "Single-Path Execution", fontsize=7.2, fontweight='bold', color=GREEN_TEXT, ha='center', zorder=3)
    ax.text(0.847, 0.275, "Zero DRAM cache thrashing;\nLinear contiguous access.", fontsize=6.5, color=C_TEXT, ha='center', zorder=3)
    ax.text(0.847, 0.220, "Latency: 2.92 ms (342.5 FPS)", fontsize=7.0, fontweight='bold', color=GREEN_TEXT, ha='center', zorder=3)

    # Bottom Badge
    draw_badge(ax, (0.03, 0.04), 0.94, 0.09,
               "Empirical Milestone: GPU latency drops from 7.12 ms to 2.92 ms on Tesla T4 (-55.2%) | Throughput: 342.5 FPS TensorRT FP16.",
               "Mathematical Rigor: Numerical output error delta strictly < 10^-5 | Grounded in custom_ablation_modules.py.",
               fill=GREEN_FILL, edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=7.2)

    save_fig(fig, "Fig1B_Proposed_RepConv_Algebraic_Fusion.png")


# =====================================================================
# FIGURE 2A: BASELINE TRANSLATION INVARIANCE FLAW
# =====================================================================
def generate_fig2a():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 2A", "Baseline: 2D Convolution Translation Invariance & Surveillance False Positives",
               "Standard 3-channel convolution is position-agnostic, yielding identical activation responses for helmets and ground distractors")

    # Panel 1: Surveillance Scene (CCTV Coordinate Plane)
    draw_panel(ax, (0.035, 0.16), 0.345, 0.71, "1. CCTV Scene (Normalized y-Axis)", fill=GRAY_FILL, edge=C_BORDER)

    # Vertical y-axis (inside panel)
    ax.annotate("", xy=(0.065, 0.25), xytext=(0.065, 0.78),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2))
    ax.text(0.075, 0.78, "y=0.0 (Sky)", fontsize=6.5, color=C_MUTED)
    ax.text(0.075, 0.25, "y=1.0 (Floor)", fontsize=6.5, color=C_MUTED)

    # Target 1: Helmet on Head
    h_box = FancyBboxPatch((0.115, 0.61), 0.245, 0.15, boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, lw=1.2, zorder=3)
    ax.add_patch(h_box)
    c1 = Circle((0.145, 0.69), 0.018, facecolor='#F59E0B', edgecolor='#B45309', lw=1.2, zorder=4)
    ax.add_patch(c1)
    ax.text(0.175, 0.71, "Safety Helmet (y ≈ 0.20)", fontsize=7.0, fontweight='bold', color=GREEN_TEXT)
    ax.text(0.175, 0.65, "Worker Head (True Target)\nYellow Dome Contour", fontsize=6.3, color=C_TEXT)

    # Target 2: Mortar Bucket on Floor
    b_box = FancyBboxPatch((0.115, 0.31), 0.245, 0.15, boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor=RED_FILL, edgecolor=RED_EDGE, lw=1.2, zorder=3)
    ax.add_patch(b_box)
    c2 = Circle((0.145, 0.39), 0.018, facecolor='#F59E0B', edgecolor='#DC2626', lw=1.2, zorder=4)
    ax.add_patch(c2)
    ax.text(0.175, 0.41, "Floor Bucket (y ≈ 0.90)", fontsize=7.0, fontweight='bold', color=RED_TEXT)
    ax.text(0.175, 0.35, "Ground Distractor\nIdentical Yellow Contour", fontsize=6.3, color=C_TEXT)

    # Panel 2: Standard 3-Channel Convolution
    draw_panel(ax, (0.40, 0.16), 0.295, 0.71, "2. Standard 2D Conv", fill='#FFFFFF', edge=C_BORDER)

    draw_cube(ax, 0.42, 0.44, 0.035, 0.18, 0.04, fill=BLUE_FILL, edge=BLUE_EDGE, label="RGB", dims=r"$3 \times H \times W$")
    draw_arrow(ax, (0.475, 0.53), (0.505, 0.53))

    draw_block(ax, (0.505, 0.43), 0.16, 0.20, "Standard 2D Conv",
               r"$K \in \mathbb{R}^{C_{\mathrm{out}} \times 3 \times 3 \times 3}$" + "\n\n" + r"$\mathcal{T}_{\Delta}[X * K] =$" + "\n" + r"$[\mathcal{T}_{\Delta} X] * K$",
               tag="Position-Blind", fill=GRAY_FILL, edge=GRAY_EDGE, text_col=GRAY_TEXT, fontsize=7.5)

    draw_arrow(ax, (0.665, 0.53), (0.715, 0.53))

    # Panel 3: Feature Map & False Positive Failure
    draw_panel(ax, (0.715, 0.16), 0.245, 0.71, "3. Detection Output", fill=RED_FILL, edge=RED_EDGE, title_col=RED_TEXT)

    # Heatmap 1: Helmet
    r1 = Rectangle((0.730, 0.62), 0.21, 0.13, facecolor='#FEF3C7', edgecolor=AMBER_EDGE, lw=1.0, zorder=2)
    ax.add_patch(r1)
    ax.text(0.835, 0.71, r"Activation at $y \approx 0.20$", fontsize=7.0, fontweight='bold', color=AMBER_TEXT, ha='center')
    ax.text(0.835, 0.66, r"$S_{\mathrm{helmet}} \approx 0.88 \rightarrow \mathbf{True\ Positive}$", fontsize=6.8, color=GREEN_TEXT, ha='center')

    # Heatmap 2: Bucket
    r2 = Rectangle((0.730, 0.32), 0.21, 0.13, facecolor='#FEE2E2', edgecolor=RED_EDGE, lw=1.0, zorder=2)
    ax.add_patch(r2)
    ax.text(0.835, 0.41, r"Activation at $y \approx 0.90$", fontsize=7.0, fontweight='bold', color=RED_TEXT, ha='center')
    ax.text(0.835, 0.36, r"$S_{\mathrm{bucket}} \approx 0.84 \rightarrow \mathbf{FALSE\ ALARM!}$", fontsize=6.8, color=RED_TEXT, ha='center')

    ax.text(0.835, 0.22, r"$\bullet$ Conv treats locations as identical" + "\n" + r"$\bullet$ Induces >28% ground false alarms", fontsize=6.5, color=RED_TEXT, ha='center')

    # Bottom Badge
    draw_badge(ax, (0.04, 0.04), 0.92, 0.09,
               "Theoretical Flaw: Convolutional weight sharing enforces spatial translation invariance regardless of object elevation.",
               "Surveillance Impact: Standard YOLO11s triggers over 28% false alarms on construction floors (yellow buckets, cones, posters).",
               fill=RED_FILL, edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.2)

    save_fig(fig, "Fig2A_Baseline_Translation_Invariance_Flaw.png")


# =====================================================================
# FIGURE 2B: PROPOSED COORDCONV SPATIAL INJECTION
# =====================================================================
def generate_fig2b():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 2B", "Proposed: CoordConv 2D Cartesian Coordinate Augmentation & Spatial Prior",
               r"Injecting normalized spatial coordinates $C_x, C_y \in [-1, 1]$ breaks translation invariance and encodes anatomical vertical priors")

    # Panel 1: Input & AddCoords Generator
    draw_panel(ax, (0.03, 0.16), 0.33, 0.71, "1. Input & AddCoords Generator", fill=TEAL_FILL, edge=TEAL_EDGE, title_col=TEAL_TEXT)

    # RGB
    draw_cube(ax, 0.05, 0.60, 0.035, 0.14, 0.04, fill=BLUE_FILL, edge=BLUE_EDGE, label="RGB", dims=r"$3 \times H \times W$")
    
    # Coordinate channels
    draw_cube(ax, 0.05, 0.42, 0.035, 0.09, 0.04, fill='#E0F2FE', edge='#0284C7', label=r"$C_x$", dims=r"$1 \times H \times W$")
    draw_cube(ax, 0.05, 0.26, 0.035, 0.09, 0.04, fill='#FEF3C7', edge=AMBER_EDGE, label=r"$C_y$", dims=r"$1 \times H \times W$")

    # Formulation inside panel
    coord_box = FancyBboxPatch((0.105, 0.23), 0.235, 0.30, boxstyle="round,pad=0,rounding_size=0.008",
                               facecolor='#FFFFFF', edgecolor=TEAL_EDGE, lw=0.9, zorder=2)
    ax.add_patch(coord_box)
    ax.text(0.115, 0.485, r"$C_x(i, j) = \frac{2j}{W-1} - 1 \in [-1, 1]$", fontsize=6.8, color=BLUE_TEXT)
    ax.text(0.115, 0.435, r"$C_y(i, j) = \frac{2i}{H-1} - 1 \in [-1, 1]$", fontsize=6.8, color=AMBER_TEXT)
    ax.text(0.115, 0.380, r"$\bullet\ C_y = -1.0$: Frame ceiling/sky", fontsize=6.4, color=C_MUTED)
    ax.text(0.115, 0.340, r"$\bullet\ C_y \approx -0.6$: Worker head elevation", fontsize=6.4, color=GREEN_TEXT)
    ax.text(0.115, 0.300, r"$\bullet\ C_y = +1.0$: Concrete ground floor", fontsize=6.4, color=RED_TEXT)
    ax.text(0.115, 0.255, "Code: custom_ablation_modules.py (L46-69)", fontsize=6.1, color=C_MUTED)

    draw_arrow(ax, (0.36, 0.53), (0.38, 0.53), lw=1.4, color=TEAL_EDGE)

    # Panel 2: 5-Channel Concatenation & Stem Conv
    draw_panel(ax, (0.38, 0.16), 0.34, 0.71, "2. 5-Channel CoordConv Stem", fill='#FFFFFF', edge=TEAL_EDGE, title_col=TEAL_TEXT)

    # 5-Channel Tensor
    draw_cube(ax, 0.410, 0.44, 0.038, 0.18, 0.045, fill=TEAL_FILL, edge=TEAL_EDGE, label="5-Ch", dims=r"$[R, G, B, C_x, C_y]$")
    draw_arrow(ax, (0.470, 0.53), (0.495, 0.53))

    # Conv Stem Block
    draw_block(ax, (0.495, 0.43), 0.20, 0.20, "CoordConv Stem",
               r"$\mathrm{ConvBNAct}(c_1=5, c_2=64)$" + "\n\n" +
               r"$S(x, y) = X * K_{\mathrm{RGB}} +$" + "\n" +
               r"$C_x * K_{Cx} + C_y * K_{Cy} + b$",
               tag="Spatial Aware", fill=TEAL_FILL, edge=TEAL_EDGE, text_col=TEAL_TEXT, sub_col=C_DARK, fontsize=7.2)

    stem_callout = FancyBboxPatch((0.40, 0.20), 0.30, 0.16, boxstyle="round,pad=0,rounding_size=0.008",
                                  facecolor='#F0FDFA', edgecolor=TEAL_EDGE, lw=0.9, zorder=2)
    ax.add_patch(stem_callout)
    ax.text(0.415, 0.320, "Learned Anatomical Spatial Prior:", fontsize=7.0, fontweight='bold', color=TEAL_TEXT)
    ax.text(0.415, 0.275, r"When $C_y > 0.5$ (ground floor), spatial filter", fontsize=6.8, color=C_TEXT)
    ax.text(0.415, 0.230, "strongly suppresses safety helmet class logits!", fontsize=6.8, color=GREEN_TEXT)

    draw_arrow(ax, (0.72, 0.53), (0.74, 0.53), lw=1.4, color=GREEN_EDGE)

    # Panel 3: Saliency Output
    draw_panel(ax, (0.74, 0.16), 0.23, 0.71, "3. Spatial Saliency Output", fill=GREEN_FILL, edge=GREEN_EDGE, title_col=GREEN_TEXT)

    # Saliency Helmet
    sh_box = Rectangle((0.755, 0.62), 0.20, 0.13, facecolor='#ECFDF5', edgecolor=GREEN_EDGE, lw=1.0, zorder=2)
    ax.add_patch(sh_box)
    ax.text(0.855, 0.71, r"Head Region ($C_y < 0$)", fontsize=7.0, fontweight='bold', color=GREEN_TEXT, ha='center')
    ax.text(0.855, 0.66, r"$S_{\mathrm{helmet}} \approx 0.96 \rightarrow \mathbf{Confirmed}$", fontsize=6.8, color=GREEN_TEXT, ha='center')

    # Saliency Bucket
    sb_box = Rectangle((0.755, 0.32), 0.20, 0.13, facecolor='#F1F5F9', edgecolor=C_BORDER, lw=1.0, zorder=2)
    ax.add_patch(sb_box)
    ax.text(0.855, 0.41, r"Floor Region ($C_y > 0.5$)", fontsize=7.0, fontweight='bold', color=C_MUTED, ha='center')
    ax.text(0.855, 0.36, r"$S_{\mathrm{bucket}} \approx 0.02 \rightarrow \mathbf{Suppressed!}$", fontsize=6.8, color=C_MUTED, ha='center')

    ax.text(0.855, 0.22, r"$\bullet$ Zeroes out floor false alarms" + "\n" + r"$\bullet$ Overhead strictly +0.06 ms", fontsize=6.5, color=GREEN_TEXT, ha='center')

    # Bottom Badge
    draw_badge(ax, (0.03, 0.04), 0.94, 0.09,
               "Empirical Outcome: Eradicates >28% of false alarms on floor objects with negligible latency overhead (+0.06 ms).",
               "Explainable AI: Grad-CAM confirms gradient concentration exclusively on human heads | custom_ablation_modules.py (Lines 46-69).",
               fill=TEAL_FILL, edge=TEAL_EDGE, text_col=TEAL_TEXT, fontsize=7.2)

    save_fig(fig, "Fig2B_Proposed_CoordConv_Spatial_Injection.png")


# =====================================================================
# FIGURE 3A: BASELINE DENSE ATTENTION BOTTLENECK
# =====================================================================
def generate_fig3a():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 3A", "Baseline: Dense Multi-Head Self-Attention Computational & Memory Bottleneck",
               r"Global pairwise dot products incur quadratic complexity $\mathcal{O}(H^2 W^2)$, triggering CUDA OOM on edge GPUs")

    # Panel 1: Full Pairwise Dense Token Graph
    draw_panel(ax, (0.04, 0.16), 0.31, 0.71, "1. Dense Pairwise Graph", fill=RED_FILL, edge=RED_EDGE, title_col=RED_TEXT)

    # Grid of tokens
    gx0, gy0 = 0.075, 0.32
    cols, rows = 6, 6
    dx_t, dy_t = 0.04, 0.06
    for r in range(rows):
        for c in range(cols):
            x_pos = gx0 + c * dx_t
            y_pos = gy0 + r * dy_t
            dot = Circle((x_pos, y_pos), 0.0055, facecolor=RED_EDGE, edgecolor='none', zorder=4)
            ax.add_patch(dot)

    # Dense crossing lines
    for r1 in range(rows):
        for c1 in range(cols):
            if (r1 + c1) % 3 == 0:
                p1 = (gx0 + c1 * dx_t, gy0 + r1 * dy_t)
                for r2 in range(rows):
                    if r2 % 2 == 0:
                        p2 = (gx0 + (5-c1) * dx_t, gy0 + r2 * dy_t)
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#FCA5A5', lw=0.4, alpha=0.5, zorder=2)

    ax.text(0.195, 0.76, r"$N = H \times W$ Tokens", fontsize=8.0, fontweight='bold', color=RED_TEXT, ha='center')
    ax.text(0.195, 0.22, r"Every pixel connects to every other pixel ($N^2$ pairs)", fontsize=6.8, color=RED_TEXT, ha='center')

    # Panel 2: Affinity Matrix Formulation & Complexity
    draw_panel(ax, (0.37, 0.16), 0.34, 0.71, "2. Quadratic Complexity", fill='#FFFFFF', edge=C_BORDER)

    aff_box = FancyBboxPatch((0.385, 0.32), 0.31, 0.45, boxstyle="round,pad=0,rounding_size=0.01",
                             facecolor='#FFF1F2', edgecolor=RED_EDGE, lw=1.0, zorder=2)
    ax.add_patch(aff_box)

    ax.text(0.40, 0.730, "Pairwise Affinity Matrix:", fontsize=7.4, fontweight='bold', color=RED_TEXT)
    ax.text(0.40, 0.685, r"$\mathbf{A} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$", fontsize=7.5, color=C_TEXT)
    
    ax.text(0.40, 0.615, r"At $640 \times 640$: $N = 40,000$ tokens", fontsize=6.8, color=C_MUTED)
    ax.text(0.40, 0.570, r"$\rightarrow (40,000)^2 = \mathbf{1.6 \times 10^9\ dot\ products}$", fontsize=7.0, fontweight='bold', color=RED_TEXT)

    ax.text(0.40, 0.505, r"At $1024 \times 1024$: $N = 102,400$ tokens", fontsize=6.8, color=C_MUTED)
    ax.text(0.40, 0.460, r"$\rightarrow (102,400)^2 = \mathbf{1.05 \times 10^{10}\ dot\ products}$", fontsize=7.0, fontweight='bold', color=RED_TEXT)

    ax.text(0.40, 0.390, r"Transient memory exceeds 4GB VRAM for $\mathbf{A}$", fontsize=6.7, color=C_MUTED)
    ax.text(0.40, 0.345, r"$\mathcal{O}(H^2 W^2)$ quadratic explosion triggers OOM.", fontsize=6.7, color=RED_TEXT)

    ax.text(0.54, 0.245, "Computational Complexity: O(H²W²)", fontsize=8.0, fontweight='bold', color=RED_TEXT, ha='center')
    ax.text(0.54, 0.195, "Dense token-to-token pairs grow quadratically.", fontsize=6.8, color=C_MUTED, ha='center')

    # Panel 3: Hardware Breakdown
    draw_panel(ax, (0.73, 0.16), 0.24, 0.71, "3. Hardware Failures", fill=RED_FILL, edge=RED_EDGE, title_col=RED_TEXT)

    f_box = FancyBboxPatch((0.745, 0.32), 0.21, 0.45, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=RED_EDGE, lw=0.9, zorder=2)
    ax.add_patch(f_box)

    ax.text(0.76, 0.730, "1. CUDA OOM Crash", fontsize=7.2, fontweight='bold', color=RED_TEXT)
    ax.text(0.76, 0.685, "Exceeds 2GB VRAM on\nedge GPUs (e.g. MX230).", fontsize=6.5, color=C_TEXT)

    ax.text(0.76, 0.600, "2. Compute Waste", fontsize=7.2, fontweight='bold', color=RED_TEXT)
    ax.text(0.76, 0.555, ">80% ops wasted on\nbare concrete & sky.", fontsize=6.5, color=C_TEXT)

    ax.text(0.76, 0.470, "3. Signal Dilution", fontsize=7.2, fontweight='bold', color=RED_TEXT)
    ax.text(0.76, 0.425, "Attention spreads thin\nover tiny helmets (<20px).", fontsize=6.5, color=C_TEXT)

    ax.text(0.85, 0.230, r"$\bullet$ Frame rate drops < 5 FPS" + "\n" + r"$\bullet$ Incompatible with edge CCTV", fontsize=6.5, color=RED_TEXT, ha='center')

    # Bottom Badge
    draw_badge(ax, (0.04, 0.04), 0.92, 0.09,
               "Algorithmic Infeasibility: Dense self-attention chokes edge surveillance frame rates below 5 FPS due to quadratic complexity O(H²W²).",
               "Edge Constraint: Transient attention map storage triggers immediate CUDA OOM on 2GB VRAM edge hardware (GeForce MX230).",
               fill=RED_FILL, edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.2)

    save_fig(fig, "Fig3A_Baseline_Dense_Attention_Bottleneck.png")


# =====================================================================
# FIGURE 3B: PROPOSED BIFORMER SPARSE ROUTING
# =====================================================================
def generate_fig3b():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 3B", "Proposed: BiFormer Bi-Level Routing Attention (BRA) with Dynamic Sparsity",
               r"Two-level routing prunes ~80% background noise and reduces attention complexity to strictly linear $\mathcal{O}(HW)$")

    # Step 1: Region Partitioning
    draw_panel(ax, (0.03, 0.16), 0.30, 0.71, "1. Region Partitioning", fill=BLUE_FILL, edge=BLUE_EDGE, title_col=BLUE_TEXT)

    draw_cube(ax, 0.05, 0.50, 0.035, 0.16, 0.045, fill=BLUE_FILL, edge=BLUE_EDGE, label="X", dims=r"$C \times H \times W$")
    draw_arrow(ax, (0.115, 0.58), (0.155, 0.58))

    # S x S coarse grid
    gx0, gy0 = 0.16, 0.46
    gw, gh = 0.024, 0.028
    for r in range(4):
        for c in range(4):
            fc = AMBER_FILL if (r == 1 and c == 2) or (r == 2 and c == 1) else '#FFFFFF'
            ec = AMBER_EDGE if (r == 1 and c == 2) or (r == 2 and c == 1) else C_BORDER
            ax.add_patch(Rectangle((gx0 + c*(gw+0.004), gy0 + r*(gh+0.004)), gw, gh,
                                   facecolor=fc, edgecolor=ec, lw=0.9, zorder=3))

    ax.text(0.22, 0.65, r"$S \times S$ Coarse Grid ($S=8$)", fontsize=7.5, fontweight='bold', color=BLUE_TEXT, ha='center')

    # Sub-card in Panel 1: completely avoids any text overflow!
    p1_box = FancyBboxPatch((0.045, 0.195), 0.27, 0.21, boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor='#FFFFFF', edgecolor=BLUE_EDGE, lw=0.9, zorder=2)
    ax.add_patch(p1_box)
    ax.text(0.060, 0.365, "Coarse Region Partitioning:", fontsize=7.2, fontweight='bold', color=BLUE_TEXT, zorder=3)
    ax.text(0.060, 0.325, r"$\bullet$ Divides $H \times W$ into $S \times S$ grid ($S=8$)", fontsize=6.5, color=C_TEXT, zorder=3)
    ax.text(0.060, 0.285, r"$\bullet$ Regional mean pooling ($S^2=64$ regions):", fontsize=6.5, color=C_TEXT, zorder=3)
    ax.text(0.080, 0.250, r"$Q^r = \operatorname{AvgPool}(Q)$", fontsize=6.5, color=BLUE_TEXT, zorder=3)
    ax.text(0.080, 0.215, r"$K^r = \operatorname{AvgPool}(K)$", fontsize=6.5, color=BLUE_TEXT, zorder=3)

    draw_arrow(ax, (0.33, 0.53), (0.35, 0.53), lw=1.4, color=AMBER_EDGE)

    # Step 2: Dynamic Region Routing (Top-k)
    draw_panel(ax, (0.35, 0.16), 0.33, 0.71, "2. Dynamic Top-k Routing", fill=AMBER_FILL, edge=AMBER_EDGE, title_col=AMBER_TEXT)

    r_box = FancyBboxPatch((0.365, 0.36), 0.30, 0.41, boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor='#FFFFFF', edgecolor=AMBER_EDGE, lw=1.0, zorder=2)
    ax.add_patch(r_box)

    ax.text(0.380, 0.730, "Region Affinity Graph:", fontsize=7.4, fontweight='bold', color=AMBER_TEXT)
    ax.text(0.380, 0.680, r"$A^r = \frac{Q^r (K^r)^T}{\sqrt{C}}$", fontsize=7.4, color=C_TEXT)
    ax.text(0.380, 0.635, r"Coarse adjacency: $\mathbf{A}^r \in \mathbb{R}^{S^2 \times S^2}$", fontsize=6.8, color=C_MUTED)

    ax.text(0.380, 0.575, "Dynamic Top-k Routing (k=4):", fontsize=7.4, fontweight='bold', color=RED_TEXT)
    ax.text(0.380, 0.530, r"$\mathrm{route\_idx} = \operatorname{topk}(A^r, k=4)$", fontsize=7.2, color=C_TEXT)

    ax.text(0.380, 0.470, "Prunes ~80% Background:", fontsize=7.4, fontweight='bold', color=GREEN_TEXT)
    ax.text(0.380, 0.430, r"$\bullet$ Sky, floor, and blank walls are pruned.", fontsize=6.6, color=C_TEXT)
    ax.text(0.380, 0.395, r"$\bullet$ Only salient worker regions routed.", fontsize=6.6, color=GREEN_TEXT)

    r_sub_box = FancyBboxPatch((0.365, 0.195), 0.30, 0.14, boxstyle="round,pad=0,rounding_size=0.008",
                               facecolor='#FFFFFF', edgecolor=AMBER_EDGE, lw=0.9, zorder=2)
    ax.add_patch(r_sub_box)
    ax.text(0.515, 0.290, "Selective Region Preservation", fontsize=7.0, fontweight='bold', color=AMBER_TEXT, ha='center', zorder=3)
    ax.text(0.515, 0.250, "Directs 100% compute to salient helmet targets.", fontsize=6.5, color=C_TEXT, ha='center', zorder=3)
    ax.text(0.515, 0.215, "Verified in custom_ablation_modules.py (L165-230)", fontsize=6.2, color=C_MUTED, ha='center', zorder=3)

    draw_arrow(ax, (0.68, 0.53), (0.70, 0.53), lw=1.4, color=GREEN_EDGE)

    # Step 3: Fine Token-to-Token Attention
    draw_panel(ax, (0.70, 0.16), 0.27, 0.71, "3. Fine Token Attention", fill=GREEN_FILL, edge=GREEN_EDGE, title_col=GREEN_TEXT)

    t_box = FancyBboxPatch((0.715, 0.36), 0.24, 0.41, boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor='#FFFFFF', edgecolor=GREEN_EDGE, lw=1.0, zorder=2)
    ax.add_patch(t_box)

    ax.text(0.730, 0.730, "Token-Level Attention:", fontsize=7.4, fontweight='bold', color=GREEN_TEXT)
    ax.text(0.730, 0.670, r"$\mathrm{Attn}(Q, K_{\mathrm{sel}}, V_{\mathrm{sel}}) = \mathrm{Softmax}\left(\frac{Q K_{\mathrm{sel}}^T}{\sqrt{d_k}}\right) V_{\mathrm{sel}}$", fontsize=6.8, color=C_TEXT)

    ax.text(0.730, 0.585, "Strictly Linear Complexity:", fontsize=7.4, fontweight='bold', color=GREEN_TEXT)
    ax.text(0.730, 0.540, r"$\mathcal{O}\left(S^2 + k \cdot \frac{HW}{S^2}\right) \approx \mathbf{\mathcal{O}(HW)}$", fontsize=7.2, color=GREEN_TEXT)

    ax.text(0.730, 0.470, "Residual Projection:", fontsize=7.0, fontweight='bold', color=C_DARK)
    ax.text(0.730, 0.430, r"Residual: $X + \mathrm{Proj}(Y)$", fontsize=6.6, color=C_MUTED)
    ax.text(0.730, 0.395, r"$\mathrm{Norm} = \mathrm{BatchNorm2d}(C)$", fontsize=6.4, color=C_MUTED)

    t_sub_box = FancyBboxPatch((0.715, 0.195), 0.24, 0.14, boxstyle="round,pad=0,rounding_size=0.008",
                               facecolor='#FFFFFF', edgecolor=GREEN_EDGE, lw=0.9, zorder=2)
    ax.add_patch(t_sub_box)
    ax.text(0.835, 0.290, "High-Res Edge Viability", fontsize=7.0, fontweight='bold', color=GREEN_TEXT, ha='center', zorder=3)
    ax.text(0.835, 0.250, "Zero VRAM overflow at 1024×1024.", fontsize=6.5, color=C_TEXT, ha='center', zorder=3)
    ax.text(0.835, 0.215, "Helmet Recall: 91.33% (+0.98% gain).", fontsize=6.4, fontweight='bold', color=GREEN_TEXT, ha='center', zorder=3)

    # Bottom Badge
    draw_badge(ax, (0.03, 0.04), 0.94, 0.09,
               "Empirical Outcome: Elevates Helmet Recall to 91.33% (+0.98%) while operating at high resolution (1024×1024) with zero VRAM overflow.",
               "Complexity Bound: Strictly linear O(HW) execution replaces quadratic bottleneck | custom_ablation_modules.py (Lines 165-230).",
               fill=AMBER_FILL, edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.2)

    save_fig(fig, "Fig3B_Proposed_BiFormer_Sparse_Routing.png")


# =====================================================================
# FIGURE 4A: BASELINE CIOU VANISHING GRADIENT
# =====================================================================
def generate_fig4a():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 4A", "Baseline: CIoU Aspect Ratio Penalty & Vanishing Gradient Failure",
               "When predicted and ground-truth bounding boxes share identical aspect ratios, the CIoU shape gradient vanishes to zero")

    # Panel 1: Geometric Ambiguity
    draw_panel(ax, (0.04, 0.16), 0.41, 0.71, "1. Aspect Ratio Ambiguity in CIoU Loss", fill=GRAY_FILL, edge=C_BORDER)

    # Frame
    b_frame = Rectangle((0.07, 0.34), 0.35, 0.43, facecolor='#FFFFFF', edgecolor=C_BORDER, lw=1.0, zorder=2)
    ax.add_patch(b_frame)

    # Ground Truth Box: w=20, h=40
    gt_box = Rectangle((0.11, 0.45), 0.08, 0.18, facecolor=BLUE_FILL, edgecolor=BLUE_EDGE, lw=1.5, zorder=4)
    ax.add_patch(gt_box)
    ax.text(0.15, 0.54, "GT Box\n" + r"$w^{gt}=20$" + "\n" + r"$h^{gt}=40$" + "\n" + r"$\frac{w^{gt}}{h^{gt}} = 0.5$",
            fontsize=6.8, fontweight='bold', color=BLUE_TEXT, ha='center', va='center', zorder=5)

    # Predicted Box: w=40, h=80 (2x larger)
    pr_box = Rectangle((0.220, 0.38), 0.170, 0.32, facecolor=AMBER_FILL, edgecolor=AMBER_EDGE, lw=1.5, ls='--', zorder=3)
    ax.add_patch(pr_box)
    ax.text(0.305, 0.54, "Predicted Box (2× Larger)\n" + r"$w=40, \ h=80$" + "\n" + r"$\frac{w}{h} = 0.5$",
            fontsize=6.3, fontweight='bold', color=AMBER_TEXT, ha='center', va='center', zorder=5)

    # Structured Contradiction Card
    c_box = FancyBboxPatch((0.07, 0.185), 0.35, 0.13, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFF1F2', edgecolor=RED_EDGE, lw=1.0, zorder=2)
    ax.add_patch(c_box)
    ax.text(0.085, 0.280, "Geometric Contradiction in CIoU:", fontsize=7.2, fontweight='bold', color=RED_TEXT, zorder=3)
    ax.text(0.085, 0.245, r"$\bullet$ Dimensions incorrect: $\Delta w = 20, \Delta h = 40$ (100% error)", fontsize=6.7, color=C_TEXT, zorder=3)
    ax.text(0.085, 0.210, r"$\bullet$ Aspect ratios identical: $\frac{w}{h} = \frac{w^{gt}}{h^{gt}} = 0.5$!", fontsize=6.7, fontweight='bold', color=RED_TEXT, zorder=3)

    # Panel 2: Mathematical Proof of Gradient Vanishing
    draw_panel(ax, (0.47, 0.16), 0.49, 0.71, "2. Analytical Proof of Gradient Vanishing", fill=RED_FILL, edge=RED_EDGE, title_col=RED_TEXT)

    m_box = FancyBboxPatch((0.485, 0.365), 0.46, 0.41, boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor='#FFFFFF', edgecolor=RED_EDGE, lw=1.0, zorder=2)
    ax.add_patch(m_box)

    ax.text(0.50, 0.730, "CIoU Shape Penalty Formulation:", fontsize=7.4, fontweight='bold', color=C_DARK)
    ax.text(0.50, 0.685, r"$v = \frac{4}{\pi^2} \left( \arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h} \right)^2$", fontsize=7.6, color=C_TEXT)

    ax.text(0.50, 0.620, "Partial Derivative with respect to width w:", fontsize=7.4, fontweight='bold', color=C_DARK)
    ax.text(0.50, 0.575, r"$\frac{\partial v}{\partial w} = \frac{8}{\pi^2} \left( \arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h} \right) \cdot \frac{h}{w^2 + h^2}$", fontsize=7.4, color=C_TEXT)

    ax.text(0.50, 0.500, r"When Aspect Ratios Match $\left( \frac{w}{h} = \frac{w^{gt}}{h^{gt}} \right)$:", fontsize=7.4, fontweight='bold', color=RED_TEXT)
    ax.text(0.50, 0.455, r"$\left( \arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h} \right) = 0$", fontsize=7.6, color=RED_TEXT)
    ax.text(0.50, 0.405, r"$\rightarrow \quad \frac{\partial v}{\partial w} = 0 \quad \text{and} \quad \frac{\partial v}{\partial h} = 0$", fontsize=8.2, fontweight='bold', color=RED_TEXT)

    # Regression Failure callout safely separated below
    f_box = FancyBboxPatch((0.485, 0.180), 0.46, 0.165, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=RED_EDGE, lw=0.9, zorder=2)
    ax.add_patch(f_box)
    ax.text(0.50, 0.310, "Fatal Regression Failure:", fontsize=7.2, fontweight='bold', color=RED_TEXT, zorder=3)
    ax.text(0.50, 0.275, r"$\bullet$ Shape gradient collapses strictly to ZERO despite large scale error.", fontsize=6.7, color=C_TEXT, zorder=3)
    ax.text(0.50, 0.240, r"$\bullet$ Cannot guide bounding box to shrink down for tiny helmets (<20px).", fontsize=6.7, color=C_TEXT, zorder=3)
    ax.text(0.50, 0.205, r"$\bullet$ Dominant body instances (111,514) drown out helmet regression.", fontsize=6.7, color=C_MUTED, zorder=3)

    # Bottom Badge
    draw_badge(ax, (0.04, 0.04), 0.92, 0.09,
               "Theoretical Flaw: CIoU loss cannot penalize bounding box dimension errors whenever relative aspect ratios match.",
               "Surveillance Impact: Tiny distant helmets (<20px) fail to scale down, causing severe false negatives under scaffold occlusion.",
               fill=RED_FILL, edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.2)

    save_fig(fig, "Fig4A_Baseline_CIoU_Vanishing_Gradient.png")


# =====================================================================
# FIGURE 4B: PROPOSED FOCAL EIOU DECOMPOSITION
# =====================================================================
def generate_fig4b():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 4B", "Proposed: Focal-EIoU & Multi-Component Geometric Bounding Box Loss",
               "Decoupled edge length penalties guarantee non-zero gradients; Focal weighting and NWD amplify tiny helmet supervision")

    # Panel 1: Geometric Bbox Decomposition (EIoU)
    draw_panel(ax, (0.03, 0.16), 0.38, 0.71, "1. EIoU Geometric Decomposition", fill=BLUE_FILL, edge=BLUE_EDGE, title_col=BLUE_TEXT)

    # Bbox illustration inside
    bb_frame = Rectangle((0.05, 0.46), 0.34, 0.32, facecolor='#FFFFFF', edgecolor=C_BORDER, lw=0.9, zorder=2)
    ax.add_patch(bb_frame)

    # Enclosing Box Cw x Ch
    enc = Rectangle((0.08, 0.49), 0.26, 0.25, facecolor='none', edgecolor=C_LINE, lw=1.0, ls=':', zorder=3)
    ax.add_patch(enc)
    ax.text(0.21, 0.75, r"Enclosing Box: $C_w \times C_h$, Diagonal $c$", fontsize=6.8, color=C_MUTED, ha='center')

    # GT Box
    gt_box = Rectangle((0.11, 0.52), 0.09, 0.15, facecolor=BLUE_FILL, edgecolor=BLUE_EDGE, lw=1.4, zorder=4)
    ax.add_patch(gt_box)
    ax.text(0.14, 0.595, r"$b^{gt}$", fontsize=7.8, fontweight='bold', color=BLUE_TEXT, ha='center', va='center', zorder=5)

    # Pred Box
    pr_box = Rectangle((0.17, 0.55), 0.14, 0.16, facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, lw=1.4, ls='--', zorder=4)
    ax.add_patch(pr_box)
    ax.text(0.25, 0.63, r"$b$", fontsize=7.8, fontweight='bold', color=GREEN_TEXT, ha='center', va='center', zorder=5)

    # Center distance vector
    ax.plot([0.155, 0.24], [0.595, 0.63], color=RED_EDGE, lw=1.2, marker='o', markersize=3, zorder=5)
    ax.text(0.20, 0.625, r"$\rho(b, b^{gt})$", fontsize=6.8, color=RED_TEXT)

    # Equations inside lower card
    e_box = FancyBboxPatch((0.045, 0.185), 0.35, 0.25, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=BLUE_EDGE, lw=0.9, zorder=2)
    ax.add_patch(e_box)
    ax.text(0.060, 0.400, r"$\mathcal{L}_{\mathrm{EIoU}} = \mathcal{L}_{\mathrm{IoU}} + \mathcal{L}_{\mathrm{dis}} + \mathcal{L}_{\mathrm{asp}}$", fontsize=7.2, fontweight='bold', color=C_DARK, zorder=3)
    ax.text(0.060, 0.360, r"$= (1 - \mathrm{IoU}) + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \frac{(w - w^{gt})^2}{C_w^2} + \frac{(h - h^{gt})^2}{C_h^2}$", fontsize=6.3, color=C_TEXT, zorder=3)
    ax.text(0.060, 0.305, "Gradient Non-Vanishing Guarantee:", fontsize=7.0, fontweight='bold', color=GREEN_TEXT, zorder=3)
    ax.text(0.060, 0.265, r"$\frac{\partial \mathcal{L}}{\partial w} = \frac{2(w - w^{gt})}{C_w^2} \neq 0, \quad \frac{\partial \mathcal{L}}{\partial h} = \frac{2(h - h^{gt})}{C_h^2} \neq 0$", fontsize=6.6, color=GREEN_TEXT, zorder=3)
    ax.text(0.060, 0.215, r"Non-zero gradients even if $\frac{w}{h} = \frac{w^{gt}}{h^{gt}}$ (fixes CIoU).", fontsize=6.2, color=C_MUTED, zorder=3)

    # Panel 2: Focal Hard-Sample Mining
    draw_panel(ax, (0.43, 0.16), 0.26, 0.71, "2. Focal Hard-Sample Mining", fill=AMBER_FILL, edge=AMBER_EDGE, title_col=AMBER_TEXT)

    f_box = FancyBboxPatch((0.445, 0.46), 0.23, 0.31, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=AMBER_EDGE, lw=0.9, zorder=2)
    ax.add_patch(f_box)

    ax.text(0.460, 0.730, "Focal Hard-Sample Mining:", fontsize=7.2, fontweight='bold', color=AMBER_TEXT, zorder=3)
    ax.text(0.460, 0.685, r"$\mathcal{L}_{\mathrm{focal}} = \mathrm{IoU}^\gamma \cdot \mathcal{L}_{\mathrm{EIoU}}$", fontsize=7.2, color=C_TEXT, zorder=3)
    ax.text(0.460, 0.645, r"Focal exponent: $\gamma = 0.5$", fontsize=6.8, color=AMBER_TEXT, zorder=3)

    ax.text(0.460, 0.590, "Adaptive Gradient Weighting:", fontsize=7.2, fontweight='bold', color=C_DARK, zorder=3)
    ax.text(0.460, 0.550, r"$\bullet$ Dynamically scales hard losses", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.460, 0.515, r"$\bullet$ Recovers occluded helmets", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.460, 0.480, r"$\bullet$ Mitigates easy-negative dominance", fontsize=6.6, color=C_MUTED, zorder=3)

    f_sub = FancyBboxPatch((0.445, 0.185), 0.23, 0.25, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=AMBER_EDGE, lw=0.9, zorder=2)
    ax.add_patch(f_sub)
    ax.text(0.460, 0.395, "Surveillance Imbalance Fix:", fontsize=7.2, fontweight='bold', color=AMBER_TEXT, zorder=3)
    ax.text(0.460, 0.355, r"$\bullet$ 111,514 bodies vs 9,044 helmets", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.460, 0.320, r"$\bullet$ 1:12 class imbalance resolved", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.460, 0.285, r"$\bullet$ Prevents sample drowning", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.460, 0.250, r"$\bullet$ Boosts helmet recall to 91.33%", fontsize=6.6, color=AMBER_TEXT, zorder=3)
    ax.text(0.460, 0.205, "custom_ablation_modules.py", fontsize=6.0, color=C_MUTED, zorder=3)

    # Panel 3: Integrated Gaussian NWD Metric
    draw_panel(ax, (0.71, 0.16), 0.26, 0.71, "3. Gaussian NWD Metric", fill=GREEN_FILL, edge=GREEN_EDGE, title_col=GREEN_TEXT)

    n_box = FancyBboxPatch((0.725, 0.46), 0.23, 0.31, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=GREEN_EDGE, lw=0.9, zorder=2)
    ax.add_patch(n_box)

    ax.text(0.740, 0.730, "Gaussian Bounding Box Model:", fontsize=7.2, fontweight='bold', color=GREEN_TEXT, zorder=3)
    ax.text(0.740, 0.685, r"$\mathbf{b} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$ for boxes $<20\mathrm{px}$", fontsize=6.8, color=C_TEXT, zorder=3)
    ax.text(0.740, 0.645, r"$\mathbf{\Sigma} = \mathrm{diag}(w^2/4, \ h^2/4)$", fontsize=6.8, color=C_TEXT, zorder=3)

    ax.text(0.740, 0.590, "Wasserstein Distance Metric:", fontsize=7.2, fontweight='bold', color=C_DARK, zorder=3)
    ax.text(0.740, 0.545, r"$\mathrm{NWD} = \exp\left(-\frac{\sqrt{W_2^2}}{C}\right)$", fontsize=7.2, color=C_TEXT, zorder=3)
    ax.text(0.740, 0.485, "Continuous metric for micro targets", fontsize=6.5, color=GREEN_TEXT, zorder=3)

    n_sub = FancyBboxPatch((0.725, 0.185), 0.23, 0.25, boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor='#FFFFFF', edgecolor=GREEN_EDGE, lw=0.9, zorder=2)
    ax.add_patch(n_sub)
    ax.text(0.740, 0.395, "Tiny Target Gradient Stability:", fontsize=7.2, fontweight='bold', color=GREEN_TEXT, zorder=3)
    ax.text(0.740, 0.355, r"$\bullet$ Discrete IoU collapses on 1-px shift", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.740, 0.320, r"$\bullet$ NWD yields smooth gradients", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.740, 0.285, r"$\bullet$ Bypasses DDP sync barriers", fontsize=6.6, color=C_TEXT, zorder=3)
    ax.text(0.740, 0.250, r"$\bullet$ mAP50: 94.88% (Ablation A4)", fontsize=6.6, fontweight='bold', color=GREEN_TEXT, zorder=3)
    ax.text(0.740, 0.205, "Code lines 239-298", fontsize=6.0, color=C_MUTED, zorder=3)

    # Bottom Badge
    draw_badge(ax, (0.03, 0.04), 0.94, 0.09,
               "Outcome: Prevents 111,514 body gradients from drowning 9,044 helmet gradients (Solves 1:12 Imbalance).",
               "Regression Milestone: Increases mAP50 to 94.88% (Ablation A4) and mAP50-95 to 62.51% | custom_ablation_modules.py (Lines 239-298).",
               fill=BLUE_FILL, edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)

    save_fig(fig, "Fig4B_Proposed_Focal_EIoU_Decomposition.png")


# =====================================================================
# FIGURE 5A: BASELINE YOLO11s ARCHITECTURE
# =====================================================================
def generate_fig5a():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 5A", "Baseline: Vanilla YOLO11s Architecture (Standard 3-Scale Pipeline)",
               "Standard CSPDarknet backbone and PAFPN neck limited to 3 detection scales (Stride 8, 16, 32); Missing high-res P2 feature map")

    # 1. Input
    draw_cube(ax, 0.045, 0.44, 0.035, 0.18, 0.04, fill=BLUE_FILL, edge=BLUE_EDGE, label="RGB", dims=r"$3 \times 640 \times 640$")
    draw_arrow(ax, (0.105, 0.53), (0.135, 0.53))

    # 2. Backbone
    draw_panel(ax, (0.135, 0.16), 0.250, 0.71, "CSPDarknet Backbone", fill=GRAY_FILL, edge=C_BORDER)
    draw_block(ax, (0.145, 0.73), 0.225, 0.055, "Conv 3×3 (s=2)", r"$\text{P1/2: } 64 \times 320 \times 320$", fill='#FFFFFF', edge=GRAY_EDGE, text_col=GRAY_TEXT, fontsize=7.0)
    draw_block(ax, (0.145, 0.63), 0.225, 0.055, "C3k2 (P2/4)", r"$128 \times 160 \times 160$", fill='#FFFFFF', edge=GRAY_EDGE, text_col=GRAY_TEXT, fontsize=7.0)
    draw_block(ax, (0.145, 0.53), 0.225, 0.055, "C3k2 (P3/8)", r"$256 \times 80 \times 80$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)
    draw_block(ax, (0.145, 0.43), 0.225, 0.055, "C3k2 (P4/16)", r"$512 \times 40 \times 40$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)
    draw_block(ax, (0.145, 0.33), 0.225, 0.055, "C3k2 + SPPF (P5/32)", r"$512 \times 20 \times 20$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)

    # Connections to Neck (centered in gap [0.385, 0.435])
    draw_arrow(ax, (0.370, 0.557), (0.445, 0.67), label="P3", text_offset=(0, 0.01))
    draw_arrow(ax, (0.370, 0.457), (0.445, 0.53), label="P4", text_offset=(0, 0.01))
    draw_arrow(ax, (0.370, 0.357), (0.445, 0.39), label="P5", text_offset=(0, 0.005))

    # 3. Neck (PAFPN)
    draw_panel(ax, (0.435, 0.16), 0.255, 0.71, "PAFPN Neck (3 Scales)", fill='#FFFFFF', edge=C_BORDER)
    draw_block(ax, (0.450, 0.64), 0.225, 0.065, "P3 Fusion", "Top-Down + Bottom-Up", fill=BLUE_FILL, edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)
    draw_block(ax, (0.450, 0.50), 0.225, 0.065, "P4 Fusion", "Top-Down + Bottom-Up", fill=BLUE_FILL, edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)
    draw_block(ax, (0.450, 0.36), 0.225, 0.065, "P5 Fusion", "Top-Down + Bottom-Up", fill=BLUE_FILL, edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)

    ax.text(0.562, 0.24, "Standard feature pyramid\nwithout micro-scale P2", fontsize=6.8, color=RED_TEXT, ha='center')

    # Connections to Heads
    draw_arrow(ax, (0.675, 0.67), (0.720, 0.67))
    draw_arrow(ax, (0.675, 0.53), (0.720, 0.53))
    draw_arrow(ax, (0.675, 0.39), (0.720, 0.39))

    # 4. Heads & Loss
    draw_panel(ax, (0.720, 0.16), 0.245, 0.71, "3 Decoupled Heads & Loss", fill=GRAY_FILL, edge=C_BORDER)
    draw_block(ax, (0.735, 0.64), 0.21, 0.065, "Head P3 (Small)", r"Stride 8 ($80 \times 80$)", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)
    draw_block(ax, (0.735, 0.50), 0.21, 0.065, "Head P4 (Medium)", r"Stride 16 ($40 \times 40$)", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)
    draw_block(ax, (0.735, 0.36), 0.21, 0.065, "Head P5 (Large)", r"Stride 32 ($20 \times 20$)", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.2)

    draw_block(ax, (0.735, 0.22), 0.21, 0.055, "Loss: CIoU + DFL", "Standard Supervision", fill=RED_FILL, edge=RED_EDGE, text_col=RED_TEXT, fontsize=7.0)

    # Bottom Badge
    draw_badge(ax, (0.03, 0.04), 0.94, 0.09,
               "Baseline Deficit: Lack of high-resolution P2 micro-scale detection head causes target vanishing for tiny distant helmets (<20 px).",
               "Scale Limitation: Downsampling by 32x erases spatial cues required to resolve occluded PPE beneath construction scaffolding.",
               fill=GRAY_FILL, edge=GRAY_EDGE, text_col=GRAY_TEXT, fontsize=7.2)

    save_fig(fig, "Fig5A_Baseline_YOLO11s_Architecture.png")


# =====================================================================
# FIGURE 5B: PROPOSED REP-YOLO11s ARCHITECTURE
# =====================================================================
def generate_fig5b():
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    add_header(ax, "Fig. 5B", "Proposed: Rep-YOLO11s-P2 AFPN Comprehensive Architecture",
               "High-resolution 4-head detector with CoordConv stem, RepConv & BiFormer backbone, AFPN neck, and composite Focal-EIoU loss")

    # 1. Input & CoordConv Stem
    draw_panel(ax, (0.02, 0.16), 0.165, 0.71, "1. CoordConv Stem", fill=TEAL_FILL, edge=TEAL_EDGE, title_col=TEAL_TEXT)
    draw_cube(ax, 0.050, 0.68, 0.030, 0.12, 0.035, fill=BLUE_FILL, edge=BLUE_EDGE, label="RGB", dims=r"$3 \times 1024 \times 1024$")
    draw_arrow(ax, (0.095, 0.67), (0.095, 0.59))
    draw_block(ax, (0.028, 0.52), 0.148, 0.065, "AddCoords", r"$+ C_x, C_y \in [-1, 1]$", fill='#FFFFFF', edge=TEAL_EDGE, text_col=TEAL_TEXT, fontsize=7.0)
    draw_arrow(ax, (0.102, 0.52), (0.102, 0.445))
    draw_block(ax, (0.028, 0.365), 0.148, 0.075, "CoordConv Stem", r"$\mathrm{Conv}(5 \to 64, s=2)$" + "\n" + r"$P_1/2: 512 \times 512$", fill='#FFFFFF', edge=TEAL_EDGE, text_col=TEAL_TEXT, fontsize=6.6)

    # Note card in Panel 1
    stem_note = FancyBboxPatch((0.028, 0.195), 0.148, 0.14, boxstyle="round,pad=0,rounding_size=0.008",
                               facecolor='#FFFFFF', edgecolor=TEAL_EDGE, lw=0.9, zorder=2)
    ax.add_patch(stem_note)
    ax.text(0.102, 0.295, "Cartesian Spatial Prior", fontsize=6.8, fontweight='bold', color=TEAL_TEXT, ha='center', zorder=3)
    ax.text(0.102, 0.252, "Encodes elevation priors\nto eliminate false alarms.", fontsize=6.3, color=C_TEXT, ha='center', zorder=3)
    ax.text(0.102, 0.215, "L46-69 in custom modules", fontsize=5.8, color=C_MUTED, ha='center', zorder=3)

    # Connection arrow from Stem output to Backbone Stage 1 (P2)
    draw_arrow(ax, (0.176, 0.4025), (0.230, 0.7425), color=TEAL_EDGE, lw=1.3, rad=-0.18, label="P1/2", label_pos=0.55, text_offset=(0.0, 0.015))

    # 2. Backbone
    draw_panel(ax, (0.220, 0.16), 0.220, 0.71, "2. Rep-BiFormer Backbone", fill=BLUE_FILL, edge=BLUE_EDGE, title_col=BLUE_TEXT)
    draw_block(ax, (0.230, 0.7125), 0.195, 0.060, "C3k2 (RepConv) P2/4", r"$128 \to 256 \quad (256 \times 256)$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)
    draw_block(ax, (0.230, 0.6125), 0.195, 0.060, "C3k2 (RepConv) P3/8", r"$256 \to 256 \quad (128 \times 128)$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)
    draw_block(ax, (0.230, 0.5075), 0.195, 0.060, "C3k2 + BiFormer P4/16", r"$256 \to 512 \quad (64 \times 64)$", fill='#FFFFFF', edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.0)
    draw_block(ax, (0.230, 0.4025), 0.195, 0.060, "C3k2 + BiFormer P5/32", r"$512 \to 512 \quad (32 \times 32)$", fill='#FFFFFF', edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.0)
    draw_block(ax, (0.230, 0.290), 0.195, 0.055, "SPPF Block (P5/32)", r"$512 \times 32 \times 32$", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)

    # Connections to Neck (centered in gap [0.440, 0.475])
    draw_arrow(ax, (0.425, 0.7425), (0.485, 0.7425), label="P2", text_offset=(0, 0.012))
    draw_arrow(ax, (0.425, 0.6425), (0.485, 0.6425), label="P3", text_offset=(0, 0.012))
    draw_arrow(ax, (0.425, 0.5375), (0.485, 0.5375), label="P4", text_offset=(0, 0.012))
    draw_arrow(ax, (0.425, 0.4325), (0.485, 0.4325), label="P5", text_offset=(0, 0.012))

    # 3. AFPN Neck (4 Scales)
    draw_panel(ax, (0.475, 0.16), 0.235, 0.71, "3. AFPN Neck (4 Scales)", fill=GREEN_FILL, edge=GREEN_EDGE, title_col=GREEN_TEXT)
    draw_block(ax, (0.485, 0.710), 0.210, 0.062, "P2 Asymptotic Fusion", "DySample + Backbone P2", fill='#FFFFFF', edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=6.8)
    draw_block(ax, (0.485, 0.610), 0.210, 0.062, "P3 Asymptotic Fusion", "Upsample + Backbone P3", fill='#FFFFFF', edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=6.8)
    draw_block(ax, (0.485, 0.505), 0.210, 0.062, "P4 Asymptotic Fusion", "Upsample + Backbone P4", fill='#FFFFFF', edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=6.8)
    draw_block(ax, (0.485, 0.400), 0.210, 0.062, "P5 Asymptotic Fusion", "Downsample + Backbone P5", fill='#FFFFFF', edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=6.8)

    ax.text(0.592, 0.250, "Preserves micro-scale P2 features\nfor tiny occluded helmets.", fontsize=6.6, color=GREEN_TEXT, ha='center')

    # Connections to Heads
    draw_arrow(ax, (0.695, 0.7425), (0.745, 0.7425))
    draw_arrow(ax, (0.695, 0.6425), (0.745, 0.6425))
    draw_arrow(ax, (0.695, 0.5375), (0.745, 0.5375))
    draw_arrow(ax, (0.695, 0.4325), (0.745, 0.4325))

    # 4. 4 Decoupled Heads & Supervision
    draw_panel(ax, (0.73, 0.16), 0.25, 0.71, "4. 4 Heads & Supervision", fill='#FFFFFF', edge=C_BORDER)
    draw_block(ax, (0.745, 0.7125), 0.22, 0.060, "P2 Micro Head (Specialist)", r"Stride 4 ($256 \times 256$)", fill=AMBER_FILL, edge=AMBER_EDGE, text_col=AMBER_TEXT, fontsize=7.0)
    draw_block(ax, (0.745, 0.6050), 0.22, 0.060, "P3 Small Head", r"Stride 8 ($128 \times 128$)", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)
    draw_block(ax, (0.745, 0.5000), 0.22, 0.060, "P4 Medium Head", r"Stride 16 ($64 \times 64$)", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)
    draw_block(ax, (0.745, 0.3950), 0.22, 0.060, "P5 Large Head", r"Stride 32 ($32 \times 32$)", fill='#FFFFFF', edge=BLUE_EDGE, text_col=BLUE_TEXT, fontsize=7.0)

    # Dedicated Composite Supervision Card
    sup_box = FancyBboxPatch((0.745, 0.185), 0.22, 0.18, boxstyle="round,pad=0,rounding_size=0.010",
                             facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, lw=1.2, zorder=3)
    ax.add_patch(sup_box)
    ax.text(0.855, 0.332, "Composite Supervision", fontsize=7.2, fontweight='bold', color=GREEN_TEXT, ha='center', zorder=4)
    ax.text(0.855, 0.292, r"$\mathcal{L}_{\mathrm{box}} = \mathrm{IoU}^{0.5} \cdot \mathcal{L}_{\mathrm{EIoU}} + \mathcal{L}_{\mathrm{NWD}}$", fontsize=6.7, color=C_TEXT, ha='center', zorder=4)
    ax.text(0.855, 0.252, r"$\mathcal{L}_{\mathrm{cls}}:$ Alpha-Balanced Focal BCE", fontsize=6.6, color=C_TEXT, ha='center', zorder=4)
    ax.text(0.855, 0.212, r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{box}} + \mathcal{L}_{\mathrm{cls}} + \mathcal{L}_{\mathrm{DFL}}$", fontsize=6.8, fontweight='bold', color=GREEN_TEXT, ha='center', zorder=4)

    # Bottom Badge
    draw_badge(ax, (0.02, 0.04), 0.96, 0.09,
               "Peak Metrics: 97.11% mAP50 · 342.5 FPS TensorRT FP16 (2.92 ms on T4) · 91.33% Helmet Recall.",
               "Architectural Rigor: 100% grounded in rep_yolo11s_p2.yaml with 4 detection heads and zero VRAM overflow at 1024×1024.",
               fill=GREEN_FILL, edge=GREEN_EDGE, text_col=GREEN_TEXT, fontsize=7.2)

    save_fig(fig, "Fig5B_Proposed_RepYOLO11s_Architecture.png")


# =====================================================================
# MAIN RUNNER
# =====================================================================
if __name__ == '__main__':
    print("Generating 10 publication-grade Q1 architectural diagrams...")
    generate_fig1a()
    generate_fig1b()
    generate_fig2a()
    generate_fig2b()
    generate_fig3a()
    generate_fig3b()
    generate_fig4a()
    generate_fig4b()
    generate_fig5a()
    generate_fig5b()
    print("ALL 10 PUBLICATION FIGURES RENDERED AND EXPORTED SUCCESSFULLY!")
