"""
Generate IEEE Q1 Publication-Quality System Pipeline Diagram for Rep-YOLO11s.
Aligned with empirical benchmarks (TensorRT FP16, 2.92 ms T4 / 5.35 ms RTX 3050, 65-95 FPS E2E).
Outputs both 300 DPI PNG and Vector PDF in paper_overleaf/figures and root directory.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MPath

def create_pipeline_diagram():
    # Set high-DPI and IEEE standard font family
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    fig, ax = plt.subplots(figsize=(15.5, 7.2), dpi=300)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.2)
    ax.axis('off')

    # Color Palette - Professional IEEE Journal Theme
    c_bg = '#FFFFFF'
    c_card_bg = '#F8FAFC'
    c_primary = '#1E3A8A'     # Navy Blue
    c_accent = '#0D9488'      # Teal / Cyan
    c_warning = '#D97706'     # Amber
    c_dark = '#0F172A'        # Slate 900
    c_text_muted = '#475569'  # Slate 600
    c_border = '#CBD5E1'      # Slate 300
    c_innov = '#ECFDF5'       # Mint Green for our innovations
    c_innov_border = '#10B981'# Emerald

    fig.patch.set_facecolor(c_bg)

    # 1. Main Title & Subtitle Banner
    title_box = patches.FancyBboxPatch(
        (0.4, 6.35), 14.7, 0.70,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor='#1E293B', edgecolor='none'
    )
    ax.add_patch(title_box)
    ax.text(7.75, 6.78, "END-TO-END INDUSTRIAL PPE COMPLIANCE & SAFETY HELMET SURVEILLANCE PIPELINE",
            ha='center', va='center', color='#FFFFFF', fontsize=11.5, fontweight='bold')
    ax.text(7.75, 6.48, "Real-Time Multi-Stream RTSP Ingestion • CoordConv Spatial Bias • Structural RepConv • TensorRT FP16 Acceleration (342.5 FPS)",
            ha='center', va='center', color='#94A3B8', fontsize=8.5, fontweight='normal')

    # 5 Major Stages Definition
    stages = [
        {
            "id": "STAGE 1",
            "title": "Multi-Stream\nRTSP Ingestion",
            "time": "3.5 - 5.0 ms",
            "x": 0.5, "w": 2.6,
            "items": [
                ("1080p RTSP Camera Streams", False),
                ("Hardware NVDEC Decoding", False),
                ("H.264 / H.265 Bitstream", False),
                ("Ring-Buffer Frame Queue", False),
            ],
            "tensor": "Tensor: (B, 3, 1080, 1920)"
        },
        {
            "id": "STAGE 2",
            "title": "Pre-Processing &\nSpatial Bias",
            "time": "1.2 - 2.0 ms",
            "x": 3.45, "w": 2.6,
            "items": [
                ("Aspect-Ratio Letterbox", False),
                ("Normalization [0, 1]", False),
                ("CoordConv Channel Injection", True), # Novelty
                ("(+2 Channels: h_i, w_j)", True),   # Novelty
            ],
            "tensor": "Tensor: (B, 5, 640, 640)"
        },
        {
            "id": "STAGE 3 (CORE)",
            "title": "Rep-YOLO11s\nTensorRT FP16 Engine",
            "time": "2.92 ms (Tesla T4) / 5.35 ms (RTX 3050)",
            "x": 6.4, "w": 3.2,
            "items": [
                ("Structural RepConv (3x3 Fused)", True),  # Novelty
                ("BiFormer Dynamic Routing Attn", True),   # Novelty
                ("FP16 Half-Precision Engine", True),      # Novelty
                ("switch_to_deploy() Zero Overhead", True),# Novelty
            ],
            "tensor": "Feature: (B, 6, 8400)"
        },
        {
            "id": "STAGE 4",
            "title": "Decoupled Head &\nBox Regression",
            "time": "1.5 - 2.8 ms",
            "x": 9.95, "w": 2.45,
            "items": [
                ("Decoupled Cls & Reg Head", False),
                ("Focal-EIoU Loss Refinement", True), # Novelty
                ("CUDA Class-Wise NMS", False),
                ("Score Threshold Tau >= 0.45", False),
            ],
            "tensor": "Filtered: (N, 6) Bboxes"
        },
        {
            "id": "STAGE 5",
            "title": "Edge Monitoring &\nIndustrial Alerting",
            "time": "2.2 - 3.4 ms",
            "x": 12.75, "w": 2.35,
            "items": [
                ("Bbox & ID Tag Overlay", False),
                ("Violation Beeper Trigger", False),
                ("H.264 WebRTC / RTSP Push", False),
                ("MQTT / Cloud Edge Log", False),
            ],
            "tensor": "Output: 1080p Overlay"
        }
    ]

    # Draw Stage Cards
    for s in stages:
        x, w = s["x"], s["w"]
        is_core = "CORE" in s["id"]
        
        # Outer Card
        card = patches.FancyBboxPatch(
            (x, 0.95), w, 5.15,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor='#F1F5F9' if not is_core else '#EFF6FF',
            edgecolor='#2563EB' if is_core else c_border,
            linewidth=1.8 if is_core else 1.2
        )
        ax.add_patch(card)

        # Stage Header Box
        hdr_box = patches.FancyBboxPatch(
            (x + 0.08, 5.15), w - 0.16, 0.85,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=c_primary if is_core else '#334155',
            edgecolor='none'
        )
        ax.add_patch(hdr_box)

        # Stage Badge
        ax.text(x + w/2, 5.75, s["id"], ha='center', va='center',
                color='#38BDF8' if is_core else '#94A3B8', fontsize=7.5, fontweight='bold')
        ax.text(x + w/2, 5.40, s["title"], ha='center', va='center',
                color='#FFFFFF', fontsize=8.5, fontweight='bold')

        # Processing Items
        y_cursor = 4.70
        for text, is_novel in s["items"]:
            if is_novel:
                item_bg = patches.FancyBboxPatch(
                    (x + 0.12, y_cursor - 0.18), w - 0.24, 0.40,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=c_innov, edgecolor=c_innov_border, linewidth=0.8
                )
                ax.add_patch(item_bg)
                ax.text(x + 0.20, y_cursor + 0.02, "[+] " + text, ha='left', va='center',
                        color='#065F46', fontsize=7.8, fontweight='bold')
            else:
                item_bg = patches.FancyBboxPatch(
                    (x + 0.12, y_cursor - 0.18), w - 0.24, 0.40,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor='#FFFFFF', edgecolor='#E2E8F0', linewidth=0.6
                )
                ax.add_patch(item_bg)
                ax.text(x + 0.20, y_cursor + 0.02, "• " + text, ha='left', va='center',
                        color=c_dark, fontsize=7.8, fontweight='normal')
            y_cursor -= 0.52

        # Tensor Output Badge
        tensor_box = patches.FancyBboxPatch(
            (x + 0.12, 1.85), w - 0.24, 0.45,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor='#1E293B', edgecolor='none'
        )
        ax.add_patch(tensor_box)
        ax.text(x + w/2, 2.07, s["tensor"], ha='center', va='center',
                color='#38BDF8', fontsize=7.5, fontweight='bold')

        # Latency / Metric Box
        time_box = patches.FancyBboxPatch(
            (x + 0.12, 1.15), w - 0.24, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor='#FEF3C7' if is_core else '#F8FAFC',
            edgecolor='#F59E0B' if is_core else '#CBD5E1',
            linewidth=1.0 if is_core else 0.8
        )
        ax.add_patch(time_box)
        ax.text(x + w/2, 1.50, "PROCESSING LATENCY" if not is_core else "PURE INFERENCE SPEED",
                ha='center', va='center', color='#B45309' if is_core else c_text_muted, fontsize=6.8, fontweight='bold')
        ax.text(x + w/2, 1.30, s["time"], ha='center', va='center',
                color='#92400E' if is_core else c_dark, fontsize=8.0, fontweight='bold')

    # Draw Connecting Arrows between Stages
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + stages[i]["w"]
        x2 = stages[i+1]["x"]
        
        # Draw Arrow
        ax.annotate(
            '', xy=(x2 - 0.02, 3.5), xytext=(x1 + 0.02, 3.5),
            arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=2.2, mutation_scale=15)
        )
        ax.annotate(
            '', xy=(x2 - 0.02, 2.07), xytext=(x1 + 0.02, 2.07),
            arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.2, mutation_scale=10, linestyle='--')
        )

    # Bottom Legend / Innovation Bar
    legend_bar = patches.FancyBboxPatch(
        (0.4, 0.18), 14.7, 0.55,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor='#F8FAFC', edgecolor=c_border, linewidth=0.8
    )
    ax.add_patch(legend_bar)
    
    # Legend Elements
    ax.text(0.7, 0.45, "LEGEND & INNOVATION HIGHLIGHTS:", ha='left', va='center',
            color=c_dark, fontsize=8.0, fontweight='bold')
    
    # Tag 1: Proposed Novel Modules
    tag1 = patches.FancyBboxPatch((4.2, 0.28), 3.3, 0.35, boxstyle="round,pad=0.02,rounding_size=0.04",
                                  facecolor=c_innov, edgecolor=c_innov_border, linewidth=0.8)
    ax.add_patch(tag1)
    ax.text(5.85, 0.45, "[*] Proposed Architectural Innovations", ha='center', va='center',
            color='#065F46', fontsize=7.5, fontweight='bold')

    # Tag 2: SOTA Efficiency
    tag2 = patches.FancyBboxPatch((7.7, 0.28), 3.4, 0.35, boxstyle="round,pad=0.02,rounding_size=0.04",
                                  facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=0.8)
    ax.add_patch(tag2)
    ax.text(9.4, 0.45, "[!] GPU Throughput: 342.5 FPS (T4) / 187.1 FPS (RTX 3050)", ha='center', va='center',
            color='#92400E', fontsize=7.2, fontweight='bold')

    # Tag 3: Multi-Stream Throughput
    tag3 = patches.FancyBboxPatch((11.3, 0.28), 3.5, 0.35, boxstyle="round,pad=0.02,rounding_size=0.04",
                                  facecolor='#EFF6FF', edgecolor='#2563EB', linewidth=0.8)
    ax.add_patch(tag3)
    ax.text(13.05, 0.45, "Multi-Camera Pipeline: 65 - 95 FPS (10.5 - 15.4 ms E2E)", ha='center', va='center',
            color='#1E40AF', fontsize=7.2, fontweight='bold')

    plt.tight_layout()

    # Save to both destinations
    out_pdf = Path("paper_overleaf/figures/shwd_system_pipeline.pdf")
    out_png = Path("paper_overleaf/figures/shwd_system_pipeline.png")
    root_pdf = Path("shwd_system_pipeline_ieee.pdf")
    root_png = Path("shwd_system_pipeline_ieee.png")

    fig.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(root_pdf, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(root_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ IEEE Pipeline Diagram successfully generated at:")
    print(f"   - {out_pdf.resolve()}")
    print(f"   - {out_png.resolve()}")
    print(f"   - {root_png.resolve()}")

if __name__ == "__main__":
    create_pipeline_diagram()
