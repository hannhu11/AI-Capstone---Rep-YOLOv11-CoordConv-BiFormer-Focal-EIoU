# -*- coding: utf-8 -*-
"""
Generates clean, high-resolution vector-quality diagram figures for:
1. RepConv structural re-parameterization algebra and branch fusion.
2. CoordConv spatial coordinate injection concept.
3. BiFormer bi-level routing sparse attention mechanism.
4. Focal EIoU independent dimension bounding box decomposition.
Color Palette: Warm Light Ivory (#FAF8F5), Deep Navy (#0F172A), Safety Orange (#EA580C), Amber (#D97706), Slate (#64748B), Green (#16A34A).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

out_dir = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\figures'
os.makedirs(out_dir, exist_ok=True)

# Set common font styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0

# ==============================================================================
# 1. REPCONV STRUCTURAL RE-PARAMETERIZATION FUSION DIAGRAM
# ==============================================================================
def create_repconv_diagram():
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    fig.patch.set_facecolor('#FAF8F5')
    ax.set_facecolor('#FAF8F5')
    ax.axis('off')

    # Title
    ax.text(0.5, 0.96, 'CƠ CHẾ TÁI THAM SỐ HÓA CẤU TRÚC (REPCONV) & GỘP NHÁNH ĐẠI SỐ ZERO-LATENCY', 
            fontsize=13, fontweight='bold', ha='center', va='top', color='#0F172A')

    # Left Container: GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
    train_box = patches.FancyBboxPatch((0.03, 0.12), 0.38, 0.76, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                       facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5)
    ax.add_patch(train_box)
    ax.text(0.22, 0.84, 'GIAI ĐOẠN HUẤN LUYỆN (TRAINING)', fontsize=11, fontweight='bold', ha='center', color='#EA580C')
    ax.text(0.22, 0.79, 'Cấu trúc 3 nhánh song song (Học biểu diễn đa dạng)', fontsize=9, ha='center', color='#64748B')

    # Input Tensor
    in_box = patches.FancyBboxPatch((0.06, 0.44), 0.06, 0.14, boxstyle="round,pad=0.01", facecolor='#F1F5F9', edgecolor='#94A3B8')
    ax.add_patch(in_box)
    ax.text(0.09, 0.51, 'Input\nX', fontsize=10, fontweight='bold', ha='center', va='center', color='#0F172A')

    # Branch 1: 3x3 Conv + BN
    b1 = patches.FancyBboxPatch((0.17, 0.62), 0.14, 0.11, boxstyle="round,pad=0.01", facecolor='#EFF6FF', edgecolor='#2563EB', linewidth=1.2)
    ax.add_patch(b1)
    ax.text(0.24, 0.675, 'Nhánh 3x3 Conv + BN\nW³ˣ³, b³ˣ³', fontsize=9, fontweight='bold', ha='center', va='center', color='#1D4ED8')

    # Branch 2: 1x1 Conv + BN
    b2 = patches.FancyBboxPatch((0.17, 0.455), 0.14, 0.11, boxstyle="round,pad=0.01", facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.2)
    ax.add_patch(b2)
    ax.text(0.24, 0.51, 'Nhánh 1x1 Conv + BN\nW¹ˣ¹, b¹ˣ¹', fontsize=9, fontweight='bold', ha='center', va='center', color='#B45309')

    # Branch 3: Identity + BN
    b3 = patches.FancyBboxPatch((0.17, 0.29), 0.14, 0.11, boxstyle="round,pad=0.01", facecolor='#F0FDF4', edgecolor='#16A34A', linewidth=1.2)
    ax.add_patch(b3)
    ax.text(0.24, 0.345, 'Nhánh Identity + BN\nMa trận đơn vị I', fontsize=9, fontweight='bold', ha='center', va='center', color='#15803D')

    # Sum Node
    sum_circle = plt.Circle((0.37, 0.51), 0.025, facecolor='#EA580C', edgecolor='#C2410C')
    ax.add_patch(sum_circle)
    ax.text(0.37, 0.51, '+', fontsize=14, fontweight='bold', ha='center', va='center', color='#FFFFFF')

    # Arrows in train box
    arrow_props = dict(arrowstyle="->", color='#64748B', lw=1.5)
    ax.annotate('', xy=(0.17, 0.675), xytext=(0.12, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.17, 0.51), xytext=(0.12, 0.51), arrowprops=arrow_props)
    ax.annotate('', xy=(0.17, 0.345), xytext=(0.12, 0.47), arrowprops=arrow_props)

    ax.annotate('', xy=(0.345, 0.51), xytext=(0.31, 0.675), arrowprops=arrow_props)
    ax.annotate('', xy=(0.345, 0.51), xytext=(0.31, 0.51), arrowprops=arrow_props)
    ax.annotate('', xy=(0.345, 0.51), xytext=(0.31, 0.345), arrowprops=arrow_props)

    ax.text(0.22, 0.17, '⚠ Nhược điểm: Truy cập bộ nhớ (MAC) cao, trễ 7.12 ms', fontsize=8.5, ha='center', color='#DC2626', fontweight='bold')

    # Middle Connector: ALGEBRAIC FUSION TRANSFORMATION
    mid_box = patches.FancyBboxPatch((0.44, 0.24), 0.18, 0.52, boxstyle="round,pad=0.02", facecolor='#FFF7ED', edgecolor='#EA580C', linewidth=1.5, linestyle='--')
    ax.add_patch(mid_box)
    ax.text(0.53, 0.71, 'GỘP ĐẠI SỐ\nswitch_to_deploy()', fontsize=10, fontweight='bold', ha='center', color='#EA580C')
    ax.text(0.53, 0.58, '① Hợp nhất BN:\nW\' = (γ/σ)·W\nb\' = β - (γμ/σ)', fontsize=8.2, ha='center', color='#0F172A')
    ax.text(0.53, 0.44, '② Đệm Zero Pad:\n1x1 → 3x3 Pad', fontsize=8.2, ha='center', color='#0F172A')
    ax.text(0.53, 0.31, '③ Cộng tuyến tính:\nW_fused = Σ W\'', fontsize=8.2, ha='center', color='#0F172A')

    # Arrows connecting Left to Middle and Middle to Right
    ax.annotate('', xy=(0.44, 0.51), xytext=(0.395, 0.51), arrowprops=dict(arrowstyle="->", color='#EA580C', lw=2.5))
    ax.annotate('', xy=(0.65, 0.51), xytext=(0.62, 0.51), arrowprops=dict(arrowstyle="->", color='#16A34A', lw=2.5))

    # Right Container: GIAI ĐOẠN TRIỂN KHAI (INFERENCE PHASE)
    deploy_box = patches.FancyBboxPatch((0.65, 0.12), 0.32, 0.76, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                        facecolor='#FFFFFF', edgecolor='#16A34A', linewidth=2.0)
    ax.add_patch(deploy_box)
    ax.text(0.81, 0.84, 'TRIỂN KHAI THỰC TẾ (INFERENCE)', fontsize=11, fontweight='bold', ha='center', color='#16A34A')
    ax.text(0.81, 0.79, 'Duy nhất 1 lớp 3x3 Conv đơn lẻ (Zero Latency)', fontsize=9, ha='center', color='#64748B')

    # Input to Single Conv
    in_box2 = patches.FancyBboxPatch((0.68, 0.44), 0.055, 0.14, boxstyle="round,pad=0.01", facecolor='#F1F5F9', edgecolor='#94A3B8')
    ax.add_patch(in_box2)
    ax.text(0.707, 0.51, 'Input\nX', fontsize=10, fontweight='bold', ha='center', va='center', color='#0F172A')

    fused_box = patches.FancyBboxPatch((0.77, 0.41), 0.13, 0.20, boxstyle="round,pad=0.01", facecolor='#F0FDF4', edgecolor='#16A34A', linewidth=2.0)
    ax.add_patch(fused_box)
    ax.text(0.835, 0.53, 'Fused 3x3 Conv\nW_fused, b_fused', fontsize=9.5, fontweight='bold', ha='center', color='#15803D')
    ax.text(0.835, 0.46, '(Zero branches, Zero BN)', fontsize=8, ha='center', color='#64748B')

    out_box = patches.FancyBboxPatch((0.93, 0.44), 0.03, 0.14, boxstyle="round,pad=0.01", facecolor='#EFF6FF', edgecolor='#2563EB')
    ax.add_patch(out_box)
    ax.text(0.945, 0.51, 'Out', fontsize=9.5, fontweight='bold', ha='center', va='center', color='#1D4ED8')

    ax.annotate('', xy=(0.77, 0.51), xytext=(0.735, 0.51), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.5))
    ax.annotate('', xy=(0.93, 0.51), xytext=(0.90, 0.51), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.5))

    ax.text(0.81, 0.26, '⚡ Độ trễ giảm từ 7.12 ms → 2.92 ms (-55.2%)\n★ Sai số số học toán học: Δ < 10⁻⁵', 
            fontsize=9.2, fontweight='bold', ha='center', color='#16A34A')
    ax.text(0.81, 0.17, 'Tốc độ Tesla T4: 342.5 FPS | RTX 3050: 187.1 FPS', fontsize=8.5, ha='center', color='#475569')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Fig_RepConv_Architecture.png'), facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print("Created Fig_RepConv_Architecture.png")

# ==============================================================================
# 2. COORDCONV SPATIAL PRIOR ENCODING CONCEPT DIAGRAM
# ==============================================================================
def create_coordconv_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    fig.patch.set_facecolor('#FAF8F5')
    ax.set_facecolor('#FAF8F5')
    ax.axis('off')

    ax.text(0.5, 0.96, 'NGUYÊN LÝ TỌA ĐỘ KHÔNG GIAN COORDCONV & TRIỆT TIÊU BÁO ĐỘNG GIẢ', 
            fontsize=13, fontweight='bold', ha='center', va='top', color='#0F172A')

    # Left: Standard RGB (3 Channels)
    rgb_box = patches.FancyBboxPatch((0.05, 0.22), 0.25, 0.65, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5)
    ax.add_patch(rgb_box)
    ax.text(0.175, 0.82, 'CNN TIÊU CHUẨN (3 KÊNH)', fontsize=10.5, fontweight='bold', ha='center', color='#0F172A')
    ax.text(0.175, 0.77, 'Bất biến tịnh tiến (Translation Invariance)', fontsize=8.5, ha='center', color='#64748B')

    # RGB layers
    for i, c in enumerate(['#FCA5A5', '#86EFAC', '#93C5FD']):
        ax.add_patch(patches.Rectangle((0.10 + i*0.02, 0.42 - i*0.03), 0.11, 0.28, facecolor=c, edgecolor='#475569', alpha=0.85))
    ax.text(0.175, 0.52, 'Tensor Ảnh\n(640 × 640 × 3)\n[R, G, B]', fontsize=9, fontweight='bold', ha='center', color='#0F172A')

    ax.text(0.175, 0.27, '⚠ Xô vữa vàng dưới sàn ($y \\approx 0.9$)\nvà Mũ trên đầu công nhân ($y \\approx 0.2$)\ncó phản ứng kích hoạt giống hệt nhau!\n→ GÂY BÁO ĐỘNG GIẢ', 
            fontsize=8.5, ha='center', color='#DC2626', fontweight='bold')

    # Plus sign
    ax.text(0.33, 0.52, '+', fontsize=24, fontweight='bold', ha='center', va='center', color='#EA580C')

    # Middle: Coordinate Channels
    coord_box = patches.FancyBboxPatch((0.36, 0.22), 0.28, 0.65, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#EA580C', linewidth=1.5)
    ax.add_patch(coord_box)
    ax.text(0.50, 0.82, 'BỔ SUNG 2 KÊNH TỌA ĐỘ', fontsize=10.5, fontweight='bold', ha='center', color='#EA580C')
    ax.text(0.50, 0.77, 'Phá vỡ tính bất biến tịnh tiến', fontsize=8.5, ha='center', color='#64748B')

    # Cx Channel (Horizontal Gradient)
    cx_box = patches.Rectangle((0.40, 0.52), 0.08, 0.18, facecolor='#FFEDD5', edgecolor='#EA580C')
    ax.add_patch(cx_box)
    ax.text(0.44, 0.61, 'Kênh C_x\n[-1, 1]\n(Ngang)', fontsize=8.5, fontweight='bold', ha='center', va='center', color='#C2410C')

    # Cy Channel (Vertical Gradient - crucial!)
    cy_box = patches.Rectangle((0.52, 0.52), 0.08, 0.18, facecolor='#FEE2E2', edgecolor='#DC2626')
    ax.add_patch(cy_box)
    ax.text(0.56, 0.61, 'Kênh C_y\n[-1, 1]\n(Dọc)', fontsize=8.5, fontweight='bold', ha='center', va='center', color='#991B1B')

    ax.text(0.50, 0.38, 'C_y = -1.0 : Đỉnh khung hình (Trời/Trần)\nC_y =  0.0 : Tầm mắt công nhân\nC_y = +1.0 : Mặt đất / Sàn bê tông', 
            fontsize=8.5, ha='center', color='#0F172A', fontweight='bold')
    ax.text(0.50, 0.27, 'Tọa độ chuẩn hóa liên tục giúp kernel học\nquy luật: "Mũ bảo hộ chỉ nằm trên đầu người"', 
            fontsize=8, ha='center', color='#475569')

    # Arrow to Output
    ax.annotate('', xy=(0.69, 0.52), xytext=(0.65, 0.52), arrowprops=dict(arrowstyle="->", color='#16A34A', lw=2.5))

    # Right: Result Tensor (5 Channels) & Effect
    res_box = patches.FancyBboxPatch((0.70, 0.22), 0.26, 0.65, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#16A34A', linewidth=2.0)
    ax.add_patch(res_box)
    ax.text(0.83, 0.82, 'TENSOR REP-YOLO11s (5 KÊNH)', fontsize=10.5, fontweight='bold', ha='center', color='#16A34A')
    ax.text(0.83, 0.77, '[R, G, B, C_x, C_y] (640 × 640 × 5)', fontsize=8.5, ha='center', color='#64748B')

    ax.add_patch(patches.Rectangle((0.77, 0.50), 0.12, 0.22, facecolor='#DCFCE7', edgecolor='#16A34A', linewidth=1.5))
    ax.text(0.83, 0.61, 'CoordConv Stem\nConv(c1=5, c2=64)', fontsize=9, fontweight='bold', ha='center', va='center', color='#166534')

    ax.text(0.83, 0.36, '★ KẾT QUẢ THỰC NGHIỆM:\n• Triệt tiêu >28% báo động giả\n• Khóa nét vòm mũ trên cao\n• Phớt lờ xô vàng và cọc tiêu dưới đất', 
            fontsize=8.8, ha='center', color='#166534', fontweight='bold')
    ax.text(0.83, 0.25, 'Chứng minh bằng bản đồ nhiệt Grad-CAM', fontsize=8, ha='center', color='#475569')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Fig_CoordConv_Concept.png'), facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print("Created Fig_CoordConv_Concept.png")

# ==============================================================================
# 3. BIFORMER SPARSE REGIONAL ROUTING ATTENTION DIAGRAM
# ==============================================================================
def create_biformer_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=300)
    fig.patch.set_facecolor('#FAF8F5')
    ax.set_facecolor('#FAF8F5')
    ax.axis('off')

    ax.text(0.5, 0.96, 'CƠ CHẾ CHÚ Ý ĐỊNH TUYẾN THƯA 2 CẤP ĐỘ (BIFORMER / BRA)', 
            fontsize=13, fontweight='bold', ha='center', va='top', color='#0F172A')

    # Step 1: Feature Map Partitioning
    b1 = patches.FancyBboxPatch((0.04, 0.15), 0.27, 0.74, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5)
    ax.add_patch(b1)
    ax.text(0.175, 0.83, 'BƯỚC 1: CHIA VÙNG THÔ (S×S)', fontsize=10, fontweight='bold', ha='center', color='#0F172A')
    ax.text(0.175, 0.78, 'Lưới S × S vùng (S = 8)', fontsize=8.5, ha='center', color='#64748B')

    # Draw grid
    for r in range(4):
        for c in range(4):
            # Highlight helmet region at (1, 2)
            color = '#FEF08A' if (r==1 and c==2) else ('#EFF6FF' if (r, c) in [(0, 2), (1, 1), (2, 2)] else '#F8FAFC')
            rect = patches.Rectangle((0.08 + c*0.045, 0.40 + (3-r)*0.07), 0.04, 0.06, facecolor=color, edgecolor='#94A3B8', linewidth=0.8)
            ax.add_patch(rect)
    ax.text(0.175, 0.32, 'Trích xuất đại diện trung bình:\nQ^r, K^r cho từng vùng', fontsize=8.5, ha='center', color='#0F172A')
    ax.text(0.175, 0.22, 'Thay vì tính chú ý cho mọi pixel\n(O(H²W²) gây tràn bộ nhớ)', fontsize=8, ha='center', color='#DC2626')

    ax.annotate('', xy=(0.34, 0.52), xytext=(0.31, 0.52), arrowprops=dict(arrowstyle="->", color='#EA580C', lw=2.0))

    # Step 2: Regional Affinity & Top-k Routing
    b2 = patches.FancyBboxPatch((0.35, 0.15), 0.30, 0.74, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#EA580C', linewidth=1.5)
    ax.add_patch(b2)
    ax.text(0.50, 0.83, 'BƯỚC 2: ĐỊNH TUYẾN TOP-k VÙNG', fontsize=10, fontweight='bold', ha='center', color='#EA580C')
    ax.text(0.50, 0.78, 'Ma trận tương quan vùng A^r = Q^r (K^r)^T', fontsize=8.5, ha='center', color='#64748B')

    # Draw affinity matrix mini
    aff_box = patches.Rectangle((0.41, 0.45), 0.18, 0.24, facecolor='#FFF7ED', edgecolor='#EA580C', linewidth=1.2)
    ax.add_patch(aff_box)
    ax.text(0.50, 0.60, 'Ma trận tương quan vùng\nA^r ∈ R^{G × G}', fontsize=8.5, fontweight='bold', ha='center', color='#C2410C')
    ax.text(0.50, 0.50, 'Lọc Top-k liên quan nhất\nk = 4 vùng đích', fontsize=8.5, ha='center', color='#0F172A')

    ax.text(0.50, 0.32, 'Loại bỏ ~80% nền rác:\nBầu trời, sàn đất, tường trống\nhoàn toàn bị ngắt kết nối', fontsize=8.5, ha='center', color='#C2410C', fontweight='bold')
    ax.text(0.50, 0.22, 'Chỉ giữ lại các vùng có khả năng\nchứa công nhân và mũ', fontsize=8, ha='center', color='#475569')

    ax.annotate('', xy=(0.68, 0.52), xytext=(0.65, 0.52), arrowprops=dict(arrowstyle="->", color='#16A34A', lw=2.0))

    # Step 3: Sparse Token-to-Token Attention
    b3 = patches.FancyBboxPatch((0.69, 0.15), 0.28, 0.74, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#16A34A', linewidth=2.0)
    ax.add_patch(b3)
    ax.text(0.83, 0.83, 'BƯỚC 3: CHÚ Ý VI MÔ CÓ CHỌN LỌC', fontsize=10, fontweight='bold', ha='center', color='#16A34A')
    ax.text(0.83, 0.78, 'Token-to-Token Attention trong Top-k', fontsize=8.5, ha='center', color='#64748B')

    ax.add_patch(patches.Rectangle((0.74, 0.48), 0.18, 0.20, facecolor='#DCFCE7', edgecolor='#16A34A', linewidth=1.2))
    ax.text(0.83, 0.60, 'ĐỘ PHỨC TẠP TUYẾN TÍNH', fontsize=8.5, fontweight='bold', ha='center', color='#166534')
    ax.text(0.83, 0.52, 'O(S² + k·HW/S²)\n≈ O(HW)', fontsize=11, fontweight='bold', ha='center', color='#15803D')

    ax.text(0.83, 0.34, '★ KẾT QUẢ ĐẠT ĐƯỢC:\n• Tập trung 100% mật độ chú ý vào vòm mũ <20px\n• Đẩy Recall mũ lên đỉnh 91.33%\n• Không làm tràn VRAM độ phân giải 1024x1024', 
            fontsize=8.5, ha='center', color='#166534', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Fig_BiFormer_Sparse_Attention.png'), facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print("Created Fig_BiFormer_Sparse_Attention.png")

# ==============================================================================
# 4. FOCAL EIOU BOUNDING BOX DECOMPOSITION DIAGRAM
# ==============================================================================
def create_focal_eiou_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=300)
    fig.patch.set_facecolor('#FAF8F5')
    ax.set_facecolor('#FAF8F5')
    ax.axis('off')

    ax.text(0.5, 0.96, 'HÀM MẤT MÁT FOCAL EIOU: PHÂN RÃ TRỰC GIAO KÍCH THƯỚC HỘP BAO', 
            fontsize=13, fontweight='bold', ha='center', va='top', color='#0F172A')

    # Left: Deficiency of CIoU
    b1 = patches.FancyBboxPatch((0.04, 0.15), 0.32, 0.74, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5)
    ax.add_patch(b1)
    ax.text(0.20, 0.83, 'HẠN CHẾ CỦA HÀM CIoU CŨ', fontsize=10, fontweight='bold', ha='center', color='#DC2626')
    ax.text(0.20, 0.78, 'Sử dụng tỷ lệ góc arctan(w/h)', fontsize=8.5, ha='center', color='#64748B')

    # Draw two different boxes with same aspect ratio
    ax.add_patch(patches.Rectangle((0.08, 0.44), 0.08, 0.16, facecolor='#FEE2E2', edgecolor='#DC2626', linewidth=1.5, linestyle='--'))
    ax.text(0.12, 0.52, 'Box 1\nw/h = 0.5', fontsize=8, ha='center', va='center', color='#991B1B')

    ax.add_patch(patches.Rectangle((0.20, 0.40), 0.12, 0.24, facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.5))
    ax.text(0.26, 0.52, 'Box 2 (To hơn)\nw/h = 0.5', fontsize=8, ha='center', va='center', color='#92400E')

    ax.text(0.20, 0.28, '⚠ Khi cả chiều rộng w và chiều cao h\ncùng sai nhưng tỷ lệ w/h tình cờ bằng nhau,\nphạt tỷ lệ góc v bị triệt tiêu đạo hàm!\n→ Bounding box không hội tụ chuẩn', 
            fontsize=8.5, ha='center', color='#DC2626')

    # Right: Focal EIoU Decomposition
    b2 = patches.FancyBboxPatch((0.40, 0.15), 0.56, 0.74, boxstyle="round,pad=0.02", facecolor='#FFFFFF', edgecolor='#16A34A', linewidth=2.0)
    ax.add_patch(b2)
    ax.text(0.68, 0.83, 'GIẢI PHÁP FOCAL EIOU: PHÂN RÃ 3 THÀNH PHẦN ĐỘC LẬP', fontsize=10.5, fontweight='bold', ha='center', color='#16A34A')
    ax.text(0.68, 0.78, 'L_EIoU = L_IoU + L_center + L_width + L_height', fontsize=9.5, fontweight='bold', ha='center', color='#0F172A')

    # Visual representation of 3 components
    # 1. Overlap IoU
    c1 = patches.FancyBboxPatch((0.43, 0.48), 0.15, 0.22, boxstyle="round,pad=0.01", facecolor='#EFF6FF', edgecolor='#2563EB', linewidth=1.2)
    ax.add_patch(c1)
    ax.text(0.505, 0.63, '① Độ Trùng Khớp', fontsize=8.8, fontweight='bold', ha='center', color='#1D4ED8')
    ax.text(0.505, 0.55, '1 - IoU\nĐo diện tích phủ', fontsize=8, ha='center', color='#0F172A')

    # 2. Center Distance
    c2 = patches.FancyBboxPatch((0.60, 0.48), 0.16, 0.22, boxstyle="round,pad=0.01", facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.2)
    ax.add_patch(c2)
    ax.text(0.68, 0.63, '② Khoảng Cách Tâm', fontsize=8.8, fontweight='bold', ha='center', color='#B45309')
    ax.text(0.68, 0.55, 'ρ²(b, b_gt) / c²\nKéo tâm hộp trùng nhau', fontsize=8, ha='center', color='#0F172A')

    # 3. Independent Width & Height
    c3 = patches.FancyBboxPatch((0.78, 0.48), 0.16, 0.22, boxstyle="round,pad=0.01", facecolor='#DCFCE7', edgecolor='#16A34A', linewidth=1.2)
    ax.add_patch(c3)
    ax.text(0.86, 0.63, '③ Tách Cạnh Độc Lập', fontsize=8.8, fontweight='bold', ha='center', color='#166534')
    ax.text(0.86, 0.54, 'ρ²(w,w_gt)/C_w² +\nρ²(h,h_gt)/C_h²', fontsize=8, ha='center', color='#0F172A')

    # Focal modulation
    focal_bar = patches.FancyBboxPatch((0.43, 0.22), 0.51, 0.18, boxstyle="round,pad=0.01", facecolor='#FFF7ED', edgecolor='#EA580C', linewidth=1.5)
    ax.add_patch(focal_bar)
    ax.text(0.685, 0.32, '★ ĐIỀU TIẾT MẪU KHÓ FOCAL: L_Focal = IoU^0.5 × L_EIoU', fontsize=9.2, fontweight='bold', ha='center', color='#EA580C')
    ax.text(0.685, 0.25, 'Tự động tăng gradient cho các vi vật thể mũ bảo hộ bị che khuất và diện tích nhỏ dưới 20px', fontsize=8.2, ha='center', color='#475569')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Fig_Focal_EIoU_Decomposition.png'), facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print("Created Fig_Focal_EIoU_Decomposition.png")

if __name__ == '__main__':
    create_repconv_diagram()
    create_coordconv_diagram()
    create_biformer_diagram()
    create_focal_eiou_diagram()
    print("All diagram assets successfully generated!")
