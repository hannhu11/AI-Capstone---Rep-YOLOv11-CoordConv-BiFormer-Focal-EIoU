"""
Render Slides for Sections 4, 5, and 6 as 300 DPI high-resolution PNG images.
Output directory: review_1_main/slide_renders_4_5_6/
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set up matplotlib style
plt.rcParams['font.sans-serif'] = ['Calibri', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

OUT_DIR = os.path.join(os.getcwd(), 'review_1_main', 'slide_renders_4_5_6')
os.makedirs(OUT_DIR, exist_ok=True)

C_BG = '#FAF8F5'
C_WHITE = '#FFFFFF'
C_BORDER = '#E2E8F0'
C_ORANGE = '#EA580C'
C_ORANGE_LIGHT = '#FFF7ED'
C_ORANGE_BORDER = '#FDBA74'
C_BLUE = '#2563EB'
C_AMBER = '#D97706'
C_GREEN = '#16A34A'
C_PURPLE = '#7C3AED'
C_SLATE = '#475569'
C_TITLE = '#0F172A'
C_BODY = '#334155'
C_MUTED = '#64748B'

def init_canvas():
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    return fig, ax

def draw_header(ax, category, slide_num, title):
    # Category tag
    ax.text(0.96, 8.45, category.upper(), fontsize=12, fontweight='bold', color=C_ORANGE, va='center')
    # Slide number
    ax.text(15.04, 8.45, slide_num, fontsize=13, color=C_MUTED, va='center', ha='right')
    # Title
    ax.text(0.96, 7.95, title, fontsize=21, fontweight='bold', color=C_TITLE, va='center')

def draw_bottom_banner(ax, text):
    banner = FancyBboxPatch((0.96, 0.40), 14.08, 0.65,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1)
    ax.add_patch(banner)

    pill = FancyBboxPatch((1.15, 0.50), 1.45, 0.45,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor=C_ORANGE, edgecolor='none')
    ax.add_patch(pill)
    ax.text(1.875, 0.725, "TAKEAWAY", fontsize=10.5, fontweight='bold', color=C_WHITE, va='center', ha='center')

    wrapped = textwrap.fill(text, width=125)
    ax.text(2.80, 0.725, wrapped, fontsize=11, fontweight='bold', color=C_TITLE, va='center')

def wrap_str(text, width=45):
    return textwrap.fill(text, width=width)

# =============================================================================
# SLIDE 1: MỤC 4.1 · BẢNG SO SÁNH ĐỐI CHUẨN SOTA CƠ SỞ
# =============================================================================
def render_slide_1():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "04 / 18",
                "Bảng So sánh Đối chuẩn SOTA Cơ sở trên Tập Chuẩn SHWD (VOC2028)")

    # Draw Table Container
    table_card = FancyBboxPatch((0.96, 3.25), 14.08, 4.35,
                                boxstyle="round,pad=0.01,rounding_size=0.1",
                                facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(table_card)

    # Table columns definitions: (Header, Width, Align)
    # Total width = 14.08. Left starts at 0.96
    cols = [
        ("Model Architecture", 3.48, 'left'),
        ("Params (M)", 1.40, 'center'),
        ("FLOPs (G)", 1.40, 'center'),
        ("mAP50 (%)", 1.55, 'center'),
        ("mAP50-95 (%)", 1.65, 'center'),
        ("AP50 hat (%)", 1.55, 'center'),
        ("Recall hat (%)", 1.65, 'center'),
        ("F1 hat", 1.40, 'center'),
    ]

    table_data = [
        ("YOLOv8n", "3.15", "8.7", "93.21", "60.26", "92.42", "87.16", "0.8879"),
        ("YOLOv8s", "11.24", "28.6", "94.89", "62.21", "94.28", "90.62", "0.9120"),
        ("YOLOv10n", "2.30", "6.7", "93.30", "60.35", "92.97", "87.13", "0.8930"),
        ("YOLOv10s", "8.00", "21.6", "94.39", "62.19", "93.36", "89.14", "0.9049"),
        ("YOLO11n", "2.60", "6.5", "93.19", "60.21", "92.33", "86.51", "0.8964"),
        ("YOLO11s (Chính)", "9.40", "21.5", "94.74", "62.54", "94.06", "90.35", "0.9073"),
        ("Rep-YOLO11s (Mục tiêu của Đề tài)", "9.85", "22.4", "94.83 (5-Fold: 96.64)", "62.54", "94.44", "91.33", "0.9190"),
    ]

    header_y = 7.30
    row_h = 0.50

    # Header Row Background
    h_bg = FancyBboxPatch((0.96, 7.05), 14.08, 0.55,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor=C_TITLE, edgecolor='none')
    ax.add_patch(h_bg)

    # Render Header Texts
    cur_x = 0.96
    for title, w, align in cols:
        pos_x = cur_x + 0.20 if align == 'left' else cur_x + w / 2.0
        ax.text(pos_x, header_y, title, fontsize=10.5, fontweight='bold', color=C_WHITE,
                va='center', ha=align)
        cur_x += w

    # Render Data Rows
    for r_idx, row in enumerate(table_data):
        y_center = 6.80 - r_idx * row_h
        is_champ = (r_idx == len(table_data) - 1)
        is_alt = (r_idx % 2 == 1) and not is_champ

        # Row background
        if is_champ:
            r_bg = FancyBboxPatch((0.96, y_center - 0.22), 14.08, row_h - 0.04,
                                  boxstyle="round,pad=0.01,rounding_size=0.08",
                                  facecolor='#FEF3C7', edgecolor=C_ORANGE_BORDER, linewidth=1.5)
            ax.add_patch(r_bg)
        elif is_alt:
            r_bg = patches.Rectangle((0.96, y_center - 0.22), 14.08, row_h - 0.04,
                                     facecolor='#F8FAFC', edgecolor='none')
            ax.add_patch(r_bg)

        # Subtle row separator
        if not is_champ:
            ax.plot([0.96, 15.04], [y_center - 0.24, y_center - 0.24], color='#F1F5F9', lw=0.8)

        cur_x = 0.96
        for c_idx, val in enumerate(row):
            w = cols[c_idx][1]
            align = cols[c_idx][2]
            pos_x = cur_x + 0.20 if align == 'left' else cur_x + w / 2.0

            font_weight = 'bold' if (is_champ or c_idx == 0) else 'normal'
            if is_champ:
                font_color = '#9A3412' if c_idx in [0, 3, 6, 7] else C_TITLE
            else:
                font_color = C_TITLE if c_idx == 0 else C_BODY

            font_size = 10.2 if is_champ else 9.6
            ax.text(pos_x, y_center, val, fontsize=font_size, fontweight=font_weight,
                    color=font_color, va='center', ha=align)
            cur_x += w

    # 3 Summary Insight Cards at Bottom
    c_w = 4.48
    c_gap = 0.32
    c_top = 1.30
    c_h = 1.70

    insights = [
        ("01", C_BLUE, "Ưu thế Recall Sinh mạng (91.33%)",
         "Recallhat đạt đỉnh 91.33%, cao hơn YOLO11s (90.35%) và YOLOv8s (90.62%). Trong an toàn lao động, bỏ sót vi phạm là rủi ro tử vong; mô hình đề tài tối đa hóa khả năng cứu sinh."),
        ("02", C_ORANGE, "Cân bằng Tham số & Độ phức tạp",
         "9.85M params & 22.4 GFLOPs xấp xỉ YOLO11s gốc (9.40M/21.5G), nhẹ hơn đáng kể so với YOLOv8s (11.24M/28.6G). Tích hợp 4 module cải tiến mà không gây bùng nổ tài nguyên tính toán."),
        ("03", C_GREEN, "Độ ổn định Thống kê 5-Fold (96.64%)",
         "Trên quy trình 5-Fold Stratified CV, mô hình đạt đỉnh 96.64% ± 0.32% mAP50. Chứng minh độ tin cậy vượt trội, loại trừ hoàn toàn yếu tố may rủi của phép chia tập dữ liệu ngẫu nhiên.")
    ]

    for idx, (b_num, b_col, ins_title, ins_desc) in enumerate(insights):
        c_left = 0.96 + idx * (c_w + c_gap)
        sc = FancyBboxPatch((c_left, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(sc)

        sb = FancyBboxPatch((c_left + 0.18, c_top + c_h - 0.42), 0.45, 0.30,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            facecolor=b_col, edgecolor='none')
        ax.add_patch(sb)
        ax.text(c_left + 0.405, c_top + c_h - 0.27, b_num, fontsize=10.5, fontweight='bold', color=C_WHITE, va='center', ha='center')

        ax.text(c_left + 0.75, c_top + c_h - 0.27, ins_title, fontsize=11.5, fontweight='bold', color=C_TITLE, va='center')

        wrapped_desc = textwrap.fill(ins_desc, width=50)
        ax.text(c_left + 0.20, c_top + 0.65, wrapped_desc, fontsize=9.6, color=C_BODY, va='center', linespacing=1.35)

    draw_bottom_banner(ax, "Bảng đối chuẩn khách quan trên 7,581 ảnh SHWD chứng minh Rep-YOLO11s vượt trội về Recallhat (91.33%) và F1hat (0.9190) với chi phí phần cứng tối ưu.")

    out_file = os.path.join(OUT_DIR, "Slide_04_1_Baseline_Comparison_Table.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 2: MỤC 4.2 · PHÂN TÍCH BASELINE 1: YOLO11s (SOTA MỚI NHẤT)
# =============================================================================
def render_slide_2():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05 / 18",
                "Phân tích Chi tiết Baseline 1: YOLO11s (Ultralytics, 10/2024 · SOTA Mới nhất)")

    # Left Card: Architecture & Specs
    c_w = 6.88
    c_gap = 0.32
    c_top = 1.30
    c_h = 6.30

    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_BLUE, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "Kiến trúc & Năng lực Thực nghiệm trên SHWD", fontsize=13.5, fontweight='bold', color=C_TITLE, va='center')

    y11_left = [
        ("Paper & Thời điểm Công bố", "Phát hành bởi Ultralytics (Tháng 10/2024), đại diện cho thế hệ mô hình phát hiện vật thể SOTA mới nhất."),
        ("Cấu trúc Mạng Đặc trưng", "Backbone CSPDarknet cải tiến với khối C3k2 và khối tự chú ý không gian C2PSA; Neck PANet trích xuất đa quy mô."),
        ("3 Tầng Đầu ra Phát hiện", "P3 (stride 8: 80x80), P4 (stride 16: 40x40), P5 (stride 32: 20x20) kết hợp Decoupled Head không neo (Anchor-free)."),
        ("Tập Dữ liệu Huấn luyện", "Pretrained trên COCO và fine-tune trực tiếp trên SHWD (7,581 ảnh authentic) với phân chia 80/20 chuẩn."),
        ("Kết quả Đối chuẩn Thực nghiệm", "mAP50 = 94.74% · mAP50-95 = 62.54% · Recallhat = 90.35% · F1hat = 0.9073 (Params: 9.40M · FLOPs: 21.5G)."),
        ("Ưu điểm Nổi bật", "Tối ưu hóa số lượng tham số tốt, tốc độ hội tụ nhanh, thông lượng xử lý cao trên các nền tảng GPU hiện đại.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y11_left:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=10.2, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=70)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.4, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    # Right Card: Bottlenecks & Improvements
    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_ORANGE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_ORANGE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "3 Điểm nghẽn Cốt lõi & Giải pháp Kế thừa / Cải tiến", fontsize=13.5, fontweight='bold', color=C_TITLE, va='center')

    y11_right = [
        ("Điểm nghẽn 1 (Mất dấu mũ ở xa)", "Các tầng downsampling tích lũy (stride 8/16/32) làm tiêu biến hoàn toàn tín hiệu của mũ bảo hộ siêu nhỏ (<20px)."),
        ("Điểm nghẽn 2 (Báo động giả sàn >28%)", "Tích chập tiêu chuẩn có tính Bất biến Tịnh tiến (Translation Invariance), gây nhầm lẫn xô vữa, biển cảnh báo màu vàng dưới nền đất thành mũ."),
        ("Điểm nghẽn 3 (Triệt tiêu Gradient CIoU)", "Hàm loss CIoU bị triệt tiêu gradient khi tỷ lệ aspect ratio w/h của bounding box dự đoán trùng với ground truth."),
        ("Nhóm học được gì từ YOLO11s?", "Học được cấu trúc khối trích xuất đặc trưng C3k2 gọn nhẹ và nguyên lý phân tách hai nhánh phân loại / định vị (Decoupled Head)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa bộ khung YOLO11s nhưng thay bằng RepConv (gộp nhánh đại số về 1 Conv 3x3), cấy CoordConv Stem, bổ sung BiFormer Neck và thay bằng Focal-EIoU.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y11_right:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=10.2, fontweight='bold', color=C_ORANGE, va='top')
        wrapped = textwrap.fill(desc, width=70)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.4, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    draw_bottom_banner(ax, "YOLO11s đại diện cho SOTA kiến trúc mới nhất (10/2024); nhóm kế thừa sườn mạng nhưng giải quyết triệt để 3 điểm nghẽn bằng 4 module toán học.")

    out_file = os.path.join(OUT_DIR, "Slide_04_2_Baseline_1_YOLO11s.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 3: MỤC 4.2 · PHÂN TÍCH BASELINE 2: YOLOv8s (CHUẨN MỰC CÔNG NGHIỆP)
# =============================================================================
def render_slide_3():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "06 / 18",
                "Phân tích Chi tiết Baseline 2: YOLOv8s (Jocher et al., 2023 · Chuẩn mực Công nghiệp)")

    c_w = 6.88
    c_gap = 0.32
    c_top = 1.30
    c_h = 6.30

    # Left Card: Architecture & Specs
    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_AMBER, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_AMBER, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "Kiến trúc & Vị thế Mốc chuẩn Công nghiệp (Gold Standard)", fontsize=13.5, fontweight='bold', color=C_TITLE, va='center')

    y8_left = [
        ("Paper & Vị thế Công nghiệp", "Được phát triển bởi Jocher et al. (2023), YOLOv8 là mốc chuẩn công nghiệp phổ biến và được ứng dụng rộng rãi nhất thế giới hiện nay."),
        ("Cấu trúc Khối Trích xuất C2f", "Cross Stage Partial with 2 Convolutions (C2f) kết hợp nhiều kết nối residual nội bộ giúp làm giàu dòng chảy gradient."),
        ("Cơ chế Gán nhãn Động (TAL)", "Task-Aligned Assigner đo lường sự kết hợp giữa độ tự tin phân loại và căn chỉnh IoU để gán nhãn động cho top-k anchors tốt nhất."),
        ("Tập Dữ liệu Huấn luyện", "Thực nghiệm trực tiếp trên tập SHWD chuẩn (7,581 ảnh authentic) trong cùng điều kiện phần cứng với các baseline khác."),
        ("Kết quả Đối chuẩn Thực nghiệm", "mAP50 = 94.89% · mAP50-95 = 62.21% · Recallhat = 90.62% · F1hat = 0.9120 (Params: 11.24M · FLOPs: 28.6G)."),
        ("Ưu điểm Nổi bật", "Độ chính xác nhận diện tổng thể mAP50 cao, cộng đồng mã nguồn mở khổng lồ, mức độ ổn định sản xuất đã được kiểm chứng.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y8_left:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=10.2, fontweight='bold', color=C_AMBER, va='top')
        wrapped = textwrap.fill(desc, width=70)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.4, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    # Right Card: Hardware Bottlenecks & Lessons
    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_PURPLE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_PURPLE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "Hạn chế Phần cứng & Bài học Kế thừa / Cải tiến", fontsize=13.5, fontweight='bold', color=C_TITLE, va='center')

    y8_right = [
        ("Hạn chế 1 (Chi phí Tính toán Nặng)", "FLOPs lên tới 28.6G và 11.24M tham số, nặng hơn 28% so với Rep-YOLO11s (22.4G / 9.85M), tạo áp lực lớn khi triển khai biên."),
        ("Hạn chế 2 (Nghẽn Băng thông Bộ nhớ DRAM)", "Cấu trúc C2f phân nhánh nội bộ liên tục sử dụng nhiều thao tác nối tensor (concat) làm tăng lưu lượng truy cập DRAM và độ trễ truy xuất."),
        ("Hạn chế 3 (Thiếu Cơ chế Không gian)", "Không có thông tin tọa độ dọc, mô hình vẫn mắc phải tỷ lệ báo động giả cao đối với các thiết bị màu vàng dưới sàn nhà."),
        ("Nhóm học được gì từ YOLOv8s?", "Học tập cơ chế gán nhãn động Task-Aligned Assigner và chiến lược cân bằng loss đa nhiệm (BCE Classification Loss + DFL Loss)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa tư tưởng Task-Aligned Assigner nhưng xây dựng trên nền tảng YOLO11s gọn nhẹ hơn, áp dụng RepConv loại bỏ hoàn toàn chi phí bộ nhớ phân nhánh khi triển khai.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y8_right:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=10.2, fontweight='bold', color=C_PURPLE, va='top')
        wrapped = textwrap.fill(desc, width=70)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.4, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    draw_bottom_banner(ax, "YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất; nhóm kế thừa cơ chế TAL gán nhãn động nhưng tối ưu triệt để chi phí tính toán và bộ nhớ DRAM.")

    out_file = os.path.join(OUT_DIR, "Slide_04_2_Baseline_2_YOLOv8s.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 4: MỤC 4.2 · SO SÁNH ĐỐI ĐẦU 2 BASELINE TRÊN 1 SLIDE (TÙY CHỌN GỌN)
# =============================================================================
def render_slide_4():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05-06 / 18",
                "Phân tích So sánh 2 Mô hình Baseline Chính: YOLO11s vs YOLOv8s")

    c_w = 6.88
    c_gap = 0.32
    c_top = 3.05
    c_h = 4.55

    # Column Left: YOLO11s
    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(card_l)

    b_l = FancyBboxPatch((1.20, c_top + c_h - 0.50), 2.80, 0.36,
                         boxstyle="round,pad=0.01,rounding_size=0.08",
                         facecolor=C_BLUE, edgecolor='none')
    ax.add_patch(b_l)
    ax.text(2.60, c_top + c_h - 0.32, "BASELINE 1 · YOLO11s", fontsize=11, fontweight='bold', color=C_WHITE, va='center', ha='center')

    ax.text(1.25, c_top + c_h - 0.82, "SOTA Mới Nhất (Ultralytics, Tháng 10/2024)", fontsize=12, fontweight='bold', color=C_TITLE, va='center')

    c1_pts = [
        ("Kiến trúc mạng:", "C3k2 Feature Blocks + C2PSA Spatial Attention + Decoupled Head."),
        ("Thông số thực nghiệm:", "9.40M Params · 21.5 GFLOPs · mAP50 = 94.74% · Recall = 90.35%."),
        ("Ưu điểm:", "Tối ưu hóa số lượng tham số xuất sắc, tính toán nhanh, hội tụ nhanh."),
        ("Hạn chế cốt lõi:", "Mất tín hiệu mũ nhỏ ở xa (<20px); Báo động giả sàn nhà do bất biến tịnh tiến; Triệt tiêu gradient tỷ lệ khung CIoU."),
        ("Kế thừa & Cải tiến:", "Kế thừa sườn C3k2; thay thế RepConv, cấy CoordConv Stem, bổ sung BiFormer Neck và Focal-EIoU Head.")
    ]
    y_pos = c_top + c_h - 1.25
    for tag, desc in c1_pts:
        ax.text(1.25, y_pos, f"• {tag}", fontsize=9.8, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=64)
        ax.text(1.45, y_pos - 0.25, wrapped, fontsize=8.8, color=C_BODY, va='top', linespacing=1.28)
        y_pos -= 0.65

    # Column Right: YOLOv8s
    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_AMBER, linewidth=1.5)
    ax.add_patch(card_r)

    b_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.50), 2.80, 0.36,
                         boxstyle="round,pad=0.01,rounding_size=0.08",
                         facecolor=C_AMBER, edgecolor='none')
    ax.add_patch(b_r)
    ax.text(0.96 + c_w + c_gap + 1.64, c_top + c_h - 0.32, "BASELINE 2 · YOLOv8s", fontsize=11, fontweight='bold', color=C_WHITE, va='center', ha='center')

    ax.text(0.96 + c_w + c_gap + 0.25, c_top + c_h - 0.82, "Chuẩn Mực Công Nghiệp (Jocher et al., 2023)", fontsize=12, fontweight='bold', color=C_TITLE, va='center')

    c2_pts = [
        ("Kiến trúc mạng:", "C2f Cross Stage Partial Blocks + Task-Aligned Assigner (TAL)."),
        ("Thông số thực nghiệm:", "11.24M Params · 28.6 GFLOPs · mAP50 = 94.89% · Recall = 90.62%."),
        ("Ưu điểm:", "Độ chính xác mAP50 cao, mã nguồn cực kỳ ổn định, cộng đồng hỗ trợ lớn."),
        ("Hạn chế cốt lõi:", "28.6 GFLOPs nặng hơn 28% so với đề tài; Khối C2f ngốn băng thông DRAM; Báo động giả do thiếu tiên đề không gian."),
        ("Kế thừa & Cải tiến:", "Kế thừa cơ chế gán nhãn động TAL nhưng chuyển sang nền sườn YOLO11s gọn hơn và áp dụng RepConv xóa bỏ overhead DRAM.")
    ]
    y_pos = c_top + c_h - 1.25
    for tag, desc in c2_pts:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}", fontsize=9.8, fontweight='bold', color=C_AMBER, va='top')
        wrapped = textwrap.fill(desc, width=64)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.25, wrapped, fontsize=8.8, color=C_BODY, va='top', linespacing=1.28)
        y_pos -= 0.65

    # Bottom Spanning Card: Scientific Justification
    c_just = FancyBboxPatch((0.96, 1.30), 14.08, 1.55,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor='#F8FAFC', edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(c_just)

    ax.text(1.25, 2.55, "TẠI SAO NHÓM CHỌN 2 BASELINE NÀY ĐỂ BÁO CÁO REVIEW 1? (SCIENTIFIC JUSTIFICATION)",
            fontsize=11.2, fontweight='bold', color=C_ORANGE, va='center')

    just_p1 = textwrap.fill("• Tính Đại diện SOTA & Uy tín Khoa học: YOLO11s đại diện cho đỉnh cao kiến trúc mới nhất (tháng 10/2024), trong khi YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất thế giới (Gold Standard). Đối chuẩn đồng thời với cả hai tạo nền tảng vững chắc và khách quan tuyệt đối trước Hội đồng chấm.", width=130)
    ax.text(1.25, 2.10, just_p1, fontsize=9.0, color=C_BODY, va='center', linespacing=1.3)

    just_p2 = textwrap.fill("• Nền tảng Minh chứng Cải tiến: Mọi cải tiến của Rep-YOLO11s đều xuất phát từ việc khắc phục chính xác các điểm nghẽn kỹ thuật được chỉ ra ở hai mô hình cơ sở này, chứng minh tính cấp thiết và giá trị khoa học thực sự của đề tài.", width=130)
    ax.text(1.25, 1.62, just_p2, fontsize=9.0, color=C_BODY, va='center', linespacing=1.3)

    draw_bottom_banner(ax, "Đối chuẩn song song SOTA mới nhất (YOLO11s) và Chuẩn công nghiệp (YOLOv8s) tạo cơ sở khoa học khách quan chứng minh tính vượt trội của đề tài.")

    out_file = os.path.join(OUT_DIR, "Slide_04_2_Baselines_Combined_Comparison.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 5: MỤC 6.1 · KIẾN TRÚC TỔNG THỂ ĐỀ XUẤT: REP-YOLO11s
# =============================================================================
def render_slide_5():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 6 · PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED METHOD)", "07 / 18",
                "Kiến trúc Tổng thể Rep-YOLO11s: Tích hợp 4 Cải tiến Toán học")

    # Left Column: Flowchart Container
    c_flow_w = 8.16
    c_top = 1.30
    c_h = 6.30

    card_flow = FancyBboxPatch((0.96, c_top), c_flow_w, c_h,
                               boxstyle="round,pad=0.01,rounding_size=0.1",
                               facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(card_flow)

    f_tag = FancyBboxPatch((1.20, c_top + c_h - 0.45), 3.80, 0.32,
                           boxstyle="round,pad=0.01,rounding_size=0.08",
                           facecolor=C_SLATE, edgecolor='none')
    ax.add_patch(f_tag)
    ax.text(3.10, c_top + c_h - 0.29, "PIPELINE KIẾN TRÚC TỔNG THỂ REP-YOLO11s", fontsize=9.5, fontweight='bold', color=C_WHITE, va='center', ha='center')

    flow_steps = [
        ("ẢNH ĐẦU VÀO [640×640×3]", "Luồng Video Camera Giám sát Công trường Xây dựng (CCTV Feed)", C_SLATE, False),
        ("CẢI TIẾN 2: COORDCONV STEM", "Cấy 2 kênh tọa độ [Cx, Cy] vào Layer 0 (5 -> 64 kênh) -> Khử báo động sàn", C_ORANGE, True),
        ("CẢI TIẾN 1: REPCONV BACKBONE", "3 nhánh huấn luyện -> Gộp đại số về 1 Conv 3x3 khi triển khai (Zero Latency)", C_BLUE, True),
        ("CẢI TIẾN 3: BIFORMER NECK", "Định tuyến thưa 2 cấp độ Top-k (Complexity O(HW)) -> Bắt trúng mũ nhỏ xa", C_PURPLE, True),
        ("CẢI TIẾN 4: FOCAL-EIoU HEAD", "Phân rã kích thước độc lập + TAL gán nhãn động -> Tránh suy biến gradient", C_AMBER, True),
        ("KẾT QUẢ ĐẦU RA (OUTPUT)", "Khung bao mũ bảo hộ sắc nét, Recall >91%, triệt tiêu báo động giả mặt sàn", C_GREEN, False)
    ]

    step_top_start = c_top + c_h - 0.75
    step_h = 0.72
    step_gap = 0.20

    for idx, (title, sub, col, is_mod) in enumerate(flow_steps):
        s_y = step_top_start - (idx + 1) * step_h - idx * step_gap

        # Draw Connector Arrow from previous step
        if idx > 0:
            arr_y = s_y + step_h + step_gap / 2.0
            ax.annotate('', xy=(5.04, s_y + step_h), xytext=(5.04, s_y + step_h + step_gap),
                        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.5, mutation_scale=12))

        # Step card
        bg_col = C_ORANGE_LIGHT if is_mod else '#F8FAFC'
        border_col = col if is_mod else C_BORDER
        sc = FancyBboxPatch((1.20, s_y), c_flow_w - 0.48, step_h,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            facecolor=bg_col, edgecolor=border_col, linewidth=1.5 if is_mod else 1)
        ax.add_patch(sc)

        # Color bar indicator
        bar = FancyBboxPatch((1.28, s_y + 0.08), 0.14, step_h - 0.16,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             facecolor=col, edgecolor='none')
        ax.add_patch(bar)

        ax.text(1.56, s_y + step_h - 0.22, title, fontsize=11, fontweight='bold', color=col, va='center')
        ax.text(1.56, s_y + 0.22, sub, fontsize=9.2, color=C_TITLE if is_mod else C_MUTED, va='center')

    # Right Column: 4 Scientific Impact Cards
    c_right_w = 5.60
    c_right_left = 9.44
    mod_cards = [
        ("Cải tiến 1: Tái tham số hóa RepConv", C_BLUE,
         "Huấn luyện đa nhánh giàu biểu diễn (3x3, 1x1, Identity) -> Gộp đại số khép kín về duy nhất 1 nhân Conv 3x3 khi suy luận (switch_to_deploy). Đạt tốc độ cực hạn với Zero Latency Overhead."),
        ("Cải tiến 2: Mã hóa Tọa độ CoordConv", C_ORANGE,
         "Cấy trực tiếp 2 kênh tọa độ không gian Cartesian [Cx, Cy] vào tầng Stem đầu vào. Phá vỡ tính bất biến tịnh tiến sai lầm, triệt tiêu dứt điểm báo động giả xô vàng và áo phản quang dưới sàn."),
        ("Cải tiến 3: Chú ý Định tuyến BiFormer", C_PURPLE,
         "Cơ chế định tuyến thưa 2 cấp độ lọc bỏ 93.75% các vùng nền rậm rạp không liên quan, tập trung năng lực tính toán vào các vùng chỏm đầu công nhân ở xa với độ phức tạp tuyến tính O(HW)."),
        ("Cải tiến 4: Hàm mất mát Focal-EIoU", C_AMBER,
         "Phân rã độc lập sai số chiều rộng và chiều cao thay vì tỷ lệ khung bao aspect ratio của CIoU, ngăn ngừa triệt tiêu gradient. Kết hợp trọng số tiêu điểm tập trung tối đa cho mẫu mục tiêu nhỏ khó.")
    ]

    card_h = 1.48
    card_gap = 0.13
    card_top_start = c_top + c_h

    for idx, (m_title, m_col, m_desc) in enumerate(mod_cards):
        m_y = card_top_start - (idx + 1) * card_h - idx * card_gap
        mc = FancyBboxPatch((c_right_left, m_y), c_right_w, card_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(mc)

        # Left color strip
        strip = FancyBboxPatch((c_right_left + 0.16, m_y + 0.12), 0.12, card_h - 0.24,
                               boxstyle="round,pad=0.01,rounding_size=0.05",
                               facecolor=m_col, edgecolor='none')
        ax.add_patch(strip)

        ax.text(c_right_left + 0.42, m_y + card_h - 0.30, m_title, fontsize=11.2, fontweight='bold', color=m_col, va='center')

        wrapped_desc = textwrap.fill(m_desc, width=54)
        ax.text(c_right_left + 0.42, m_y + 0.52, wrapped_desc, fontsize=8.8, color=C_BODY, va='center', linespacing=1.35)

    draw_bottom_banner(ax, "Rep-YOLO11s kết hợp 4 cải tiến toán học giải quyết đồng thời 4 rào cản: Khử báo động sàn, suy luận không độ trễ, tập trung mục tiêu xa và hồi quy hộp chính xác.")

    out_file = os.path.join(OUT_DIR, "Slide_06_1_Proposed_Rep_YOLO11s_Architecture.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

if __name__ == '__main__':
    render_slide_1()
    render_slide_2()
    render_slide_3()
    render_slide_4()
    render_slide_5()
    print("All slides for Sections 4, 5, 6 successfully rendered at 300 DPI!")
