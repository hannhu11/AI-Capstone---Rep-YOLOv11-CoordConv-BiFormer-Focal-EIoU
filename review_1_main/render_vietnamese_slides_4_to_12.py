"""
Render the complete Vietnamese slide deck (Sections 4 through 12)
as 300 DPI high-resolution PNG images.

Output directory: review_1_main/slide_renders_tieng_viet/
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Calibri', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

OUT_DIR = os.path.join(os.getcwd(), 'review_1_main', 'slide_renders_tieng_viet')
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
C_RED = '#DC2626'
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
    ax.text(0.96, 8.45, category.upper(), fontsize=11.5, fontweight='bold', color=C_ORANGE, va='center')
    ax.text(15.04, 8.45, slide_num, fontsize=12.5, color=C_MUTED, va='center', ha='right')
    ax.text(0.96, 7.95, title, fontsize=20, fontweight='bold', color=C_TITLE, va='center')

def draw_bottom_banner(ax, text):
    banner = FancyBboxPatch((0.96, 0.40), 14.08, 0.65,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1)
    ax.add_patch(banner)

    pill = FancyBboxPatch((1.15, 0.50), 1.45, 0.45,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor=C_ORANGE, edgecolor='none')
    ax.add_patch(pill)
    ax.text(1.875, 0.725, "ĐÚC KẾT", fontsize=10.5, fontweight='bold', color=C_WHITE, va='center', ha='center')

    wrapped = textwrap.fill(text, width=125)
    ax.text(2.80, 0.725, wrapped, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')

# =============================================================================
# SLIDE 1: MỤC 4.1 · BẢNG SO SÁNH ĐỐI CHUẨN SOTA CƠ SỞ
# =============================================================================
def render_slide_1():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "04 / 18",
                "Bảng So sánh Đối chuẩn SOTA Cơ sở trên Tập Chuẩn SHWD (VOC2028)")

    table_card = FancyBboxPatch((0.96, 3.25), 14.08, 4.35,
                                boxstyle="round,pad=0.01,rounding_size=0.1",
                                facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(table_card)

    cols = [
        ("Kiến trúc Mô hình", 3.48, 'left'),
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
        ("YOLO11s (Baseline Chính)", "9.40", "21.5", "94.74", "62.54", "94.06", "90.35", "0.9073"),
        ("Rep-YOLO11s (Mục tiêu Đề tài)", "9.85", "22.4", "94.83 (5-Fold: 96.64)", "62.54", "94.44", "91.33", "0.9190"),
    ]

    header_y = 7.30
    row_h = 0.50

    h_bg = FancyBboxPatch((0.96, 7.05), 14.08, 0.55,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor=C_TITLE, edgecolor='none')
    ax.add_patch(h_bg)

    cur_x = 0.96
    for title, w, align in cols:
        pos_x = cur_x + 0.20 if align == 'left' else cur_x + w / 2.0
        ax.text(pos_x, header_y, title, fontsize=10.2, fontweight='bold', color=C_WHITE,
                va='center', ha=align)
        cur_x += w

    for r_idx, row in enumerate(table_data):
        y_center = 6.80 - r_idx * row_h
        is_champ = (r_idx == len(table_data) - 1)
        is_alt = (r_idx % 2 == 1) and not is_champ

        if is_champ:
            r_bg = FancyBboxPatch((0.96, y_center - 0.22), 14.08, row_h - 0.04,
                                  boxstyle="round,pad=0.01,rounding_size=0.08",
                                  facecolor='#FEF3C7', edgecolor=C_ORANGE_BORDER, linewidth=1.5)
            ax.add_patch(r_bg)
        elif is_alt:
            r_bg = patches.Rectangle((0.96, y_center - 0.22), 14.08, row_h - 0.04,
                                     facecolor='#F8FAFC', edgecolor='none')
            ax.add_patch(r_bg)

        if not is_champ:
            ax.plot([0.96, 15.04], [y_center - 0.24, y_center - 0.24], color='#F1F5F9', lw=0.8)

        cur_x = 0.96
        for c_idx, val in enumerate(row):
            w = cols[c_idx][1]; align = cols[c_idx][2]
            pos_x = cur_x + 0.20 if align == 'left' else cur_x + w / 2.0
            font_weight = 'bold' if (is_champ or c_idx == 0) else 'normal'
            font_color = ('#9A3412' if c_idx in [0, 3, 6, 7] else C_TITLE) if is_champ else (C_TITLE if c_idx == 0 else C_BODY)
            font_size = 10.0 if is_champ else 9.4
            ax.text(pos_x, y_center, val, fontsize=font_size, fontweight=font_weight,
                    color=font_color, va='center', ha=align)
            cur_x += w

    c_w = 4.48; c_gap = 0.32; c_top = 1.30; c_h = 1.70
    insights = [
        ("01", C_BLUE, "Ưu thế Recall Sinh mạng (91.33%)",
         "Recallhat đạt đỉnh 91.33%, cao hơn YOLO11s (90.35%) và YOLOv8s (90.62%). Trong an toàn lao động, bỏ sót vi phạm là rủi ro tử vong; mô hình đề tài tối đa hóa khả năng cứu sinh."),
        ("02", C_ORANGE, "Cân bằng Tham số & Độ phức tạp",
         "9.85M params & 22.4 GFLOPs xấp xỉ YOLO11s gốc (9.40M/21.5G), nhẹ hơn 28% so với YOLOv8s (11.24M/28.6G). Tích hợp 4 module cải tiến mà không gây bùng nổ tài nguyên tính toán."),
        ("03", C_GREEN, "Độ ổn định Thống kê 5-Fold (96.64%)",
         "Trên quy trình 5-Fold Stratified CV, mô hình đạt đỉnh 96.64% +- 0.32% mAP50. Chứng minh độ tin cậy vượt trội, loại trừ hoàn toàn yếu tố may rủi khi phân chia tập dữ liệu.")
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
        ax.text(c_left + 0.75, c_top + c_h - 0.27, ins_title, fontsize=11.0, fontweight='bold', color=C_TITLE, va='center')

        wrapped_desc = textwrap.fill(ins_desc, width=54)
        ax.text(c_left + 0.20, c_top + 0.65, wrapped_desc, fontsize=9.2, color=C_BODY, va='center', linespacing=1.35)

    draw_bottom_banner(ax, "Bảng đối chuẩn khách quan trên 7,581 ảnh SHWD chứng minh Rep-YOLO11s vượt trội về Recallhat (91.33%) và F1hat (0.9190) với chi phí phần cứng tối ưu.")

    out_file = os.path.join(OUT_DIR, "Slide_04_1_Bang_So_Sanh_Baseline.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 2: MỤC 4.2 · BASELINE 1: YOLO11s
# =============================================================================
def render_slide_2():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05 / 18",
                "Phân tích Chi tiết Baseline 1: YOLO11s (Ultralytics, 10/2024 · SOTA Mới nhất)")

    c_w = 6.88; c_gap = 0.32; c_top = 1.30; c_h = 6.30

    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_BLUE, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "Kiến trúc & Năng lực Thực nghiệm trên SHWD", fontsize=13.0, fontweight='bold', color=C_TITLE, va='center')

    y11_left = [
        ("Công bố & Phát hành", "Phát hành bởi Ultralytics (Tháng 10/2024), đại diện cho thế hệ mô hình phát hiện vật thể SOTA mới nhất."),
        ("Cấu trúc Mạng Đặc trưng", "Backbone CSPDarknet cải tiến với các khối C3k2 và khối tự chú ý không gian C2PSA; Neck PANet trích xuất đa quy mô."),
        ("3 Tầng Đầu ra Phát hiện", "P3 (stride 8: 80x80), P4 (stride 16: 40x40), P5 (stride 32: 20x20) kết hợp Decoupled Head không neo (Anchor-free)."),
        ("Tập Dữ liệu Huấn luyện", "Pretrained trên COCO và fine-tune trực tiếp trên SHWD (7,581 ảnh authentic) với phân chia 80/20 chuẩn."),
        ("Kết quả Đối chuẩn Thực nghiệm", "mAP50 = 94.74% · mAP50-95 = 62.54% · Recallhat = 90.35% · F1hat = 0.9073 (Params: 9.40M · FLOPs: 21.5G)."),
        ("Ưu điểm Nổi bật", "Tối ưu hóa số lượng tham số tốt, tốc độ hội tụ nhanh, thông lượng xử lý cao trên các nền tảng GPU hiện đại.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y11_left:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=10.0, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.2, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_ORANGE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_ORANGE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "3 Điểm nghẽn Cốt lõi & Bài học Kế thừa / Cải tiến", fontsize=13.0, fontweight='bold', color=C_TITLE, va='center')

    y11_right = [
        ("Điểm nghẽn 1 (Mất dấu mũ ở xa)", "Các tầng downsampling tích lũy (stride 8/16/32) làm tiêu biến hoàn toàn tín hiệu của mũ bảo hộ siêu nhỏ (<20px)."),
        ("Điểm nghẽn 2 (Báo động giả sàn >28%)", "Tích chập tiêu chuẩn có tính Bất biến Tịnh tiến, gây nhầm lẫn xô vữa, biển cảnh báo màu vàng dưới sàn thành mũ."),
        ("Điểm nghẽn 3 (Triệt tiêu Gradient CIoU)", "Hàm loss CIoU bị triệt tiêu gradient khi tỷ lệ aspect ratio w/h của bounding box dự đoán trùng với ground truth."),
        ("Nhóm học được gì từ YOLO11s?", "Học được cấu trúc khối trích xuất đặc trưng C3k2 gọn nhẹ và nguyên lý phân tách hai nhánh phân loại / định vị (Decoupled Head)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa bộ khung YOLO11s nhưng thay bằng RepConv (gộp đại số về 1 Conv 3x3), cấy CoordConv Stem, bổ sung BiFormer Neck và thay bằng Focal-EIoU.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y11_right:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=10.0, fontweight='bold', color=C_ORANGE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.2, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    draw_bottom_banner(ax, "YOLO11s đại diện cho SOTA kiến trúc mới nhất (10/2024); nhóm kế thừa sườn mạng nhưng giải quyết triệt để 3 điểm nghẽn bằng 4 module toán học.")

    out_file = os.path.join(OUT_DIR, "Slide_04_2_Baseline_1_YOLO11s.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 3: MỤC 4.2 · BASELINE 2: YOLOv8s
# =============================================================================
def render_slide_3():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "06 / 18",
                "Phân tích Chi tiết Baseline 2: YOLOv8s (Jocher et al., 2023 · Chuẩn mực Công nghiệp)")

    c_w = 6.88; c_gap = 0.32; c_top = 1.30; c_h = 6.30

    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_AMBER, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_AMBER, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "Kiến trúc & Vị thế Mốc chuẩn Công nghiệp (Gold Standard)", fontsize=13.0, fontweight='bold', color=C_TITLE, va='center')

    y8_left = [
        ("Vị thế Công nghiệp Toàn cầu", "Được phát triển bởi Jocher et al. (2023), YOLOv8 là mốc chuẩn công nghiệp phổ biến và được ứng dụng rộng rãi nhất thế giới hiện nay."),
        ("Cấu trúc Khối Trích xuất C2f", "Cross Stage Partial with 2 Convolutions (C2f) kết hợp nhiều kết nối residual nội bộ giúp làm giàu dòng chảy gradient."),
        ("Cơ chế Gán nhãn Động (TAL)", "Task-Aligned Assigner đo lường sự kết hợp giữa độ tự tin phân loại và căn chỉnh IoU để gán nhãn động cho top-k anchors tốt nhất."),
        ("Tập Dữ liệu Huấn luyện", "Thực nghiệm trực tiếp trên tập SHWD chuẩn (7,581 ảnh) trong cùng điều kiện phần cứng với các baseline khác."),
        ("Kết quả Đối chuẩn Thực nghiệm", "mAP50 = 94.89% · mAP50-95 = 62.21% · Recallhat = 90.62% · F1hat = 0.9120 (Params: 11.24M · FLOPs: 28.6G)."),
        ("Ưu điểm Nổi bật", "Độ chính xác nhận diện tổng thể mAP50 cao, cộng đồng mã nguồn mở khổng lồ, mức độ ổn định sản xuất đã được kiểm chứng.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y8_left:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=10.0, fontweight='bold', color=C_AMBER, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.2, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_PURPLE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_PURPLE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "Hạn chế Phần cứng & Bài học Kế thừa / Cải tiến", fontsize=13.0, fontweight='bold', color=C_TITLE, va='center')

    y8_right = [
        ("Hạn chế 1 (Chi phí Tính toán Nặng)", "FLOPs lên tới 28.6G và 11.24M tham số, nặng hơn 28% so với Rep-YOLO11s (22.4G / 9.85M), tạo áp lực lớn khi triển khai biên."),
        ("Hạn chế 2 (Nghẽn Băng thông Bộ nhớ DRAM)", "Cấu trúc C2f phân nhánh nội bộ liên tục sử dụng nhiều thao tác nối tensor (concat) làm tăng lưu lượng truy cập DRAM và độ trễ truy xuất."),
        ("Hạn chế 3 (Thiếu Cơ chế Không gian)", "Không có thông tin tọa độ dọc, mô hình vẫn mắc phải tỷ lệ báo động giả cao đối với các thiết bị màu vàng dưới sàn nhà."),
        ("Nhóm học được gì từ YOLOv8s?", "Học tập cơ chế gán nhãn động Task-Aligned Assigner và chiến lược cân bằng loss đa nhiệm (BCE Classification Loss + DFL Loss)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa tư tưởng Task-Aligned Assigner nhưng xây dựng trên nền tảng YOLO11s gọn nhẹ hơn, áp dụng RepConv loại bỏ hoàn toàn chi phí bộ nhớ phân nhánh khi triển khai.")
    ]

    y_pos = c_top + c_h - 0.95
    for tag, desc in y8_right:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=10.0, fontweight='bold', color=C_PURPLE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.2, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    draw_bottom_banner(ax, "YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất; nhóm kế thừa cơ chế TAL gán nhãn động nhưng tối ưu triệt để chi phí tính toán và bộ nhớ DRAM.")

    out_file = os.path.join(OUT_DIR, "Slide_04_2_Baseline_2_YOLOv8s.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 4: MỤC 4.2 · SO SÁNH ĐỐI ĐẦU 2 BASELINES
# =============================================================================
def render_slide_4():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05-06 / 18",
                "Phân tích So sánh 2 Mô hình Baseline Chính: YOLO11s vs YOLOv8s")

    c_w = 6.88; c_gap = 0.32; c_top = 3.05; c_h = 4.55

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
    ax.text(1.25, c_top + c_h - 0.82, "SOTA Mới Nhất (Ultralytics, Tháng 10/2024)", fontsize=11.8, fontweight='bold', color=C_TITLE, va='center')

    c1_pts = [
        ("Kiến trúc mạng:", "C3k2 Feature Blocks + C2PSA Attention + Decoupled Head."),
        ("Thông số thực nghiệm:", "9.40M Params · 21.5 GFLOPs · mAP50 = 94.74% · Recall = 90.35%."),
        ("Ưu điểm:", "Tối ưu hóa số lượng tham số xuất sắc, tính toán nhanh, hội tụ nhanh."),
        ("Hạn chế cốt lõi:", "Mất tín hiệu mũ nhỏ ở xa (<20px); Báo động giả sàn nhà do bất biến tịnh tiến; Triệt tiêu gradient tỷ lệ khung CIoU."),
        ("Kế thừa & Cải tiến:", "Kế thừa sườn C3k2; thay thế RepConv, cấy CoordConv Stem, bổ sung BiFormer Neck và Focal-EIoU Head.")
    ]
    y_pos = c_top + c_h - 1.25
    for tag, desc in c1_pts:
        ax.text(1.25, y_pos, f"• {tag}", fontsize=9.6, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=66)
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
    ax.text(0.96 + c_w + c_gap + 0.25, c_top + c_h - 0.82, "Chuẩn Mực Công Nghiệp (Jocher et al., 2023)", fontsize=11.8, fontweight='bold', color=C_TITLE, va='center')

    c2_pts = [
        ("Kiến trúc mạng:", "C2f Cross Stage Partial Blocks + Task-Aligned Assigner (TAL)."),
        ("Thông số thực nghiệm:", "11.24M Params · 28.6 GFLOPs · mAP50 = 94.89% · Recall = 90.62%."),
        ("Ưu điểm:", "Độ chính xác mAP50 cao, mã nguồn cực kỳ ổn định, cộng đồng hỗ trợ lớn."),
        ("Hạn chế cốt lõi:", "28.6 GFLOPs nặng hơn 28% so với đề tài; Khối C2f ngốn băng thông DRAM; Báo động giả do thiếu tiên đề không gian."),
        ("Kế thừa & Cải tiến:", "Kế thừa cơ chế gán nhãn động TAL nhưng chuyển sang nền sườn YOLO11s gọn hơn và áp dụng RepConv xóa bỏ overhead DRAM.")
    ]
    y_pos = c_top + c_h - 1.25
    for tag, desc in c2_pts:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}", fontsize=9.6, fontweight='bold', color=C_AMBER, va='top')
        wrapped = textwrap.fill(desc, width=66)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.25, wrapped, fontsize=8.8, color=C_BODY, va='top', linespacing=1.28)
        y_pos -= 0.65

    # Bottom Spanning Card
    c_just = FancyBboxPatch((0.96, 1.30), 14.08, 1.55,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor='#F8FAFC', edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(c_just)

    ax.text(1.25, 2.55, "TẠI SAO NHÓM CHỌN 2 BASELINE NÀY ĐỂ BÁO CÁO REVIEW 1?",
            fontsize=11.2, fontweight='bold', color=C_ORANGE, va='center')

    just_p1 = textwrap.fill("• Tính Đại diện SOTA & Uy tín Khoa học: YOLO11s đại diện cho đỉnh cao kiến trúc mới nhất (tháng 10/2024), trong khi YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất thế giới (Gold Standard). Đối chuẩn đồng thời với cả hai tạo nền tảng vững chắc và khách quan tuyệt đối trước Hội đồng chấm.", width=130)
    ax.text(1.25, 2.10, just_p1, fontsize=9.0, color=C_BODY, va='center', linespacing=1.3)

    just_p2 = textwrap.fill("• Nền tảng Minh chứng Cải tiến: Mọi cải tiến của Rep-YOLO11s đều xuất phát từ việc khắc phục chính xác các điểm nghẽn kỹ thuật được chỉ ra ở hai mô hình cơ sở này, chứng minh tính cấp thiết và giá trị khoa học thực sự của đề tài.", width=130)
    ax.text(1.25, 1.62, just_p2, fontsize=9.0, color=C_BODY, va='center', linespacing=1.3)

    draw_bottom_banner(ax, "Đối chuẩn song song SOTA mới nhất (YOLO11s) và Chuẩn công nghiệp (YOLOv8s) tạo cơ sở khoa học khách quan chứng minh tính vượt trội của đề tài.")

    out_file = os.path.join(OUT_DIR, "Slide_04_2_So_Sanh_Doi_Dau_Baselines.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 5: MỤC 6.1 · KIẾN TRÚC TỔNG THỂ REP-YOLO11s
# =============================================================================
def render_slide_5():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 6 · PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED METHOD)", "07 / 18",
                "Kiến trúc Tổng thể Rep-YOLO11s: Tích hợp 4 Cải tiến Toán học Đột phá")

    c_flow_w = 8.16; c_top = 1.30; c_h = 6.30
    card_flow = FancyBboxPatch((0.96, c_top), c_flow_w, c_h,
                               boxstyle="round,pad=0.01,rounding_size=0.1",
                               facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(card_flow)

    f_tag = FancyBboxPatch((1.20, c_top + c_h - 0.45), 4.20, 0.32,
                           boxstyle="round,pad=0.01,rounding_size=0.08",
                           facecolor=C_SLATE, edgecolor='none')
    ax.add_patch(f_tag)
    ax.text(3.30, c_top + c_h - 0.29, "PIPELINE KIẾN TRÚC TỔNG THỂ REP-YOLO11s", fontsize=9.5, fontweight='bold', color=C_WHITE, va='center', ha='center')

    flow_steps = [
        ("ẢNH ĐẦU VÀO [640×640×3]", "Luồng Video Camera Giám sát Công trường Xây dựng (CCTV Feed)", C_SLATE, False),
        ("CẢI TIẾN 2: COORDCONV STEM", "Cấy 2 kênh tọa độ [Cx, Cy] vào Layer 0 (5 -> 64 kênh) -> Khử báo động sàn", C_ORANGE, True),
        ("CẢI TIẾN 1: REPCONV BACKBONE", "3 nhánh huấn luyện -> Gộp đại số về 1 Conv 3x3 khi triển khai (Zero Latency)", C_BLUE, True),
        ("CẢI TIẾN 3: BIFORMER NECK", "Định tuyến thưa 2 cấp độ Top-k (Complexity O(HW)) -> Bắt trúng mũ nhỏ xa", C_PURPLE, True),
        ("CẢI TIẾN 4: FOCAL-EIoU HEAD", "Phân rã kích thước độc lập + TAL gán nhãn động -> Tránh suy biến gradient", C_AMBER, True),
        ("KẾT QUẢ ĐẦU RA (OUTPUT)", "Khung bao mũ bảo hộ sắc nét, Recall >91%, triệt tiêu báo động giả mặt sàn", C_GREEN, False)
    ]

    step_top_start = c_top + c_h - 0.75
    step_h = 0.72; step_gap = 0.20

    for idx, (title, sub, col, is_mod) in enumerate(flow_steps):
        s_y = step_top_start - (idx + 1) * step_h - idx * step_gap
        if idx > 0:
            ax.annotate('', xy=(5.04, s_y + step_h), xytext=(5.04, s_y + step_h + step_gap),
                        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.5, mutation_scale=12))

        bg_col = C_ORANGE_LIGHT if is_mod else '#F8FAFC'
        border_col = col if is_mod else C_BORDER
        sc = FancyBboxPatch((1.20, s_y), c_flow_w - 0.48, step_h,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            facecolor=bg_col, edgecolor=border_col, linewidth=1.5 if is_mod else 1)
        ax.add_patch(sc)

        bar = FancyBboxPatch((1.28, s_y + 0.08), 0.14, step_h - 0.16,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             facecolor=col, edgecolor='none')
        ax.add_patch(bar)

        ax.text(1.56, s_y + step_h - 0.22, title, fontsize=10.5, fontweight='bold', color=col, va='center')
        ax.text(1.56, s_y + 0.22, sub, fontsize=8.8, color=C_TITLE if is_mod else C_MUTED, va='center')

    # Right Column: 4 Cards
    c_right_w = 5.60; c_right_left = 9.44
    mod_cards = [
        ("Cải tiến 1: Tái tham số hóa RepConv", C_BLUE,
         "Huấn luyện đa nhánh giàu biểu diễn (3x3, 1x1, Identity) -> Gộp đại số khép kín về duy nhất 1 nhân Conv 3x3 khi suy luận (`switch_to_deploy`). Đạt tốc độ cực hạn với Zero Latency Overhead."),
        ("Cải tiến 2: Mã hóa Tọa độ CoordConv", C_ORANGE,
         "Cấy trực tiếp 2 kênh tọa độ không gian Cartesian [Cx, Cy] vào tầng Stem đầu vào. Phá vỡ tính bất biến tịnh tiến sai lầm, triệt tiêu dứt điểm báo động giả xô vàng và áo phản quang dưới sàn."),
        ("Cải tiến 3: Chú ý Định tuyến BiFormer", C_PURPLE,
         "Cơ chế định tuyến thưa 2 cấp độ lọc bỏ 93.75% các vùng nền rậm rạp không liên quan, tập trung năng lực tính toán vào các vùng chỏm đầu công nhân ở xa với độ phức tạp tuyến tính O(HW)."),
        ("Cải tiến 4: Hàm mất mát Focal-EIoU", C_AMBER,
         "Phân rã độc lập sai số chiều rộng và chiều cao thay vì tỷ lệ khung bao aspect ratio của CIoU, ngăn ngừa triệt tiêu gradient. Kết hợp trọng số tiêu điểm tập trung tối đa cho mẫu mục tiêu nhỏ khó.")
    ]

    card_h = 1.48; card_gap = 0.13; card_top_start = c_top + c_h
    for idx, (m_title, m_col, m_desc) in enumerate(mod_cards):
        m_y = card_top_start - (idx + 1) * card_h - idx * card_gap
        mc = FancyBboxPatch((c_right_left, m_y), c_right_w, card_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(mc)

        strip = FancyBboxPatch((c_right_left + 0.16, m_y + 0.12), 0.12, card_h - 0.24,
                               boxstyle="round,pad=0.01,rounding_size=0.05",
                               facecolor=m_col, edgecolor='none')
        ax.add_patch(strip)

        ax.text(c_right_left + 0.42, m_y + card_h - 0.30, m_title, fontsize=10.8, fontweight='bold', color=m_col, va='center')
        wrapped_desc = textwrap.fill(m_desc, width=54)
        ax.text(c_right_left + 0.42, m_y + 0.52, wrapped_desc, fontsize=8.6, color=C_BODY, va='center', linespacing=1.35)

    draw_bottom_banner(ax, "Rep-YOLO11s kết hợp 4 cải tiến toán học giải quyết đồng thời 4 rào cản: Khử báo động sàn, suy luận không độ trễ, tập trung mục tiêu xa và hồi quy hộp chính xác.")

    out_file = os.path.join(OUT_DIR, "Slide_06_1_Kien_Truc_Rep_YOLO11s.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 6: MỤC 7 · KỸ THUẬT DỮ LIỆU & QUY TRÌNH DATA ENGINEERING
# =============================================================================
def render_slide_6():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 7 · TẬP DỮ LIỆU & QUY TRÌNH DATA ENGINEERING", "08 / 18",
                "Kỹ thuật Dữ liệu: Làm sạch Dị biệt, Cân bằng Lớp & Chuẩn hóa Không gian Nhãn C*")

    c_w = 6.88; c_gap = 0.32; c_top = 1.30; c_h = 6.30

    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_BLUE, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "Tập Dữ liệu Huấn luyện Trong miền: SHWD (VOC2028)", fontsize=12.2, fontweight='bold', color=C_TITLE, va='center')

    shwd_pts = [
        ("Quy mô & Nguồn gốc", "7,581 ảnh chụp công trường xây dựng thực tế với độ phân giải cao và góc quay camera giám sát đa dạng."),
        ("Phân chia Chuẩn 80/20", "6,064 ảnh TrainVal và 1,517 ảnh Test độc lập. Gom cụm các khung hình liên tiếp để bảo đảm Zero Data Leakage tuyệt đối."),
        ("Lọc sạch Dị biệt (Cleaning)", "Phát hiện và thanh lọc triệt để 3 nhãn chú thích XML rác bị gán nhầm thành 'dog' trong tập gốc SHWD."),
        ("Mất cân bằng Cực đoan 1:12", "Gồm 9,044 nhãn mũ bảo hộ ('hat') áp đảo bởi 111,514 nhãn thân người ('person')."),
        ("Khắc phục Mất cân bằng", "Áp dụng Focal BCE Loss kết hợp Task-Aligned Assigner (TAL) giúp mô hình tập trung gradient vào mũ bảo hộ mà không làm suy giảm độ chính xác thân người."),
        ("Data Augmentation", "Mosaic (p=1.0, tắt 10 epoch cuối), Mixup (p=0.15), Random Affine (+-10%), lật ngang ảnh (p=0.5) và HSV Jitter.")
    ]
    y_pos = c_top + c_h - 0.95
    for tag, desc in shwd_pts:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=9.8, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.0, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_ORANGE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_ORANGE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "5 Tập Dữ liệu Mở rộng & Không gian Nhãn C*", fontsize=12.2, fontweight='bold', color=C_TITLE, va='center')

    cross_pts = [
        ("GDUT-HWD (13,499 ảnh)", "Môi trường công nhân cực kỳ đông đúc (15–30 người/khung hình) với mật độ che khuất lẫn nhau dày đặc."),
        ("SHEL5K (5,000 ảnh)", "Góc nhìn camera quan sát CCTV góc cao, bao quát từ trên xuống với thách thức che khuất đa quy mô và tính đa nghĩa nhãn."),
        ("Hard Hat Workers (7,000 ảnh)", "Công trường xây dựng ngoài trời với độ tương phản ánh sáng gắt và bóng đổ phức tạp."),
        ("SHD & SFCHD Benchmark", "Nhà máy luyện kim, xưởng đóng tàu và tổ hợp hóa dầu với bụi bẩn và ánh sáng nhân tạo yếu."),
        ("Chuẩn hóa Nhãn Chung C*", "Đồng bộ hóa các quy ước gán nhãn xung đột về một không gian nhãn thống nhất: C* = {0: 'hat' (Mũ bảo hộ), 1: 'person' (Người lao động)}."),
        ("Ý nghĩa Khoa học", "Đảm bảo khả năng chuyển giao tổng quát hóa Zero-Shot vững chắc trên mọi hệ thống camera công trường thực tế chưa từng thấy khi huấn luyện.")
    ]
    y_pos = c_top + c_h - 0.95
    for tag, desc in cross_pts:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=9.8, fontweight='bold', color=C_ORANGE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.0, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.88

    draw_bottom_banner(ax, "Quy trình tiền xử lý nghiêm ngặt loại bỏ rò rỉ dữ liệu, khử nhãn rác, cân bằng tỉ lệ 1:12 và đồng bộ nhãn C* trên hơn 25,000 ảnh đa miền.")

    out_file = os.path.join(OUT_DIR, "Slide_07_Ky_Thuat_Du_Lieu_Data_Engineering.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 7: MỤC 8 · CHIẾN LƯỢC ĐÁNH GIÁ (EVALUATION STRATEGY)
# =============================================================================
def render_slide_7():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 8 · CHIẾN LƯỢC ĐÁNH GIÁ (EVALUATION STRATEGY)", "09 / 18",
                 "Chiến lược Đánh giá Đa chỉ số & Nguyên tắc Ưu tiên Recall Sinh mạng")

    card_specs = [
        (0.96, 4.45, "01", C_BLUE, "Các Chỉ số Nhận diện & Định vị Cốt lõi", [
            "mAP50 (Thước đo Chuẩn mực): Đo lường độ chính xác phát hiện tổng thể tại ngưỡng IoU 0.50.",
            "mAP50-95 (Độ ôm khít Khung bao): Trung bình nghiêm ngặt qua 10 bước IoU [0.50:0.05:0.95], phạt nặng các khung bao xộc xệch hoặc lệch tâm.",
            "Precision & F1-Score: Đánh giá sự cân bằng giữa phát hiện chính xác và hạn chế báo động nhầm.",
            "Ràng buộc Thời gian thực: Đảm bảo độ trễ suy luận đáp ứng luồng video mượt mà >25 FPS."
        ]),
        (8.16, 4.45, "02", C_ORANGE, "Ưu tiên Sinh mạng: Recall Quan trọng hơn Precision", [
            "Đặc thù An toàn Lao động: Bỏ sót một công nhân không đội mũ là nguy cơ tử vong không thể cứu vãn.",
            "Phân tích Chi phí Sai số Không đối xứng: Cảnh báo thừa (False Alarm) chỉ mất 2 giây kiểm tra; Bỏ sót (False Negative) đánh đổi bằng tính mạng con người.",
            "Ngưỡng Tối ưu hóa: Tinh chỉnh hàm mất mát để tối đa hóa Recallhat (>91%) mà vẫn giữ vững mAP50.",
            "Mục tiêu Cứu sinh: Mô hình được thiết kế hướng tới sứ mệnh bảo vệ an toàn tối đa cho công nhân."
        ]),
        (0.96, 1.30, "03", C_PURPLE, "Kiểm định Chéo 5 Lớp (5-Fold Stratified CV)", [
            "Phân chia Đồng đều 5 Fold: Bảo đảm tỉ lệ nhãn 1:12.33 không bị lệch lạc giữa các fold.",
            "Loại trừ May rủi Dữ liệu: Đánh giá mô hình trên 5 lần phân chia độc lập để tránh hiện tượng cherry-picking.",
            "Báo cáo Trung bình & Độ lệch chuẩn: Trình bày số liệu dạng mu +- sigma khẳng định tính tin cậy khoa học.",
            "Thống kê Vững chắc: Đạt 96.64% mAP50 (+-0.32%) chứng minh sự ổn định tuyệt đối của kiến trúc."
        ]),
        (8.16, 1.30, "04", C_GREEN, "Minh bạch Mô hình & Trích xuất Nhiệt XAI (Grad-CAM)", [
            "Trích xuất Gradient Tầng Cổ (Neck): Quan sát trực quan vùng trọng tâm kích hoạt của mạng nơ-ron.",
            "Khử Nhiễu Bối cảnh: Kiểm chứng mô hình tập trung vào chỏm đầu công nhân thay vì bị đánh lừa bởi áo phản quang hay xô vữa dưới sàn.",
            "Phân tích Sai số Trực quan: Khoanh vùng chính xác các trường hợp dự đoán sai để làm sáng tỏ bản chất bài toán.",
            "Minh bạch Học thuật Tuyệt đối: Đảm bảo mô hình có thể giải thích được trước Hội đồng Đánh giá."
        ])
    ]

    for c_left, c_top, b_num, b_col, card_title, card_items in card_specs:
        sc = FancyBboxPatch((c_left, c_top), 6.88, 3.00,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(sc)

        sb = FancyBboxPatch((c_left + 0.20, c_top + 3.00 - 0.45), 0.45, 0.30,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            facecolor=b_col, edgecolor='none')
        ax.add_patch(sb)
        ax.text(c_left + 0.425, c_top + 3.00 - 0.30, b_num, fontsize=10.5, fontweight='bold', color=C_WHITE, va='center', ha='center')
        ax.text(c_left + 0.78, c_top + 3.00 - 0.30, card_title, fontsize=11.2, fontweight='bold', color=C_TITLE, va='center')

        y_p = c_top + 3.00 - 0.75
        for itm in card_items:
            wrapped = textwrap.fill(itm, width=70)
            ax.text(c_left + 0.25, y_p, "• " + wrapped, fontsize=8.8, color=C_BODY, va='top', linespacing=1.28)
            y_p -= 0.54

    draw_bottom_banner(ax, "Chiến lược đánh giá toàn diện ưu tiên bảo vệ sinh mạng công nhân, kết hợp kiểm định thống kê 5-Fold và giải thích trực quan bằng Grad-CAM.")

    out_file = os.path.join(OUT_DIR, "Slide_08_Chien_Luoc_Danh_Gia_Evaluation.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 8: MỤC 9 · KẾ HOẠCH THỰC HIỆN KHÓA LUẬN (PROJECT PLAN)
# =============================================================================
def render_slide_8():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 9 · KẾ HOẠCH THỰC HIỆN KHÓA LUẬN (PROJECT PLAN)", "10 / 18",
                "Kế hoạch Triển khai Khóa luận & Biểu đồ Mốc tiến độ Gantt (W1 - W15)")

    phase_cards = [
        ("Giai đoạn 1 (Tuần 1-5)", C_BLUE, "Nền tảng & Báo cáo Review 1", [
            "Khảo sát 30 nghiên cứu quốc tế, xác định RQ & Research Gap.",
            "Làm sạch tập dữ liệu SHWD, khử nhãn rác dog, đồng bộ nhãn C*.",
            "Thiết lập các mô hình mốc chuẩn SOTA: YOLOv8s, YOLO11s.",
            "Sản phẩm: Slide & Báo cáo Review 1, Tập dữ liệu chuẩn hóa."
        ]),
        ("Giai đoạn 2 (Tuần 6-8)", C_ORANGE, "Kỹ thuật Module Cốt lõi & Review 2", [
            "Lập trình module toán học: RepConv, CoordConv, BiFormer, Focal-EIoU.",
            "Thực hiện chuỗi thực nghiệm Ablation Study A0 -> A6.",
            "Đo đạc thực nghiệm sơ bộ trên phần cứng GPU T4.",
            "Sản phẩm: Mã nguồn module hoàn chỉnh, Bảng số liệu Ablation."
        ]),
        ("Giai đoạn 3 (Tuần 9-11)", C_PURPLE, "Kiểm chứng Thống kê & Review 3", [
            "Thực thi kiểm định chéo 5-Fold Stratified Cross-Validation.",
            "Đánh giá năng lực Zero-shot trên 5 tập dữ liệu ngoại miền.",
            "Trích xuất bản đồ kích hoạt nhiệt Grad-CAM XAI.",
            "Sản phẩm: Bảng số liệu thống kê 5-Fold, Trọng số mô hình tốt nhất."
        ]),
        ("Giai đoạn 4-5 (Tuần 12-15)", C_GREEN, "Ứng dụng, Bài báo & Bảo vệ", [
            "Xây dựng ứng dụng giám sát video thời gian thực RTSP.",
            "Hoàn thiện bản thảo bài báo nghiên cứu khoa học chuẩn IEEE.",
            "Viết toàn văn Báo cáo Khóa luận Tốt nghiệp & Dựng slide bảo vệ.",
            "Sản phẩm: Ứng dụng demo RTSP, Bản thảo Paper, Quyển khóa luận."
        ])
    ]

    p_w = 3.28; p_gap = 0.32
    for idx, (p_phase, p_col, p_title, p_items) in enumerate(phase_cards):
        p_left = 0.96 + idx * (p_w + p_gap)
        sc = FancyBboxPatch((p_left, 3.10), p_w, 4.50,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(sc)

        hbar = FancyBboxPatch((p_left + 0.15, 7.15), p_w - 0.30, 0.35,
                              boxstyle="round,pad=0.01,rounding_size=0.08",
                              facecolor=p_col, edgecolor='none')
        ax.add_patch(hbar)
        ax.text(p_left + p_w / 2.0, 7.325, p_phase, fontsize=9.8, fontweight='bold', color=C_WHITE, va='center', ha='center')

        ax.text(p_left + 0.15, 6.85, p_title, fontsize=10.5, fontweight='bold', color=C_TITLE, va='center')

        y_p = 6.45
        for itm in p_items:
            wrapped = textwrap.fill(itm, width=32)
            ax.text(p_left + 0.15, y_p, "• " + wrapped, fontsize=8.4, color=C_BODY, va='top', linespacing=1.25)
            y_p -= 0.82

    # Bottom Workload Container
    c_alloc = FancyBboxPatch((0.96, 1.30), 14.08, 1.60,
                             boxstyle="round,pad=0.01,rounding_size=0.1",
                             facecolor='#F8FAFC', edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(c_alloc)

    ax.text(1.25, 2.65, "PHÂN CÔNG TRÁCH NHIỆM & ĐẢM BẢO CHẤT LƯỢNG KHÓA LUẬN (AGILE WORKLOAD ALLOCATION)",
            fontsize=10.8, fontweight='bold', color=C_ORANGE, va='center')

    alloc_pts = [
        ("Nguyễn Hàn Như (Trưởng nhóm / AI Lead):", "Thiết kế kiến trúc Rep-YOLO11s, lập trình module toán học, quy trình huấn luyện và chấp bút bản thảo bài báo khoa học."),
        ("Nguyễn Văn Thành (Data Engineering Lead):", "Lọc nhãn rác, làm sạch dữ liệu SHWD, chuẩn hóa không gian nhãn C*, thực thi kiểm định thống kê 5-Fold Cross-Validation."),
        ("Trần Phạm Tuấn Dũng (System & RTSP Lead):", "Xây dựng pipeline thu nhận luồng video RTSP, giải mã phần cứng NVDEC, tích hợp giao diện cảnh báo và chuẩn bị hồ sơ kỹ thuật.")
    ]
    y_al = 2.25
    for m_role, m_desc in alloc_pts:
        ax.text(1.25, y_al, f"• {m_role}", fontsize=9.2, fontweight='bold', color=C_TITLE, va='center')
        ax.text(5.40, y_al, m_desc, fontsize=9.0, color=C_BODY, va='center')
        y_al -= 0.40

    draw_bottom_banner(ax, "Kế hoạch 15 tuần theo mô hình Agile lặp gắn chặt với 3 mốc Review, phân công trách nhiệm minh bạch, bảo đảm hoàn thành 100% đúng tiến độ.")

    out_file = os.path.join(OUT_DIR, "Slide_09_Ke_Hoach_Thuc_Hien_Gantt.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 9: MỤC 10 · KẾ HOẠCH BÀI BÁO KHOA HỌC CHUẨN QUỐC TẾ
# =============================================================================
def render_slide_9():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 10 · KẾ HOẠCH XUẤT BẢN BÀI BÁO KHOA HỌC", "11 / 18",
                 "Kế hoạch Xuất bản Bài báo Nghiên cứu Khoa học Chuẩn Quốc tế (IEEE Format)")

    c_w = 6.88; c_gap = 0.32; c_top = 1.30; c_h = 6.30

    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_BLUE, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "Bản thảo Nghiên cứu Khoa học Chuẩn Sẵn sàng Nộp", fontsize=12.2, fontweight='bold', color=C_TITLE, va='center')

    paper_pts = [
        ("Định dạng Trình bày", "Soạn thảo hoàn chỉnh theo chuẩn định dạng IEEE hai cột (Two-Column Conference/Journal Template) với đầy đủ phương trình toán học và biểu đồ đối chuẩn."),
        ("Cấu trúc Học thuật Toàn diện", "Bao gồm Abstract, 4 Câu hỏi Nghiên cứu, Phân tích 30 Công trình SOTA, Kiến trúc Đề xuất, Kết quả Đối chuẩn Đa chiều và Phân tích XAI."),
        ("Chiến lược Xuất bản An toàn", "Chuẩn bị bản thảo nghiên cứu hoàn chỉnh, chất lượng cao, sẵn sàng trình Hội đồng Đánh giá Khóa luận và nộp tới các hội nghị / tạp chí quốc tế uy tín."),
        ("Minh bạch Học thuật", "Mọi đồ thị Pareto, bảng số liệu thực nghiệm và sơ đồ kiến trúc đều được trích xuất 100% từ mã nguồn PyTorch thực thi, không suy diễn."),
        ("Chia sẻ Mã nguồn Mở", "Toàn bộ checkpoints mô hình, kịch bản huấn luyện và dữ liệu làm sạch được công khai trên GitHub phục vụ mục đích tái lập nghiên cứu (Reproducibility).")
    ]
    y_pos = c_top + c_h - 0.95
    for tag, desc in paper_pts:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=9.8, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.0, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.98

    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_ORANGE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_ORANGE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "4 Đóng góp Học thuật Cốt lõi của Bản thảo", fontsize=12.2, fontweight='bold', color=C_TITLE, va='center')

    contrib_pts = [
        ("Đóng góp 1 (RepConv không độ trễ)", "Chứng minh bằng đại số khả năng gộp 3 nhánh huấn luyện về 1 nhân Conv 3x3 duy nhất, triệt tiêu phân mảnh bộ nhớ và đạt Zero Latency Overhead."),
        ("Đóng góp 2 (CoordConv khử báo động)", "Cung cấp tiên nghiệm vị trí thẳng đứng phá vỡ tính bất biến tịnh tiến của CNN, triệt tiêu dứt điểm báo động giả xô vàng và đồ bảo hộ dưới sàn nhà."),
        ("Đóng góp 3 (BiFormer bắt mục tiêu xa)", "Đưa độ phức tạp tính toán chú ý từ bậc hai O(H^2W^2) về tuyến tính O(HW), giải quyết triệt để nguy cơ sập bộ nhớ CUDA OOM khi bắt mục tiêu mũ bảo hộ ở xa."),
        ("Đóng góp 4 (Phân rã Focal-EIoU Loss)", "Khắc phục hiện tượng triệt tiêu gradient của hàm CIoU bằng cách phân rã độc lập sai số kích thước w và h, giúp hồi quy khung bao nhỏ chính xác tối đa.")
    ]
    y_pos = c_top + c_h - 0.95
    for tag, desc in contrib_pts:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=9.8, fontweight='bold', color=C_ORANGE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.0, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 1.15

    draw_bottom_banner(ax, "Đề tài định hướng xây dựng bản thảo bài báo khoa học chuẩn quốc tế, chứng minh 4 đóng góp toán học thực chất và sẵn sàng trình Hội đồng nghiệm thu.")

    out_file = os.path.join(OUT_DIR, "Slide_10_Ke_Hoach_Bai_Bao_Khoa_Hoc.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 10: MỤC 11 · MA TRẬN QUẢN TRỊ RỦI RO (RISK MANAGEMENT)
# =============================================================================
def render_slide_10():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 11 · MA TRẬN QUẢN TRỊ RỦI RO & KHẮC PHỤC CHỦ ĐỘNG", "12 / 18",
                 "Ma trận Nhận diện Rủi ro Kỹ thuật & Phương án Khắc phục Chủ động")

    risk_cards = [
        (0.96, 4.45, "01", C_RED, "Rủi ro Thiếu hụt Tài nguyên Tính toán GPU", [
            "Mô tả Rủi ro: Thời gian huấn luyện lâu hoặc bị ngắt quãng do giới hạn quota GPU miễn phí trên nền tảng đám mây.",
            "Xác suất: Trung bình | Tác động: Cao.",
            "Biện pháp Khắc phục: Phân bổ mô hình huấn luyện sang Google Colab Pro và Kaggle GPU T4 x 2; sử dụng Mixed Precision FP16 và cơ chế Checkpoint Resume tự động lưu mỗi 5 epoch."
        ]),
        (8.16, 4.45, "02", C_ORANGE, "Rủi ro Quá khớp (Overfitting) & Rò rỉ Dữ liệu", [
            "Mô tả Rủi ro: Mô hình ghi nhớ dữ liệu tập train dẫn tới suy giảm nghiêm trọng độ chính xác khi đưa ra môi trường công trường thực tế.",
            "Xác suất: Trung bình | Tác động: Cao.",
            "Biện pháp Khắc phục: Kiểm định chéo 5-Fold Stratified CV, nhóm sequence ảnh camera ngăn ngừa Data Leakage; áp dụng Mosaic, Mixup và HSV Jitter tăng tính đa dạng."
        ]),
        (0.96, 1.30, "03", C_PURPLE, "Rủi ro Bất đồng Nhãn & Sai lệch Đa miền", [
            "Mô tả Rủi ro: 5 tập dữ liệu kiểm thử ngoại miền có quy ước gán nhãn mâu thuẫn (nhãn vẽ toàn thân vs vẽ chỏm đầu), gây suy giảm mAP.",
            "Xác suất: Cao | Tác động: Trung bình.",
            "Biện pháp Khắc phục: Thiết lập bộ quy chuẩn ánh xạ không gian nhãn chung C* = {0: 'hat', 1: 'person'}, tự động đồng bộ hóa toàn bộ bounding boxes ngoại miền."
        ]),
        (8.16, 1.30, "04", C_BLUE, "Rủi ro Trễ Hạn Hoàn thành Bài báo & Báo cáo", [
            "Mô tả Rủi ro: Khối lượng công việc nghiên cứu và kỹ thuật lớn dẫn tới nguy cơ trễ hạn nộp bản thảo bài báo và quyển khóa luận tốt nghiệp.",
            "Xác suất: Thấp | Tác động: Cao.",
            "Biện pháp Khắc phục: Áp dụng quy trình Agile sprint hàng tuần với bảng Kanban; Trưởng nhóm rà soát tiến độ định kỳ; hoàn thiện bản thảo song song với quá trình chạy mã nguồn."
        ])
    ]

    for c_left, c_top, b_num, b_col, card_title, card_items in risk_cards:
        sc = FancyBboxPatch((c_left, c_top), 6.88, 3.00,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=1)
        ax.add_patch(sc)

        sb = FancyBboxPatch((c_left + 0.20, c_top + 3.00 - 0.45), 0.45, 0.30,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            facecolor=b_col, edgecolor='none')
        ax.add_patch(sb)
        ax.text(c_left + 0.425, c_top + 3.00 - 0.30, b_num, fontsize=10.5, fontweight='bold', color=C_WHITE, va='center', ha='center')
        ax.text(c_left + 0.78, c_top + 3.00 - 0.30, card_title, fontsize=11.2, fontweight='bold', color=C_TITLE, va='center')

        y_p = c_top + 3.00 - 0.75
        for itm in card_items:
            wrapped = textwrap.fill(itm, width=70)
            ax.text(c_left + 0.25, y_p, "• " + wrapped, fontsize=8.8, color=C_BODY, va='top', linespacing=1.28)
            y_p -= 0.68

    draw_bottom_banner(ax, "Nhóm chủ động nhận diện 4 rủi ro kỹ thuật và thiết lập phương án dự phòng chi tiết, bảo đảm dự án vận hành an toàn và về đích đúng kế hoạch.")

    out_file = os.path.join(OUT_DIR, "Slide_11_Ma_Tran_Quan_Tri_Rui_Ro.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

# =============================================================================
# SLIDE 11: MỤC 12 · KẾT QUẢ KỲ VỌNG & TỔNG KẾT
# =============================================================================
def render_slide_11():
    fig, ax = init_canvas()
    draw_header(ax, "MỤC 12 · KẾT QUẢ KỲ VỌNG & TỔNG KẾT BẢO VỆ KHÓA LUẬN", "13 / 18",
                 "Sản phẩm Kỳ vọng Bàn giao & Tổng hợp Giá trị Đóng góp Khóa luận")

    c_w = 6.88; c_gap = 0.32; c_top = 1.30; c_h = 6.30

    card_l = FancyBboxPatch((0.96, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_GREEN, linewidth=1.5)
    ax.add_patch(card_l)

    badge_l = FancyBboxPatch((1.20, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_GREEN, edgecolor='none')
    ax.add_patch(badge_l)
    ax.text(1.45, c_top + c_h - 0.34, "01", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(1.85, c_top + c_h - 0.34, "5 Sản phẩm Kỳ vọng Bàn giao (Deliverables)", fontsize=12.2, fontweight='bold', color=C_TITLE, va='center')

    deliv_pts = [
        ("Mô hình Trọng số Rep-YOLO11s", "Bộ trọng số PyTorch (.pt) và ONNX/TensorRT đã tái tham số hóa, đạt mAP50 >94.8% và Recallhat >91%."),
        ("Pipeline Video Camera RTSP", "Ứng dụng xử lý luồng camera giám sát CCTV với giải mã phần cứng NVDEC, hiển thị cảnh báo vi phạm trực quan >25 FPS."),
        ("Bản thảo Bài báo Chuẩn IEEE", "Tài liệu khoa học hoàn chỉnh, sẵn sàng trình Hội đồng Đánh giá và nộp tới các hội nghị / tạp chí quốc tế uy tín."),
        ("Bộ Dữ liệu Chuẩn hóa C*", "Bộ dữ liệu SHWD đã làm sạch nhãn rác, ánh xạ C* cho 5 tập ngoại vi và kho mã nguồn GitHub minh bạch."),
        ("Toàn văn Báo cáo (Thesis Book)", "Quyển báo cáo khóa luận tốt nghiệp chi tiết, giải trình đầy đủ cơ sở lý thuyết, quy trình thực nghiệm và phân tích chuyên sâu.")
    ]
    y_pos = c_top + c_h - 0.95
    for tag, desc in deliv_pts:
        ax.text(1.25, y_pos, f"• {tag}:", fontsize=9.8, fontweight='bold', color=C_GREEN, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(1.45, y_pos - 0.28, wrapped, fontsize=9.0, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.98

    card_r = FancyBboxPatch((0.96 + c_w + c_gap, c_top), c_w, c_h,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=C_WHITE, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(card_r)

    badge_r = FancyBboxPatch((0.96 + c_w + c_gap + 0.24, c_top + c_h - 0.52), 0.50, 0.36,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor=C_BLUE, edgecolor='none')
    ax.add_patch(badge_r)
    ax.text(0.96 + c_w + c_gap + 0.49, c_top + c_h - 0.34, "02", fontsize=12, fontweight='bold', color=C_WHITE, va='center', ha='center')
    ax.text(0.96 + c_w + c_gap + 0.89, c_top + c_h - 0.34, "Tổng kết Giá trị & Khẳng định Khóa luận", fontsize=12.2, fontweight='bold', color=C_TITLE, va='center')

    synth_pts = [
        ("Ý nghĩa Thực tiễn Cấp bách", "Giải quyết bài toán an toàn lao động tối quan trọng: ngăn ngừa tai nạn chấn thương đầu tử vong tại công trường xây dựng."),
        ("Cơ sở Khoa học Vững chắc", "Khắc phục triệt để các khoảng trống nghiên cứu được chỉ ra từ hơn 30 công trình quốc tế giai đoạn 2019-2026."),
        ("4 Cải tiến Toán học Đột phá", "Kết hợp RepConv (Zero Latency), CoordConv (Khử báo động sàn), BiFormer (Bắt mũ nhỏ) và Focal-EIoU (Hồi quy sắc nét)."),
        ("Đối chuẩn Khách quan & Vượt trội", "Chứng minh vượt trội về Recallhat (91.33%) và F1hat (0.9190) trước cả hai baseline YOLO11s và YOLOv8s."),
        ("Kết hợp Nghiên cứu & Ứng dụng", "Cầu nối hài hòa giữa sự chuẩn mực, khắt khe của nghiên cứu học thuật và giá trị thực tế bảo vệ sinh mạng con người.")
    ]
    y_pos = c_top + c_h - 0.95
    for tag, desc in synth_pts:
        ax.text(0.96 + c_w + c_gap + 0.25, y_pos, f"• {tag}:", fontsize=9.8, fontweight='bold', color=C_BLUE, va='top')
        wrapped = textwrap.fill(desc, width=72)
        ax.text(0.96 + c_w + c_gap + 0.45, y_pos - 0.28, wrapped, fontsize=9.0, color=C_BODY, va='top', linespacing=1.3)
        y_pos -= 0.98

    draw_bottom_banner(ax, "Một khóa luận AI hoàn chỉnh và chuẩn mực, kết hợp hài hòa giữa độ chính xác cứu sinh, phần mềm thực thi mạnh mẽ và giá trị khoa học đích thực.")

    out_file = os.path.join(OUT_DIR, "Slide_12_Ket_Qua_Ky_Vong_Tong_Ket.png")
    plt.savefig(out_file, dpi=300, facecolor=C_BG)
    plt.close()
    print("Rendered:", os.path.basename(out_file))

if __name__ == '__main__':
    render_slide_1()
    render_slide_2()
    render_slide_3()
    render_slide_4()
    render_slide_5()
    render_slide_6()
    render_slide_7()
    render_slide_8()
    render_slide_9()
    render_slide_10()
    render_slide_11()
    print("All 11 Vietnamese slides successfully rendered at 300 DPI!")
