"""
Script to generate PowerPoint slides for Sections 4, 5, and 6
of the Capstone Review 1 presentation.

Section 4.1: SOTA Baseline Comparison Table
Section 4.2: Baseline 1 (YOLO11s) & Baseline 2 (YOLOv8s) Analysis
Section 6.1: Overall Proposed Architecture (Rep-YOLO11s)

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

    # Theme Colors
    c_bg = RGBColor(0xFA, 0xF8, 0xF5)
    c_white = RGBColor(0xFF, 0xFF, 0xFF)
    c_border = RGBColor(0xE2, 0xE8, 0xF0)
    c_orange = RGBColor(0xEA, 0x58, 0x0C)
    c_orange_light = RGBColor(0xFF, 0xF7, 0xED)
    c_orange_border = RGBColor(0xFD, 0xBA, 0x74)
    c_blue = RGBColor(0x25, 0x63, 0xEB)
    c_amber = RGBColor(0xD9, 0x77, 0x06)
    c_green = RGBColor(0x16, 0xA3, 0x4A)
    c_purple = RGBColor(0x7C, 0x3A, 0xED)
    c_slate = RGBColor(0x47, 0x55, 0x69)
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

    # =========================================================================
    # SLIDE 1: MỤC 4.1 · BẢNG SO SÁNH ĐỐI CHUẨN SOTA CƠ SỞ (BASELINE COMPARISON)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s1, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "04 / 18",
                       "Bảng So sánh Đối chuẩn SOTA Cơ sở trên Tập Chuẩn SHWD (VOC2028)")

    # Table Shape
    t_rows, t_cols = 8, 8
    t_shape = s1.shapes.add_table(t_rows, t_cols, Inches(0.8), Inches(1.35), Inches(11.733), Inches(3.2))
    table = t_shape.table

    # Column widths
    col_widths = [Inches(3.133), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3)]
    for j, w in enumerate(col_widths):
        table.columns[j].width = w

    headers = [
        "Model Architecture", "Params (M)", "FLOPs (G)", "mAP50 (%)", "mAP50-95 (%)", "AP50 hat (%)", "Recall hat (%)", "F1 hat"
    ]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p = cell.text_frame.paragraphs[0]
        p.font.name = 'Calibri'
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = c_white
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    table_data = [
        ("YOLOv8n", "3.15", "8.7", "93.21", "60.26", "92.42", "87.16", "0.8879"),
        ("YOLOv8s", "11.24", "28.6", "94.89", "62.21", "94.28", "90.62", "0.9120"),
        ("YOLOv10n", "2.30", "6.7", "93.30", "60.35", "92.97", "87.13", "0.8930"),
        ("YOLOv10s", "8.00", "21.6", "94.39", "62.19", "93.36", "89.14", "0.9049"),
        ("YOLO11n", "2.60", "6.5", "93.19", "60.21", "92.33", "86.51", "0.8964"),
        ("YOLO11s (Chính)", "9.40", "21.5", "94.74", "62.54", "94.06", "90.35", "0.9073"),
        ("Rep-YOLO11s (Mục tiêu của Đề tài)", "9.85", "22.4", "94.83 (5-Fold: 96.64)", "62.54", "94.44", "91.33", "0.9190"),
    ]

    for i, row in enumerate(table_data):
        is_champion = (i == len(table_data) - 1)
        is_alt = (i % 2 == 1) and not is_champion

        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            if is_champion:
                cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF3, 0xC7)  # Light Amber/Orange highlight
            elif is_alt:
                cell.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)
            else:
                cell.fill.fore_color.rgb = c_white

            p = cell.text_frame.paragraphs[0]
            p.font.name = 'Calibri'
            p.font.size = Pt(9.2) if not is_champion else Pt(9.5)
            p.font.bold = is_champion or (j == 0)
            if is_champion:
                p.font.color.rgb = RGBColor(0x9A, 0x34, 0x12) if j in [0, 3, 6, 7] else c_title
            else:
                p.font.color.rgb = c_title if j == 0 else c_body
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    # 3 Summary Insight Cards at Bottom
    c_w = Inches(3.724)
    c_gap = Inches(0.28)
    c_top = Inches(4.75)
    c_h = Inches(1.68)

    insights = [
        ("01", c_blue, "Ưu thế Recall Sinh mạng (91.33%)",
         "Recallhat đạt 91.33%, vượt trội cả YOLO11s (90.35%) và YOLOv8s (90.62%). Trong an toàn lao động, bỏ sót vi phạm là nguy cơ tai nạn tử vong; mô hình đề xuất tối đa hóa khả năng cứu sinh."),
        ("02", c_orange, "Cân bằng Tham số & Độ phức tạp",
         "9.85M params & 22.4 GFLOPs xấp xỉ YOLO11s gốc (9.40M/21.5G), nhẹ hơn đáng kể so với YOLOv8s (11.24M/28.6G). Tích hợp 4 module cải tiến mà không gây bùng nổ tài nguyên."),
        ("03", c_green, "Độ ổn định Thống kê 5-Fold (96.64%)",
         "Trên quy trình 5-Fold Stratified CV, mô hình đạt đỉnh 96.64% mAP50 (±0.32%). Chứng minh độ tin cậy vượt trội, loại trừ hoàn toàn yếu tố may rủi của phép chia tập dữ liệu ngẫu nhiên.")
    ]

    for idx, (b_num, b_col, ins_title, ins_desc) in enumerate(insights):
        c_left = Inches(0.8) + idx * (c_w + c_gap)
        sc = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, c_top, c_w, c_h)
        sc.fill.solid(); sc.fill.fore_color.rgb = c_white; sc.line.color.rgb = c_border; sc.line.width = Pt(1)

        sb = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.12), c_top + Inches(0.12), Inches(0.38), Inches(0.28))
        sb.fill.solid(); sb.fill.fore_color.rgb = b_col; sb.line.fill.background()
        sb.text_frame.text = b_num; sb.text_frame.paragraphs[0].font.name = 'Calibri'; sb.text_frame.paragraphs[0].font.size = Pt(10); sb.text_frame.paragraphs[0].font.bold = True; sb.text_frame.paragraphs[0].font.color.rgb = c_white

        tf_ins = s1.shapes.add_textbox(c_left + Inches(0.58), c_top + Inches(0.10), c_w - Inches(0.68), c_h - Inches(0.15)).text_frame
        tf_ins.word_wrap = True; tf_ins.margin_left = tf_ins.margin_right = 0
        p = tf_ins.paragraphs[0]; p.text = ins_title; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
        p = tf_ins.add_paragraph(); p.text = ins_desc; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(3)

    add_bottom_banner(s1, "Bảng đối chuẩn khách quan trên 7,581 ảnh SHWD chứng minh Rep-YOLO11s vượt trội về Recallhat (91.33%) và F1hat (0.9190) với chi phí phần cứng tối ưu.")

    # =========================================================================
    # SLIDE 2: MỤC 4.2 · PHÂN TÍCH BASELINE 1: YOLO11s (SOTA MỚI NHẤT)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s2, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05 / 18",
                       "Phân tích Chi tiết Baseline 1: YOLO11s (Ultralytics, 10/2024 · SOTA Mới nhất)")

    # Left Card: Architecture & Specs
    c_y11_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y11_l.fill.solid(); c_y11_l.fill.fore_color.rgb = c_white; c_y11_l.line.color.rgb = c_blue; c_y11_l.line.width = Pt(1.5)
    b_y11_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y11_l.fill.solid(); b_y11_l.fill.fore_color.rgb = c_blue; b_y11_l.line.fill.background()
    b_y11_l.text_frame.text = "01"; b_y11_l.text_frame.paragraphs[0].font.name = 'Calibri'; b_y11_l.text_frame.paragraphs[0].font.size = Pt(11); b_y11_l.text_frame.paragraphs[0].font.bold = True; b_y11_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y11_l = s2.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame
    tf_y11_l.word_wrap = True
    p = tf_y11_l.paragraphs[0]; p.text = "Kiến trúc & Năng lực Thực nghiệm trên SHWD"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y11_left_points = [
        ("Paper & Thời điểm Công bố:", "Phát hành bởi Ultralytics (Tháng 10/2024), đại diện cho thế hệ mô hình phát hiện vật thể SOTA mới nhất."),
        ("Cấu trúc Mạng Đặc trưng:", "Backbone CSPDarknet cải tiến với các khối C3k2 và khối tự chú ý không gian C2PSA; Neck PANet trích xuất đa quy mô."),
        ("3 Tầng Đầu ra Phát hiện:", "P3 (stride 8: 80x80), P4 (stride 16: 40x40), P5 (stride 32: 20x20) kết hợp Decoupled Head không neo (Anchor-free)."),
        ("Tập Dữ liệu Huấn luyện:", "Pretrained trên COCO và fine-tune trực tiếp trên SHWD (7,581 ảnh authentic) với phân chia 80/20 chuẩn."),
        ("Kết quả Đối chuẩn Thực nghiệm:", "mAP50 = 94.74% · mAP50-95 = 62.54% · Recallhat = 90.35% · F1hat = 0.9073 (Params: 9.40M · FLOPs: 21.5G)."),
        ("Ưu điểm Nổi bật:", "Tối ưu hóa số lượng tham số tốt, tốc độ hội tụ nhanh, thông lượng xử lý cao trên các nền tảng GPU hiện đại.")
    ]
    for tag, desc in y11_left_points:
        p = tf_y11_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    # Right Card: Bottlenecks & Improvements
    c_y11_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y11_r.fill.solid(); c_y11_r.fill.fore_color.rgb = c_white; c_y11_r.line.color.rgb = c_orange; c_y11_r.line.width = Pt(1.5)
    b_y11_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y11_r.fill.solid(); b_y11_r.fill.fore_color.rgb = c_orange; b_y11_r.line.fill.background()
    b_y11_r.text_frame.text = "02"; b_y11_r.text_frame.paragraphs[0].font.name = 'Calibri'; b_y11_r.text_frame.paragraphs[0].font.size = Pt(11); b_y11_r.text_frame.paragraphs[0].font.bold = True; b_y11_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y11_r = s2.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame
    tf_y11_r.word_wrap = True
    p = tf_y11_r.paragraphs[0]; p.text = "3 Điểm nghẽn Cốt lõi & Giải pháp Kế thừa / Cải tiến"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y11_right_points = [
        ("Điểm nghẽn 1 (Mất dấu mũ ở xa):", "Các tầng downsampling tích lũy (stride 8/16/32) làm tiêu biến hoàn toàn tín hiệu của mũ bảo hộ siêu nhỏ (<20px)."),
        ("Điểm nghẽn 2 (Báo động giả sàn >28%):", "Tích chập tiêu chuẩn có tính Bất biến Tịnh tiến (Translation Invariance), gây nhầm lẫn xô vữa, biển cảnh báo màu vàng dưới nền đất thành mũ."),
        ("Điểm nghẽn 3 (Triệt tiêu Gradient CIoU):", "Hàm loss CIoU bị triệt tiêu gradient khi tỷ lệ aspect ratio w/h của bounding box dự đoán trùng với ground truth."),
        ("Nhóm học được gì từ YOLO11s?", "Học được cấu trúc khối trích xuất đặc trưng C3k2 gọn nhẹ và nguyên lý phân tách hai nhánh phân loại / định vị (Decoupled Head)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa bộ khung YOLO11s nhưng thay thế bằng RepConv (3 nhánh huấn luyện -> 1 nhân khi suy luận), cấy CoordConv Stem, bổ sung BiFormer Neck và thay bằng Focal-EIoU Loss.")
    ]
    for tag, desc in y11_right_points:
        p = tf_y11_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    add_bottom_banner(s2, "YOLO11s đại diện cho SOTA kiến trúc mới nhất (10/2024); nhóm kế thừa sườn mạng nhưng giải quyết triệt để 3 điểm nghẽn bằng 4 module toán học.")

    # =========================================================================
    # SLIDE 3: MỤC 4.2 · PHÂN TÍCH BASELINE 2: YOLOv8s (CHUẨN MỰC CÔNG NGHIỆP)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s3, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "06 / 18",
                       "Phân tích Chi tiết Baseline 2: YOLOv8s (Jocher et al., 2023 · Chuẩn mực Công nghiệp)")

    # Left Card: Architecture & Specs
    c_y8_l = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y8_l.fill.solid(); c_y8_l.fill.fore_color.rgb = c_white; c_y8_l.line.color.rgb = c_amber; c_y8_l.line.width = Pt(1.5)
    b_y8_l = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y8_l.fill.solid(); b_y8_l.fill.fore_color.rgb = c_amber; b_y8_l.line.fill.background()
    b_y8_l.text_frame.text = "01"; b_y8_l.text_frame.paragraphs[0].font.name = 'Calibri'; b_y8_l.text_frame.paragraphs[0].font.size = Pt(11); b_y8_l.text_frame.paragraphs[0].font.bold = True; b_y8_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y8_l = s3.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame
    tf_y8_l.word_wrap = True
    p = tf_y8_l.paragraphs[0]; p.text = "Kiến trúc & Vị thế Mốc chuẩn Công nghiệp (Gold Standard)"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y8_left_points = [
        ("Paper & Vị thế Công nghiệp:", "Được phát triển bởi Jocher et al. (2023), YOLOv8 là mốc chuẩn công nghiệp phổ biến và được ứng dụng rộng rãi nhất thế giới hiện nay."),
        ("Cấu trúc Khối Trích xuất C2f:", "Cross Stage Partial with 2 Convolutions (C2f) kết hợp nhiều kết nối residual nội bộ giúp làm giàu dòng chảy gradient."),
        ("Cơ chế Gán nhãn Động (TAL):", "Task-Aligned Assigner đo lường sự kết hợp giữa độ tự tin phân loại và căn chỉnh IoU để gán nhãn động cho top-k anchors tốt nhất."),
        ("Tập Dữ liệu Huấn luyện:", "Thực nghiệm trực tiếp trên tập SHWD chuẩn (7,581 ảnh) trong cùng điều kiện phần cứng với các baseline khác."),
        ("Kết quả Đối chuẩn Thực nghiệm:", "mAP50 = 94.89% · mAP50-95 = 62.21% · Recallhat = 90.62% · F1hat = 0.9120 (Params: 11.24M · FLOPs: 28.6G)."),
        ("Ưu điểm Nổi bật:", "Độ chính xác nhận diện tổng thể mAP50 cao, cộng đồng mã nguồn mở khổng lồ, mức độ ổn định sản xuất đã được kiểm chứng.")
    ]
    for tag, desc in y8_left_points:
        p = tf_y8_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    # Right Card: Limitations & Lessons
    c_y8_r = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y8_r.fill.solid(); c_y8_r.fill.fore_color.rgb = c_white; c_y8_r.line.color.rgb = c_purple; c_y8_r.line.width = Pt(1.5)
    b_y8_r = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y8_r.fill.solid(); b_y8_r.fill.fore_color.rgb = c_purple; b_y8_r.line.fill.background()
    b_y8_r.text_frame.text = "02"; b_y8_r.text_frame.paragraphs[0].font.name = 'Calibri'; b_y8_r.text_frame.paragraphs[0].font.size = Pt(11); b_y8_r.text_frame.paragraphs[0].font.bold = True; b_y8_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y8_r = s3.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame
    tf_y8_r.word_wrap = True
    p = tf_y8_r.paragraphs[0]; p.text = "Hạn chế Phần cứng & Bài học Kế thừa / Cải tiến"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y8_right_points = [
        ("Hạn chế 1 (Chi phí Tính toán Nặng):", "FLOPs lên tới 28.6G và 11.24M tham số, nặng hơn 28% so với Rep-YOLO11s (22.4G / 9.85M), tạo áp lực lớn khi triển khai biên."),
        ("Hạn chế 2 (Nghẽn Băng thông Bộ nhớ DRAM):", "Cấu trúc C2f phân nhánh nội bộ liên tục sử dụng nhiều thao tác nối tensor (concat) làm tăng lưu lượng truy cập bộ nhớ DRAM và độ trễ trễ truy xuất."),
        ("Hạn chế 3 (Thiếu Cơ chế Không gian & Báo động sàn):", "Không có thông tin tọa độ dọc, mô hình vẫn mắc phải tỷ lệ báo động giả cao đối với các thiết bị màu vàng dưới sàn nhà."),
        ("Nhóm học được gì từ YOLOv8s?", "Học tập cơ chế gán nhãn động Task-Aligned Assigner và chiến lược cân bằng loss đa nhiệm (BCE Classification Loss + DFL Loss)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa tư tưởng Task-Aligned Assigner nhưng xây dựng trên nền tảng YOLO11s gọn nhẹ hơn, áp dụng RepConv loại bỏ hoàn toàn chi phí bộ nhớ phân nhánh khi triển khai.")
    ]
    for tag, desc in y8_right_points:
        p = tf_y8_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    add_bottom_banner(s3, "YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất; nhóm kế thừa cơ chế TAL gán nhãn động nhưng tối ưu triệt để chi phí tính toán và bộ nhớ DRAM.")

    # =========================================================================
    # SLIDE 4 (TÙY CHỌN GỌN): SO SÁNH ĐỐI ĐẦU 2 BASELINE TRÊN 1 SLIDE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s4, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05-06 / 18",
                       "Phân tích So sánh 2 Mô hình Baseline Chính: YOLO11s vs YOLOv8s")

    # Column Left: Baseline 1 (YOLO11s)
    c_cmp1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(3.65))
    c_cmp1.fill.solid(); c_cmp1.fill.fore_color.rgb = c_white; c_cmp1.line.color.rgb = c_blue; c_cmp1.line.width = Pt(1.5)
    b_cmp1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.48), Inches(2.2), Inches(0.32))
    b_cmp1.fill.solid(); b_cmp1.fill.fore_color.rgb = c_blue; b_cmp1.line.fill.background()
    b_cmp1.text_frame.text = "BASELINE 1 · YOLO11s"
    b_cmp1.text_frame.paragraphs[0].font.name = 'Calibri'; b_cmp1.text_frame.paragraphs[0].font.size = Pt(9.5); b_cmp1.text_frame.paragraphs[0].font.bold = True; b_cmp1.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_cmp1 = s4.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(3.05)).text_frame
    tf_cmp1.word_wrap = True
    p = tf_cmp1.paragraphs[0]; p.text = "SOTA Mới Nhất (Ultralytics, Tháng 10/2024)"; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    c1_pts = [
        "Kiến trúc: C3k2 Feature Blocks + C2PSA Attention + Decoupled Head.",
        "Thông số: 9.40M Params · 21.5 GFLOPs · mAP50 = 94.74% · Recall = 90.35%.",
        "Ưu điểm: Tối ưu tham số xuất sắc, tính toán nhanh, hội tụ ổn định.",
        "Hạn chế cốt lõi: Mất tín hiệu mũ nhỏ ở xa (<20px); Báo động giả sàn nhà do bất biến tịnh tiến; Triệt tiêu gradient tỷ lệ khung CIoU.",
        "Đóng góp của Nhóm: Kế thừa sườn C3k2; thay thế RepConv, cấy CoordConv Stem, bổ sung BiFormer Neck và Focal-EIoU Head."
    ]
    for pt in c1_pts:
        p = tf_cmp1.add_paragraph(); p.text = "• " + pt; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2.5)

    # Column Right: Baseline 2 (YOLOv8s)
    c_cmp2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(3.65))
    c_cmp2.fill.solid(); c_cmp2.fill.fore_color.rgb = c_white; c_cmp2.line.color.rgb = c_amber; c_cmp2.line.width = Pt(1.5)
    b_cmp2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.48), Inches(2.2), Inches(0.32))
    b_cmp2.fill.solid(); b_cmp2.fill.fore_color.rgb = c_amber; b_cmp2.line.fill.background()
    b_cmp2.text_frame.text = "BASELINE 2 · YOLOv8s"
    b_cmp2.text_frame.paragraphs[0].font.name = 'Calibri'; b_cmp2.text_frame.paragraphs[0].font.size = Pt(9.5); b_cmp2.text_frame.paragraphs[0].font.bold = True; b_cmp2.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_cmp2 = s4.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(3.05)).text_frame
    tf_cmp2.word_wrap = True
    p = tf_cmp2.paragraphs[0]; p.text = "Chuẩn Mực Công Nghiệp (Jocher et al., 2023)"; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    c2_pts = [
        "Kiến trúc: C2f Cross Stage Partial Blocks + Task-Aligned Assigner (TAL).",
        "Thông số: 11.24M Params · 28.6 GFLOPs · mAP50 = 94.89% · Recall = 90.62%.",
        "Ưu điểm: Độ chính xác mAP50 cao, mã nguồn cực kỳ ổn định và phổ biến.",
        "Hạn chế cốt lõi: 28.6 GFLOPs nặng hơn 28% so với mô hình đề tài; Khối C2f ngốn băng thông DRAM; Báo động giả do thiếu tiên đề không gian.",
        "Đóng góp của Nhóm: Kế thừa tư tưởng gán nhãn động TAL nhưng chuyển sang nền sườn YOLO11s gọn hơn và áp dụng RepConv xóa bỏ overhead DRAM."
    ]
    for pt in c2_pts:
        p = tf_cmp2.add_paragraph(); p.text = "• " + pt; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2.5)

    # Bottom Spanning Box: Scientific Justification
    c_just = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.12), Inches(11.733), Inches(1.35))
    c_just.fill.solid(); c_just.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xFA)
    c_just.line.color.rgb = c_border; c_just.line.width = Pt(1)

    tf_just = s4.shapes.add_textbox(Inches(1.0), Inches(5.18), Inches(11.3), Inches(1.2)).text_frame
    tf_just.word_wrap = True
    p = tf_just.paragraphs[0]; p.text = "TẠI SAO NHÓM CHỌN 2 BASELINE NÀY ĐỂ BÁO CÁO REVIEW 1?"; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_orange; p.font.name = 'Calibri'
    p = tf_just.add_paragraph(); p.text = "• Tính Đại diện SOTA & Uy tín Khoa học: YOLO11s đại diện cho đỉnh cao kiến trúc mới nhất (tháng 10/2024), trong khi YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất thế giới (Gold Standard). Đối chuẩn đồng thời với cả hai tạo nền tảng vững chắc và khách quan tuyệt đối trước Hội đồng chấm."; p.font.size = Pt(9.0); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2)
    p = tf_just.add_paragraph(); p.text = "• Nền tảng Minh chứng Cải tiến: Mọi cải tiến của Rep-YOLO11s đều xuất phát từ việc khắc phục chính xác các điểm nghẽn kỹ thuật được chỉ ra ở hai mô hình cơ sở này, chứng minh tính cấp thiết và giá trị khoa học thực sự của đề tài."; p.font.size = Pt(9.0); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(1)

    add_bottom_banner(s4, "Đối chuẩn song song SOTA mới nhất (YOLO11s) và Chuẩn công nghiệp (YOLOv8s) tạo cơ sở khoa học khách quan chứng minh tính vượt trội của đề tài.")

    # =========================================================================
    # SLIDE 5: MỤC 6.1 · KIẾN TRÚC TỔNG THỂ ĐỀ XUẤT: REP-YOLO11s
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    setup_slide_header(s5, "MỤC 6 · PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED METHOD)", "07 / 18",
                       "Kiến trúc Tổng thể Rep-YOLO11s: Tích hợp 4 Cải tiến Toán học")

    # Left Container: Flowchart Architecture Pipeline
    c_flow = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(6.8), Inches(5.05))
    c_flow.fill.solid(); c_flow.fill.fore_color.rgb = c_white; c_flow.line.color.rgb = c_border; c_flow.line.width = Pt(1)

    # Header for Flowchart Container
    f_tag = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.48), Inches(3.2), Inches(0.28))
    f_tag.fill.solid(); f_tag.fill.fore_color.rgb = c_slate; f_tag.line.fill.background()
    f_tag.text_frame.text = "PIPELINE KIẾN TRÚC REP-YOLO11s"
    f_tag.text_frame.paragraphs[0].font.name = 'Calibri'; f_tag.text_frame.paragraphs[0].font.size = Pt(9); f_tag.text_frame.paragraphs[0].font.bold = True; f_tag.text_frame.paragraphs[0].font.color.rgb = c_white

    flow_steps = [
        ("ẢNH ĐẦU VÀO [640×640×3]", "Luồng Video Camera Giám sát Công trường Xây dựng", c_slate, False),
        ("CẢI TIẾN 2: COORDCONV STEM", "Cấy 2 kênh tọa độ [Cx, Cy] vào Layer 0 (5 -> 64 kênh) -> Khử báo động sàn", c_orange, True),
        ("CẢI TIẾN 1: REPCONV BACKBONE", "3 nhánh huấn luyện -> Gộp đại số về 1 Conv 3x3 khi triển khai (Zero Latency)", c_blue, True),
        ("CẢI TIẾN 3: BIFORMER NECK", "Định tuyến thưa 2 cấp độ Top-k (Complexity O(HW)) -> Bắt trúng mũ nhỏ xa", c_purple, True),
        ("CẢI TIẾN 4: FOCAL-EIoU HEAD", "Phân rã kích thước độc lập + TAL gán nhãn động -> Tránh suy biến gradient", c_amber, True),
        ("KẾT QUẢ ĐẦU RA (OUTPUT)", "Khung bao mũ bảo hộ sắc nét, Recall >91%, triệt tiêu báo động giả mặt sàn", c_green, False)
    ]

    step_top_start = Inches(1.85)
    step_h = Inches(0.60)
    step_gap = Inches(0.16)

    for idx, (title, sub, col, is_mod) in enumerate(flow_steps):
        s_y = step_top_start + idx * (step_h + step_gap)

        # Step card
        sc = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), s_y, Inches(6.4), step_h)
        sc.fill.solid()
        sc.fill.fore_color.rgb = c_orange_light if is_mod else RGBColor(0xF8, 0xFA, 0xFC)
        sc.line.color.rgb = col if is_mod else c_border
        sc.line.width = Pt(1.5) if is_mod else Pt(1)

        # Left color bar
        bar = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.05), s_y + Inches(0.06), Inches(0.12), step_h - Inches(0.12))
        bar.fill.solid(); bar.fill.fore_color.rgb = col; bar.line.fill.background()

        tf_s = s5.shapes.add_textbox(Inches(1.25), s_y + Inches(0.04), Inches(6.1), step_h - Inches(0.08)).text_frame
        tf_s.word_wrap = True; tf_s.margin_left = tf_s.margin_right = 0
        p = tf_s.paragraphs[0]; p.text = title; p.font.bold = True; p.font.size = Pt(9.5); p.font.color.rgb = col; p.font.name = 'Calibri'
        p = tf_s.add_paragraph(); p.text = sub; p.font.size = Pt(8.3); p.font.color.rgb = c_title if is_mod else c_muted; p.font.name = 'Calibri'; p.space_before = Pt(1)

    # Right Column: 4 Scientific Impact Cards
    c_right_w = Inches(4.733)
    c_right_left = Inches(7.8)
    mod_cards = [
        ("Cải tiến 1: Tái tham số hóa RepConv", c_blue,
         "Huấn luyện đa nhánh giàu biểu diễn (3x3, 1x1, Identity) -> Gộp đại số khép kín về duy nhất 1 nhân Conv 3x3 khi suy luận (`switch_to_deploy`). Đạt tốc độ suy luận cực hạn với Zero Latency Overhead."),
        ("Cải tiến 2: Mã hóa Tọa độ CoordConv", c_orange,
         "Cấy trực tiếp 2 kênh tọa độ không gian Cartesian chuẩn hóa [Cx, Cy] vào tầng Stem đầu vào. Phá vỡ tính bất biến tịnh tiến sai lầm, triệt tiêu dứt điểm báo động giả xô vàng và áo phản quang dưới sàn."),
        ("Cải tiến 3: Chú ý Định tuyến BiFormer", c_purple,
         "Cơ chế định tuyến thưa 2 cấp độ lọc bỏ 93.75% các vùng nền rậm rạp không liên quan, tập trung năng lực tính toán vào các vùng chỏm đầu công nhân ở xa với độ phức tạp tuyến tính O(HW)."),
        ("Cải tiến 4: Hàm mất mát Focal-EIoU", c_amber,
         "Phân rã độc lập sai số chiều rộng và chiều cao thay vì tỷ lệ khung bao aspect ratio của CIoU, ngăn ngừa triệt tiêu gradient. Kết hợp trọng số tiêu điểm tập trung tối đa cho các mẫu mục tiêu nhỏ khó.")
    ]

    card_h = Inches(1.18)
    card_gap = Inches(0.11)
    card_top_start = Inches(1.35)

    for idx, (m_title, m_col, m_desc) in enumerate(mod_cards):
        m_y = card_top_start + idx * (card_h + card_gap)
        mc = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right_left, m_y, c_right_w, card_h)
        mc.fill.solid(); mc.fill.fore_color.rgb = c_white; mc.line.color.rgb = c_border; mc.line.width = Pt(1)

        # Badge pill
        bp = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_right_left + Inches(0.12), m_y + Inches(0.10), Inches(0.1), card_h - Inches(0.20))
        bp.fill.solid(); bp.fill.fore_color.rgb = m_col; bp.line.fill.background()

        tf_m = s5.shapes.add_textbox(c_right_left + Inches(0.28), m_y + Inches(0.08), c_right_w - Inches(0.38), card_h - Inches(0.16)).text_frame
        tf_m.word_wrap = True; tf_m.margin_left = tf_m.margin_right = 0
        p = tf_m.paragraphs[0]; p.text = m_title; p.font.bold = True; p.font.size = Pt(9.8); p.font.color.rgb = m_col; p.font.name = 'Calibri'
        p = tf_m.add_paragraph(); p.text = m_desc; p.font.size = Pt(8.3); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2)

    add_bottom_banner(s5, "Rep-YOLO11s kết hợp 4 cải tiến toán học giải quyết đồng thời 4 rào cản: Khử báo động sàn, suy luận không độ trễ, tập trung mục tiêu xa và hồi quy hộp chính xác.")

    # Universal Calibri font enforcement
    for slide in [s1, s2, s3, s4, s5]:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    p.font.name = 'Calibri'

    prs.save(output_pptx_path)
    print(f"Successfully generated PowerPoint deck: {output_pptx_path}")

if __name__ == '__main__':
    out_dir = os.path.join(os.getcwd(), 'review_1_main')
    pptx_path = os.path.join(out_dir, 'Slides_Muc_4_5_6.pptx')
    create_pptx_deck(pptx_path)
