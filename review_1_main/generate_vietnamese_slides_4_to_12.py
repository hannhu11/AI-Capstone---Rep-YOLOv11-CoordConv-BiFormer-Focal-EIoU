"""
Script to generate the complete Vietnamese parallel slide deck for
Sections 4 through 12 of the Capstone Review 1 presentation.

Deck contains 11 slides:
- Slide 1 (Muc 4.1): Bang So Sanh Doi Chuan SOTA Co So (SHWD VOC2028)
- Slide 2 (Muc 4.2): Phan Tich Chi Tiet Baseline 1 - YOLO11s
- Slide 3 (Muc 4.2): Phan Tich Chi Tiet Baseline 2 - YOLOv8s
- Slide 4 (Muc 4.2): So Sanh Doi Dau 2 Baseline & Co So Khoa Hoc
- Slide 5 (Muc 6.1): Kien Truc Tong The Rep-YOLO11s & 4 Cai Tien Toan Hoc
- Slide 6 (Muc 7.0): Ky Thuat Du Lieu & Quy Trinh Data Engineering (SHWD & 5 Mien)
- Slide 7 (Muc 8.0): Chien Luoc Danh Gia Da Chi So & Uu Tien Recall Sinh Mang
- Slide 8 (Muc 9.0): Ke Hoach Thuc Hien Khoa Luan & Bieu Do Gantt W1-W15
- Slide 9 (Muc 10.0): Ke Hoach Xuat Ban Bai Bao Nghien Cuu Khoa Hoc Chuan Quoc Te
- Slide 10 (Muc 11.0): Ma Tran Quan Tri Rui Ro & Bien Phap Khac Phuc Chu Dong
- Slide 11 (Muc 12.0): Ket Qua Ky Vong Ban Giao & Tong Ket Gia Tri Khoa Luan

Output: review_1_main/Slides_Tieng_Viet_Muc_4_den_12.pptx
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_vietnamese_deck(output_pptx_path):
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
    c_blue = RGBColor(0x25, 0x63, 0xEB)
    c_amber = RGBColor(0xD9, 0x77, 0x06)
    c_green = RGBColor(0x16, 0xA3, 0x4A)
    c_purple = RGBColor(0x7C, 0x3A, 0xED)
    c_slate = RGBColor(0x47, 0x55, 0x69)
    c_red = RGBColor(0xDC, 0x26, 0x26)
    c_title = RGBColor(0x0F, 0x17, 0x2A)
    c_body = RGBColor(0x33, 0x41, 0x55)
    c_muted = RGBColor(0x64, 0x74, 0x8B)

    def setup_header(slide, cat_text, slide_num, title_text):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid(); bg.fill.fore_color.rgb = c_bg; bg.line.fill.background()

        # Category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(9.5), Inches(0.28))
        tf_cat = cat_box.text_frame; tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p = tf_cat.paragraphs[0]; p.text = cat_text.upper(); p.font.name = 'Calibri'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = c_orange

        # Slide number
        num_box = slide.shapes.add_textbox(Inches(11.0), Inches(0.35), Inches(1.5), Inches(0.28))
        tf_num = num_box.text_frame; tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p = tf_num.paragraphs[0]; p.text = slide_num; p.alignment = PP_ALIGN.RIGHT; p.font.name = 'Calibri'; p.font.size = Pt(11); p.font.color.rgb = c_muted

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.55))
        tf_t = title_box.text_frame; tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p = tf_t.paragraphs[0]; p.text = title_text; p.font.name = 'Calibri'; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = c_title

    def add_banner(slide, text):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.6), Inches(11.733), Inches(0.52))
        box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC); box.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1); box.line.width = Pt(1)

        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.95), Inches(6.68), Inches(1.2), Inches(0.35))
        pill.fill.solid(); pill.fill.fore_color.rgb = c_orange; pill.line.fill.background()
        p = pill.text_frame.paragraphs[0]; p.text = "ĐÚC KẾT"; p.alignment = PP_ALIGN.CENTER; p.font.name = 'Calibri'; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = c_white

        t_box = slide.shapes.add_textbox(Inches(2.25), Inches(6.62), Inches(10.1), Inches(0.48))
        tf = t_box.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = text; p.font.name = 'Calibri'; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = c_title

    # =========================================================================
    # SLIDE 1: MỤC 4.1 · BẢNG SO SÁNH ĐỐI CHUẨN SOTA CƠ SỞ
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    setup_header(s1, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "04 / 18",
                 "Bảng So sánh Đối chuẩn SOTA Cơ sở trên Tập Chuẩn SHWD (VOC2028)")

    t_shape = s1.shapes.add_table(8, 8, Inches(0.8), Inches(1.35), Inches(11.733), Inches(3.2))
    table = t_shape.table
    col_widths = [Inches(3.133), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3)]
    for j, w in enumerate(col_widths): table.columns[j].width = w

    headers = ["Kiến trúc Mô hình", "Params (M)", "FLOPs (G)", "mAP50 (%)", "mAP50-95 (%)", "AP50 hat (%)", "Recall hat (%)", "F1 hat"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j); cell.text = h; cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p = cell.text_frame.paragraphs[0]; p.font.name = 'Calibri'; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = c_white
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    table_data = [
        ("YOLOv8n", "3.15", "8.7", "93.21", "60.26", "92.42", "87.16", "0.8879"),
        ("YOLOv8s", "11.24", "28.6", "94.89", "62.21", "94.28", "90.62", "0.9120"),
        ("YOLOv10n", "2.30", "6.7", "93.30", "60.35", "92.97", "87.13", "0.8930"),
        ("YOLOv10s", "8.00", "21.6", "94.39", "62.19", "93.36", "89.14", "0.9049"),
        ("YOLO11n", "2.60", "6.5", "93.19", "60.21", "92.33", "86.51", "0.8964"),
        ("YOLO11s (Baseline Chính)", "9.40", "21.5", "94.74", "62.54", "94.06", "90.35", "0.9073"),
        ("Rep-YOLO11s (Mục tiêu Đề tài)", "9.85", "22.4", "94.83 (5-Fold: 96.64)", "62.54", "94.44", "91.33", "0.9190"),
    ]
    for i, row in enumerate(table_data):
        is_champ = (i == len(table_data) - 1)
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j); cell.text = val; cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF3, 0xC7) if is_champ else (RGBColor(0xF1, 0xF5, 0xF9) if i%2==1 else c_white)
            p = cell.text_frame.paragraphs[0]; p.font.name = 'Calibri'; p.font.size = Pt(9.2) if not is_champ else Pt(9.5)
            p.font.bold = is_champ or (j == 0)
            p.font.color.rgb = (RGBColor(0x9A, 0x34, 0x12) if j in [0, 3, 6, 7] else c_title) if is_champ else (c_title if j == 0 else c_body)
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    insights = [
        ("01", c_blue, "Ưu thế Recall Sinh mạng (91.33%)",
         "Recallhat đạt đỉnh 91.33%, cao hơn YOLO11s (90.35%) và YOLOv8s (90.62%). Trong an toàn lao động, bỏ sót vi phạm là rủi ro tử vong; mô hình đề tài tối đa hóa khả năng cứu sinh."),
        ("02", c_orange, "Cân bằng Tham số & Độ phức tạp",
         "9.85M tham số & 22.4 GFLOPs xấp xỉ YOLO11s (9.40M/21.5G), nhẹ hơn 28% so với YOLOv8s (11.24M/28.6G). Tích hợp 4 module cải tiến mà không gây bùng nổ tài nguyên."),
        ("03", c_green, "Độ ổn định Thống kê 5-Fold (96.64%)",
         "Trên quy trình 5-Fold Stratified CV, mô hình đạt đỉnh 96.64% mAP50 (+-0.32%). Chứng minh độ tin cậy vượt trội, loại trừ hoàn toàn yếu tố may rủi khi phân chia dữ liệu.")
    ]
    for idx, (b_num, b_col, ins_title, ins_desc) in enumerate(insights):
        c_left = Inches(0.8) + idx * (Inches(3.724) + Inches(0.28))
        sc = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, Inches(4.75), Inches(3.724), Inches(1.68))
        sc.fill.solid(); sc.fill.fore_color.rgb = c_white; sc.line.color.rgb = c_border; sc.line.width = Pt(1)

        sb = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.12), Inches(4.87), Inches(0.38), Inches(0.28))
        sb.fill.solid(); sb.fill.fore_color.rgb = b_col; sb.line.fill.background()
        sb.text_frame.text = b_num; sb.text_frame.paragraphs[0].font.name = 'Calibri'; sb.text_frame.paragraphs[0].font.size = Pt(10); sb.text_frame.paragraphs[0].font.bold = True; sb.text_frame.paragraphs[0].font.color.rgb = c_white

        tf_ins = s1.shapes.add_textbox(c_left + Inches(0.58), Inches(4.85), Inches(3.04), Inches(1.53)).text_frame
        tf_ins.word_wrap = True; tf_ins.margin_left = tf_ins.margin_right = 0
        p = tf_ins.paragraphs[0]; p.text = ins_title; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
        p = tf_ins.add_paragraph(); p.text = ins_desc; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(3)

    add_banner(s1, "Bảng đối chuẩn khách quan trên 7,581 ảnh SHWD chứng minh Rep-YOLO11s vượt trội về Recallhat (91.33%) và F1hat (0.9190) với chi phí phần cứng tối ưu.")

    # =========================================================================
    # SLIDE 2: MỤC 4.2 · BASELINE 1: YOLO11s (SOTA MỚI NHẤT)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    setup_header(s2, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05 / 18",
                 "Phân tích Chi tiết Baseline 1: YOLO11s (Ultralytics, 10/2024 · SOTA Mới nhất)")

    # Left Card
    c_y11_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y11_l.fill.solid(); c_y11_l.fill.fore_color.rgb = c_white; c_y11_l.line.color.rgb = c_blue; c_y11_l.line.width = Pt(1.5)
    b_y11_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y11_l.fill.solid(); b_y11_l.fill.fore_color.rgb = c_blue; b_y11_l.line.fill.background()
    b_y11_l.text_frame.text = "01"; b_y11_l.text_frame.paragraphs[0].font.name = 'Calibri'; b_y11_l.text_frame.paragraphs[0].font.size = Pt(11); b_y11_l.text_frame.paragraphs[0].font.bold = True; b_y11_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y11_l = s2.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf_y11_l.word_wrap = True
    p = tf_y11_l.paragraphs[0]; p.text = "Kiến trúc & Năng lực Thực nghiệm trên SHWD"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y11_left_points = [
        ("Công bố & Phát hành:", "Phát hành bởi Ultralytics (Tháng 10/2024), đại diện cho thế hệ mô hình phát hiện vật thể SOTA mới nhất."),
        ("Cấu trúc Mạng Đặc trưng:", "Backbone CSPDarknet cải tiến với các khối C3k2 và khối tự chú ý không gian C2PSA; Neck PANet trích xuất đa quy mô."),
        ("3 Tầng Đầu ra Phát hiện:", "P3 (stride 8: 80x80), P4 (stride 16: 40x40), P5 (stride 32: 20x20) kết hợp Decoupled Head không neo (Anchor-free)."),
        ("Tập Dữ liệu Huấn luyện:", "Pretrained trên COCO và fine-tune trực tiếp trên SHWD (7,581 ảnh authentic) với phân chia 80/20 chuẩn."),
        ("Kết quả Đối chuẩn Thực nghiệm:", "mAP50 = 94.74% · mAP50-95 = 62.54% · Recallhat = 90.35% · F1hat = 0.9073 (Params: 9.40M · FLOPs: 21.5G)."),
        ("Ưu điểm Nổi bật:", "Tối ưu hóa số lượng tham số tốt, tốc độ hội tụ nhanh, thông lượng xử lý cao trên các nền tảng GPU hiện đại.")
    ]
    for tag, desc in y11_left_points:
        p = tf_y11_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    # Right Card
    c_y11_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y11_r.fill.solid(); c_y11_r.fill.fore_color.rgb = c_white; c_y11_r.line.color.rgb = c_orange; c_y11_r.line.width = Pt(1.5)
    b_y11_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y11_r.fill.solid(); b_y11_r.fill.fore_color.rgb = c_orange; b_y11_r.line.fill.background()
    b_y11_r.text_frame.text = "02"; b_y11_r.text_frame.paragraphs[0].font.name = 'Calibri'; b_y11_r.text_frame.paragraphs[0].font.size = Pt(11); b_y11_r.text_frame.paragraphs[0].font.bold = True; b_y11_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y11_r = s2.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf_y11_r.word_wrap = True
    p = tf_y11_r.paragraphs[0]; p.text = "3 Điểm nghẽn Cốt lõi & Bài học Kế thừa / Cải tiến"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y11_right_points = [
        ("Điểm nghẽn 1 (Mất dấu mũ ở xa):", "Các tầng downsampling tích lũy (stride 8/16/32) làm tiêu biến hoàn toàn tín hiệu của mũ bảo hộ siêu nhỏ (<20px)."),
        ("Điểm nghẽn 2 (Báo động giả sàn >28%):", "Tích chập tiêu chuẩn có tính Bất biến Tịnh tiến, gây nhầm lẫn xô vữa, biển cảnh báo màu vàng dưới sàn thành mũ."),
        ("Điểm nghẽn 3 (Triệt tiêu Gradient CIoU):", "Hàm loss CIoU bị triệt tiêu gradient khi tỷ lệ aspect ratio w/h của bounding box dự đoán trùng với ground truth."),
        ("Nhóm học được gì từ YOLO11s?", "Học được cấu trúc khối trích xuất đặc trưng C3k2 gọn nhẹ và nguyên lý phân tách hai nhánh phân loại / định vị (Decoupled Head)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa bộ khung YOLO11s nhưng thay bằng RepConv (gộp đại số về 1 Conv 3x3), cấy CoordConv Stem, bổ sung BiFormer Neck và thay bằng Focal-EIoU.")
    ]
    for tag, desc in y11_right_points:
        p = tf_y11_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    add_banner(s2, "YOLO11s đại diện cho SOTA kiến trúc mới nhất (10/2024); nhóm kế thừa sườn mạng nhưng giải quyết triệt để 3 điểm nghẽn bằng 4 module toán học.")

    # =========================================================================
    # SLIDE 3: MỤC 4.2 · BASELINE 2: YOLOv8s (CHUẨN MỰC CÔNG NGHIỆP)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    setup_header(s3, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "06 / 18",
                 "Phân tích Chi tiết Baseline 2: YOLOv8s (Jocher et al., 2023 · Chuẩn mực Công nghiệp)")

    c_y8_l = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y8_l.fill.solid(); c_y8_l.fill.fore_color.rgb = c_white; c_y8_l.line.color.rgb = c_amber; c_y8_l.line.width = Pt(1.5)
    b_y8_l = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y8_l.fill.solid(); b_y8_l.fill.fore_color.rgb = c_amber; b_y8_l.line.fill.background()
    b_y8_l.text_frame.text = "01"; b_y8_l.text_frame.paragraphs[0].font.name = 'Calibri'; b_y8_l.text_frame.paragraphs[0].font.size = Pt(11); b_y8_l.text_frame.paragraphs[0].font.bold = True; b_y8_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y8_l = s3.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf_y8_l.word_wrap = True
    p = tf_y8_l.paragraphs[0]; p.text = "Kiến trúc & Vị thế Mốc chuẩn Công nghiệp (Gold Standard)"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y8_left_points = [
        ("Vị thế Công nghiệp Toàn cầu:", "Được phát triển bởi Jocher et al. (2023), YOLOv8 là mốc chuẩn công nghiệp phổ biến và được ứng dụng rộng rãi nhất thế giới hiện nay."),
        ("Cấu trúc Khối Trích xuất C2f:", "Cross Stage Partial with 2 Convolutions (C2f) kết hợp nhiều kết nối residual nội bộ giúp làm giàu dòng chảy gradient."),
        ("Cơ chế Gán nhãn Động (TAL):", "Task-Aligned Assigner đo lường sự kết hợp giữa độ tự tin phân loại và căn chỉnh IoU để gán nhãn động cho top-k anchors tốt nhất."),
        ("Tập Dữ liệu Huấn luyện:", "Thực nghiệm trực tiếp trên tập SHWD chuẩn (7,581 ảnh) trong cùng điều kiện phần cứng với các baseline khác."),
        ("Kết quả Đối chuẩn Thực nghiệm:", "mAP50 = 94.89% · mAP50-95 = 62.21% · Recallhat = 90.62% · F1hat = 0.9120 (Params: 11.24M · FLOPs: 28.6G)."),
        ("Ưu điểm Nổi bật:", "Độ chính xác nhận diện tổng thể mAP50 cao, cộng đồng mã nguồn mở khổng lồ, mức độ ổn định sản xuất đã được kiểm chứng.")
    ]
    for tag, desc in y8_left_points:
        p = tf_y8_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    # Right Card
    c_y8_r = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c_y8_r.fill.solid(); c_y8_r.fill.fore_color.rgb = c_white; c_y8_r.line.color.rgb = c_purple; c_y8_r.line.width = Pt(1.5)
    b_y8_r = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b_y8_r.fill.solid(); b_y8_r.fill.fore_color.rgb = c_purple; b_y8_r.line.fill.background()
    b_y8_r.text_frame.text = "02"; b_y8_r.text_frame.paragraphs[0].font.name = 'Calibri'; b_y8_r.text_frame.paragraphs[0].font.size = Pt(11); b_y8_r.text_frame.paragraphs[0].font.bold = True; b_y8_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_y8_r = s3.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf_y8_r.word_wrap = True
    p = tf_y8_r.paragraphs[0]; p.text = "Hạn chế Phần cứng & Bài học Kế thừa / Cải tiến"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    y8_right_points = [
        ("Hạn chế 1 (Chi phí Tính toán Nặng):", "FLOPs lên tới 28.6G và 11.24M tham số, nặng hơn 28% so với Rep-YOLO11s (22.4G / 9.85M), tạo áp lực lớn khi triển khai biên."),
        ("Hạn chế 2 (Nghẽn Băng thông Bộ nhớ DRAM):", "Cấu trúc C2f phân nhánh nội bộ liên tục sử dụng nhiều thao tác nối tensor (concat) làm tăng lưu lượng truy cập DRAM và độ trễ truy xuất."),
        ("Hạn chế 3 (Thiếu Cơ chế Không gian):", "Không có thông tin tọa độ dọc, mô hình vẫn mắc phải tỷ lệ báo động giả cao đối với các thiết bị màu vàng dưới sàn nhà."),
        ("Nhóm học được gì từ YOLOv8s?", "Học tập cơ chế gán nhãn động Task-Aligned Assigner và chiến lược cân bằng loss đa nhiệm (BCE Classification Loss + DFL Loss)."),
        ("Nhóm kế thừa & Cải tiến gì?", "Kế thừa tư tưởng Task-Aligned Assigner nhưng xây dựng trên nền tảng YOLO11s gọn nhẹ hơn, áp dụng RepConv loại bỏ hoàn toàn chi phí bộ nhớ phân nhánh khi triển khai.")
    ]
    for tag, desc in y8_right_points:
        p = tf_y8_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.2); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    add_banner(s3, "YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất; nhóm kế thừa cơ chế TAL gán nhãn động nhưng tối ưu triệt để chi phí tính toán và bộ nhớ DRAM.")

    # =========================================================================
    # SLIDE 4: MỤC 4.2 · SO SÁNH ĐỐI ĐẦU 2 BASELINE TRÊN 1 SLIDE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    setup_header(s4, "MỤC 4 & MỤC 5 · RELATED WORK & SOTA BASELINES", "05-06 / 18",
                 "Phân tích So sánh 2 Mô hình Baseline Chính: YOLO11s vs YOLOv8s")

    # Left Column: YOLO11s
    c_cmp1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(3.65))
    c_cmp1.fill.solid(); c_cmp1.fill.fore_color.rgb = c_white; c_cmp1.line.color.rgb = c_blue; c_cmp1.line.width = Pt(1.5)
    b_cmp1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.48), Inches(2.2), Inches(0.32))
    b_cmp1.fill.solid(); b_cmp1.fill.fore_color.rgb = c_blue; b_cmp1.line.fill.background()
    b_cmp1.text_frame.text = "BASELINE 1 · YOLO11s"
    b_cmp1.text_frame.paragraphs[0].font.name = 'Calibri'; b_cmp1.text_frame.paragraphs[0].font.size = Pt(9.5); b_cmp1.text_frame.paragraphs[0].font.bold = True; b_cmp1.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_cmp1 = s4.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(3.05)).text_frame; tf_cmp1.word_wrap = True
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

    # Right Column: YOLOv8s
    c_cmp2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(3.65))
    c_cmp2.fill.solid(); c_cmp2.fill.fore_color.rgb = c_white; c_cmp2.line.color.rgb = c_amber; c_cmp2.line.width = Pt(1.5)
    b_cmp2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.48), Inches(2.2), Inches(0.32))
    b_cmp2.fill.solid(); b_cmp2.fill.fore_color.rgb = c_amber; b_cmp2.line.fill.background()
    b_cmp2.text_frame.text = "BASELINE 2 · YOLOv8s"
    b_cmp2.text_frame.paragraphs[0].font.name = 'Calibri'; b_cmp2.text_frame.paragraphs[0].font.size = Pt(9.5); b_cmp2.text_frame.paragraphs[0].font.bold = True; b_cmp2.text_frame.paragraphs[0].font.color.rgb = c_white

    tf_cmp2 = s4.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.3), Inches(3.05)).text_frame; tf_cmp2.word_wrap = True
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

    # Bottom Justification
    c_just = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.12), Inches(11.733), Inches(1.35))
    c_just.fill.solid(); c_just.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xFA); c_just.line.color.rgb = c_border; c_just.line.width = Pt(1)
    tf_just = s4.shapes.add_textbox(Inches(1.0), Inches(5.18), Inches(11.3), Inches(1.2)).text_frame; tf_just.word_wrap = True
    p = tf_just.paragraphs[0]; p.text = "TẠI SAO NHÓM CHỌN 2 BASELINE NÀY ĐỂ BÁO CÁO REVIEW 1?"; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_orange; p.font.name = 'Calibri'
    p = tf_just.add_paragraph(); p.text = "• Tính Đại diện SOTA & Uy tín Khoa học: YOLO11s đại diện cho đỉnh cao kiến trúc mới nhất (tháng 10/2024), trong khi YOLOv8s là mốc chuẩn công nghiệp phổ biến nhất thế giới (Gold Standard). Đối chuẩn đồng thời với cả hai tạo nền tảng vững chắc và khách quan tuyệt đối trước Hội đồng chấm."; p.font.size = Pt(9.0); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2)
    p = tf_just.add_paragraph(); p.text = "• Nền tảng Minh chứng Cải tiến: Mọi cải tiến của Rep-YOLO11s đều xuất phát từ việc khắc phục chính xác các điểm nghẽn kỹ thuật được chỉ ra ở hai mô hình cơ sở này, chứng minh tính cấp thiết và giá trị khoa học thực sự của đề tài."; p.font.size = Pt(9.0); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(1)

    add_banner(s4, "Đối chuẩn song song SOTA mới nhất (YOLO11s) và Chuẩn công nghiệp (YOLOv8s) tạo cơ sở khoa học khách quan chứng minh tính vượt trội của đề tài.")

    # =========================================================================
    # SLIDE 5: MỤC 6.1 · KIẾN TRÚC TỔNG THỂ ĐỀ XUẤT: REP-YOLO11s
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    setup_header(s5, "MỤC 6 · PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED METHOD)", "07 / 18",
                 "Kiến trúc Tổng thể Rep-YOLO11s: Tích hợp 4 Cải tiến Toán học")

    c_flow = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(6.8), Inches(5.05))
    c_flow.fill.solid(); c_flow.fill.fore_color.rgb = c_white; c_flow.line.color.rgb = c_border; c_flow.line.width = Pt(1)

    f_tag = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.48), Inches(3.2), Inches(0.28))
    f_tag.fill.solid(); f_tag.fill.fore_color.rgb = c_slate; f_tag.line.fill.background()
    f_tag.text_frame.text = "PIPELINE KIẾN TRÚC REP-YOLO11s"
    f_tag.text_frame.paragraphs[0].font.name = 'Calibri'; f_tag.text_frame.paragraphs[0].font.size = Pt(9); f_tag.text_frame.paragraphs[0].font.bold = True; f_tag.text_frame.paragraphs[0].font.color.rgb = c_white

    flow_steps = [
        ("ẢNH ĐẦU VÀO [640×640×3]", "Luồng Video Camera Giám sát Công trường Xây dựng (CCTV Feed)", c_slate, False),
        ("CẢI TIẾN 2: COORDCONV STEM", "Cấy 2 kênh tọa độ [Cx, Cy] vào Layer 0 (5 -> 64 kênh) -> Khử báo động sàn", c_orange, True),
        ("CẢI TIẾN 1: REPCONV BACKBONE", "3 nhánh huấn luyện -> Gộp đại số về 1 Conv 3x3 khi triển khai (Zero Latency)", c_blue, True),
        ("CẢI TIẾN 3: BIFORMER NECK", "Định tuyến thưa 2 cấp độ Top-k (Complexity O(HW)) -> Bắt trúng mũ nhỏ xa", c_purple, True),
        ("CẢI TIẾN 4: FOCAL-EIoU HEAD", "Phân rã kích thước độc lập + TAL gán nhãn động -> Tránh suy biến gradient", c_amber, True),
        ("KẾT QUẢ ĐẦU RA (OUTPUT)", "Khung bao mũ bảo hộ sắc nét, Recall >91%, triệt tiêu báo động giả mặt sàn", c_green, False)
    ]
    for idx, (title, sub, col, is_mod) in enumerate(flow_steps):
        s_y = Inches(1.85) + idx * (Inches(0.60) + Inches(0.16))
        sc = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), s_y, Inches(6.4), Inches(0.60))
        sc.fill.solid(); sc.fill.fore_color.rgb = c_orange_light if is_mod else RGBColor(0xF8, 0xFA, 0xFC)
        sc.line.color.rgb = col if is_mod else c_border; sc.line.width = Pt(1.5) if is_mod else Pt(1)

        bar = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.05), s_y + Inches(0.06), Inches(0.12), Inches(0.48))
        bar.fill.solid(); bar.fill.fore_color.rgb = col; bar.line.fill.background()

        tf_s = s5.shapes.add_textbox(Inches(1.25), s_y + Inches(0.04), Inches(6.1), Inches(0.52)).text_frame
        tf_s.word_wrap = True; tf_s.margin_left = tf_s.margin_right = 0
        p = tf_s.paragraphs[0]; p.text = title; p.font.bold = True; p.font.size = Pt(9.5); p.font.color.rgb = col; p.font.name = 'Calibri'
        p = tf_s.add_paragraph(); p.text = sub; p.font.size = Pt(8.3); p.font.color.rgb = c_title if is_mod else c_muted; p.font.name = 'Calibri'; p.space_before = Pt(1)

    # Right Cards
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
    for idx, (m_title, m_col, m_desc) in enumerate(mod_cards):
        m_y = Inches(1.35) + idx * (Inches(1.18) + Inches(0.11))
        mc = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.8), m_y, Inches(4.733), Inches(1.18))
        mc.fill.solid(); mc.fill.fore_color.rgb = c_white; mc.line.color.rgb = c_border; mc.line.width = Pt(1)

        bp = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.92), m_y + Inches(0.10), Inches(0.1), Inches(0.98))
        bp.fill.solid(); bp.fill.fore_color.rgb = m_col; bp.line.fill.background()

        tf_m = s5.shapes.add_textbox(Inches(8.08), m_y + Inches(0.08), Inches(4.35), Inches(1.02)).text_frame
        tf_m.word_wrap = True; tf_m.margin_left = tf_m.margin_right = 0
        p = tf_m.paragraphs[0]; p.text = m_title; p.font.bold = True; p.font.size = Pt(9.8); p.font.color.rgb = m_col; p.font.name = 'Calibri'
        p = tf_m.add_paragraph(); p.text = m_desc; p.font.size = Pt(8.3); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2)

    add_banner(s5, "Rep-YOLO11s kết hợp 4 cải tiến toán học giải quyết đồng thời 4 rào cản: Khử báo động sàn, suy luận không độ trễ, tập trung mục tiêu xa và hồi quy hộp chính xác.")

    # =========================================================================
    # SLIDE 6: MỤC 7 · KỸ THUẬT DỮ LIỆU & QUY TRÌNH DATA ENGINEERING
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    setup_header(s6, "MỤC 7 · TẬP DỮ LIỆU & QUY TRÌNH DATA ENGINEERING", "08 / 18",
                 "Kỹ thuật Dữ liệu: Làm sạch Dị biệt, Cân bằng Lớp & Chuẩn hóa Không gian Nhãn C*")

    # Left Card: In-Domain SHWD
    c7_l = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c7_l.fill.solid(); c7_l.fill.fore_color.rgb = c_white; c7_l.line.color.rgb = c_blue; c7_l.line.width = Pt(1.5)
    b7_l = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b7_l.fill.solid(); b7_l.fill.fore_color.rgb = c_blue; b7_l.line.fill.background()
    b7_l.text_frame.text = "01"; b7_l.text_frame.paragraphs[0].font.name = 'Calibri'; b7_l.text_frame.paragraphs[0].font.size = Pt(11); b7_l.text_frame.paragraphs[0].font.bold = True; b7_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf7_l = s6.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf7_l.word_wrap = True
    p = tf7_l.paragraphs[0]; p.text = "Tập Dữ liệu Huấn luyện Trong miền: SHWD (VOC2028)"; p.font.bold = True; p.font.size = Pt(12.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    shwd_pts = [
        ("Quy mô & Nguồn gốc:", "7,581 ảnh chụp công trường xây dựng thực tế với độ phân giải cao và góc quay camera giám sát đa dạng."),
        ("Phân chia Chuẩn 80/20:", "6,064 ảnh TrainVal và 1,517 ảnh Test độc lập. Gom cụm các khung hình liên tiếp để bảo đảm Zero Data Leakage tuyệt đối."),
        ("Lọc sạch Dị biệt (Anomaly Cleaning):", "Phát hiện và thanh lọc triệt để 3 nhãn chú thích XML rác bị gán nhầm thành 'dog' trong tập gốc SHWD."),
        ("Mất cân bằng Cực đoan 1:12:", "Gồm 9,044 nhãn mũ bảo hộ ('hat') áp đảo bởi 111,514 nhãn thân người ('person')."),
        ("Chiến lược Khắc phục Mất cân bằng:", "Áp dụng Focal BCE Loss kết hợp Task-Aligned Assigner (TAL) giúp mô hình tập trung gradient vào mũ bảo hộ mà không làm suy giảm độ chính xác thân người."),
        ("Data Augmentation Đa quy mô:", "Mosaic (p=1.0, tắt 10 epoch cuối), Mixup (p=0.15), Random Affine (+-10%), lật ngang ảnh (p=0.5) và HSV Jitter.")
    ]
    for tag, desc in shwd_pts:
        p = tf7_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.0); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(3.5)

    # Right Card: Cross-Domain & Label Harmonization
    c7_r = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c7_r.fill.solid(); c7_r.fill.fore_color.rgb = c_white; c7_r.line.color.rgb = c_orange; c7_r.line.width = Pt(1.5)
    b7_r = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b7_r.fill.solid(); b7_r.fill.fore_color.rgb = c_orange; b7_r.line.fill.background()
    b7_r.text_frame.text = "02"; b7_r.text_frame.paragraphs[0].font.name = 'Calibri'; b7_r.text_frame.paragraphs[0].font.size = Pt(11); b7_r.text_frame.paragraphs[0].font.bold = True; b7_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf7_r = s6.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf7_r.word_wrap = True
    p = tf7_r.paragraphs[0]; p.text = "5 Tập Dữ liệu Mở rộng & Chuẩn hóa Không gian Nhãn C*"; p.font.bold = True; p.font.size = Pt(12.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    cross_pts = [
        ("GDUT-HWD (13,499 ảnh):", "Môi trường công nhân cực kỳ đông đúc (15–30 người/khung hình) với mật độ che khuất lẫn nhau dày đặc."),
        ("SHEL5K (5,000 ảnh):", "Góc nhìn camera quan sát CCTV góc cao, bao quát từ trên xuống với thách thức che khuất đa quy mô và tính đa nghĩa nhãn."),
        ("Hard Hat Workers (7,000 ảnh):", "Công trường xây dựng ngoài trời với độ tương phản ánh sáng gắt và bóng đổ phức tạp."),
        ("SHD & SFCHD Benchmark:", "Nhà máy luyện kim, xưởng đóng tàu và tổ hợp hóa dầu với bụi bẩn và ánh sáng nhân tạo yếu."),
        ("Chuẩn hóa Nhãn Chung C*:", "Đồng bộ hóa các quy ước gán nhãn xung đột về một không gian nhãn thống nhất: C* = {0: 'hat' (Mũ bảo hộ), 1: 'person' (Người lao động)}."),
        ("Ý nghĩa Khoa học:", "Đảm bảo khả năng chuyển giao tổng quát hóa Zero-Shot vững chắc trên mọi hệ thống camera công trường thực tế chưa từng thấy khi huấn luyện.")
    ]
    for tag, desc in cross_pts:
        p = tf7_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.0); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(3.5)

    add_banner(s6, "Quy trình tiền xử lý nghiêm ngặt loại bỏ rò rỉ dữ liệu, khử nhãn rác, cân bằng tỉ lệ 1:12 và đồng bộ nhãn C* trên hơn 25,000 ảnh đa miền.")

    # =========================================================================
    # SLIDE 7: MỤC 8 · CHIẾN LƯỢC ĐÁNH GIÁ (EVALUATION STRATEGY)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    setup_header(s7, "MỤC 8 · CHIẾN LƯỢC ĐÁNH GIÁ (EVALUATION STRATEGY)", "09 / 18",
                 "Chiến lược Đánh giá Đa chỉ số & Nguyên tắc Ưu tiên Recall Sinh mạng")

    # 4 Cards 2x2 Layout
    card_specs = [
        (Inches(0.8), Inches(1.35), "01", c_blue, "Các Chỉ số Nhận diện & Định vị Cốt lõi", [
            "mAP50 (Thước đo Chuẩn mực): Đo lường độ chính xác phát hiện tổng thể tại ngưỡng IoU 0.50.",
            "mAP50-95 (Độ ôm khít Khung bao): Trung bình nghiêm ngặt qua 10 bước IoU [0.50:0.05:0.95], phạt nặng các khung bao xộc xệch hoặc lệch tâm.",
            "Precision & F1-Score: Đánh giá sự cân bằng giữa phát hiện chính xác và hạn chế báo động nhầm.",
            "Ràng buộc Thời gian thực: Đảm bảo độ trễ suy luận đáp ứng luồng video mượt mà >25 FPS."
        ]),
        (Inches(6.8), Inches(1.35), "02", c_orange, "Ưu tiên Sinh mạng: Recall Quan trọng hơn Precision", [
            "Đặc thù An toàn Lao động: Bỏ sót một công nhân không đội mũ là nguy cơ tử vong không thể cứu vãn.",
            "Phân tích Chi phí Sai số Không đối xứng: Cảnh báo thừa (False Alarm) chỉ mất 2 giây kiểm tra; Bỏ sót (False Negative) đánh đổi bằng tính mạng con người.",
            "Ngưỡng Tối ưu hóa: Tinh chỉnh hàm mất mát để tối đa hóa Recallhat (>91%) mà vẫn giữ vững mAP50.",
            "Mục tiêu Cứu sinh: Mô hình được thiết kế hướng tới sứ mệnh bảo vệ an toàn tối đa cho công nhân."
        ]),
        (Inches(0.8), Inches(3.95), "03", c_purple, "Kiểm định Chéo 5 Lớp (5-Fold Stratified CV)", [
            "Phân chia Đồng đều 5 Fold: Bảo đảm tỉ lệ nhãn 1:12.33 không bị lệch lạc giữa các fold.",
            "Loại trừ May rủi Dữ liệu: Đánh giá mô hình trên 5 lần phân chia độc lập để tránh hiện tượng cherry-picking.",
            "Báo cáo Trung bình & Độ lệch chuẩn: Trình bày số liệu dạng mu +- sigma khẳng định tính tin cậy khoa học.",
            "Thống kê Vững chắc: Đạt 96.64% mAP50 (+-0.32%) chứng minh sự ổn định tuyệt đối của kiến trúc."
        ]),
        (Inches(6.8), Inches(3.95), "04", c_green, "Minh bạch Mô hình & Trích xuất Nhiệt XAI (Grad-CAM)", [
            "Trích xuất Gradient Tầng Cổ (Neck): Quan sát trực quan vùng trọng tâm kích hoạt của mạng nơ-ron.",
            "Khử Nhiễu Bối cảnh: Kiểm chứng mô hình tập trung vào chỏm đầu công nhân thay vì bị đánh lừa bởi áo phản quang hay xô vữa dưới sàn.",
            "Phân tích Sai số Trực quan: Khoanh vùng chính xác các trường hợp dự đoán sai để làm sáng tỏ bản chất bài toán.",
            "Minh bạch Học thuật Tuyệt đối: Đảm bảo mô hình có thể giải thích được trước Hội đồng Đánh giá."
        ])
    ]

    for c_left, c_top, b_num, b_col, card_title, card_items in card_specs:
        sc = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, c_top, Inches(5.7), Inches(2.45))
        sc.fill.solid(); sc.fill.fore_color.rgb = c_white; sc.line.color.rgb = c_border; sc.line.width = Pt(1)

        sb = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.2), c_top + Inches(0.15), Inches(0.42), Inches(0.32))
        sb.fill.solid(); sb.fill.fore_color.rgb = b_col; sb.line.fill.background()
        sb.text_frame.text = b_num; sb.text_frame.paragraphs[0].font.name = 'Calibri'; sb.text_frame.paragraphs[0].font.size = Pt(11); sb.text_frame.paragraphs[0].font.bold = True; sb.text_frame.paragraphs[0].font.color.rgb = c_white

        tf = s7.shapes.add_textbox(c_left + Inches(0.75), c_top + Inches(0.13), Inches(4.75), Inches(2.2)).text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = card_title; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
        for itm in card_items:
            p = tf.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2.5)

    add_banner(s7, "Chiến lược đánh giá toàn diện ưu tiên bảo vệ sinh mạng công nhân, kết hợp kiểm định thống kê 5-Fold và giải thích trực quan bằng Grad-CAM.")

    # =========================================================================
    # SLIDE 8: MỤC 9 · KẾ HOẠCH THỰC HIỆN KHÓA LUẬN (PROJECT PLAN)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    setup_header(s8, "MỤC 9 · KẾ HOẠCH THỰC HIỆN KHÓA LUẬN (PROJECT PLAN)", "10 / 18",
                 "Kế hoạch Triển khai Khóa luận & Biểu đồ Mốc tiến độ Gantt (W1 - W15)")

    phase_cards = [
        ("Giai đoạn 1 (Tuần 1-5)", c_blue, "Nền tảng & Báo cáo Review 1", [
            "Khảo sát 30 nghiên cứu quốc tế, xác định RQ & Research Gap.",
            "Làm sạch tập dữ liệu SHWD, khử nhãn rác dog, đồng bộ nhãn C*.",
            "Thiết lập các mô hình mốc chuẩn SOTA: YOLOv8s, YOLO11s.",
            "Sản phẩm: Slide & Báo cáo Review 1, Tập dữ liệu chuẩn hóa."
        ]),
        ("Giai đoạn 2 (Tuần 6-8)", c_orange, "Kỹ thuật Module Cốt lõi & Review 2", [
            "Lập trình module toán học: RepConv, CoordConv, BiFormer, Focal-EIoU.",
            "Thực hiện chuỗi thực nghiệm Ablation Study A0 -> A6.",
            "Đo đạc thực nghiệm sơ bộ trên phần cứng GPU T4.",
            "Sản phẩm: Mã nguồn module hoàn chỉnh, Bảng số liệu Ablation."
        ]),
        ("Giai đoạn 3 (Tuần 9-11)", c_purple, "Kiểm chứng Thống kê & Review 3", [
            "Thực thi kiểm định chéo 5-Fold Stratified Cross-Validation.",
            "Đánh giá năng lực Zero-shot trên 5 tập dữ liệu ngoại miền.",
            "Trích xuất bản đồ kích hoạt nhiệt Grad-CAM XAI.",
            "Sản phẩm: Bảng số liệu thống kê 5-Fold, Trọng số mô hình tốt nhất."
        ]),
        ("Giai đoạn 4-5 (Tuần 12-15)", c_green, "Ứng dụng, Bài báo & Bảo vệ", [
            "Xây dựng ứng dụng giám sát video thời gian thực RTSP.",
            "Hoàn thiện bản thảo bài báo nghiên cứu khoa học chuẩn IEEE.",
            "Viết toàn văn Báo cáo Khóa luận Tốt nghiệp & Dựng slide bảo vệ.",
            "Sản phẩm: Ứng dụng demo RTSP, Bản thảo Paper, Quyển khóa luận."
        ])
    ]

    p_w = Inches(2.72)
    p_gap = Inches(0.28)
    for idx, (p_phase, p_col, p_title, p_items) in enumerate(phase_cards):
        p_left = Inches(0.8) + idx * (p_w + p_gap)
        sc = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, p_left, Inches(1.35), p_w, Inches(3.60))
        sc.fill.solid(); sc.fill.fore_color.rgb = c_white; sc.line.color.rgb = c_border; sc.line.width = Pt(1)

        hbar = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, p_left + Inches(0.12), Inches(1.48), p_w - Inches(0.24), Inches(0.32))
        hbar.fill.solid(); hbar.fill.fore_color.rgb = p_col; hbar.line.fill.background()
        hbar.text_frame.text = p_phase; hbar.text_frame.paragraphs[0].font.name = 'Calibri'; hbar.text_frame.paragraphs[0].font.size = Pt(9.5); hbar.text_frame.paragraphs[0].font.bold = True; hbar.text_frame.paragraphs[0].font.color.rgb = c_white

        tf = s8.shapes.add_textbox(p_left + Inches(0.12), Inches(1.88), p_w - Inches(0.24), Inches(2.95)).text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = p_title; p.font.bold = True; p.font.size = Pt(10.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
        for itm in p_items:
            p = tf.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(8.4); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2.5)

    # Workload Allocation Box
    c_alloc = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.08), Inches(11.733), Inches(1.38))
    c_alloc.fill.solid(); c_alloc.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xFA); c_alloc.line.color.rgb = c_border; c_alloc.line.width = Pt(1)

    tf_al = s8.shapes.add_textbox(Inches(1.0), Inches(5.14), Inches(11.3), Inches(1.25)).text_frame; tf_al.word_wrap = True
    p = tf_al.paragraphs[0]; p.text = "PHÂN CÔNG TRÁCH NHIỆM & ĐẢM BẢO CHẤT LƯỢNG KHÓA LUẬN (AGILE WORKLOAD)"; p.font.bold = True; p.font.size = Pt(10.2); p.font.color.rgb = c_orange; p.font.name = 'Calibri'
    alloc_pts = [
        ("Nguyễn Hàn Như (Trưởng nhóm / AI Lead):", "Thiết kế kiến trúc Rep-YOLO11s, lập trình module toán học, quy trình huấn luyện và chấp bút bản thảo bài báo khoa học."),
        ("Nguyễn Văn Thành (Data Engineering Lead):", "Lọc nhãn rác, làm sạch dữ liệu SHWD, chuẩn hóa không gian nhãn C*, thực thi kiểm định thống kê 5-Fold Cross-Validation."),
        ("Trần Phạm Tuấn Dũng (System & RTSP Lead):", "Xây dựng pipeline thu nhận luồng video RTSP, giải mã phần cứng NVDEC, tích hợp giao diện cảnh báo và chuẩn bị hồ sơ kỹ thuật.")
    ]
    for m_role, m_desc in alloc_pts:
        p = tf_al.add_paragraph(); p.text = f"• {m_role} {m_desc}"; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2)

    add_banner(s8, "Kế hoạch 15 tuần theo mô hình Agile lặp gắn chặt với 3 mốc Review, phân công trách nhiệm minh bạch, bảo đảm hoàn thành 100% đúng tiến độ.")

    # =========================================================================
    # SLIDE 9: MỤC 10 · KẾ HOẠCH BÀI BÁO KHOA HỌC CHUẨN QUỐC TẾ
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    setup_header(s9, "MỤC 10 · KẾ HOẠCH XUẤT BẢN BÀI BÁO KHOA HỌC", "11 / 18",
                 "Kế hoạch Xuất bản Bài báo Nghiên cứu Khoa học Chuẩn Quốc tế (IEEE Format)")

    # Left Card: Publication Plan
    c10_l = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c10_l.fill.solid(); c10_l.fill.fore_color.rgb = c_white; c10_l.line.color.rgb = c_blue; c10_l.line.width = Pt(1.5)
    b10_l = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b10_l.fill.solid(); b10_l.fill.fore_color.rgb = c_blue; b10_l.line.fill.background()
    b10_l.text_frame.text = "01"; b10_l.text_frame.paragraphs[0].font.name = 'Calibri'; b10_l.text_frame.paragraphs[0].font.size = Pt(11); b10_l.text_frame.paragraphs[0].font.bold = True; b10_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf10_l = s9.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf10_l.word_wrap = True
    p = tf10_l.paragraphs[0]; p.text = "Bản thảo Nghiên cứu Khoa học Chuẩn Sẵn sàng Nộp"; p.font.bold = True; p.font.size = Pt(12.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    paper_pts = [
        ("Định dạng Trình bày:", "Soạn thảo hoàn chỉnh theo chuẩn định dạng IEEE hai cột (Two-Column Conference/Journal Template) với đầy đủ phương trình toán học và biểu đồ đối chuẩn."),
        ("Cấu trúc Học thuật Toàn diện:", "Bao gồm Abstract, 4 Câu hỏi Nghiên cứu, Phân tích 30 Công trình SOTA, Kiến trúc Đề xuất, Kết quả Đối chuẩn Đa chiều và Phân tích XAI."),
        ("Chiến lược Xuất bản An toàn:", "Chuẩn bị bản thảo nghiên cứu hoàn chỉnh, chất lượng cao, sẵn sàng trình Hội đồng Đánh giá Khóa luận và nộp tới các hội nghị / tạp chí quốc tế uy tín."),
        ("Cam kết Minh bạch Học thuật:", "Mọi đồ thị Pareto, bảng số liệu thực nghiệm và sơ đồ kiến trúc đều được trích xuất 100% từ mã nguồn PyTorch thực thi, không suy diễn."),
        ("Chia sẻ Mã nguồn Mở:", "Toàn bộ checkpoints mô hình, kịch bản huấn luyện và dữ liệu làm sạch được công khai trên GitHub phục vụ mục đích tái lập nghiên cứu (Reproducibility).")
    ]
    for tag, desc in paper_pts:
        p = tf10_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.1); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    # Right Card: 4 Core Contributions
    c10_r = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c10_r.fill.solid(); c10_r.fill.fore_color.rgb = c_white; c10_r.line.color.rgb = c_orange; c10_r.line.width = Pt(1.5)
    b10_r = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b10_r.fill.solid(); b10_r.fill.fore_color.rgb = c_orange; b10_r.line.fill.background()
    b10_r.text_frame.text = "02"; b10_r.text_frame.paragraphs[0].font.name = 'Calibri'; b10_r.text_frame.paragraphs[0].font.size = Pt(11); b10_r.text_frame.paragraphs[0].font.bold = True; b10_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf10_r = s9.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf10_r.word_wrap = True
    p = tf10_r.paragraphs[0]; p.text = "4 Đóng góp Học thuật Cốt lõi của Bản thảo"; p.font.bold = True; p.font.size = Pt(12.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    contrib_pts = [
        ("Đóng góp 1 (Tái tham số hóa RepConv):", "Chứng minh bằng đại số khả năng gộp 3 nhánh huấn luyện về 1 nhân Conv 3x3 duy nhất, triệt tiêu phân mảnh bộ nhớ và đạt Zero Latency Overhead."),
        ("Đóng góp 2 (Mã hóa Tọa độ CoordConv):", "Cung cấp tiên nghiệm vị trí thẳng đứng phá vỡ tính bất biến tịnh tiến của CNN, triệt tiêu dứt điểm báo động giả xô vàng và đồ bảo hộ dưới sàn nhà."),
        ("Đóng góp 3 (Định tuyến Thưa BiFormer):", "Đưa độ phức tạp tính toán chú ý từ bậc hai O(H^2W^2) về tuyến tính O(HW), giải quyết triệt để nguy cơ sập bộ nhớ CUDA OOM khi bắt mục tiêu mũ bảo hộ ở xa."),
        ("Đóng góp 4 (Phân rã Focal-EIoU Loss):", "Khắc phục hiện tượng triệt tiêu gradient của hàm CIoU bằng cách phân rã độc lập sai số kích thước w và h, giúp hồi quy khung bao nhỏ chính xác tối đa.")
    ]
    for tag, desc in contrib_pts:
        p = tf10_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.1); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    add_banner(s9, "Đề tài định hướng xây dựng bản thảo bài báo khoa học chuẩn quốc tế, chứng minh 4 đóng góp toán học thực chất và sẵn sàng trình Hội đồng nghiệm thu.")

    # =========================================================================
    # SLIDE 10: MỤC 11 · MA TRẬN QUẢN TRỊ RỦI RO (RISK MANAGEMENT)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    setup_header(s10, "MỤC 11 · MA TRẬN QUẢN TRỊ RỦI RO & KHẮC PHỤC CHỦ ĐỘNG", "12 / 18",
                 "Ma trận Nhận diện Rủi ro Kỹ thuật & Phương án Khắc phục Chủ động")

    risk_cards = [
        (Inches(0.8), Inches(1.35), "01", c_red, "Rủi ro Thiếu hụt Tài nguyên Tính toán GPU", [
            "Mô tả Rủi ro: Thời gian huấn luyện lâu hoặc bị ngắt quãng do giới hạn quota GPU miễn phí trên nền tảng đám mây.",
            "Xác suất: Trung bình | Tác động: Cao.",
            "Biện pháp Khắc phục: Phân bổ mô hình huấn luyện sang Google Colab Pro và Kaggle GPU T4 x 2; sử dụng Mixed Precision FP16 và cơ chế Checkpoint Resume tự động lưu mỗi 5 epoch."
        ]),
        (Inches(6.8), Inches(1.35), "02", c_orange, "Rủi ro Quá khớp (Overfitting) & Rò rỉ Dữ liệu", [
            "Mô tả Rủi ro: Mô hình ghi nhớ dữ liệu tập train dẫn tới suy giảm nghiêm trọng độ chính xác khi đưa ra môi trường công trường thực tế.",
            "Xác suất: Trung bình | Tác động: Cao.",
            "Biện pháp Khắc phục: Kiểm định chéo 5-Fold Stratified CV, nhóm sequence ảnh camera ngăn ngừa Data Leakage; áp dụng Mosaic, Mixup và HSV Jitter tăng tính đa dạng."
        ]),
        (Inches(0.8), Inches(3.95), "03", c_purple, "Rủi ro Bất đồng Nhãn & Sai lệch Đa miền", [
            "Mô tả Rủi ro: 5 tập dữ liệu kiểm thử ngoại miền có quy ước gán nhãn mâu thuẫn (nhãn vẽ toàn thân vs vẽ chỏm đầu), gây suy giảm mAP.",
            "Xác suất: Cao | Tác động: Trung bình.",
            "Biện pháp Khắc phục: Thiết lập bộ quy chuẩn ánh xạ không gian nhãn chung C* = {0: 'hat', 1: 'person'}, tự động đồng bộ hóa toàn bộ bounding boxes ngoại miền."
        ]),
        (Inches(6.8), Inches(3.95), "04", c_blue, "Rủi ro Trễ Hạn Hoàn thành Bài báo & Báo cáo", [
            "Mô tả Rủi ro: Khối lượng công việc nghiên cứu và kỹ thuật lớn dẫn tới nguy cơ trễ hạn nộp bản thảo bài báo và quyển khóa luận tốt nghiệp.",
            "Xác suất: Thấp | Tác động: Cao.",
            "Biện pháp Khắc phục: Áp dụng quy trình Agile sprint hàng tuần với bảng Kanban; Trưởng nhóm rà soát tiến độ định kỳ; hoàn thiện bản thảo song song với quá trình chạy mã nguồn."
        ])
    ]

    for c_left, c_top, b_num, b_col, card_title, card_items in risk_cards:
        sc = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, c_top, Inches(5.7), Inches(2.45))
        sc.fill.solid(); sc.fill.fore_color.rgb = c_white; sc.line.color.rgb = c_border; sc.line.width = Pt(1)

        sb = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.2), c_top + Inches(0.15), Inches(0.42), Inches(0.32))
        sb.fill.solid(); sb.fill.fore_color.rgb = b_col; sb.line.fill.background()
        sb.text_frame.text = b_num; sb.text_frame.paragraphs[0].font.name = 'Calibri'; sb.text_frame.paragraphs[0].font.size = Pt(11); sb.text_frame.paragraphs[0].font.bold = True; sb.text_frame.paragraphs[0].font.color.rgb = c_white

        tf = s10.shapes.add_textbox(c_left + Inches(0.75), c_top + Inches(0.13), Inches(4.75), Inches(2.2)).text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = card_title; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
        for itm in card_items:
            p = tf.add_paragraph(); p.text = "• " + itm; p.font.size = Pt(8.8); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(2.5)

    add_banner(s10, "Nhóm chủ động nhận diện 4 rủi ro kỹ thuật và thiết lập phương án dự phòng chi tiết, bảo đảm dự án vận hành an toàn và về đích đúng kế hoạch.")

    # =========================================================================
    # SLIDE 11: MỤC 12 · KẾT QUẢ KỲ VỌNG & TỔNG KẾT BẢO VỆ KHÓA LUẬN
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    setup_header(s11, "MỤC 12 · KẾT QUẢ KỲ VỌNG & TỔNG KẾT BẢO VỆ KHÓA LUẬN", "13 / 18",
                 "Sản phẩm Kỳ vọng Bàn giao & Tổng hợp Giá trị Đóng góp Khóa luận")

    # Left Column: Deliverables
    c12_l = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c12_l.fill.solid(); c12_l.fill.fore_color.rgb = c_white; c12_l.line.color.rgb = c_green; c12_l.line.width = Pt(1.5)
    b12_l = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b12_l.fill.solid(); b12_l.fill.fore_color.rgb = c_green; b12_l.line.fill.background()
    b12_l.text_frame.text = "01"; b12_l.text_frame.paragraphs[0].font.name = 'Calibri'; b12_l.text_frame.paragraphs[0].font.size = Pt(11); b12_l.text_frame.paragraphs[0].font.bold = True; b12_l.text_frame.paragraphs[0].font.color.rgb = c_white

    tf12_l = s11.shapes.add_textbox(Inches(1.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf12_l.word_wrap = True
    p = tf12_l.paragraphs[0]; p.text = "5 Sản phẩm Kỳ vọng Bàn giao (Deliverables)"; p.font.bold = True; p.font.size = Pt(12.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    deliv_pts = [
        ("Mô hình Trọng số Rep-YOLO11s Tối ưu:", "Bộ trọng số PyTorch (.pt) và ONNX/TensorRT đã tái tham số hóa, đạt mAP50 >94.8% và Recallhat >91%."),
        ("Pipeline Video Camera RTSP Thời gian thực:", "Ứng dụng xử lý luồng camera giám sát CCTV với giải mã phần cứng NVDEC, hiển thị cảnh báo vi phạm trực quan >25 FPS."),
        ("Bản thảo Bài báo Khoa học Chuẩn IEEE:", "Tài liệu khoa học hoàn chỉnh, sẵn sàng trình Hội đồng Đánh giá và nộp tới các hội nghị / tạp chí quốc tế uy tín."),
        ("Tập Dữ liệu Chuẩn hóa C* & Kho Code:", "Bộ dữ liệu SHWD đã làm sạch nhãn rác, ánh xạ C* cho 5 tập ngoại vi và kho mã nguồn GitHub minh bạch."),
        ("Toàn văn Báo cáo Khóa luận (Thesis Book):", "Quyển báo cáo khóa luận tốt nghiệp chi tiết, giải trình đầy đủ cơ sở lý thuyết, quy trình thực nghiệm và phân tích chuyên sâu.")
    ]
    for tag, desc in deliv_pts:
        p = tf12_l.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.1); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    # Right Column: Synthesis & Conclusion
    c12_r = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(5.7), Inches(5.05))
    c12_r.fill.solid(); c12_r.fill.fore_color.rgb = c_white; c12_r.line.color.rgb = c_blue; c12_r.line.width = Pt(1.5)
    b12_r = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(0.42), Inches(0.32))
    b12_r.fill.solid(); b12_r.fill.fore_color.rgb = c_blue; b12_r.line.fill.background()
    b12_r.text_frame.text = "02"; b12_r.text_frame.paragraphs[0].font.name = 'Calibri'; b12_r.text_frame.paragraphs[0].font.size = Pt(11); b12_r.text_frame.paragraphs[0].font.bold = True; b12_r.text_frame.paragraphs[0].font.color.rgb = c_white

    tf12_r = s11.shapes.add_textbox(Inches(7.55), Inches(1.48), Inches(4.8), Inches(4.8)).text_frame; tf12_r.word_wrap = True
    p = tf12_r.paragraphs[0]; p.text = "Tổng kết Giá trị & Khẳng định Khóa luận"; p.font.bold = True; p.font.size = Pt(12.5); p.font.color.rgb = c_title; p.font.name = 'Calibri'
    synth_pts = [
        ("Ý nghĩa Thực tiễn Cấp bách:", "Giải quyết bài toán an toàn lao động tối quan trọng: ngăn ngừa tai nạn chấn thương đầu tử vong tại công trường xây dựng."),
        ("Cơ sở Khoa học Vững chắc:", "Khắc phục triệt để các khoảng trống nghiên cứu được chỉ ra từ hơn 30 công trình quốc tế giai đoạn 2019-2026."),
        ("4 Cải tiến Toán học Đột phá:", "Kết hợp RepConv (Zero Latency), CoordConv (Khử báo động sàn), BiFormer (Bắt mũ nhỏ) và Focal-EIoU (Hồi quy sắc nét)."),
        ("Đối chuẩn Khách quan & Vượt trội:", "Chứng minh vượt trội về Recallhat (91.33%) và F1hat (0.9190) trước cả hai baseline YOLO11s và YOLOv8s."),
        ("Kết hợp Hoàn hảo Nghiên cứu & Ứng dụng:", "Cầu nối hài hòa giữa sự chuẩn mực, khắt khe của nghiên cứu học thuật và giá trị thực tế bảo vệ sinh mạng con người.")
    ]
    for tag, desc in synth_pts:
        p = tf12_r.add_paragraph(); p.text = f"• {tag} {desc}"; p.font.size = Pt(9.1); p.font.color.rgb = c_body; p.font.name = 'Calibri'; p.space_before = Pt(4)

    add_banner(s11, "Một khóa luận AI hoàn chỉnh và chuẩn mực, kết hợp hài hòa giữa độ chính xác cứu sinh, phần mềm thực thi mạnh mẽ và giá trị khoa học đích thực.")

    # Calibri font enforcement
    for slide in [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    p.font.name = 'Calibri'

    prs.save(output_pptx_path)
    print(f"Successfully generated Vietnamese PowerPoint deck: {output_pptx_path}")

if __name__ == '__main__':
    out_dir = os.path.join(os.getcwd(), 'review_1_main')
    pptx_path = os.path.join(out_dir, 'Slides_Tieng_Viet_Muc_4_den_12.pptx')
    create_vietnamese_deck(pptx_path)
