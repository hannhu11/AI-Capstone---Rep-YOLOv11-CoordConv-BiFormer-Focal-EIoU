# -*- coding: utf-8 -*-
"""
Builds the 18-slide Vietnamese Visual Master Presentation for FPT University AI Capstone Defense.
Key Principles:
1. 100% TIẾNG VIỆT CHUẨN MỰC: Toàn bộ tiêu đề, nội dung, chú thích và thẻ số liệu.
2. ÍT CHỮ (MINIMAL TEXT, HIGH SIGNAL): Từ khóa đắt giá, câu ngắn gọn, số liệu to bản nổi bật, không có khối chữ dày đặc.
3. NHIỀU HÌNH ẢNH & SƠ ĐỒ LỚN: Sử dụng 100% hình ảnh thực tế từ bài báo (Fig 1 đến Fig 6) và 4 sơ đồ giải phẫu kiến trúc.
4. THUYẾT TRÌNH MƯỢT MÀ: Bố cục thông thoáng, nhịp điệu logic tự nhiên từ Đặt vấn đề -> Giải pháp -> Đóng góp -> Thực nghiệm -> Ứng dụng.
5. KHÔNG CÓ TỪ CẤM: Loại bỏ hoàn toàn IEEE, Review, Reviews.
"""
import os
import sys
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Warm Light Academic Palette
    BG_COLOR = RGBColor(250, 248, 245)      # #FAF8F5 Warm Ivory
    CARD_BG = RGBColor(255, 255, 255)       # #FFFFFF Pure White
    TEXT_DARK = RGBColor(15, 23, 42)        # #0F172A Deep Slate / Navy
    TEXT_MUTED = RGBColor(100, 116, 139)    # #64748B Slate Grey
    TEXT_BODY = RGBColor(51, 65, 85)        # #334155 Body Charcoal
    BORDER_COLOR = RGBColor(226, 232, 240)  # #E2E8F0 Subtle Border
    ORANGE_ACCENT = RGBColor(234, 88, 12)   # #EA580C Industrial Safety Orange
    AMBER_ACCENT = RGBColor(217, 119, 6)    # #D97706 Warm Amber
    BLUE_ACCENT = RGBColor(37, 99, 235)     # #2563EB Technical Blue
    GREEN_ACCENT = RGBColor(22, 163, 74)    # #16A34A Success Green
    LIGHT_ORANGE = RGBColor(255, 247, 237)  # #FFF7ED
    LIGHT_BLUE = RGBColor(239, 246, 255)    # #EFF6FF
    LIGHT_GREEN = RGBColor(240, 253, 244)   # #F0FDF4

    fig_dir = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\figures'

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title, category, slide_num_str):
        # Category tracker
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.32), Inches(9.5), Inches(0.28))
        tf_cat = tb_cat.text_frame
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ORANGE_ACCENT

        # Slide Number
        tb_num = slide.shapes.add_textbox(Inches(11.3), Inches(0.32), Inches(1.2), Inches(0.28))
        tf_num = tb_num.text_frame
        tf_num.margin_left = tf_num.margin_right = tf_num.margin_top = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = slide_num_str
        p_num.font.size = Pt(11)
        p_num.font.bold = True
        p_num.font.color.rgb = TEXT_MUTED
        p_num.alignment = PP_ALIGN.RIGHT

        # Big Clear Title
        tb_t = slide.shapes.add_textbox(Inches(0.8), Inches(0.62), Inches(11.7), Inches(0.55))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(20)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_DARK

    def add_card(slide, left_in, top_in, width_in, height_in, bg_color=CARD_BG, border_color=BORDER_COLOR, border_width=1.0):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(border_width)
        else:
            shape.line.fill.background()
        return shape

    # =========================================================================
    # SLIDE 1: TRANG BÌA (TITLE SLIDE)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout); set_slide_bg(s1)
    
    # Institution badge
    add_card(s1, 0.8, 0.45, 5.8, 0.38, LIGHT_ORANGE, ORANGE_ACCENT)
    tb = s1.shapes.add_textbox(Inches(0.92), Inches(0.50), Inches(5.6), Inches(0.28))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    tb.text_frame.paragraphs[0].text = 'ĐẠI HỌC FPT · BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỐT NGHIỆP KỸ SƯ AI'
    tb.text_frame.paragraphs[0].font.size = Pt(9.5); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = ORANGE_ACCENT

    # Main Project Title
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.05), Inches(7.5), Inches(1.10))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    tb.text_frame.paragraphs[0].text = 'Rep-YOLO11s'
    tb.text_frame.paragraphs[0].font.size = Pt(44); tb.text_frame.paragraphs[0].font.bold = True; tb.text_frame.paragraphs[0].font.color.rgb = TEXT_DARK

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.20), Inches(7.5), Inches(1.30))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = 'Phát Hiện Mũ Bảo Hộ Thời Gian Thực Trong Giám Sát Công Trường:\nTái Tham Số Hóa Cấu Trúc, Mã Hóa Tọa Độ & Khái Quát Hóa Đa Miền'
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT_MUTED

    # Right Hero Metric Showcase
    add_card(s1, 8.5, 1.05, 4.0, 3.40, CARD_BG, ORANGE_ACCENT, border_width=1.5)
    tb = s1.shapes.add_textbox(Inches(8.75), Inches(1.20), Inches(3.5), Inches(3.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'ĐỈNH 5-FOLD CV mAP50'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
    p = tf.add_paragraph(); p.text = '97.11%'; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    p = tf.add_paragraph(); p.text = '✓ Vượt trội Baseline YOLO11s (+0.47% mAP50)'; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT; p.space_after = Pt(6)
    
    stats_lines = [
        ('Độ trễ Tesla T4', '2.92 ms (342.5 FPS)'),
        ('Độ trễ RTX 3050', '5.35 ms (187.1 FPS)'),
        ('Laptop rẻ MX230 (2GB)', '36.0 ms (27.8 FPS)'),
        ('Tập dữ liệu chuẩn hóa', '> 33,000 ảnh đa miền')
    ]
    for lbl, val in stats_lines:
        p = tf.add_paragraph()
        p.text = f'• {lbl}: '
        p.font.size = Pt(9.2); p.font.color.rgb = TEXT_MUTED
        run = p.add_run()
        run.text = val; run.font.bold = True; run.font.color.rgb = TEXT_DARK

    # Bottom 3 Info Cards
    info_boxes = [
        ('NHÓM NGHIÊN CỨU', 'Nguyễn Hàn Như · SE183644 (Trưởng nhóm)\nNguyễn Văn Thành · SE183645\nNguyễn Tuấn Dũng · SE183646', 'Phụ trách: Mô hình, Tối ưu hóa TensorRT & Pipeline RTSP', ORANGE_ACCENT),
        ('GIẢNG VIÊN HƯỚNG DẪN', 'ThS. Vũ Hải Anh (anhvh@fe.edu.vn)\nBộ môn Trí tuệ Nhân tạo — Khoa CNTT\nTrường Đại học FPT Hà Nội', 'Định hướng: Nghiên cứu ứng dụng công nghiệp chất lượng cao', BLUE_ACCENT),
        ('ĐỊNH HƯỚNG CÔNG BỐ', 'Kỷ Yếu / Tạp Chí Quốc Tế Hạng Q1\nChuyên ngành Tin học Công nghiệp\n& Thị giác Máy tính', 'Chuẩn mực: Toán học giải tích, thực nghiệm kiểm chứng 100%', GREEN_ACCENT)
    ]
    for idx, (head, main_txt, sub_txt, col) in enumerate(info_boxes):
        left_pos = 0.8 + idx * 4.0
        add_card(s1, left_pos, 4.65, 3.8, 2.30)
        tb = s1.shapes.add_textbox(Inches(left_pos + 0.18), Inches(4.78), Inches(3.44), Inches(2.05))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = head; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = main_txt; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(4)
        p = tf.add_paragraph(); p.text = sub_txt; p.font.size = Pt(9); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: BỐI CẢNH CÔNG TRƯỜNG & 4 NÚT THẮT THỰC TẾ
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout); set_slide_bg(s2)
    add_header(s2, 'Bối Cảnh Thực Tế Công Trường & 4 Nút Thắt Kỹ Thuật', 'Tiêu Chí 1 · Mục Tiêu Đề Tài & Tính Cấp Thiết', '02 / 18')

    # Left: 4 Bottleneck visual cards (Compact, punchy)
    bottlenecks = [
        ('01', 'Vi Vật Thể Từ Xa (<20 px)', 'Camera góc cao 15–30m, mũ bị mờ nhòe và biến mất qua các tầng trích xuất sâu.', ORANGE_ACCENT),
        ('02', 'Mất Cân Bằng Nhãn (1:12)', '9,044 mũ bảo hộ đối đầu 111,514 thân người, gradient lớp người áp đảo hoàn toàn.', BLUE_ACCENT),
        ('03', 'Nhiễu Màu Sắc & Báo Động Giả', 'Xô vữa vàng, cọc tiêu, biển báo nguy hiểm bị nhận diện nhầm do CNN bất biến tịnh tiến.', AMBER_ACCENT),
        ('04', 'Rào Cản Phần Cứng Biên', 'Đòi hỏi >= 25 FPS thời gian thực trên laptop/GPU giá rẻ, không dùng cụm server đắt đỏ.', GREEN_ACCENT)
    ]
    for idx, (num, h_text, desc, col) in enumerate(bottlenecks):
        top_pos = 1.35 + idx * 1.35
        add_card(s2, 0.8, top_pos, 5.6, 1.25)
        # Accent pill
        add_card(s2, 0.95, top_pos + 0.15, 0.45, 0.35, col, col)
        tb_num = s2.shapes.add_textbox(Inches(0.95), Inches(top_pos + 0.16), Inches(0.45), Inches(0.35))
        tb_num.text_frame.margin_left = tb_num.text_frame.margin_right = tb_num.text_frame.margin_top = tb_num.text_frame.margin_bottom = 0
        tb_num.text_frame.paragraphs[0].text = num
        tb_num.text_frame.paragraphs[0].font.size = Pt(11); tb_num.text_frame.paragraphs[0].font.bold = True; tb_num.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255); tb_num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        tb = s2.shapes.add_textbox(Inches(1.52), Inches(top_pos + 0.12), Inches(4.7), Inches(1.05))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_text; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = desc; p.font.size = Pt(9.2); p.font.color.rgb = TEXT_BODY

    # Right: Large Authentic CCTV Image
    fig1_path = os.path.join(fig_dir, 'Fig1_site_overview_challenges.jpg')
    if os.path.exists(fig1_path):
        add_card(s2, 6.7, 1.35, 5.8, 5.30, CARD_BG, BORDER_COLOR)
        s2.shapes.add_picture(fig1_path, Inches(6.85), Inches(1.50), Inches(5.5), Inches(4.40))
        tb_cap = s2.shapes.add_textbox(Inches(6.85), Inches(6.00), Inches(5.5), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Hình 1: Hiện trường công trường thực tế — Góc quay camera trên cao, vi vật thể nhỏ, công nhân bị giàn giáo che khuất và bề mặt bê tông gây lóa sáng.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: KHẢO SÁT 32 BÀI BÁO & KHOẢNG TRỐNG SOTA
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout); set_slide_bg(s3)
    add_header(s3, 'Khảo Sát 32 Công Trình Quốc Tế & Khoảng Trống Nghiên Cứu', 'Tiêu Chí 3 · Tính Khả Thi & Cơ Sở Khoa Học', '03 / 18')

    RED_ACCENT = RGBColor(239, 68, 68)

    models_comp = [
        ('YOLOv8 / YOLO11 Gốc', '94.74% mAP50', '6.52 ms (153.3 FPS)', 'Tốc độ tốt nhưng thiếu nhận biết tọa độ; báo động giả nhiều ở xô vữa mặt đất.', RED_ACCENT),
        ('EC-YOLOv8 (Zhang 2024)', '95.70% mAP50', '5.80 ms (172.4 FPS)', 'Độ chính xác cao nhờ CARAFE nhưng cấu trúc quá nặng, tốc độ bị giới hạn.', AMBER_ACCENT),
        ('YOLO-CBF (Li 2023)', '95.60% mAP50', '12.40 ms (80.6 FPS)', 'Kết hợp CoordConv & BiFormer nhưng cồng kềnh (37.2M params, 104.5 GFLOPs).', AMBER_ACCENT),
        ('YOLOv8n-FADS (Fu 2024)', '79.70% mAP50', 'Khó vận hành biên', 'Mở rộng nhánh P2 gây bùng nổ tính toán, độ chính xác thực tế tụt giảm sâu.', RED_ACCENT),
        ('Rep-YOLO11s (Đề Tài Này)', '94.83% - 97.11%', '2.92 ms (342.5 FPS)', 'Cơ chế W_fused triệt tiêu độ trễ; CoordConv dập tắt báo động giả; chạy mượt MX230.', GREEN_ACCENT)
    ]

    for idx, (m_name, m_map, m_speed, m_weak, col) in enumerate(models_comp):
        top_pos = 1.35 + idx * 0.95
        is_ours = idx == 4
        bg_c = LIGHT_GREEN if is_ours else CARD_BG
        brd_c = GREEN_ACCENT if is_ours else BORDER_COLOR
        add_card(s3, 0.8, top_pos, 11.7, 0.85, bg_c, brd_c, border_width=1.8 if is_ours else 1.0)

        # Name
        tb = s3.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.12), Inches(3.0), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = m_name; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = col if is_ours else TEXT_DARK
        
        # Stats
        tb = s3.shapes.add_textbox(Inches(4.1), Inches(top_pos + 0.12), Inches(2.3), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = f'mAP50: {m_map}'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = f'Độ trễ: {m_speed}'; p.font.size = Pt(9); p.font.color.rgb = TEXT_MUTED

        # Description
        tb = s3.shapes.add_textbox(Inches(6.5), Inches(top_pos + 0.12), Inches(5.8), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = m_weak; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_BODY

    # Bottom Takeaway Card
    add_card(s3, 0.8, 6.25, 11.7, 0.80, LIGHT_ORANGE, ORANGE_ACCENT)
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(6.32), Inches(11.3), Inches(0.65))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = '★ KHOẢNG TRỐNG KHOA HỌC: Chưa có công trình nào đạt được đồng thời độ chính xác vi vật thể đỉnh cao mà KHÔNG phải trả giá bằng độ trễ suy luận hoặc sự phình to tham số khi triển khai thực tế.'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT

    # =========================================================================
    # SLIDE 4: PHẠM VI & 5 SẢN PHẨM BÀN GIAO CỤ THỂ
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout); set_slide_bg(s4)
    add_header(s4, 'Phạm Vi Đề Tài & Cam Kết 5 Sản Phẩm Bàn Giao', 'Tiêu Chí 2 · Kết Quả Cuối Cùng & Sản Phẩm Đạt Được', '04 / 18')

    deliverables = [
        ('01', 'BÁO CÁO KHOA HỌC', 'Chuyên luận 9 trang đầy đủ chứng minh giải tích, đối sánh SOTA, Ablation & giải thích Grad-CAM.', 'Chuẩn Q1 Quốc tế', ORANGE_ACCENT),
        ('02', 'MÔ HÌNH REP-YOLO11s', 'Trọng số hoàn chỉnh đạt 94.83% Single Test, 96.64% 5-Fold, 97.03% trên Hard Hat Workers.', 'Chính xác & Khái quát', BLUE_ACCENT),
        ('03', 'SOFTWARE PROTOTYPE', 'Phần mềm giải mã RTSP đa luồng, đệm chống giật khung hình, đạt 65–95 FPS trên RTX 3050.', 'Vận hành thực tế', GREEN_ACCENT),
        ('04', 'BỘ ENGINE TỐI ƯU', 'Đóng gói TensorRT 11.2 FP16 (.engine), ONNX INT8 (.onnx), OpenVINO sẵn sàng cho mọi thiết bị biên.', 'Zero-latency Deploy', AMBER_ACCENT),
        ('05', 'DATASET CHUẨN HÓA', 'Bộ dữ liệu chuẩn hóa >33,000 ảnh từ 6 nguồn công nghiệp, làm sạch triệt để nhãn rác VOC2028.', '33,000+ ảnh chuẩn', RGBColor(147, 51, 234))
    ]

    for idx, (num, h_text, desc, tag, col) in enumerate(deliverables):
        top_pos = 1.35 + idx * 1.00
        add_card(s4, 0.8, top_pos, 11.7, 0.88)
        
        # Pill
        add_card(s4, 1.0, top_pos + 0.16, 0.45, 0.35, col, col)
        tb_num = s4.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.17), Inches(0.45), Inches(0.35))
        tb_num.text_frame.margin_left = tb_num.text_frame.margin_right = tb_num.text_frame.margin_top = tb_num.text_frame.margin_bottom = 0
        tb_num.text_frame.paragraphs[0].text = num
        tb_num.text_frame.paragraphs[0].font.size = Pt(11); tb_num.text_frame.paragraphs[0].font.bold = True; tb_num.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255); tb_num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # Title
        tb = s4.shapes.add_textbox(Inches(1.6), Inches(top_pos + 0.14), Inches(3.2), Inches(0.60))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_text; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = tag; p.font.size = Pt(9); p.font.bold = True; p.font.color.rgb = col

        # Description
        tb = s4.shapes.add_textbox(Inches(4.9), Inches(top_pos + 0.18), Inches(7.4), Inches(0.55))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = desc; p.font.size = Pt(10); p.font.color.rgb = TEXT_BODY

    # Bottom Status Banner
    add_card(s4, 0.8, 6.45, 11.7, 0.60, LIGHT_GREEN, GREEN_ACCENT)
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(6.50), Inches(11.3), Inches(0.50))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = '✓ HIỆN TRẠNG TIẾN ĐỘ: Đã hoàn thiện đầy đủ 5/5 sản phẩm cam kết, đạt ~90% khối lượng toàn khóa ngay tại Giai đoạn 1!'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT

    # Helper for Side-by-Side Comparison Slides (Slides 05 to 09)
    def add_side_by_side_slide(slide, title, category, slide_num,
                               left_title, left_sub, left_img_name, left_bullets,
                               right_title, right_sub, right_img_name, right_bullets):
        add_header(slide, title, category, slide_num)
        
        card_w = 5.75
        card_h = 5.85
        top_pos = 1.25
        sci_dir = os.path.join(fig_dir, 'scientific_exports')
        
        # LEFT CARD (Baseline / Flaw)
        left_x = 0.80
        add_card(slide, left_x, top_pos, card_w, card_h, CARD_BG, BORDER_COLOR, border_width=1.0)
        
        # Left Title Header Pill
        add_card(slide, left_x + 0.15, top_pos + 0.12, card_w - 0.30, 0.48, LIGHT_ORANGE, ORANGE_ACCENT, border_width=1.0)
        tb_lh = slide.shapes.add_textbox(Inches(left_x + 0.25), Inches(top_pos + 0.14), Inches(card_w - 0.50), Inches(0.44))
        tf_lh = tb_lh.text_frame; tf_lh.word_wrap = True
        tf_lh.margin_left = tf_lh.margin_right = tf_lh.margin_top = tf_lh.margin_bottom = 0
        p = tf_lh.paragraphs[0]; p.text = left_title; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
        p = tf_lh.add_paragraph(); p.text = left_sub; p.font.size = Pt(8.2); p.font.color.rgb = TEXT_MUTED
        
        # Left Picture
        left_img_path = os.path.join(sci_dir, left_img_name)
        if not os.path.exists(left_img_path):
            left_img_path = os.path.join(fig_dir, left_img_name)
        if os.path.exists(left_img_path):
            slide.shapes.add_picture(left_img_path, Inches(left_x + 0.15), Inches(top_pos + 0.68), Inches(card_w - 0.30), Inches(2.95))
            
        # Left Bullets Box
        add_card(slide, left_x + 0.15, top_pos + 3.72, card_w - 0.30, 1.98, RGBColor(254, 242, 242), RGBColor(252, 165, 165), border_width=0.8)
        tb_lb = slide.shapes.add_textbox(Inches(left_x + 0.25), Inches(top_pos + 3.80), Inches(card_w - 0.50), Inches(1.82))
        tf_lb = tb_lb.text_frame; tf_lb.word_wrap = True
        tf_lb.margin_left = tf_lb.margin_right = tf_lb.margin_top = tf_lb.margin_bottom = 0
        for i, (badge, text) in enumerate(left_bullets):
            p = tf_lb.paragraphs[0] if i == 0 else tf_lb.add_paragraph()
            p.text = f"{badge} "
            p.font.size = Pt(9.0); p.font.bold = True; p.font.color.rgb = RGBColor(185, 28, 28)
            run = p.add_run()
            run.text = text
            run.font.size = Pt(8.8); run.font.bold = False; run.font.color.rgb = TEXT_BODY
            p.space_after = Pt(2)

        # RIGHT CARD (Proposed / Innovation)
        right_x = 6.78
        add_card(slide, right_x, top_pos, card_w, card_h, CARD_BG, BORDER_COLOR, border_width=1.0)
        
        # Right Title Header Pill
        add_card(slide, right_x + 0.15, top_pos + 0.12, card_w - 0.30, 0.48, LIGHT_GREEN, GREEN_ACCENT, border_width=1.0)
        tb_rh = slide.shapes.add_textbox(Inches(right_x + 0.25), Inches(top_pos + 0.14), Inches(card_w - 0.50), Inches(0.44))
        tf_rh = tb_rh.text_frame; tf_rh.word_wrap = True
        tf_rh.margin_left = tf_rh.margin_right = tf_rh.margin_top = tf_rh.margin_bottom = 0
        p = tf_rh.paragraphs[0]; p.text = right_title; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
        p = tf_rh.add_paragraph(); p.text = right_sub; p.font.size = Pt(8.2); p.font.color.rgb = TEXT_MUTED

        # Right Picture
        right_img_path = os.path.join(sci_dir, right_img_name)
        if not os.path.exists(right_img_path):
            right_img_path = os.path.join(fig_dir, right_img_name)
        if os.path.exists(right_img_path):
            slide.shapes.add_picture(right_img_path, Inches(right_x + 0.15), Inches(top_pos + 0.68), Inches(card_w - 0.30), Inches(2.95))

        # Right Bullets Box
        add_card(slide, right_x + 0.15, top_pos + 3.72, card_w - 0.30, 1.98, LIGHT_GREEN, RGBColor(134, 239, 172), border_width=0.8)
        tb_rb = slide.shapes.add_textbox(Inches(right_x + 0.25), Inches(top_pos + 3.80), Inches(card_w - 0.50), Inches(1.82))
        tf_rb = tb_rb.text_frame; tf_rb.word_wrap = True
        tf_rb.margin_left = tf_rb.margin_right = tf_rb.margin_top = tf_rb.margin_bottom = 0
        for i, (badge, text) in enumerate(right_bullets):
            p = tf_rb.paragraphs[0] if i == 0 else tf_rb.add_paragraph()
            p.text = f"{badge} "
            p.font.size = Pt(9.0); p.font.bold = True; p.font.color.rgb = RGBColor(21, 128, 61)
            run = p.add_run()
            run.text = text
            run.font.size = Pt(8.8); run.font.bold = False; run.font.color.rgb = TEXT_BODY
            p.space_after = Pt(2)

    # =========================================================================
    # SLIDE 5: ĐỐI CHIẾU KIẾN TRÚC YOLO11s (3 HEADS) VS. REP-YOLO11s-P2 (4 HEADS)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout); set_slide_bg(s5)
    add_side_by_side_slide(
        s5,
        title='Đối Chiếu Kiến Trúc: Baseline YOLO11s (3 Heads) vs. Rep-YOLO11s-P2 AFPN (4 Heads)',
        category='ĐỀ XUẤT MÔ HÌNH · GIẢI PHẪU KIẾN TRÚC NƠ-RON',
        slide_num='05 / 18',
        left_title='BASELINE: VANILLA YOLO11s (3 HEADS)',
        left_sub='Chỉ có 3 đầu dò P3, P4, P5 (Stride 8, 16, 32) · Bỏ sót mục tiêu < 20px',
        left_img_name='Fig5A_Baseline_YOLO11s_Architecture.png',
        left_bullets=[
            ('[❌ Nút Thắt P5 Stride 32]', 'Downsampling 32 lần làm mũ 16×16 px co rút dưới 0.5 pixel trên feature map, đặc trưng biến mất.'),
            ('[❌ Giới Hạn Cự Ly Xa]', 'Bỏ sót hoàn toàn công nhân ở xa cự ly > 25m (Recall mũ bảo hộ bị nghẽn ở mức 90.35%).'),
            ('[❌ Thiếu Dẫn Truyền P2]', 'Không có nhánh kết nối trực tiếp đặc trưng nông P2 lên các tầng cổ mạng (Neck).')
        ],
        right_title='PROPOSED: REP-YOLO11s-P2 AFPN (4 HEADS)',
        right_sub='Bổ sung nhánh P2 (Stride 4, 160×160) · Bảo toàn 100% hình học vi vật thể',
        right_img_name='Fig5B_Proposed_RepYOLO11s_Architecture.png',
        right_bullets=[
            ('[✅ Nhánh P2 Stride 4]', 'Duy trì bản đồ đặc trưng 160×160 px, vi vật thể mũ bảo hộ chiếm trọn 4×4 cells trên bản đồ.'),
            ('[⚡ Bứt Phá Recall 91.33%]', 'Tăng vọt khả năng bắt dính mục tiêu nhỏ cự ly xa (đỉnh 5-Fold Cross-Validation đạt 93.34%).'),
            ('[⚡ Cắt Giảm 51.9% Params]', 'Tối ưu hóa độ rộng kênh mạng, cắt giảm tham số từ 9.85M xuống 4.74M mà độ chính xác vẫn vượt trội.')
        ]
    )

    # =========================================================================
    # SLIDE 6: ĐỘT PHÁ 1 - TÁI THAM SỐ HÓA REPCONV & GỘP NHÁNH ĐẠI SỐ
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout); set_slide_bg(s6)
    add_side_by_side_slide(
        s6,
        title='Đột Phá 1: Tái Tham Số Hóa Cấu Trúc (RepConv) & Gộp Nhánh Đại Số',
        category='ĐÓNG GÓP KỸ THUẬT 1 · TỐI ƯU HÓA ĐỘ TRỄ SUY LUẬN',
        slide_num='06 / 18',
        left_title='BASELINE: MULTI-BRANCH TRAINING BOTTLENECK',
        left_sub='3 nhánh song song (3×3, 1×1, Identity) · Phân mảnh bộ nhớ GPU',
        left_img_name='Fig1A_Baseline_MultiBranch_Bottleneck.png',
        left_bullets=[
            ('[❌ Memory Access Cost]', 'GPU phải phân bổ 3 bộ đệm bộ nhớ riêng biệt cho từng nhánh, gây nghẽn băng thông nghiêm trọng.'),
            ('[❌ 3 Kernel Launches]', 'Tốn 3 lần gọi nhân tính toán CUDA độc lập trên từng lớp tích chập, gây hiện tượng Cache Thrashing.'),
            ('[❌ Độ Trễ Kéo Dài 7.12 ms]', 'Tốc độ thực tế chỉ đạt 140.4 FPS trên Tesla T4, không đáp ứng được luồng camera đa kênh 4K.')
        ],
        right_title='PROPOSED: REPCONV ALGEBRAIC FUSION LIFECYCLE',
        right_sub='switch_to_deploy() gộp giải tích thành duy nhất 1 lớp Conv 3×3',
        right_img_name='Fig1B_Proposed_RepConv_Algebraic_Fusion.png',
        right_bullets=[
            ('[✅ Gộp Đại Số Khép Kín]', 'BatchNorm Folding + Zero-Padding 1×1 + Dirac Delta Kernel (điều kiện Cin==Cout và s=1).'),
            ('[⚡ Giảm 55.2% Độ Trễ]', 'Thời gian suy luận giảm ngoạn mục từ 7.12 ms xuống 2.92 ms trên Tesla T4 (5.35 ms trên RTX 3050).'),
            ('[⚡ Tốc Độ 342.5 FPS]', 'Vận hành siêu thời gian thực với sai số giải tích số học cực tiểu Δ < 10⁻⁵ so với bản đa nhánh.')
        ]
    )

    # =========================================================================
    # SLIDE 7: ĐỘT PHÁ 2 - MÃ HÓA TỌA ĐỘ COORDCONV DẬP TẮT BÁO ĐỘNG GIẢ
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout); set_slide_bg(s7)
    add_side_by_side_slide(
        s7,
        title='Đột Phá 2: Mã Hóa Tọa Độ CoordConv Dập Tắt Báo Động Giả Dưới Sàn',
        category='ĐÓNG GÓP KỸ THUẬT 2 · PHÁ VỠ TÍNH BẤT BIẾN TỊNH TIẾN',
        slide_num='07 / 18',
        left_title='BASELINE: TRANSLATION INVARIANCE FLAW',
        left_sub='Tích chập tiêu chuẩn xử lý xô vàng dưới sàn giống hệt mũ trên đầu',
        left_img_name='Fig2A_Baseline_Translation_Invariance_Flaw.png',
        left_bullets=[
            ('[❌ Chia Sẻ Trọng Số]', 'Bộ lọc tích chập 3 kênh RGB không có nhận thức không gian về độ cao vật thể trong khung hình.'),
            ('[❌ Nhầm Lẫn Hình Học]', 'Xô vữa vàng và cọc tiêu ở sàn (y ≈ 0.9) kích hoạt phản hồi hệt như mũ bảo hộ trên đầu (y ≈ 0.2).'),
            ('[❌ Báo Động Giả Tràn Lan]', 'Hơn 28% cảnh báo sai xuất hiện ở khu vực sàn đất và máy móc công trường do màu sắc tương đồng.')
        ],
        right_title='PROPOSED: 5-CHANNEL COORDCONV STEM INJECTION',
        right_sub='Cấy 2 kênh tọa độ không gian đối xứng Cx, Cy ∈ [-1, 1] tại tầng Stem',
        right_img_name='Fig2B_Proposed_CoordConv_Spatial_Injection.png',
        right_bullets=[
            ('[✅ Tiên Đề Hình Học]', 'Ép mạng học tiên nghiệm giải phẫu: Mũ bảo hộ nằm trên đầu người (Cy < 0.3), không nằm ở sàn đất.'),
            ('[⚡ Dập Tắt >28% Cảnh Báo Sai]', 'Logit lớp mũ tại sàn nhà bị triệt tiêu về 0 ngay tại tầng Stem (Conv c1=5 -> c2=64).'),
            ('[⚡ Chi Phí Cực Nhỏ (+0.06 ms)]', 'Giữ nguyên các tầng sâu là tích chập chuẩn, đạt hiệu quả tối đa với chi phí tính toán tối thiểu.')
        ]
    )

    # =========================================================================
    # SLIDE 8: ĐỘT PHÁ 3 - CHÚ Ý THƯA 2 CẤP ĐỘ BIFORMER
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout); set_slide_bg(s8)
    add_side_by_side_slide(
        s8,
        title='Đột Phá 3: Cơ Chế Chú Ý Định Tuyến Thưa 2 Cấp Độ (BiFormer)',
        category='ĐÓNG GÓP KỸ THUẬT 3 · KHÓA NÉT VI VẬT THỂ MŨ < 20 PX',
        slide_num='08 / 18',
        left_title='BASELINE: DENSE SELF-ATTENTION BOTTLENECK',
        left_sub='Độ phức tạp bậc hai O(H²W²) · Gây tràn bộ nhớ CUDA OOM',
        left_img_name='Fig3A_Baseline_Dense_Attention_Bottleneck.png',
        left_bullets=[
            ('[❌ Bùng Nổ Tính Toán]', 'Ở độ phân giải 1024px, ma trận ái lực A cần >10¹⁰ phép tính, ngốn >4 GB VRAM tạm thời.'),
            ('[❌ Lãng Phí Tài Nguyên]', 'Hơn 80% tính toán chú ý rơi vào nền vô nghĩa (bầu trời, mặt sàn bê tông, mảng tường trống).'),
            ('[❌ Sập Nguồn Thiết Bị Biên]', 'Gây lỗi tràn bộ nhớ CUDA Out-of-Memory trên GPU biên 2GB–4GB (Jetson Nano / MX230).')
        ],
        right_title='PROPOSED: BIFORMER DYNAMIC SPARSE ROUTING',
        right_sub='Định tuyến 2 cấp độ: Lưới vùng S=8, Top-k=4 · Độ phức tạp tuyến tính O(HW)',
        right_img_name='Fig3B_Proposed_BiFormer_Sparse_Routing.png',
        right_bullets=[
            ('[✅ Lọc 80% Nền Rác]', 'Chỉ định tuyến thông tin qua 4 vùng thô quan trọng nhất chứa chi tiết công nhân và vòm mũ.'),
            ('[⚡ Độ Phức Tạp Tuyến Tính]', 'Chuyển từ O((HW)²) thành O(HW), giải phóng hoàn toàn áp lực bộ nhớ VRAM khi xử lý ảnh lớn.'),
            ('[⚡ Khóa Nét Viền Mũ Xa]', 'Dồn 100% chú ý vào mục tiêu li ti, giúp mô hình chạy mượt ở 1024px mà không bao giờ bị OOM.')
        ]
    )

    # =========================================================================
    # SLIDE 9: ĐỘT PHÁ 4 - HÀM MẤT MÁT FOCAL EIOU HỒI QUY KHUNG BAO
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout); set_slide_bg(s9)
    add_side_by_side_slide(
        s9,
        title='Đột Phá 4: Focal EIoU Loss Phân Rã Cạnh & Hồi Quy Vi Vật Thể Khó',
        category='ĐÓNG GÓP KỸ THUẬT 4 · HỒI QUY KHUNG BAO CHÍNH XÁC CAO',
        slide_num='09 / 18',
        left_title='BASELINE: CIOU GRADIENT VANISHING FLAW',
        left_sub='Phạt tỷ lệ góc v ∝ (arctan(w_gt/h_gt) - arctan(w/h))² triệt tiêu đạo hàm',
        left_img_name='Fig4A_Baseline_CIoU_Vanishing_Gradient.png',
        left_bullets=[
            ('[❌ Đạo Hàm Triệt Tiêu Về 0]', 'Khi tỷ lệ cạnh w/h == w_gt/h_gt, ∂v/∂w ≡ 0 dù diện tích hộp dự đoán bị lệch gấp đôi.'),
            ('[❌ Kẹt Hộp Bao Dự Đoán]', 'Hộp bao không thể co giãn để khớp chính xác mũ bảo hộ bị giàn giáo công trường che khuất.'),
            ('[❌ Mù Kích Thước Tuyệt Đối]', 'Chỉ phạt tỷ lệ góc tương đối mà không tối ưu trực tiếp sai lệch chiều rộng và chiều cao.')
        ],
        right_title='PROPOSED: FOCAL-EIoU INDEPENDENT DECOMPOSITION',
        right_sub='Phân rã độc lập sai số (w-w_gt)² + (h-h_gt)² + Điều tiết mẫu khó (IoU)^0.5',
        right_img_name='Fig4B_Proposed_Focal_EIoU_Decomposition.png',
        right_bullets=[
            ('[✅ Gradient Luôn Tồn Tại]', '∂L/∂w = 2(w-w_gt)/Cw² ≠ 0 liên tục dẫn dắt tối ưu cả chiều rộng và chiều cao độc lập.'),
            ('[⚡ Khai Phá Mẫu Khó]', 'Trọng số (IoU)^0.5 tập trung tối ưu biên cho các mũ bảo hộ bị che khuất và kích thước nhỏ.'),
            ('[⚡ Đồng Thuận TAL + BCE]', 'Phối hợp Task-Aligned Assigner giải quyết mất cân bằng 1:12, mAP50-95 đạt đỉnh 65.91% (Harmonized 78.93%).')
        ]
    )

    # =========================================================================
    # SLIDE 10: TỰ PHÁT TRIỂN & ĐÓNG GÓP MÃ NGUỒN LÕI
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout); set_slide_bg(s10)
    add_header(s10, 'Khẳng Định Bản Quyền: Tự Phát Triển & Tùy Biến Mã Nguồn Lõi', 'Tiêu Chí 3 · Tính Tự Chủ Kỹ Thuật Của Nhóm', '10 / 18')

    code_contributions = [
        ('01', 'TỰ VIẾT 100% PYTORCH', 'custom_ablation_modules.py', 'Tự lập trình từ đầu 360 dòng mã nguồn thuần PyTorch cho CoordConv, RepConv, BiFormer và Focal EIoU; không dùng thư viện ngoài.', ORANGE_ACCENT),
        ('02', 'CAN THIỆP VẬT LÝ FILE LOSS.PY', 'Physical File Hard-Patch', 'Khắc phục hạn chế cô lập tiến trình của PyTorch DDP trên Kaggle bằng kỹ thuật ghi đè trực tiếp loss.py để ép tính loss tùy biến.', BLUE_ACCENT),
        ('03', 'BẺ KHÓA BẢO MẬT PYTORCH 2.6', 'add_safe_globals Integration', 'Xử lý lỗi chặn nạp trọng số weights_only=True của PyTorch 2.6, chuyển giao thành công 117 layer tensor tương thích.', GREEN_ACCENT),
        ('04', 'SÁNG CHẾ GIAO THỨC HARMONIZED', 'Harmonized PPE Protocol', 'Phát hiện và giải mã hiện tượng Sụp đổ IoU do xung đột định nghĩa nhãn thân người, cứu vãn mAP ngoại miền lên 97.03%.', AMBER_ACCENT)
    ]

    for idx, (num, h_txt, sub_txt, desc, col) in enumerate(code_contributions):
        top_pos = 1.35 + idx * 1.25
        add_card(s10, 0.8, top_pos, 11.7, 1.12)
        
        # Pill
        add_card(s10, 1.0, top_pos + 0.22, 0.45, 0.35, col, col)
        tb_num = s10.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.23), Inches(0.45), Inches(0.35))
        tb_num.text_frame.margin_left = tb_num.text_frame.margin_right = tb_num.text_frame.margin_top = tb_num.text_frame.margin_bottom = 0
        tb_num.text_frame.paragraphs[0].text = num
        tb_num.text_frame.paragraphs[0].font.size = Pt(11); tb_num.text_frame.paragraphs[0].font.bold = True; tb_num.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255); tb_num.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # Title + Module
        tb = s10.shapes.add_textbox(Inches(1.6), Inches(top_pos + 0.15), Inches(3.5), Inches(0.80))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_txt; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = sub_txt; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = col

        # Detailed Impact
        tb = s10.shapes.add_textbox(Inches(5.2), Inches(top_pos + 0.20), Inches(7.1), Inches(0.70))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = desc; p.font.size = Pt(10); p.font.color.rgb = TEXT_BODY

    add_card(s10, 0.8, 6.45, 11.7, 0.60, LIGHT_ORANGE, ORANGE_ACCENT)
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(6.50), Inches(11.3), Inches(0.50))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    p = tb.text_frame.paragraphs[0]
    p.text = '★ KHẲNG ĐỊNH: Nhóm kế thừa khung sườn Ultralytics nhưng tự chủ hoàn toàn 100% thuật toán tùy biến và kỹ nghệ can thiệp tầng thấp!'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT

    # =========================================================================
    # SLIDE 11: KỸ NGHỆ DỮ LIỆU & CHUẨN HÓA 33,000+ ẢNH
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout); set_slide_bg(s11)
    add_header(s11, 'Kỹ Nghệ Dữ Liệu: Thanh Lọc VOC2028 & Chuẩn Hóa Đa Miền', 'Tiêu Chí 3 · Tính Toàn Vẹn & Quy Chuẩn Tập Dữ Liệu', '11 / 18')

    # Left: In-domain SHWD details
    add_card(s11, 0.8, 1.35, 5.6, 4.90)
    tb = s11.shapes.add_textbox(Inches(1.05), Inches(1.50), Inches(5.1), Inches(4.60))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'TẬP DỮ LIỆU NGUỒN CHUẨN (VOC2028 / SHWD)'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT
    p = tf.add_paragraph(); p.text = '7,581 Ảnh Công Trường Thực Tế'; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)
    
    shwd_points = [
        ('Phân chia nghiêm ngặt 80/20:', '6,064 ảnh TrainVal / 1,517 ảnh Test độc lập (Zero Leakage).'),
        ('Thanh lọc nhãn rác:', 'Phát hiện & xóa bỏ 3 nhãn dị thường gán nhãn "dog" trong file XML 000377.'),
        ('Tỷ lệ mất cân bằng cực đoan:', '1:12 (9,044 nhãn mũ đối đầu 111,514 nhãn thân người).'),
        ('Độ phân giải chuẩn hóa:', '960×960 native và 640×640 / 1024×1024 đa tỉ xích.')
    ]
    for lbl, val in shwd_points:
        p = tf.add_paragraph()
        p.text = f'• {lbl} '
        p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(4)

    # Right: Multi-Domain benchmark suite
    add_card(s11, 6.7, 1.35, 5.8, 4.90)
    tb = s11.shapes.add_textbox(Inches(6.95), Inches(1.50), Inches(5.3), Inches(4.60))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = '5 TẬP DỮ LIỆU KIỂM THỬ NGOẠI MIỀN (>25,000 ẢNH)'; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT
    p = tf.add_paragraph(); p.text = 'Thử Thách Khái Quát Hóa Cực Đoan'; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

    external_datasets = [
        ('GDUT-HWD (13,499 ảnh):', 'Đám đông công nhân cực dày đặc (15–30 người/khung hình).'),
        ('SHEL5K (5,000 ảnh):', 'Góc nhìn Flycam thẳng đứng từ trên cao (70–90°), mũ cực nhỏ <15px.'),
        ('Hard Hat Workers (7,000 ảnh):', 'Giám sát công trường ngoài trời với ánh sáng tự nhiên phức tạp.'),
        ('SHD & SFCHD Benchmark:', 'Hạ tầng nhà máy luyện kim, dầu khí và xưởng đóng tàu công nghiệp.')
    ]
    for lbl, val in external_datasets:
        p = tf.add_paragraph()
        p.text = f'• {lbl} '
        p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(4)

    # Bottom Unified Ontology
    add_card(s11, 0.8, 6.35, 11.7, 0.70, LIGHT_BLUE, BLUE_ACCENT)
    tb = s11.shapes.add_textbox(Inches(1.0), Inches(6.42), Inches(11.3), Inches(0.55))
    tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    p = tb.text_frame.paragraphs[0]
    p.text = '★ KHÔNG GIAN NHÃN THỐNG NHẤT C* = {0: "hat" (Mũ bảo hộ), 1: "person" (Người / Công nhân)} — Đồng bộ hóa toàn bộ 33,000+ ảnh!'
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = BLUE_ACCENT

    # =========================================================================
    # SLIDE 12: NGHIÊN CỨU CẮT BỎ THỰC NGHIỆM ABLATION (FIGURE 6)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout); set_slide_bg(s12)
    add_header(s12, 'Thực Nghiệm Cắt Bỏ (Ablation Study A0–A6): Minh Chứng Khoa Học', 'Tiêu Chí 3 · Kiểm Định Đóng Góp Từng Thành Phần', '12 / 18')

    # Left: Concise summary of A0 to A6
    add_card(s12, 0.8, 1.35, 4.8, 5.40)
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(4.4), Inches(5.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'ĐÓNG GÓP TỪNG MODULE'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    ablation_items = [
        ('A0 (Baseline YOLO11s):', '94.74% mAP50 · 6.52 ms latency.'),
        ('A1 (+ P2 Small Head):', '94.81% mAP50 · Trễ vọt lên 8.94 ms (+37%) -> Bị loại bỏ vì không tối ưu.'),
        ('A2 (+ CoordConv):', '94.78% mAP50 · Tăng độ nhận diện không gian mà không tốn chi phí.'),
        ('A3 (+ RepConv):', '94.81% mAP50 · Làm giàu gradient huấn luyện.'),
        ('A4 (+ Focal EIoU):', '94.88% mAP50 · Bắt dính bounding box vi vật thể.'),
        ('A5 (+ BiFormer):', '94.80% mAP50 · Đẩy Recall mũ lên đỉnh 91.15%.'),
        ('A6 (Full Fusion):', '94.83% mAP50 · Độ trễ giảm kỷ lục còn 2.92 ms!')
    ]
    for lbl, val in ablation_items:
        p = tf.add_paragraph()
        p.text = f'• {lbl} '
        p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(3)

    # Right: High-Res Dual-Axis Chart
    fig6_path = os.path.join(fig_dir, 'Fig6_ablation_A0_A6_tradeoff.png')
    if os.path.exists(fig6_path):
        add_card(s12, 5.8, 1.35, 6.7, 5.40, CARD_BG, BORDER_COLOR)
        s12.shapes.add_picture(fig6_path, Inches(5.95), Inches(1.50), Inches(6.4), Inches(4.50))
        tb_cap = s12.shapes.add_textbox(Inches(5.95), Inches(6.15), Inches(6.4), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Hình 6: Biểu đồ đánh đổi thực nghiệm A0–A6 — Sau khi gộp nhánh W_fused, độ trễ giảm 55.2% trong khi độ chính xác duy trì ở mức tối ưu.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 13: ĐỐI SÁNH SOTA TRÊN ĐƯỜNG BIÊN PARETO (FIGURE 5)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout); set_slide_bg(s13)
    add_header(s13, 'Đối Sánh SOTA: Vị Thế Thống Trị Trên Đường Biên Pareto', 'Tiêu Chí 4 · Ý Nghĩa Khoa Học & Giá Trị Đột Phá', '13 / 18')

    # Left: SOTA superiority highlights
    add_card(s13, 0.8, 1.35, 4.8, 5.40)
    tb = s13.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(4.4), Inches(5.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'ƯU THẾ TUYỆT ĐỐI CỦA REP-YOLO11s'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    pareto_points = [
        ('Nhanh gấp 2.23x Baseline:', 'Độ trễ 2.92 ms vs 6.52 ms của YOLO11s gốc (342.5 FPS trên Tesla T4).'),
        ('Nhanh gấp 4.25x YOLO-CBF:', 'YOLO-CBF mất 12.4 ms (80.6 FPS), mô hình của đề tài chỉ mất 2.92 ms.'),
        ('Vượt trội EC-YOLOv8 về FPS:', 'EC-YOLOv8 đạt 172.4 FPS, mô hình của đề tài đạt 342.5 FPS (nhanh gấp đôi).'),
        ('Đỉnh cao 5-Fold Stratified:', 'Đạt 96.64% ± 0.32% mAP50, đỉnh Fold 3 đạt 97.11% mAP50 và F1: 0.9396.'),
        ('Chiếm lĩnh góc trên - bên trái:', 'Vị trí lý tưởng nhất trên biểu đồ Pareto: Độ chính xác cao nhất với độ trễ thấp nhất.')
    ]
    for lbl, val in pareto_points:
        p = tf.add_paragraph()
        p.text = f'★ {lbl} '
        p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        run = p.add_run()
        run.text = val; run.font.bold = False; run.font.color.rgb = TEXT_BODY
        p.space_after = Pt(4)

    # Right: High-Res Pareto Frontier
    fig5_path = os.path.join(fig_dir, 'Fig5_efficiency_frontier_latency_vs_map.png')
    if os.path.exists(fig5_path):
        add_card(s13, 5.8, 1.35, 6.7, 5.40, CARD_BG, BORDER_COLOR)
        s13.shapes.add_picture(fig5_path, Inches(5.95), Inches(1.50), Inches(6.4), Inches(4.50))
        tb_cap = s13.shapes.add_textbox(Inches(5.95), Inches(6.15), Inches(6.4), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Hình 5: Đường biên hiệu quả Pareto (Latency vs mAP50-95) — Rep-YOLO11s đứng đầu về tỷ lệ chính xác trên mỗi mili-giây suy luận.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 14: THỊ GIÁC GIẢI THÍCH GRAD-CAM XAI (FIGURE 3)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout); set_slide_bg(s14)
    add_header(s14, 'Giải Thích Thị Giác (Grad-CAM XAI): Dập Tắt Báo Động Giả', 'Tiêu Chí 4 · Tính Minh Bạch & Khả Năng Giải Thích Của AI', '14 / 18')

    # Left: 3 Hard Scenarios explanation
    add_card(s14, 0.8, 1.35, 4.4, 5.40)
    tb = s14.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(4.0), Inches(5.10))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'KIỂM CHỨNG 3 KỊCH BẢN THỰC TẾ'; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    scenarios = [
        ('KỊCH BẢN 1: ÁO CAM PHẢN QUANG', 'Baseline bị phân tán gradient vào ngực áo và giàn giáo; Rep-YOLO11s tập trung duy nhất vào vòm mũ xanh (conf 0.84).'),
        ('KỊCH BẢN 2: LÓA SÁNG NGƯỢC NGUY HIỂM', 'Ánh sáng cửa kính làm mờ độ tương phản; BiFormer duy trì cụm kích hoạt đậm đặc tại đỉnh đầu (conf 0.89).'),
        ('KỊCH BẢN 3: BIỂN BÁO TAM GIÁC VÀNG', 'Baseline kích hoạt nhầm biển báo nguy hiểm màu vàng; kênh tọa độ CoordConv dập tắt hoàn toàn báo động giả!')
    ]
    for lbl, val in scenarios:
        p = tf.add_paragraph(); p.text = lbl; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = val; p.font.size = Pt(9.2); p.font.color.rgb = TEXT_BODY; p.space_after = Pt(4)

    # Right: High-Res Grad-CAM 3-Row Panel
    fig3_path = os.path.join(fig_dir, 'Fig3_gradcam_xai_saliency_comparison.png')
    if os.path.exists(fig3_path):
        add_card(s14, 5.4, 1.35, 7.1, 5.40, CARD_BG, BORDER_COLOR)
        s14.shapes.add_picture(fig3_path, Inches(5.55), Inches(1.48), Inches(6.8), Inches(4.55))
        tb_cap = s14.shapes.add_textbox(Inches(5.55), Inches(6.15), Inches(6.8), Inches(0.50))
        tb_cap.text_frame.margin_left = tb_cap.text_frame.margin_right = tb_cap.text_frame.margin_top = tb_cap.text_frame.margin_bottom = 0
        tb_cap.text_frame.word_wrap = True
        p = tb_cap.text_frame.paragraphs[0]
        p.text = 'Hình 3: Bản đồ nhiệt Grad-CAM đối sánh đối đầu trực tiếp — Cột (a) Ảnh gốc; (b) Baseline YOLO11s bị nhiễu; (c) Rep-YOLO11s khóa nét đỉnh đầu; (d) Hộp phát hiện cuối cùng.'
        p.font.size = Pt(8.5); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 15: KIỂM THỬ KHÁI QUÁT HÓA ĐA MIỀN & GIẢI MÃ IOU COLLAPSE
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout); set_slide_bg(s15)
    add_header(s15, 'Kiểm Thử Khái Quát Hóa Đa Miền & Giải Mã Hiện Tượng Sụp Đổ IoU', 'Tiêu Chí 4 · Thử Thách Chuyển Giao Công Nghiệp', '15 / 18')

    # Top: 3 External Dataset Scorecards
    ext_cards = [
        ('HARD HAT WORKERS (7,000 ẢNH)', '97.03%', 'Giao thức Harmonized PPE (Hat-Only)', 'Đạt độ chính xác tương đương tập nguồn SHWD; biểu diễn mũ chuyển giao hoàn hảo.', GREEN_ACCENT),
        ('GDUT-HWD (13,499 ẢNH)', '74.27%', 'Precision: 90.26% (Đám đông 15-30 người)', 'Chống chịu tuyệt vời trong điều kiện công nhân đứng san sát che khuất lẫn nhau.', BLUE_ACCENT),
        ('SHEL5K (5,000 ẢNH FLYCAM)', '41.15%', 'Precision: 85.62% (Góc nhìn 90° cực đoan)', 'Mô hình thà bỏ sót vi vật thể <15px chứ không báo động giả vào sỏi đá, mặt đất.', AMBER_ACCENT)
    ]
    for idx, (h_txt, val_txt, sub_txt, desc_txt, col) in enumerate(ext_cards):
        left_pos = 0.8 + idx * 4.0
        add_card(s15, left_pos, 1.35, 3.8, 2.50)
        tb = s15.shapes.add_textbox(Inches(left_pos + 0.15), Inches(1.48), Inches(3.5), Inches(2.25))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_txt; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = val_txt; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = sub_txt; p.font.size = Pt(9); p.font.bold = True; p.font.color.rgb = col; p.space_after = Pt(2)
        p = tf.add_paragraph(); p.text = desc_txt; p.font.size = Pt(8.8); p.font.color.rgb = TEXT_MUTED

    # Bottom: The IoU Collapse Autopsy (Crucial Academic Insight)
    add_card(s15, 0.8, 4.05, 11.7, 2.70, CARD_BG, ORANGE_ACCENT)
    tb = s15.shapes.add_textbox(Inches(1.05), Inches(4.18), Inches(11.2), Inches(2.45))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'GIẢI MÃ TOÁN HỌC: HIỆN TƯỢNG SỤP ĐỔ IOU (IOU COLLAPSE AUTOPSY)'; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = '• Xung đột định nghĩa nhãn học thuật: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    run = p.add_run()
    run.text = 'Tập nguồn SHWD gán nhãn "person" là TOÀN THÂN (Full-body), trong khi Hard Hat Workers gán nhãn "person" chỉ là NỬA ĐẦU (Head-only).'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    p = tf.add_paragraph()
    p.text = '• Sụp đổ toán học: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    run = p.add_run()
    run.text = 'Khi detector dự đoán đúng thân người, hộp ground-truth chỉ là cái đầu nằm lọt bên trong. Chỉ số IoU = Diện tích đầu / Diện tích thân ≈ 0.07 – 0.14 << 0.50! Hệ thống bị phạt đồng thời cả False Positive và False Negative, kéo mAP gộp tụt xuống 74.40%.'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    p = tf.add_paragraph()
    p.text = '• Bằng chứng minh oan: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
    run = p.add_run()
    run.text = 'Khi áp dụng Giao thức Harmonized PPE (đánh giá riêng lớp mũ "hat"), mAP lập tức nhảy vọt lên 97.03%, chứng minh mô hình học bản chất thị giác cực chuẩn!'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 16: HIỆN THỰC HÓA TRÊN PHẦN CỨNG GIÁ RẺ (EDGE AI)
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout); set_slide_bg(s16)
    add_header(s16, 'Hiện Thực Hóa Phần Cứng: Chạy Mượt Trên GPU Laptop MX230 2GB', 'Tiêu Chí 4 · Tính Khả Thi Triển Khai Thực Tiễn', '16 / 18')

    hw_tiers = [
        ('SERVER / CLOUD GPU', 'NVIDIA Tesla T4', '2.92 ms', '342.5 FPS', 'TensorRT 11.2 FP16 · 640×640\nChạy song song 12 luồng camera.', ORANGE_ACCENT),
        ('LAPTOP GPU CẬN CAO CẤP', 'RTX 3050 Laptop', '5.35 ms', '187.1 FPS', 'TensorRT 11.2 FP16 · 640×640\nTrạm giám sát công trường di động.', BLUE_ACCENT),
        ('LAPTOP VĂN PHÒNG GIÁ RẺ', 'GeForce MX230 (2GB VRAM)', '36.0 ms', '27.8 FPS', 'Native PyTorch FP32 · 640×640\nVƯỢT NGƯỠNG 24 FPS REAL-TIME!', GREEN_ACCENT)
    ]
    for idx, (tier_h, dev_name, lat_val, fps_val, note_txt, col) in enumerate(hw_tiers):
        left_pos = 0.8 + idx * 4.0
        add_card(s16, left_pos, 1.35, 3.8, 2.70, LIGHT_GREEN if idx==2 else CARD_BG, col, border_width=1.8 if idx==2 else 1.0)
        tb = s16.shapes.add_textbox(Inches(left_pos + 0.15), Inches(1.48), Inches(3.5), Inches(2.45))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = tier_h; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = dev_name; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_DARK
        p = tf.add_paragraph(); p.text = f'Độ trễ: {lat_val}'; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = TEXT_MUTED
        p = tf.add_paragraph(); p.text = fps_val; p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = col; p.space_after = Pt(3)
        p = tf.add_paragraph(); p.text = note_txt; p.font.size = Pt(8.8); p.font.color.rgb = TEXT_BODY

    # Bottom Callout: Real-world Value
    add_card(s16, 0.8, 4.30, 11.7, 2.45)
    tb = s16.shapes.add_textbox(Inches(1.05), Inches(4.45), Inches(11.2), Inches(2.20))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = 'Ý NGHĨA KINH TẾ & THỰC TIỄN CỦA KẾT QUẢ GEFORCE MX230 (2GB VRAM)'; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT; p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = '• Bẻ gãy rào cản chi phí: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = TEXT_DARK
    run = p.add_run()
    run.text = 'NVIDIA GeForce MX230 là card đồ họa đời cũ kiến trúc Pascal (2019), không hề có Tensor Core và chỉ có 2GB VRAM hạn hẹp. Đa số các mô hình AI hiện nay đều bị tràn bộ nhớ (CUDA OOM) hoặc giật lag <10 FPS trên phần cứng này.'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    p = tf.add_paragraph()
    p.text = '• Khả thi 100% tại mọi công trường: '
    p.font.size = Pt(9.8); p.font.bold = True; p.font.color.rgb = GREEN_ACCENT
    run = p.add_run()
    run.text = 'Rep-YOLO11s vận hành ổn định ở 27.8 FPS (> 24 FPS chuẩn điện ảnh). Doanh nghiệp chỉ cần tận dụng laptop văn phòng cũ của chỉ huy trưởng để giám sát an toàn, TIẾT KIỆM HÀNG TRĂM TRIỆU ĐỒNG đầu tư server!'
    run.font.bold = False; run.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 17: PIPELINE GIÁM SÁT VIDEO RTSP ĐA LUỒNG (FIGURE 4)
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout); set_slide_bg(s17)
    add_header(s17, 'Hệ Thống Giám Sát Camera RTSP Đa Luồng Toàn Trình', 'Sản Phẩm Prototype · Triển Khai Phần Mềm Giám Sát', '17 / 18')

    # Top: Large RTSP Pipeline Diagram (Fig 4)
    fig4_path = os.path.join(fig_dir, 'Fig4_industrial_rtsp_surveillance_pipeline.png')
    if os.path.exists(fig4_path):
        add_card(s17, 0.8, 1.25, 11.7, 4.30, CARD_BG, BORDER_COLOR)
        s17.shapes.add_picture(fig4_path, Inches(0.95), Inches(1.35), Inches(11.4), Inches(4.10))

    # Bottom 3 Pipeline Stage Metrics
    rtsp_stages = [
        ('GIẢI MÃ PHẦN CỨNG H.264/H.265', 'Đọc luồng RTSP đa luồng độc lập, đệm vòng chống giật khung hình, duy trì độ trễ giải mã < 4.5 ms.', ORANGE_ACCENT),
        ('SUY LUẬN TENSORRT TĂNG TỐC', 'Tiền xử lý Letterbox song song, nạp batch suy luận TensorRT FP16 với độ trễ forward 2.92 ms – 5.35 ms.', BLUE_ACCENT),
        ('BẬT BÁO ĐỘNG & VẼ HUD THỜI GIAN THỰC', 'Hậu xử lý Non-Maximum Suppression (NMS), vẽ bounding box vi phạm và ghi log sự kiện ở tốc độ 65–95 FPS.', GREEN_ACCENT)
    ]
    for idx, (h_txt, desc_txt, col) in enumerate(rtsp_stages):
        left_pos = 0.8 + idx * 4.0
        add_card(s17, left_pos, 5.65, 3.8, 1.45)
        tb = s17.shapes.add_textbox(Inches(left_pos + 0.15), Inches(5.72), Inches(3.5), Inches(1.30))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = h_txt; p.font.size = Pt(10.2); p.font.bold = True; p.font.color.rgb = col; p.space_after = Pt(2)
        p = tf.add_paragraph(); p.text = desc_txt; p.font.size = Pt(9); p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 18: TỔNG KẾT & KẾ HOẠCH GIAI ĐOẠN TIẾP THEO
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout); set_slide_bg(s18)
    add_header(s18, 'Tổng Kết Giai Đoạn 1 & Kế Hoạch Thực Hiện Giai Đoạn 2 - 3', 'Lộ Trình Đồ Án Tốt Nghiệp · Khoa CNTT - Đại Học FPT', '18 / 18')

    phases = [
        ('GIAI ĐOẠN 1 (ĐÃ HOÀN TẤT 90%)', 'ĐÃ HOÀN THÀNH', [
            '✓ Hoàn thiện cơ sở lý thuyết toán học & 4 module tùy biến.',
            '✓ Hoàn thành thực nghiệm A0–A6 & kiểm định 5-Fold (97.11%).',
            '✓ Chứng minh tính khả thi trên GPU phổ thông MX230 (27.8 FPS).',
            '✓ Đã viết hoàn chỉnh bản thảo báo cáo khoa học 9 trang.'
        ], GREEN_ACCENT),

        ('GIAI ĐOẠN 2 (TIẾP THEO)', 'KẾ HOẠCH THỰC HIỆN', [
            '▸ Hoàn thiện giao diện phần mềm Desktop GUI & Web Dashboard.',
            '▸ Tích hợp tính năng quản lý nhiều camera RTSP đồng thời.',
            '▸ Hệ thống cảnh báo âm thanh và gửi email tự động khi có vi phạm.',
            '▸ Thử nghiệm vận hành thực tế tại công trường đối tác.'
        ], BLUE_ACCENT),

        ('GIAI ĐOẠN 3 (BẢO VỆ CHÍNH THỨC)', 'MỤC TIÊU CUỐI KHÓA', [
            '▸ Triển khai Chưng cất Tri thức (Knowledge Distillation YOLO11x -> s).',
            '▸ Nâng cao độ chính xác nhận diện góc nhìn Flycam trên SHEL5K.',
            '▸ Hoàn thiện hồ sơ đồ án tốt nghiệp chính thức.',
            '▸ Nộp công bố bài báo khoa học ra hội nghị / tạp chí quốc tế.'
        ], ORANGE_ACCENT)
    ]

    for idx, (p_title, p_badge, items, col) in enumerate(phases):
        left_pos = 0.8 + idx * 4.0
        add_card(s18, left_pos, 1.35, 3.8, 3.75)
        
        # Header + Badge
        tb = s18.shapes.add_textbox(Inches(left_pos + 0.18), Inches(1.50), Inches(3.44), Inches(0.80))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.text = p_badge; p.font.size = Pt(9); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph(); p.text = p_title; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = TEXT_DARK; p.space_after = Pt(6)

        # Bullets
        tb_b = s18.shapes.add_textbox(Inches(left_pos + 0.18), Inches(2.35), Inches(3.44), Inches(2.65))
        tf_b = tb_b.text_frame; tf_b.word_wrap = True
        tf_b.margin_left = tf_b.margin_right = tf_b.margin_top = tf_b.margin_bottom = 0
        for b_txt in items:
            p = tf_b.add_paragraph()
            p.text = b_txt
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_DARK if '✓' in b_txt else TEXT_BODY
            p.font.bold = True if '✓' in b_txt else False
            p.space_after = Pt(5)

    # Bottom Contact Card & Ready to Defend
    add_card(s18, 0.8, 5.30, 11.7, 1.70, CARD_BG, ORANGE_ACCENT, border_width=1.5)
    tb = s18.shapes.add_textbox(Inches(1.05), Inches(5.45), Inches(11.2), Inches(1.40))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = 'LỜI CẢM ƠN & SẴN SÀNG GIẢI ĐÁP PHẢN BIỆN'
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = ORANGE_ACCENT; p.space_after = Pt(2)
    p = tf.add_paragraph()
    p.text = 'Nhóm nghiên cứu xin chân thành cảm ơn ThS. Vũ Hải Anh đã tận tình hướng dẫn và Quý Thầy Cô Hội đồng đã lắng nghe bài báo cáo!\nNhóm rất mong nhận được những nhận xét, đóng góp quý báu của Quý Thầy Cô để hoàn thiện đề tài trong các giai đoạn tiếp theo.'
    p.font.size = Pt(10.2); p.font.color.rgb = TEXT_BODY; p.space_after = Pt(3)
    p = tf.add_paragraph()
    p.text = 'Sinh viên thuyết trình: Nguyễn Hàn Như (SE183644) · Bộ môn Trí tuệ Nhân tạo — Đại học FPT Hà Nội'
    p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SAVE OUTPUT PRESENTATIONS & EXPORT PDF
    # =========================================================================
    import shutil
    pptx_out1 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep-YOLO11s___Capstone_Review_1_Defense_20260920062556.pptx'
    pptx_out2 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep_YOLO11s_FPT_Defense_Master_Deck.pptx'
    pdf_out1 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep-YOLO11s___Capstone_Review_1_Defense_20260920062625.pdf'
    pdf_out2 = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\Rep_YOLO11s_FPT_Defense_Master_Deck.pdf'
    render_dir = r'c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\slide_renders'
    os.makedirs(render_dir, exist_ok=True)
    
    prs.save(pptx_out1)
    print(f"Successfully saved primary PPTX to: {pptx_out1}")
    try:
        shutil.copyfile(pptx_out1, pptx_out2)
        print(f"Successfully copied secondary PPTX to: {pptx_out2}")
    except Exception as e:
        print(f"Note: Secondary PPTX locked by viewer ({e}), primary is updated.")

    # COM Automation for PDF export
    try:
        import win32com.client
        ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        ppt_app.Visible = 1
        deck = ppt_app.Presentations.Open(os.path.abspath(pptx_out1), WithWindow=False)
        deck.SaveAs(os.path.abspath(pdf_out1), 32)  # 32 = ppSaveAsPDF
        deck.Close()
        ppt_app.Quit()
        print(f"Successfully exported PDF to: {pdf_out1}")
        try:
            shutil.copyfile(pdf_out1, pdf_out2)
        except Exception:
            pass

        # Render to PNGs
        import fitz
        doc = fitz.open(pdf_out1)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            png_p = os.path.join(render_dir, f"slide_{i+1:02d}.png")
            pix.save(png_p)
        print(f"Rendered {len(doc)} slides to PNGs in: {render_dir}")
    except Exception as e:
        print(f"PDF export / render warning: {e}")

if __name__ == '__main__':
    build_deck()
