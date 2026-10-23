"""
Script to generate the Vietnamese version of the Research Proposal document:
'NTD_Cong nghe thong tin_TiengViet.docx' and 'NTD_Cong nghe thong tin_TiengViet.pdf'
Faithfully translating all sections of 'NTD_Cong nghe thong tin.docx'.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import win32com.client
import pythoncom

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="003366", sz="6", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="none"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def format_run(run, font_name="Times New Roman", size_pt=11, bold=False, italic=False, color_rgb=(0,0,0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_p(doc, text="", font_name="Times New Roman", size_pt=11, bold=False, italic=False, 
          align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6, line_spacing=1.15):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        format_run(run, font_name, size_pt, bold, italic)
    return p

def add_heading_styled(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if level == 1:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        format_run(run, "Times New Roman", 13, bold=True, color_rgb=(0, 51, 102))
    elif level == 2:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        format_run(run, "Times New Roman", 12, bold=True, color_rgb=(0, 0, 0))
    elif level == 3:
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        format_run(run, "Times New Roman", 11, bold=True, italic=True, color_rgb=(50, 50, 50))
    return p

def convert_docx_to_pdf(docx_path, pdf_path):
    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.ExportAsFixedFormat(os.path.abspath(pdf_path), 17) # 17 = wdExportFormatPDF
        doc.Close()
        print(f"Successfully converted to PDF: {pdf_path}")
    except Exception as e:
        print(f"Error during Word export: {e}")
    finally:
        try:
            word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()

def main():
    doc = docx.Document()
    
    # Page setup: A4, 1.0 inch margins all around
    for sec in doc.sections:
        sec.page_width = Inches(8.27)
        sec.page_height = Inches(11.69)
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        
    # Title / Header
    p_top = add_p(doc, "ĐỀ XUẤT NGHIÊN CỨU KHOA HỌC (BẢN TIẾNG VIỆT)", size_pt=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=2)
    p_top.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    p_sub = add_p(doc, "CUỘC THI NGHIÊN CỨU KHOA HỌC SINH VIÊN ĐẠI HỌC FPT (FA26)", size_pt=11, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)
    p_sub.runs[0].font.color.rgb = RGBColor(100, 100, 100)
    
    # Metadata Table (Table 0)
    tbl0 = doc.add_table(rows=5, cols=2)
    tbl0.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl0, color="003366", sz="8")
    
    col_widths0 = [Inches(2.1), Inches(4.17)]
    for row in tbl0.rows:
        for i, w in enumerate(col_widths0):
            row.cells[i].width = w
            set_cell_margins(row.cells[i], top=100, bottom=100, left=140, right=140)
            row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    meta_items = [
        ("Tên đề tài\n(Tiếng Anh & Tiếng Việt)", (
            "Tiếng Anh: Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance\n\n"
            "Tiếng Việt: Tái tham số hóa cấu trúc, mã hóa tọa độ không gian và độ bền vững liên miền phục vụ phát hiện mũ bảo hộ thời gian thực trong camera giám sát công trường xây dựng"
        )),
        ("Tiểu ban", "Công nghệ thông tin (Information Technology)"),
        ("Tên nhóm", "NTD"),
        ("Tác giả", (
            "1. Nguyễn Hàn Như - MSSV: SE183644\n"
            "   Email: nhunhse183644@fpt.edu.vn (Trưởng nhóm / Leader)\n"
            "2. Nguyễn Văn Thành - MSSV: SE180387\n"
            "   Email: thanhnvSE180387@fpt.edu.vn (Thành viên / Member)\n"
            "3. Trần Phạm Tuấn Dũng - MSSV: SE180382\n"
            "   Email: dungtptse180382@fpt.edu.vn (Thành viên / Member)"
        )),
        ("Giảng viên hướng dẫn", "Vũ Hà Anh\nEmail: AnhVH54@fe.edu.vn\nKhoa Trí tuệ Nhân tạo, Trường Đại học FPT TP.HCM")
    ]
    
    for idx, (label, val) in enumerate(meta_items):
        cell_lbl = tbl0.rows[idx].cells[0]
        cell_val = tbl0.rows[idx].cells[1]
        set_cell_background(cell_lbl, "F0F4F8")
        
        p_l = cell_lbl.paragraphs[0]
        p_l.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run_l = p_l.add_run(label)
        format_run(run_l, size_pt=10.5, bold=True, color_rgb=(0, 51, 102))
        
        p_v = cell_val.paragraphs[0]
        p_v.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run_v = p_v.add_run(val)
        format_run(run_v, size_pt=10.5, bold=(idx == 0 or idx == 1 or idx == 2))

    add_p(doc, "", space_before=6, space_after=6)

    # ABSTRACT
    add_heading_styled(doc, "Tóm tắt (Abstract)", level=1)
    abstract_text = (
        "Việc giám sát tự động tuân thủ Trang thiết bị Bảo hộ Lao động (PPE) qua camera an ninh là yếu tố sống còn nhằm duy trì "
        "an toàn lao động tại các công trường xây dựng công nghiệp. Tuy nhiên, các hệ thống camera giám sát trên thiết bị biên đối mặt "
        "với những nút thắt hoạt động nghiêm trọng, bao gồm: mục tiêu mũ bảo hộ siêu nhỏ ở khoảng cách xa, mất cân bằng nhãn dữ liệu "
        "nghiêm trọng giữa công nhân và trang bị an toàn, mật độ che khuất vật lý dày đặc, và nhiễu nền phức tạp. Để vượt qua các giới hạn này, "
        "chúng tôi đề xuất Rep-YOLO11s, một kiến trúc dò tìm đối tượng trên thiết bị biên hiệu năng cao được thiết kế tối ưu phần cứng dành "
        "riêng cho các thiết bị biên bị giới hạn bộ nhớ. Được phát triển từ mô hình nền tảng Ultralytics YOLO11s, kiến trúc của chúng tôi "
        "giải quyết nút thắt độ trễ luồng đơn bằng cách tích hợp kỹ thuật tái tham số hóa cấu trúc nhằm gập (fold) các nhánh tích chập huấn luyện "
        "song song thành một phép tích chập 3x3 duy nhất trong giai đoạn triển khai, giúp cắt giảm tối đa chi phí truy xuất bộ nhớ DRAM (MAC). "
        "Để triệt tiêu cảnh báo giả từ rác thải mặt đất, tích chập tọa độ (CoordConv) ở tầng Stem cung cấp các thông tin tiên nghiệm không gian "
        "theo phương đứng. Ngoài ra, cơ chế chú ý định tuyến hai cấp (Bi-Level Routing Attention) được tích hợp vào các tầng cổ mạng (neck) để "
        "lọc bỏ linh hoạt các token nền và dồn tài nguyên tính toán vào các mục tiêu siêu nhỏ. Quá trình huấn luyện mô hình được ổn định hóa "
        "bằng kỹ thuật giám sát đa tác vụ tách rời kết hợp hàm mất mát hồi quy Focal EIoU với thuật toán gán mẫu theo căn chỉnh tác vụ (TAL) "
        "nhằm xử lý bài toán lệch phân phối dữ liệu nghiêm trọng. Các đánh giá toàn diện trên bộ dữ liệu Safety Helmet Wearing Dataset (SHWD) "
        "và bộ dữ liệu kiểm chứng độc lập Hard Hat Workers (HHW) chứng minh rằng Rep-YOLO11s đạt độ chính xác phát hiện vượt trội và độ bền vững "
        "liên miền mạnh mẽ, đồng thời duy trì thông lượng xử lý thời gian thực rất cao trên các dòng GPU biên thông dụng và các nền tảng di động "
        "bị giới hạn công suất. Đường ống giám sát toàn diện từ đầu đến cuối đáp ứng hoàn hảo các yêu cầu xử lý đa luồng video đồng thời, "
        "mang lại một giải pháp thực tiễn, độ chính xác cao cho công tác kiểm soát an toàn xây dựng tự động."
    )
    add_p(doc, abstract_text, italic=True)
    
    # Keywords
    p_kw = add_p(doc, "Từ khóa: ", bold=True, space_before=2, space_after=10)
    run_kw = p_kw.add_run("Phát hiện mũ bảo hộ (Safety Helmet Detection); Tái tham số hóa cấu trúc (Structural Re-Parameterization); Tích chập tọa độ (CoordConv); Cơ chế chú ý định tuyến hai cấp (Bi-Level Routing Attention); Trí tuệ nhân tạo biên thời gian thực (Real-Time Edge AI); Khả năng tổng quát hóa liên miền (Cross-Domain Generalization).")
    format_run(run_kw, bold=False, italic=False)

    # SECTION 1
    add_heading_styled(doc, "1. Giới thiệu (Introduction)", level=1)
    
    # 1.1 Literature Review
    add_heading_styled(doc, "1.1. Tổng quan tài liệu nghiên cứu (Literature Review)", level=2)
    lr_p1 = (
        "Đảm bảo an toàn lao động tại các đại công trường xây dựng công nghiệp hiện đại là vấn đề xã hội và kinh tế cấp bách. "
        "Theo các báo cáo lao động quốc tế, các chấn thương sọ não do vật thể rơi từ trên cao và té ngã chiếm tỷ lệ cao nhất trong "
        "các vụ tai nạn tử vong tại nơi làm việc. Mặc dù việc đội mũ bảo hộ đạt chuẩn giúp giảm thiểu đáng kể nguy cơ chấn thương não, "
        "việc kiểm tra thủ công bằng mắt thường của giám sát viên trên các công trường nhiều tầng, quy mô rộng lớn là bất khả thi, "
        "tốn kém nhân lực và thiếu liên tục. Do đó, thị giác máy tính dựa trên học sâu (deep learning) đã trở thành hướng tiếp cận tối ưu nhất "
        "để tự động hóa việc kiểm tra tuân thủ PPE."
    )
    add_p(doc, lr_p1)
    
    lr_p2 = (
        "Các nghiên cứu của các nhà khoa học trong và ngoài nước đã phát triển qua các giai đoạn từ mô hình dò tìm hai giai đoạn "
        "(two-stage detectors như Faster R-CNN) đến các kiến trúc một giai đoạn (single-stage detectors như SSD, YOLOv5, YOLOv8 và YOLO11) [1]–[4]. "
        "Mặc dù các mô hình hai giai đoạn đạt độ định vị tốt, độ phức tạp tính toán bậc hai và việc sinh vùng đề xuất dày đặc dẫn đến độ trễ rất cao "
        "(>80 ms/khung hình), hoàn toàn không thể triển khai trên thiết bị biên. Ngược lại, các mô hình nhẹ một giai đoạn lại thường bị suy giảm độ chính xác "
        "nghiêm trọng khi hoạt động trên công trường thực tế do 4 nút thắt: (1) mục tiêu mũ bảo hiểm ở xa chỉ chiếm kích thước siêu nhỏ (<16×16 pixel); "
        "(2) sự mất cân bằng dữ liệu nghiêm trọng giữa thân người và mũ bảo hộ (tỷ lệ 12 thân người chỉ có 1 mũ trong dữ liệu giám sát); "
        "(3) sự che khuất vật lý dày đặc bởi giàn giáo và máy móc; và (4) cảnh báo giả do rác thải mặt đất (thùng sơn vàng, cọc tiêu giao thông, "
        "mảnh vụn bê tông) [5], [6]."
    )
    add_p(doc, lr_p2)

    lr_p3 = (
        "Để cải thiện các mục tiêu nhỏ, các nghiên cứu gần đây thường bổ sung các nhánh phụ, mạng kim tự tháp đặc trưng đa tỷ lệ (PANet/FPN) [9], [10] "
        "hoặc các cơ chế tự chú ý (Vision Transformers). Tuy nhiên, cơ chế tự chú ý dày đặc (dense self-attention) có độ phức tạp tăng theo hàm bậc hai "
        "O(N^2), gây tràn bộ nhớ SRAM trên chip biên. Hơn nữa, việc chia nhiều nhánh tích chập song song làm tăng vọt chi phí truy xuất bộ nhớ DRAM "
        "(Memory Access Cost - MAC), gây nghẽn quá trình khởi chạy kernel GPU khi xử lý luồng video tuần tự (single-batch streaming). "
        "Kỹ thuật tái tham số hóa cấu trúc, tiên phong bởi RepVGG (Ding et al. [2]) và được YOLOv6 [7] ứng dụng, đã chứng minh rằng cấu trúc nhiều nhánh "
        "phức tạp lúc huấn luyện có thể gập đại số thành một nhánh tích chập duy nhất lúc triển khai, loại bỏ bộ đệm thừa. Đồng thời, kỹ thuật Tích chập Tọa độ "
        "(CoordConv của Liu et al. [3]) bổ sung thông tin tọa độ không gian để phá vỡ tính bất biến tịnh tiến của tích chập truyền thống. "
        "Cơ chế chú ý định tuyến hai cấp (BiFormer của Zhu et al. [4]) cho phép định tuyến linh hoạt để bỏ qua các vùng hậu cảnh. Xuất phát từ thực tế rằng "
        "các nghiên cứu trước đây chỉ tối ưu tốc độ dựa trên FLOPs lý thuyết mà bỏ quên giới hạn băng thông bộ nhớ và công suất vật lý thật, nghiên cứu "
        "của chúng tôi tập trung: tích hợp có chọn lọc và tối ưu hóa phần cứng các cơ chế RepConv, CoordConv và BiFormer trên nền tảng YOLO11s [1] "
        "để tạo ra một bộ dò biên chuyên dụng."
    )
    add_p(doc, lr_p3)

    # 1.2 The Necessity of the Research
    add_heading_styled(doc, "1.2. Tính cần thiết của nghiên cứu (The Necessity of the Research)", level=2)
    nec_p1 = (
        "Tại Việt Nam, tiến trình công nghiệp hóa và hiện đại hóa xây dựng đang diễn ra mạnh mẽ, đi kèm với các quy chuẩn an toàn lao động ngày càng khắt khe "
        "(Thông tư 10/2020/TT-BXD và các nghị định an toàn lao động). Tuy nhiên, các hệ thống camera hiện nay tại công trường hầu hết chỉ ghi hình thụ động "
        "mà không có khả năng phát hiện vi phạm theo thời gian thực. Khi thử nghiệm các giải pháp AI tự động, đa số đều phụ thuộc vào máy chủ đám mây (cloud computing). "
        "Việc truyền tải đồng thời nhiều luồng video 4K/1080p lên cloud làm tốn kém băng thông viễn thông khủng khiếp, rủi ro rò rỉ dữ liệu riêng tư và độ trễ "
        "đường truyền lớn (>200–500 ms), khiến không thể kích hoạt chuông cảnh báo kịp thời khi công nhân vừa tháo mũ bước vào vùng nguy hiểm."
    )
    add_p(doc, nec_p1)

    nec_p2 = (
        "Vì vậy, việc nghiên cứu một bộ dò AI biên có khả năng chạy độc lập hoàn toàn tại chỗ (on-premises edge nodes) với tốc độ khung hình cao (>150 FPS), "
        "tiêu thụ điện năng thấp (<15W) là một đòi hỏi khoa học và thực tiễn bức thiết. Tính mới và giá trị khoa học của đề tài nằm ở việc tích hợp một cách "
        "có kỷ luật kỹ thuật tái tham số hóa cấu trúc, tiên nghiệm tọa độ đứng và cơ chế chú ý định tuyến linh hoạt trên nền YOLO11s, mang lại một giải pháp "
        "vừa có độ chính xác cao, vừa chạy siêu nhanh trên phần cứng phổ thông và có khả năng nhân rộng thương mại tại Việt Nam."
    )
    add_p(doc, nec_p2)

    # 1.3 Feasibility of Research
    add_heading_styled(doc, "1.3. Tính khả thi của nghiên cứu (Feasibility of Research)", level=2)
    feas_intro = (
        "Dự án đề xuất thể hiện tính khả thi khoa học vững chắc trên các phương diện nền tảng toán học, nguồn dữ liệu thực nghiệm, hạ tầng phần cứng "
        "và phương pháp thực thi cụ thể như sau:"
    )
    add_p(doc, feas_intro)

    feas_points = [
        ("Câu hỏi nghiên cứu (Research Question)", "Liệu việc tích hợp tái tham số hóa cấu trúc (RepConv), mã hóa tọa độ nhận thức độ cao (CoordConv) và định tuyến chú ý thưa linh hoạt (BiFormer) vào YOLO11s có thể đồng thời nâng cao độ chính xác phát hiện mũ bảo hộ nhỏ và duy trì độ trễ TensorRT dưới 5ms trên phần cứng biên tiết kiệm điện hay không?"),
        ("Thiết kế nghiên cứu (Research Design)", "Phương pháp thực nghiệm deep learning chuẩn mực: (1) Biến đổi đại số gập trọng số tích chập và nhúng tọa độ; (2) Thực nghiệm triệt tiêu thành phần (ablation study) đánh giá đóng góp từng cấu phần; (3) Kiểm định chéo phân tầng theo cảnh quay (scene-stratified 5-fold CV); (4) Thử nghiệm liên miền không tinh chỉnh (zero-shot transfer) trên công trường lạ; (5) Đo kiểm trực tiếp trên 4 tầng phần cứng vật lý thật (GPU máy chủ Tesla T4, GPU laptop RTX 3050 kèm profile 15W, GPU MX230 và CPU)."),
        ("Mẫu dữ liệu & Quần thể (Sample Size and Population)", "Quần thể mục tiêu là công nhân và mũ bảo hộ dưới nhiều tư thế, điều kiện ánh sáng và khoảng cách. Mẫu dữ liệu gồm 2 bộ chuẩn: (i) Safety Helmet Wearing Dataset (SHWD) gồm 7,581 ảnh phân giải cao, 9,044 mũ bảo hộ và 111,514 nhãn người; và (ii) Hard Hat Workers (HHW) gồm 7,000 ảnh và 26,989 hộp bao công nhân."),
        ("Nguồn dữ liệu (Data Sources)", "Cả SHWD [12] và HHW [8] đều là các bộ dữ liệu thị giác máy tính công khai, có giấy phép mở. Mọi chú thích đã được làm sạch (loại bỏ hộp ngược tọa độ, hộp diện tích 0) và chuẩn hóa theo hệ tọa độ VOC/COCO, loại bỏ hoàn toàn các rủi ro pháp lý."),
        ("Ngân sách & Kinh phí (Budget and Funding)", "Mọi tài nguyên tính toán đều được tự trang bị nội bộ. Huấn luyện và kiểm chứng sử dụng máy trạm nội bộ (NVIDIA RTX 3050 Laptop GPU, 4GB VRAM) kết hợp GPU Dual Tesla T4 thông qua nền tảng Kaggle. Không phát sinh chi phí thương mại hay mua bản quyền bên ngoài."),
        ("Kế hoạch thời gian (Timeframe)", "Lộ trình nghiên cứu kéo dài 12 tuần (01/10/2026 – 20/12/2026), bám sát tuyệt đối các mốc tiến độ Cuộc thi NCKH FA26 của Đại học FPT."),
        ("Thách thức tiềm ẩn & Giải pháp (Potential Challenges and Risks)", "(a) Tràn bộ nhớ GPU biên: giải quyết bằng tối ưu hóa bộ nhớ TensorRT FP16 và gập nhánh RepConv; (b) Báo động giả dưới mặt đất: triệt tiêu bằng kênh tọa độ CoordConv; (c) Thiếu gradient cho mục tiêu nhỏ do lệch dữ liệu 1:12: giải quyết bằng Focal EIoU loss và Task-Aligned Assigner.")
    ]
    for lbl, desc in feas_points:
        p_pt = add_p(doc, f"• {lbl}: ", bold=True, space_before=2, space_after=4)
        run_desc = p_pt.add_run(desc)
        format_run(run_desc, bold=False)

    # SECTION 2
    add_heading_styled(doc, "2. Mục tiêu nghiên cứu (Research Objectives)", level=1)
    obj_p1 = (
        "Mục tiêu hàng đầu của dự án là phát triển, kiểm chứng và đo kiểm phần cứng mô hình Rep-YOLO11s—một bộ dò kết hợp tích chập và chú ý "
        "tối ưu phần cứng cho bài toán kiểm soát mũ bảo hộ thời gian thực trên camera giám sát biên, xây dựng dựa trên kiến trúc Ultralytics YOLO11s [1]."
    )
    add_p(doc, obj_p1)
    
    obj_specific = [
        "Mục tiêu 1 (Tích hợp Tái tham số hóa cấu trúc): Tích hợp các khối RepConv vào đường dẫn tổng hợp đặc trưng ở cổ mạng (neck), gập các nhánh tích chập song song lúc huấn luyện thành một nhân 3x3 lúc chạy suy luận, giúp cắt giảm 66.3% lưu lượng truy xuất bộ nhớ DRAM (-0.50 ms thời gian chạy kernel).",
        "Mục tiêu 2 (Tiên nghiệm không gian độ cao): Bổ sung 2 kênh tọa độ CoordConv vào tầng Stem đầu tiên của mạng, cung cấp tọa độ độ cao tuyệt đối để triệt tiêu cảnh báo giả từ rác thải mặt đất mà không làm tăng chi phí tính toán ở các tầng sâu.",
        "Mục tiêu 3 (Định tuyến chú ý thưa linh hoạt): Tích hợp cơ chế chú ý định tuyến hai cấp (BiFormer) vào các tầng P4/P5 ở cổ mạng để lọc bỏ các token nền không cần thiết và dồn sự chú ý vào các mục tiêu mũ siêu nhỏ ở xa.",
        "Mục tiêu 4 (Giám sát đa tác vụ tách rời): Kết hợp hàm mất mát hồi quy Focal EIoU với thuật toán gán mẫu Task-Aligned Assignment (TAL) để loại bỏ hiện tượng thống trị gradient do mất cân bằng tỷ lệ 1:12 giữa mũ và người.",
        "Mục tiêu 5 (Thực nghiệm & Đo kiểm đa phần cứng): Đạt độ chính xác hàng đầu trên SHWD (94.83% mAP50, 62.54% mAP50-95; 96.64% ± 0.32% trên 5-fold CV), chứng minh khả năng tổng quát hóa liên miền trên HHW (97.03% hat-only mAP50), đạt tốc độ 228.6 FPS (TensorRT FP16) trên RTX 3050 và duy trì 182.4 FPS ở mức giới hạn điện năng 15W."
    ]
    for item in obj_specific:
        p_item = add_p(doc, f"• {item}", space_before=2, space_after=3)

    # SECTION 3
    add_heading_styled(doc, "3. Phạm vi nghiên cứu (Research Scope)", level=1)
    scope_p1 = (
        "Phạm vi của nghiên cứu này được xác định rõ ràng trên 3 phương diện hoạt động chính:"
    )
    add_p(doc, scope_p1)
    
    scope_items = [
        ("Phạm vi bài toán (Problem Domain)", "Giám sát tự động việc tuân thủ PPE, tập trung cụ thể vào mũ bảo hộ lao động và đầu/thân người công nhân ghi nhận qua camera an ninh cố định hoặc camera PTZ tại các công trường xây dựng."),
        ("Phạm vi thuật toán (Algorithmic Scope)", "Mô hình thị giác học sâu phát triển dựa trên nền tảng Ultralytics YOLO11s [1]. Việc cải tiến kiến trúc tập trung vào: Tích chập Stem CoordConv [3], Khối Neck RepConv [2], Cơ chế Neck BiFormer Attention [4], và Hàm mất mát Focal EIoU [5] kết hợp TAL [6]."),
        ("Phạm vi phần cứng triển khai (Deployment Target Scope)", "Các thiết bị phần cứng biên thông dụng, cụ thể là GPU biên (NVIDIA RTX 3050 Laptop, Tesla T4, MX230) và các nền tảng di động chạy pin (chế độ tiết kiệm điện 15W), sử dụng công cụ tối ưu hóa TensorRT FP16.")
    ]
    for lbl, desc in scope_items:
        p_sc = add_p(doc, f"• {lbl}: ", bold=True, space_before=2, space_after=4)
        run_desc = p_sc.add_run(desc)
        format_run(run_desc, bold=False)

    # SECTION 4
    add_heading_styled(doc, "4. Cách tiếp cận và Phương pháp nghiên cứu (Approach and Method)", level=1)
    app_p1 = (
        "Để đồng thời đạt được độ chính xác phát hiện cao và tốc độ suy luận cực nhanh, Rep-YOLO11s triển khai một đường ống kiến trúc 4 giai đoạn gắn kết chặt chẽ:"
    )
    add_p(doc, app_p1)

    app_stages = [
        ("Giai đoạn 1: Tích chập tọa độ tầng Stem (CoordConv [3])", 
         "Tích chập 2D thông thường có tính bất biến tịnh tiến nên không phân biệt được vị trí trên/dưới. Chúng tôi bổ sung 2 ma trận tọa độ chuẩn hóa Cx, Cy in [-1, 1] vào ảnh đầu vào, tạo thành tensor 5 kênh đưa vào Stem (c1=5, c2=64, k=3, s=2). Mô hình học được trọng số phụ thuộc vào độ cao, loại bỏ triệt để cảnh báo giả từ các vật màu vàng ở chân công trình với chi phí trễ cực nhỏ (+0.06 ms)."),
        
        ("Giai đoạn 2: Tái tham số hóa cấu trúc (RepConv [2])", 
         "Khi huấn luyện, cấu trúc 3 nhánh (3x3 conv, 1x1 conv, nhánh đồng nhất identity) tạo ra sự đa dạng gradient phong phú. Khi kết thúc huấn luyện, toàn bộ trọng số của các nhánh và Batch Normalization được gập đại số thành một ma trận trọng số 3x3 và một vector bias duy nhất: W_fused = W'_3x3 + Pad_3x3(W'_1x1) + W'_id. Mạng trở về dạng đơn luồng thẳng tắp, giảm dung lượng truy xuất DRAM từ ~9.8 MB xuống ~3.3 MB (-66.3%) và gập 3 lần gọi kernel (1.28 ms) thành 1 lần gọi (0.78 ms, tiết kiệm 0.50 ms)."),
        
        ("Giai đoạn 3: Cơ chế chú ý định tuyến hai cấp (BiFormer [4])", 
         "Khung hình camera giám sát chứa 80%–90% là cảnh nền không liên quan. Chúng tôi chia bản đồ đặc trưng P4/P5 thành các cụm S x S vùng (S = 8), tính ma trận tương đồng vùng và định tuyến truy vấn chỉ đến k=4 vùng tiềm năng nhất. Phép tự chú ý chi tiết chỉ diễn ra trong các vùng được chọn, giảm độ phức tạp xuống tuyến tính O(HW) và phát hiện rõ nét các mũ bảo hộ ở rất xa mà độ trễ chỉ tăng nhẹ +0.24 ms."),
        
        ("Giai đoạn 4: Giám sát đa tác vụ tách rời (Focal EIoU [5] + TAL [6])", 
         "Để khắc phục nhược điểm triệt tiêu gradient của hàm CIoU/Wise-IoU khi hộp dự đoán và nhãn có cùng tỷ lệ khung hình, chúng tôi áp dụng Focal EIoU Loss phân rã độc lập sai số độ trùng lặp, khoảng cách tâm, chiều rộng và chiều cao: L_EIoU = L_IoU + L_center + L_width + L_height. Đồng thời, thuật toán gán mẫu TAL tính điểm căn chỉnh để chủ động gán các neo chất lượng cao cho mục tiêu mũ bảo hộ ít ỏi giữa biển thân người.")
    ]
    for lbl, desc in app_stages:
        p_stg = add_p(doc, f"• {lbl}: ", bold=True, space_before=3, space_after=4)
        run_desc = p_stg.add_run(desc)
        format_run(run_desc, bold=False)

    app_eval = (
        "Quy trình đánh giá: Mô hình được kiểm chứng thông qua kiểm định chéo 5-fold CV phân tầng theo cảnh trên SHWD để đảm bảo tính vững chắc "
        "về mặt thống kê. Khả năng tổng quát hóa liên miền (zero-shot transfer) được đánh giá trên tập HHW (7,000 ảnh) [8] hoàn toàn không tinh chỉnh "
        "(fine-tuning). Độ trễ, thông lượng (FPS), công suất tiêu thụ và hiệu suất năng lượng (FPS/Watt) được đo đạc trực tiếp trên các tầng phần cứng "
        "vật lý bằng các sự kiện đồng bộ CUDA dưới TensorRT 11.2 FP16 và PyTorch 2.5.1."
    )
    add_p(doc, app_eval, space_before=4, space_after=6)

    # SECTION 5
    add_heading_styled(doc, "5. Kế hoạch nghiên cứu (Research Plan)", level=1)
    plan_intro = (
        "Kế hoạch thực hiện dự án được tổ chức thành 5 mốc tiến độ chính kéo dài từ 01/10/2026 đến 20/12/2026, phù hợp hoàn hảo với lịch trình của cuộc thi:"
    )
    add_p(doc, plan_intro)

    # Table 1: Research Plan
    tbl1 = doc.add_table(rows=6, cols=5)
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl1, color="003366", sz="6")
    
    col_widths1 = [Inches(0.5), Inches(1.4), Inches(2.2), Inches(1.4), Inches(1.3)]
    for row in tbl1.rows:
        for i, w in enumerate(col_widths1):
            row.cells[i].width = w
            set_cell_margins(row.cells[i], top=80, bottom=80, left=100, right=100)
            row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    headers1 = ["STT", "Thời gian", "Nhiệm vụ", "Sản phẩm đầu ra", "Người phụ trách"]
    for i, h in enumerate(headers1):
        cell = tbl1.rows[0].cells[i]
        set_cell_background(cell, "003366")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        format_run(run, size_pt=10, bold=True, color_rgb=(255, 255, 255))

    plan_rows = [
        ("1", "01/10/2026 –\n20/10/2026", 
         "Khảo cứu tài liệu, làm sạch và chuẩn hóa dữ liệu (SHWD & HHW), đo benchmark baseline YOLO11s, xây dựng bản đề xuất nghiên cứu.", 
         "Bản đề cương được duyệt; Bộ dữ liệu chuẩn hóa.", 
         "Nguyễn Hàn Như\n(Leader)"),
        ("2", "21/10/2026 –\n05/11/2026", 
         "Triển khai mã nguồn gập trọng số RepConv, Stem CoordConv, Neck BiFormer Attention, và hàm mất mát Focal EIoU.", 
         "Kho mã nguồn PyTorch Rep-YOLO11s & Unit tests.", 
         "Nguyễn Hàn Như,\nNguyễn Văn Thành"),
        ("3", "06/11/2026 –\n20/11/2026", 
         "Huấn luyện mô hình đa seed, kiểm định chéo 5-fold CV, thực nghiệm triệt tiêu thành phần (A0–A6), kiểm định thống kê.", 
         "Checkpoints mô hình; Toàn bộ log thực nghiệm và số liệu.", 
         "Nguyễn Hàn Như,\nTrần Phạm Tuấn Dũng"),
        ("4", "21/11/2026 –\n07/12/2026", 
         "Thử nghiệm liên miền trên HHW, đo đạc phần cứng vật lý (TensorRT, profile giới hạn 15W), viết toàn văn bài báo 10 trang.", 
         "Toàn văn bài báo nghiên cứu (chuẩn IEEE); Engine TensorRT.", 
         "Cả nhóm"),
        ("5", "08/12/2026 –\n20/12/2026", 
         "Chuẩn bị slide thuyết trình, quay video demo đa luồng trực tiếp, tổng duyệt chuẩn bị bảo vệ Chung kết.", 
         "Bộ Slide thuyết trình; Video Demo; Sẵn sàng bảo vệ.", 
         "Cả nhóm")
    ]

    for r_idx, r_data in enumerate(plan_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = tbl1.rows[r_idx].cells[c_idx]
            if r_idx % 2 == 1:
                set_cell_background(cell, "F9FBFD")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 1] else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            format_run(run, size_pt=9.5, bold=(c_idx == 0))

    add_p(doc, "", space_before=6, space_after=6)

    # SECTION 6
    add_heading_styled(doc, "6. Kết quả dự kiến (Expected Results)", level=1)
    res_intro = (
        "Dự án sẽ mang lại các kết quả có giá trị khoa học, công nghệ và ứng dụng thực tiễn cao:"
    )
    add_p(doc, res_intro)

    results_items = [
        ("Đóng góp về mặt phương pháp & kiến trúc", 
         "Cung cấp một bộ dò biên được thiết kế tối ưu phần cứng, chứng minh rằng việc tái tham số hóa cấu trúc, nhúng tọa độ không gian và định tuyến chú ý thưa có thể triệt tiêu nghẽn băng thông bộ nhớ mà không làm giảm năng lực biểu diễn sâu. Các thực nghiệm triệt tiêu thành phần chứng minh rõ nét đóng góp của từng cấu phần."),
        ("Thành tựu thực nghiệm", 
         "Đạt độ chính xác hàng đầu trên SHWD: 94.83% mAP50 và 62.54% mAP50-95 (đạt 96.64% ± 0.32% trên 5-fold CV). Với riêng lớp mũ bảo hộ (hat), độ chính xác đạt 96.10% AP50 và 64.80% AP50-95 (+4.70% so với baseline). Chứng minh khả năng tổng quát hóa liên miền vượt trội trên HHW (97.03% hat-only mAP50, cao hơn baseline +2.91%)."),
        ("Khả năng triển khai biên thực tế & Tiết kiệm năng lượng", 
         "Đạt tốc độ cực nhanh trên GPU biên thông dụng (RTX 3050 Laptop): 228.6 FPS (4.37 ms) trên TensorRT FP16. Dưới mức giới hạn công suất di động 15W, hệ thống duy trì 182.4 FPS (hiệu suất 12.16 FPS/Watt), cho phép lắp đặt trên các thiết bị giám sát di động chạy pin và xử lý mượt mà đa luồng camera RTSP thời gian thực (65–95 FPS)."),
        ("Công bố khoa học & Chia sẻ mã nguồn", 
         "Hoàn thành bài báo toàn văn gửi Hội nghị quốc tế IEEE AAIML 2027 và báo cáo 10 trang Cuộc thi NCKH FA26. Công khai toàn bộ mã nguồn PyTorch, trọng số tiền huấn luyện và các script sinh engine TensorRT phục vụ tái lập nghiên cứu.")
    ]
    for lbl, desc in results_items:
        p_res = add_p(doc, f"• {lbl}: ", bold=True, space_before=2, space_after=4)
        run_desc = p_res.add_run(desc)
        format_run(run_desc, bold=False)

    # REFERENCES
    add_heading_styled(doc, "Tài liệu tham khảo (References)", level=1)
    refs = [
        "[1] G. Jocher and J. Qiu, \"Ultralytics YOLO11 documentation and architecture,\" 2024. [Online]. Available: https://docs.ultralytics.com/models/yolo11",
        "[2] X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun, \"RepVGG: Making VGG-style ConvNets great again,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 13733–13742.",
        "[3] R. Liu, J. Lehman, P. Molino, F. P. Such, E. Frank, A. Sergeev, and J. Yosinski, \"An intriguing failing of convolutional neural networks and the CoordConv solution,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 31, 2018, pp. 9605–9616.",
        "[4] L. Zhu, X. Wang, Z. Ke, W. Zhang, and R. W. H. Lau, \"BiFormer: Vision transformer with bi-level routing attention,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 10323–10333.",
        "[5] Y.-F. Zhang, W. Ren, Z. Zhang, Z. Jia, L. Wang, and T. Tan, \"Focal and efficient IOU loss for accurate bounding box regression,\" Neurocomputing, vol. 506, pp. 146–157, 2022.",
        "[6] C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, \"TOOD: Task-aligned one-stage object detection,\" in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 3490–3499.",
        "[7] C. Li, L. Li, Y. Geng, H. Jiang, M. Cheng, B. Zhang, Z. Ke, X. Xu, and X. Chu, \"YOLOv6 v3.0: A full-scale reloading,\" arXiv preprint arXiv:2301.05586, 2023.",
        "[8] Andrew MVD, \"Hard hat workers dataset,\" 2021. [Online]. Available: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection",
        "[9] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, \"Feature pyramid networks for object detection,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2117–2125.",
        "[10] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, \"Path aggregation network for instance segmentation,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 8759–8768.",
        "[11] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, \"Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 21002–21012.",
        "[12] NJVisionPower, \"Safety helmet wearing dataset (SHWD),\" 2020. [Online]. Available: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset"
    ]
    for r in refs:
        p_ref = add_p(doc, r, size_pt=9.5, space_before=1, space_after=3)

    # COMMITMENT
    add_p(doc, "", space_before=10, space_after=10)
    add_heading_styled(doc, "BẢN CAM KẾT (COMMITMENT)", level=1)
    p_com_title = add_p(doc, "VỀ SỬ DỤNG TRÍ TUỆ NHÂN TẠO CÓ TRÁCH NHIỆM TRONG NGHIÊN CỨU", size_pt=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=10)
    p_com_title.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    p_com1 = add_p(doc, (
        "Chúng tôi là nhóm tác giả của đề tài nghiên cứu mang tên “Structural Re-Parameterization, Spatial Coordinate Encoding, "
        "and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance”, cam kết rằng tất cả "
        "các thành viên tuân thủ nghiêm túc các quy định và hướng dẫn đạo đức liên quan đến Trí tuệ nhân tạo (AI) trong thực hành "
        "nghiên cứu như sau:"
    ))
    
    com_bullets = [
        "AI chỉ được sử dụng như một công cụ hỗ trợ sửa lỗi ngữ pháp, cấu trúc câu và diễn đạt ngôn ngữ trong văn phong học thuật;",
        "Việc sử dụng AI KHÔNG được tham gia vào việc sáng tác nội dung, viết bản thảo bài báo, tạo dữ liệu giả mạo hoặc tổng hợp tài liệu hư cấu;",
        "Việc sử dụng AI phải được kiểm tra và xác nhận bởi tất cả các tác giả. Mỗi tác giả chịu hoàn toàn trách nhiệm về tính chính xác, tính minh bạch và tính nguyên bản của phần nội dung do mình phụ trách, bao gồm các trích dẫn và tài liệu tham khảo;",
        "Tất cả các tác giả phải công khai việc sử dụng AI tạo sinh và các công nghệ hỗ trợ AI với các đồng tác giả khác, giúp hiểu rõ AI đã được áp dụng ở đâu và như thế nào trong quá trình chuẩn bị nghiên cứu."
    ]
    for b in com_bullets:
        add_p(doc, f"• {b}", space_before=2, space_after=3)

    p_com_end = add_p(doc, "Bằng văn bản này, chúng tôi xin xác nhận và chịu hoàn toàn trách nhiệm về tính chính xác và tính minh bạch của các thông tin đã cung cấp.", space_before=6, space_after=14)

    # Table 2: Author Signatures (3 authors) with signatures applied directly
    tbl2 = doc.add_table(rows=2, cols=3)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl2, color="003366", sz="6")
    
    col_widths2 = [Inches(2.1), Inches(2.1), Inches(2.1)]
    for row in tbl2.rows:
        for i, w in enumerate(col_widths2):
            row.cells[i].width = w
            set_cell_margins(row.cells[i], top=100, bottom=100, left=100, right=100)
            row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    auth_headers = ["Tác giả 1 (Trưởng nhóm)", "Tác giả 2 (Thành viên)", "Tác giả 3 (Thành viên)"]
    for i, h in enumerate(auth_headers):
        cell = tbl2.rows[0].cells[i]
        set_cell_background(cell, "F0F4F8")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        format_run(run, size_pt=10.5, bold=True, color_rgb=(0, 51, 102))

    # Cell content with signatures:
    auth_data = [
        ("Như", "Nguyễn Hàn Như", "SE183644"),
        ("Thành", "Nguyễn Văn Thành", "SE180387"),
        ("Dũng", "Trần Phạm Tuấn Dũng", "SE180382"),
    ]
    for i, (sig, name, mssv) in enumerate(auth_data):
        cell = tbl2.rows[1].cells[i]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Label
        r_top = p.add_run("(Ký tên & Họ tên đầy đủ)\n\n")
        format_run(r_top, size_pt=9.5, italic=True, color_rgb=(120, 120, 120))
        
        # Handwritten-style signature
        r_sig = p.add_run(f"{sig}\n")
        format_run(r_sig, font_name="Segoe Script", size_pt=16, bold=True, italic=True, color_rgb=(0, 51, 153))
        
        # Underline and name
        r_name = p.add_run(f"_______________________\n{name}\n")
        format_run(r_name, size_pt=10.5, bold=True, color_rgb=(0, 0, 0))
        
        # MSSV
        r_mssv = p.add_run(f"MSSV: {mssv}")
        format_run(r_mssv, size_pt=9.5, bold=False, color_rgb=(80, 80, 80))

    # Notes section
    add_p(doc, "", space_before=14, space_after=6)
    p_note_title = add_p(doc, "Lưu ý (Note):", bold=True, space_before=4, space_after=2)
    notes = [
        "Đề xuất nghiên cứu phải được trình bày bằng Tiếng Anh/Tiếng Nhật, khổ giấy A4, định dạng một cột, font chữ Times New Roman cỡ 11, độ dài tối đa 10 trang (không bao gồm tài liệu tham khảo và phụ lục).",
        "Đối với trích dẫn và tài liệu tham khảo: Chuẩn IEEE áp dụng cho tiểu ban Công nghệ Thông tin (https://libguides.murdoch.edu.au/IEEE/home)."
    ]
    for n in notes:
        add_p(doc, f"• {n}", size_pt=10, italic=True, space_before=1, space_after=2)

    # Save to disk
    out_docx_vn = r"C:\Users\ADMIN\Downloads\capstone AI\NTD_Cong nghe thong tin_TiengViet.docx"
    doc.save(out_docx_vn)
    print(f"Successfully saved Vietnamese docx: {out_docx_vn}")

    # Convert to PDF
    out_pdf_vn = r"C:\Users\ADMIN\Downloads\capstone AI\NTD_Cong nghe thong tin_TiengViet.pdf"
    convert_docx_to_pdf(out_docx_vn, out_pdf_vn)

if __name__ == "__main__":
    main()
