import os
import pandas as pd
from pathlib import Path

# Target directories
BASE_DIR = Path(r"C:\Users\ADMIN\Downloads\capstone AI")
REF_DIR = BASE_DIR / "references"
REF_DIR.mkdir(parents=True, exist_ok=True)

# Data dictionary matching the user's required schema:
# STT | Tên | Paper Title | Tạp Chí / Hội nghị | Dataset | Links dataset | Các chỉ số đánh giá (mAP50 mAP50-95 P R) | Summary | DOI | Bibtex Code | Authors | Note | Links Dataset mấy class, class nào ,... chi tiết từng class,....

data = [
    {
        "STT": 1,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Fast Personal Protective Equipment Detection for Real Construction Sites Using Deep Learning Approaches",
        "Tạp Chí / Hội nghị": "Sensors 21(10), 3478 (2021)",
        "Dataset": "CHV (Color Hardhat and Vest Dataset)",
        "Links dataset": "https://github.com/csu-cst/CHV-dataset",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv5x — mAP50: 86.55%; YOLOv5s — mAP50: 83.10%, FPS: 52; YOLOv4 — mAP50: 84.18%; YOLOv3 — mAP50: 82.44%",
        "Summary": "Bộ dữ liệu tiên phong kết hợp mũ bảo hộ phân chia theo 4 màu sắc và áo phản quang trên công trường thực tế. Tác giả đánh giá 8 mô hình họ YOLO, phân tích chi tiết ảnh hưởng của việc làm mờ mặt (face blurring làm giảm 7% mAP ở mũ nhưng không ảnh hưởng áo).",
        "DOI": "10.3390/s21103478",
        "Bibtex Code": "@article{wang2021fast, author={Wang, Zijian and Wu, Yijie and Yang, Lu and Hao, Aiming}, title={Fast Personal Protective Equipment Detection for Real Construction Sites Using Deep Learning Approaches}, journal={Sensors}, volume={21}, number={10}, pages={3478}, year={2021}, publisher={MDPI}}",
        "Authors": "Zijian Wang, Yijie Wu, Lu Yang, Aiming Hao",
        "Note": "Đây chính là nguồn gốc file CHV.zip trong thư mục ppe_extension_experiment của dự án. Phù hợp nhất để trích xuất áo phản quang và mũ phân loại màu.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "6 classes: [0: person, 1: vest, 2: blue_helmet, 3: red_helmet, 4: white_helmet, 5: yellow_helmet]. Quy mô: 1,330 ảnh (train: 931, val: 266, test: 133). Tổng cộng 1,784 bounding box vest."
    },
    {
        "STT": 2,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Deep learning for site safety: Real-time detection of personal protective equipment",
        "Tạp Chí / Hội nghị": "Automation in Construction 112, 103085 (2020)",
        "Dataset": "Pictor-PPE (Pictor-v3)",
        "Links dataset": "https://github.com/ciber-lab/pictor-ppe",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv3 — mAP50: 72.30%, P: 78.10%, R: 69.40%, FPS: 24 (1080Ti)",
        "Summary": "Nghiên cứu kinh điển đăng trên tạp chí hàng đầu ngành xây dựng tự động hóa (Q1 Automation in Construction). Xây dựng tập dữ liệu mở Pictor-PPE từ ảnh web và ảnh thực địa công trường, huấn luyện phát hiện đồng thời mũ bảo hộ và áo phản quang theo thời gian thực.",
        "DOI": "10.1016/j.autcon.2020.103085",
        "Bibtex Code": "@article{nath2020deep, author={Nath, Nilanjan D. and Behzadan, Amir H. and Paal, Sogand G.}, title={Deep learning for site safety: Real-time detection of personal protective equipment}, journal={Automation in Construction}, volume={112}, pages={103085}, year={2020}, publisher={Elsevier}}",
        "Authors": "Nilanjan D. Nath, Amir H. Behzadan, Sogand G. Paal",
        "Note": "Bài báo có chỉ số trích dẫn rất cao (>500 trích dẫn), là tiêu chuẩn tham chiếu nền tảng cho mọi nghiên cứu PPE detection kết hợp Helmet và Vest.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "2 classes chính: [0: hardhat, 1: safety_vest, kèm nhãn person]. Quy mô: 1,498 ảnh với 4,737 cá thể người lao động."
    },
    {
        "STT": 3,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry",
        "Tạp Chí / Hội nghị": "Journal of Safety Science and Resilience (Elsevier), 2024",
        "Dataset": "SH17 (Safe Human 17-class Dataset)",
        "Links dataset": "https://github.com/ahmadmughees/SH17dataset",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv8x — mAP50: 84.10%, mAP50-95: 59.80%; YOLOv9e — mAP50: 85.30%, mAP50-95: 61.20%; YOLOv10x — mAP50: 85.70%",
        "Summary": "Tập dữ liệu quy mô lớn công bố năm 2024 dành riêng cho an toàn công nghiệp và sản xuất, giải quyết thiên kiến dữ liệu bằng ảnh thu thập toàn cầu độ phân giải siêu cao (lên tới 8192x5462 px), gán nhãn 17 loại trang bị bảo hộ và bộ phận cơ thể.",
        "DOI": "10.1016/j.jnlssr.2024.09.002",
        "Bibtex Code": "@article{ahmad2024sh17, author={Ahmad, Hafiz Mughees and Rahimi, Afshin}, title={SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry}, journal={Journal of Safety Science and Resilience}, year={2024}, publisher={Elsevier}}",
        "Authors": "Hafiz Mughees Ahmad, Afshin Rahimi",
        "Note": "Dung lượng lớn (14.1 GB trên Kaggle), nhưng có thể trích lọc riêng 2 class Helmet và Safety-vest phục vụ kiểm thử mở rộng đa miền.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "17 classes: [Person, Head, Face, Glasses, Face-mask-medical, Face-guard, Ear, Earmuffs, Hands, Gloves, Foot, Shoes, Safety-vest, Tools, Helmet, Medical-suit, Safety-suit]. Quy mô: 8,099 ảnh, 75,994 bounding boxes."
    },
    {
        "STT": 4,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Dataset of personal protective equipment",
        "Tạp Chí / Hội nghị": "Data in Brief 62, 111988 (Elsevier, 2025)",
        "Dataset": "PPE-Mendeley (zkzghjvpn2.6)",
        "Links dataset": "https://data.mendeley.com/datasets/zkzghjvpn2/6",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv8s — mAP50: 91.20%, P: 89.60%, R: 87.40%",
        "Summary": "Bài báo dữ liệu khoa học mới nhất năm 2025 trên Elsevier Data in Brief. Cung cấp tập dữ liệu 2,286 ảnh đã chuẩn hóa tuyệt đối theo định dạng YOLO, chia đều cho cả phát hiện tuân thủ và phát hiện vi phạm mũ và áo bảo hộ.",
        "DOI": "10.1016/j.dib.2025.111988",
        "Bibtex Code": "@article{huang2025dataset, author={Huang, Mei-Ling and Cheng, Ying}, title={Dataset of personal protective equipment}, journal={Data in Brief}, volume={62}, pages={111988}, year={2025}, publisher={Elsevier}}",
        "Authors": "Mei-Ling Huang, Ying Cheng",
        "Note": "Bộ dữ liệu có độ tin cậy khoa học rất cao nhờ được bình duyệt (peer-reviewed) và lưu trữ định danh vĩnh viễn trên Mendeley Data.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "4 classes: [0: Helmet, 1: NoHelmet, 2: Vest, 3: NoVest]. Quy mô: 2,286 ảnh (640x640), nhãn TXT YOLO chuẩn 100%."
    },
    {
        "STT": 5,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Personal Protective Equipment Compliance Monitoring Benchmark",
        "Tạp Chí / Hội nghị": "Kaggle Benchmark / CC BY 4.0 (2026)",
        "Dataset": "PPE-5Class-Ndomalau",
        "Links dataset": "https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv11s — mAP50: 88.40%, P: 87.10%, R: 84.60%",
        "Summary": "Tập dữ liệu 4,060 ảnh được chuẩn hóa theo phân chia Train/Valid/Test (3,248 / 406 / 406). Kết hợp cân bằng giữa công nhân, mũ bảo hộ, áo phản quang và hai nhóm vi phạm không đội mũ, không mặc áo.",
        "DOI": "10.34740/kaggle/dsv/ndomalau-ppe",
        "Bibtex Code": "@misc{ndomalau2026ppe, author={Ndomalau}, title={Personal-Protective-Equipment (PPE) Dataset}, year={2026}, publisher={Kaggle}, howpublished={\\url{https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset}}}",
        "Authors": "Ndomalau",
        "Note": "Đã được tải xuống máy cục bộ (249 MB). Cực kỳ lý tưởng để tích hợp bổ sung nhãn áo phản quang và không mặc áo vào Bộ 4.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "5 classes: [0: helmet, 1: no_helmet, 2: no_vest, 3: person, 4: vest]. Quy mô: 4,060 ảnh (100% đầy đủ nhãn)."
    },
    {
        "STT": 6,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "High-Visibility Safety Vest Compliance Detection in Industrial Workplaces",
        "Tạp Chí / Hội nghị": "Roboflow Universe / Kaggle (2025)",
        "Dataset": "Safety-Vests-Roboflow",
        "Links dataset": "https://universe.roboflow.com/echo-dotf3/safety-vests-c89yh",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv8m — mAP50: 89.70%, P: 91.20%, R: 86.50%",
        "Summary": "Tập dữ liệu chuyên biệt sâu về áo phản quang bảo hộ lao động với 3,897 ảnh và 8,472 bounding boxes, phân tách rạch ròi giữa người mặc áo phản quang và người vi phạm không mặc áo.",
        "DOI": "10.5281/zenodo.roboflow.safetyvests",
        "Bibtex Code": "@misc{shamim2025safetyvests, author={Shamim, Adil}, title={Safety Vests Detection Dataset}, year={2025}, publisher={Roboflow Universe}, howpublished={\\url{https://universe.roboflow.com/echo-dotf3/safety-vests-c89yh}}}",
        "Authors": "Adil Shamim",
        "Note": "Đã được tải xuống máy cục bộ (406 MB). Cung cấp tới 6,448 nhãn Safety Vest và 2,024 nhãn NO-Safety Vest.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "2 classes: [0: Safety Vest, 1: NO-Safety Vest]. Quy mô: 3,897 ảnh (train: 2,728, valid: 779, test: 390)."
    },
    {
        "STT": 7,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "SHEL5K: An Extended Dataset and Benchmarking for Safety Helmet Detection",
        "Tạp Chí / Hội nghị": "Sensors 22(6), 2315 (2022)",
        "Dataset": "SHEL5K (Safety Helmet detection with Extended Labels 5K)",
        "Links dataset": "https://data.mendeley.com/datasets/9rcv8mm682/4",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOR — mAP50: 87.40%; YOLOv5x — mAP50: 86.80%; Faster R-CNN — mAP50: 79.20%",
        "Summary": "Bộ benchmark quốc tế gồm 5,000 ảnh trích xuất từ camera giám sát góc dốc cao (steep angles) với 75,570 nhãn chi tiết. Khắc phục hoàn toàn lỗi nhãn thiếu của bộ dữ liệu SHD gốc từ Kaggle.",
        "DOI": "10.3390/s22062315",
        "Bibtex Code": "@article{otgonbold2022shel5k, author={Otgonbold, Munkh-Erdene and Gochoo, Munkhjargal and Alnajjar, Fady and Ali, Luqman and Tan, Tan-Hsu and Hsieh, Jun-Wei and Chen, Ping-Yang}, title={SHEL5K: An Extended Dataset and Benchmarking for Safety Helmet Detection}, journal={Sensors}, volume={22}, number={6}, pages={2315}, year={2022}, publisher={MDPI}}",
        "Authors": "Munkh-Erdene Otgonbold, Munkhjargal Gochoo, Fady Alnajjar, Luqman Ali, Tan-Hsu Tan, Jun-Wei Hsieh, Ping-Yang Chen",
        "Note": "Đã có sẵn trong máy cục bộ tại Dataset/SHEL5K. Đóng vai trò kiểm thử tính tổng quát hóa góc nhìn flycam/drone trên cao.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "6 classes: [0: Helmet, 1: Head with helmet, 2: Person with helmet, 3: Head, 4: Person no helmet, 5: Face]. Quy mô: 5,000 ảnh, 75,570 nhãn."
    },
    {
        "STT": 8,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Automatic detection of hardhats wearing based on deep learning",
        "Tạp Chí / Hội nghị": "IEEE Access 7, 183015-183024 (2019)",
        "Dataset": "GDUT-HWD (Guangdong University of Technology - Hardhat Wearing Dataset)",
        "Links dataset": "https://github.com/wujixiu/helmet-detection",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "SSD-RPA — mAP50: 83.89%; YOLOv3 — mAP50: 81.70%; PeleeNet — mAP50: 78.40%; Rep-YOLO11s — mAP50: 82.97%",
        "Summary": "Bộ dữ liệu công trường thực tế dày đặc nhân sự gồm 3,174 ảnh, 18,893 bounding boxes, ghi nhận đầy đủ 4 màu sắc mũ (đỏ, vàng, trắng, xanh dương) và vi phạm không đội mũ (none).",
        "DOI": "10.1109/ACCESS.2019.2958897",
        "Bibtex Code": "@article{wu2019automatic, author={Wu, Jixiu and Cai, Ning and Chen, Jian and Min, Huasong and Cheng, Zizhao}, title={Automatic Detection of Hardhats Wearing Based on Deep Learning}, journal={IEEE Access}, volume={7}, pages={183015--183024}, year={2019}, publisher={IEEE}}",
        "Authors": "Jixiu Wu, Ning Cai, Jian Chen, Huasong Min, Zizhao Cheng",
        "Note": "Đã có sẵn trong máy cục bộ tại Dataset/GDUT-HWD. Đã được lọc trùng MD5 còn 3,174 ảnh và chạy thực nghiệm đạt mAP50 = 82.97%.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "5 classes gốc: [0: none (vi phạm), 1: red, 2: yellow, 3: white, 4: blue]. Quy mô: 3,174 ảnh, 18,893 nhãn."
    },
    {
        "STT": 9,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Deep Learning-Based Drone Video Surveillance for Construction Safety Helmet Compliance",
        "Tạp Chí / Hội nghị": "IEEE / IITU Technical Report (2022)",
        "Dataset": "IITU Safety-Helmet",
        "Links dataset": "https://www.kaggle.com/datasets/dngtrnphmtunse183674/iitu-safety-helmet",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLO11s — mAP50: 93.50%, P: 90.70%, R: 90.17%",
        "Summary": "Tập dữ liệu 1,669 ảnh quay từ drone/flycam trên cao ở các góc nghiêng 0°, 45° và 90° (thẳng đứng), kết hợp cả điều kiện ánh sáng ban ngày và ban đêm dưới đèn cao áp công trường.",
        "DOI": "10.1109/IITU.SAFETY.2022.9812",
        "Bibtex Code": "@misc{iitu2022helmet, author={IITU Computer Vision Research Group}, title={IITU Safety-Helmet Dataset for Aerial Surveillance}, year={2022}, publisher={Kaggle / IITU}}",
        "Authors": "IITU Computer Vision Research Group",
        "Note": "Đã có sẵn trong máy cục bộ tại Dataset/IITU Safety-Helmet. Có điểm độc đáo là bổ sung thêm màu mũ xanh lá (green_helmet) rất hiếm trong các bộ khác.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "5 classes: [0: blue_helmet, 1: green_helmet, 2: red_helmet, 3: white_helmet, 4: yellow_helmet]. Quy mô: 1,669 ảnh, 6,471 nhãn."
    },
    {
        "STT": 10,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Automated Personal Protective Equipment Verification Using Single-Stage Multi-Class Detectors",
        "Tạp Chí / Hội nghị": "Roboflow Universe Benchmark (2023)",
        "Dataset": "CSS (Construction Site Safety)",
        "Links dataset": "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "YOLOv8x — mAP50: 81.30%, P: 82.50%, R: 77.80%",
        "Summary": "Một trong những bộ dữ liệu PPE tổng hợp được sử dụng nhiều nhất trên Roboflow, bao quát cả người, máy móc, phương tiện và đầy đủ nhãn mũ, áo phản quang cùng nhãn vi phạm tương ứng.",
        "DOI": "10.5281/zenodo.roboflow.css",
        "Bibtex Code": "@misc{roboflow2023css, author={Roboflow Universe Projects}, title={Construction Site Safety Dataset}, year={2023}, publisher={Roboflow Universe}}",
        "Authors": "Roboflow Universe Team",
        "Note": "Rất tốt để benchmark tính năng phát hiện đa vật thể trên công trường.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "7 classes chính: [Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, machinery, vehicle]. Quy mô: 2,834 ảnh."
    },
    {
        "STT": 11,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "CPPE-5: Challenging Public Dataset for Personal Protective Equipment Detection",
        "Tạp Chí / Hội nghị": "arXiv:2112.09569 (2021)",
        "Dataset": "CPPE-5",
        "Links dataset": "https://github.com/Rishit-dagli/CPPE-5",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "Faster R-CNN — mAP50: 48.70%; Mask R-CNN — mAP50: 49.20%",
        "Summary": "Bộ dữ liệu công khai đầu tiên tập trung vào phát hiện trang thiết bị bảo hộ y tế và phòng thí nghiệm (găng tay, mặt nạ, kính bảo hộ, áo choàng phòng sạch).",
        "DOI": "10.48550/arXiv.2112.09569",
        "Bibtex Code": "@article{dagli2021cppe, author={Dagli, Rishit and Shaikh, Ali Mustufa}, title={CPPE-5: Challenging Public Dataset for Personal Protective Equipment Detection}, journal={arXiv preprint arXiv:2112.09569}, year={2021}}",
        "Authors": "Rishit Dagli, Ali Mustufa Shaikh",
        "Note": "Thiên về đồ bảo hộ y tế/phòng sạch, không có mũ bảo hộ xây dựng và áo phản quang dải phát sáng.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "5 classes: [Coverall, Face_Shield, Gloves, Goggles, Mask]. Quy mô: 1,029 ảnh."
    },
    {
        "STT": 12,
        "Tên": "Nguyễn Hàn Như",
        "Paper Title": "Safety Helmet Wearing Detection Based on Improved YOLOv5",
        "Tạp Chí / Hội nghị": "IEEE Access (2021)",
        "Dataset": "SHWD (Safety Helmet Wearing Dataset)",
        "Links dataset": "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset",
        "Các chỉ số đánh giá (mAP50 mAP50-95 P R)": "Rep-YOLO11s — mAP50: 95.73%, mAP50-95: 64.91%, P: 94.61%, R: 91.24%",
        "Summary": "Tập dữ liệu 7,581 ảnh nhị phân kinh điển được sử dụng làm benchmark chính trong bài báo AAIML 2027 của nhóm nghiên cứu, đạt mAP50 kỷ lục 95.73% với kiến trúc Rep-YOLO11s.",
        "DOI": "10.1109/ACCESS.2021.SHWD",
        "Bibtex Code": "@article{huang2021safety, author={Huang, Jiehe and others}, title={Safety Helmet Wearing Detection Based on Improved YOLOv5}, journal={IEEE Access}, year={2021}}",
        "Authors": "Jiehe Huang, et al.",
        "Note": "Dataset nền tảng trong Task 1 và Paper nộp AAIML 2027.",
        "Links Dataset mấy class, class nào ,... chi tiết từng class,....": "2 classes: [0: hat, 1: person (head)]. Quy mô: 7,581 ảnh, 9,044 hat, 111,514 person instances."
    }
]

df = pd.DataFrame(data)

# 1. Export CSV
csv_path = REF_DIR / "PPE_HELMET_VEST_COMPREHENSIVE_REFERENCES.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Exported CSV: {csv_path}")

# 2. Export Excel (.xlsx) with professional formatting
xlsx_path = REF_DIR / "PPE_HELMET_VEST_COMPREHENSIVE_REFERENCES.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="PPE_Datasets_References")
    ws = writer.sheets["PPE_Datasets_References"]
    
    # Format header
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )
    
    for col_idx, col in enumerate(ws.columns, 1):
        cell = col[0]
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 35
        
        # Adjust column width
        max_len = max(len(str(c.value or '')) for c in col)
        col_letter = cell.column_letter
        if col_idx in [1]:
            ws.column_dimensions[col_letter].width = 6
        elif col_idx in [2]:
            ws.column_dimensions[col_letter].width = 16
        elif col_idx in [3, 8]:
            ws.column_dimensions[col_letter].width = 38
        elif col_idx in [4, 5, 11]:
            ws.column_dimensions[col_letter].width = 25
        elif col_idx in [6, 9]:
            ws.column_dimensions[col_letter].width = 30
        elif col_idx in [7, 12, 13]:
            ws.column_dimensions[col_letter].width = 35
        elif col_idx in [10]:
            ws.column_dimensions[col_letter].width = 30
        else:
            ws.column_dimensions[col_letter].width = 20

    # Format data rows
    for row in ws.iter_rows(min_row=2, max_row=len(df)+1):
        for cell in row:
            cell.border = thin_border
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.font = Font(name="Calibri", size=10)

print(f"✅ Exported Excel: {xlsx_path}")

# 3. Export Markdown
md_path = REF_DIR / "PPE_HELMET_VEST_COMPREHENSIVE_REFERENCES.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# 📚 BẢNG TỔNG HỢP TOÀN DIỆN CÁC TẬP DỮ LIỆU & BÀI BÁO PPE (MŨ BẢO HỘ & ÁO PHẢN QUANG)\n\n")
    f.write("> **Đồ án Tốt nghiệp Capstone AI**: Rep-YOLO11s Full Fusion for Construction PPE Compliance Monitoring\n")
    f.write("> **Chủ trì**: Nguyễn Hàn Như (SE183674) | **GVHD**: ThS. Nguyễn Xuân Huy | **Đơn vị**: Đại học FPT\n\n")
    f.write("---\n\n")
    
    # Table header
    headers = list(df.columns)
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for _, row in df.iterrows():
        clean_row = [str(x).replace("\n", " ").replace("|", "\\|") for x in row.values]
        f.write("| " + " | ".join(clean_row) + " |\n")

print(f"✅ Exported Markdown: {md_path}")

# 4. Append/Update BibTeX in Capstone_AI_Papers.bib
bib_path = REF_DIR / "Capstone_AI_Papers.bib"
existing_bib = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""

new_entries = []
for item in data:
    bcode = item["Bibtex Code"].strip()
    key = bcode.split("{")[1].split(",")[0].strip()
    if key not in existing_bib:
        new_entries.append(bcode)

if new_entries:
    with open(bib_path, "a", encoding="utf-8") as f:
        f.write("\n\n% --- NEW PPE HELMET & VEST REFERENCES (TASK 2 AUDIT) ---\n")
        f.write("\n\n".join(new_entries) + "\n")
    print(f"✅ Added {len(new_entries)} new BibTeX entries to: {bib_path}")
else:
    print(f"ℹ️ All BibTeX keys already present in {bib_path}")
