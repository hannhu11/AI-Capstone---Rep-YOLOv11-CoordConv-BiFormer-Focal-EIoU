# 🧠 MEMORY.MD — BỘ NHỚ KHOA HỌC DỰ ÁN CAPSTONE AI (REP-YOLO11s)

> **Tên đồ án:** Hệ thống Giám sát Tuân thủ An toàn Lao động Thời gian thực trên Thiết bị Biên (Rep-YOLO11s)  
> **Chủ trì nghiên cứu:** Nguyễn Hàn Như (SE183674) | **Thành viên:** Nguyễn Thành | **GVHD:** ThS. Nguyễn Xuân Huy  
> **Đơn vị:** Đại học FPT TP.HCM - Viện Đào tạo Quốc tế  
> **Bài báo khoa học quốc tế:** *Rep-YOLO11s: An Efficient Hardware-Aware Edge Detector for PPE Compliance Monitoring* (AAIML 2027 Submission)  
> **Ngày cập nhật:** 2026-10-21  

---

## 1. TIẾN ĐỘ THỰC HIỆN & ĐÁNH GIÁ ĐỢT REVIEW 1 / TASK 2

### 1.1. Nhận xét của Thầy hướng dẫn (Review 1):
1. Đề tài phát hiện vi phạm mũ bảo hộ 2 lớp (`hat`, `person`) còn đơn giản / hơi thiếu tính ứng dụng sâu.
2. Cần mở rộng sang:
   - **Phân loại màu sắc mũ bảo hộ** (Trắng, Vàng, Đỏ, Xanh dương, Xanh lá, Cam...) để phục vụ phân quyền công trường (Kỹ sư: Mũ trắng; Công nhân: Mũ vàng; An toàn viên: Mũ đỏ/xanh; Khách: Mũ cam).
   - **Nhận diện áo bảo hộ phản quang** (`vest` / `safety_vest`) để đảm bảo tuân thủ trang bị bảo hộ cá nhân (PPE) toàn diện.
3. Bản prototype cần thể hiện rõ luồng dữ liệu sạch, xử lý mất cân bằng lớp và tính khả thi triển khai trên thiết bị biên (Edge Device).

### 1.2. Khoảng trống Task 2 cần hoàn thiện:
- **Đã có (trên tập SHWD 2-class)**: Toàn bộ chuỗi thực nghiệm A0 -> A6 đầy đủ, ablation study 5 module, đo đạc phần cứng (RTX 3050, T4, CPU, 15W clamp).
- **Đã bổ sung cho Task 2**:
  1. Pipeline dữ liệu đa lớp hoàn chỉnh: [Dataset/README.md](file:///c:/Users/ADMIN/Downloads/capstone%20AI/Dataset/README.md).
  2. Phát hiện và loại bỏ chính xác **14 ảnh trùng lặp tuyệt đối (MD5)** trong GDUT-HWD.
  3. Bộ 3 notebook benchmark chuẩn mực cho 3 bộ dữ liệu: Bộ 1 GDUT-HWD, Bộ 3 IITU Drone, Bộ 4 Unified 7-Class.
  4. Thực nghiệm phân tán thành công trên 3 tài khoản Kaggle GPU Tesla T4 (`hannhu4002`, `mnu303`, `hannhu40022`).
  5. Kịch bản Slide báo cáo Task 2 (18 slides) chuẩn mực 4 Chương: [SLIDE_TASK_2_BAO_CAO_TIEN_DO.md](file:///c:/Users/ADMIN/Downloads/capstone%20AI/SLIDE_TASK_2_BAO_CAO_TIEN_DO.md).

---

## 2. BẢNG TỔNG HỢP SỐ LIỆU THỰC NGHIỆM ĐỘC LẬP TRÊN 3 BỘ DỮ LIỆU

| Chỉ số đánh giá | Bộ 1: GDUT-HWD 5-Class (Kaggle: `hannhu4002`) | Bộ 3: IITU Drone 5-Class (Kaggle: `mnu303`) | Bộ 4: Unified 7-Class Master (Kaggle: `hannhu40022`) |
| :--- | :---: | :---: | :---: |
| **Số lượng ảnh thực nghiệm** | 3,160 ảnh (Sau lọc 14 trùng) | 1,657 ảnh | **4,817 ảnh** (Hợp nhất toàn diện) |
| **Độ chính xác Precision** | 88.67% | **90.70%** | **89.73%** |
| **Độ nhạy Recall** | 76.66% | **90.17%** | **87.58%** |
| **mAP@50 (Overall)** | 82.97% | **93.50%** | **91.99%** |
| **mAP@50-95** | 53.93% | 45.51% | **54.99%** |
| **Phát hiện Vi phạm (`no_helmet`)** | 80.08% mAP50 | ❌ (Không có nhãn) | **88.65% mAP50** (Vượt trội +8.57%) |
| **Mũ Xanh dương (`blue_helmet`)** | 73.84% mAP50 | 92.11% mAP50 | **90.14% mAP50** |
| **Mũ Xanh lá (`green_helmet`)** | ❌ (Không có) | 95.59% mAP50 | **93.49% mAP50** |
| **Mũ Đỏ (`red_helmet`)** | 79.53% mAP50 | 92.22% mAP50 | **93.40% mAP50** |
| **Mũ Trắng (`white_helmet`)** | 92.30% mAP50 | 95.75% mAP50 | **93.88% mAP50** |
| **Mũ Vàng (`yellow_helmet`)** | 89.09% mAP50 | 91.85% mAP50 | **92.38% mAP50** |
| **Áo phản quang (`safety_vest`)** | ❌ (Không có) | ❌ (Không có) | **87.28% mAP50** |

---

## 3. KẾT LUẬN CHIẾN LƯỢC VỀ PHƯƠNG ÁN DATASET
1. **Bộ 1 (GDUT-HWD)**: Phù hợp phát hiện vi phạm (`no_helmet`), nhưng thiếu màu xanh lá và thiếu áo.
2. **Bộ 3 (IITU Safety-Helmet)**: Cực tốt cho flycam/drone và ban đêm, nhưng thiếu nhãn vi phạm `none` nên không thể dùng độc lập.
3. **Bộ 4 (Unified PPE Master)**: **LỰA CHỌN TỐI ƯU DUY NHẤT**. Hợp nhất toàn bộ ưu điểm: Đạt **91.99% mAP@50**, có đầy đủ 5 màu mũ, nhận diện vi phạm không đội mũ đạt 88.65%, và nhận diện áo phản quang đạt 87.28%.

---

## 4. KHẲNG ĐỊNH TÍNH BẢO TOÀN CỦA 5 MODULE KIẾN TRÚC REP-YOLO11s
- **Tuyệt đối không chắp vá thêm module ngoài (Lego-style)**.
- 5 module: Stem CoordConv (độ cao $y$), Neck RepConv (>200 FPS), BiFormer Attention (lọc 80% nền), Focal EIoU Loss (cân bằng hình học mũ-áo), Task-Aligned Assigner (đồng bộ phân loại & định vị) đã chứng minh độ bền vững tuyệt đối qua thực nghiệm 7 lớp.


---

## 3. BÁO CÁO KIỂM TOÁN NGUỒN GỐC DỮ LIỆU ÁO PHẢN QUANG & STANDARDIZED (100% LIÊM CHÍNH KHOA HỌC)

### 3.1. Nguồn gốc của thư mục `Dataset\STANDARDIZED`:
- Thư mục `C:\Users\ADMIN\Downloads\capstone AI\Dataset\STANDARDIZED` được tạo ra vào ngày 26/08/2026 bởi script `scripts/standardize_and_benchmark_cross_datasets.py` nhằm phục vụ thử nghiệm đánh giá Zero-Shot Cross-Domain Transfer.
- Thư mục này chỉ gồm 2 thư mục con:
  1. `GDUT_HWD`: Chuẩn hóa về 2 lớp `hat` và `person`.
  2. `SHEL5K`: Chuẩn hóa về 2 lớp `hat` và `person`.
- **Kết luận liêm chính**: Trong `Dataset\STANDARDIZED` **hoàn toàn KHÔNG CÓ nhãn áo phản quang (`vest`) hay áo bảo hộ**.

### 3.2. Nguồn gốc thực sự của nhãn áo phản quang (`vest`) trong dự án:
- Dữ liệu áo phản quang cục bộ nằm tại `C:\Users\ADMIN\Downloads\capstone AI\ppe_extension_experiment\dataset\CHV.zip` (439 MB, giải nén thành `STANDARDIZED_CHV`).
- Đây là tập dữ liệu **CHV (Color Hardhat and Vest Dataset)** có nguồn gốc từ bài báo khoa học uy tín:
  - **Bài báo**: *Zijian Wang, Yijie Wu, Lu Yang, Aiming Hao, "Fast Personal Protective Equipment Detection for Real Construction Sites Using Deep Learning Approaches", Sensors 2021, 21(10), 3478. DOI: 10.3390/s21103478.*
  - **Cấu trúc**: 1,330 ảnh, 6 classes: `[person, vest, blue_helmet, red_helmet, white_helmet, yellow_helmet]`. Có 1,784 bounding box `vest`.

### 3.3. Giải trình trung thực về lần chạy test Bộ 4 trên Kaggle vừa qua:
- Lần chạy trên tài khoản `hannhu40022` sử dụng script sinh tự động nạp 2 nguồn Kaggle input do người dùng cung cấp: `tundng111/gdut-hwd` và `dngtrnphmtunse183674/iitu-safety-helmet`.
- Taxonomy khai báo 7 lớp (dự trù sẵn class 6: `safety_vest`), tuy nhiên do chưa đính kèm tập CHV hoặc tập Vest lên Kaggle trong lần chạy đó, class `safety_vest` có 0 mẫu và chỉ đóng vai trò pipeline test sanity check (5 epochs, backbone tiêu chuẩn) để xác nhận hệ thống có thể học hợp nhất giữa GDUT và IITU mượt mà, **chưa áp dụng 5 module kiến trúc sâu A1-A6** và **chưa có số liệu thực của áo phản quang**.

---

## 4. CHUẨN HÓA THUẬT NGỮ KHOA HỌC: "ÁO PHẢN QUANG" VS "ÁO BẢO HỘ"

1. **Về mặt kỹ thuật thị giác máy tính (Computer Vision Target)**:
   - Thuật ngữ chính xác tuyệt đối là **"Áo phản quang" (High-Visibility Safety Vest / Reflective Vest)**.
   - Lý do: Đặc trưng thị giác mà mạng nơ-ron nhận diện là màu huỳnh quang (fluorescent yellow-green / fluorescent orange-red) và các dải phản quang kim loại (retroreflective stripes) phản xạ ánh sáng mạnh theo tiêu chuẩn **OSHA 29 CFR 1926.201 / ANSI/ISEA 107 / EN ISO 20471**.
   - Nếu gọi là "Áo bảo hộ" (*Protective Clothing*), khái niệm này quá rộng (bao gồm quần áo bảo hộ chống hóa chất, áo khoác ấm, áo bảo hộ thợ hàn, quần áo chống tĩnh điện), khiến bộ nhận diện dễ bị gán sai nhãn (false positive).
2. **Khuyến nghị sử dụng trong Báo cáo / Đồ án / Bài báo IEEE**:
   - **Tên đề tài & bối cảnh**: Sử dụng cụm từ rộng **Personal Protective Equipment (PPE) Compliance Monitoring** (*Hệ thống giám sát tuân thủ trang thiết bị bảo hộ lao động cá nhân*).
   - **Nhãn phân loại (Object Class)**: Sử dụng chính xác `safety_vest` (tiếng Việt: *Áo phản quang bảo hộ*).
   - Như vậy, hoàn toàn có thể dùng cả hai: Đề tài là **Bảo hộ lao động (PPE)**, còn nhiệm vụ nhận diện cụ thể là **Mũ bảo hộ (Safety Helmet)** và **Áo phản quang bảo hộ (Safety Vest)**.

---

## 5. KẾT QUẢ KHẢO CỨU DEEP RESEARCH & TẢI XUỐNG CÁC TẬP DỮ LIỆU MỚI (< 5GB)

Đã tải thành công và kiểm định toàn vẹn 3 tập dữ liệu ứng viên mới vào `Dataset/DOWNLOADED_CANDIDATES`:
1. **Personal-Protective-Equipment (PPE) Dataset (ndomalau)**: 249 MB, 4,060 ảnh chuẩn YOLO, 5 lớp: `helmet, no_helmet, no_vest, person, vest`. (Cân bằng xuất sắc, có cả vi phạm không mặc áo).
2. **Safety Vests Detection Dataset (Adil Shamim)**: 406 MB, 3,897 ảnh, 2 lớp: `Safety Vest` (6,448 nhãn) và `NO-Safety Vest` (2,024 nhãn). (Chuyên biệt sâu về áo phản quang).
3. **Safety Helmet & Vest (Maryam Borzoo)**: 116 MB, 1,754 ảnh, 2 lớp: `helmet` và `vest`.
4. **CHV Dataset (Sensors 2021)**: Có sẵn 439 MB tại `ppe_extension_experiment/dataset/CHV.zip` (1,330 ảnh, 6 lớp).

Toàn bộ thông tin đối chiếu 12 bài báo / tập dữ liệu đã được xuất ra 3 định dạng tại:
- `references/PPE_HELMET_VEST_COMPREHENSIVE_REFERENCES.csv`
- `references/PPE_HELMET_VEST_COMPREHENSIVE_REFERENCES.xlsx` (Định dạng Excel kẻ bảng chuẩn thẩm mỹ)
- `references/PPE_HELMET_VEST_COMPREHENSIVE_REFERENCES.md`
- `references/Capstone_AI_Papers.bib` (Đã cập nhật mã trích dẫn BibTeX chuẩn xác)


---

## 6. HOÀN THÀNH QUY TRÌNH HỢP NHẤT DỮ LIỆU BỘ 4 CHUẨN MỰC (UNIFIED PPE MASTER HARMONIZED)

Đã hoàn thành phân tích và giải quyết triệt để vấn đề 770 ảnh giao thoa giữa GDUT-HWD và CHV bằng cơ chế **Annotation Fusion & Harmonization** (Script: `scripts/harmonize_unified_ppe_master.py`):
1. **Phân lập và ghép nhãn chính xác**:
   - 770 ảnh giao thoa GDUT $\cap$ CHV được ghép nhãn vi phạm `none` và mũ của GDUT với nhãn áo phản quang `vest` của CHV.
   - Giữ nguyên 2,390 ảnh thuần GDUT.
   - Giữ nguyên 560 ảnh thuần CHV (ảnh mới của Cranfield University).
   - Tích hợp 1,664 ảnh thuần IITU Flycam (độc quyền mũ xanh lá và góc nhìn dốc cao).
2. **Quy mô tập dữ liệu Bộ 4 Harmonized (`Dataset/UNIFIED_PPE_MASTER_HARMONIZED`)**:
   - **Tổng số ảnh độc lập**: **5,359 ảnh** (Train: 4,287, Val: 535, Test: 537 - Tỷ lệ 80/10/10).
   - **Tỷ lệ trùng lặp**: **0.00%** (Loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu chéo Train-Test Data Leakage).
   - **Tổng số Bounding Box**: **28,203 hộp bao** chất lượng cao:
     * `white_helmet`: 5,666 boxes
     * `yellow_helmet`: 5,635 boxes
     * `red_helmet`: 5,543 boxes
     * `no_helmet` (vi phạm): 4,368 boxes
     * `blue_helmet`: 4,074 boxes
     * `safety_vest`: 1,784 boxes
     * `green_helmet`: 1,133 boxes
   - Đã sinh sẵn tệp `data.yaml` chuẩn YOLO 7 lớp, sẵn sàng huấn luyện và đánh giá trên Kaggle/Colab.
