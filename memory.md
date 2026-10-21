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
