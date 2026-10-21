# 🧠 MEMORY.MD — BỘ NHỚ KHOA HỌC DỰ ÁN CAPSTONE AI (REP-YOLO11s)

> **Tên đồ án:** Hệ thống Giám sát Tuân thủ An toàn Lao động Thời gian thực trên Thiết bị Biên (Rep-YOLO11s)  
> **Chủ trì nghiên cứu:** Nguyễn Hàn Như (SE183674) | **GVHD:** ThS. Nguyễn Xuân Huy  
> **Đơn vị:** Đại học FPT - Viện Đào tạo Quốc tế  
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
- **Cần bổ sung cho đợt báo cáo lần 2 (Task 2)**:
  1. Data Pipeline đa lớp: Báo cáo làm sạch, loại bỏ trùng lặp, phân tích mất cân bằng lớp định lượng, chia Train/Val/Test 80/10/10.
  2. Baseline 1 (YOLO11s) và Baseline 2 (YOLOv8s) cùng mô hình đề xuất trên dataset multi-class.
  3. Báo cáo phân tích lỗi (Failure Case Analysis) và tính khả thi sản phẩm.
  4. Bộ Slide thuyết trình Task 2 (18-25 slide theo đúng chuẩn mẫu 4 Chương).

---

## 2. KHO DỮ LIỆU & PHÂN TÍCH 4 PHƯƠNG ÁN QUYẾT ĐỊNH

### 2.1. Thống kê nguồn dữ liệu đã kiểm định toàn vẹn:
1. **GDUT-HWD** (`tundng111/gdut-hwd`):
   - 3,174 ảnh, 3,174 XML Pascal VOC.
   - Làm sạch: Loại bỏ chính xác **14 ảnh trùng lặp tuyệt đối (MD5 hash)**.
   - Phân bố: `none` (4,371 - 23.1%), `red` (4,231 - 22.4%), `yellow` (3,898 - 20.6%), `white` (3,781 - 20.0%), `blue` (2,612 - 13.8%).
   - Ưu điểm: Có lớp vi phạm `none`. Nhược điểm: Thiếu mũ xanh lá và thiếu áo.
2. **IITU Safety-Helmet** (`dngtrnphmtunse183674/iitu-safety-helmet`):
   - 1,669 ảnh, 1,664 txt YOLO (5 ảnh rỗng).
   - Không trùng lặp nội bộ, không trùng lặp chéo với GDUT.
   - Góc quay Drone đa dạng (0°, 45°, 90°), ban ngày và ban đêm.
   - Phân bố: `white` (1,401), `yellow` (1,362), `blue` (1,359), `red` (1,216), `green` (1,133 - có màu xanh lá hiếm).
   - Nhược điểm: Thiếu lớp vi phạm `none` và thiếu áo.
3. **STANDARDIZED_CHV**:
   - 1,330 ảnh (1,064 train, 133 val, 133 test).
   - Phân bố: `hat` (3,538), `person` (3,887), `vest` (1,784).
   - Cung cấp đặc trưng phản quang cho áo bảo hộ.

### 2.2. Đánh giá 4 Phương án Quyết định Dataset:
- **Bộ 1 (GDUT-HWD)**: 5 class (`none`, red, yellow, white, blue). Tốt cho phát hiện vi phạm nhưng thiếu xanh lá và áo.
- **Bộ 2 (GDUT-HWD + Mũ cam)**: 6 class. Thêm được mũ cam nhưng vẫn thiếu áo và xanh lá.
- **Bộ 3 (IITU)**: 5 class màu. Cực tốt cho flycam/ban đêm nhưng KHÔNG CÓ nhãn vi phạm `none`.
- **Bộ 4 (Unified PPE Master - Hợp nhất GDUT + IITU + Vest)**: **LỰA CHỌN TỐI ƯU SỐ 1**.
  * 7 class chuẩn hóa: `[0: blue_helmet, 1: green_helmet, 2: red_helmet, 3: white_helmet, 4: yellow_helmet, 5: no_helmet, 6: safety_vest]` (sẵn sàng mở rộng lên 8 class khi có mũ cam).
  * Đạt điểm 10/10 tiêu chí đánh giá của Thầy và Hội đồng ĐATN.

---

## 3. KHẲNG ĐỊNH TÍNH BẢO TOÀN CỦA 5 MODULE KIẾN TRÚC REP-YOLO11s

**Tuyệt đối không phá vỡ kiến trúc đồng nhất để chắp vá các module bên ngoài.** 5 module cốt lõi của Rep-YOLO11s là các inductive bias phần cứng độc lập với số lượng class:
1. **Stem CoordConv**: Mã hóa tọa độ không gian $y$ (mũ ở đỉnh đầu, áo ở thân trên, nhiễu ở sàn).
2. **Neck RepConv**: Trích xuất đa nhánh đặc trưng phản quang và vỏ mũ nhựa, gập nhánh lúc suy luận $\to$ Duy trì >200 FPS.
3. **Neck BiFormer Attention**: Lọc 80% token nền, tập trung tính toán vào mũ xa và sọc phản quang.
4. **Focal EIoU Loss**: Tối ưu trực tiếp tỷ lệ khung hình (mũ vuông, áo đứng) và cân bằng mất cân bằng lớp.
5. **Task-Aligned Assigner (TAL)**: Đồng bộ điểm số nhận diện màu sắc/áo với độ chính xác bounding box.

---

## 4. HỆ THỐNG KAGGLE DISPATCHER VÀ 3 TÀI KHOẢN THỰC NGHIỆM

Đã triển khai hệ thống quản trị phân tán `scripts/kaggle_multitoken_dispatcher.py` với 3 tài khoản:
1. **Tài khoản 1 (`hannhu4002`)**: Chạy Bộ 1 (`dataset-bo1-5class-gdut-hwd-benchmark`).
2. **Tài khoản 2 (`mnu303`)**: Chạy Bộ 3 (`dataset-bo3-5class-iitu-drone-benchmark`).
3. **Tài khoản 3 (`hannhu40022`)**: Chạy Bộ 4 (`dataset-bo4-7class-unified-ppe-master`).

**Nguyên tắc phòng chống khóa tài khoản Kaggle**:
- Luôn kiểm soát số lượng job song song $\le 1$ GPU job / tài khoản.
- Không sử dụng vòng lặp polling tần suất cao (áp dụng độ trễ $\ge 3$s giữa các request).
- Đảm bảo metadata chứa đầy đủ `kernelspec` chuẩn Python 3 để tránh lỗi Papermill.
