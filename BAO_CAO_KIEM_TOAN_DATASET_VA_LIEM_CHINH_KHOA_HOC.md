# BÁO CÁO GIÁM ĐỊNH PHÁP Y DỮ LIỆU & BẢO VỆ LIÊM CHÍNH KHOA HỌC
## ĐỒ ÁN TỐT NGHIỆP: REP-YOLO11s — GIÁM SÁT AN TOÀN LAO ĐỘNG TRÊN THIẾT BỊ BIÊN
**Chủ trì:** Nguyễn Hàn Như (SE183674) | **Thành viên:** Nguyễn Thành | **GVHD:** ThS. Nguyễn Xuân Huy  
**Đơn vị:** Viện Đào tạo Quốc tế - Đại học FPT TP.HCM  
**Thời gian kiểm định:** 26/10/2026  

---

## 1. TỔNG QUAN VẤN ĐỀ & CHỈ ĐẠO CỦA THẦY HƯỚNG DẪN
Trong quá trình nghiệm thu, Thầy hướng dẫn đã đưa ra nhận định mang tính sống còn đối với phương pháp luận nghiên cứu:
> *"Tuyệt đối không được chấp nhận dùng dữ liệu do Teacher Model sinh nhãn (Pseudo-Labeling) trên tập Test rồi học và đánh giá, vì kết quả bị bão hòa ảo (97-98%). Phải kiểm tra toàn diện tính toàn vẹn dữ liệu, rò rỉ dữ liệu (Data Leakage), ảnh trùng lặp, xung đột nhãn ground-truth và chỉ sử dụng mô hình liêm chính, trung thực."*

Báo cáo này đóng vai trò một **Reviewer Độc lập & Khắt khe** thuộc phân nhóm Q1 IEEE TPAMI / CVPR, thực hiện giám định pháp y nhị phân (Binary Forensic Audit) trên toàn bộ hệ sinh thái dữ liệu của dự án.

---

## 2. KẾT QUẢ KIỂM TOÁN PHÁP Y TRÊN TỪNG BỘ DỮ LIỆU

### 2.1. Kiểm định Rò rỉ Dữ liệu Nội bộ qua Mã băm MD5 (Cross-Split Data Leakage)
Đã thực hiện băm MD5 trên từng tệp ảnh nhị phân độc lập tại script `scripts/audit_dataset_leakage_and_integrity.py`:

| Bộ Dữ Liệu | Số ảnh Train | Số ảnh Val | Số ảnh Test | Trùng lặp Train $\cap$ Val | Trùng lặp Train $\cap$ Test | Trùng lặp Val $\cap$ Test | Đánh giá Liêm chính |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CHV_6CLASS_OFFICIAL_CLEAN** | 1,064 | 133 | 133 | **0 ảnh (0%)** | **0 ảnh (0%)** | **0 ảnh (0%)** | **ĐẠT CHUẨN 100%** |
| **CHV_7CLASS_CONFLICT_FREE** | 1,017 | 130 | 120 | **0 ảnh (0%)** | **0 ảnh (0%)** | **0 ảnh (0%)** | **ĐẠT CHUẨN 100%** |
| **UNIFIED_IITU_CHV_COMPLETE** | 2,386 | 295 | 306 | **0 ảnh (0%)** | **0 ảnh (0%)** | **0 ảnh (0%)** | Đạt về MD5, nhưng dính Pseudo-label |
| **UNIFIED_PPE_MASTER_HARMONIZED** | 4,287 | 535 | 537 | **0 ảnh (0%)** | **0 ảnh (0%)** | **0 ảnh (0%)** | **ĐẠT CHUẨN 100%** |

**Kết luận 1**: Không có bất kỳ hiện tượng trùng lặp ảnh thô (duplicate image files) nào giữa các tập Train, Val, và Test trong cả 4 phân hoạch.

---

### 2.2. Giải mã "Tử huyệt" Giao thoa 770 ảnh giữa GDUT-HWD và CHV (Cross-Dataset Leakage)
Qua thuật toán quét băm MD5 chéo giữa các thư mục:
- `Dataset/GDUT-HWD` (Jie et al.): 3,160 ảnh độc lập.
- `Dataset/CHV_dataset` (Wang et al., Sensors 2021): 1,330 ảnh độc lập.
- **Phát hiện pháp y**: **Chính xác 770 ảnh của bộ CHV có mã băm MD5 trùng khớp 100% với ảnh trong GDUT-HWD**.
  * CHV thực chất kế thừa 770 ảnh từ GDUT-HWD và bổ sung 560 ảnh mới từ Cranfield University.
  * **Xung đột nhãn nghiêm trọng**: Trong GDUT-HWD gốc, các tác giả chỉ gán nhãn 5 lớp mũ (`blue, red, white, yellow, none`) và **hoàn toàn bỏ quên nhãn áo phản quang (`safety_vest = 0`)** dù công nhân trong ảnh có mặc áo. Trong khi đó, nhóm tác giả CHV đã gán nhãn lại cả mũ và áo phản quang cho 770 ảnh này.
  * **Hậu quả nếu ghép thô không qua xử lý**: 
    1. Nếu Train lấy GDUT và Test lấy CHV (hoặc ngược lại), tập Test sẽ bị rò rỉ tới **57.9%** dữ liệu huấn luyện.
    2. Các nhãn áo phản quang không được gán trong GDUT sẽ bị coi là False Positives, bóp méo chỉ số Vest AP50 tụt dốc xuống ~51%.

---

### 2.3. Bóc trần Hiện tượng Bão hòa Ảo 98% (Circular Pseudo-Labeling Effect)
Tại sao ở lần chạy trước, cả 6 thế hệ YOLO (v8s, v9s, v10s, 11s, 25s, 26s) và các module ablation A0-A6 đều đạt mAP@50 từ **97.5% đến 98.65%**?
Thực tế kiểm tra mã nguồn tại `scripts/refine_unified_gold_dataset.py` đã làm sáng tỏ:
1. **Cơ chế nắn nhãn tự động**: Tọa độ các hộp trong tập Test đã bị làm mịn theo dự đoán của mô hình YOLO giáo viên ($0.72 \times B_{\text{pred}} + 0.28 \times B_{\text{gt}}$).
2. **Tự động điền nhãn thiếu vào Test set**: Các nhãn `person` và `safety_vest` còn thiếu ở tập IITU Flycam đã được mô hình Teacher dự đoán và ghi thẳng vào tệp nhãn của tập Test (`test/labels/*.txt`).
3. **Bản chất Circular Evaluation**: Khi các mô hình học sinh (YOLO baseline hoặc Rep-YOLO11s) thực hiện suy luận trên tập Test, chúng dự đoán lại đúng thiên vị quy nạp (inductive bias) của Teacher Model. Mô hình đang "tự chấm bài thi do chính mình soạn đề".
4. **Hiệu ứng tương quan chuỗi video Drone**: Tập dữ liệu IITU được ghi hình bằng Drone theo từng chuyến bay. Các frame trong tập Train và Test cách nhau chỉ vài giây tại cùng một góc bay, dẫn đến việc mạng nơ-ron nhận diện công nhân cực kỳ dễ dàng (AP50 của green helmet đạt 99.5%).

$\implies$ **Quyết định bác bỏ**: Tuyệt đối **KHÔNG SỬ DỤNG** số liệu 98% từ tập pseudo-labeled này trong báo cáo đồ án chính thức và bài báo khoa học.

---

### 2.4. Bóc trần 4 Lỗi Chí Mạng của Tập `Dataset/GDUT-custome.v1i.yolov11`
Qua quét nhị phân 11,735 tệp ảnh trong thư mục này:
1. **110 ảnh AI sinh từ ChatGPT (DALL-E 3)**: Mang phong cách hoạt họa / ảo giác quang học (tệp `A-nh-ChatGPT-*-thg-9-2026`), vi phạm nghiêm trọng tính chân thực của dữ liệu thực nghiệm IEEE.
2. **4,816 ảnh (41%) trùng lặp chéo** với GDUT và IITU.
3. **Thư mục `valid/labels` rỗng (0 tệp)**: Làm tiến trình huấn luyện YOLO bị sập (runtime crash).
4. **Xóa sổ 100% nhãn áo và người** trong 6,804 ảnh Roboflow, chỉ giữ lại mũ vàng, phá hủy khả năng học nhận diện áo bảo hộ.

---

## 3. PHÁN QUYẾT REVIEWER & MA TRẬN CHỌN LỰA TẬP DỮ LIỆU CHUẨN

Đóng vai trò Reviewer khắt khe, chúng tôi phân định rõ ràng 3 cấp độ dữ liệu:

```
+-----------------------------------------------------------------------------------+
|                            MA TRẬN DỮ LIỆU LIÊM CHÍNH                             |
+-----------------------------------------------------------------------------------+
| 1. BENCHMARK CHÍNH THỨC SỐ 1 (GOLD STANDARD - 100% HUMAN LABELED):                |
|    Dataset: CHV_6CLASS_OFFICIAL_CLEAN (1,330 ảnh) / CHV_7CLASS_CONFLICT_FREE      |
|    - 100% Nhãn do con người gán thủ công (Sensors 2021).                          |
|    - 0% Circular Pseudo-labeling, 0% Data Leakage.                                |
|    - Baseline YOLO11s: mAP50 = 90.56% | Proposed Rep-YOLO11s: Precision = 92.66%  |
|    - Ý nghĩa: Minh chứng giảm báo động giả (+2.60% Precision), Red Helmet +4.69%  |
+-----------------------------------------------------------------------------------+
| 2. BENCHMARK ĐA NGUỒN CÓ KIỂM SOÁT (MULTI-SOURCE HARMONIZED):                     |
|    Dataset: UNIFIED_PPE_MASTER_HARMONIZED (5,359 ảnh, 28,203 boxes)               |
|    - Ghép nối nhãn vi phạm 'none' của GDUT với nhãn áo CHV qua thumbnail MSE.     |
|    - Loại bỏ hoàn toàn 770 ảnh giao thoa rò rỉ.                                   |
|    - Chỉ sử dụng khi không áp dụng pseudo-label smoothing trên tập test.          |
+-----------------------------------------------------------------------------------+
| 3. TẬP DỮ LIỆU BỊ CẤM / BÁC BỎ (REJECTED FROM FORMAL EVALUATION):                 |
|    - GDUT-custome.v1i.yolov11 (Dính ảnh DALL-E, mất nhãn áo, rỗng valid).        |
|    - Mọi tập test đã qua tinh chỉnh nhãn bởi Teacher Model (Circular Smoothing).  |
+-----------------------------------------------------------------------------------+
```

---

## 4. CHIẾN LƯỢC TRIỂN KHAI CCTV BIÊN HIỆN TẠI (DECOUPLED ARCHITECTURE)

Để đáp ứng yêu cầu của Thầy về việc xây dựng hệ thống mô phỏng CCTV, Docker và cảnh báo Zalo/Telegram ngay trong khi chờ đợt huấn luyện lại:
1. **Trọng số triển khai hiện tại**:
   - Nạp trọng số tốt nhất đã đạt kiểm định từ ablation sạch:  
     `Output/kaggle_results/hannhu40022_warmstart_ablations/rep_yolo11s_chv7_best_deploy.pt`
   - Mô hình chạy FP16 CUDA trên GPU RTX 3050 Laptop, đạt độ trễ suy luận **17.7 - 21.7 ms** (~33.8 FPS mượt mà trên video 1080p).
2. **Kiến trúc Phân rã Hoàn toàn (Decoupled Design)**:
   - Toàn bộ đường dẫn mô hình, ngưỡng nhận diện, tham số camera, và bot token được quản lý qua biến môi trường (`.env` / `src/config.py`).
   - Khi có kết quả của mô hình mới nhất (không dính pseudo-labeling), chỉ cần sửa 1 dòng `MODEL_PATH=...` trong `.env`, hệ thống Web và Docker sẽ tự động nạp trọng số mới mà **KHÔNG CẦN VIẾT LẠI CODE**.
3. **Khắc phục lỗi phát hiện nhãn phủ định (Negative Class Trap)**:
   - Các nhãn `no_helmet`, `none`, `no_vest` được cô lập riêng biệt. Nếu phát hiện công nhân không đội mũ hoặc không mặc áo, hệ thống ngay lập tức cưỡng chế trạng thái `has_helmet = False` / `has_vest = False`, ngăn chặn hoàn toàn việc nhận diện nhầm thành có bảo hộ.
4. **Bổ sung tính năng cắt chân dung đối tượng vi phạm (Person Crop)**:
   - Khi xác nhận vi phạm liên tục qua bộ lọc Debounce ($N \ge 8$ khung hình), hệ thống trích xuất vùng cắt công nhân (`frame[y1:y2, x1:x2]`), phóng to và nhúng trực tiếp dưới dạng khung Picture-in-Picture (PiP) nổi bật ở góc phải trên của ảnh snapshot, đồng thời lưu trữ tệp crop riêng biệt vào `snapshots/crop_worker_*.jpg`.
   - Cảnh báo gửi về Telegram Bot (`@safety_helmet_cctv_bot`) và Zalo Bot (`Bot safety helmet CCTV`) có ảnh snapshot rõ nét kèm chân dung phóng to, giúp cán bộ an toàn nhận diện ngay tức khắc trên điện thoại di động.

---

## 5. KẾT LUẬN & CAM KẾT LIÊM CHÍNH
Toàn bộ quy trình kiểm toán pháp y đã được tài liệu hóa minh bạch. Dự án cam kết bảo vệ kết quả nghiên cứu trước Hội đồng Đánh giá Đồ án và các Reviewer quốc tế bằng phương pháp luận trung thực, dữ liệu sạch 100% và mã nguồn chuẩn mực công nghiệp.
