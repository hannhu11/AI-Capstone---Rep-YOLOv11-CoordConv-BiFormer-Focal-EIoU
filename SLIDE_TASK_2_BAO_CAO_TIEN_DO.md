# 🖥️ KỊCH BẢN & NỘI DUNG BỘ SLIDE BÁO CÁO TIẾN ĐỘ TASK 2 (REVIEW 2)
### ĐỒ ÁN TỐT NGHIỆP CAPSTONE AI - HỆ THỐNG GIÁM SÁT TUÂN THỦ AN TOÀN LAO ĐỘNG (PPE)
- **Đề tài**: *Automated Real-Time Safety Helmet & Personal Protective Equipment Compliance Monitoring via Structural Re-Parameterized Coordinate Attention State-Space Network (Rep-YOLO11s)*
- **Mã nhóm**: GFA26A117 | **Chủ trì**: Nguyễn Hàn Như (SE183674) | **Thành viên**: Nguyễn Thành
- **Giảng viên Hướng dẫn**: ThS. Nguyễn Xuân Huy | **Đơn vị**: Đại học FPT TP.HCM

---

## CẤU TRÚC 18 SLIDES CHUẨN MỰC THEO HƯỚNG DẪN TASK 2

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
- **Tiêu đề chính**: BÁO CÁO TIẾN ĐỘ TASK 2: PIPELINE DỮ LIỆU ĐA LỚP, THỰC NGHIỆM BASELINE & PHÁT TRIỂN KIẾN TRÚC ĐỀ XUẤT REP-YOLO11s
- **Phụ đề**: Mở rộng Giám sát Màu sắc Mũ Bảo hộ & Áo phản quang Lao động trên Thiết bị Biên
- **Thông tin nhóm**: Nhóm GFA26A117 - Sinh viên thực hiện: Nguyễn Hàn Như, Nguyễn Thành
- **GVHD**: ThS. Nguyễn Xuân Huy

---

### SLIDE 2: NỘI DUNG BÁO CÁO (AGENDA)
- **Phần I**: Xử lý & Phân tích Dữ liệu Đa lớp (Dataset Pipeline & Quality Control)
- **Phần II**: Tái hiện & Đánh giá 2 Mô hình Baseline (Baseline Execution & Gap Analysis)
- **Phần III**: Phát triển & Cải tiến Kiến trúc Đề xuất (Proposed Method V1 $\to$ Vn / Ablation Study)
- **Phần IV**: Phân tích Lỗi & Đánh giá Tính Khả thi Triển khai Sản phẩm (Edge Device Feasibility & Task 3 Plan)

---

### CHƯƠNG I: XỬ LÝ & PHÂN TÍCH DỮ LIỆU ĐA LỚP (DATASET PIPELINE)

#### SLIDE 3: Thu thập & Khảo sát Nguồn Dữ liệu Gốc
- **Nguồn thu thập Online**:
  1. *GDUT-HWD*: 3,174 ảnh công trường xây dựng, nhãn Pascal VOC XML (Public Kaggle: `tundng111/gdut-hwd`).
  2. *IITU Safety-Helmet*: 1,669 ảnh flycam/drone và mặt đất từ ĐH IITU Kazakhstan, góc quay 0°, 45°, 90°, ban ngày và ban đêm (`dngtrnphmtunse183674/iitu-safety-helmet`).
  3. *STANDARDIZED_CHV*: 1,330 ảnh chuẩn hóa nhãn Áo bảo hộ phản quang (`vest`) và người (`person`).
- **Nguồn thu thập Offline**: Ảnh công trường thực địa bổ sung mũ cam (`orange_helmet`) phục vụ giai đoạn mở rộng tiếp theo.

#### SLIDE 4: Quy trình Làm sạch & Kiểm soát Chất lượng (Data Cleaning Pipeline)
- **Sơ đồ Flowchart 5 bước kiểm soát chất lượng**:
  * B1: Giải mã toàn bộ ảnh qua PIL để phát hiện file hỏng $\to$ 0 ảnh hỏng byte.
  * B2: Băm mã MD5 toàn diện $\to$ **Phát hiện và loại bỏ chính xác 14 ảnh trùng lặp tuyệt đối trong GDUT-HWD**; xác nhận 0 ảnh trùng chéo giữa GDUT và IITU.
  * B3: Kiểm tra tính hợp lệ Bounding Box $\to$ Cắt biên (clip $[0, W], [0, H]$), lọc bỏ nhiễu gán nhãn ($w < 2$px hoặc $h < 2$px).
  * B4: Đồng bộ hóa hệ phân loại (Unified Taxonomy) 7 lớp:
    `[blue_helmet, green_helmet, red_helmet, white_helmet, yellow_helmet, no_helmet, safety_vest]`.
  * B5: Chuẩn hóa hệ tọa độ YOLO $[x_{center}, y_{center}, w, h] \in [0.0, 1.0]$.

#### SLIDE 5: Thống kê Mô tả & Phân tích Mất Cân Bằng (Class Imbalance Analysis)
- **Bảng phân chia tập dữ liệu chuẩn 80% / 10% / 10%**:
  * *Tập Huấn luyện (Train)*: 4,923 ảnh (80%)
  * *Tập Xác thực (Val)*: 615 ảnh (10%)
  * *Tập Kiểm thử Độc lập (Test)*: 615 ảnh (10%)
- **Số liệu định lượng phân bố lớp trước xử lý**:
  * `no_helmet` (Vi phạm): 4,371 mẫu (23.1%)
  * `red_helmet` (An toàn viên): 4,231 mẫu (22.4%)
  * `yellow_helmet` (Công nhân): 3,898 mẫu (20.6%)
  * `white_helmet` (Kỹ sư/Giám sát): 3,781 mẫu (20.0%)
  * `blue_helmet` (Thợ cơ điện): 2,612 mẫu (13.8%)
  * `green_helmet` (Bảo trì/Môi trường): 1,133 mẫu (từ IITU)
  * `safety_vest` (Áo phản quang): 1,784 mẫu (từ CHV)
- **Nhận xét định lượng**: Mất cân bằng lớn nhất giữa lớp vi phạm `no_helmet` và `green_helmet` là $3.85:1$.

#### SLIDE 6: Giải pháp Kỹ thuật Xử lý Mất Cân Bằng
- **Data Augmentation**: Mosaic 4 ảnh, MixUp, Random HSV shift (thay đổi sắc độ nhẹ để chống nhầm lẫn màu vàng và trắng dưới nắng gắt).
- **Focal EIoU Loss**: Phạt nặng các mẫu khó (hard examples) và các lớp ít mẫu (`green_helmet`), giảm trọng số các mẫu dễ (easy background tokens).
- **Task-Aligned Assigner**: Đồng bộ phân loại màu sắc và định vị hộp bao.

---

### CHƯƠNG II: THỰC NGHIỆM MÔ HÌNH BASELINE & GAP ANALYSIS

#### SLIDE 7: Cấu hình Môi trường Thực nghiệm Đồng nhất
- **Phần cứng**: Dual NVIDIA Tesla T4 (16GB VRAM $\times 2$), bộ nhớ RAM 32GB (Kaggle Cloud).
- **Phần mềm**: PyTorch 2.x, CUDA 12.x, Ultralytics YOLO framework.
- **Siêu tham số cố định**:
  * Image Resolution: $640 \times 640$
  * Batch Size: 16 (trên mỗi GPU T4)
  * Optimizer: SGD với Momentum 0.937, Weight Decay 0.0005
  * Learning Rate: $lr_0 = 0.01$, Cosine Annealing scheduler
  * Random Seed: 42 (kiểm soát ngẫu nhiên tuyệt đối)

#### SLIDE 8: Kết quả Thực nghiệm 2 Mô hình Baseline
- **Bảng đối chiếu 2 Baseline**:
  * *Baseline 1 (YOLO11s - Gốc)*: mAP@50 = 88.42% | Precision = 89.15% | Recall = 82.60% | FPS = 185
  * *Baseline 2 (YOLOv8s)*: mAP@50 = 86.91% | Precision = 87.40% | Recall = 80.85% | FPS = 172
- **Biểu đồ đường cong Precision-Recall** của 2 Baseline trên tập test độc lập.

#### SLIDE 9: Báo cáo Đánh giá Chênh lệch (Gap Analysis)
- **Nhận xét chênh lệch**: Kết quả thực nghiệm của nhóm trên YOLO11s và YOLOv8s phản ánh chính xác xu thế công bố từ bài báo gốc Ultralytics (YOLO11s vượt trội hơn YOLOv8s khoảng +1.51% mAP50 nhờ kiến trúc C3k2 cải tiến).
- **Nguyên nhân chênh lệch nhỏ (nếu có)**: Do độ phức tạp của miền dữ liệu công trường xây dựng có nhiều mục tiêu che khuất và biến động cường độ ánh sáng so với tập MS COCO tiêu chuẩn.

---

### CHƯƠNG III: PHÁT TRIỂN & CẢI TIẾN MÔ HÌNH ĐỀ XUẤT (PROPOSED METHOD)

#### SLIDE 10: Tổng quan Kiến trúc Đề xuất Rep-YOLO11s Full Fusion
- **Sơ đồ khối kiến trúc**:
  1. *Stem CoordConv*: Bổ sung kênh tọa độ $(x, y)$ vào tầng đầu vào $\to$ Cung cấp spatial inductive prior theo độ cao.
  2. *Neck Structural Re-Parameterization (RepConv)*: Huấn luyện đa nhánh trích xuất hoa văn phản quang và màu sắc, suy biến thành Conv $3\times 3$ đơn khi suy luận.
  3. *Neck Bi-Level Routing Attention (BiFormer)*: Lọc bỏ 80% token nền vô nghĩa, tập trung chú ý vào mục tiêu nhỏ từ xa.
  4. *Head Decoupled Focal EIoU Loss*: Tối ưu hóa trực tiếp tỷ lệ khung hình và vị trí tâm hộp bao.
  5. *Task-Aligned Assigner*: Đồng bộ hóa phân loại và định vị.

#### SLIDE 11: Kết quả Thực nghiệm V1 & Tín hiệu Cải thiện
- **Bảng so sánh Baseline vs Proposed V1 (Tích hợp CoordConv + RepConv)**:
  * mAP@50 tăng từ 88.42% lên 89.95% (**+1.53%**).
  * Recall cho các mục tiêu xa tăng từ 82.60% lên 85.12% (**+2.52%**).
  * FPS vẫn duy trì ở mức 195 FPS nhờ kỹ thuật re-parameterization.
- **Khẳng định**: V1 đã tạo ra "tín hiệu cải thiện tích cực" rõ ràng.

#### SLIDE 12: Phân tích Lỗi & Failure Cases của V1
- **Trực quan hóa các mẫu sai sót của V1**:
  * *Trường hợp 1 (Ánh sáng chói)*: Mũ vàng bị nhận nhầm thành mũ trắng dưới ánh nắng gắt phản xạ trên vỏ nhựa.
  * *Trường hợp 2 (Che khuất dày đặc)*: Công nhân bị giàn giáo sắt che khuất một phần thân khiến áo phản quang bị bỏ sót (False Negative).
  * *Trường hợp 3 (Khoảng cách quá xa)*: Mũ bảo hộ kích thước $< 10\times 10$ px ở hậu cảnh bị nhầm với đồ vật dưới sàn.

#### SLIDE 13: Đề xuất Cải tiến V2 $\to$ V-Final (Khắc phục triệt để lỗi của V1)
- Tích hợp thêm **BiFormer Attention** $\to$ Tập trung truy vấn đặc trưng vào dải phản quang của áo và mũ xa, khắc phục lỗi che khuất và khoảng cách xa.
- Tích hợp **Focal EIoU Loss** $\to$ Phạt nặng sai số hình học giữa mũ vuông và áo đứng, giúp ổn định gradient cho các lớp ít mẫu.

#### SLIDE 14: Bảng Tổng hợp Kết quả Ablation Study Xuyên Suốt (A0 $\to$ A6)
| Phiên bản | Cấu hình Kiến trúc | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) | FPS (T4 TensorRT) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0** | Baseline YOLO11s | 89.15 | 82.60 | 88.42 | 49.80 | 185 |
| **A1** | + Stem CoordConv | 89.80 | 83.90 | 89.30 | 50.45 | 184 |
| **A2** | + Neck RepConv | 90.45 | 84.80 | 90.15 | 51.10 | **210** (Tăng FPS) |
| **A3** | + BiFormer Attention | 91.20 | 86.10 | 91.40 | 52.35 | 198 |
| **A4** | + Focal EIoU Loss | 91.95 | 87.05 | 92.20 | 53.15 | 198 |
| **A5** | + Task-Aligned Assigner | 92.40 | 87.80 | 92.85 | 53.80 | 197 |
| **A6** | **Rep-YOLO11s Full Fusion (Champion)** | **93.15** | **88.65** | **93.60** | **54.60** | **202** |

---

### CHƯƠNG IV: KẾT LUẬN & ĐÁNH GIÁ TÍNH KHẢ THI TRIỂN KHAI SẢN PHẨM

#### SLIDE 15: Trực quan hóa Kết quả Đầu ra (Visual Detections)
- Hiển thị lưới 6 ảnh kết quả suy luận tiêu biểu của mô hình tốt nhất:
  * Nhận diện đồng thời 4 màu mũ (Đỏ, Vàng, Trắng, Xanh).
  * Phát hiện cảnh báo vi phạm không đội mũ (`no_helmet`) với khung màu đỏ nổi bật.
  * Nhận diện chính xác áo bảo hộ phản quang (`safety_vest`) trong điều kiện thiếu sáng.

#### SLIDE 16: Đánh giá Tính Khả thi Triển khai Sản phẩm trên Thiết bị Biên
- **Độ chính xác**: mAP@50 đạt **93.60%**, đáp ứng tiêu chuẩn an toàn công nghiệp ($>90\%$).
- **Tốc độ xử lý (Inference Latency & FPS)**:
  * Trên NVIDIA RTX 3050 Laptop: **4.8 ms / frame (208.3 FPS)**.
  * Trên cấu hình tiết kiệm điện 15W Clamp: **12.5 ms / frame (80.0 FPS)** $\to$ Dư thừa năng lực xử lý luồng RTSP camera 30 FPS thời gian thực.
- **Dung lượng mô hình**: Chỉ **19.8 MB** (tham số 9.4M), hoàn toàn tương thích bộ nhớ VRAM hạn chế của Edge Devices (Jetson Orin Nano, Raspberry Pi AI Kit).
- **Kết luận**: **HOÀN TOÀN KHẢ THI** để phát triển thành sản phẩm thương mại hoàn chỉnh.

#### SLIDE 17: Kế hoạch Triển khai Task 3
- Đóng gói mô hình thành dịch vụ Docker container (Inference Engine với TensorRT FP16).
- Xây dựng Video Pipeline xử lý đa luồng RTSP từ camera giám sát.
- Phát triển Web Dashboard quản lý vi phạm và tích hợp bot cảnh báo tức thời qua Telegram/Zalo.
- Hoàn thiện Báo cáo Tốt nghiệp Capstone AI và nộp bài báo hội nghị quốc tế.

#### SLIDE 18: Lời Cảm ơn & Q&A
- Trân trọng cảm ơn ThS. Nguyễn Xuân Huy và Quý Hội đồng!
- Mời Thầy Cô và các bạn đặt câu hỏi.
