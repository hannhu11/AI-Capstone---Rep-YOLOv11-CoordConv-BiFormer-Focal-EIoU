# 🏛️ BÁO CÁO GIẢI TRÌNH & BẢO VỆ ĐỒ ÁN TỐT NGHIỆP KỸ SƯ AI — GIAI ĐOẠN 1 (TRƯỜNG ĐẠI HỌC FPT)
## ĐỀ TÀI: Rep-YOLO11s: Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance

**Nhóm thực hiện:** Nhu Han (Lead Researcher, SE183644), Van-Thanh Nguyen, Tuan-Dung Nguyen  
**Giảng viên Hướng dẫn:** ThS. Vũ Hà Anh (anhvh@fe.edu.vn)  
**Hội đồng Giảng viên Đánh giá:** ThS. Nguyễn Trọng Tài, ThS. Lê Phú Nguyên, ThS. Nguyễn Quốc Trung, ThS. Nguyễn Hồng Hải, TS. Nguyễn Xuân Huy  
**Tài liệu tham chiếu chuẩn:** `Rep-YOLO11s_Master_Paper_Final.pdf` (Định dạng bài báo khoa học chuẩn Q1 quốc tế)

---

## 📋 BẢNG TỔNG HỢP CÂU HỎI VÀ TRẢ LỜI ĐÁP ỨNG TIÊU CHÍ GIAI ĐOẠN 1

### TIÊU CHÍ 1: MỤC TIÊU CỦA ĐỀ TÀI (PROBLEM STATEMENT)
#### ❓ Câu hỏi của Hội đồng: *Phát biểu bài toán có đủ rõ ràng hay không?*

**Trả lời đáp ứng chuẩn học thuật & thực tiễn:**
* **Bối cảnh & Tầm quan trọng**: Tại các đại công trường xây dựng, tai nạn chấn thương sọ não do vật thể rơi từ trên cao chiếm tỷ lệ tử vong hàng đầu. Việc giám sát thủ công bằng bảo vệ công trường có nhược điểm chí mạng: không liên tục, bị hạn chế tầm nhìn và dễ bỏ sót vi phạm. Do đó, bài toán đặt ra là **xây dựng hệ thống thị giác máy tính tự động giám sát tuân thủ an toàn lao động (đội mũ bảo hộ - Safety Helmet Wearing Detection) thông qua luồng video CCTV/RTSP thời gian thực**.
* **4 Nút thắt Kỹ thuật Cốt lõi (Core Technical Challenges)** được định nghĩa tường minh:
  1. **Vật thể siêu nhỏ ở cự ly xa (Distant Tiny Objects)**: Camera CCTV gắn trên cao (cột đèn, cẩu tháp 15–30m) khiến kích thước mũ bảo hộ bị co rút dưới $20 \times 20$ pixels, thậm chí $<15 \times 15$ pixels. Qua các tầng tích chập sâu (stride 16, 32), đặc trưng mũ bảo hộ bị tiêu biến hoàn toàn (Spatial Feature Vanishing).
  2. **Mất cân bằng dữ liệu cực đoan nội khung (Extreme Class Imbalance)**: Trong tập dữ liệu chuẩn công nghiệp SHWD (VOC2028), tỷ lệ số lượng hộp đối tượng là $9,044$ mũ (`hat`) so với $111,514$ thân người (`person`), tương đương tỷ lệ lệch tới **$1 : 12$**. Hàm mất mát thông thường bị chi phối bởi các mẫu background và thân người dễ học, bỏ quên mũ nhỏ.
  3. **Nhiễu không gian và vật thể gây nhầm lẫn thị giác (Spatial Context & Color Clutter)**: Công trường xây dựng chứa đầy các vật thể có màu vàng, cam, tròn tương tự mũ bảo hộ: xô nhựa vàng, cọc tiêu giao thông, đèn cao áp ban đêm, biển cảnh báo nguy hiểm. Tích chập thông thường có tính bất biến tịnh tiến (Translation Invariance), không phân biệt được "màu vàng ở trên đầu người" (mũ thật) với "màu vàng đặt dưới mặt đất" (xô vữa/cọc tiêu).
  4. **Rào cản Phần cứng Biên Thời gian thực (Strict Latency & Edge Deployment Bottlenecks)**: Hệ thống giám sát thực tế yêu cầu xử lý đa luồng camera ($60-90$ FPS) trên thiết bị biên chi phí thấp (Edge AI / Laptop công trường / Jetson). Các kiến trúc Transformer nặng nề gây nghẽn băng thông bộ nhớ (Memory Access Cost - MAC), không thể ứng dụng thực tế.
* **Mục tiêu lượng hóa cụ thể**: Đạt độ chính xác $mAP_{50} \ge 94.5\%$, $mAP_{50-95} \ge 62.0\%$, tốc độ xử lý video thực tế $\ge 60$ FPS trên GPU biên và $\ge 24$ FPS trên thiết bị giá rẻ tối thiểu (2GB VRAM).

---

### TIÊU CHÍ 2: KẾT QUẢ CUỐI CÙNG & SẢN PHẨM MONG MUỐN (PROPOSED SOLUTION)
#### ❓ Câu hỏi của Hội đồng: *Kết quả cuối cùng là gì? Kết quả mong muốn phải đạt được của đề tài là gì?*

Nhóm cam kết bàn giao trọn vẹn **5 sản phẩm đầu ra (Deliverables)** đạt tiêu chuẩn cao nhất của một Đồ án Tốt nghiệp ngành TTNT:

1. **Báo cáo Khoa học Hoàn chỉnh (Scientific Research Paper)**:
   - Bài báo khoa học chuẩn quốc tế Q1 dài **9 trang**, đầy đủ công thức toán học, chứng minh giải phẫu bệnh lý, so sánh SOTA, Ablation Study $A_0 \to A_6$, bản đồ nhiệt Grad-CAM XAI và Benchmark đa miền dữ liệu.
   - Tệp nguồn LaTeX Overleaf chuẩn mực kèm bộ trích dẫn 32 tài liệu tham khảo chất lượng cao (BibTeX).
2. **Mô hình Trí tuệ Nhân tạo Mới (Novel Deep Learning Model)**:
   - Mô hình mạng **Rep-YOLO11s** với trọng số tối ưu (`yolo11s_best.pt`), đạt $mAP_{50} = 94.83\%$ (Test cố định), $mAP_{50} = 96.64 \pm 0.32\%$ (5-Fold CV), và $97.03\%$ trên tập ngoại miền Hard Hat Workers.
3. **Software Prototype / Ứng dụng Giám sát Hoàn chỉnh (Industrial Video Analytics System)**:
   - Pipeline xử lý luồng video RTSP/CCTV thời gian thực khép kín: Tích hợp giải mã phần cứng H.264/H.265 $\to$ Tiền xử lý CoordConv $\to$ Suy luận TensorRT FP16 $\to$ Hậu xử lý NMS $\to$ Giao diện cảnh báo vi phạm trực quan với tốc độ đạt **$65 - 95$ FPS**.
4. **Bộ Động cơ Biên dịch Triển khai Tối ưu (Engine Optimization Artifacts)**:
   - Bộ trọng số chuyển đổi TensorRT 11.2 FP16 (`.engine`), ONNX Runtime INT8 (`.onnx`), và OpenVINO phục vụ triển khai zero-overhead trên NVIDIA Tesla T4, RTX 3050 Laptop, và CPU Edge.
5. **Bộ Dữ liệu Chuẩn hóa Công nghiệp (Standardized Multi-Domain PPE Datasets)**:
   - Đã làm sạch và chuẩn hóa nhãn thống nhất $\mathcal{C}^*$ cho hơn $33,000$ ảnh công trường thực tế (VOC2028: $7,581$ ảnh, GDUT-HWD: $13,499$ ảnh, SHEL5K: $5,000$ ảnh, Hard Hat Workers: $7,000$ ảnh).

---

### TIÊU CHÍ 3: TÍNH KHẢ THI & XÁC ĐỊNH PHẠM VI (FEASIBILITY & SCOPE)

#### ❓ 1. Dựa trên hướng tiếp cận nào; phương pháp/thuật toán/mô hình nào? Số lượng tìm hiểu là bao nhiêu?
* **Hướng tiếp cận**: Single-Stage Object Detection kết hợp Nguyên lý Tái tham số hóa Cấu trúc (Structural Re-parameterization), Cơ chế Chú ý Định tuyến Thưa (Sparse Routing Attention), và Mã hóa Tọa độ Không gian Tuyệt đối (Spatial Coordinate Injection).
* **Số lượng công trình nghiên cứu đã khảo sát chuyên sâu**: **32 bài báo khoa học đỉnh cao (2019 – 2026)** về phát hiện thiết bị bảo hộ lao động và kiến trúc thị giác máy tính biên (quản lý đồng bộ qua tệp `Capstone_AI_Papers.bib` và Zotero).
* **Các kiến trúc nền tảng và SOTA đối sánh trực tiếp**: 
  - Mô hình nền tảng: YOLOv8 (v8n, v8s), YOLOv10 (v10n, v10s), YOLO11 (11n, 11s).
  - Các công trình chuyên sâu về PPE: EC-YOLOv8 (Zhang et al., 2024), YOLO-CBF (Li et al., 2023), YOLOv8n-FADS (Fu et al., 2024).

#### ❓ 2. Dự kiến triển khai từ đầu hay tái sử dụng lại API/Framework/Platform có sẵn?
* Nhóm **kết hợp tối ưu hóa hai tầng**:
  - **Tái sử dụng có chọn lọc**: Kế thừa framework huấn luyện mã nguồn mở Ultralytics YOLO11, thư viện gia tốc phần cứng PyTorch 2.5.1, CUDA 12.4, TensorRT 11.2, và OpenCV VideoCapture/NVDEC cho luồng RTSP.
  - **Tự phát triển và tích hợp từ đầu (From Scratch Architecture Modifications)**: Tự thiết kế và cài đặt mã nguồn các khối tùy biến:
    1. Lớp sinh tọa độ không gian `CoordConv` trong Tensor đầu vào.
    2. Khối tích chập tái tham số hóa `RepConv` 3 nhánh (Conv 3x3, Conv 1x1, Identity) kèm thuật toán gộp nhân `switch_to_deploy` suy biến đại số tuyến tính về 1 nhân $3\times3$ duy nhất.
    3. Khối chú ý định tuyến thưa `BiFormer` (Bi-Level Routing Attention) với ma trận độ tương đồng vùng thô và chọn top-$k$.
    4. Hàm mất mát hồi quy bounding box `Focal EIoU Loss` có phân rã sai số chiều dài/chiều rộng và trọng số mũ cứng.

#### ❓ 3. Có cải tiến hay đưa ra mô hình, thuật toán mới không? Số lượng là bao nhiêu?
* Nhóm đề xuất **4 cải tiến kỹ thuật cốt lõi (4 Novel Technical Contributions)** đã được chứng minh định lượng:
  1. **Structural Re-parameterization (RepConv)**: Giải quyết mâu thuẫn giữa độ chính xác khi train và độ trễ khi chạy. Khi train mô hình có 3 nhánh đa dạng đặc trưng, khi deploy gộp toán học thành 1 nhánh duy nhất $\to$ Giảm độ trễ suy luận từ $7.12$ ms xuống còn **$2.92$ ms** trên Tesla T4 ($5.35$ ms trên RTX 3050), overhead độ trễ bằng 0!
  2. **Spatial Coordinate Injection (CoordConv)**: Bổ sung 2 kênh tọa độ chuẩn hóa $(x, y) \in [-1, 1]$ phá vỡ tính bất biến tịnh tiến của CNN, cung cấp tiên đề không gian giúp mạng loại bỏ các dương tính giả từ xô vữa vàng, đèn cao áp, cọc tiêu dưới mặt đất.
  3. **Bi-Level Routing Attention (BiFormer)**: Lọc bỏ các vùng phông nền vô nghĩa ở mức thô $S \times S$, dồn 100% tài nguyên tính toán vào các vùng ứng viên chứa mũ bảo hộ siêu nhỏ, giảm độ phức tạp từ bậc 2 $\mathcal{O}(H^2W^2)$ xuống mức tuyến tính thân thiện với thiết bị biên.
  4. **Focal EIoU Bounding Box Loss**: Thay thế CIoU truyền thống bằng cách phân rã trực tiếp sai lệch cạnh $\frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$ và nhân trọng số Focal $\text{IoU}^\gamma$, giúp định vị chính xác mép vành mũ trong các trường hợp bị che khuất một phần.

#### ❓ 4. Tập dữ liệu mẫu tự xây dựng hay có sẵn? Có thay đổi gì trên tập dữ liệu có sẵn hay không?
* **Tập nguồn chính**: Sử dụng tập **Safety Helmet Wearing Dataset (SHWD / VOC2028)** gồm **$7,581$ ảnh** công trường thực tế.
* **Các thay đổi và làm sạch bắt buộc đã thực hiện**:
  1. Phát hiện và loại bỏ các nhãn rác (Label Noise) có sẵn trong XML gốc (ví dụ: 3 nhãn dị biệt "dog" trong tệp VOC).
  2. Chuẩn hóa chuyển đổi định dạng từ Pascal VOC XML sang ma trận nhãn YOLO chuẩn tắc tọa độ tâm $[x_{center}, y_{center}, w, h] \in [0, 1]$.
  3. Phân chia tập huấn luyện nghiêm ngặt: $80\%$ Train ($6,064$ ảnh) và $20\%$ Validation/Test ($1,517$ ảnh) cố định, loại bỏ hoàn toàn rò rỉ dữ liệu (Data Leakage).
* **Mở rộng kiểm thử ngoại vi đa miền (Cross-Domain Benchmark - Hơn 25,000 ảnh)**:
  - Tích hợp thêm **5 bộ dữ liệu công trường độc lập**: GDUT-HWD ($13,499$ ảnh), SHEL5K ($5,000$ ảnh), Hard Hat Workers ($7,000$ ảnh), Safety Helmet Detection (SHD), và SFCHD.
  - Chuẩn hóa toàn bộ về không gian nhãn chung $\mathcal{C}^* = \{0: \text{'hat'}, 1: \text{'person'}\}$.
  - Giải quyết bài toán xung đột quy chuẩn gán nhãn: Nhãn `person` ở SHWD là Full-Body trong khi ở Hard Hat Workers là Head-Only, dẫn tới hiện tượng **Sụp đổ IoU (IoU Collapse, IoU $\approx 0.07-0.14 \ll 0.50$)**. Nhóm đã đề xuất giao thức **Harmonized PPE (Hat-Only)** giúp đánh giá đúng bản chất mũ bảo hộ, đạt điểm số chuyển giao ngoại miền xuất sắc **$97.03\%$**.

---

### TIÊU CHÍ 4: GIÁ TRỊ CỦA ĐỀ TÀI (VALUE & IMPACT)

#### ❓ 1. Đề tài có tính thực tế không? Có thể vận dụng được trong thực tế hay không?
* **Tính thực tế cực kỳ cao**:
  - Hệ thống được thiết kế đo đạc trực tiếp trên luồng camera giám sát công trường thật (`construction_site_workers_1080p.mp4`).
  - Đạt thông lượng xử lý luồng video hoàn chỉnh (bao gồm giải mã RTSP, tiền xử lý CoordConv, suy luận TensorRT, NMS và hiển thị giao diện cảnh báo) đạt **$65 - 95$ FPS** trên RTX 3050 Laptop, vượt xa ngưỡng yêu cầu thời gian thực của camera công trường chuẩn ($25 - 30$ FPS).
  - **Khả năng phổ cập hóa trên thiết bị cấu hình siêu yếu**: Thử nghiệm thực tế trên Laptop văn phòng phổ thông trang bị GPU **NVIDIA GeForce MX230 (kiến trúc Pascal, chỉ có 2GB VRAM, không có nhân Tensor Cores)** chạy PyTorch Native FP32 vẫn đạt **$27.8$ FPS ($36.0$ ms)**. Điều này chứng minh giải pháp hoàn toàn có thể chạy trên máy tính giá rẻ tại các lán trại chỉ huy công trường mà không cần đầu tư máy chủ đắt đỏ.

#### ❓ 2. Đề tài có ý nghĩa khoa học không?
* Đề tài giải quyết triệt để bài toán khoa học về **sự đánh đổi giữa Độ chính xác (Accuracy) và Tốc độ thực thi (Latency)** trên thiết bị biên.
* Đưa ra lời giải toán học sáng tỏ cho 2 hiện tượng gây tranh cãi trong cộng đồng nghiên cứu:
  1. *Giải thích nghịch lý mAP giữa Single Run ($94.83\%$) và 5-Fold Partition CV ($96.64\%$)* thông qua phương trình kỳ vọng hỗn hợp có trọng số:
     $$\mathbb{E}[mAP_{50}] = 0.80 \times 97.10\% + 0.20 \times 94.83\% = 96.65\% \approx 96.64\%$$
  2. *Giải thích hiện tượng sụp đổ IoU khi chuyển miền dữ liệu* và xây dựng chuẩn đánh giá Harmonized PPE Protocol.
* Được chứng minh trực quan bằng bản đồ nhiệt giải thích học sâu **Grad-CAM XAI**: Chứng minh các khối CoordConv và BiFormer triệt tiêu hiện tượng phân tán chú ý vào áo bảo hộ dạ quang hay biển báo giả mạo, tập trung gradient vào đúng mũ công nhân.

#### ❓ 3. Đề tài có mới? Có sáng tạo không?
* **Tính mới**: Đây là công trình đầu tiên kết hợp đồng thời 4 cơ chế: Structural Re-parameterization (RepConv), Coordinate Encoding (CoordConv), Bi-Level Routing Attention (BiFormer), và Focal EIoU Loss vào họ mô hình mới nhất YOLO11.
* **Tính sáng tạo**:
  - Không tăng tham số hay độ trễ khi suy luận nhờ cơ chế đại số RepConv $3\times3 + 1\times1 + Identity \to 3\times3$ Conv.
  - Vượt qua tất cả các mô hình SOTA hiện hữu (YOLOv8s, YOLOv10s, YOLO11s, EC-YOLOv8, YOLO-CBF) cả về $mAP_{50}$ lẫn tốc độ suy luận thuần GPU ($2.92$ ms trên T4, nhanh gấp đôi so với các mô hình cùng kích thước).
