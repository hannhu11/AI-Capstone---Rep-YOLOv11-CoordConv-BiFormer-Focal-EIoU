# 📊 KỊCH BẢN NỘI DUNG SLIDE CAPSTONE REVIEW 1 (SLIDE-BY-SLIDE DECK)
## ĐỒ ÁN TỐT NGHIỆP NGÀNH TRÍ TUỆ NHÂN TẠO - FPT UNIVERSITY
**Đề tài:** Rep-YOLO11s: Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance  
**Nguồn trích xuất:** 100% từ `Rep-YOLO11s_Master_Paper_IEEE_Final.pdf`

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề chính:** Rep-YOLO11s: Phát hiện Mũ Bảo hộ Thời gian thực trong Giám sát Công trường
* **Tiêu đề phụ:** Tái tham số hóa Cấu trúc, Mã hóa Tọa độ Không gian và Năng lực Tổng quát hóa Đa miền
* **Bối cảnh:** Báo cáo Đồ án Tốt nghiệp Ngành Trí tuệ Nhân tạo - Đợt Review 1
* **Giảng viên hướng dẫn:** ThS. Vũ Hà Anh
* **Thành viên nhóm:**
  - Nguyễn Hàn Như (Trưởng nhóm - SE183644)
  - Nguyễn Văn Thành
  - Nguyễn Tuấn Dũng
* **Đơn vị:** Khoa Công nghệ Thông tin, Đại học FPT Hà Nội
* **Mục tiêu Review 1:** Xác định tính rõ ràng của bài toán, phạm vi nghiên cứu, tính khả thi kiến trúc và giá trị khoa học/thực tiễn.

---

### SLIDE 2: ĐẶT VẤN ĐỀ & Ý NGHĨA THỰC TIỄN (PROBLEM BACKGROUND)
* **Bối cảnh thực tiễn:**
  - Tai nạn chấn thương sọ não do vật thể rơi chiếm tỷ lệ tử vong và thương tật vĩnh viễn cao nhất tại các công trường xây dựng công nghiệp.
  - Việc kiểm tra thủ công bằng bảo vệ/cán bộ an toàn lao động bộc lộ hạn chế chí mạng: không liên tục, tầm nhìn bị che khuất và chi phí nhân sự lớn.
* **Giải pháp tự động hóa:**
  - Ứng dụng thị giác máy tính và Deep Learning giám sát qua camera giám sát CCTV/RTSP sẵn có tại công trường.
* **Mục tiêu cốt lõi:**
  - Phát hiện chính xác công nhân đội mũ/không đội mũ bảo hộ trong luồng video thời gian thực với độ trễ thấp, chi phí phần cứng tối thiểu.
* **Hình ảnh tham chiếu:** `figures/Fig1_site_overview_challenges.jpg` (Minh họa hiện trường công trường thực tế với góc máy camera trên cao, nhiều công nhân và vật cản).

---

### SLIDE 3: 4 NÚT THẮT KỸ THUẬT CỐT LÕI (TECHNICAL BOTTLENECKS)
* **1. Suy giảm đặc trưng vật thể siêu nhỏ ở cự ly xa (Distant Tiny Targets):**
  - Camera gắn trên cao ($15-30m$) khiến mũ bảo hộ chỉ chiếm kích thước $<20\times20$ pixels (thậm chí $<15\times15$ px). Qua các tầng downsampling stride 16/32, tín hiệu mục tiêu bị suy biến hoàn toàn.
* **2. Mất cân bằng dữ liệu cực đoan nội khung (Extreme Class Imbalance):**
  - Tỷ lệ mũ (`hat`) và thân người (`person`) trên tập chuẩn SHWD là $9,044 : 111,514$ (lệch tới **$1:12$**), khiến hàm loss bị áp đảo bởi các mẫu background dễ học.
* **3. Nhiễu không gian và vật thể gây nhầm lẫn thị giác (Spatial Clutter):**
  - Công trường chứa nhiều vật thể hình tròn, màu vàng/cam như xô nhựa, cọc tiêu, đèn pha. CNN truyền thống bất biến tịnh tiến nên dễ báo động giả.
* **4. Rào cản phần cứng biên & độ trễ thời gian thực (Strict Edge Latency):**
  - Giám sát video thực tế đòi hỏi thông lượng $\ge 60$ FPS, trong khi thiết bị biên tại công trường thường có tài nguyên hạn chế (GPU laptop giá rẻ hoặc CPU).

---

### SLIDE 4: TỔNG QUAN NGHIÊN CỨU & KHOẢNG TRỐNG KHOA HỌC (LITERATURE REVIEW)
* **Khảo sát chuyên sâu 32 công trình nghiên cứu quốc tế (2019 – 2026):**
  - **YOLO Series (YOLOv8, YOLOv10, YOLO11):** Nhanh, độ chính xác tổng quát cao nhưng thiếu cơ chế chuyên biệt cho vật thể siêu nhỏ và dễ nhầm lẫn màu sắc môi trường công trường.
  - **EC-YOLOv8 (Zhang et al., 2024):** Đạt $95.7\%$ $mAP_{50}$ nhưng sử dụng CARAFE upsampling làm tăng độ phức tạp, tốc độ chỉ đạt $172$ FPS trên desktop GPU.
  - **YOLO-CBF (Li et al., 2023):** Thêm CoordConv + BiFormer nhưng số lượng tham số lên tới $37.2$ M ($104.5$ GFLOPs), tốc độ rơi xuống $80.6$ FPS.
  - **YOLOv8n-FADS (Fu et al., 2024):** Thêm nhánh P2 cho mỏ than nhưng làm bùng nổ độ phân giải feature map, gây thắt cổ chai độ trễ.
* **Khoảng trống khoa học (Research Gap):**
  - Chưa có giải pháp nào đạt được đồng thời: **Độ chính xác cao trên vật thể nhỏ + Zero-overhead độ trễ khi suy luận + Năng lực chuyển giao ngoại miền vững chắc**.

---

### SLIDE 5: MỤC TIÊU & PHẠM VI NGHIÊN CỨU (OBJECTIVES & DELIVERABLES)
* **Phạm vi nghiên cứu:**
  - Tập trung vào bài toán phát hiện mũ bảo hộ (Safety Helmet Wearing Detection) từ luồng video CCTV công trường xây dựng.
  - Môi trường huấn luyện: Dual NVIDIA Tesla T4 GPUs (Kaggle). Môi trường suy luận: NVIDIA RTX 3050 Laptop, GeForce MX230 (2GB VRAM) và CPU Edge.
* **5 Sản phẩm đầu ra cam kết (Deliverables):**
  1. **Paper khoa học IEEE Transactions (9 trang):** Đầy đủ chứng minh toán học, Ablation Study, XAI Grad-CAM và Cross-domain.
  2. **Mô hình Rep-YOLO11s:** Đạt $94.83\%$ $mAP_{50}$ (Test cố định), $96.64 \pm 0.32\%$ (5-Fold CV).
  3. **Ứng dụng Giám sát RTSP Hoàn chỉnh:** Xử lý end-to-end $65 - 95$ FPS trên RTX 3050.
  4. **Bộ Động cơ Biên dịch Triển khai Tối ưu:** TensorRT FP16, ONNX Runtime INT8.
  5. **Bộ Dữ liệu Chuẩn hóa Công nghiệp:** Hơn $33,000$ ảnh công trường (VOC2028, GDUT-HWD, SHEL5K, HHW).

---

### SLIDE 6: TỔNG THỂ KIẾN TRÚC REP-YOLO11S (NEURAL ARCHITECTURE)
* **Nguyên lý thiết kế:**
  - End-to-end single-stage detector tích hợp 3 phân hệ tối ưu:
    1. **Backbone:** CSPDarknet kết hợp các khối Tái tham số hóa Cấu trúc (RepConv) và tầng đầu vào CoordConv.
    2. **Neck:** Path Aggregation Network (PAN) tích hợp cơ chế Chú ý Định tuyến Thưa hai tầng BiFormer (Bi-Level Routing Attention).
    3. **Head:** Anchor-Free Decoupled Detection Head với hàm mất mát Focal EIoU Loss.
* **Hình ảnh tham chiếu:** `figures/Fig2_rep_yolo11s_neural_architecture.png` (Sơ đồ toàn bộ kiến trúc mạng từ input đến output, chi tiết RepConv training/deploy và BiFormer routing).

---

### SLIDE 7: ĐỘT PHÁ 1 - TÁI THAM SỐ HÓA CẤU TRÚC (REPCONV)
* **Giải quyết mâu thuẫn cốt lõi:** Làm thế nào để mô hình học biểu diễn phi tuyến tính mạnh mẽ nhưng không tốn thêm tài nguyên khi chạy thực tế?
* **Cơ chế hoạt động:**
  - **Giai đoạn Huấn luyện (Training Phase):** Thiết kế 3 nhánh song song: Conv $3\times3$, Conv $1\times1$, và Identity branch. Đa dạng hóa dòng gradient, tăng khả năng trích xuất đặc trưng cạnh và kết cấu.
  - **Giai đoạn Triển khai (Deployment Phase - `switch_to_deploy`):**
    $$\text{Conv-BN} \to W' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W, \quad b' = \beta - \frac{\gamma\mu}{\sqrt{\sigma^2+\epsilon}}$$
    $$W_{\text{fused}} = W'_{3\times3} + \text{Pad}(W'_{1\times1}) + W'_{\text{id}}$$
  - Toàn bộ cấu trúc đa nhánh được suy biến đại số tuyến tính về duy nhất một nhân $3\times3$.
* **Hiệu quả thực tế:**
  - Giảm độ trễ từ $7.12$ ms xuống **$2.92$ ms** trên Tesla T4 ($5.35$ ms trên RTX 3050) với sai số độ chính xác bằng 0!

---

### SLIDE 8: ĐỘT PHÁ 2 - MÃ HÓA TỌA ĐỘ (COORDCONV) & ĐỊNH TUYẾN THƯA (BIFORMER)
* **1. Spatial Coordinate Convolution (CoordConv):**
  - Ghép thêm 2 kênh tọa độ không gian chuẩn hóa $C_x, C_y \in [-1, 1]$ vào tensor ảnh đầu vào ($640\times640\times5$).
  - **Tác dụng:** Phá vỡ tính bất biến tịnh tiến của CNN. Cung cấp tiên đề hình học (Spatial Prior): "Mũ bảo hộ chỉ xuất hiện phía trên thân người; các vật thể vàng dưới mặt đất là xô vữa/cọc tiêu".
* **2. Bi-Level Routing Attention (BiFormer):**
  - Chia feature map thành các vùng thô $S \times S$. Tính ma trận tương quan vùng $A^r = Q^r (K^r)^T$ và lọc lấy Top-$k$ vùng có liên quan nhất.
  - Chú ý chi tiết mức token chỉ được tính trên các vùng được định tuyến:
    $$\text{Complexity: } \mathcal{O}\left(S^2 + k \cdot \frac{HW}{S^2}\right) \ll \mathcal{O}(H^2W^2)$$
  - **Tác dụng:** Tập trung 100% tài nguyên tính toán vào các vùng ứng viên chứa mũ bảo hộ siêu nhỏ, loại bỏ nhiễu phông nền phức tạp.

---

### SLIDE 9: ĐỘT PHÁ 3 - HÀM MẤT MÁT FOCAL EIOU LOSS
* **Hạn chế của CIoU truyền thống:**
  - CIoU sử dụng tỷ lệ khung hình tương đối $v \propto (\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h})^2$, dễ bị bão hòa gradient khi kích thước chiều dài/chiều rộng tuyệt đối có sai số lớn.
* **Cải tiến của Focal EIoU Loss:**
  - Phân rã trực tiếp sai lệch chiều dài và chiều rộng độc lập:
    $$\mathcal{L}_{\text{EIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$$
  - Kết hợp trọng số tiêu điểm (Focal Weighting) $\text{IoU}^\gamma$ ($\gamma=0.5$):
    $$\mathcal{L}_{\text{Focal-EIoU}} = \text{IoU}^\gamma \cdot \mathcal{L}_{\text{EIoU}}$$
* **Tác dụng:** Tăng tốc độ hội tụ và định vị sắc nét mép mũ bảo hộ bị che khuất một phần trong môi trường công trường rậm rạp.

---

### SLIDE 10: TẬP DỮ LIỆU & QUY TRÌNH CHUẨN HÓA CÔNG NGHIỆP
* **Tập nguồn chính (SHWD / VOC2028):**
  - $7,581$ ảnh công trường thực tế ($6,064$ trainval, $1,517$ test).
  - Khắc phục nhãn rác (loại bỏ 3 nhãn dị biệt "dog" trong XML gốc).
  - Phân chia $80\%$ Train / $20\%$ Val không rò rỉ dữ liệu (No Data Leakage).
* **Mở rộng kiểm thử ngoại vi đa miền (Hơn 25,000 ảnh):**
  - **GDUT-HWD:** $13,499$ ảnh công trường Trung Quốc (mật độ đông 15-30 người/ảnh).
  - **SHEL5K:** $5,000$ ảnh góc nhìn Drone/Flycam chụp thẳng đứng từ đỉnh đầu.
  - **Hard Hat Workers (AndrewMVD):** $7,000$ ảnh công trường ngoài trời.
  - **Safety Helmet Detection (SHD) & SFCHD:** Camera giám sát giao thông và nhà máy.
* **Chuẩn hóa không gian nhãn chung $\mathcal{C}^*$:**
  $$\mathcal{C}^* = \{0: \text{'hat'} \text{ (Mũ bảo hộ)}, 1: \text{'person'} \text{ (Đầu trần / Người)}\}$$

---

### SLIDE 11: KẾT QUẢ THỰC NGHIỆM & SO SÁNH VỚI SOTA
* **Bảng so sánh định lượng (Trích từ Table I trong Paper):**
  - **YOLO11s:** $94.74\%$ $mAP_{50}$ | $62.54\%$ $mAP_{50-95}$ | $6.52$ ms
  - **YOLOv8s:** $94.89\%$ $mAP_{50}$ | $62.21\%$ $mAP_{50-95}$ | $6.10$ ms
  - **EC-YOLOv8 (2024):** $95.70\%$ $mAP_{50}$ | $5.80$ ms ($172$ FPS)
  - **YOLO-CBF (2023):** $95.60\%$ $mAP_{50}$ | $12.40$ ms ($80.6$ FPS)
  - **Rep-YOLO11s (Ours - Single Run):** **$94.83\%$** $mAP_{50}$ | **$62.54\%$** $mAP_{50-95}$ | **$2.92$ ms** ($342.5$ FPS trên T4, $5.35$ ms trên RTX 3050)
  - **Rep-YOLO11s (Ours - 5-Fold CV Mean):** **$96.64 \pm 0.32\%$** $mAP_{50}$ (Đỉnh Fold 3: **$97.11\%$**) | **$65.91 \pm 0.46\%$** $mAP_{50-95}$ | F1-score: **$0.9396$**
* **Nhận xét:**
  - Rep-YOLO11s đạt độ chính xác tương đương hoặc vượt trội SOTA nhưng có tốc độ suy luận nhanh gấp đôi đối thủ nhờ cấu trúc RepConv tối ưu.
* **Bảng tham chiếu:** `tables/Table1_sota_comparison.md`

---

### SLIDE 12: NGHIÊN CỨU CẮT BỎ THÀNH PHẦN (ABLATION STUDY A0 -> A6)
* **Minh chứng định lượng từng mô-đun đề xuất (Trích từ Table II):**
  - **$A_0$ (Baseline YOLO11s):** $94.74\%$ $mAP_{50}$, $62.34\%$ $mAP_{50-95}$, Latency: $6.52$ ms.
  - **$A_1$ (+ Nhánh P2):** $94.81\%$ $mAP_{50}$ nhưng độ trễ tăng vọt lên $8.94$ ms $\to$ Loại bỏ để đảm bảo thời gian thực.
  - **$A_2$ (+ CoordConv):** $94.78\%$ $mAP_{50}$, tăng cường độ chính xác định vị không gian, độ trễ giữ nguyên $6.58$ ms.
  - **$A_3$ (+ RepConv):** $94.81\%$ $mAP_{50}$, đa dạng hóa biểu diễn.
  - **$A_4$ (+ Focal EIoU):** $94.88\%$ $mAP_{50}$, cải thiện mạnh khả năng bắt dính biên hộp.
  - **$A_5$ (+ BiFormer Attention):** Tăng Recall lên $91.15\%$.
  - **$A_6$ (Full Fusion Rep-YOLO11s):** **$94.83\%$** $mAP_{50}$, **$62.54\%$** $mAP_{50-95}$, và độ trễ giảm đột phá về **$2.92$ ms** sau `switch_to_deploy`!
* **Bảng tham chiếu:** `tables/Table2_ablation_study.md`

---

### SLIDE 13: ĐÁNH GIÁ TRỰC QUAN XAI GRAD-CAM (SALIENCY ANALYSIS)
* **Chứng minh mạng học đúng bản chất thay vì học vẹt:**
  - **Tình huống 1 (Áo bảo hộ dạ quang & Giàn giáo gỗ):** Baseline YOLO11s bị phân tán gradient vào áo phản quang màu cam. Rep-YOLO11s tập trung 100% điểm nhiệt vào 4 chiếc mũ bảo hộ xanh (Confidence $0.84$).
  - **Tình huống 2 (Mục tiêu siêu nhỏ bị ngược sáng mạnh):** BiFormer định tuyến token chính xác, khóa chặt chiếc mũ nhỏ bên cửa sổ ngược sáng (Confidence $0.89$).
  - **Tình huống 3 (Biển cảnh báo màu vàng tam giác):** CoordConv áp đặt tiên đề vị trí giải phẫu, loại bỏ hoàn toàn biển báo tam giác giả mạo, chỉ bắt đúng mũ công nhân.
* **Hình ảnh tham chiếu:** `figures/Fig3_gradcam_xai_saliency_comparison.png` (So sánh trực quan 3 kịch bản: Ảnh gốc, Baseline bị nhiễu, Rep-YOLO11s tập trung và Bounding Box).

---

### SLIDE 14: ĐÁNH GIÁ NĂNG LỰC TỔNG QUÁT HÓA NGOẠI MIỀN (CROSS-DOMAIN)
* **Kết quả Zero-Shot trên 5 bộ dữ liệu độc lập (Table III):**
  - **Hard Hat Workers (7,000 ảnh):** Đạt đỉnh **$97.03\%$** $mAP_{50}$ theo chuẩn Harmonized PPE (Hat-Only).
  - **GDUT-HWD (13,499 ảnh):** Duy trì **$74.27\%$** $mAP_{50}$ trong bối cảnh công nhân cực kỳ đông đúc ($15-30$ người/ảnh).
  - **SHEL5K (5,000 ảnh):** Đạt **$41.15\%$** $mAP_{50}$ do góc nhìn Flycam thẳng đứng từ trên trời triệt tiêu bối cảnh giải phẫu người và vật thể quá nhỏ ($<15$ px).
* **Phát hiện khoa học về Hiện tượng Sụp đổ IoU (IoU Collapse):**
  - Xung đột nhãn `person` (SHWD: Full-body vs HHW: Head-only) khiến $\text{IoU} \approx 0.07-0.14 \ll 0.50$, kéo sập mAP Joint xuống $74.40\%$.
  - Khi chuẩn hóa Harmonized PPE (chỉ đo lớp `hat`), mAP phục hồi lên **$97.03\%$**, chứng minh năng lực trích xuất đặc trưng mũ là hoàn hảo.
* **Bảng tham chiếu:** `tables/Table3_cross_domain_generalization.md`

---

### SLIDE 15: TRIỂN KHAI PHẦN CỨNG BIÊN & PIPELINE RTSP THỜI GIAN THỰC
* **Benchmark đa nền tảng phần cứng thực tế (Table IV):**
  - **NVIDIA Tesla T4 (Server GPU):** $2.92$ ms (**$342.5$ FPS**) với TensorRT 11.2 FP16.
  - **NVIDIA RTX 3050 Laptop (Edge GPU):** $5.35$ ms (**$187.1$ FPS**) với TensorRT FP16; $12.43$ ms ($80.4$ FPS) với PyTorch Native FP32.
  - **NVIDIA GeForce MX230 (Budget Edge Laptop):** Cấu hình siêu hạn chế (Pascal, **2GB VRAM**, không có Tensor Cores) vẫn đạt **$36.0$ ms ($27.8$ FPS)** với PyTorch Native FP32, vượt ngưỡng thời gian thực chuẩn ($>24$ FPS).
* **Phân tích bóc tách độ trễ luồng RTSP hoàn chỉnh (End-to-End Pipeline):**
  - Giải mã H.264 ($3.5-5.0$ ms) + Tiền xử lý CoordConv ($1.2-2.0$ ms) + Suy luận TensorRT ($2.92-5.35$ ms) + Hậu xử lý NMS ($1.5-2.8$ ms) + Vẽ giao diện UI ($2.2-3.4$ ms) = **$10.54 - 15.38$ ms**.
  - Đạt thông lượng thực tế **$65 - 95$ FPS** trên camera giám sát RTSP.
* **Hình ảnh & Bảng tham chiếu:** `figures/Fig4_industrial_rtsp_surveillance_pipeline.png`, `figures/Fig5_efficiency_frontier_latency_vs_map.png`, `tables/Table4_deployment_benchmark.md`

---

### SLIDE 16: KẾ HOẠCH CHO REVIEW 2, REVIEW 3 & KẾT LUẬN
* **Tóm tắt đóng góp tại Review 1:**
  - Hoàn thành trọn vẹn phát biểu bài toán, cơ sở lý thuyết, kiến trúc mạng mới Rep-YOLO11s và các chứng minh định lượng SOTA/Ablation/Deployment.
  - Khẳng định tính khả thi 100% với các đo đạc vật lý trên phần cứng thật.
* **Kế hoạch triển khai cho Review 2:**
  - Hoàn thiện đóng gói phần mềm Desktop GUI / Web Dashboard giám sát cảnh báo vi phạm an toàn lao động.
  - Tích hợp thêm các bài toán quản trị dự án, tinh chỉnh tham số chuyên sâu và mở rộng thử nghiệm đa camera.
* **Kế hoạch triển khai cho Review 3 & Bảo vệ Hội đồng:**
  - Tích hợp chưng cất tri thức (Knowledge Distillation) cho góc máy Drone (SHEL5K).
  - Hoàn tất báo cáo KLTN chính thức và nộp bài báo khoa học chuẩn IEEE.
* **Cam kết nhóm:** Đảm bảo toàn bộ tiêu chí đánh giá loại Xuất sắc của Hội đồng ĐATN ngành Trí tuệ Nhân tạo FPT University!
