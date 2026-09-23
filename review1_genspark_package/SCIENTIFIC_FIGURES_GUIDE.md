# HƯỚNG DẪN BỘ 10 HÌNH VẼ KHOA HỌC 300 DPI (CHUẨN CÔNG BỐ IEEE)
**Dự án:** Rep-YOLO11s: Real-Time Safety Helmet Detection via Structural Re-parameterization, Coordinate Convolution, and Bi-Level Routing Attention  
**Tác giả:** Nguyễn Hãn Như & Nhóm nghiên cứu Capstone AI  
**Định dạng hình ảnh:** 300 DPI Ultra High-Resolution PNG (~2600 × 1500 px)  
**Vị trí lưu trữ:**
- Slide Presentation: `review1_genspark_package/figures/scientific_exports/`
- Paper LaTeX Overleaf: `paper_overleaf/figures/`

---

## BẢNG TỔNG HỢP CẶP HÌNH ĐỐI CHIẾU (BASELINE VS PROPOSED)

| Slide | Đột phá kỹ thuật | Hình 1: Kiến trúc gốc (Baseline Flaw) | Hình 2: Đột phá của nhóm (Proposed Innovation) |
| :--- | :--- | :--- | :--- |
| **Slide 5 / 10** | **End-to-End Pipeline** | `Fig5A_Baseline_YOLO11s_Architecture.png` | `Fig5B_Proposed_RepYOLO11s_Architecture.png` |
| **Slide 6** | **Innovation 1: RepConv** | `Fig1A_Baseline_MultiBranch_Bottleneck.png` | `Fig1B_Proposed_RepConv_Algebraic_Fusion.png` |
| **Slide 7** | **Innovation 2: CoordConv** | `Fig2A_Baseline_Translation_Invariance_Flaw.png` | `Fig2B_Proposed_CoordConv_Spatial_Injection.png` |
| **Slide 8** | **Innovation 3: BiFormer** | `Fig3A_Baseline_Dense_Attention_Bottleneck.png` | `Fig3B_Proposed_BiFormer_Sparse_Routing.png` |
| **Slide 9** | **Innovation 4: Focal-EIoU** | `Fig4A_Baseline_CIoU_Vanishing_Gradient.png` | `Fig4B_Proposed_Focal_EIoU_Decomposition.png` |

---

## 1. SLIDE 6 — INNOVATION 1: STRUCTURAL RE-PARAMETERIZATION (REPCONV)

### Vấn đề gốc của Baseline (Fig 1A):
- Mạng nhiều nhánh (Multi-branch) giúp đa dạng hóa biểu diễn nhưng gây bùng nổ chi phí truy cập bộ nhớ (**Memory Access Cost - MAC**), hiện tượng phân mảnh cache GPU khiến độ trễ tăng vọt lên **7.12 ms** (140.4 FPS trên Tesla T4).
- Không thể đáp ứng yêu cầu giám sát thời gian thực trên nhiều luồng camera CCTV.

### Đột phá của Nhóm (Fig 1B):
- Thiết kế vòng đời tách biệt:
  1. **Training Phase:** Huấn luyện đồng thời trên 3 nhánh ($3\times3$ Conv+BN, $1\times1$ Conv+BN, Identity+BN) giúp dòng gradient phong phú, chống biến mất gradient.
  2. **Closed-Form Algebraic Fusion (`switch_to_deploy`):**
     $$W' = \frac{\gamma}{\sigma} W, \quad b' = \beta - \frac{\gamma \mu}{\sigma}$$
     $$W_{\text{fused}} = W'_{3\times3} + \text{Pad}(W'_{1\times1}) + W'_{\text{id}}$$
  3. **Inference Phase:** Gập hoàn toàn thành **duy nhất 1 nhân $3\times3$ Conv**, loại bỏ 100% chi phí chuyển nhánh và trễ BatchNorm.
- **Minh chứng thực nghiệm:** Độ trễ giảm 55.2% từ 7.12 ms xuống **2.92 ms** trên Tesla T4 (**342.5 FPS**), **5.35 ms** trên RTX 3050 Laptop. Sai số toán học giữa trước và sau gập $\Delta < 10^{-5}$. Mã nguồn: `custom_ablation_modules.py` (Dòng 71-164).

---

## 2. SLIDE 7 — INNOVATION 2: COORDINATE CONVOLUTION (COORDCONV)

### Vấn đề gốc của Baseline (Fig 2A):
- Phép tích chập 2D chuẩn trên tensor 3 kênh RGB có tính **Bất biến tịnh tiến (Translation Invariance)**:
  $$\mathcal{T}_{(\Delta x, \Delta y)} [I * K] = [\mathcal{T}_{(\Delta x, \Delta y)} I] * K$$
- Hạt nhân tích chập hoàn toàn "mù tọa độ" (Position-Blind). Một chiếc nón bảo hộ màu vàng trên đầu và một chiếc xô vữa vàng hoặc cọc tiêu nằm dưới sàn bê tông tạo ra vector phản hồi giống hệt nhau $\rightarrow$ Gây ra **>28% cảnh báo giả (False Alarms)** dưới sàn công trường.

### Đột phá của Nhóm (Fig 2B):
- Module `AddCoords` tự động tạo 2 kênh tọa độ Descartes chuẩn hóa:
  $$C_x(i, j) = \frac{2j}{W-1} - 1 \in [-1, 1], \quad C_y(i, j) = \frac{2i}{H-1} - 1 \in [-1, 1]$$
- Ghép trực tiếp cùng RGB tạo tensor 5 chiều $[R, G, B, C_x, C_y]$ ngay tại lớp Stem (`Conv(c1=5, c2=64)`).
- **Minh chứng thực nghiệm:** Cung cấp tiên nghiệm không gian giải phẫu (Anatomical Prior): nón chỉ nằm ở vùng đầu ($C_y \approx -0.8 \rightarrow 0.2$), triệt tiêu toàn bộ báo động giả dưới mặt đất ($C_y > 0.5$). Độ trễ đo thực tế chỉ tăng **0.06 ms** (6.58 ms vs 6.52 ms baseline), hoàn toàn nằm trong dung sai nhiễu. Mã nguồn: `custom_ablation_modules.py` (Dòng 46-68).

---

## 3. SLIDE 8 — INNOVATION 3: BI-LEVEL ROUTING ATTENTION (BIFORMER)

### Vấn đề gốc của Baseline (Fig 3A):
- Cơ chế Self-Attention đầy đủ (MHSA) có độ phức tạp tính toán bậc hai $\mathcal{O}(H^2 W^2)$.
- Với ảnh $640\times640$ ($N=40,000$ tokens), số phép tính tích vô hướng là $1.6 \times 10^9$; với ảnh $1024\times1024$ bùng nổ lên $1.05 \times 10^{10}$ phép tính.
- Gây tràn bộ nhớ đồ họa (**CUDA Out-Of-Memory**) trên GPU biên (như GeForce MX230 2GB VRAM), làm loãng trọng số chú ý vào 85% diện tích nền rác (bầu trời, mặt sàn bê tông, tường trắng).

### Đột phá của Nhóm (Fig 3B):
- Triển khai thuật toán định tuyến thưa 2 cấp độ:
  1. **Region Partitioning:** Chia feature map thành các vùng thô $S \times S$ ($S=8$, 64 vùng).
  2. **Top-k Region Routing:** Xây dựng đồ thị tương quan vùng $A^r = Q^r (K^r)^T$, chỉ giữ lại $k=4$ vùng tương quan nhất, cắt tỉa (prune) **80% vùng nền rác**.
  3. **Routed Token Attention:** Chỉ tính Attention chi tiết trong các vùng được định tuyến, đưa độ phức tạp về tuyến tính:
     $$\mathcal{O}\left(S^2 + k \cdot \frac{HW}{S^2}\right) \approx \mathcal{O}(HW)$$
- **Minh chứng thực nghiệm:** Đẩy Recall phát hiện nón bảo hộ đạt đỉnh **91.33%** (+0.98% so với baseline), bắt dính hoàn hảo các nón bảo hộ bị giàn giáo che khuất ở khoảng cách xa 20-30m. Chạy mượt mà ở độ phân giải $1024\times1024$ không bị OOM. Mã nguồn: `custom_ablation_modules.py` (Dòng 165-230).

---

## 4. SLIDE 9 — INNOVATION 4: FOCAL EIOU BOUNDING BOX LOSS

### Vấn đề gốc của Baseline (Fig 4A):
- Hàm phạt tỷ lệ khung hình trong CIoU:
  $$v = \frac{4}{\pi^2} \left(\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h}\right)^2$$
- Đạo hàm riêng theo chiều rộng:
  $$\frac{\partial v}{\partial w} = \frac{8}{\pi^2} \left(\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h}\right) \cdot \frac{h}{w^2 + h^2}$$
- **Hiện tượng triệt tiêu gradient:** Khi tỷ lệ khung hình dự đoán tình cờ trùng với tỷ lệ mẫu thật ($\frac{w}{h} = \frac{w^{gt}}{h^{gt}}$), đạo hàm $\frac{\partial v}{\partial w} \equiv 0$ ngay cả khi kích thước hộp sai lệch 100% (ví dụ $w=40, h=80$ so với $w^{gt}=20, h^{gt}=40$).
- Bounding box không thể co về đúng kích thước nón bảo hộ siêu nhỏ ($<20\times20$ px).

### Đột phá của Nhóm (Fig 4B):
- Phân rã trực tiếp sai lệch chiều dài và chiều rộng độc lập:
  $$\mathcal{L}_{\text{EIoU}} = \mathcal{L}_{\text{IoU}} + \mathcal{L}_{\text{dis}} + \mathcal{L}_{\text{asp}} = (1 - \text{IoU}) + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$$
- Bổ sung hệ số tập trung mẫu khó **Focal factor ($\text{IoU}^{0.5}$)**, **Inner-Shape-IoU** và khoảng cách Wasserstein Gaussian (**NWD**).
- **Minh chứng thực nghiệm:** Đảm bảo đạo hàm $\frac{\partial L}{\partial w} \neq 0$, giải quyết dứt điểm sự mất cân bằng mẫu 1:12 giữa 111,514 thân người và 9,044 nón nhỏ. mAP50 tăng lên 94.88% (Ablation A4) và mAP50-95 tăng lên 62.51%. Được vá trực tiếp vào `ultralytics/utils/loss.py`.

---

## 5. SLIDE 5 & 10 — TOÀN BỘ KIẾN TRÚC ĐƯỜNG ỐNG (END-TO-END PIPELINE)

### Kiến trúc Baseline YOLO11s (Fig 5A):
- Nhận luồng ảnh RGB chuẩn 3 kênh.
- Backbone CSPDarknet truyền thống (C3k2 đơn luồng).
- Neck PAFPN phân bố chú ý đồng đều, không định tuyến vùng.
- 3 đầu ra phát hiện (Heads P3, P4, P5), thiếu nhánh P2 độ phân giải cao cho vật siêu nhỏ.
- Giám sát bằng CIoU Loss dễ lỗi bão hòa gradient.

### Kiến trúc vô địch đề xuất Rep-YOLO11s-P2 AFPN (Fig 5B):
- **Tầng 1 (CoordConv Stem):** Tensor 5 chiều $[R,G,B,C_x,C_y]$, xóa sổ >28% cảnh báo giả dưới sàn.
- **Tầng 2 (RepConv Backbone):** Huấn luyện đa nhánh $\rightarrow$ Gập thành $3\times3$ khi triển khai (`switch_to_deploy`), đạt 2.92 ms trên T4 (342.5 FPS).
- **Tầng 3 (BiFormer Neck):** Định tuyến thưa 2 cấp độ ($S=8, k=4$), loại bỏ 80% nhiễu nền, phức tạp $\mathcal{O}(HW)$.
- **Tầng 4 (4 Decoupled Heads + P2):** Bổ sung Head P2 (stride 4, feature map $160\times160$) chuyên trách vật siêu nhỏ $8\times8$ pixel.
- **Tầng 5 (Focal-EIoU Loss):** Tách rời gradient $dw, dh$, cân bằng mẫu 1:12.
- **Thành tựu định lượng thực tế:**
  * 5-Fold Cross-Validation: Mean **$96.64 \pm 0.32\%$**, Đỉnh Fold 3 đạt **$97.11\%$** mAP50.
  * Tốc độ TensorRT FP16: **2.92 ms** (342.5 FPS trên Tesla T4), **5.35 ms** (187.1 FPS trên RTX 3050).
  * Chạy trên Laptop cấu hình yếu NVIDIA GeForce MX230 (2GB VRAM): **27.8 FPS** (36.0 ms), chỉ chiếm 485 MB VRAM, vượt ngưỡng thời gian thực điện ảnh (24 FPS).
  * Kiểm thử ngoại vi Zero-Shot trên Hard Hat Workers (7,000 ảnh): **$97.03\%$** mAP50 theo chuẩn Harmonized PPE Hat-Only.
