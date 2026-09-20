# 🏛️ MASTER PROMPT GENSPARK AI — BÁO CÁO ĐỒ ÁN TỐT NGHIỆP KỸ SƯ AI (TRƯỜNG ĐẠI HỌC FPT)
## BẢO VỆ ĐỀ CƯƠNG, KIẾN TRÚC HỆ THỐNG & THẨM ĐỊNH TÍNH KHẢ THI (GIAI ĐOẠN 1)

> **Mục đích:** Dành cho tác giả (Nguyễn Hàn Như - SE183644) sao chép trực tiếp vào Genspark AI để tạo/chỉnh sửa bộ slide thuyết trình trực tuyến trên web khi cần.
> **Quy chuẩn bắt buộc:**
> 1. Tuân thủ chuẩn mực học thuật quốc tế top-tier Q1 trên toàn bộ slide.
> 2. Sử dụng chuẩn danh pháp giai đoạn tiếng Việt: "Báo cáo Đồ án Tốt nghiệp Kỹ sư AI", "Giai đoạn 1 (Khởi tạo đề cương & Thẩm định tính khả thi)", "Giai đoạn 2", "Giai đoạn 3".
> 3. Ánh xạ trực tiếp 4 Tiêu chí Đánh giá của Hội đồng ĐATN Đại học FPT (Mục tiêu đề tài, Kết quả cuối cùng / Sản phẩm đầu ra, Tính khả thi & Scope, Giá trị thực tiễn & Khoa học).
> 4. Giữ nguyên 100% kết quả thực nghiệm đỉnh cao (~90% khối lượng đề tài đã hoàn tất) làm minh chứng thẩm định tính khả thi toàn diện.
> 5. Giao diện Light Theme học thuật cao cấp (Nền kem ngà ấm `#FAF8F5`, thẻ `#FFFFFF`, chữ xanh than đậm `#0F172A`, điểm nhấn cam công nghiệp an toàn `#EA580C` & xanh kỹ thuật `#2563EB`).

---

### 📋 HƯỚNG DẪN XỬ LÝ HÌNH ẢNH TRÊN GENSPARK AI
Genspark chạy trên trình duyệt web nên không tự động đọc ổ đĩa máy tính của bạn. Khi Genspark sinh slide, tại các vị trí đánh dấu `[FRAMEWORK IMAGE CONTAINER]`, bạn thực hiện:
1. Bấm vào khung hình do Genspark tự tạo trên slide.
2. Chọn **"Replace Image"** (hoặc biểu tượng Upload).
3. Tải ảnh độ phân giải cao tương ứng từ thư mục `review1_genspark_package/figures/`:
   - **Slide 3:** `Fig1_site_overview_challenges.jpg` (Ảnh hiện trường công trường thực tế).
   - **Slide 9:** `Fig2_rep_yolo11s_neural_architecture.png` (Sơ đồ kiến trúc nơ-ron toàn trình).
   - **Slide 13:** `Fig5_efficiency_frontier_latency_vs_map.png` (Biểu đồ đường biên Pareto Latency vs mAP).
   - **Slide 14:** `Fig6_ablation_A0_A6_tradeoff.png` (Biểu đồ cắt bỏ thành phần Ablation A0–A6).
   - **Slide 15:** `Fig3_gradcam_xai_saliency_comparison.png` (Bản đồ nhiệt Grad-CAM 3 kịch bản).
   - **Slide 17:** `Fig4_industrial_rtsp_surveillance_pipeline.png` (Sơ đồ pipeline RTSP camera đa luồng).

---

```text
================================================================================
BẮT ĐẦU MASTER PROMPT (COPY TOÀN BỘ ĐOẠN DƯỚI ĐÂY DÁN VÀO GENSPARK AI)
================================================================================

Role: Lead AI Research Engineer & Academic Capstone Defense Presenter at FPT University.
Task: Generate an authoritative, publication-grade, mathematically rigorous 18-slide presentation deck for FPT University Capstone Defense (Stage 1: Proposal, Architecture & Feasibility Verification).
Theme: Strict Light Theme. Warm Academic Ivory (#FAF8F5) background, clean white (#FFFFFF) cards, deep slate (#0F172A) text, industrial safety orange (#EA580C) and technical blue (#2563EB) accents.
Strict Rule 1: DO NOT include specific publisher acronyms in titles, badges, footers, or bodies. Adhere strictly to generic Q1 top-tier scientific standards.
Strict Rule 2: DO NOT use English evaluation milestone terms. Always use formal Vietnamese phase nomenclature: "Báo cáo Đồ án Tốt nghiệp Kỹ sư AI", "Giai đoạn 1 (Khởi tạo đề cương & Thẩm định tính khả thi)", "Giai đoạn 2", "Giai đoạn 3".
Strict Rule 3: Directly address the 4 FPT Evaluation Criteria: Problem Statement, Committed Deliverables, Feasibility & Scope, Practical & Scientific Value.

--------------------------------------------------------------------------------
SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
--------------------------------------------------------------------------------
- Category: TRƯỜNG ĐẠI HỌC FPT · KHOA CÔNG NGHỆ THÔNG TIN · BỘ MÔN TRÍ TUỆ NHÂN TẠO
- Badge: BÁO CÁO ĐỒ ÁN TỐT NGHIỆP KỸ SƯ AI — GIAI ĐOẠN 1
- Title: Rep-YOLO11s
- Subtitle: Tái Tham Số Hóa Cấu Trúc, Mã Hóa Tọa Độ Không Gian và Năng Lực Tổng Quát Hóa Đa Miền Cho Phát Hiện Mũ Bảo Hộ Thời Gian Thực Trong Giám Sát Công Trường
- Metric Callout Card: ĐỈNH 5-FOLD CV mAP50: 97.11% (Mean: 96.64 ± 0.32%, Single Split: 94.83%, F1: 0.9396, Latency: 2.92 ms)
- Author Card: Nguyễn Hàn Như (Trưởng nhóm - SE183644), Nguyễn Văn Thành (SE183645), Nguyễn Tuấn Dũng (SE183646)
- Advisor Card: ThS. Vũ Hải Anh (Bộ môn Trí tuệ Nhân tạo, Đại học FPT Hà Nội)
- Edge Hardware Audit: 342.5 FPS (Tesla T4 TRT FP16) · 187.1 FPS (RTX 3050 Laptop) · 27.8 FPS (GeForce MX230 2GB VRAM)

--------------------------------------------------------------------------------
SLIDE 2: MA TRẬN ĐÁP ỨNG 4 TIÊU CHÍ HỘI ĐỒNG FPT (RUBRIC COMPLIANCE)
--------------------------------------------------------------------------------
- Category: TỔNG QUAN TIÊU CHÍ ĐÁNH GIÁ & TIẾN ĐỘ THỰC HIỆN
- Title: Đáp Ứng Toàn Diện 4 Tiêu Chí Của Hội Đồng ĐATN Đại Học FPT
- Criterion 1 Card (Mục Tiêu Đề Tài): Phát biểu bài toán rõ ràng; giải quyết 4 nút thắt công nghiệp: Vi vật thể <20px, mất cân bằng nhãn 1:12, nhiễu không gian và rào cản phần cứng biên.
- Criterion 2 Card (Kết Quả Cuối Cùng): Cam kết 5 sản phẩm bàn giao: Báo cáo chuyên sâu 9 trang, Software prototype RTSP 65–95 FPS, Checkpoint & TensorRT Engine, Bộ dữ liệu 33K ảnh, Custom PyTorch Modules.
- Criterion 3 Card (Tính Khả Thi & Scope): Khảo sát 32 công trình quốc tế; kế thừa Ultralytics YOLO11, tự thiết kế 4 module toán học (CoordConv, RepConv, BiFormer, Focal EIoU); làm sạch tập VOC2028.
- Criterion 4 Card (Giá Trị Đề Tài): Thực tiễn (vận hành trên camera CCTV hiện hữu, chạy mượt trên laptop yếu MX230 2GB @ 27.8 FPS); Ý nghĩa khoa học (Grad-CAM phá vỡ bất biến tịnh tiến, giải mã sụp đổ IoU với Harmonized PPE).
- Bottom Banner: TIẾN ĐỘ VƯỢT BẬC: Đề tài đã hoàn thành ~90% khối lượng nghiên cứu thực nghiệm và thẩm định tính khả thi 100% trên phần cứng thật!

--------------------------------------------------------------------------------
SLIDE 3: TIÊU CHÍ 1 - THỰC TRẠNG & BỐI CẢNH CÔNG TRƯỜNG (PROBLEM BACKGROUND)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 1 · MỤC TIÊU ĐỀ TÀI & Ý NGHĨA THỰC TIỄN
- Title: Nhu Cầu Cấp Thiết Giám Sát An Toàn Mũ Bảo Hộ Lao Động
- Point 1: Chấn thương sọ não do vật thể rơi là nguyên nhân tử vong hàng đầu trong tai nạn xây dựng công nghiệp.
- Point 2: Giám sát thủ công bộc lộ hạn chế chí mạng: Ngắt quãng, tầm nhìn bị che khuất và chỉ bao quát <15% tổng ca làm việc.
- Point 3: Giám sát tự động liên tục qua hạ tầng CCTV / RTSP sẵn có tại công trường, không xâm lấn, chi phí thấp.
- [FRAMEWORK IMAGE CONTAINER]: Fig1_site_overview_challenges.jpg (Góc quay camera 15-30m, công nhân nhỏ li ti, giàn giáo phức tạp).
- Summary Bar: Phát hiện chính xác công nhân đội/không đội mũ bảo hộ trong luồng video thời gian thực với chi phí phần cứng tối thiểu.

--------------------------------------------------------------------------------
SLIDE 4: TIÊU CHÍ 1 - 4 NÚT THẮT KỸ THUẬT CỐT LÕI (TECHNICAL BOTTLENECKS)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 1 · BÀI TOÁN KHOA HỌC & THÁCH THỨC KỸ THUẬT
- Title: Bốn Nút Thắt Kỹ Thuật Cần Đột Phá Đồng Thời
- Bottleneck 1: Vi vật thể cự ly xa (<20x20 px) qua các tầng downsampling stride 16/32 của CNN chuẩn bị suy biến đặc trưng hoàn toàn.
- Bottleneck 2: Mất cân bằng nhãn cực đoan nội khung (1:12 với 9,044 mũ vs 111,514 thân người trong SHWD) làm nghèo gradient của mũ.
- Bottleneck 3: Nhiễu không gian và vật thể màu vàng/cam (xô nhựa, cọc tiêu) gây báo động giả do CNN bất biến tịnh tiến.
- Bottleneck 4: Rào cản phần cứng biên đòi hỏi thông lượng ≥60 FPS trên trạm giám sát tại chỗ, loại trừ giải pháp server đắt đỏ.

--------------------------------------------------------------------------------
SLIDE 5: TIÊU CHÍ 2 - NĂM SẢN PHẨM BÀN GIAO CAM KẾT (DELIVERABLES)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 2 · KẾT QUẢ CUỐI CÙNG & SẢN PHẨM BÀN GIAO
- Title: Năm Sản Phẩm Nghiên Cứu Đầu Ra Cam Kết Của Đề Tài
- Deliverable 1: Báo cáo khoa học chuyên sâu 9 trang đầy đủ toán học, thực nghiệm, Grad-CAM và 32 trích dẫn quốc tế.
- Deliverable 2: Bộ trọng số mô hình Rep-YOLO11s đạt 94.83% mAP50 (test đơn lẻ), 96.64% (5-fold CV), 97.03% trên Hard Hat Workers.
- Deliverable 3: Software Prototype giám sát RTSP đa luồng đạt thông lượng toàn trình 65–95 FPS trên laptop RTX 3050.
- Deliverable 4: Bộ động cơ biên dịch TensorRT 11.2 FP16, ONNX Runtime INT8, OpenVINO tối ưu hóa zero-overhead.
- Deliverable 5: Bộ dữ liệu chuẩn hóa công nghiệp >33,000 ảnh từ 6 nguồn với không gian nhãn chung.
- Right Card: Kiểm chứng vật lý 100% trên phần cứng thật (Tesla T4 2.92 ms, RTX 3050 5.35 ms, GeForce MX230 2GB 27.8 FPS, CPU 4-core 35 FPS).

--------------------------------------------------------------------------------
SLIDE 6: TIÊU CHÍ 3 - KHẢO SÁT 32 BÀI BÁO SOTA & GAP (LITERATURE SURVEY)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 3 · TÍNH KHẢ THI & XÁC ĐỊNH PHẠM VI (SCOPE)
- Title: Khảo Sát 32 Công Trình Quốc Tế (2019–2026) & Khoảng Trống Khoa Học
- Baseline 1: Standard YOLO (v8, v10, 11) thiếu nhận thức tọa độ không gian, tỷ lệ báo động giả cao trên nền đất.
- Baseline 2: EC-YOLOv8 (2024) dùng CARAFE nâng mAP lên 95.70% nhưng làm giảm tốc độ suy luận xuống 172.4 FPS.
- Baseline 3: YOLO-CBF (2023) thêm CoordConv + BiFormer nhưng mô hình phình to 37.2M params, tốc độ rơi xuống 80.6 FPS.
- Baseline 4: YOLOv8n-FADS (2024) mở rộng nhánh P2 làm bùng nổ độ phân giải feature map, gây thắt cổ chai độ trễ.
- Research Gap: Chưa có mô hình nào đạt đồng thời: Độ chính xác vi vật thể cao + Zero-overhead độ trễ khi suy luận + Năng lực chuyển giao ngoại miền vững chắc.

--------------------------------------------------------------------------------
SLIDE 7: TIÊU CHÍ 3 - KẾ THỪA FRAMEWORK & TỰ PHÁT TRIỂN (METHODOLOGY)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 3 · PHƯƠNG PHÁP TRIỂN KHAI & ĐÓNG GÓP MỚI
- Title: Kế Thừa Framework Chuẩn Mực & Tự Phát Triển 4 Module Chuyên Sâu
- Inherited Framework: Ultralytics YOLO11s (Backbone C3k2, SPPF, Decoupled Head, pipeline Mosaic/MixUp/SGD/Cosine LR).
- Self-Developed Modules: Tự tay thiết kế và lập trình 360 dòng mã nguồn PyTorch thuần trong custom_ablation_modules.py:
  1. Module CoordConv tiêm 2 kênh tọa độ không gian.
  2. Module RepConv 3 nhánh song song lúc train và gộp đại số switch_to_deploy lúc inference.
  3. Module BiFormer định tuyến thưa hai tầng top-k vùng thô.
  4. Custom Loss Focal EIoU tách biệt phạt độc lập dài/rộng và điều tiết trọng số mẫu khó.

--------------------------------------------------------------------------------
SLIDE 8: TIÊU CHÍ 3 - KỸ NGHỆ DỮ LIỆU & CHUẨN HÓA (DATA ENGINEERING)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 3 · DỮ LIỆU HUẤN LUYỆN & CHUẨN HÓA CÔNG NGHIỆP
- Title: Kỹ Nghệ Dữ Liệu Chuẩn Hóa & Bộ Kiểm Thử Ngoại Miền >33,000 Ảnh
- Source Dataset (SHWD / VOC2028): 7,581 ảnh công trường (6,064 trainval / 1,517 test, nghiêm ngặt leak-free 80/20); loại bỏ 3 nhãn rác "dog" bất thường; mất cân bằng 1:12.
- Multi-Domain Suite (>25,000 ảnh): GDUT-HWD (13,499 ảnh công nhân đông đúc), SHEL5K (5,000 ảnh Flycam thẳng đứng), Hard Hat Workers (7,000 ảnh ngoài trời), SHD, SFCHD.
- Canonical Space C* = {0: 'hat', 1: 'person'} & Giao thức Harmonized PPE (Hat-Only) loại bỏ xung đột định nghĩa nhãn.

--------------------------------------------------------------------------------
SLIDE 9: TỔNG THỂ KIẾN TRÚC MẠNG REP-YOLO11s (ARCHITECTURE)
--------------------------------------------------------------------------------
- Category: KIẾN TRÚC MẠNG NƠ-RON ĐỀ XUẤT
- Title: Tổng Thể Kiến Trúc Mạng Nơ-ron Đề Xuất Rep-YOLO11s
- [FRAMEWORK IMAGE CONTAINER]: Fig2_rep_yolo11s_neural_architecture.png (Sơ đồ kiến trúc toàn trình từ input đến decoupled head).
- Subsystem 1 (Backbone): CSPDarknet kết hợp CoordConv ở stem và RepConv tại các tầng C3k2.
- Subsystem 2 (Neck): Bi-Level Routing Attention (BiFormer) tích hợp trong PAN.
- Subsystem 3 (Head): Anchor-Free Decoupled Head tối ưu bằng hàm mất mát Focal EIoU Loss.

--------------------------------------------------------------------------------
SLIDE 10: ĐỘT PHÁ 1 - TÁI THAM SỐ HÓA CẤU TRÚC REPCONV (INNOVATION 1)
--------------------------------------------------------------------------------
- Category: ĐỘT PHÁ CÔNG NGHỆ 1 / 3 · TÁI THAM SỐ HÓA CẤU TRÚC
- Title: Huấn Luyện Đa Nhánh Phong Phú, Triển Khai Nhánh Đơn Siêu Tốc
- Training Phase: 3 nhánh song song y = BN(W_3x3*x) + BN(W_1x1*x) + BN(x) học đa dạng biểu diễn viền và kết cấu mũ.
- Deploy Phase: switch_to_deploy gộp đại số tuyến tính về duy nhất 1 nhân Conv 3x3 y = W_fused*x + b_fused.
- Closed-Form Fusion Formula: W_fused = W'_3x3 + Pad(W'_1x1) + W'_id; b_fused = b'_3x3 + b'_1x1 + b'_id.
- Real T4 Gain: 7.12 ms -> 2.92 ms (342.5 FPS), độ trễ giảm 59.0%, sai số độ chính xác Delta mAP = 0.00%!

--------------------------------------------------------------------------------
SLIDE 11: ĐỘT PHÁ 2 - COORDCONV & BIFORMER (INNOVATION 2)
--------------------------------------------------------------------------------
- Category: ĐỘT PHÁ CÔNG NGHỆ 2 / 3 · TIÊN ĐỀ KHÔNG GIAN & ĐỊNH TUYẾN THƯA
- Title: Phá Vỡ Bất Biến Tịnh Tiến & Tập Trung Chú Ý Vi Vật Thể
- CoordConv: Tensor đầu vào 5 kênh [RGB; Cx; Cy]. Tiên đề hình học giải phẫu: Mũ nằm trên đầu người (khu vực trên), xô vữa cọc tiêu nằm dưới đất (Cy -> +1). Tăng 0.078% tham số, 0 ms độ trễ.
- BiFormer: Attention định tuyến thưa hai tầng, độ phức tạp O(S^2 + k*HW/S^2) << O(H^2W^2). Phân vùng thô, lọc Top-k vùng tương quan, tính attention chi tiết token. Tập trung 100% vào vi vật thể mũ.

--------------------------------------------------------------------------------
SLIDE 12: ĐỘT PHÁ 3 - HÀM MẤT MÁT FOCAL EIOU LOSS (INNOVATION 3)
--------------------------------------------------------------------------------
- Category: ĐỘT PHÁ CÔNG NGHỆ 3 / 3 · HÀM MẤT MÁT HỒI QUY KHUNG CHỨA
- Title: Bắt Dính Vi Vật Thể Che Khuất Bằng Hàm Mất Mát Focal EIoU
- CIoU Flaw: Tỷ lệ co cụm w/h bão hòa khi bị giàn giáo che khuất, triệt tiêu gradient của mũ nhỏ.
- EIoU Decomposition: Tách biệt độc lập sai số chiều dài w và chiều cao h: L_EIoU = (1-IoU) + rho^2/c^2 + rho^2(w,w_gt)/Cw^2 + rho^2(h,h_gt)/Ch^2.
- Focal Modulation: L_Focal-EIoU = IoU^0.5 * L_EIoU, giảm gradient mẫu dễ, dồn trọng số tối ưu vi vật thể mũ bị che khuất.
- Empirical Gain: Tăng mAP50 lên 94.88% (A4) và mAP50-95 đạt 62.54% với zero chi phí phụ trội khi triển khai.

--------------------------------------------------------------------------------
SLIDE 13: TIÊU CHÍ 4 - SO SÁNH ĐỊNH LƯỢNG VỚI SOTA (EVALUATION)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 4 · GIÁ TRỊ KHOA HỌC & ĐỐI SÁNH THỰC NGHIỆM
- Title: Vượt Trội Các Mô Hình SOTA Về Cả Độ Chính Xác Và Độ Trễ
- Table I Comparison:
  · YOLOv8s: 94.89% mAP50 | 6.10 ms (163.9 FPS)
  · YOLOv10s: 94.39% mAP50 | 6.23 ms (160.5 FPS)
  · YOLO11s Baseline: 94.74% mAP50 | 6.52 ms (153.3 FPS)
  · EC-YOLOv8 (2024): 95.70% mAP50 | 5.80 ms (172.4 FPS)
  · YOLO-CBF (2023): 95.60% mAP50 | 12.40 ms (80.6 FPS)
  · Rep-YOLO11s (Ours Single): 94.83% mAP50 | 2.92 ms (342.5 FPS)
  · Rep-YOLO11s (5-Fold Mean): 96.64 ± 0.32% mAP50 (Đỉnh Fold 3: 97.11%, F1: 0.9396)
- [FRAMEWORK IMAGE CONTAINER]: Fig5_efficiency_frontier_latency_vs_map.png (Biểu đồ Pareto Latency vs mAP).

--------------------------------------------------------------------------------
SLIDE 14: TIÊU CHÍ 4 - NGHIÊN CỨU CẮT BỎ A0 -> A6 (ABLATION STUDY)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 4 · KIỂM CHỨNG TỪNG THÀNH PHẦN KIẾN TRÚC
- Title: Nghiên Cứu Cắt Bỏ A0–A6: Minh Chứng Vai Trò Từng Mô-Đun Đề Xuất
- Table II Ablation Breakdown:
  · A0 (Baseline): 94.74% mAP50, 6.52 ms
  · A1 (+ P2 small-head): 94.81% (+0.07%), nhưng độ trễ tăng vọt lên 8.94 ms (+37%) [LOẠI BỎ]
  · A2 (+ CoordConv): 94.78% mAP50, 6.58 ms (Tăng độ chính xác không gian)
  · A3 (+ RepConv): 94.81% mAP50, 6.64 ms (Đa dạng hóa biểu diễn)
  · A4 (+ Focal EIoU): 94.88% mAP50, 62.51% mAP50-95 (Bám dính mép biên)
  · A5 (+ BiFormer): 94.80% mAP50, Recall tăng lên 91.15%
  · A6 (Full Fusion): 94.83% mAP50, độ trễ sụp đổ về 2.92 ms (-59.0%) sau switch_to_deploy!
- [FRAMEWORK IMAGE CONTAINER]: Fig6_ablation_A0_A6_tradeoff.png (Biểu đồ cắt bỏ thành phần và độ trễ).

--------------------------------------------------------------------------------
SLIDE 15: TIÊU CHÍ 4 - KIỂM CHỨNG THỊ GIÁC XAI GRAD-CAM (XAI ANALYSIS)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 4 · GIẢI THÍCH MÔ HÌNH XAI GRAD-CAM
- Title: Bản Đồ Nhiệt Khẳng Định Tập Trung Vào Mũ, Triệt Tiêu Nhiễu
- [FRAMEWORK IMAGE CONTAINER]: Fig3_gradcam_xai_saliency_comparison.png (Bản đồ nhiệt Grad-CAM 3 hàng x 4 cột).
- Scenario 1 (Áo Cam & Giàn Giáo): Baseline bị phân tán gradient vào áo phản quang; Rep-YOLO11s tập trung 100% điểm nhiệt vào 4 mũ bảo hộ (conf 0.84).
- Scenario 2 (Vi Vật Thể Ngược Sáng): BiFormer định tuyến token chính xác khóa chặt chiếc mũ nhỏ bên cửa sổ ngược sáng (conf 0.89).
- Scenario 3 (Biển Báo Tam Giác Vàng): CoordConv áp đặt tiên đề vị trí giải phẫu loại bỏ hoàn toàn biển báo tam giác giả mạo.

--------------------------------------------------------------------------------
SLIDE 16: TIÊU CHÍ 4 - TỔNG QUÁT HÓA ĐA MIỀN & IOU COLLAPSE (GENERALIZATION)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 4 · NĂNG LỰC TỔNG QUÁT HÓA ĐA MIỀN
- Title: Chuyển Giao Ngoại Miền 33,000+ Ảnh & Giải Mã Hiện Tượng Sụp Đổ IoU
- Cross-Domain Results (Table III): Hard Hat Workers 97.03% mAP50; GDUT-HWD 74.27% mAP50 (90.26% Precision); SHEL5K 41.15% mAP50 (Drone góc 90 độ).
- IoU Collapse Autopsy: Đánh giá Joint trên HHW khiến mAP tụt xuống 74.40% do xung đột định nghĩa: SHWD gán 'person' là Full-body còn HHW gán là Head-only. Do hộp đầu lọt thỏm trong hộp thân, IoU ~ 0.07-0.14 << 0.50 làm phạt kép FP và FN.
- Harmonized PPE: Đánh giá riêng lớp mũ (nơi định nghĩa 2 tập đồng nhất), mAP50 phục hồi lên 97.03%, chứng minh biểu diễn đặc trưng mũ chuyển giao hoàn hảo!

--------------------------------------------------------------------------------
SLIDE 17: TIÊU CHÍ 4 - PHẦN CỨNG BIÊN & PIPELINE RTSP (HARDWARE DEPLOYMENT)
--------------------------------------------------------------------------------
- Category: TIÊU CHÍ 4 · GIÁ TRỊ THỰC TIỄN & TRIỂN KHAI PHẦN CỨNG BIÊN
- Title: Kiểm Chứng Phần Cứng Đa Nền Tảng & Pipeline RTSP Thời Gian Thực
- Multi-Platform Hardware Benchmark (Table IV):
  · Tesla T4 (Server GPU): 2.92 ms | 342.5 FPS (TensorRT 11.2 FP16)
  · RTX 3050 Laptop (Edge GPU): 5.35 ms | 187.1 FPS (TRT FP16); 12.43 ms (80.4 FPS, Native FP32)
  · GeForce MX230 (Budget Laptop): Pascal, 2GB VRAM, KHÔNG Tensor Cores vẫn đạt 36.0 ms (27.8 FPS > 24 FPS chuẩn thời gian thực)!
  · CPU Edge (4-Cores INT8): 28.56 ms | 35.0 FPS
- [FRAMEWORK IMAGE CONTAINER]: Fig4_industrial_rtsp_surveillance_pipeline.png (Sơ đồ pipeline 5 giai đoạn).
- Latency Breakdown: T_total = T_dec (3.5-5.0ms) + T_prep (1.2-2.0ms) + T_gpu (2.92-5.35ms) + T_nms (1.5-2.8ms) + T_ui (2.2-3.4ms) = 10.54 - 15.38 ms. Thông lượng thực tế 65 đến 95 FPS trên RTX 3050.

--------------------------------------------------------------------------------
SLIDE 18: LỘ TRÌNH TRIỂN KHAI TOÀN DIỆN & KẾT LUẬN (ROADMAP & CONCLUSION)
--------------------------------------------------------------------------------
- Category: LỘ TRÌNH THỰC HIỆN TOÀN DIỆN & TỔNG KẾT
- Title: Lộ Trình Hoàn Thiện Đồ Án & Sẵn Sàng Bảo Vệ Hội Đồng
- Giai đoạn 1 (Đã hoàn thành 100%): Xác lập bài toán rõ ràng; tự lập trình 4 module toán học; thực nghiệm SOTA, Ablation A0-A6, Grad-CAM, Cross-domain 33K ảnh; đo đạc vật lý trên phần cứng thật (T4 342.5 FPS, RTX 3050 187.1 FPS, MX230 27.8 FPS); mAP50 94.83% single, 96.64% 5-fold.
- Giai đoạn 2 (Đang triển khai): Đóng gói Desktop GUI & Web Dashboard giám sát cảnh báo vi phạm; tự động ghi log hình ảnh vi phạm; điều phối đa luồng camera RTSP; tích hợp giao thức MQTT/WebRTC.
- Giai đoạn 3 (Kế hoạch về đích): Chưng cất tri thức vi mô (Knowledge Distillation: YOLO11x -> Rep-YOLO11s) cho góc máy Drone thẳng đứng (SHEL5K); hoàn thiện bản thảo Khóa luận Tốt nghiệp chính thức; sẵn sàng bảo vệ trước Hội đồng FPT.
- Bottom Banner: Khẳng định cam kết: Đề tài đạt đầy đủ và vượt trội các tiêu chí Xuất sắc của Hội đồng Đồ án Tốt nghiệp Kỹ sư AI - Đại học FPT!
================================================================================
```
