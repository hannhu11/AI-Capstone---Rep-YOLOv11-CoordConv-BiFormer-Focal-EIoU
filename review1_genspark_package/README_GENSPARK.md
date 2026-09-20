# 📦 HƯỚNG DẪN TRỌN GÓI SOẠN SLIDE VÀ BẢO VỆ ĐỒ ÁN TỐT NGHIỆP GIAI ĐOẠN 1 (ĐATN AI - FPT UNIVERSITY)

Thư mục này chứa đầy đủ tài nguyên, hình ảnh trích xuất trực tiếp từ file PDF, bảng biểu thực nghiệm và prompt tối ưu hóa cho công cụ **Genspark AI Slides** phục vụ Giai đoạn 1 Đồ án Tốt nghiệp.

---

## 📁 1. CẤU TRÚC THƯ MỤC `review1_genspark_package`

```text
review1_genspark_package/
│
├── figures/                                              # Hình ảnh trích xuất 100% từ Paper PDF
│   ├── Fig1_site_overview_challenges.jpg                 # Hình 1: Thách thức công trường thực tế
│   ├── Fig2_rep_yolo11s_neural_architecture.png          # Hình 2: Toàn bộ kiến trúc mạng Rep-YOLO11s
│   ├── Fig3_gradcam_xai_saliency_comparison.png          # Hình 3: Bản đồ nhiệt Grad-CAM XAI
│   ├── Fig4_industrial_rtsp_surveillance_pipeline.png    # Hình 4: Quy trình xử lý luồng RTSP thời gian thực
│   ├── Fig5_efficiency_frontier_latency_vs_map.png       # Hình 5: Biểu đồ đánh đổi Latency vs mAP
│   └── Fig6_ablation_A0_A6_tradeoff.png                  # Hình 6: Biểu đồ cột mAP và Latency Ablation
│
├── tables/                                               # Các bảng số liệu định lượng (định dạng CSV & Markdown)
│   ├── Table1_sota_comparison.csv / .md                  # Bảng 1: So sánh với các mô hình SOTA
│   ├── Table2_ablation_study.csv / .md                   # Bảng 2: Nghiên cứu cắt bỏ thành phần A0 -> A6
│   ├── Table3_cross_domain_generalization.csv / .md      # Bảng 3: Kiểm thử ngoại miền 5 tập dữ liệu
│   └── Table4_deployment_benchmark.csv / .md             # Bảng 4: Benchmark phần cứng (T4, 3050, MX230, CPU)
│
├── REVIEW1_QA_DEFENSE_REPORT.md                          # Báo cáo trả lời chi tiết toàn bộ câu hỏi Giai đoạn 1
├── SLIDE_DECK_CONTENT.md                                 # Kịch bản chi tiết 16 Slide (nội dung + chỉ định ảnh)
├── GENSPARK_MASTER_PROMPT_FPT_DEFENSE.txt                # Prompt chuẩn 18 slide không chứa từ cấm
├── GENSPARK_MASTER_PROMPT_FPT_DEFENSE.md                 # Hướng dẫn chi tiết kèm prompt 18 slide
└── README_GENSPARK.md                                    # Hướng dẫn thao tác này
```

---

## 🎯 2. LỰA CHỌN SKILL PHÙ HỢP NHẤT TRÊN GENSPARK AI SLIDES

Trong danh sách các Skills của Genspark hiển thị trên màn hình của bạn:
1. **LỰA CHỌN TỐT NHẤT (KHUYẾN NGHỊ SỐ 1)**:
   - **Tên Skill:** `Write a Senior Capstone Defense like a Top R1-University Senior` (Nằm trong mục **Coursework**).
   - **Lý do lựa chọn:** Đây là template thiết kế chuyên biệt cho buổi bảo vệ đồ án tốt nghiệp kỹ sư / cử nhân tại các trường đại học nghiên cứu hàng đầu (R1-University). Cấu trúc của skill này phân tầng mạch lạc: Đặt vấn đề $\to$ Khảo sát tài liệu $\to$ Phương pháp & Kiến trúc đề xuất $\to$ Kết quả thực nghiệm định lượng $\to$ Tính khả thi triển khai phần cứng $\to$ Lộ trình các giai đoạn tiếp theo. Template này hỗ trợ hiển thị bảng biểu học thuật rất đẹp và bố cục khoa học.
2. **LỰA CHỌN DỰ PHÒNG SỐ 2**:
   - **Tên Skill:** `Write an Academic Thesis Deck like a Top Journal Author` (Mục **Academic**). Phù hợp nếu bạn muốn nhấn mạnh khía cạnh bài báo nghiên cứu khoa học chuẩn Q1 quốc tế.
3. **LỰA CHỌN DỰ PHÒNG SỐ 3**:
   - **Tên Skill:** `Write an Engineering Senior Thesis Defense like a Top Engineering Senior the Night Before Defense` (Mục **Coursework**). Nhấn mạnh khía cạnh kỹ thuật phần cứng và thời gian thực.

---

## 🚀 3. HƯỚNG DẪN 3 BƯỚC TẠO SLIDE TỰ ĐỘNG TRÊN GENSPARK

* **Bước 1:** Mở trang tạo slide của Genspark (`https://www.genspark.ai/slides`), chọn Skill `Write a Senior Capstone Defense like a Top R1-University Senior`. Chọn chế độ **Professional**, tỷ lệ màn hình **16:9** hoặc **Auto Ratio**.
* **Bước 2:** Mở tệp `GENSPARK_MASTER_PROMPT_FPT_DEFENSE.txt`, nhấn `Ctrl + A` và `Ctrl + C` để copy toàn bộ nội dung. Dán vào ô nhập lệnh (prompt) của Genspark.
* **Bước 3:** Nhấn nút Generate. Sau khi Genspark tạo xong khung slide:
   - Bạn có thể tải các ảnh trong thư mục `figures/` lên các slide tương ứng.
   - Các slide đối sánh và cắt bỏ đã có sẵn bảng số liệu định lượng chuẩn xác 100% lấy từ bài báo khoa học Q1.

---

## 💡 4. CHIẾN LƯỢC TRẢ LỜI CỦA BẠN TRƯỚC HỘI ĐỒNG ĐÁNH GIÁ GIAI ĐOẠN 1

1. **Về tính rõ ràng của bài toán:** Nêu bật 4 nút thắt kỹ thuật mà CNN thông thường gặp phải: (1) Vật thể siêu nhỏ $<20\times20$ px; (2) Mất cân bằng $1:12$; (3) Nhầm lẫn xô vàng/cọc tiêu dưới đất; (4) Nghẽn băng thông trên thiết bị biên.
2. **Về tính khả thi phần cứng:** Trình bày bảng Table IV với điểm nhấn: Ngay cả trên Laptop văn phòng yếu nhất (**NVIDIA GeForce MX230, 2GB VRAM, kiến trúc Pascal cũ không có Tensor Cores**), mô hình chạy PyTorch Native FP32 vẫn đạt **$27.8$ FPS ($36.0$ ms)**, vượt ngưỡng an toàn thời gian thực ($>24$ FPS). Trên RTX 3050 Laptop đạt **$187.1$ FPS**, luồng RTSP hoàn chỉnh đạt **$65 - 95$ FPS**.
3. **Về tính khoa học:** Giải thích hiện tượng **Sụp đổ IoU ($IoU \approx 0.07-0.14$)** khi đánh giá nhãn lệch (Full-body vs Head-only) trên tập ngoại vi Hard Hat Workers, và cách nhóm chuẩn hóa Harmonized PPE để đạt $97.03\%$ mAP50.
