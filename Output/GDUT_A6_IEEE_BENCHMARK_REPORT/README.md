# Benchmark Thực Nghiệm GDUT-HWD: Rep-YOLO11s A6 vs. Rep-YOLOv8s A6
**Tập Dữ Liệu:** GDUT-HWD (Guangdong University of Technology Hardhat Wear Detection)  
**Phân Loại Thực Nghiệm:** Stage 2 Full-Fusion Ablation (A6 = CoordConv + RepConv + BiFormer Attention + Focal-EIoU Loss)  
**Tác Vụ:** Giám sát an toàn lao động, nhận diện 5 lớp mũ bảo hộ và phát hiện trực tiếp vi phạm không đội mũ (`none`).

---

## 1. Tổng Quan & Nguồn Gốc Trọng Số (Provenance)

- **Checkpoint gốc Kaggle:** `Output/GDUT_Bộ số 1_stage2-full-train-run-a6/GDUT_A6_CLEAN_VAL_RESULTS/`
  - YOLO11s A6: `YOLO11s/weights/GDUT_A6_YOLO11s_best.pt` (SHA-256: `61a93e428d5c17613d80880e8aa0a5c4db85e6ec98f4409a5f0c371c565b0414`)
  - YOLOv8s A6: `YOLOv8s/weights/GDUT_A6_YOLOv8s_best.pt` (SHA-256: `7fa5749801a3459bf4d270349483f4235eddd1eed1e65d9cd8140d2ab4f1f093`)
- **Trọng số Deploy độc lập (Clean Deployable Checkpoints):**
  - `Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt` (36.55 MB, FP32)
  - `Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolov8s_gdut_a6_deploy.pt` (46.79 MB, FP32)
- **Cải tiến triển khai (Deployment Decoupling):**
  - Toàn bộ các nhánh re-parameterization đa nhánh (3x3 + 1x1 + identity) của `RepConv` đã được kết hợp (fused) thành một lớp `Conv2d` 3x3 duy nhất qua `switch_to_deploy()`.
  - Loại bỏ hoàn toàn sự phụ thuộc unpickler vào các module nghiên cứu nội bộ (`research_trainer`, `model_code`). Checkpoint có thể nạp trực tiếp qua `ultralytics.YOLO()` chuẩn mà không phát sinh lỗi pickling.

---

## 2. Taxonomy & Kiến Trúc Bộ Dữ Liệu GDUT 5-Class

Khác với bộ dữ liệu CHV (7 lớp: người, áo phản quang, 5 màu mũ), bộ dữ liệu GDUT là tập dữ liệu chuyên biệt về đầu/mũ bảo hộ gồm 5 lớp:
```python
{
    0: "blue",    # Mũ bảo hộ màu xanh dương
    1: "white",   # Mũ bảo hộ màu trắng
    2: "yellow",  # Mũ bảo hộ màu vàng
    3: "red",     # Mũ bảo hộ màu đỏ
    4: "none"     # Đầu trần / Không đội mũ bảo hộ (VI PHẠM TRỰC TIẾP)
}
```

### Dynamic Taxonomy Adapter trong Hệ Thống
Hệ thống giám sát CCTV tại `src/engine/detector.py`, `src/engine/spatial_safety.py`, và `src/engine/pipeline.py` đã được tích hợp bộ điều hợp phân loại động:
1. **Chống nhận diện sai lớp 0 (`blue`):** Hệ thống kiểm tra sự tồn tại của lớp người (`has_person_class`). Vì GDUT không có lớp người, `cls == 0` tuyệt đối không bị gán nhầm thành công nhân (person).
2. **Chống nhận diện sai lớp 1 (`white`):** Lớp 1 trong GDUT là mũ trắng, không bị nhầm lẫn với áo phản quang (safety vest).
3. **Helmet Violation Mode (Chế độ vi phạm mũ bảo hộ trực tiếp):**
   - Khi chạy mô hình 5-class không có bounding box toàn thân công nhân, ByteTrack không nhận diện được box người để ghép cặp.
   - `SpatialSafetyEngine` tự động kích hoạt **Helmet Violation Mode**:
     - Các phát hiện `blue`, `white`, `yellow`, `red` được phân loại là **COMPLIANT** (tuân thủ).
     - Phát hiện `none` được phân loại trực tiếp là **VIOLATION (MISSING_SAFETY_HELMET)**.
   - Áp dụng cơ chế khử rung không gian (Spatial-Centroid Debouncing, ngưỡng cách tâm chuẩn hóa $0.08$) và làm nguội ($30$ giây) để ngăn chặn việc cảnh báo lặp lại liên tục trên cùng một vị trí.

---

## 3. Kiến Trúc Docker Đơn Nhất (1 Docker Container vs 2 Containers)

**Câu hỏi thiết kế của người dùng:** Cần tạo 2 docker containers hay dùng 1 docker container?  
**Kết luận kỹ thuật tối ưu:** **Chỉ cần duy nhất 1 Docker Container**.
- Mã nguồn hệ thống đã hỗ trợ hoán đổi trọng số nóng (hot-swapping model weights) và cấu hình qua biến môi trường `.env`:
  ```bash
  # Để chạy mô hình GDUT Rep-YOLO11s A6
  MODEL_PATH=Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt
  ```
- Bộ Dynamic Taxonomy Adapter tự động quét thuộc tính `model.names`:
  - Nếu phát hiện tập 7 lớp CHV: Chạy chế độ suy luận không gian 2 tầng (Spatial Relational Engine) với ByteTrack theo dõi công nhân + ghép nối mũ & áo phản quang.
  - Nếu phát hiện tập 5 lớp GDUT: Chạy chế độ trực tiếp (Helmet Violation Mode), xử lý vi phạm không mũ `none` mà không gây treo/crash ByteTrack.
- Do đó, việc duy trì 1 container giúp tiết kiệm tối đa tài nguyên GPU VRAM, đơn giản hóa quy trình vận hành và tránh xung đột cổng dịch vụ.

---

## 4. Danh Mục Tài Liệu Báo Cáo

- [gdut_a6_evaluation_report.md](file:///c:/Users/ADMIN/Downloads/capstone%20AI/Output/GDUT_A6_IEEE_BENCHMARK_REPORT/gdut_a6_evaluation_report.md): Báo cáo chi tiết toàn diện số liệu thực nghiệm, phân tích so sánh và nội dung trình chiếu slide cho hội nghị AAIML.
- [gdut_a6_ieee_tables.tex](file:///c:/Users/ADMIN/Downloads/capstone%20AI/Output/GDUT_A6_IEEE_BENCHMARK_REPORT/gdut_a6_ieee_tables.tex): Bảng biểu định dạng chuẩn IEEE Transactions / Conference với gói `booktabs` sẵn sàng nhúng vào bài báo LaTeX.
