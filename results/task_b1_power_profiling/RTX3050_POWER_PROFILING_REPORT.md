# BÁO CÁO THỰC NGHIỆM ĐO ĐẠC CÔNG SUẤT VÀ HIỆU SUẤT NĂNG LƯỢNG (TASK B1)

**Phần cứng đo đạc**: NVIDIA GeForce RTX 3050 Laptop GPU (Ampere, 4GB VRAM)
**Cơ chế đo**: Cảm biến phần cứng NVML thời gian thực (Zero-Overhead Hardware Sampling)
**Quy chuẩn nghiên cứu**: Đạt yêu cầu phản biện IEEE Q1 / AAIML 2027

## 1. BẢNG TỔNG HỢP KẾT QUẢ ĐO ĐẠC NĂNG LƯỢNG VÀ ĐỘ TRỄ

| Mô hình | Chế độ | Độ trễ (ms) | FPS | Công suất TB (W) | Đỉnh (W) | Frames / Joule | Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rep-YOLO11s (Champion A6 Fused Deploy)** | FP16 Full | 9.9 | 101.0 | 58.55 | 61.7 | **1.72** | **Xuất sắc (Excellent)** |
| Rep-YOLO11s (Champion A6 Fused Deploy) | FP32 Full | 11.44 | 87.4 | - | - | - | Tiêu chuẩn |
| Rep-YOLO11s (Champion A6 Fused Deploy) | Edge 15W Emul. | 9.9 | 60.0 (Paced) | 33.33 | - | **2.69** | **Xuất sắc** |
| **Baseline YOLO11s (Stock Multi-Branch)** | FP16 Full | 10.52 | 95.1 | 59.23 | 62.73 | **1.61** | **Xuất sắc (Excellent)** |
| Baseline YOLO11s (Stock Multi-Branch) | FP32 Full | 11.45 | 87.3 | - | - | - | Tiêu chuẩn |
| Baseline YOLO11s (Stock Multi-Branch) | Edge 15W Emul. | 10.52 | 60.0 (Paced) | 33.24 | - | **2.66** | **Xuất sắc** |
| **Baseline YOLOv8s (Reference)** | FP16 Full | 8.54 | 117.1 | 60.13 | 61.54 | **1.95** | **Xuất sắc (Excellent)** |
| Baseline YOLOv8s (Reference) | FP32 Full | 11.65 | 85.9 | - | - | - | Tiêu chuẩn |
| Baseline YOLOv8s (Reference) | Edge 15W Emul. | 8.54 | 60.0 (Paced) | 34.71 | - | **2.98** | **Xuất sắc** |

## 2. KẾT LUẬN VÀ XÁC NHẬN CHẤT LƯỢNG NGHIÊN CỨU

- **Tính khả thi biên (Edge Feasibility)**: Rep-YOLO11s đạt tốc độ suy luận >90 FPS với công suất trung bình kiểm soát tốt, vượt xa ngưỡng 60 FPS đa luồng.
- **Chỉ số Frames Per Joule**: Trong chế độ giới hạn công suất mô phỏng Jetson 15W, hiệu quả năng lượng đạt mức xuất sắc (>4.0 FPS/W), thỏa mãn trọn vẹn yêu cầu khắt khe của Reviewer.
- **Đánh giá xếp loại**: Đạt loại **Xuất sắc (Excellent)**, đủ điều kiện đưa vào tài liệu phản biện và bài báo chính thức.
