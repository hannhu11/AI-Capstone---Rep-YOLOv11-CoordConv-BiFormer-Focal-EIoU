# Rep-YOLO11s: Edge Hardware Deployment Benchmark

Báo cáo này trình bày chi tiết các thông số kỹ thuật và kết quả đánh giá (benchmark) tốc độ suy luận của mô hình **Rep-YOLO11s** khi triển khai thực tế trên thiết bị phần cứng có cấu hình giới hạn (Edge Device/Low-end Laptop).

## 1. Thông số phần cứng (Hardware Specifications)
Hệ thống thử nghiệm là một Laptop văn phòng tiêu chuẩn, đại diện cho các thiết bị Edge Computing với tài nguyên hạn chế:
*   **CPU:** Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz (4 Cores)
*   **RAM:** 8.0 GB
*   **GPU:** NVIDIA GeForce MX230 (Kiến trúc Pascal)
*   **VRAM:** 2 GB GDDR5

## 2. Môi trường thực thi (Execution Environment)
*   **Framework:** PyTorch Native
*   **Precision (Độ phân giải tính toán):** FP32 (Single-Precision). 
    *   *Ghi chú:* Chế độ FP16 (Half-Precision) đã được tắt vì kiến trúc Pascal (MX230) không có nhân Tensor Cores hỗ trợ phần cứng cho FP16. Việc sử dụng FP32 nguyên bản giúp tránh lỗi thắt cổ chai do mô phỏng phần mềm.
*   **Optimization:** `torch.backends.cudnn.benchmark = True` (Tối ưu hóa thuật toán Convolution tự động).
*   **Video Test:** `construction_site_workers_1080p.mp4`

## 3. Kết quả Benchmark (Performance Metrics)

Dưới đây là bảng tổng hợp độ trễ (Latency) và băng thông khung hình (Throughput/FPS) đo được qua quá trình nội suy trực tiếp từ GPU. (Mức FPS ghi nhận đạt ngưỡng ~27.5 - 28 FPS).

| Hardware Target | Execution Engine | Resolution | Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| NVIDIA MX230 Laptop (Edge) | PyTorch Native FP32 | 640 × 640 | ~35.71 - 37.03 | ~27.0 - 28.0 |

### Chi tiết phân tích độ trễ (Latency Breakdown) trung bình:
*   **Pre-process (Xử lý ảnh đầu vào):** ~2.5 ms
*   **Inference (Thời gian GPU tính toán):** ~32.0 ms
*   **Post-process (Lọc NMS):** ~1.5 ms
*   **Total Latency (Độ trễ tổng):** ~36.0 ms

## 4. Kết luận
Mặc dù triển khai trên nền tảng phần cứng rất yếu (MX230 - 2GB VRAM), mô hình Rep-YOLO11s vẫn thể hiện hiệu năng xuất sắc khi duy trì được mức **~28 FPS** ở độ phân giải gốc `640x640`. Mức Throughput này hoàn toàn đáp ứng được tiêu chuẩn thời gian thực (Real-time) cho các ứng dụng giám sát an toàn lao động tại công trường (mức tối thiểu thường yêu cầu >24 FPS), chứng minh tính khả thi của mô hình khi triển khai trên các thiết bị Edge giá rẻ.
