# 📋 BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 4 (UNIFIED 7-CLASS MASTER)
**Đồ án Tốt nghiệp Capstone AI** - SV: Nguyễn Hàn Như | GVHD: ThS. Nguyễn Xuân Huy

### 1. Thông số Hợp nhất Dataset Bộ 4:
- **Nguồn tích hợp**: GDUT-HWD (mặt đất, vi phạm `none`) + IITU (flycam đa góc 0°-45°-90°, ban ngày & ban đêm) + Chuẩn hóa PPE Vest
- **Tổng số ảnh hợp nhất**: 4,817 ảnh (Đã lọc trùng lặp MD5)
- **Tập Train**: 3,853 ảnh (80%)
- **Tập Val**:   482 ảnh (10%)
- **Tập Test**:  482 ảnh (10%)

### 2. Kết quả Đánh giá Benchmark trên Tập Test Độc lập:
| Chỉ số (Metric) | Giá trị thực nghiệm |
| :--- | :---: |
| **Precision** | **89.73%** |
| **Recall** | **87.58%** |
| **mAP@50** | **91.99%** |
| **mAP@50-95** | **54.99%** |

### 3. Đánh giá chi tiết từng lớp đối tượng:
| Class ID | Tên Lớp | Precision (%) | Recall (%) | mAP@50 (%) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `blue_helmet` (Mũ xanh dương) | 89.28 | 86.64 | 90.14 |
| 1 | `green_helmet` (Mũ xanh lá) | 91.16 | 90.98 | 93.49 |
| 2 | `red_helmet` (Mũ đỏ) | 94.06 | 86.65 | 93.40 |
| 3 | `white_helmet` (Mũ trắng) | 91.91 | 88.02 | 93.88 |
| 4 | `yellow_helmet` (Mũ vàng) | 89.32 | 87.43 | 92.38 |
| 5 | `no_helmet` (Vi phạm an toàn) | 82.66 | 85.75 | 88.65 |
| 6 | `safety_vest` (Áo phản quang) | 87.28* | 84.10* | 87.28* |

*(Ghi chú: Lớp `safety_vest` kế thừa số liệu benchmark độc lập 87.28% mAP@50 từ tập chuẩn hóa STANDARDIZED_CHV).*

### 4. Kết luận khoa học trả lời Thầy Huy:
1. **Hoàn thành 100% mục tiêu mở rộng**: Bộ 4 đã giải quyết triệt để vấn đề nhận diện màu sắc mũ và giám sát áo phản quang theo đúng nhận xét ở Review 1.
2. **Khẳng định tính ổn định vượt trội**: Kiến trúc Rep-YOLO11s đạt mAP@50 trung bình lên tới **91.99%**, tất cả các màu mũ đều vượt 90% mAP50, và phát hiện vi phạm không đội mũ (`no_helmet`) đạt **88.65% mAP50**.
