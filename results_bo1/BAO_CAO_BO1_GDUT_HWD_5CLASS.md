# 📋 BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 1 (GDUT-HWD 5-CLASS)
**Đồ án Tốt nghiệp Capstone AI** - SV: Nguyễn Hàn Như | GVHD: ThS. Nguyễn Xuân Huy

### 1. Thông số Dataset GDUT-HWD sau chuẩn hóa:
- **Tổng số ảnh sau làm sạch**: 3160 (Đã loại bỏ 14 ảnh trùng MD5)
- **Tập Train**: 2528 ảnh (80%)
- **Tập Val**:   316 ảnh (10%)
- **Tập Test**:  316 ảnh (10%)

### 2. Kết quả Đánh giá Benchmark trên Tập Test Độc lập:
| Chỉ số (Metric) | Giá trị thực nghiệm |
| :--- | :---: |
| **Precision** | **88.67%** |
| **Recall** | **76.66%** |
| **mAP@50** | **82.97%** |
| **mAP@50-95** | **53.93%** |

### 3. Đánh giá chi tiết từng lớp:
| Class ID | Tên Lớp | Precision (%) | Recall (%) | mAP@50 (%) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `no_helmet` (Vi phạm) | 93.97 | 67.86 | 80.08 |
| 1 | `red_helmet` | 81.78 | 72.98 | 79.53 |
| 2 | `yellow_helmet` | 88.84 | 84.79 | 89.09 |
| 3 | `white_helmet` | 88.30 | 88.29 | 92.30 |
| 4 | `blue_helmet` | 90.46 | 69.38 | 73.84 |

### 4. Kết luận khoa học:
1. Tập GDUT-HWD hỗ trợ phát hiện tốt cả 4 màu sắc mũ cơ bản và phát hiện vi phạm không đội mũ (`no_helmet`).
2. Tuy nhiên, tập này thiếu lớp Mũ xanh lá (`green_helmet`) và chưa có Áo bảo hộ phản quang (`vest`). Cần kết hợp với IITU và STANDARDIZED_CHV ở Bộ 4 để hoàn thiện bài toán PPE toàn diện.
