# 📋 BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 3 (IITU DRONE 5-CLASS)
**Đồ án Tốt nghiệp Capstone AI** - SV: Nguyễn Hàn Như | GVHD: ThS. Nguyễn Xuân Huy

### 1. Thông số Dataset IITU sau chuẩn hóa:
- **Tổng số ảnh hợp lệ**: 1657 ảnh
- **Tập Train**: 1325 ảnh (80%)
- **Tập Val**:   165 ảnh (10%)
- **Tập Test**:  167 ảnh (10%)

### 2. Kết quả Đánh giá Benchmark trên Tập Test Độc lập:
| Chỉ số (Metric) | Giá trị thực nghiệm |
| :--- | :---: |
| **Precision** | **90.70%** |
| **Recall** | **90.17%** |
| **mAP@50** | **93.50%** |
| **mAP@50-95** | **45.51%** |

### 3. Đánh giá chi tiết 5 màu mũ bảo hộ:
| Class ID | Màu Mũ Bảo Hộ | Precision (%) | Recall (%) | mAP@50 (%) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `blue_helmet` (Xanh dương) | 86.30 | 91.11 | 92.11 |
| 1 | `green_helmet` (Xanh lá) | 94.55 | 89.67 | 95.59 |
| 2 | `red_helmet` (Đỏ) | 89.05 | 86.65 | 92.22 |
| 3 | `white_helmet` (Trắng) | 89.74 | 92.20 | 95.75 |
| 4 | `yellow_helmet` (Vàng) | 93.87 | 91.24 | 91.85 |

### 4. Kết luận khoa học:
1. IITU là tập dữ liệu tuyệt vời để nhận diện màu sắc mũ từ góc quay flycam/drone trên cao (0°, 45°, 90°) và môi trường ban đêm.
2. Tuy nhiên, tập này HOÀN TOÀN KHÔNG CÓ nhãn người không đội mũ (`none`) và áo bảo hộ (`vest`).
3. Khuyến nghị: Phải dùng phương án Bộ 4 (Tổng hợp GDUT + IITU + Vest) để giải quyết toàn diện bài toán bảo vệ an toàn công trường.
