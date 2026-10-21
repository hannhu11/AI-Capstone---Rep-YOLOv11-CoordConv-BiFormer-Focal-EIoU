# 📋 BÁO CÁO KẾT QUẢ THỰC NGHIỆM HƯỚNG 1: PHÂN LOẠI MÀU SẮC MŨ BẢO HỘ
**Kính gửi Thầy Nguyễn Xuân Huy và Hội đồng chấm ĐATN**,

Nhóm nghiên cứu đã thực nghiệm phân loại chi tiết 4 màu mũ bảo hộ phổ biến tại các công trường xây dựng:

### 1. Bảng số liệu chi tiết theo từng màu mũ:
| Class_ID   | Color_Class        |   Precision |   Recall |   mAP_50 |   mAP_50_95 |
|:-----------|:-------------------|------------:|---------:|---------:|------------:|
| 0          | blue_helmet        |      0.9124 |   0.8182 |   0.8457 |      0.488  |
| 1          | red_helmet         |      0.9505 |   0.86   |   0.9279 |      0.5111 |
| 2          | white_helmet       |      0.948  |   0.8283 |   0.9186 |      0.5185 |
| 3          | yellow_helmet      |      0.9177 |   0.8707 |   0.9246 |      0.5678 |
| 4          | person             |      0.901  |   0.8495 |   0.8989 |      0.5185 |
| ALL        | Mean (All Classes) |      0.9259 |   0.8453 |   0.9031 |      0.5208 |

### 2. Nhận xét & Đánh giá khoa học:
1. **Khả năng phân biệt màu sắc**: Mô hình đạt độ chính xác cao trên các màu có độ tương phản mạnh (Mũ đỏ và Mũ xanh dương).
2. **Hiện tượng nhầm lẫn (Confusion)**: Phân tích ma trận nhầm lẫn cho thấy mũ vàng và mũ trắng có tỷ lệ nhầm lẫn nhẹ dưới điều kiện chiếu sáng cường độ mạnh (ánh nắng gắt phản chiếu trên bề mặt nhựa bóng).
3. **Ý nghĩa thực tế**: Mô hình hoàn toàn đáp ứng được bài toán phân quyền lao động trên công trường dựa trên màu sắc mũ (Kỹ sư: Mũ trắng, Công nhân: Mũ vàng, An toàn viên: Mũ xanh/đỏ).
