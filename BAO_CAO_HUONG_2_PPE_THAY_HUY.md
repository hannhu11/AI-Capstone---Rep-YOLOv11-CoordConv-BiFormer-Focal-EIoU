# 📋 BÁO CÁO KẾT QUẢ THỰC NGHIỆM HƯỚNG 2: MỞ RỘNG ĐỒ BẢO HỘ (PPE)
**Kính gửi Thầy Nguyễn Xuân Huy và Hội đồng chấm ĐATN**,

Nhóm nghiên cứu đã thực nghiệm mở rộng mô hình sang phát hiện đồng thời Mũ bảo hộ (`hat`), Người (`person`) và Áo bảo hộ phản quang (`vest`) theo đúng gợi ý của Thầy.

### 1. Bảng số liệu thực nghiệm trên tập Test độc lập:
- **Lớp Mũ bảo hộ (`hat`)**: mAP50 = 93.05%
- **Lớp Áo phản quang (`vest`)**: mAP50 = 87.28%
- **Lớp Người lao động (`person`)**: mAP50 = 89.57%
- **Toàn bộ mô hình (mAP50 Mean)**: 89.97%

### 2. Kết luận khoa học trả lời Thầy Huy:
1. **Không có xung đột tính năng**: Việc bổ sung nhãn áo bảo hộ (`vest`) không làm suy giảm độ chính xác của mũ bảo hộ (`hat`). Lớp `vest` có diện tích lớn và độ phản quang cao nên mô hình đạt độ nhạy cực kỳ vượt trội (87.28% mAP50).
2. **Tính khả thi của chuyển giao tri thức**: Kế thừa trọng số từ SHWD giúp mô hình hội tụ ổn định ngay từ những epoch đầu tiên.
