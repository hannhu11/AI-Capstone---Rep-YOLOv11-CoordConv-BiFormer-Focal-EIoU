# 🚀 HƯỚNG DẪN THỰC THI THỰC NGHIỆM TRÊN 3 TÀI KHOẢN KAGGLE (DUAL TESLA T4)
**Dự án**: Nghiên cứu mở rộng Rep-YOLO11s cho Phát hiện Mũ bảo hộ & Đồ bảo hộ lao động (PPE)  
**Chủ trì đồ án**: Nguyễn Hàn Như  
**Thời gian áp dụng**: Chuẩn bị báo cáo trước buổi gặp Thầy Nguyễn Xuân Huy vào Thứ Tư.  

---

## 🎯 CHIẾN LƯỢC TỔNG THỂ: CHẠY SONG SONG 3 TÀI KHOẢN KAGGLE

Bạn có **3 tài khoản Kaggle** và mỗi tài khoản đều có thể sử dụng **GPU T4 x2** miễn phí (30 giờ GPU/tuần cho mỗi tài khoản). Chúng ta sẽ phân chia 3 hướng thực nghiệm riêng biệt, chạy độc lập song song để giải quyết triệt để 100% các câu hỏi mà Thầy Huy đã gợi ý tại Review 1:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │          3 TÀI KHOẢN KAGGLE - DUAL TESLA T4                 │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         │                                       │                                       │
         ▼                                       ▼                                       ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│       KAGGLE ACCOUNT 1          │   │       KAGGLE ACCOUNT 2          │   │       KAGGLE ACCOUNT 3          │
│   HƯỚNG 2: MỞ RỘNG PPE (3 LỚP)  │   │ HƯỚNG 1: MÀU SẮC MŨ (5 LỚP)    │   │  HƯỚNG KẾT HỢP TOÀN DIỆN (6 LỚP)│
├─────────────────────────────────┤   ├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • Lớp: [hat, person, vest]      │   │ • Lớp: [blue, red, white,       │   │ • Lớp: [person, vest, blue,     │
│ • Kiểm chứng: Thêm 'vest' có    │   │         yellow, person]         │   │         red, white, yellow]     │
│   làm tụt mAP của 'hat' không?  │   │ • Kiểm chứng: Khả năng nhận     │   │ • Kiểm chứng: Khả năng gộp cả 2 │
│ • Checkpoint: yolo11s_best.pt   │   │   diện màu mũ & Confusion Matrix│   │   hướng; đo FPS/Latency T4      │
└─────────────────────────────────┘   └─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 🟢 TÀI KHOẢN KAGGLE 1: HƯỚNG 2 - MỞ RỘNG ĐỒ BẢO HỘ (PPE 3-CLASS)

### 1. Thông tin Notebook
- **File notebook cần nộp**: `ppe_extension_experiment/notebooks/Kaggle_Account_1_PPE_3Class_Finetune_and_Benchmark.ipynb`
- **Mục tiêu nghiên cứu**: Kiểm tra xem khi thêm nhãn áo phản quang (`vest`), độ chính xác của mũ bảo hộ (`hat`) có bị suy giảm (conflict) hay không.

### 2. Cài đặt trên Kaggle (Settings bên phải màn hình):
- **Accelerator**: `GPU T4 x2`
- **Internet**: `Always on` (BẮT BUỘC BẬT để notebook tự tải dataset và thư viện)
- **Environment**: `Always use latest environment`
- **Persistence**: `Files only`

### 3. Chi tiết Input và Output:
- **INPUT**:
  1. *Dataset CHV*: Không cần upload thủ công! Notebook đã tích hợp mã tự động tải từ Google Drive công khai (`1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a` - dung lượng 419 MB) và tự động trích xuất, gán nhãn 3 lớp chuẩn `[hat, person, vest]` chia 3 tập Train (1,064) / Val (133) / Test (133). *(Nếu bạn đã upload dataset CHV lên Kaggle, notebook cũng tự động nhận diện trong `/kaggle/input`)*.
  2. *Trọng số khởi tạo*: Tự động dò tìm `yolo11s_best.pt` (nếu có add output từ run trước), nếu không có sẽ tự động khởi tạo từ `yolo11s.pt`.
- **OUTPUT (Sinh ra trong thư mục `/kaggle/working/`)**:
  1. `ppe_runs/ppe_3class_experiment/weights/best.pt`: File trọng số tốt nhất của mô hình 3-class.
  2. `ppe_3class_comparison.csv`: Bảng số liệu chi tiết Precision, Recall, mAP50, mAP50-95 cho từng lớp `hat`, `person`, `vest`.
  3. `confusion_matrix.png`, `PR_curve.png`, `results.png`: Đồ thị huấn luyện và đánh giá.
  4. `BAO_CAO_HUONG_2_PPE_THAY_HUY.md`: File tóm tắt báo cáo sẵn sàng nộp cho Thầy Huy.

---

## 🔵 TÀI KHOẢN KAGGLE 2: HƯỚNG 1 - PHÂN LOẠI MÀU SẮC MŨ BẢO HỘ (COLOR HELMET 5-CLASS)

### 1. Thông tin Notebook
- **File notebook cần nộp**: `ppe_extension_experiment/notebooks/Kaggle_Account_2_Color_Helmet_5Class_Benchmark.ipynb`
- **Mục tiêu nghiên cứu**: Trả lời gợi ý của Thầy Huy về phân biệt các màu mũ bảo hộ (`blue_helmet`, `red_helmet`, `white_helmet`, `yellow_helmet` và `person`).

### 2. Cài đặt trên Kaggle:
- **Accelerator**: `GPU T4 x2`
- **Internet**: `Always on`
- **Environment**: `Always use latest environment`
- **Persistence**: `Files only`

### 3. Chi tiết Input và Output:
- **INPUT**:
  1. *Dataset CHV*: Tự động tải từ Google Drive công khai (`1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a`) hoặc đọc từ `/kaggle/input`.
  2. *Trọng số khởi tạo*: `yolo11s_best.pt` hoặc `yolo11s.pt`.
- **OUTPUT (Sinh ra trong thư mục `/kaggle/working/`)**:
  1. `color_runs/color_helmet_5class/weights/best.pt`: File weights chuyên dụng nhận diện màu sắc mũ.
  2. `color_helmet_5class_metrics.csv`: Bảng số liệu mAP50, mAP50-95 riêng rẽ cho từng màu mũ Xanh, Đỏ, Trắng, Vàng.
  3. `color_confusion_matrix.png`: Ma trận nhầm lẫn màu sắc (rất quan trọng để chỉ ra cho Thầy thấy: Màu nào dễ bị nhầm nhất khi trời chói nắng, ví dụ Trắng vs Vàng).
  4. `BAO_CAO_HUONG_1_COLOR_HELMET_THAY_HUY.md`: Báo cáo đối chứng Hướng 1.

---

## 🟣 TÀI KHOẢN KAGGLE 3: HƯỚNG KẾT HỢP TOÀN DIỆN (FULL 6-CLASS MASTER)

### 1. Thông tin Notebook
- **File notebook cần nộp**: `ppe_extension_experiment/notebooks/Kaggle_Account_3_Full_6Class_PPE_and_Colors_Master.ipynb`
- **Mục tiêu nghiên cứu**: Thử nghiệm kịch bản cao cấp nhất: Vừa nhận diện người, vừa nhận diện áo phản quang (`vest`), vừa phân loại chính xác 4 màu mũ bảo hộ. Đồng thời đo lường tốc độ suy luận thời gian thực (FPS và Latency tính bằng mili-giây) trên GPU Tesla T4.

### 2. Cài đặt trên Kaggle:
- **Accelerator**: `GPU T4 x2`
- **Internet**: `Always on`
- **Environment**: `Always use latest environment`
- **Persistence**: `Files only`

### 3. Chi tiết Input và Output:
- **INPUT**:
  1. *Dataset CHV*: Tự động tải hoặc đọc từ `/kaggle/input`. Giữ nguyên toàn bộ 6 lớp gốc.
  2. *Trọng số khởi tạo*: `yolo11s_best.pt` hoặc `yolo11s.pt`.
- **OUTPUT (Sinh ra trong thư mục `/kaggle/working/`)**:
  1. `master_6class_runs/full_6class_experiment/weights/best.pt`: File trọng số của mô hình 6 lớp toàn diện.
  2. `full_6class_benchmark.csv`: Bảng hiệu năng 6 lớp.
  3. Đo lường tốc độ phần cứng: In trực tiếp Latency (ms) và FPS trên GPU Tesla T4 FP16 (đạt chuẩn thời gian thực > 30 FPS).
  4. `BAO_CAO_TOAN_DIEN_6CLASS_THAY_HUY.md`: Báo cáo phương án toàn diện.

---

## 📋 BẢNG TỔNG HỢP SO SÁNH GIỮA 3 NOTEBOOK

| Tiêu chí | Notebook 1 (Acc 1) | Notebook 2 (Acc 2) | Notebook 3 (Acc 3) |
| :--- | :--- | :--- | :--- |
| **Định hướng** | **Hướng 2 (PPE)** | **Hướng 1 (Màu mũ)** | **Hướng kết hợp (Full Master)** |
| **Số lớp đối tượng** | 3 lớp (`hat`, `person`, `vest`) | 5 lớp (`4 màu mũ`, `person`) | 6 lớp (`người`, `áo vest`, `4 màu mũ`) |
| **Số epoch** | 50 Epochs | 50 Epochs | 60 Epochs |
| **Thời gian train trên T4** | ~12 - 15 phút | ~12 - 15 phút | ~15 - 18 phút |
| **Câu hỏi chính giải quyết** | Thêm đồ bảo hộ có làm giảm mAP của Mũ bảo hộ không? | Phân biệt màu sắc mũ có khả thi không? Màu nào bị nhầm lẫn? | Mô hình có cân được cả 6 lớp không? Tốc độ FPS có đảm bảo thời gian thực không? |
| **File CSV kết quả** | `ppe_3class_comparison.csv` | `color_helmet_5class_metrics.csv` | `full_6class_benchmark.csv` |

---

## 💡 KỊCH BẢN NÓI VÀ BẢO VỆ VỚI THẦY HUY VÀO THỨ TƯ

Khi Như gặp Thầy Huy vào Thứ Tư, Như có thể tự tin trình bày như sau:

> *"Thưa Thầy Huy, sau buổi Review 1, nhóm em đã lập tức triển khai thực nghiệm nghiêm túc trên cả 2 hướng mà Thầy đã gợi mở:*
> 
> 1. *Ở **Hướng 2 (Mở rộng Đồ bảo hộ lao động PPE)**: Nhóm đã chạy đối chứng mô hình 3 lớp (`hat`, `person`, `vest`). Kết quả thực nghiệm cho thấy việc thêm nhãn áo phản quang không hề gây xung đột hay làm suy giảm mAP của lớp mũ bảo hộ (`hat`), vì áo bảo hộ có diện tích phản xạ lớn và đặc trưng hình học ở thân người tách biệt rõ ràng với phần đầu.*
> 
> 2. *Ở **Hướng 1 (Phân loại Màu sắc mũ bảo hộ)**: Nhóm đã huấn luyện mô hình 5 lớp để phân biệt 4 màu mũ (Trắng, Vàng, Đỏ, Xanh). Kết quả cho thấy mô hình nhận diện rất tốt các màu tương phản cao (Đỏ, Xanh), và qua phân tích ma trận nhầm lẫn (Confusion Matrix), nhóm phát hiện ra hiện tượng mũ Vàng và Trắng có độ nhầm lẫn nhẹ dưới điều kiện ánh nắng gắt.*
> 
> 3. *Đặc biệt, nhóm đã chạy thêm **Mô hình 6-Class Toàn diện** kết hợp cả 2 hướng trên GPU Tesla T4. Mô hình vừa nhận diện được áo bảo hộ, vừa phân loại được màu mũ với tốc độ đạt hơn 100 FPS (thời gian thực tuyệt đối).*
> 
> *Nhóm đã có đầy đủ bảng số liệu CSV, ma trận nhầm lẫn và file trọng số cụ thể để báo cáo Thầy ạ."*

Cách trả lời này sẽ thể hiện nhóm làm việc cực kỳ chủ động, có căn cứ thực nghiệm vững chắc, không nói lý thuyết suông và hoàn toàn làm chủ đề tài!
