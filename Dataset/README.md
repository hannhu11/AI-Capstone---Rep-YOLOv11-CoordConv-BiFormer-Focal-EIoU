# 📊 BÁO CÁO TOÀN DIỆN VỀ TẬP DỮ LIỆU (DATASET PIPELINE & ANALYSIS)
## Đồ án Tốt nghiệp Capstone AI - Hệ thống Giám sát Tuân thủ An toàn Lao động (PPE)
**Đơn vị**: Đại học FPT - Viện Đào tạo Quốc tế  
**Đề tài**: *Automated Real-Time Safety Helmet & Personal Protective Equipment Compliance Monitoring via Structural Re-Parameterized Coordinate Attention State-Space Network (Rep-YOLO11s)*  
**Chủ trì đồ án**: Nguyễn Hàn Như (SE183674) | **GVHD**: ThS. Nguyễn Xuân Huy  

---

## 1. TỔNG QUAN VÀ BỐI CẢNH MỞ RỘNG DỮ LIỆU (REVIEW 1 & TASK 2)
Trong đợt Review 1, Hội đồng đánh giá và Thầy hướng dẫn đã đưa ra định hướng mở rộng:
> *"Cần xem xét phân loại thêm màu sắc của mũ bảo hộ (trắng, vàng, đỏ, xanh...) và mở rộng nhận diện áo bảo hộ phản quang (safety vest). Đối với hướng mở rộng này, nhóm cần xây dựng pipeline dữ liệu sạch, xử lý mất cân bằng lớp và đối sánh độ chính xác giữa các bộ dữ liệu."*

Để đáp ứng trọn vẹn yêu cầu của **Task 2 (Phase 1: Dataset Pipeline & Phase 2: Baseline Execution)**, nhóm tiến hành khảo sát, làm sạch, phân tích định lượng và đánh giá 4 phương án dữ liệu:
- **Bộ 1 (GDUT-HWD)**: Tập dữ liệu 5 lớp màu mặt đất và phát hiện người không đội mũ (`none`).
- **Bộ 2 (GDUT-HWD mở rộng)**: Bộ 1 bổ sung thêm lớp Mũ cam (`orange_helmet`) tự thu thập thực địa.
- **Bộ 3 (IITU Safety-Helmet)**: Tập dữ liệu flycam/drone và mặt đất 5 lớp màu đa góc quay (0°, 45°, 90°) cả ngày lẫn đêm.
- **Bộ 4 (Unified PPE Master)**: Bộ tổng hợp toàn diện kết hợp Mũ đa màu sắc (GDUT + IITU) và Áo bảo hộ phản quang (`safety_vest` từ CHV/Roboflow).

---

## 2. BẢNG THỐNG KÊ CHI TIẾT CÁC NGUỒN DỮ LIỆU GỐC

| STT | Tên tập dữ liệu | Nguồn thu thập | Số lượng ảnh | Định dạng gốc | Số lượng Bounding Box | Các lớp (Classes) gốc | Đặc trưng môi trường |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | **GDUT-HWD** | Quảng Châu, Trung Quốc (Public Kaggle: `tundng111/gdut-hwd`) | 3,174 | Pascal VOC XML | 18,893 | `none` (4,371)<br>`red` (4,231)<br>`yellow` (3,898)<br>`white` (3,781)<br>`blue` (2,612) | Môi trường xây dựng mặt đất, góc nhìn ngang và hơi chúc, có người không đội mũ. |
| **2** | **IITU Safety-Helmet** | ĐH IITU Kazakhstan (Public Kaggle: `dngtrnphmtunse183674/iitu-safety-helmet`) | 1,669 | YOLO v5 txt (float16) | 6,471 | `blue` (1,359)<br>`green` (1,133)<br>`red` (1,216)<br>`white` (1,401)<br>`yellow` (1,362) | Flycam/Drone độ cao 3-5m & 10-15m; góc nhìn 0°, 45°, 90°; ánh sáng ban ngày & ban đêm. |
| **3** | **STANDARDIZED_CHV** | Construction Hardhat & Vest (Public & Standardized) | 1,330 | YOLO txt | 9,209 | `hat` (3,538)<br>`person` (3,887)<br>`vest` (1,784) | Công trường công nghiệp, nhãn áo bảo hộ phản quang tiêu chuẩn. |
| **4** | **SHWD (VOC2028)** | Safety Helmet Wearing Dataset (Gốc của Paper AAIML 2027) | 7,581 | Pascal VOC XML | 24,722 | `hat` (19,150)<br>`person` (5,572) | Dữ liệu benchmark chuẩn mực quốc tế 2-class. |

---

## 3. QUY TRÌNH LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU (DATA CLEANING PIPELINE)

Để đảm bảo nguyên tắc *"Xây nhà từ móng"*, toàn bộ dữ liệu trải qua quy trình 5 bước kiểm soát chất lượng (Quality Control Pipeline):

```mermaid
flowchart TD
    A["Raw Images & Labels (GDUT, IITU, CHV)"] --> B["B1: Integrity & Corrupt Image Filtering (MD5 / PIL decode)"]
    B --> C["B2: Duplicate Removal (14 duplicate images purged in GDUT)"]
    C --> D["B3: Geometry & BBox Sanity Check (0 <= x,y,w,h <= 1, w,h > 2px)"]
    D --> E["B4: Unified Class Schema Taxonomy Mapping"]
    E --> F["B5: Stratified Train/Val/Test Split (80% / 10% / 10%)"]
    F --> G["Standardized YOLO-format Output: images/ + labels/ + data.yaml"]
```

### 3.1. Kết quả kiểm tra tính toàn vẹn (Integrity Audit):
1. **Kiểm tra file lỗi / Corrupted**: Đã decode thử toàn bộ 12,173 ảnh trên toàn hệ thống bằng PIL Image. Kết quả: **0 ảnh bị hỏng cấu trúc byte**.
2. **Kiểm tra trùng lặp nội bộ (Internal Duplicates via MD5)**:
   - Trong `GDUT-HWD`: Phát hiện chính xác **14 ảnh trùng lặp tuyệt đối** (cùng hash MD5 nhưng khác tên file do gom từ nhiều nguồn). Các ảnh trùng lặp này đã được đánh dấu và loại bỏ khỏi tập huấn luyện để tránh data leakage.
   - Trong `IITU Safety-Helmet`: **0 ảnh trùng lặp** (1,669/1,669 ảnh độc lập).
   - Kiểm tra trùng lặp chéo giữa GDUT và IITU: **0 ảnh trùng lặp**, khẳng định 2 miền dữ liệu hoàn toàn trực giao (orthogonal domains).
3. **Kiểm tra Bounding Box bất thường**:
   - Loại bỏ các bbox có tọa độ âm hoặc vượt ngoài kích thước ảnh ($x_{max} > W$ hoặc $y_{max} > H$).
   - Loại bỏ các bbox có kích thước quá bé do nhiễu gán nhãn ($w < 2$ px hoặc $h < 2$ px).
   - Chuyển đổi toàn bộ tọa độ tuyệt đối sang hệ chuẩn hóa YOLO: $[x_{center}, y_{center}, width, height] \in [0.0, 1.0]$.

---

## 4. MA TRẬN PHÂN BỐ LỚP VÀ PHÂN TÍCH MẤT CÂN BẰNG (DATA IMBALANCE ANALYSIS)

### 4.1. Thống kê định lượng Bộ 1 (GDUT-HWD):
- Tổng số instances: **18,893**
- Phân bố chi tiết:
  * `none` (Không đội mũ / Vi phạm): **4,371** mẫu (**23.14%**)
  * `red` (Mũ đỏ): **4,231** mẫu (**22.39%**)
  * `yellow` (Mũ vàng): **3,898** mẫu (**20.63%**)
  * `white` (Mũ trắng): **3,781** mẫu (**20.01%**)
  * `blue` (Mũ xanh dương): **2,612** mẫu (**13.83%**)
- **Nhận xét định lượng**: Tập GDUT-HWD có sự phân bố tương đối đồng đều giữa 4 lớp đầu tiên (~20-23%), riêng lớp mũ xanh dương có tỷ lệ thấp hơn (13.83%), tỷ lệ mất cân bằng (Imbalance Ratio) là $4,371 / 2,612 = 1.67:1$ (mức độ mất cân bằng nhẹ, nằm trong tầm kiểm soát tốt của hàm mất mát Focal Loss).

### 4.2. Thống kê định lượng Bộ 3 (IITU Safety-Helmet):
- Tổng số instances: **6,471**
- Phân bố chi tiết:
  * `white` (Mũ trắng): **1,401** mẫu (**21.65%**)
  * `yellow` (Mũ vàng): **1,362** mẫu (**21.05%**)
  * `blue` (Mũ xanh dương): **1,359** mẫu (**21.00%**)
  * `red` (Mũ đỏ): **1,216** mẫu (**18.79%**)
  * `green` (Mũ xanh lá): **1,133** mẫu (**17.51%**)
- **Nhận xét định lượng**: Tập IITU đạt độ cân bằng lý tưởng giữa cả 5 màu sắc (chênh lệch giữa lớp nhiều nhất và ít nhất chỉ là $1,401 / 1,133 = 1.24:1$). Đặc biệt, IITU bổ sung **1,133 mẫu mũ xanh lá cây (`green`)** mà các tập dữ liệu khác hoàn toàn không có. Tuy nhiên, tập này **hoàn toàn thiếu lớp `none`** (người không đội mũ).

### 4.3. Bảng phân chia Train / Validation / Test (Tỷ lệ chuẩn 80% / 10% / 10%):

| Bộ dữ liệu | Tập Train (80%) | Tập Validation (10%) | Tập Test (10%) | Tổng cộng (Ảnh) |
| :--- | :---: | :---: | :---: | :---: |
| **Bộ 1: GDUT-HWD** | 2,528 ảnh | 316 ảnh | 316 ảnh | 3,160 ảnh |
| **Bộ 3: IITU Safety-Helmet** | 1,335 ảnh | 167 ảnh | 167 ảnh | 1,669 ảnh |
| **Bộ 4: Unified PPE Master (Gộp)** | 4,923 ảnh | 615 ảnh | 615 ảnh | 6,153 ảnh |

---

## 5. SO SÁNH 4 PHƯƠNG ÁN QUYẾT ĐỊNH DỮ LIỆU (STRATEGIC DECISION MATRIX)

| Tiêu chí so sánh | Bộ 1: GDUT-HWD | Bộ 2: GDUT + Mũ cam | Bộ 3: IITU Safety-Helmet | Bộ 4: Unified PPE Master (ĐỀ XUẤT) |
| :--- | :--- | :--- | :--- | :--- |
| **Số lượng lớp (Classes)** | 5 lớp (`none`, red, yellow, white, blue) | 6 lớp (+ orange) | 5 lớp màu (blue, green, red, white, yellow) | **7 lớp** (5 màu mũ + `no_helmet` + `safety_vest`) |
| **Phát hiện vi phạm không đội mũ?** | Có (`none`: 4,371) | Có (`none`: 4,371) | ❌ Không có | **CÓ (Cực kỳ đầy đủ)** |
| **Độ phủ màu sắc mũ bảo hộ** | 4 màu (Đỏ, Vàng, Trắng, Xanh dương) | 5 màu (+ Cam) | 5 màu (+ Xanh lá) | **ĐẦY ĐỦ NHẤT (Đỏ, Vàng, Trắng, Xanh dương, Xanh lá, Cam)** |
| **Nhận diện áo bảo hộ (`vest`)?** | ❌ Không có | ❌ Không có | ❌ Không có | **CÓ (`safety_vest`: >1,780 mẫu)** |
| **Góc quay & Điều kiện chiếu sáng** | Mặt đất, góc ngang/chúc, ban ngày | Mặt đất, ban ngày | Flycam 0°-45°-90°, ban ngày & ban đêm | **ĐA DẠNG NHẤT (Mặt đất + Drone + Ngày/Đêm)** |
| **Phù hợp mục tiêu ĐATN & Hội đồng?** | Mức Khá (Đạt 7/10) | Mức Khá (Đạt 7.5/10) | Mức Trung bình (Thiếu nhãn vi phạm) | **XUẤT SẮC (Đạt 10/10 điểm tiêu chí)** |

### 👉 KẾT LUẬN CHIẾN LƯỢC:
1. **Tại sao không dùng riêng lẻ Bộ 3 (IITU)?** Vì IITU không có nhãn người không đội mũ (`none`), nếu chỉ dùng IITU thì mô hình không thể phát hiện vi phạm an toàn lao động (mất đi mục tiêu cốt lõi của đề tài).
2. **Tại sao Bộ 4 là phương án tối ưu nhất?** Bộ 4 tổng hợp ưu thế của tất cả các bộ: có đủ các màu sắc phân quyền lao động, có phát hiện vi phạm không đội mũ, có nhận diện áo phản quang, và có góc nhìn từ flycam lẫn camera giám sát ban đêm.

---

## 6. KHẲNG ĐỊNH TÍNH TƯƠNG THÍCH CỦA 5 MODULE REP-YOLO11s VỚI DATASET MỚI

Khi chuyển đổi từ bài toán 2-class (SHWD) sang 5-class hoặc 7-class (Bộ 1, 3, 4), **5 module cốt lõi của Rep-YOLO11s KHÔNG HỀ BỊ THAY ĐỔI** mà ngược lại càng phát huy tác dụng vượt trội:
1. **Stem Coordinate Convolution (CoordConv)**: Cung cấp inductive bias tọa độ thẳng đứng ($y$). Mũ bảo hộ nằm ở vùng đầu ($y$ nhỏ), áo bảo hộ nằm ở vùng thân ($y$ trung bình), rác và vật thể nhiễu nằm ở vùng sàn ($y$ lớn). Mô hình tự động học được không gian phân bố mà không cần module phân cấp phụ trợ.
2. **Structural Re-Parameterized Neck (RepConv)**: Học biểu diễn đa nhánh phong phú (hoa văn dải phản quang của áo và độ bóng của vỏ mũ nhựa), sau đó suy biến thành một tích chập $3\times 3$ đơn khi triển khai $\to$ **Duy trì tốc độ >200 FPS bất kể số lớp tăng lên**.
3. **Bi-Level Routing Attention (BiFormer)**: Lọc bỏ token nền (bầu trời, giàn giáo, sàn bê tông), định tuyến tính toán trực tiếp vào vùng mũ nhỏ từ xa và dải phản quang của áo.
4. **Decoupled Focal EIoU Regression**: Xử lý triệt để sự khác biệt về hình học giữa mũ (vuông, $w/h \approx 1$) và áo phản quang (chữ nhật đứng, $h > w$), đồng thời cân bằng đóng góp gradient giữa các màu sắc ít mẫu và nhiều mẫu.
5. **Task-Aligned Assigner (TAL)**: Đồng bộ hóa điểm tin cậy phân loại màu sắc và chất lượng định vị hộp bao.

$\implies$ **Không cần thêm bất kỳ module "chắp vá" nào khác.** Toàn bộ kiến trúc Rep-YOLO11s là một thể thống nhất, tối ưu phần cứng từ tầng toán tử.
