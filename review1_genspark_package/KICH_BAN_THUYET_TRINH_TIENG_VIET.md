# 🎙️ KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN TỐT NGHIỆP KỸ SƯ AI (GIAI ĐOẠN 1)
## ĐỀ TÀI: REP-YOLO11s — GIÁM SÁT AN TOÀN LAO ĐỘNG CÔNG TRƯỜNG THỜI GIAN THỰC
### TRƯỜNG ĐẠI HỌC FPT · BỘ MÔN TRÍ TUỆ NHÂN TẠO

---

> **Sinh viên thuyết trình:** Nguyễn Hàn Như (Trưởng nhóm - SE183644)  
> **Thành viên nhóm:** Nguyễn Văn Thành (SE183645), Nguyễn Tuấn Dũng (SE183646)  
> **Giảng viên hướng dẫn:** ThS. Vũ Hải Anh  
> **Tài liệu sử dụng:** 
> - File trình chiếu PPTX: `Rep-YOLO11s___Capstone_Review_1_Defense_20260920062556.pptx` (hoặc `Rep_YOLO11s_FPT_Defense_Master_Deck.pptx`)
> - Bản in PDF: `Rep_YOLO11s_FPT_Defense_Master_Deck.pdf` (18 slide, Theme nền sáng, 100% Tiếng Việt)
> - Bài báo nghiên cứu khoa học: `Rep-YOLO11s_Master_Paper_IEEE_Final.pdf` (9 trang)
> **Tổng thời lượng đề xuất:** 12 – 15 phút trình bày + 10 – 15 phút hỏi đáp phản biện với Hội đồng.

---

## 🎯 CHIẾN LƯỢC TÂM LÝ & NGUYÊN TẮC THUYẾT TRÌNH TRƯỚC HỘI ĐỒNG FPT

1. **Vị thế tự tin vượt trội (Thẩm định tính khả thi đạt 100%):**
   - Đa số các nhóm ở đợt đánh giá đề cương ban đầu (Giai đoạn 1) chỉ mới dừng lại ở ý tưởng lý thuyết hoặc khảo sát tài liệu.
   - Nhóm chúng ta **đã hoàn thành hơn 90% khối lượng thực nghiệm kỹ thuật lõi**: Có bài báo khoa học 9 trang chuẩn quốc tế, có kiểm định 5-Fold Cross-Validation đạt **97.11% mAP50**, có đầy đủ nghiên cứu cắt bỏ A0–A6, bản đồ nhiệt giải thích Grad-CAM XAI, và đo đạc vật lý trên phần cứng thật từ máy chủ Tesla T4 đến laptop sinh viên MX230 (27.8 FPS).
   - Hãy trình bày với phong thái đĩnh đạc, rõ ràng, luôn dẫn chứng bằng **số liệu đo đạc thực tế** và **công thức giải tích**, tuyệt đối không dùng từ võ đoán ("em đoán", "chắc là").

2. **Khắc cốt ghi tâm 4 Tiêu chí Rubric của Bộ môn AI - ĐH FPT:**
   - Khi chuyển slide, hãy chủ động liên kết với 4 tiêu chí đánh giá:
     * **Tiêu chí 1:** Mục tiêu đề tài & Tính cấp thiết (Slide 2: 4 nút thắt thực tế công trường).
     * **Tiêu chí 2:** Kết quả đầu ra cụ thể (Slide 4: 5 sản phẩm cam kết bàn giao).
     * **Tiêu chí 3:** Tính khả thi & Phạm vi (Slide 3: Khảo sát 32 bài báo; Slide 5-10: 4 module tự phát triển; Slide 11: Làm sạch 33,000+ ảnh).
     * **Tiêu chí 4:** Giá trị thực tiễn & Ý nghĩa khoa học (Slide 12-16: SOTA Pareto, Grad-CAM, kiểm thử đa miền, chạy trên GPU giá rẻ).

3. **Ngôn từ chuẩn mực tuyệt đối:**
   - Toàn bộ bài báo cáo dùng thuật ngữ học thuật trang trọng: *"Báo cáo Đồ án Tốt nghiệp Kỹ sư AI — Giai đoạn 1 (Khởi tạo đề cương & Thẩm định tính khả thi)"*, *"Giai đoạn 2"*, *"Giai đoạn 3"*.

---

## 📋 LỜI THOẠI THUYẾT TRÌNH CHI TIẾT TỪNG SLIDE (SLIDE-BY-SLIDE)

---

### 📌 SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** Khung số liệu nổi bật bên phải (`97.11%`, `2.92 ms`, `27.8 FPS`).
- **Lời thoại thuyết trình:**
  > "Kính thưa quý Thầy Cô trong Hội đồng Đánh giá Đồ án Tốt nghiệp ngành Trí tuệ Nhân tạo - Trường Đại học FPT.  
  > Em tên là **Nguyễn Hàn Như**, đại diện cho nhóm nghiên cứu gồm em và hai bạn **Nguyễn Văn Thành**, **Nguyễn Tuấn Dũng**, dưới sự hướng dẫn chuyên môn tận tình của **Thầy Vũ Hải Anh**, xin được phép báo cáo tiến độ và thẩm định tính khả thi của Đồ án Tốt nghiệp với đề tài:  
  > **'Rep-YOLO11s: Tái Tham Số Hóa Cấu Trúc, Mã Hóa Tọa Độ Không Gian và Năng Lực Khái Quát Hóa Đa Miền Cho Phát Hiện Mũ Bảo Hộ Thời Gian Thực Trong Giám Sát Công Trường'**.  
  > Dù đây là đợt thẩm định đề cương Giai đoạn 1, nhóm chúng em đã chủ động triển khai sớm và hoàn thành hơn 90% khối lượng thực nghiệm kỹ thuật, đạt độ chính xác đỉnh **97.11% mAP50** trên kiểm định 5-Fold, đồng thời đạt tốc độ suy luận vật lý vượt trội **342.5 FPS** trên GPU Tesla T4 và **27.8 FPS** ngay trên laptop phổ thông giá rẻ. Sau đây, em xin phép đi vào chi tiết bài báo cáo."

---

### 📌 SLIDE 2: BỐI CẢNH THỰC TẾ & 4 NÚT THẮT KỸ THUẬT (TIÊU CHÍ 1)
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Hình ảnh thực tế bên phải (Hình 1: Camera góc cao) và 4 thẻ nút thắt bên trái.
- **Lời thoại thuyết trình:**
  > "Kính thưa Thầy Cô, đáp ứng **Tiêu chí 1 về Tính cấp thiết của đề tài**, tai nạn chấn thương sọ não do vật thể rơi từ trên cao là nguyên nhân hàng đầu gây tử vong tại các công trường xây dựng. Việc giám sát thủ công hiện nay chỉ bao quát được dưới 15% thời gian ca trực.  
  > Tuy nhiên, như Thầy Cô quan sát ở **Hình 1** trên slide, khi triển khai camera thị giác máy tính vào thực tế, hệ thống gặp phải **4 nút thắt kỹ thuật cốt lõi**:  
  > 1. **Vi vật thể từ xa:** Camera gắn ở độ cao 15–30 mét khiến mũ bảo hộ chỉ chiếm kích thước dưới 20 pixels, rất dễ bị mờ nhòe và biến mất qua các tầng trích xuất sâu.  
  > 2. **Mất cân bằng nhãn cực đoan:** Tỉ lệ giữa mũ bảo hộ và thân người lên tới 1:12, khiến mô hình bị thiên lệch gradient vào các mẫu dễ nhận diện.  
  > 3. **Nhiễu màu sắc gây báo động giả:** Công trường tràn ngập xô vữa vàng, cọc tiêu cam, biển báo nguy hiểm. Do mạng CNN tiêu chuẩn có tính 'Bất biến tịnh tiến', mô hình thường nhận diện nhầm các vật thể màu vàng dưới đất thành mũ bảo hộ.  
  > 4. **Rào cản phần cứng biên:** Hệ thống đòi hỏi tốc độ xử lý thời gian thực từ 25 FPS trở lên trên máy tính giá rẻ tại công trường, không thể phụ thuộc vào máy chủ đám mây đắt tiền."

---

### 📌 SLIDE 3: KHẢO SÁT 32 CÔNG TRÌNH QUỐC TẾ & KHOẢNG TRỐNG SOTA (TIÊU CHÍ 3)
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 4 thẻ so sánh các dòng mô hình hiện hữu và banner khoảng trống khoa học ở đáy slide.
- **Lời thoại thuyết trình:**
  > "Để thẩm định tính khả thi theo **Tiêu chí 3**, nhóm đã khảo sát chuyên sâu **32 công trình khoa học quốc tế uy tín** từ năm 2019 đến 2026.  
  > Khảo sát cho thấy các mô hình hiện tại đều mắc phải những đánh đổi lớn:  
  > - Dòng **YOLO chuẩn (v8, v10, 11)** chạy rất nhanh nhưng thiếu cơ chế tọa độ, gây báo động giả nghiêm trọng với xô vữa vàng.  
  > - Mô hình **EC-YOLOv8** đạt mAP cao nhưng dùng toán tử CARAFE nặng nề, làm tụt tốc độ xử lý.  
  > - Mô hình **YOLO-CBF** tích hợp CoordConv và BiFormer nhưng phình to tới 37.2 triệu tham số, tốc độ rơi xuống chỉ còn 80 FPS.  
  > - Mô hình **YOLOv8n-FADS** mở nhánh P2 cho vi vật thể nhưng gây nghẽn độ trễ phần cứng.  
  > **Khoảng trống khoa học (Research Gap):** Chưa có công trình nào đạt được đồng thời: Độ chính xác vi vật thể cao + Zero-overhead độ trễ suy luận + Khả năng thích ứng đa miền bền vững. Đây chính là động lực để nhóm đề xuất kiến trúc **Rep-YOLO11s**."

---

### 📌 SLIDE 4: XÁC ĐỊNH PHẠM VI & 5 SẢN PHẨM BÀN GIAO CỤ THỂ (TIÊU CHÍ 2)
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** Cột 5 sản phẩm cam kết bên trái và Bảng kết quả phần cứng đo đạc bên phải.
- **Lời thoại thuyết trình:**
  > "Để trả lời trực diện cho **Tiêu chí 2 về Kết quả đầu ra cụ thể**, nhóm cam kết bàn giao trọn vẹn **5 sản phẩm đầu ra hoàn chỉnh**:  
  > 1. **Báo cáo khoa học 9 trang** viết bằng LaTeX chuẩn mực, đầy đủ chứng minh toán học giải tích và thực nghiệm.  
  > 2. **Mô hình Rep-YOLO11s** tối ưu đạt 94.83% mAP50 trên tập kiểm thử và 97.11% trên 5-Fold Cross-Validation.  
  > 3. **Software Prototype** giám sát video RTSP đa luồng thời gian thực đạt 65–95 FPS.  
  > 4. **Bộ động cơ biên dịch triển khai** gồm TensorRT FP16, ONNX Runtime và OpenVINO cho chip biên.  
  > 5. **Bộ dữ liệu chuẩn hóa công nghiệp** với hơn 33,000 ảnh từ 6 nguồn công trường khác nhau.  
  > Toàn bộ số đo độ trễ trên bảng bên phải đều được đo vật lý bằng CUDA Events trên thiết bị thật, loại bỏ hoàn toàn các số liệu ảo."

---

### 📌 SLIDE 5: SƠ ĐỒ KIẾN TRÚC TOÀN DIỆN 5 TẦNG REP-YOLO11s
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Sơ đồ kiến trúc cực lớn ở giữa (Hình 2) và 3 thẻ tóm tắt Backbone - Neck - Head & Loss bên dưới.
- **Lời thoại thuyết trình:**
  > "Trên slide 5 là **Hình 2 — Sơ đồ kiến trúc toàn diện 5 tầng** của Rep-YOLO11s do nhóm thiết kế:  
  > - **Tầng 1 & 2 (Backbone Không Gian):** Mạng CSPDarknet nhận ảnh đầu vào được bổ sung 2 kênh tọa độ không gian chuẩn hóa thông qua tầng **CoordConv Stem**, kết hợp các khối **RepConv** học biểu diễn đa nhánh.  
  > - **Tầng 3 (Neck Định Tuyến Thưa):** Mạng PAN được nhúng cơ chế chú ý 2 cấp độ **BiFormer**, tự động lọc bỏ 80% phông nền xà bần công trường và gom tụ chú ý vào vi vật thể.  
  > - **Tầng 4 & 5 (Head & Loss Tối Ưu):** Đầu dò Decoupled không anchor kết hợp hàm mất mát **Focal EIoU**, giải quyết triệt để vấn đề mất cân bằng nhãn và sự biến dạng hộp bao vi vật thể.  
  > Tiếp theo, em xin đi sâu vào bản chất toán học của 4 điểm mới này."

---

### 📌 SLIDE 6: ĐIỂM MỚI 1 - TÁI THAM SỐ HÓA CẤU TRÚC (REPCONV)
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Sơ đồ gộp nhánh đại số ở giữa và công thức $W_{fused}$, $b_{fused}$.
- **Lời thoại thuyết trình:**
  > "Điểm mới đầu tiên là kỹ thuật **Tái tham số hóa cấu trúc RepConv**.  
  > - Trong **giai đoạn huấn luyện**, mô hình sử dụng cấu trúc 3 nhánh song song: tích chập 3x3, tích chập 1x1 và nhánh Identity ma trận đơn vị. Thiết kế này giúp dòng gradient phân nhánh phong phú, tránh bão hòa và tối đa hóa khả năng biểu diễn đặc trưng.  
  > - Khi **triển khai suy luận**, hàm `switch_to_deploy()` sử dụng phép biến đổi đại số tuyến tính: Hợp nhất Batch Normalization vào trọng số tích chập, đệm zero kernel 1x1 thành kích thước 3x3, và chuyển nhánh Identity thành ma trận Dirac delta.  
  > - Cả 3 nhánh được cộng gộp chính xác thành **duy nhất một lớp Conv 3x3 đơn lẻ**.  
  > Kết quả: Độ trễ suy luận giảm ngoạn mục **55.2%** (từ 7.12 ms xuống 2.92 ms), đạt thông lượng **342.5 FPS** trên Tesla T4 với sai số toán học đại số $\Delta < 10^{-5}$."

---

### 📌 SLIDE 7: ĐIỂM MỚI 2 - MÃ HÓA TỌA ĐỘ COORDCONV DẬP TẮT BÁO ĐỘNG GIẢ
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Sơ đồ tiêm 2 kênh $C_x, C_y$ và giải thích quy luật giải phẫu công trường.
- **Lời thoại thuyết trình:**
  > "Điểm mới thứ hai giải quyết trực diện nút thắt báo động giả: **Tầng mã hóa tọa độ không gian CoordConv**.  
  > Mạng CNN truyền thống có tính chất *bất biến tịnh tiến* (Translation Invariance) — nghĩa là một khối pixel màu vàng ở trên đỉnh đầu hay nằm dưới mặt đất đều tạo ra phản ứng kích hoạt tương đương nhau. Do đó, các xô vữa vàng hay cọc tiêu dưới đất thường xuyên bị báo động giả thành mũ bảo hộ.  
  > Nhóm đã phá vỡ giới hạn này bằng cách bổ sung trực tiếp 2 kênh tọa độ chuẩn hóa $C_x$ và $C_y$ trong đoạn $[-1, 1]$ vào tensor ảnh đầu vào (tạo thành tensor 5 kênh).  
  > Nhờ kênh $C_y$ phản ánh vị trí tương đối từ đỉnh khung hình ($C_y = -1.0$) xuống mặt sàn ($C_y = +1.0$), các kernel tích chập học được tiên đề hình học tự nhiên: **Mũ bảo hộ luôn nằm trên phần thân trên của công nhân ($C_y < 0$), không thể nằm trôi nổi dưới mặt đất**.  
  > Kỹ thuật này dập tắt hoàn toàn hơn **28% số lượng cảnh báo sai**, và được chứng minh trực quan bằng bản đồ nhiệt Grad-CAM ở slide sau."

---

### 📌 SLIDE 8: ĐIỂM MỚI 3 - CHÚ Ý ĐỊNH TUYẾN THƯA BIFORMER
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Sơ đồ chia lưới $S \times S$, đồ thị tương đồng và Top-$k$ routing.
- **Lời thoại thuyết trình:**
  > "Điểm mới thứ ba là cơ chế **Chú ý định tuyến thưa 2 cấp độ BiFormer** tích hợp tại tầng Neck.  
  > Cơ chế Self-Attention chuẩn của Vision Transformer có độ phức tạp bậc hai $\mathcal{O}((HW)^2)$, gây nghẽn phần cứng nghiêm trọng. BiFormer giải quyết vấn đề này qua 2 bước:  
  > - Bước 1: Chia feature map thành lưới $S \times S$ vùng khu vực (với $S=7$), xây dựng ma trận tương đồng giữa các vùng.  
  > - Bước 2: Dùng phép định tuyến thưa Top-$k$ (chọn $k=4$), chỉ giữ lại các cặp vùng có độ tương quan ngữ nghĩa cao nhất và loại bỏ hơn 80% phông nền xà bần, giàn giáo gây nhiễu.  
  > - Cuối cùng, tính toán Token-to-Token Attention chi tiết trên tập vùng đã lọc.  
  > Thuật toán này đưa độ phức tạp về bậc tuyến tính $\mathcal{O}(HW)$, giúp mô hình tập trung năng lượng tính toán vào các vi vật thể mũ bảo hộ cự ly xa mà không làm tăng độ trễ."

---

### 📌 SLIDE 9: ĐIỂM MỚI 4 - HÀM MẤT MÁT FOCAL EIOU
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Sơ đồ phân rã hình học 3 thành phần của Focal EIoU.
- **Lời thoại thuyết trình:**
  > "Điểm mới thứ tư là việc áp dụng **Hàm mất mát Focal EIoU** cho đầu dò hộp bao.  
  > Hàm CIoU truyền thống gặp nhược điểm lớn khi chỉ tối ưu tỉ lệ co $w/h$. Nếu chiều rộng và chiều cao cùng tăng hoặc giảm tỉ lệ thuận, đạo hàm của tỉ lệ co sẽ bằng 0, làm mô hình không thể tối ưu kích thước hộp bao vi vật thể.  
  > Focal EIoU phân rã hàm mục tiêu thành 3 thành phần hình học độc lập:  
  > 1. Trùng khớp diện tích IoU chuẩn;  
  > 2. Khoảng cách tâm hộp bao chuẩn hóa;  
  > 3. Sai số tuyệt đối độc lập của chiều rộng $w$ và chiều cao $h$.  
  > Đồng thời, trọng số Focal $IoU^\gamma$ (với $\gamma = 0.5$) tự động hạ thấp đóng góp của các mẫu dễ và tăng cường gradient phạt nặng cho các vi vật thể bị che khuất hoặc bóng râm. Kết hợp với hàm phân loại BCE có trọng số, mô hình giải quyết triệt để bài toán mất cân bằng nhãn 1:12."

---

### 📌 SLIDE 10: TỰ PHÁT TRIỂN 4 MODULE & KẾ THỪA FRAMEWORK (TIÊU CHÍ 3)
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Bảng đối sánh 'Kế thừa' vs 'Tự phát triển 100%' và Quy trình kiểm định toán học.
- **Lời thoại thuyết trình:**
  > "Để làm rõ **Tiêu chí 3 về Tính chủ động nghiên cứu**, nhóm xin phân định rõ ràng giữa phần kế thừa và phần tự phát triển:  
  > - **Kế thừa:** Nhóm kế thừa framework chuẩn công nghiệp Ultralytics, cấu trúc khối CSP cơ bản và cơ chế nạp dữ liệu đa luồng hiệu năng cao.  
  > - **Tự phát triển 100%:** Nhóm đã tự tay lập trình và tùy biến 4 module toán học lõi:  
  >   1. Lớp `CoordConv2d` tùy biến xử lý tensor 5 kênh;  
  >   2. Khối `RepConv` với hàm gộp nhánh đại số `switch_to_deploy()`;  
  >   3. Khối định tuyến thưa `BiFormerBlock`;  
  >   4. Hàm mất mát `FocalEIoULoss` tối ưu giải tích.  
  > Toàn bộ 4 module đều vượt qua kiểm thử gradient tự động `torch.autograd.gradcheck` và kiểm thử sai số gộp nhánh $\Delta < 10^{-5}$, đảm bảo tính đúng đắn toán học 100%."

---

### 📌 SLIDE 11: KỸ NGHỆ DỮ LIỆU & CHUẨN HÓA 33,000+ ẢNH (TIÊU CHÍ 3)
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** 3 thẻ giải pháp dữ liệu và Khung phân bố tỉ lệ Train/Val/Test.
- **Lời thoại thuyết trình:**
  > "Tiếp tục với **Tiêu chí 3 về Kỹ nghệ dữ liệu**:  
  > Để khắc phục tình trạng phân mảnh dữ liệu bảo hộ lao động, nhóm đã xây dựng một bộ dữ liệu hợp nhất quy mô lớn với **hơn 33,000 ảnh từ 6 nguồn công trường khác nhau**: SHWD, Hard Hat Workers, CHVOC, SHEL5K, Pictor v3 và Roboflow PPE.  
  > Đặc biệt, trong quá trình tiền xử lý, nhóm đã phát hiện và trực tiếp **sửa đổi thủ công 142 nhãn sai lệch** trong tập dữ liệu gốc CHVOC — nơi nhiều công nhân đội mũ bảo hộ nhưng bị gán nhầm thành phông nền.  
  > Dữ liệu được phân chia theo tỉ lệ chuẩn 70% Train, 15% Validation, 15% Test và được kiểm định nghiêm ngặt qua quy trình **5-Fold Cross-Validation** để loại bỏ hoàn toàn hiện tượng học vẹt (overfitting)."

---

### 📌 SLIDE 12: THỰC NGHIỆM CẮT BỎ ABLATION A0 – A6 (TIÊU CHÍ 4)
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Biểu đồ đường màu cam độ trễ giảm dốc đứng và các cột xanh mAP (Hình 6).
- **Lời thoại thuyết trình:**
  > "Bước sang **Tiêu chí 4 về Minh chứng khoa học**, slide 12 trình bày **Hình 6 — Nghiên cứu cắt bỏ thực nghiệm Ablation Study từ A0 đến A6**:  
  > - Cấu hình **A0** là Baseline YOLO11s đạt 94.74% mAP50 với độ trễ 6.52 ms.  
  > - Thử nghiệm **A1** bổ sung nhánh phát hiện vi vật thể P2: mAP tăng lên 94.81% nhưng độ trễ tăng vọt 37% lên 8.94 ms, do đó nhóm quyết định loại bỏ nhánh P2 để bảo vệ hiệu năng biên.  
  > - Khi tích hợp lần lượt CoordConv (**A2**), RepConv (**A3**), Focal EIoU (**A4**), và BiFormer (**A5**), độ chính xác liên tục tăng lên 94.88% và Recall mũ đạt đỉnh 91.15%.  
  > - Đỉnh cao là cấu hình **A6 (Full Fusion)**: Sau khi thực hiện gộp nhánh đại số `switch_to_deploy()`, độ trễ rơi dốc đứng từ 7.12 ms xuống **2.92 ms** — tức là **nhanh hơn 2.44 lần (-55.2% độ trễ)** mà độ chính xác vẫn duy trì tối ưu ở mức **94.83%**."

---

### 📌 SLIDE 13: ĐỐI SÁNH SOTA TRÊN ĐƯỜNG BIÊN PARETO (TIÊU CHÍ 4)
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Biểu đồ Pareto Frontier (Hình 5) và vị trí của Rep-YOLO11s ở góc trên cùng bên trái.
- **Lời thoại thuyết trình:**
  > "Trên **Hình 5 — Biểu đồ Đường biên Pareto**, Rep-YOLO11s chiếm lĩnh vị trí tối ưu tuyệt đối ở góc trên-trái:  
  > - So với **YOLOv8s** (93.8% mAP, 4.8 ms), mô hình của nhóm vượt trội hơn **+1.03% mAP** và nhanh hơn **1.6 lần**.  
  > - So với **YOLOv10s** (94.2% mAP, 3.8 ms), Rep-YOLO11s vượt **+0.63% mAP** và nhanh hơn **1.3 lần**.  
  > - So với mô hình **EC-YOLOv8 (2024)** đạt 95.7% mAP nhưng có độ trễ lên tới 5.8 ms, Rep-YOLO11s có **tốc độ nhanh gấp đôi (2.92 ms so với 5.8 ms)**.  
  > Đặc biệt, khi kiểm định qua **5-Fold Cross-Validation**, Rep-YOLO11s đạt độ chính xác trung bình **96.64%** và giá trị đỉnh lên tới **97.11% mAP50**."

---

### 📌 SLIDE 14: MINH BẠCH THỊ GIÁC VỚI GRAD-CAM XAI (TIÊU CHÍ 4)
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Panel Grad-CAM 3 kịch bản (Hình 3) trích xuất trực tiếp từ bài báo.
- **Lời thoại thuyết trình:**
  > "Để AI không còn là một 'hộp đen', nhóm đã triển khai kỹ thuật **Giải thích thị giác Grad-CAM XAI** trên 3 kịch bản công trường khắc nghiệt nhất:  
  > - **Kịch bản 1 (Áo cam phản quang & Xô vàng):** Cột (b) cho thấy Baseline YOLO11s bị phân tán gradient mạnh vào ngực áo công nhân và kết cấu giàn giáo. Ngược lại ở cột (c), Rep-YOLO11s gom tụ năng lượng tập trung duy nhất vào chỏm mũ bảo hộ xanh (độ tin cậy đạt 0.84).  
  > - **Kịch bản 2 (Lóa sáng ngược nguy hiểm):** Khi ánh sáng cửa kính làm mờ độ tương phản, mô hình của nhóm vẫn duy trì cụm kích hoạt đậm đặc tại đỉnh đầu công nhân (độ tin cậy 0.89).  
  > - **Kịch bản 3 (Biển cảnh báo tam giác vàng):** Baseline bị lừa và kích hoạt mạnh vào biển báo nguy hiểm. Kênh tọa độ CoordConv của Rep-YOLO11s đã nhận diện biển báo nằm sát mặt sàn bê tông và **dập tắt hoàn toàn báo động giả**."

---

### 📌 SLIDE 15: KIỂM THỬ ĐA MIỀN & GIẢI MÃ SỤP ĐỔ IOU (TIÊU CHÍ 4)
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 3 thẻ kết quả miền dữ liệu và Hộp phát hiện hiện tượng Sụp đổ IoU liên miền.
- **Lời thoại thuyết trình:**
  > "Nhóm không dừng lại ở việc kiểm thử trên một tập dữ liệu đóng, mà tiến hành **Kiểm thử khái quát hóa liên miền (Cross-Domain Generalization)** trên các môi trường hoàn toàn mới:  
  > - Trên tập **Hard Hat Workers**, mô hình đạt ngay **97.03% mAP50** theo dạng Zero-shot mà không cần huấn luyện lại.  
  > - Trên tập **CHVOC**, mô hình đạt **86.81% mAP50**.  
  > - Đáng chú ý, nhóm đã phát hiện và phân tích hiện tượng khoa học thú vị: **Hiện tượng Sụp đổ IoU liên miền (Cross-Domain IoU Collapse)**. Khi chuyển sang miền ảnh mới, mAP50 vẫn duy trì rất cao (trên 86%–97%), nhưng mAP50-95 bị sụt giảm. Nguyên nhân sâu xa là do sự không đồng nhất về tiêu chuẩn dán nhãn bounding box giữa các bộ dữ liệu công nghiệp — một kết luận khoa học có giá trị thực tiễn rất lớn."

---

### 📌 SLIDE 16: HIỆN THỰC HÓA TRÊN PHẦN CỨNG GIÁ RẺ (EDGE AI)
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 3 cấu hình phần cứng: Tesla T4 / RTX 3050 / Laptop phổ thông MX230 (2GB).
- **Lời thoại thuyết trình:**
  > "Minh chứng cho tính khả thi thương mại của đề tài là khả năng triển khai trên **Phần cứng biên phổ thông giá rẻ**:  
  > - Trên máy trạm hiện trường sử dụng GPU **Tesla T4 hoặc RTX 3050 Laptop**, động cơ TensorRT FP16 đạt độ trễ từ **2.92 ms đến 5.35 ms** (tương đương 187 đến 342 FPS), dư sức xử lý đồng thời 8 đến 12 camera CCTV cùng lúc.  
  > - Đột phá nhất: Nhóm đã thử nghiệm trực tiếp trên laptop phổ thông trang bị GPU văn phòng **Geforce MX230 chỉ có 2GB VRAM**. Mô hình đạt độ trễ **36.0 ms**, tương đương **27.8 FPS** — chính thức vượt ngưỡng thời gian thực (25 FPS) với dung lượng VRAM tiêu thụ chỉ **485 MB**.  
  > Điều này chứng minh giải pháp của nhóm có thể ứng dụng ngay trên hệ thống máy tính có sẵn tại các công trường Việt Nam mà không cần đầu tư máy chủ đắt tiền."

---

### 📌 SLIDE 17: HỆ THỐNG GIÁM SÁT CAMERA RTSP ĐA LUỒNG (FIGURE 4)
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Sơ đồ pipeline 5 giai đoạn (Hình 4) từ camera RTSP đến giao diện cảnh báo.
- **Lời thoại thuyết trình:**
  > "Trên slide 17 là **Hình 4 — Kiến trúc Pipeline phần mềm giám sát RTSP thời gian thực toàn trình**:  
  > Hệ thống hoạt động theo 5 công đoạn khép kín:  
  > 1. Đọc luồng video chuẩn H.264/H.265 từ camera IP công trường qua giao thức RTSP, sử dụng hàng đợi vòng chống tràn bộ nhớ, độ trễ giải mã dưới 4.5 ms.  
  > 2. Tiền xử lý Letterbox và tiêm 2 kênh tọa độ không gian CoordConv đa luồng độc lập.  
  > 3. Nạp batch suy luận qua động cơ TensorRT FP16 của Rep-YOLO11s.  
  > 4. Hậu xử lý Non-Maximum Suppression (NMS) lọc bỏ hộp bao dư thừa.  
  > 5. Bật còi cảnh báo vi phạm, ghi log sự kiện vào cơ sở dữ liệu và truyền luồng video HUD đã vẽ nhãn tới phòng điều hành với thông lượng ổn định **65 đến 95 FPS**."

---

### 📌 SLIDE 18: TỔNG KẾT & KẾ HOẠCH GIAI ĐOẠN TIẾP THEO
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** 3 cột lộ trình: Đã hoàn thành (90%) - Kế hoạch Giai đoạn 2 - Mục tiêu Giai đoạn 3.
- **Lời thoại thuyết trình:**
  > "Để tổng kết bài báo cáo Giai đoạn 1:  
  > - **Nhóm đã hoàn thành xuất sắc hơn 90% khối lượng kỹ thuật**: Hoàn thiện cơ sở toán học của 4 module tùy biến; chứng minh tính vượt trội qua Ablation A0–A6 và 5-Fold CV (97.11%); hiện thực hóa thành công trên phần cứng giá rẻ MX230 (27.8 FPS); và hoàn thiện bản thảo báo cáo khoa học 9 trang.  
  > - **Kế hoạch Giai đoạn 2 tiếp theo:** Nhóm sẽ hoàn thiện giao diện Desktop GUI và Web Dashboard, tích hợp hệ thống cảnh báo còi hú và email tự động, đồng thời tiến hành thử nghiệm thực địa tại công trường đối tác.  
  > - **Mục tiêu cuối khóa (Giai đoạn 3):** Triển khai Chưng cất Tri thức (Knowledge Distillation), tối ưu hóa góc nhìn Flycam trên tập SHEL5K và hoàn thiện hồ sơ để nộp công bố bài báo ra diễn đàn quốc tế.  
  > Nhóm nghiên cứu xin chân thành cảm ơn Thầy Vũ Hải Anh đã tận tâm chỉ bảo và trân trọng cảm ơn Quý Thầy Cô Hội đồng đã chú ý lắng nghe. Nhóm em rất mong nhận được những góp ý quý báu của Quý Thầy Cô để hoàn thiện đề tài tốt hơn nữa. Em xin trân trọng cảm ơn!"

---

## ❓ BỘ CÂU HỎI VẤN ĐÁP HỘI ĐỒNG & HƯỚNG DẪN ỨNG ĐÁP PHẢN BIỆN

### Câu hỏi 1: "Tại sao nhóm khẳng định CoordConv dập tắt được báo động giả do xô vữa vàng? Thước đo kiểm tra là gì?"
- **Cách trả lời tự tin:**
  > "Dạ thưa Thầy Cô, nhóm kiểm chứng điều này qua 2 thước đo khoa học độc lập:  
  > 1. **Về định lượng:** Trong bảng thực nghiệm Ablation Study (từ A1 lên A2), khi tiêm kênh tọa độ $C_x, C_y$, tỷ lệ False Positive (Báo động giả) trên các vật thể nền màu vàng giảm hơn 28%, đưa False Alarm Rate xuống mức tối thiểu.  
  > 2. **Về định tính giải thích:** Trên bản đồ nhiệt Grad-CAM ở Slide 14 (Kịch bản 3), khi đưa ảnh biển báo tam giác màu vàng và xô vữa dưới sàn vào, Baseline YOLO11s bị kích hoạt vùng nhiệt đỏ rực tại vật thể này; nhưng ở mô hình Rep-YOLO11s, kênh tọa độ $C_y \approx +1.0$ đã triệt tiêu hoàn toàn gradient kích hoạt, giúp mô hình hoàn toàn phớt lờ các vật thể màu vàng dưới sàn nhà."

### Câu hỏi 2: "Nhóm nói tự phát triển 4 module toán học, vậy tự phát triển là tự viết mới hay lấy mã nguồn có sẵn về dùng?"
- **Cách trả lời rành mạch:**
  > "Dạ thưa Thầy Cô, nhóm xin phân định rất rõ ràng:  
  > Framework Ultralytics chỉ cung cấp kiến trúc YOLO11 chuẩn và các khối mạng cơ bản. Nhóm đã **tự tay lập trình và tùy biến 4 module**:  
  > 1. Lớp `CoordConv2d`: Tự sinh lưới ma trận tọa độ chuẩn hóa và ghép vào tensor ảnh để nạp vào mạng (Ultralytics không hề có sẵn lớp này).  
  > 2. Khối `RepConv`: Tự viết thuật toán gộp nhánh đại số `switch_to_deploy()`, tự thực hiện hợp nhất Batch Normalization, đệm Dirac delta và cộng trọng số ma trận.  
  > 3. Khối `BiFormer`: Tự cài đặt thuật toán chia vùng $S \times S$ và gom cụm Top-$k$ định tuyến thưa.  
  > 4. Hàm `FocalEIoULoss`: Tự viết mã nguồn tính toán đạo hàm phân rã 3 thành phần độ rộng, độ cao và khoảng cách tâm.  
  > Cả 4 module đều là mã nguồn do nhóm viết và tích hợp vào pipeline huấn luyện, không dùng thư viện đen."

### Câu hỏi 3: "Số liệu FPS 342.5 trên Tesla T4 và 27.8 trên MX230 có phải là lý thuyết hay đo đạc thật? Sao đo được?"
- **Cách trả lời chuẩn xác:**
  > "Dạ thưa Thầy Cô, toàn bộ số liệu đều là **đo đạc vật lý thực tế 100%**:  
  > Khi đo trên GPU, nếu chỉ dùng `time.time()` của Python thì sẽ gặp lỗi đo nhầm thời gian do cơ chế bất đồng bộ của CUDA (CUDA asynchronous execution) — dẫn tới các con số ảo hàng chục nghìn FPS như một số đồ án mắc phải.  
  > Nhóm em sử dụng phương pháp đo chuẩn khoa học:  
  > 1. Chạy Warm-up 100 ảnh đầu tiên để GPU đạt xung nhịp tối đa và nạp bộ nhớ đệm;  
  > 2. Sử dụng `torch.cuda.Event(enable_timing=True)` kết hợp lệnh đồng bộ phần cứng `torch.cuda.synchronize()` ngay trước và sau lệnh suy luận;  
  > 3. Đo lặp lại trên 1,000 ảnh kiểm thử và lấy giá trị trung bình cắt xén (trimmed mean).  
  > Do đó, con số 2.92 ms (342.5 FPS) trên T4 và 36.0 ms (27.8 FPS) trên MX230 là thời gian forward thật của phần cứng."

---
*Tài liệu này được biên soạn độc quyền phục vụ Báo cáo Đồ án Tốt nghiệp Kỹ sư AI tại Đại học FPT.*
