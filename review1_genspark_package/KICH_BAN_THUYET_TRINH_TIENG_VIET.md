# 🎙️ KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN TỐT NGHIỆP KỸ SƯ AI (GIAI ĐOẠN 1)
## SLIDE TIẾNG ANH · THUYẾT TRÌNH TIẾNG VIỆT CHUYÊN NGHIỆP & MƯỢT MÀ
### ĐỀ TÀI: REP-YOLO11s — GIÁM SÁT AN TOÀN LAO ĐỘNG CÔNG TRƯỜNG THỜI GIAN THỰC
**TRƯỜNG ĐẠI HỌC FPT · BỘ MÔN TRÍ TUỆ NHÂN TẠO**

---

> **Sinh viên thuyết trình:** Nguyễn Hàn Như (Trưởng nhóm - SE183644)  
> **Thành viên nhóm:** Nguyễn Văn Thành (SE183645), Nguyễn Tuấn Dũng (SE183646)  
> **Giảng viên hướng dẫn:** ThS. Vũ Hải Anh  
> **Quy cách trình chiếu:**
> - **Ngôn ngữ trên Slide:** 100% Tiếng Anh học thuật quốc tế (Chuẩn báo cáo đề tài AI / CVPR / IEEE).
> - **Ngôn ngữ thuyết trình:** 100% Tiếng Việt diễn đạt tự nhiên, rành mạch, chuẩn phong thái Kỹ sư Nghiên cứu.
> - **Bộ Slide PowerPoint:** `Rep-YOLO11s___Capstone_Review_1_Defense_20260920062556.pptx` (hoặc `Rep_YOLO11s_FPT_Defense_Master_Deck.pptx`)
> - **Bản in / Trình chiếu PDF:** `Rep_YOLO11s_FPT_Defense_Master_Deck.pdf` (18 slide, Theme nền sáng, sơ đồ lớn)
> - **Thời lượng đề xuất:** 12 – 15 phút trình bày + 10 – 15 phút vấn đáp Hội đồng.

---

## 🎯 NGUYÊN TẮC THUYẾT TRÌNH "SLIDE TIẾNG ANH - NÓI TIẾNG VIỆT"

1. **Khớp nối tự nhiên giữa chữ trên màn hình và lời nói:**
   - Slide hiển thị các thuật ngữ tiếng Anh chuẩn mực (`Structural Re-parameterization`, `Coordinate Encoding`, `Translation Invariance`, `Sparse Routing Attention`, `Cross-Domain Generalization`).
   - Người thuyết trình nói bằng tiếng Việt tự nhiên, giải thích trực diện bản chất kỹ thuật để các Thầy Cô trong Hội đồng vừa nhìn thấy chuẩn mực quốc tế trên slide, vừa nắm bắt trọn vẹn lập luận logic bằng tiếng Việt.
2. **Tận dụng tối đa 10 Sơ đồ & Biểu đồ lớn (Visual Cues):**
   - Không đọc slide! Mỗi slide chỉ có 3–4 gạch đầu dòng ngắn, hãy dùng tay hoặc con trỏ chuột hướng Hội đồng nhìn vào sơ đồ kiến trúc (Hình 2), sơ đồ gộp nhánh (Slide 6), sơ đồ tọa độ (Slide 7), bản đồ nhiệt Grad-CAM (Hình 3) và biểu đồ Pareto (Hình 5).
3. **Phong thái tự tin của nhóm dẫn đầu:**
   - Dù là đợt thẩm định đề cương Giai đoạn 1, nhóm đã hoàn thành **~90% khối lượng thực nghiệm kỹ thuật** của toàn bộ đồ án, có kiểm định 5-Fold đạt **97.11% mAP50**, chứng minh được tính khả thi trên GPU laptop phổ thông **MX230 (27.8 FPS)** và đã viết xong bài báo khoa học 9 trang.

---

## 📋 LỜI THOẠI THUYẾT TRÌNH CHI TIẾT TỪNG SLIDE (SLIDE 1 ĐẾN 18)

---

### 📌 SLIDE 1: TITLE SLIDE (TRANG TIÊU ĐỀ)
- **Tiêu đề Slide:** `Rep-YOLO11s: Real-Time Safety Helmet Detection in Construction Surveillance`
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** Hướng người nghe nhìn vào khung số liệu cam bên phải (`97.11% mAP50`, `2.92 ms`, `27.8 FPS`).
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Kính thưa quý Thầy Cô trong Hội đồng Đánh giá Đồ án Tốt nghiệp ngành Trí tuệ Nhân tạo - Trường Đại học FPT.  
  > Em tên là **Nguyễn Hàn Như**, đại diện nhóm nghiên cứu gồm em và hai bạn **Nguyễn Văn Thành**, **Nguyễn Tuấn Dũng**, dưới sự hướng dẫn chuyên môn của **Thầy Vũ Hải Anh**, xin được phép báo cáo tiến độ và thẩm định tính khả thi của Đồ án Tốt nghiệp Giai đoạn 1 với đề tài:  
  > **'Rep-YOLO11s: Tái Tham Số Hóa Cấu Trúc, Mã Hóa Tọa Độ Không Gian và Năng Lực Khái Quát Hóa Đa Miền Cho Phát Hiện Mũ Bảo Hộ Thời Gian Thực Trong Giám Sát Công Trường'**.  
  > Nhằm hướng tới chuẩn mực công bố khoa học quốc tế, toàn bộ hệ thống slide được nhóm biên soạn bằng tiếng Anh học thuật. Dù đây là đợt đánh giá đề cương khởi đầu, nhóm chúng em đã hoàn thành hơn 90% khối lượng thực nghiệm kỹ thuật, đạt độ chính xác đỉnh **97.11% mAP50** trên kiểm định 5-Fold Cross-Validation, đồng thời tối ưu hóa tốc độ suy luận đạt **342.5 FPS** trên Tesla T4 và **27.8 FPS** ngay trên GPU laptop phổ thông giá rẻ. Sau đây, em xin phép đi vào chi tiết bài báo cáo."

---

### 📌 SLIDE 2: CRITERION 1 - REAL CONSTRUCTION CONTEXT & 4 TECHNICAL BOTTLENECKS
- **Tiêu đề Slide:** `Real Construction Context & 4 Technical Bottlenecks`
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Hình ảnh thực tế CCTV công trường bên phải (Fig. 1) và 4 thẻ nút thắt bên trái.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Kính thưa Thầy Cô, đáp ứng **Tiêu chí 1 về Tính cấp thiết của đề tài**, chấn thương do vật thể rơi tự do là nguyên nhân hàng đầu gây tử vong tại công trường. Giám sát thủ công hiện chỉ bao quát được dưới 15% thời gian ca trực.  
  > Tuy nhiên, như Thầy Cô quan sát ở **Hình 1** trên slide, khi triển khai camera thị giác máy tính vào thực tế, hệ thống gặp phải **4 nút thắt kỹ thuật cốt lõi**:  
  > 1. **Distant Tiny Objects (<20 px):** Camera CCTV treo cao 15–30 mét khiến mũ bảo hộ chỉ chiếm kích thước dưới 20 pixels, rất dễ bị mờ nhòe và biến mất qua các tầng trích xuất sâu của mạng nơ-ron.  
  > 2. **Extreme Class Imbalance (1:12):** Tỉ lệ nhãn mũ so với thân người là 1:12, khiến gradient bị áp đảo hoàn toàn bởi các mẫu thân người.  
  > 3. **Color Noise & False Alarms:** Công trường tràn ngập xô vữa vàng, cọc tiêu cam, biển báo nguy hiểm. Do mạng CNN tiêu chuẩn có tính 'Bất biến tịnh tiến' (Translation Invariance), mô hình thường nhận diện nhầm các vật thể màu vàng dưới đất thành mũ bảo hộ, gây báo động giả liên tục.  
  > 4. **Edge Hardware Constraints:** Đòi hỏi tốc độ xử lý thời gian thực từ 25 FPS trở lên trên laptop hoặc thiết bị biên giá rẻ tại công trường mà không cần máy chủ đám mây đắt tiền."

---

### 📌 SLIDE 3: CRITERION 3 - LITERATURE REVIEW & SOTA RESEARCH GAP
- **Tiêu đề Slide:** `Literature Review of 32 Works & SOTA Research Gap`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Bảng so sánh 5 dòng mô hình và hộp Khoảng trống nghiên cứu (Research Gap) màu cam dưới đáy.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Về **Tiêu chí 3: Cơ sở khoa học**, nhóm đã khảo sát chuyên sâu **32 công trình quốc tế từ năm 2019 đến 2026**.  
  > Khảo sát cho thấy các mô hình SOTA hiện hữu đều vấp phải sự đánh đổi:  
  > - Dòng **YOLO chuẩn (v8, v11)** chạy nhanh nhưng thiếu nhận thức tọa độ không gian, gây báo động giả liên tục với xô vữa vàng dưới sàn.  
  > - **EC-YOLOv8** đạt mAP cao nhưng dùng toán tử CARAFE nặng nề, làm tụt tốc độ khung hình.  
  > - **YOLO-CBF** tích hợp CoordConv và BiFormer nhưng kiến trúc bị phình to tới 37.2 triệu tham số, tốc độ rơi xuống chỉ còn 80 FPS.  
  > - **YOLOv8n-FADS** mở nhánh P2 cho vi vật thể nhưng gây bùng nổ tính toán trên phần cứng biên.  
  > **Khoảng trống nghiên cứu (Research Gap):** Chưa có công trình nào đạt được đồng thời: Độ chính xác vi vật thể cao + Zero-overhead độ trễ khi suy luận + Khả năng khái quát hóa đa miền bền vững. Đây chính là mục tiêu mà kiến trúc **Rep-YOLO11s** giải quyết trọn vẹn."

---

### 📌 SLIDE 4: CRITERION 2 - PROJECT SCOPE & 5 CONCRETE DELIVERABLES
- **Tiêu đề Slide:** `Project Scope & 5 Concrete Committed Deliverables`
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** 5 thẻ sản phẩm cam kết và banner xanh khẳng định tiến độ đạt 90% ngay tại Giai đoạn 1.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Để trả lời cho **Tiêu chí 2 về Kết quả đầu ra cụ thể**, nhóm cam kết và đã hoàn thành trọn vẹn **5 sản phẩm bàn giao thực tế**:  
  > 1. **Báo cáo khoa học 9 trang** viết bằng LaTeX học thuật chỉn chu, đầy đủ chứng minh toán học giải tích và thực nghiệm.  
  > 2. **Trọng số mô hình Rep-YOLO11s** tối ưu đạt 94.83% mAP50 trên tập kiểm thử và 97.11% trên 5-Fold Cross-Validation.  
  > 3. **Software Prototype** giám sát video RTSP thời gian thực đạt 65–95 FPS trên GPU phổ thông.  
  > 4. **Bộ động cơ biên dịch triển khai** gồm TensorRT 11.2 FP16, ONNX INT8 và OpenVINO cho chip biên.  
  > 5. **Bộ dữ liệu chuẩn hóa công nghiệp** với hơn 33,000 ảnh từ 6 nguồn công trường khác nhau.  
  > Tất cả số đo độ trễ đều được đo vật lý bằng CUDA Events trên phần cứng thật, cam kết 100% không có số liệu mô phỏng ảo."

---

### 📌 SLIDE 5: COMPREHENSIVE 5-STAGE NEURAL ARCHITECTURE (FIGURE 2)
- **Tiêu đề Slide:** `Comprehensive 5-Stage Neural Architecture of Rep-YOLO11s`
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Sơ đồ toàn cảnh Hình 2 (trích từ bài báo gốc) và 3 thẻ tóm tắt Backbone - Neck - Head bên dưới.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Trên slide 5 là **Hình 2 — Sơ đồ kiến trúc toàn diện 5 tầng** của Rep-YOLO11s do nhóm thiết kế:  
  > - **Tầng 1 & 2 (Spatial Backbone):** Mạng CSPDarknet nhận ảnh đầu vào được bổ sung 2 kênh tọa độ chuẩn hóa thông qua tầng **CoordConv Stem**, kết hợp các khối **RepConv** học biểu diễn đa nhánh phong phú trong pha huấn luyện.  
  > - **Tầng 3 (Sparse Routing Neck):** Mạng PAN được nhúng cơ chế chú ý 2 cấp độ **BiFormer**, tự động lọc bỏ 80% phông nền xà bần và gom tụ năng lượng chú ý vào vi vật thể.  
  > - **Tầng 4 & 5 (Decoupled Head & Loss):** Đầu dò Decoupled không anchor kết hợp hàm mất mát **Focal EIoU**, giải quyết triệt để vấn đề mất cân bằng nhãn và sai lệch kích thước hộp bao vi vật thể.  
  > Sau đây, em xin đi sâu vào bản chất toán học của 4 điểm mới này."

---

### 📌 SLIDE 6: INNOVATION 1 - STRUCTURAL RE-PARAMETERIZATION (REPCONV)
- **Tiêu đề Slide:** `Innovation 1: Structural Re-parameterization (RepConv) & Algebraic Fusion`
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Sơ đồ chuyển đổi từ 3 nhánh sang 1 nhánh đơn $3\times3$ thông qua hộp gộp đại số ở giữa.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Điểm mới đầu tiên là kỹ thuật **Tái tham số hóa cấu trúc RepConv**.  
  > - Trong **pha huấn luyện (Training Phase)**, khối RepConv duy trì 3 nhánh song song: tích chập 3x3, tích chập 1x1 và nhánh Identity ma trận đơn vị. Cấu trúc này làm giàu dòng gradient, giúp mô hình học được không gian đặc trưng đa dạng.  
  > - Khi **triển khai suy luận (Inference Phase)**, hàm `switch_to_deploy()` sử dụng phép biến đổi đại số tuyến tính: Hợp nhất Batch Normalization vào trọng số tích chập, đệm zero kernel 1x1 thành kích thước 3x3, và chuyển nhánh Identity thành ma trận Dirac delta.  
  > - Cả 3 nhánh được cộng gộp chính xác thành **duy nhất một lớp Conv 3x3 đơn lẻ**.  
  > Kết quả: Độ trễ suy luận giảm ngoạn mục **55.2%** (từ 7.12 ms xuống 2.92 ms), đạt tốc độ **342.5 FPS** trên Tesla T4 với sai số toán học đại số $\Delta < 10^{-5}$."

---

### 📌 SLIDE 7: INNOVATION 2 - COORDCONV SPATIAL ENCODING SUPPRESSES FALSE ALARMS
- **Tiêu đề Slide:** `Innovation 2: CoordConv Spatial Encoding Suppresses False Alarms`
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Sơ đồ tiêm 2 kênh $C_x, C_y$ và giải thích quy luật giải phẫu vị trí mũ bảo hộ.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Điểm mới thứ hai giải quyết triệt để nút thắt báo động giả: **Tầng mã hóa tọa độ không gian CoordConv**.  
  > Mạng CNN truyền thống có tính chất *bất biến tịnh tiến* (Translation Invariance) — nghĩa là cùng một cụm pixel màu vàng dù nằm trên đầu công nhân hay nằm dưới mặt sàn bê tông đều tạo ra phản ứng kích hoạt tương tự nhau, dẫn tới báo động giả liên tục với xô vữa vàng hay cọc tiêu.  
  > Nhóm đã phá vỡ giới hạn này bằng cách tiêm trực tiếp 2 kênh tọa độ chuẩn hóa $C_x$ và $C_y$ trong đoạn $[-1, 1]$ vào tensor ảnh đầu vào (tạo thành tensor 5 kênh).  
  > Kênh $C_y$ phản ánh độ cao thẳng đứng: $C_y = -1.0$ là đỉnh khung hình và $C_y = +1.0$ là mặt sàn. Qua đó, các kernel tích chập học được tiên đề hình học tự nhiên: **Mũ bảo hộ luôn nằm trên phần thân trên của công nhân ($C_y < 0$), không thể nằm trôi nổi sát mặt đất**.  
  > Kỹ thuật này triệt tiêu hoàn toàn hơn **28% số lượng cảnh báo sai**, và được chứng minh bằng bản đồ nhiệt Grad-CAM ở slide tiếp theo."

---

### 📌 SLIDE 8: INNOVATION 3 - BIFORMER 2-LEVEL SPARSE ROUTING ATTENTION
- **Tiêu đề Slide:** `Innovation 3: BiFormer 2-Level Sparse Routing Attention`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 3 bước xử lý: Chia lưới $S \times S$, Lọc Top-$k$ vùng, và Chú ý Token-to-Token tuyến tính $\mathcal{O}(HW)$.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Điểm mới thứ ba là cơ chế **Chú ý định tuyến thưa 2 cấp độ BiFormer** tích hợp tại tầng Neck.  
  > Cơ chế Self-Attention chuẩn của Vision Transformer có độ phức tạp bậc hai $\mathcal{O}((HW)^2)$, gây nghẽn phần cứng nghiêm trọng. BiFormer giải quyết vấn đề này qua quy trình 3 bước:  
  > - Bước 1: Chia feature map thành lưới $S \times S$ vùng khu vực (với $S=8$), tính vector đại diện cho từng vùng.  
  > - Bước 2: Xây dựng ma trận tương đồng vùng và dùng phép định tuyến thưa Top-$k$ (chọn $k=4$), loại bỏ hơn 80% phông nền xà bần, giàn giáo không liên quan.  
  > - Bước 3: Tính toán chú ý chi tiết Token-to-Token chỉ trong các vùng Top-$k$ đã chọn.  
  > Thuật toán này đưa độ phức tạp về bậc tuyến tính $\mathcal{O}(HW)$, giúp mô hình tập trung 100% năng lượng vào vi vật thể mũ bảo hộ cự ly xa mà không làm tràn bộ nhớ VRAM."

---

### 📌 SLIDE 9: INNOVATION 4 - FOCAL EIOU LOSS & SMALL TARGET OPTIMIZATION
- **Tiêu đề Slide:** `Innovation 4: Focal EIoU Loss & Small Target Optimization`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Sơ đồ phân rã hình học 3 thành phần của Focal EIoU (Trùng khớp, Khoảng cách tâm, Tách cạnh độc lập).
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Điểm mới thứ tư là việc áp dụng **Hàm mất mát Focal EIoU** cho bài toán hồi quy hộp bao.  
  > Hàm CIoU truyền thống gặp hạn chế lớn khi chỉ phạt tỷ lệ co $w/h$. Nếu chiều rộng và chiều cao cùng sai lệch nhưng tỷ lệ $w/h$ tình cờ bằng nhau, đạo hàm phạt góc sẽ bị triệt tiêu, khiến hộp bao vi vật thể không thể hội tụ chuẩn xác.  
  > Focal EIoU phân rã hàm mục tiêu thành 3 thành phần hình học độc lập:  
  > 1. Trùng khớp diện tích IoU chuẩn;  
  > 2. Khoảng cách tâm hộp bao chuẩn hóa;  
  > 3. Sai số tuyệt đối độc lập của chiều rộng $w$ và chiều cao $h$.  
  > Đồng thời, hệ số điều tiết Focal $IoU^{0.5}$ tự động hạ thấp đóng góp của các mẫu dễ và tăng cường gradient cho các mũ bảo hộ bị che khuất một phần dưới giàn giáo, giải quyết hiệu quả bài toán mất cân bằng nhãn 1:12."

---

### 📌 SLIDE 10: CRITERION 3 - INDEPENDENT DEVELOPMENT: 4 CUSTOM MODULES
- **Tiêu đề Slide:** `Independent Development: 4 Custom Mathematical Modules`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 4 thẻ đóng góp mã nguồn lõi và banner cam khẳng định tính tự chủ kỹ thuật 100%.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Để làm rõ **Tiêu chí 3 về Tính tự chủ nghiên cứu**, nhóm xin phân định rành mạch giữa phần kế thừa và phần tự phát triển:  
  > - **Kế thừa:** Nhóm kế thừa framework chuẩn Ultralytics để tận dụng hạ tầng nạp dữ liệu đa luồng và cấu trúc CSP cơ bản.  
  > - **Tự phát triển 100%:** Nhóm đã tự tay lập trình từ đầu 360 dòng mã nguồn thuần PyTorch cho 4 module toán học lõi:  
  >   1. Lớp `CoordConv2d` tùy biến xử lý tensor 5 kênh;  
  >   2. Khối `RepConv` với hàm gộp nhánh đại số `switch_to_deploy()`;  
  >   3. Khối định tuyến thưa `BiFormerBlock`;  
  >   4. Hàm mất mát `FocalEIoULoss` tối ưu giải tích.  
  > Ngoài ra, nhóm đã tự can thiệp tầng thấp vào tệp `loss.py` để vượt qua giới hạn cô lập tiến trình DDP trên Kaggle và xử lý cơ chế bảo mật `weights_only=True` của PyTorch 2.6, đảm bảo tính đúng đắn toán học 100%."

---

### 📌 SLIDE 11: CRITERION 3 - DATA ENGINEERING & MULTI-DOMAIN HARMONIZATION
- **Tiêu đề Slide:** `Data Engineering: Multi-Domain Harmonization & Anomaly Cleaning`
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** 2 thẻ dữ liệu lớn: Tập nguồn SHWD (7,581 ảnh) và 5 tập kiểm định ngoại miền (>25,000 ảnh).
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Về **Kỹ nghệ dữ liệu ở Tiêu chí 3**:  
  > Nhóm đã xây dựng một bộ chuẩn hóa công nghiệp với **hơn 33,000 ảnh từ 6 nguồn công trường khác nhau**:  
  > - Tập dữ liệu nguồn **SHWD (VOC2028)** gồm 7,581 ảnh, được phân chia nghiêm ngặt 80% TrainVal và 20% Test độc lập (hoàn toàn không rò rỉ dữ liệu). Đặc biệt, nhóm đã phát hiện và loại bỏ các nhãn bất thường (như nhãn 'dog' trong ảnh 000377).  
  > - Hệ thống **5 tập dữ liệu kiểm thử ngoại miền (>25,000 ảnh)** với các thử thách khắc nghiệt: GDUT-HWD với đám đông cực dày đặc 15–30 người/khung hình; SHEL5K với góc quay Flycam thẳng đứng từ trên cao; Hard Hat Workers ngoài trời; và các bộ dữ liệu nhà máy luyện kim SHD, SFCHD.  
  > Toàn bộ được đồng bộ hóa về không gian nhãn thống nhất: 0 là Mũ bảo hộ ('hat') và 1 là Thân người ('person')."

---

### 📌 SLIDE 12: CRITERION 4 - ABLATION STUDY A0–A6 EMPIRICAL VERIFICATION
- **Tiêu đề Slide:** `Ablation Study (A0–A6): Rigorous Component Verification`
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Biểu đồ thực nghiệm Hình 6 bên phải với đường màu cam biểu diễn độ trễ giảm dốc đứng khi gộp nhánh.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Bước sang **Tiêu chí 4 về Minh chứng khoa học**, slide 12 trình bày **Hình 6 — Nghiên cứu cắt bỏ thực nghiệm Ablation Study từ A0 đến A6**:  
  > - Cấu hình **A0** là Baseline YOLO11s đạt 94.74% mAP50 với độ trễ 6.52 ms.  
  > - Thử nghiệm **A1** thêm nhánh P2: mAP tăng nhẹ lên 94.81% nhưng độ trễ tăng vọt 37% lên 8.94 ms, do đó nhóm quyết định loại bỏ nhánh P2 để bảo toàn hiệu năng phần cứng.  
  > - Khi tích hợp lần lượt CoordConv (**A2**), RepConv (**A3**), Focal EIoU (**A4**), và BiFormer (**A5**), độ chính xác liên tục tăng lên 94.88% và Recall mũ đạt đỉnh 91.15%.  
  > - Đột phá xuất hiện ở cấu hình **A6 (Full Fusion)**: Sau khi thực hiện gộp nhánh đại số `switch_to_deploy()`, độ trễ rơi dốc đứng từ 7.12 ms xuống **2.92 ms** — tức là **nhanh hơn 2.44 lần (-55.2% độ trễ)** trong khi độ chính xác duy trì tối ưu ở mức **94.83%**."

---

### 📌 SLIDE 13: CRITERION 4 - SOTA BENCHMARKING ON PARETO FRONTIER
- **Tiêu đề Slide:** `State-of-the-Art Benchmarking: Pareto Frontier Dominance`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Biểu đồ Pareto Frontier Hình 5 bên phải, chỉ vào điểm Rep-YOLO11s ở góc trên cùng bên trái.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Trên **Hình 5 — Biểu đồ Đường biên Pareto**, Rep-YOLO11s chiếm lĩnh vị trí tối ưu tuyệt đối ở góc trên-trái:  
  > - So với **Baseline YOLO11s**, mô hình của nhóm **nhanh gấp 2.23 lần** (2.92 ms so với 6.52 ms trên Tesla T4).  
  > - So với mô hình **YOLO-CBF**, Rep-YOLO11s **nhanh gấp 4.25 lần** (2.92 ms so với 12.4 ms).  
  > - So với **EC-YOLOv8 (2024)** đạt 172.4 FPS, mô hình của nhóm đạt **342.5 FPS — nhanh gấp đôi**.  
  > Đặc biệt, khi kiểm định qua **5-Fold Stratified Cross-Validation**, Rep-YOLO11s đạt độ chính xác trung bình **96.64%** và giá trị đỉnh Fold 3 đạt **97.11% mAP50** với chỉ số F1 đạt 0.9396."

---

### 📌 SLIDE 14: CRITERION 4 - VISUAL EXPLAINABILITY VIA GRAD-CAM XAI
- **Tiêu đề Slide:** `Visual Explainability (Grad-CAM XAI): Saliency & Noise Suppression`
- **Thời lượng:** 60 giây
- **Điểm nhìn thị giác:** Panel bản đồ nhiệt Grad-CAM Hình 3 bên phải trích trực tiếp từ bài báo.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Để AI không còn là một 'hộp đen', nhóm đã triển khai kỹ thuật **Giải thích thị giác Grad-CAM XAI** trên 3 kịch bản công trường khắc nghiệt:  
  > - **Kịch bản 1 (Áo cam phản quang):** Cột (b) cho thấy Baseline bị phân tán gradient mạnh vào thân áo và giàn giáo. Ngược lại ở cột (c), Rep-YOLO11s gom tụ năng lượng tập trung duy nhất vào vòm mũ bảo hộ xanh (độ tin cậy đạt 0.84).  
  > - **Kịch bản 2 (Lóa sáng ngược nguy hiểm):** Khi ánh sáng cửa kính làm mờ độ tương phản, mô hình của nhóm vẫn duy trì cụm kích hoạt đậm đặc tại đỉnh đầu công nhân (độ tin cậy 0.89).  
  > - **Kịch bản 3 (Biển cảnh báo tam giác vàng):** Baseline bị đánh lừa và kích hoạt mạnh vào biển báo nguy hiểm. Kênh tọa độ CoordConv của Rep-YOLO11s đã nhận diện biển báo nằm sát sàn bê tông và **dập tắt hoàn toàn báo động giả**."

---

### 📌 SLIDE 15: CRITERION 4 - CROSS-DOMAIN GENERALIZATION & IOU COLLAPSE AUTOPSY
- **Tiêu đề Slide:** `Cross-Domain Generalization & IoU Collapse Autopsy`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 3 thẻ điểm kiểm thử ngoại miền ở trên và Hộp giải mã hiện tượng Sụp đổ IoU ở dưới.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Nhóm tiến hành kiểm thử **Khái quát hóa liên miền (Cross-Domain Generalization)** trên các môi trường hoàn toàn mới:  
  > - Trên tập **Hard Hat Workers**, mô hình đạt ngay **97.03% mAP50** theo dạng Zero-shot mà không cần huấn luyện lại.  
  > - Trên tập **GDUT-HWD**, mô hình đạt Precision ấn tượng **90.26%** trong điều kiện công nhân đứng san sát che khuất lẫn nhau.  
  > - Đáng chú ý, nhóm đã phát hiện và phân tích hiện tượng khoa học sâu sắc: **Hiện tượng Sụp đổ IoU liên miền (Cross-Domain IoU Collapse)**. Khi chuyển sang tập Hard Hat Workers, mAP gộp bị tụt do tập nguồn gán nhãn thân người là 'Toàn thân' (Full-body), còn tập đích chỉ gán nhãn là 'Nửa đầu' (Head-only). Khi áp dụng Giao thức Harmonized PPE đánh giá riêng lớp mũ, mAP lập tức nhảy vọt lên 97.03%, chứng minh các đặc trưng thị giác học được là cực kỳ vững chắc."

---

### 📌 SLIDE 16: CRITERION 4 - EDGE HARDWARE REALIZATION ON BUDGET GPU (MX230 2GB)
- **Tiêu đề Slide:** `Edge Hardware Realization: Real-Time on 2GB MX230 Laptop GPU`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** 3 tầng phần cứng và Hộp màu xanh lá nổi bật giải thích ý nghĩa kinh tế của GPU MX230 2GB.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Minh chứng cho tính khả thi thương mại của đề tài là khả năng triển khai trên **Phần cứng biên phổ thông giá rẻ**:  
  > - Trên máy trạm hiện trường sử dụng GPU **Tesla T4 hoặc RTX 3050 Laptop**, động cơ TensorRT FP16 đạt độ trễ từ **2.92 ms đến 5.35 ms** (tương đương 187 đến 342 FPS), dư sức xử lý đồng thời 8 đến 12 camera CCTV cùng lúc.  
  > - Đột phá nhất: Nhóm đã thử nghiệm trực tiếp trên laptop văn phòng phổ thông trang bị GPU **GeForce MX230 chỉ có 2GB VRAM** (ra đời từ năm 2019, không có Tensor Core). Mô hình đạt độ trễ **36.0 ms**, tương đương **27.8 FPS** — chính thức vượt ngưỡng thời gian thực (24 FPS) với lượng tiêu thụ VRAM chỉ **485 MB**.  
  > Điều này chứng minh giải pháp của nhóm có thể ứng dụng ngay trên máy tính xách tay cũ của chỉ huy trưởng công trường mà không cần đầu tư máy chủ đắt đỏ, giúp tiết kiệm hàng trăm triệu đồng chi phí đầu tư!"

---

### 📌 SLIDE 17: PROTOTYPE PRODUCT - MULTI-STREAM RTSP PIPELINE (FIGURE 4)
- **Tiêu đề Slide:** `End-to-End Industrial Multi-Stream RTSP Surveillance Pipeline`
- **Thời lượng:** 50 giây
- **Điểm nhìn thị giác:** Sơ đồ pipeline toàn trình Hình 4 trích từ bài báo gốc.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Trên slide 17 là **Hình 4 — Kiến trúc Pipeline phần mềm giám sát RTSP thời gian thực toàn trình**:  
  > Hệ thống hoạt động theo quy trình 5 bước khép kín:  
  > 1. Đọc luồng video H.264/H.265 từ camera IP qua giao thức RTSP, sử dụng hàng đợi vòng chống tràn bộ nhớ, độ trễ giải mã dưới 4.5 ms.  
  > 2. Tiền xử lý Letterbox và tiêm 2 kênh tọa độ CoordConv đa luồng song song.  
  > 3. Nạp batch suy luận qua động cơ TensorRT FP16 của Rep-YOLO11s.  
  > 4. Hậu xử lý Non-Maximum Suppression (NMS) lọc bỏ hộp bao dư thừa.  
  > 5. Kích hoạt còi hú cảnh báo, ghi log sự kiện và truyền luồng video HUD đã vẽ nhãn tới phòng giám sát với tốc độ ổn định **65 đến 95 FPS**."

---

### 📌 SLIDE 18: SUMMARY & ROADMAP FOR STAGES 2 - 3 (TỔNG KẾT & LỘ TRÌNH)
- **Tiêu đề Slide:** `Stage 1 Accomplishments & Roadmap for Stages 2 – 3`
- **Thời lượng:** 45 giây
- **Điểm nhìn thị giác:** 3 cột lộ trình: Giai đoạn 1 (Đã đạt 90%) - Giai đoạn 2 - Giai đoạn 3, kết thúc bằng lời cảm ơn Hội đồng.
- **Lời thoại thuyết trình (Tiếng Việt):**
  > "Để tổng kết bài báo cáo Giai đoạn 1:  
  > - **Nhóm đã hoàn thành xuất sắc hơn 90% khối lượng kỹ thuật**: Hoàn thiện cơ sở toán học của 4 module tùy biến; chứng minh tính vượt trội qua Ablation A0–A6 và 5-Fold CV (97.11%); hiện thực hóa thành công trên phần cứng giá rẻ MX230 (27.8 FPS); và hoàn thiện bản thảo bài báo khoa học 9 trang.  
  > - **Kế hoạch Giai đoạn 2 tiếp theo:** Nhóm sẽ hoàn thiện giao diện Desktop GUI và Web Dashboard, tích hợp hệ thống cảnh báo còi hú và email tự động, đồng thời tiến hành thử nghiệm thực địa tại công trường đối tác.  
  > - **Mục tiêu cuối khóa (Giai đoạn 3):** Triển khai Chưng cất Tri thức (Knowledge Distillation), tối ưu hóa góc nhìn Flycam trên tập SHEL5K và hoàn thiện hồ sơ để nộp công bố bài báo ra diễn đàn quốc tế.  
  > Nhóm nghiên cứu xin chân thành cảm ơn Thầy Vũ Hải Anh đã tận tâm chỉ bảo và trân trọng cảm ơn Quý Thầy Cô Hội đồng đã chú ý lắng nghe. Nhóm em rất mong nhận được những góp ý quý báu của Quý Thầy Cô. Em xin trân trọng cảm ơn!"

---

## ❓ BỘ CÂU HỎI VẤN ĐÁP HỘI ĐỒNG & MẸO ỨNG ĐÁP PHẢN BIỆN NHANH

### Câu hỏi 1: "Slide ghi tiếng Anh nhưng nhóm bảo vệ bằng tiếng Việt, tại sao lại làm vậy?"
- **Cách trả lời đĩnh đạc:**
  > "Dạ thưa Thầy Cô, nhóm chuẩn bị bộ slide bằng tiếng Anh nhằm mục tiêu kép:  
  > 1. Toàn bộ nghiên cứu của nhóm được thiết kế theo chuẩn mực quốc tế, bài báo 9 trang viết bằng tiếng Anh để nộp công bố tại các diễn đàn khoa học uy tín, nên thuật ngữ trên slide thể hiện tính chuẩn xác của ngành.  
  > 2. Thuyết trình bằng tiếng Việt giúp nhóm truyền đạt trọn vẹn và tự nhiên nhất các lập luận kỹ thuật, bảo đảm tính mạch lạc và tương tác tốt nhất với Hội đồng trong buổi thẩm định đề cương hôm nay."

### Câu hỏi 2: "Tại sao nhóm khẳng định CoordConv dập tắt được báo động giả do xô vữa vàng? Thước đo kiểm tra là gì?"
- **Cách trả lời tự tin:**
  > "Dạ thưa Thầy Cô, nhóm kiểm chứng điều này qua 2 thước đo khoa học độc lập:  
  > 1. **Về định lượng:** Trong bảng thực nghiệm Ablation Study (từ A1 lên A2), khi tiêm kênh tọa độ $C_x, C_y$, tỷ lệ False Positive trên các vật thể nền màu vàng giảm hơn 28%, đưa False Alarm Rate xuống mức tối thiểu.  
  > 2. **Về định tính giải thích:** Trên bản đồ nhiệt Grad-CAM ở Slide 14 (Kịch bản 3), khi đưa ảnh biển báo tam giác màu vàng và xô vữa dưới sàn vào, Baseline YOLO11s bị kích hoạt vùng nhiệt đỏ rực tại vật thể này; nhưng ở mô hình Rep-YOLO11s, kênh tọa độ $C_y \approx +1.0$ đã triệt tiêu hoàn toàn gradient kích hoạt, giúp mô hình hoàn toàn phớt lờ các vật thể màu vàng dưới sàn nhà."

### Câu hỏi 3: "Nhóm nói tự phát triển 4 module toán học, vậy tự phát triển là tự viết mới hay lấy mã nguồn có sẵn về dùng?"
- **Cách trả lời rành mạch:**
  > "Dạ thưa Thầy Cô, nhóm xin phân định rất rõ ràng:  
  > Framework Ultralytics chỉ cung cấp kiến trúc YOLO11 chuẩn và các khối mạng cơ bản. Nhóm đã **tự tay lập trình và tùy biến 4 module**:  
  > 1. Lớp `CoordConv2d`: Tự sinh lưới ma trận tọa độ chuẩn hóa và ghép vào tensor ảnh để nạp vào mạng (Ultralytics không hề có sẵn lớp này).  
  > 2. Khối `RepConv`: Tự viết thuật toán gộp nhánh đại số `switch_to_deploy()`, tự thực hiện hợp nhất Batch Normalization, đệm Dirac delta và cộng trọng số ma trận.  
  > 3. Khối `BiFormer`: Tự cài đặt thuật toán chia vùng $S \times S$ và gom cụm Top-$k$ định tuyến thưa.  
  > 4. Hàm `FocalEIoULoss`: Tự viết mã nguồn tính toán đạo hàm phân rã 3 thành phần độ rộng, độ cao và khoảng cách tâm.  
  > Cả 4 module đều là mã nguồn do nhóm viết và tích hợp vào pipeline huấn luyện, không dùng thư viện đen."

### Câu hỏi 4: "Số liệu FPS 342.5 trên Tesla T4 và 27.8 trên MX230 có phải là lý thuyết hay đo đạc thật? Sao đo được?"
- **Cách trả lời chuẩn xác:**
  > "Dạ thưa Thầy Cô, toàn bộ số liệu đều là **đo đạc vật lý thực tế 100%**:  
  > Khi đo trên GPU, nếu chỉ dùng `time.time()` của Python thì sẽ gặp lỗi đo nhầm thời gian do cơ chế bất đồng bộ của CUDA (CUDA asynchronous execution) — dẫn tới các con số ảo hàng chục nghìn FPS như một số đồ án mắc phải.  
  > Nhóm em sử dụng phương pháp đo chuẩn khoa học:  
  > 1. Chạy Warm-up 100 ảnh đầu tiên để GPU đạt xung nhịp tối đa và nạp bộ nhớ đệm;  
  > 2. Sử dụng `torch.cuda.Event(enable_timing=True)` kết hợp lệnh đồng bộ phần cứng `torch.cuda.synchronize()` ngay trước và sau lệnh suy luận;  
  > 3. Đo lặp lại trên 1,000 ảnh kiểm thử và lấy giá trị trung bình cắt xén (trimmed mean).  
  > Do đó, con số 2.92 ms (342.5 FPS) trên T4 và 36.0 ms (27.8 FPS) trên MX230 là thời gian forward thật của phần cứng."

---
*Tài liệu được biên soạn độc quyền phục vụ Báo cáo Đồ án Tốt nghiệp Kỹ sư AI tại Đại học FPT.*
