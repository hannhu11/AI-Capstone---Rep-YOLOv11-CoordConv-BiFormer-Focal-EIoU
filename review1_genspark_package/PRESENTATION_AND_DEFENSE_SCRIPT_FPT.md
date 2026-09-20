# 🎙️ KỊCH BẢN THUYẾT TRÌNH & PHẢN BIỆN HỘI ĐỒNG BẢO VỆ ĐATN KỸ SƯ AI
## ĐỀ TÀI: REP-YOLO11s — GIÁM SÁT AN TOÀN LAO ĐỘNG CÔNG TRƯỜNG THỜI GIAN THỰC
### TRƯỜNG ĐẠI HỌC FPT · BỘ MÔN TRÍ TUỆ NHÂN TẠO (GIAI ĐOẠN 1)

> **Tác giả:** Nguyễn Hàn Như (Trưởng nhóm - SE183644), Nguyễn Văn Thành (SE183645), Nguyễn Tuấn Dũng (SE183646)  
> **Giảng viên hướng dẫn:** ThS. Vũ Hải Anh  
> **Tài liệu tham chiếu:** `Rep-YOLO11s_Master_Paper_Final.pdf`, `AI_Capstone_Defense_Template.xlsx`  
> **Bộ Slide trình chiếu:** `Rep-YOLO11s___Capstone_Review_1_Defense_20260920062556.pptx` (18 slides, 100% Light Theme, tuân thủ chuẩn danh pháp Q1 và các Giai đoạn Đồ án)  
> **Tổng thời lượng:** 15 – 20 phút thuyết trình + 10 – 15 phút vấn đáp phản biện.

---

# PHẦN 1: CHIẾN LƯỢC TÂM LÝ & NGUYÊN TẮC BẢO VỆ HỘI ĐỒNG

1. **Vị thế tự tin vượt trội (Thẩm định khả thi 100%):**
   - Đa số các nhóm ở giai đoạn khởi đầu bảo vệ đề cương chỉ có ý tưởng lý thuyết trên giấy. Tuy nhiên, nhóm chúng ta **đã hoàn thành ~90% khối lượng nghiên cứu kỹ thuật**, có bài báo khoa học 9 trang, có thực nghiệm SOTA, Ablation A0–A6, XAI Grad-CAM, Cross-domain 33,000 ảnh và đo đạc vật lý trên phần cứng thật (Tesla T4, RTX 3050, MX230 2GB).
   - Hãy trình bày với phong thái của một **Kỹ sư Nghiên cứu AI Ứng dụng thực thụ**: Nói bằng số liệu đo đạc vật lý, bằng công thức toán học tường minh, không dùng từ ngữ phỏng đoán ("em nghĩ là", "chắc là").
2. **Khắc cốt ghi tâm 4 Tiêu chí của Hội đồng FPT:**
   - Khi chuyển slide, luôn gắn kết với 4 tiêu chí đánh giá trong rubric:
     * *Tiêu chí 1:* Phát biểu bài toán rõ ràng, cấp thiết (4 nút thắt).
     * *Tiêu chí 2:* Kết quả đầu ra cụ thể (5 sản phẩm bàn giao cam kết).
     * *Tiêu chí 3:* Tính khả thi & Scope (32 bài báo, kế thừa Ultralytics + 4 module tự phát triển, làm sạch dữ liệu).
     * *Tiêu chí 4:* Giá trị thực tiễn & Ý nghĩa khoa học (Chạy trên MX230 2GB @ 27.8 FPS, T4 @ 342.5 FPS, RTSP @ 65-95 FPS, Grad-CAM, Harmonized PPE).
3. **Tuyệt đối tuân thủ ngôn từ quy chuẩn:**
   - Quy chuẩn danh pháp: Sử dụng thuật ngữ học thuật chuẩn "Báo cáo Đồ án Tốt nghiệp", "Giai đoạn 1 (Khởi tạo đề cương & Thẩm định khả thi)", "Giai đoạn 2", "Giai đoạn 3". Không dùng các danh xưng hội nghị hay các từ tiếng Anh không chính thức.

---

# PHẦN 2: LỜI THOẠI THUYẾT TRÌNH CHI TIẾT TỪNG SLIDE (SLIDE-BY-SLIDE SCRIPT)

### 📌 SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
- **Thời lượng:** 0:45
- **Lời thoại:**
  > "Kính thưa quý Thầy, Cô trong Hội đồng Đánh giá Đồ án Tốt nghiệp ngành Trí tuệ Nhân tạo - Trường Đại học FPT.  
  > Em tên là Nguyễn Hàn Như, đại diện nhóm nghiên cứu gồm em và hai bạn Nguyễn Văn Thành, Nguyễn Tuấn Dũng, dưới sự hướng dẫn chuyên môn của Thầy Vũ Hải Anh, xin được phép báo cáo tiến độ và thẩm định tính khả thi của Đồ án Tốt nghiệp với đề tài:  
  > **'Rep-YOLO11s: Tái Tham Số Hóa Cấu Trúc, Mã Hóa Tọa Độ Không Gian và Năng Lực Tổng Quát Hóa Đa Miền Cho Phát Hiện Mũ Bảo Hộ Thời Gian Thực Trong Giám Sát Công Trường'**.  
  > Mặc dù đây là đợt đánh giá đề cương khởi đầu (Giai đoạn 1), nhóm chúng em đã chủ động hoàn thành hơn 90% khối lượng thực nghiệm kỹ thuật, đạt độ chính xác đỉnh **97.11% mAP50** trên 5-Fold Cross-Validation và kiểm chứng tốc độ suy luận vật lý vượt trội **342.5 FPS** trên GPU Tesla T4 và **27.8 FPS** trên laptop phổ thông giá rẻ. Sau đây, em xin phép được trình bày chi tiết."

---

### 📌 SLIDE 2: MA TRẬN ĐÁP ỨNG 4 TIÊU CHÍ HỘI ĐỒNG FPT
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Để quý Thầy Cô tiện theo dõi và đối chiếu trực tiếp với biểu mẫu đánh giá của Bộ môn AI, Slide 2 tóm lược Ma trận đáp ứng trọn vẹn 4 Tiêu chí cốt lõi:  
  > - **Tiêu chí 1 về Mục tiêu đề tài:** Chúng em định nghĩa bài toán phát hiện mũ bảo hộ lao động thời gian thực thông qua việc giải quyết triệt để 4 nút thắt vật lý và toán học của công trường.  
  > - **Tiêu chí 2 về Kết quả cuối cùng:** Nhóm cam kết và đã xây dựng 5 sản phẩm đầu ra hoàn chỉnh: từ Báo cáo khoa học chuyên sâu 9 trang, Software prototype RTSP 65–95 FPS, Checkpoint và TensorRT Engine, đến Bộ dữ liệu chuẩn hóa 33,000 ảnh.  
  > - **Tiêu chí 3 về Tính khả thi và Scope:** Nhóm khảo sát chuyên sâu 32 công trình quốc tế; kế thừa framework chuẩn Ultralytics và tự tay phát triển 4 module toán học chuyên biệt; xử lý làm sạch tập VOC2028 loại bỏ các nhãn sai lệch.  
  > - **Tiêu chí 4 về Giá trị đề tài:** Đảm bảo khả năng vận hành thực tế trên hạ tầng camera CCTV sẵn có, chạy mượt mà ngay trên laptop yếu MX230 2GB VRAM; đồng thời mang ý nghĩa khoa học sâu sắc khi phá vỡ tính bất biến tịnh tiến của CNN và giải mã hiện tượng Sụp đổ IoU liên miền.  
  > Toàn bộ nội dung tiếp theo sẽ là minh chứng cụ thể cho 4 tiêu chí này."

---

### 📌 SLIDE 3: TIÊU CHÍ 1 - THỰC TRẠNG & BỐI CẢNH THỰC TIỄN
- **Thời lượng:** 1:00
- **Lời thoại:**
  > "Bước vào Tiêu chí 1, tại sao bài toán này lại cấp thiết?  
  > Theo thống kê an toàn lao động công nghiệp toàn cầu, chấn thương sọ não do vật thể rơi tự do từ trên cao là nguyên nhân hàng đầu gây tử vong và tàn phế vĩnh viễn cho công nhân xây dựng. Chiếc mũ bảo hộ là lằn ranh sinh tử.  
  > Tuy nhiên, công tác giám sát an toàn hiện nay gần như phụ thuộc vào việc đi tuần thủ công của cán bộ an toàn. Phương pháp này chỉ bao quát được dưới 15% thời lượng ca trực, tầm nhìn bị che khuất và chi phí nhân sự rất lớn.  
  > Như Thầy Cô có thể thấy ở **Hình 1** trên slide, được trích xuất từ hiện trường công trường thực tế: Camera CCTV thường được lắp đặt ở độ cao 15 đến 30 mét, góc nhìn nghiêng dốc, công nhân bị giàn giáo và vật liệu che lấp. Mục tiêu của đề tài chúng em là biến từng mắt camera CCTV/RTSP hiện hữu thành một kiểm toán viên an toàn tự động, giám sát liên tục 24/7 với chi phí phần cứng tối thiểu."

---

### 📌 SLIDE 4: TIÊU CHÍ 1 - BỐN NÚT THẮT KỸ THUẬT CỐT LÕI
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Khi đưa các mô hình thị giác máy tính vào giám sát công trường thực tế, chúng ta phải đối mặt với 4 nút thắt kỹ thuật mà các mạng CNN thông thường không thể xử lý tốt:  
  > 1. **Vi vật thể cự ly xa:** Camera góc cao khiến mũ bảo hộ chỉ chiếm kích thước dưới 20x20 pixels, thậm chí dưới 15 pixels. Qua các tầng downsampling stride 16 và 32 của backbone chuẩn, đặc trưng không gian của mũ bị triệt tiêu hoàn toàn.  
  > 2. **Mất cân bằng nhãn cực đoan:** Trên tập dữ liệu chuẩn SHWD, số lượng mũ chỉ có 9,044 so với 111,514 thân người (tỷ lệ lệch tới 1:12). Gradient bị áp đảo hoàn toàn bởi các mẫu thân người và phông nền dễ học.  
  > 3. **Nhiễu không gian và vật thể gây nhầm lẫn:** Công trường tràn ngập xô vữa nhựa màu vàng, cọc tiêu cam, biển báo nguy hiểm tam giác. Mạng CNN truyền thống có tính 'Bất biến tịnh tiến' (Translation Invariance) nên không phân biệt được màu vàng trên đầu người hay màu vàng của vật tư dưới đất, gây báo động giả liên tục.  
  > 4. **Rào cản phần cứng biên:** Hệ thống đòi hỏi thông lượng thời gian thực từ 60 FPS trở lên trên máy tính trạm đặt tại công trường, không thể phụ thuộc vào máy chủ đám mây đắt tiền."

---

### 📌 SLIDE 5: TIÊU CHÍ 2 - NĂM SẢN PHẨM BÀN GIAO CAM KẾT
- **Thời lượng:** 1:00
- **Lời thoại:**
  > "Để trả lời cho câu hỏi ở Tiêu chí 2: 'Kết quả cuối cùng là gì?', nhóm cam kết và đã hoàn thành 5 sản phẩm đầu ra cụ thể:  
  > 1. **Báo cáo nghiên cứu khoa học chuyên sâu 9 trang:** Viết bằng LaTeX học thuật chỉn chu, đầy đủ chứng minh toán học, Ablation Study A0–A6, XAI Grad-CAM và 32 tài liệu tham khảo quốc tế.  
  > 2. **Bộ trọng số mô hình Rep-YOLO11s tối ưu:** Đạt 94.83% mAP50 trên tập kiểm thử đơn lẻ, 96.64% trên 5-Fold Cross-Validation và 97.03% trên tập Hard Hat Workers.  
  > 3. **Software Prototype giám sát RTSP thời gian thực:** Tiếp nhận luồng H.264/H.265, tiền xử lý, suy luận và vẽ giao diện cảnh báo với tốc độ 65 đến 95 FPS trên laptop RTX 3050.  
  > 4. **Bộ động cơ biên dịch triển khai đa nền tảng:** Xuất sẵn TensorRT 11.2 FP16, ONNX Runtime INT8 và OpenVINO cho chip biên.  
  > 5. **Bộ dữ liệu chuẩn hóa công nghiệp:** Hơn 33,000 ảnh từ 6 nguồn công trường khác nhau.  
  > Toàn bộ số đo độ trễ trên bảng bên phải đều được đo vật lý bằng CUDA Events trên phần cứng thật, cam kết 100% không có số liệu ảo."

---

### 📌 SLIDE 6: TIÊU CHÍ 3 - KHẢO SÁT 32 CÔNG TRÌNH QUỐC TẾ & KHOẢNG TRỐNG KHOA HỌC
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Về Tiêu chí 3: Tính khả thi và Xác định Scope, nhóm đã khảo sát chuyên sâu 32 công trình khoa học quốc tế uy tín từ năm 2019 đến 2026.  
  > Phân tích các hướng tiếp cận hiện hữu:  
  > - Dòng YOLO chuẩn (v8, v10, 11) có tốc độ cao nhưng thiếu tiên đề tọa độ không gian, dễ nhận diện nhầm các vật thể màu vàng dưới mặt đất thành mũ bảo hộ.  
  > - Mô hình EC-YOLOv8 năm 2024 đạt 95.70% mAP nhưng dùng toán tử CARAFE nặng nề, kéo thông lượng GPU xuống 172 FPS.  
  > - Mô hình YOLO-CBF năm 2023 đưa CoordConv và BiFormer vào nhưng kiến trúc bị phình to tới 37.2 triệu tham số, tốc độ rơi xuống 80.6 FPS.  
  > - Mô hình YOLOv8n-FADS năm 2024 mở rộng thêm nhánh P2 cho mỏ than nhưng làm bùng nổ kích thước feature map, gây nghẽn độ trễ.  
  > **Khoảng trống khoa học (Research Gap):** Chưa có công trình nào đạt được đồng thời: Độ chính xác vi vật thể cao + Zero-overhead độ trễ khi suy luận + Khả năng chuyển giao ngoại miền vững chắc. Đề tài của chúng em ra đời để giải quyết chính xác khoảng trống này."

---

### 📌 SLIDE 7: TIÊU CHÍ 3 - KẾ THỪA FRAMEWORK & TỰ PHÁT TRIỂN 4 MODULE
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Để trả lời câu hỏi phản biện của Hội đồng: 'Triển khai từ đầu hay tái sử dụng thư viện? Có đóng góp thuật toán mới không?':  
  > - **Kế thừa:** Nhóm kế thừa framework Ultralytics YOLO11s phiên bản mới nhất, tận dụng kiến trúc Backbone C3k2, SPPF, Decoupled Head và quy trình huấn luyện hiện đại (Mosaic, MixUp, Cosine LR). Điều này giúp nhóm không phải lãng phí thời gian 'phát minh lại bánh xe'.  
  > - **Tự phát triển & Tùy biến:** Nhóm đã tự tay thiết kế và viết 360 dòng mã nguồn PyTorch thuần trong file `custom_ablation_modules.py`, bao gồm 4 module toán học chuyên sâu:  
  >   1. Module `CoordConv`: Tự tạo 2 kênh tọa độ tiêm vào tầng stem conv.  
  >   2. Module `RepConv`: Tái tham số hóa cấu trúc 3 nhánh lúc train và giải thuật đại số tuyến tính `switch_to_deploy` đưa về nhánh đơn lúc inference.  
  >   3. Module `BiFormer`: Cơ chế chú ý định tuyến thưa hai tầng lọc top-$k$ vùng đặc trưng.  
  >   4. Custom Loss `Focal EIoU`: Hard-patch trực tiếp vào hệ thống tính loss của Ultralytics.  
  > Nhóm khẳng định: Chúng em không chỉ lấy model có sẵn về train, mà đã can thiệp sâu vào cấu trúc toán học và đồ thị tính toán bên dưới!"

---

### 📌 SLIDE 8: TIÊU CHÍ 3 - KỸ NGHỆ DỮ LIỆU & CHUẨN HÓA CÔNG NGHIỆP
- **Thời lượng:** 1:00
- **Lời thoại:**
  > "Về dữ liệu nghiên cứu:  
  > - **Tập nguồn chính:** Nhóm sử dụng tập chuẩn SHWD / VOC2028 gồm 7,581 ảnh công trường thực tế. Nhóm đã tự kiểm tra và làm sạch dữ liệu, loại bỏ 3 nhãn dị biệt rác là nhãn 'dog' trong các file XML gốc; chia tập nghiêm ngặt 80% trainval / 20% test không rò rỉ dữ liệu (No Data Leakage).  
  > - **Bộ kiểm thử ngoại miền (>25,000 ảnh):** Để kiểm chứng năng lực tổng quát hóa trong thực tế, nhóm đã chuẩn hóa thêm 5 tập dữ liệu công trường bên ngoài: GDUT-HWD với 13,499 ảnh mật độ đông đúc; SHEL5K với 5,000 ảnh góc nhìn Flycam thẳng đứng; Hard Hat Workers với 7,000 ảnh ngoài trời; và các tập SHD, SFCHD.  
  > - Toàn bộ các tập này được nhóm chuẩn hóa về cùng một Không gian Nhãn Chung $\mathcal{C}^* = \{0: \text{'hat'}, 1: \text{'person'}\}$ và thiết lập Giao thức Harmonized PPE (Hat-Only) để triệt tiêu sự sai lệch nhãn giữa các tập."

---

### 📌 SLIDE 9: TỔNG THỂ KIẾN TRÚC MẠNG NƠ-RON REP-YOLO11s
- **Thời lượng:** 1:00
- **Lời thoại:**
  > "Trên **Hình 2** là Sơ đồ Kiến trúc Mạng Nơ-ron Rep-YOLO11s do nhóm thiết kế. Kiến trúc được chia thành 3 phân hệ liên hoàn:  
  > 1. **Backbone:** Bắt đầu bằng tầng Stem Conv nhận tensor 5 kênh từ CoordConv để phá vỡ tính bất biến tịnh tiến; tiếp theo là các khối CSPDarknet tích hợp các khối RepConv đa nhánh tại các tầng P2, P3, P4, P5 để trích xuất đặc trưng đa dạng.  
  > 2. **Neck:** Sử dụng mạng Path Aggregation Network (PAN) tích hợp cơ chế Chú ý Định tuyến Thưa BiFormer, giúp gom tụ 100% năng lực tính toán vào các vùng ứng viên chứa mũ bảo hộ siêu nhỏ.  
  > 3. **Head:** Cấu trúc Anchor-Free Decoupled Head tách biệt nhánh phân loại và nhánh định vị hộp bao, được huấn luyện tối ưu bằng hàm mất mát Focal EIoU Loss."

---

### 📌 SLIDE 10: ĐỘT PHÁ 1 - TÁI THAM SỐ HÓA CẤU TRÚC (REPCONV)
- **Thời lượng:** 1:30
- **Lời thoại:**
  > "Bây giờ em xin đi sâu vào 3 đột phá công nghệ của đề tài.  
  > **Đột phá số 1 là RepConv:** Giải quyết mâu thuẫn giữa việc học biểu diễn phong phú và yêu cầu suy luận siêu tốc.  
  > - **Lúc huấn luyện:** Mô hình chạy 3 nhánh song song: Conv 3x3, Conv 1x1, và nhánh Identity. Mỗi nhánh có một lớp Batch Normalization riêng. Cấu trúc đa nhánh này tạo ra dòng gradient đa dạng, giúp mô hình bắt dính viền cong của mũ bảo hộ và bề mặt giàn giáo.  
  > - **Lúc triển khai (`switch_to_deploy`):** Trước khi đưa ra camera thực tế, chúng em sử dụng đại số tuyến tính: Đầu tiên gộp Batch Normalization vào trọng số tích chập thông qua công thức $W' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W$. Sau đó zero-pad nhân 1x1 thành 3x3, biến identity thành ma trận Dirac 3x3, và cộng trực tiếp 3 ma trận lại thành duy nhất một nhân Conv 3x3 đơn!  
  > - **Kết quả thực nghiệm vật lý:** Trên GPU Tesla T4, độ trễ suy luận giảm ngoạn mục từ 7.12 ms xuống **2.92 ms** (giảm 59.0%), tốc độ đạt **342.5 FPS** mà sai số độ chính xác bằng 0.00%!"

---

### 📌 SLIDE 11: ĐỘT PHÁ 2 - COORDCONV & BIFORMER DYNAMIC ROUTING
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Đột phá số 2 kết hợp hai cơ chế không gian:  
  > - **Thứ nhất là CoordConv:** Nhóm bổ sung 2 kênh tọa độ không gian chuẩn hóa $C_x, C_y \in [-1, 1]$ vào tensor ảnh đầu vào ($640\times640\times5$).  
  >   *Tác dụng:* Phá vỡ tính bất biến tịnh tiến của CNN. Cung cấp tiên đề hình học giải phẫu: 'Mũ bảo hộ chỉ xuất hiện ở nửa trên của thân người; còn các vật thể màu vàng dưới mặt đất có tọa độ $C_y \to +1$ là xô vữa hoặc cọc tiêu'. Chi phí của kỹ thuật này chỉ là tăng thêm 2 kênh ở stem conv (0.078% tham số), độ trễ suy luận tăng thêm đúng 0 mili-giây!  
  > - **Thứ hai là Bi-Level Routing Attention (BiFormer):** Thay vì tính Self-Attention toàn cục với độ phức tạp bậc hai $\mathcal{O}(H^2W^2)$, BiFormer chia feature map thành các vùng thô $S \times S$, tính ma trận tương quan vùng và chỉ định tuyến đến Top-$k$ vùng có liên quan nhất. Độ phức tạp giảm xuống dạng tuyến tính $\mathcal{O}(S^2 + k \frac{HW}{S^2})$, loại bỏ hơn 85% nhiễu phông nền và dồn toàn bộ sự chú ý vào mũ bảo hộ."

---

### 📌 SLIDE 12: ĐỘT PHÁ 3 - HÀM MẤT MÁT FOCAL EIOU LOSS
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Đột phá số 3 nằm ở hàm mất mát định vị hộp bao Focal EIoU Loss.  
  > - **Hạn chế của CIoU truyền thống:** CIoU sử dụng tỷ lệ khung hình tương đối $v \propto (\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h})^2$. Khi mũ bảo hộ bị giàn giáo che khuất một phần, tỷ lệ $w/h$ bị bão hòa gradient dù kích thước tuyệt đối sai lệch lớn.  
  > - **Cải tiến của EIoU:** Nhóm phân rã trực tiếp sai số khoảng cách tâm, chiều rộng $w$ và chiều cao $h$ thành 3 thành phần độc lập, tối ưu hóa chính xác kích thước thực tế của hộp.  
  > - **Kết hợp trọng số tiêu điểm Focal:** Nhân thêm hệ số $\text{IoU}^{0.5}$ để triệt tiêu gradient từ các mẫu nền dễ học, dồn toàn bộ trọng số cập nhật cho các vi vật thể mũ bảo hộ bị che khuất.  
  > Kết quả thực nghiệm cho thấy Focal EIoU giúp mAP50 tăng thêm 0.14% và mAP50-95 đạt 62.54% mà không tốn thêm bất kỳ chi phí tính toán nào lúc chạy thực tế."

---

### 📌 SLIDE 13: TIÊU CHÍ 4 - SO SÁNH ĐỊNH LƯỢNG VỚI SOTA (TABLE I)
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Đi vào Tiêu chí 4: Giá trị của đề tài, **Bảng I** và **Hình 5** đối sánh Rep-YOLO11s với 5 dòng mô hình SOTA trên cùng tập dữ liệu chuẩn SHWD:  
  > - So với baseline YOLO11s gốc: Rep-YOLO11s tăng mAP50 từ 94.74% lên 94.83% trên Single Split, và độ trễ suy luận giảm từ 6.52 ms xuống **2.92 ms** (nhanh hơn gấp 2.2 lần!).  
  > - Đánh giá trên 5-Fold Cross-Validation: Mô hình đạt mAP50 trung bình **96.64 ± 0.32%**, trong đó đỉnh Fold 3 đạt **97.11%** với chỉ số F1 đạt 0.9396.  
  > - So với các công trình cải tiến quốc tế như EC-YOLOv8 (5.80 ms) hay YOLO-CBF (12.40 ms): Rep-YOLO11s có tốc độ nhanh hơn từ 2 đến 4 lần.  
  > Biểu đồ Đường biên Hiệu quả Pareto trên Hình 5 chứng minh: Rep-YOLO11s chiếm lĩnh vị trí tối ưu tuyệt đối ở góc trên-trái, đạt tỷ lệ độ chính xác trên mỗi mili-giây cao nhất hiện nay."

---

### 📌 SLIDE 14: TIÊU CHÍ 4 - NGHIÊN CỨU CẮT BỎ THÀNH PHẦN A0 -> A6 (TABLE II)
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Để chứng minh khoa học rằng mỗi mô-đun đề xuất đều đóng góp giá trị thực sự chứ không phải ngẫu nhiên, nhóm đã thực hiện chuỗi thực nghiệm Ablation Study có kiểm soát từ $A_0$ đến $A_6$ (Bảng II và Hình 6):  
  > - $A_0$ là Baseline YOLO11s (94.74% mAP50, 6.52 ms).  
  > - Nhóm thử nghiệm thêm nhánh vi vật thể $P_2$ ($A_1$): mAP tăng nhẹ lên 94.81% nhưng độ trễ tăng vọt lên 8.94 ms (+37%). Nhóm **kiên quyết loại bỏ** $A_1$ để giữ chuẩn thời gian thực.  
  > - Tích hợp CoordConv ($A_2$): mAP tăng lên 94.78%, độ trễ không đổi (6.58 ms).  
  > - Tích hợp RepConv ($A_3$) và Focal EIoU ($A_4$): mAP tăng liên tục lên 94.88%, khả năng bám dính biên cải thiện rõ rệt.  
  > - Tích hợp BiFormer ($A_5$): Recall tăng vọt lên 91.15%.  
  > - Cuối cùng, cấu hình $A_6$ (Full Fusion): Sau khi kích hoạt `switch_to_deploy`, độ trễ sụp đổ ngoạn mục từ 7.12 ms về **2.92 ms** (-59.0%) mà mAP50 giữ vững ở mức 94.83%!"

---

### 📌 SLIDE 15: TIÊU CHÍ 4 - KIỂM CHỨNG THỊ GIÁC XAI GRAD-CAM
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Để chứng minh mạng nơ-ron học đúng bản chất của chiếc mũ bảo hộ chứ không phải học vẹt bối cảnh, nhóm đã triển khai công cụ Explainable AI với kỹ thuật Grad-CAM trên **Hình 3** qua 3 kịch bản công trường khắc nghiệt:  
  > - **Kịch bản 1 (Áo bảo hộ cam & Giàn giáo):** Baseline YOLO11s bị phân tán gradient đỏ rực vào áo phản quang và thanh giàn giáo gỗ. Trong khi đó, Rep-YOLO11s tập trung 100% điểm nhiệt vào đúng 4 chiếc mũ bảo hộ xanh với độ tin cậy 0.84.  
  > - **Kịch bản 2 (Vi vật thể bị ngược sáng mạnh):** Ánh sáng cửa sổ làm mất tương phản của đối tượng. Nhờ BiFormer định tuyến token, mạng vẫn khóa chặt chiếc mũ nhỏ bên cửa sổ với độ tin cậy 0.89.  
  > - **Kịch bản 3 (Biển báo tam giác màu vàng):** Baseline kích hoạt mạnh trên biển báo tam giác màu vàng và báo động giả. Nhờ tiên đề tọa độ CoordConv, mạng nhận biết đây là biển báo dưới thấp và loại bỏ hoàn toàn, chỉ nhận diện đúng mũ trên đầu công nhân!"

---

### 📌 SLIDE 16: TIÊU CHÍ 4 - TỔNG QUÁT HÓA ĐA MIỀN & GIẢI MÃ SỤP ĐỔ IOU
- **Thời lượng:** 1:30
- **Lời thoại:**
  > "Về năng lực khái quát hóa ngoại miền (Tiêu chí 4), nhóm đã kiểm thử Zero-shot trên hơn 25,000 ảnh từ 5 bộ dữ liệu độc lập (Bảng III):  
  > - Trên tập **Hard Hat Workers (7,000 ảnh)**: Mô hình đạt đỉnh **97.03% mAP50**, tương đương độ chính xác trên tập nguồn gốc!  
  > - Trên tập **GDUT-HWD (13,499 ảnh)**: Đạt 74.27% mAP50 với Precision lên tới **90.26%** trong bối cảnh công nhân cực kỳ đông đúc (15–30 người/ảnh).  
  > **Đặc biệt, nhóm đã giải mã thành công hiện tượng Sụp đổ IoU (IoU Collapse):**  
  > Khi đánh giá chung cả lớp mũ và người trên Hard Hat Workers, mAP tụt xuống 74.40%. Nhóm đã điều tra và phát hiện đây không phải lỗi mô hình, mà do xung đột định nghĩa nhãn: Tập SHWD định nghĩa 'person' là Toàn thân (Full-body), trong khi Hard Hat Workers lại định nghĩa 'person' là Riêng phần đầu (Head-only). Khi mô hình bắt đúng toàn thân, hộp đầu người bị lọt thỏm bên trong làm $\text{IoU} \approx 0.07-0.14 \ll 0.50$, khiến thuật toán chấm điểm phạt cả False Positive lẫn False Negative!  
  > Khi nhóm chuẩn hóa đánh giá theo giao thức **Harmonized PPE (Hat-Only)**, mAP lập tức phục hồi lên **97.03%**, chứng minh năng lực trích xuất đặc trưng mũ của Rep-YOLO11s là hoàn hảo!"

---

### 📌 SLIDE 17: TIÊU CHÍ 4 - TRIỂN KHAI PHẦN CỨNG BIÊN & PIPELINE RTSP
- **Thời lượng:** 1:15
- **Lời thoại:**
  > "Về giá trị thực tiễn triển khai (Tiêu chí 4), nhóm đã tiến hành benchmark vật lý trên 4 nền tảng phần cứng thực tế (Bảng IV và Hình 4):  
  > - Trên **NVIDIA Tesla T4**: Đạt 2.92 ms (**342.5 FPS**) với TensorRT 11.2 FP16.  
  > - Trên **RTX 3050 Laptop**: Đạt 5.35 ms (**187.1 FPS**) với TensorRT FP16, và 12.43 ms (80.4 FPS) với PyTorch gốc.  
  > - **Điểm sáng vượt bậc:** Trên chiếc laptop văn phòng cũ chạy card **NVIDIA GeForce MX230** chỉ có **2GB VRAM**, kiến trúc Pascal cũ và hoàn toàn KHÔNG có nhân Tensor Cores, mô hình vẫn đạt **36.0 ms (27.8 FPS)** với PyTorch Native FP32 — vượt qua ngưỡng thời gian thực chuẩn 24 FPS!  
  > - Đồng thời, nhóm đã xây dựng hoàn chỉnh **Pipeline RTSP đa luồng** (Hình 4): Bóc tách toàn bộ thời gian giải mã H.264 (3.5–5.0 ms), tiền xử lý CoordConv (1.2–2.0 ms), suy luận TensorRT (2.92–5.35 ms), hậu xử lý NMS (1.5–2.8 ms) và hiển thị UI (2.2–3.4 ms). Tổng độ trễ toàn trình chỉ từ 10.54 đến 15.38 ms, đảm bảo thông lượng camera thực tế đạt **65 đến 95 FPS** trên laptop RTX 3050!"

---

### 📌 SLIDE 18: LỘ TRÌNH TRIỂN KHAI TOÀN DIỆN & KẾT LUẬN
- **Thời lượng:** 1:00
- **Lời thoại:**
  > "Cuối cùng, nhìn lại toàn bộ quá trình:  
  > - **Giai đoạn 1 (Đã hoàn thành 100%):** Nhóm đã hoàn thành trọn vẹn phát biểu bài toán, cơ sở lý thuyết, kiến trúc mạng Rep-YOLO11s, tự lập trình 4 module toán học, thực nghiệm SOTA, Ablation A0–A6, XAI Grad-CAM, Cross-domain và kiểm chứng vật lý trên phần cứng thật.  
  > - **Giai đoạn 2 (Đang triển khai):** Nhóm tập trung đóng gói phần mềm Desktop GUI và Web Dashboard giám sát trực quan; hoàn thiện cơ chế tự động kích hoạt âm thanh cảnh báo và ghi log vi phạm; tối ưu hóa điều phối đa luồng camera RTSP.  
  > - **Giai đoạn 3 (Kế hoạch về đích):** Nhóm sẽ áp dụng kỹ thuật Chưng cất Tri thức vi mô (Knowledge Distillation) từ YOLO11x sang Rep-YOLO11s để cải thiện góc nhìn Drone thẳng đứng từ đỉnh đầu; hoàn thiện bản thảo Khóa luận Tốt nghiệp chính thức để bảo vệ trước Hội đồng.  
  > Nhóm xin khẳng định: Đề tài đã được thiết kế, lập trình và đo đạc thực tế đầy đủ, hoàn toàn đáp ứng và vượt trội các tiêu chí Xuất sắc của Hội đồng Đồ án Tốt nghiệp Kỹ sư AI Đại học FPT!  
  > Chúng em xin chân thành cảm ơn quý Thầy Cô và rất mong nhận được những câu hỏi góp ý quý báu."

---

# PHẦN 3: BỘ CÂU HỎI VẤN ĐÁP PHẢN BIỆN HỘI ĐỒNG (Q&A DEFENSE PLAYBOOK)

### ❓ CÂU 1 (TIÊU CHÍ 3): "Tại sao nhóm không viết lại toàn bộ từ đầu (from scratch) mà lại dùng Ultralytics? Đóng góp 'tự phát triển' ở đây thực chất là gì?"
- **Đáp viên:** Nguyễn Hàn Như
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, trong nghiên cứu AI ứng dụng và kỹ nghệ thị giác máy tính hiện đại, việc tự viết lại toàn bộ từ đầu các thành phần cơ bản (như cấu trúc mạng tích chập, bộ nạp dữ liệu dataloader, hàm tối ưu SGD) là không cần thiết và dễ dẫn đến lỗi kỹ thuật không đáng có.  
  > Kế thừa framework Ultralytics giúp nhóm tận dụng hạ tầng huấn luyện chuẩn mực. Đóng góp 'Tự phát triển' thực chất của nhóm chúng em thể hiện ở 4 điểm mấu chốt:  
  > 1. Chúng em tự nhận diện 4 tử huyệt kỹ thuật của bài toán công trường (vi vật thể <20px, mất cân bằng 1:12, nhiễu bất biến tịnh tiến và giới hạn phần cứng).  
  > 2. Chúng em tự thiết kế đồ thị tính toán và tự viết 360 dòng code PyTorch thuần trong `custom_ablation_modules.py` cho 4 module: CoordConv, RepConv, BiFormer và Focal EIoU.  
  > 3. Chúng em can thiệp sâu (hard-patch) vào mã nguồn lõi tính hàm mất mát của Ultralytics (`ultralytics/utils/loss.py`) để thay thế hàm CIoU mặc định bằng Focal EIoU Loss.  
  > 4. Chúng em tự xây dựng giải thuật gộp đại số `switch_to_deploy` và pipeline RTSP đa luồng đồng bộ CUDA Events.  
  > Đây chính là bản chất của Kỹ nghệ Tùy biến Kiến trúc và Tích hợp Thuật toán Chuyên sâu trong AI ứng dụng."

---

### ❓ CÂU 2 (TIÊU CHÍ 3): "Kỹ thuật Tái tham số hóa (RepConv) nhóm tự nghĩ ra hay lấy từ bài báo RepVGG? Có cải tiến gì không?"
- **Đáp viên:** Nguyễn Hàn Như / Nguyễn Văn Thành
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, nguyên lý toán học về việc gộp Batch Normalization và cộng ma trận nhân tích chập được đề xuất gốc trong bài báo RepVGG (Ding et al., CVPR 2021).  
  > Tuy nhiên, đóng góp của nhóm chúng em là:  
  > 1. RepVGG nguyên bản áp dụng cho bài toán Phân loại ảnh (Image Classification) trên mạng VGG đường dẫn đơn. Nhóm chúng em đã **cải tiến và tích hợp thành công RepConv vào kiến trúc CSPDarknet hiện đại của YOLO11s** cho bài toán Phát hiện Vật thể Vi mô (Tiny Object Detection).  
  > 2. Chúng em giải quyết bài toán tương thích kênh khi tích hợp RepConv vào các khối residual C3k2, đảm bảo luồng gradient huấn luyện đa nhánh phong phú nhưng khi suy luận thì suy biến hoàn toàn về nhân Conv 3x3 duy nhất, giúp giảm 59% độ trễ trên GPU thật mà không suy giảm 0.01% độ chính xác."

---

### ❓ CÂU 3 (TIÊU CHÍ 4): "Tại sao trong bảng Ablation A0 -> A6 mAP50 trên Single Test chỉ tăng từ 94.74% lên 94.83% (+0.09%)? Có đáng để thêm 4 module phức tạp như vậy không?"
- **Đáp viên:** Nguyễn Hàn Như
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, đây là một câu hỏi rất sâu sắc về mặt bản chất toán học của các chỉ số đánh giá. Nhóm xin giải trình 3 nguyên nhân cốt lõi:  
  > 1. **Mặt nạ mất cân bằng dữ liệu 1:12:** Trên tập SHWD, số lượng nhãn `person` là 111,514 trong khi `hat` chỉ có 9,044. Chỉ số mAP tổng hợp là trung bình cộng của 2 lớp. Bản thân lớp `person` đã đạt trên 97% nên bị bão hòa. Bước nhảy vọt thực sự nằm ở **Recall của lớp mũ bảo hộ (`Recall_hat`), tăng từ 86.51% lên 91.15% (+4.64%)** — điều này có ý nghĩa sinh tử vì công trường sợ nhất là bỏ sót công nhân không đội mũ!  
  > 2. **Độ chính xác định vị không gian (Precision):** Nhờ CoordConv và BiFormer, các trường hợp báo động giả trên xô vữa vàng và áo cam bị triệt tiêu, giúp Precision duy trì ở mức 94.94%.  
  > 3. **Giá trị lớn nhất là Giảm 59% Độ Trễ Suy Luận:** Thay vì tăng độ phức tạp như các công trình khác (EC-YOLOv8 tụt xuống 172 FPS, YOLO-CBF tụt xuống 80 FPS), cấu trúc của chúng em sau khi gộp nhánh đưa độ trễ từ 7.12 ms xuống **2.92 ms (342.5 FPS)**. Nghĩa là giữ vững độ chính xác SOTA ở tốc độ nhanh gấp đôi đối thủ!"

---

### ❓ CÂU 4 (TIÊU CHÍ 4): "Tại sao kết quả 5-Fold Cross-Validation đạt 96.64% trong khi Single Test chỉ đạt 94.83%? Có bị Data Leakage không?"
- **Đáp viên:** Nguyễn Hàn Như / Nguyễn Tuấn Dũng
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, nhóm khẳng định 100% **hoàn toàn KHÔNG có hiện tượng Data Leakage (rò rỉ dữ liệu)**:  
  > 1. **Phân chia độc lập:** Tập dữ liệu SHWD / VOC2028 gồm 7,581 ảnh. Trong thực nghiệm Single Split, tập Test cố định gồm 1,517 ảnh (20%) được cô lập hoàn toàn, mô hình chỉ huấn luyện trên 6,064 ảnh Trainval.  
  > 2. **Bản chất toán học của 5-Fold:** Trong 5-Fold Cross-Validation, toàn bộ 7,581 ảnh được phân chia ngẫu nhiên phân tầng (Stratified K-Fold) thành 5 phần bằng nhau. Ở mỗi fold, mô hình được train trên 80% (6,064 ảnh) và validate trên 20% (1,517 ảnh) hoàn toàn tách biệt.  
  > 3. Do tập test cố định ban đầu của SHWD chứa tỷ lệ ảnh khó và ảnh góc cao nhiều hơn mức trung bình của toàn bộ tập dữ liệu, nên khi lấy kỳ vọng trung bình trên cả 5 phân vùng đại diện của toàn tập dữ liệu, giá trị kỳ vọng thực tế đạt mức $96.64 \pm 0.32\%$. Cả 5 fold đều cho kết quả rất ổn định (Fold 1: 96.48%, Fold 2: 96.72%, Fold 3: 97.11%, Fold 4: 96.35%, Fold 5: 96.54%), chứng minh mô hình không bị phụ thuộc vào một phân chia may rủi."

---

### ❓ CÂU 5 (TIÊU CHÍ 4): "Nhóm khẳng định mô hình chạy được trên NVIDIA GeForce MX230 2GB VRAM đạt 27.8 FPS, điều này có thật không? Đo đạc như thế nào?"
- **Đáp viên:** Nguyễn Hàn Như
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, con số 27.8 FPS trên GeForce MX230 là kết quả đo đạc vật lý thật 100% trên máy tính cá nhân của thành viên nhóm em:  
  > 1. Card NVIDIA GeForce MX230 là dòng GPU laptop giá rẻ phát hành năm 2019, kiến trúc Pascal, chỉ có 2GB VRAM GDDR5 và hoàn toàn KHÔNG có nhân Tensor Cores.  
  > 2. Mô hình Rep-YOLO11s sau khi gọi hàm `switch_to_deploy()` đã loại bỏ hoàn toàn các nhánh 1x1 và Identity phụ, đưa toàn bộ mạng về dạng tuần tự (sequential) Conv 3x3 duy nhất.  
  > 3. Số lượng tham số chỉ là 9.85 triệu, kích thước file trọng số PyTorch chỉ khoảng 20MB. Khi nạp vào VRAM với batch size = 1 và độ phân giải 640x640, bộ nhớ VRAM chiếm dụng chỉ khoảng 650MB (rất an toàn trong giới hạn 2GB của card).  
  > 4. Nhóm đo bằng `torch.cuda.Event(enable_timing=True)` có `torch.cuda.synchronize()`, chạy warm-up 50 frames và lấy trung bình trên 300 frames suy luận liên tục: thời gian suy luận là **36.00 ms**, tương đương **27.78 FPS**, vượt qua mốc chuẩn thời gian thực 24 FPS của video."

---

### ❓ CÂU 6 (TIÊU CHÍ 4): "Hiện tượng Sụp đổ IoU (IoU Collapse) trên tập Hard Hat Workers là gì? Tại sao đánh giá lớp mũ lại phục hồi lên 97.03%?"
- **Đáp viên:** Nguyễn Hàn Như
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, đây là một phát hiện học thuật rất thú vị mà nhóm đã giải mã:  
  > - Tập dữ liệu nguồn SHWD định nghĩa nhãn `person` là **Toàn thân công nhân (Full-body)**. Do đó mô hình Rep-YOLO11s được học để vẽ hộp bao quanh cả cơ thể người.  
  > - Ngược lại, tập dữ liệu Hard Hat Workers của AndrewMVD lại gán nhãn `person` cho **Riêng phần đầu người (Head-only)**.  
  > - Khi chạy Zero-shot trên Hard Hat Workers, mô hình dự đoán đúng toàn thân công nhân. Tuy nhiên, hộp ground-truth của tác giả chỉ bao quanh đầu người và bị nằm lọt thỏm bên trong. Tỷ lệ diện tích hộp đầu trên diện tích hộp thân chỉ khoảng 7% đến 14%, dẫn đến chỉ số $\text{IoU} \approx 0.07-0.14 \ll 0.50$.  
  > - Theo chuẩn đánh giá COCO mAP, khi IoU < 0.50, thuật toán sẽ coi dự đoán là một False Positive (dương tính giả) và gán nhãn ground-truth là False Negative (âm tính giả), làm chỉ số mAP chung bị kéo tụt xuống 74.40%.  
  > - Tuy nhiên, đối với nhãn `hat` (Mũ bảo hộ), cả hai tập dữ liệu đều định nghĩa giống hệt nhau (chỉ bao quanh chiếc mũ). Khi nhóm áp dụng giao thức **Harmonized PPE (Hat-Only)** để đánh giá riêng lớp mũ, mAP50 lập tức đạt **97.03%**. Điều này chứng minh 100% năng lực biểu diễn đặc trưng mũ của mô hình không hề bị suy giảm khi chuyển giao sang môi trường mới!"

---

### ❓ CÂU 7 (TIÊU CHÍ 1): "Góc máy Drone trên tập SHEL5K kết quả chỉ đạt 41.15%, vậy hệ thống có khả thi trên Flycam không? Nhóm khắc phục thế nào?"
- **Đáp viên:** Nguyễn Hàn Như
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, nhóm xin thẳng thắn thừa nhận: Kết quả 41.15% trên SHEL5K phản ánh đúng bản chất vật lý của góc nhìn Drone thẳng đứng từ trên cao 90 độ (Nadir View):  
  > 1. Khi Flycam chụp thẳng đứng từ đỉnh đầu xuống, toàn bộ bối cảnh giải phẫu cơ thể người (vai, ngực, thân) bị triệt tiêu hoàn toàn. Đồng thời kích thước mũ ở độ cao này chỉ dưới 10x10 pixels.  
  > 2. Đề tài của nhóm trong Giai đoạn 1 xác định rõ phạm vi (Scope) là tập trung tối ưu cho **Camera giám sát CCTV/RTSP cố định tại công trường** (góc nhìn nghiêng từ 15 đến 45 độ, độ cao 15-30m), nơi mô hình đạt hiệu quả xuất sắc từ 74% đến 97% mAP.  
  > 3. Đối với góc máy Drone thẳng đứng, đây chính là **kế hoạch của Giai đoạn 3**: Nhóm sẽ áp dụng kỹ thuật **Chưng cất Tri thức Vi mô (Knowledge Distillation)** từ mô hình giáo viên cỡ lớn YOLO11x sang Rep-YOLO11s kết hợp tiền xử lý siêu phân giải cục bộ (Patch Cropping) để nâng cao độ chính xác trên góc máy này."

---

### ❓ CÂU 8 (TIÊU CHÍ 2): "Software Prototype của nhóm đã chạy được thật chưa? Luồng RTSP hoạt động ra sao?"
- **Đáp viên:** Nguyễn Hàn Như / Nguyễn Văn Thành
- **Chiến lược trả lời:**
  > "Thưa Thầy Cô, Software Prototype của nhóm **đã hoạt động hoàn chỉnh và sẵn sàng demo trực tiếp**:  
  > 1. Nhóm đã xây dựng mã nguồn `scripts/ultra_fast_rtsp_engine.py` sử dụng thư viện OpenCV VideoCapture kết hợp hàng đợi đa luồng bất đồng bộ (Multi-threading Queue).  
  > 2. Luồng đọc camera (Camera Ingestion Thread) chạy độc lập để giải mã gói tin H.264 qua phần cứng NVDEC; luồng suy luận (Inference Thread) gọi trực tiếp TensorRT FP16 Engine với CUDA Stream.  
  > 3. Cơ chế Drop-Oldest-Frame trong hàng đợi giúp pipeline không bao giờ bị nghẽn (zero frame lag), đảm bảo hiển thị bounding box và cảnh báo vi phạm với độ trễ toàn trình chỉ 10–15 ms, đạt thông lượng 65–95 FPS trên laptop RTX 3050.  
  > Nhóm chúng em đã chuẩn bị sẵn video test và máy tính demo, xin phép được trình chiếu cho Hội đồng ngay khi Thầy Cô yêu cầu!"

---

# PHẦN 4: LỜI KẾT VÀ THÔNG ĐIỆP BẢO VỆ

> "Kính thưa Hội đồng, Đồ án Tốt nghiệp Rep-YOLO11s của nhóm chúng em không phải là một bài tập mô phỏng trên lý thuyết, mà là một công trình nghiên cứu ứng dụng nghiêm túc:  
> - Đã có bài báo khoa học 9 trang đầy đủ nền tảng toán học.  
> - Đã tự phát triển 4 module PyTorch can thiệp sâu vào cấu trúc mạng.  
> - Đã giải mã các hiện tượng vật lý và toán học của công trường (CoordConv, IoU Collapse).  
> - Đã kiểm chứng vật lý trên phần cứng thật từ server T4 đến laptop giá rẻ MX230 2GB.  
> Nhóm cam kết sẽ tiếp tục hoàn thiện trọn vẹn sản phẩm phần mềm và quyển Khóa luận Tốt nghiệp trong các giai đoạn tiếp theo để bảo vệ đạt kết quả Xuất sắc nhất. Xin trân trọng cảm ơn quý Thầy Cô!"
