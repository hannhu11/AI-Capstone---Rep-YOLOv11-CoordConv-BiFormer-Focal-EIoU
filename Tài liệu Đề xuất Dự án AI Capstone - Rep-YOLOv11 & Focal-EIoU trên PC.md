# **TÀI LIỆU ĐỀ XUẤT VÀ TƯ LIỆU ĐẦU VÀO DỰ ÁN AI CAPSTONE**

**Tên đề tài:** Kiến trúc YOLO Tái Tham Số Hóa Cấu Trúc Kết Hợp Hàm Mất Mát Focal-EIoU Cho Phát Hiện Mũ Bảo Hộ Theo Thời Gian Thực Trên Máy Tính Cá Nhân (PC Execution)

## **I. TÊN ĐỀ TÀI VÀ MÔ TẢ DỰ ÁN (PROJECT DESCRIPTIONS)**

### **1\. Tên đề tài chính thức**

> * **Tiếng Việt:** Kiến trúc YOLO tái tham số hóa cấu trúc kết hợp hàm mất mát Focal-EIoU cho phát hiện mũ bảo hộ theo thời gian thực trên máy tính cá nhân.  
> * **Tiếng Anh:** Structural Re-parameterized YOLO Architecture with Focal-EIoU Loss for Real-time Safety Helmet Detection on Personal PC.

### **2\. Mô tả đề tài bằng Tiếng Việt**

**Bài toán:** Môi trường công trường xây dựng có đặc thù bị che khuất phức tạp bởi giàn giáo, máy móc, vật liệu xây dựng, kết hợp với độ phân giải không đồng đều từ các camera an ninh CCTV. Yếu tố này đặt ra thách thức lớn đối với việc nhận diện chính xác công nhân có đội mũ bảo hộ hay không theo thời gian thực với thông lượng cao.  
**Giải pháp:** Dự án đề xuất phát triển dựa trên kiến trúc YOLO hiện đại (YOLOv11 / SD-YOLO) tích hợp Kỹ thuật Tái tham số hóa Cấu trúc (Structural Re-parameterization \- RepConv/RepC3) và Hàm mất mát Focal Efficient IoU (Focal-EIoU Loss):

> * **Trong quá trình huấn luyện:** Mô hình sử dụng cấu trúc đa nhánh (multi-branch) nhằm trích xuất đa dạng biểu diễn độ dốc và viền cong chi tiết của mũ bảo hộ, giúp tách biệt đối tượng khỏi các cấu trúc giàn giáo hình học phức tạp xung quanh.  
> * **Trong quá trình suy luận (Inference):** Sử dụng đại số tuyến tính để hợp nhất (fuse) toàn bộ các nhánh song song về một lớp Convolution 3x3 đường dẫn đơn duy nhất (single-path). Cơ chế này loại bỏ hoàn toàn chi phí phụ trội về kiến trúc, giúp đạt tốc độ suy luận cực đại trên GPU máy tính cá nhân.  
> * **Hàm mất mát Focal-EIoU:** Tự động tái đánh trọng số để tập trung học các mẫu khó (bị che khuất nặng) và tối ưu hóa chính xác tỷ lệ khung chứa (bounding box) có khía cạnh cực đoan.

**Môi trường triển khai:** Hệ thống được thiết kế để thực thi trực tiếp trên GPU máy tính cá nhân (PC GPU như RTX 3060/4070/T4), giải mã mượt mà nhiều luồng video camera RTSP CCTV cùng lúc mà không bị giật lag hay mất khung hình.

### **3\. Mô tả đề tài bằng Tiếng Anh (English Description \- For Copying)**

Describe:  
Construction site environments are heavily occluded by scaffolding, machinery, and building materials, coupled with uneven security camera resolutions. These adverse factors pose significant challenges for reliable, high-throughput, real-time safety helmet detection on personal PC environments.

Context:  
This project proposes developing upon modern YOLO architectures (SD-YOLO or YOLOv11) integrated with a Structural Re-parameterization Technique (RepConv/RepC3). During training, multi-branch decoupling captures rich gradient representations and detailed curved contours of safety helmets, isolating them from adjacent complex geometric structures. At inference time, these branches are mathematically fused into a single-path 3x3 convolution layer, eliminating architectural overhead while maintaining peak feature extraction capability. Furthermore, the Focal Efficient Intersection over Union (Focal-EIoU) loss function is employed to re-weight hard, heavily occluded samples and optimize bounding box regression for targets with extreme aspect ratios.

Objectives:  
\- Integrate a Structural Re-parameterization Module into YOLOv11 / SD-YOLO to achieve multi-branch feature learning during training and zero-overhead single-path execution during inference.  
\- Implement the Focal-EIoU Loss function to dynamically balance easy and hard occluded samples and refine bounding box regression precision in crowded construction environments.  
\- Surpass the fundamental state-of-the-art baseline of 88.5% mAP established by Darknet53 on the SHWD dataset, targeting \>= 95.0% mAP@0.5.  
\- Achieve ultra-high throughput (\> 120 FPS) and ultra-low latency (\< 5 ms) on a Personal PC GPU environment.  
\- Ensure seamless multi-stream decoding and real-time processing of standard RTSP CCTV streams without frame drops.

## **II. CÁC Ý CHÍNH VÀ MỤC TIÊU ĐỊNH LƯỢNG (KEY POINTS & TARGETS)**

| Hạng mục | Mô tả chi tiết kỹ thuật | Mục tiêu định lượng   |
| :---- | :---- | :---- |
| **Đổi mới Kiến trúc** | Sử dụng Structural Re-parameterization (RepConv). Học đa nhánh lúc Train và gộp ma trận trọng số thành 1 nhánh Conv 3x3 đơn lúc Inference. | Zero-overhead lúc suy luận. |
| **Hàm Mất Mát Cải Tiến** | Focal-EIoU Loss: Tách biệt phạt độ rộng/chiều cao Bounding Box \+ Hệ số Focal cân bằng mẫu khó bị giàn giáo che khuất. | Giảm sai số Bounding Box regression. |
| **Độ Chính Xác (mAP)** | Đánh giá trên tập dữ liệu chuẩn SHWD (Safety Helmet Wearing Dataset). | **mAP@0.5 \>= 95.0%** (Vượt xa Baseline Darknet53 88.5%). |
| **Tốc Độ & Độ Trễ trên PC** | Thực thi trên GPU PC cá nhân kết hợp xuất mô hình TensorRT FP16 Engine. | **FPS \> 120**, **Latency \< 5 ms**. |
| **Triển Khai Luồng RTSP** | Nhận và giải mã đa luồng camera CCTV công trường qua giao thức RTSP. | **4 \- 8 luồng RTSP 1080p** đồng thời, không drop frame. |

## **III. QUY TRÌNH TRIỂN KHAI TOÀN TRÌNH (END-TO-END PIPELINE)**

Quy trình xử lý toàn trình của dự án được cấu trúc thành 5 giai đoạn liên hoàn:

> 1. **Bước 1: Data Pipeline & Pre-processing:**  
   * Thu thập dữ liệu SHWD \+ gộp dữ liệu công trường từ Roboflow.  
   * Áp dụng kỹ thuật tăng cường dữ liệu chuyên biệt (Albumentations): *Random Shadow* (mô phỏng bóng mây/giàn giáo), *Cutout / Random Erasing* (mô phỏng bị che khuất một phần), và *Mosaic / MixUp*.  
> 2. **Bước 2: Model Architecture & Training:**  
   * Tích hợp khối RepConv/RepC3 vào cấu trúc Backbone và Neck của YOLOv11 / SD-YOLO.  
   * Thay thế hàm mất mát mặc định trong Ultralytics bằng **Focal-EIoU Loss**.  
   * Huấn luyện mô hình trên GPU PC cá nhân cho đến khi các chỉ số hội tụ tối ưu.  
> 3. **Bước 3: Structural Re-parameterization Fusion (Switch-to-Deploy):**  
   * Thực hiện hàm đại số switch\_to\_deploy() để triệt tiêu các nhánh phụ (1x1, Identity).  
   * Cộng nhập ma trận trọng số (kernel fusion) và hằng số lệch (bias fusion) về thành 1 lớp Convolution 3x3 duy nhất.  
> 4. **Bước 4: TensorRT FP16 Optimization:**  
   * Xuất mô hình PyTorch (.pt) \-\> ONNX format.  
   * Biên dịch ONNX thành **TensorRT FP16 Engine (.engine)** tận dụng nhân CUDA trên GPU PC để tối ưu hóa tốc độ suy luận.  
> 5. **Bước 5: Multi-stream RTSP Deployment & Dashboard:**  
   * Xây dựng module Python (OpenCV Multi-threading / DeepStream) nhận đa luồng RTSP camera.  
   * Đưa khung hình qua TensorRT Engine và hiển thị kết quả cảnh báo trực quan trên Dashboard.

## **IV. TẬP DỮ LIỆU (DATASETS)**

| Tên Dataset | Quy mô & Đặc điểm nhãn | Vai trò trong Dự án   |
| :---- | :---- | :---- |
| **SHWD (Safety Helmet Wearing Dataset)** | 7,581 hình ảnh; 9,044 đối tượng đội mũ (hat) và 111,514 đối tượng đầu người/không đội mũ (head/person). | **Tập Benchmark chính.** Bắt buộc dùng làm Test Set chuẩn để so sánh đối chiếu mAP với Darknet53 Baseline (88.5%). |
| **Construction Site Safety (Roboflow Universe)** | Hàng nghìn ảnh công trường thực tế có góc quay camera từ trên cao xuống, giàn giáo chăng lưới phức tạp. | **Tập bổ sung mẫu khó.** Trộn 20-30% vào Train Set để huấn luyện hàm Focal-EIoU học các góc che khuất nặng. |
| **Hard Hat Workers Dataset (Roboflow)** | Hơn 7,000 ảnh với đa dạng màu sắc mũ bảo hộ (vàng, trắng, xanh, đỏ) và góc nghiêng cực đoan. | **Tập bổ sung tổng quát.** Tăng tính đa dạng mẫu cho mô hình không bị phụ thuộc vào môi trường cố định. |

## **V. CÁC BÀI BÁO KHOA HỌC TƯƠNG QUAN VÀ TƯƠNG ĐỒNG (SOTA PAPERS)**

Danh mục các bài báo khoa học chuẩn mực làm nền tảng lý thuyết và đối chiếu thực nghiệm cho dự án:

> 1. **Focal and Efficient IOU Loss for Accurate Bounding Box Regression (Neurocomputing, 2021\)**  
>    *Nội dung:* Công trình đề xuất Focal-EIoU Loss. Chứng minh việc kết hợp hệ số Focal với EIoU giúp tăng 1.5% \- 3.0% mAP trên các tập dữ liệu có đối tượng bị che khuất và có tỷ lệ khung hình cực đoan.  
> 2. **RepVGG: Making VGG-style ConvNets Great Again (IEEE/CVF CVPR, 2021\)**  
>    *Nội dung:* Bài báo gốc về Structural Re-parameterization. Đề xuất phương pháp đại số tuyến tính gộp mạng đa nhánh phức tạp lúc huấn luyện thành một nhánh đơn Conv 3x3 duy nhất lúc suy luận, giúp đạt tốc độ FPS tối đa.  
> 3. **YOLO-CBF: Optimized YOLOv7 Algorithm for Helmet Detection in Road Environments (MDPI Electronics, 2023\)**  
>    *Nội dung:* Tích hợp Focal-EIoU Loss vào họ mạng YOLO cho bài toán nhận diện mũ bảo hộ, đạt độ chính xác 95.6% mAP trong môi trường che khuất và mật độ đông đúc.  
> 4. **An Algorithm for Safety Helmet Detection Based on Improved YOLOv8 (EC-YOLOv8) (MDPI Applied Sciences, 2024\)**  
>    *Nội dung:* Ứng dụng EIoU Loss trên tập dữ liệu SHWD, cải thiện mAP từ mức baseline 88.5% lên 95.7% mAP.  
> 5. **YOLOv8n-FADS: A Study for Enhancing Miners' Helmet Detection Accuracy in Complex Underground Environments (IEEE / ResearchGate, 2024\)**  
>    *Nội dung:* Áp dụng kỹ thuật Tái tham số hóa cấu trúc (RepConv) vào YOLO cho bài toán phát hiện mũ bảo hộ công nhân trong môi trường mỏ ngầm hẹp và che khuất phức tạp.