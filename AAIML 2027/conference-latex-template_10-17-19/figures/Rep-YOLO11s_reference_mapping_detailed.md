# REP-YOLO11s — BẢN ĐỒ DẪN CHỨNG KHOA HỌC CHO TỪNG KHỐI KIẾN TRÚC

> **Mục đích của tài liệu**  
> Tài liệu này tổng hợp các dẫn chứng học thuật/nguồn kỹ thuật phù hợp nhất cho **từng khối** trong kiến trúc Rep-YOLO11s hiện tại của nhóm.  
> Mục tiêu là giúp nhóm:
> 1. biết khối nào cần dẫn paper gốc;
> 2. biết khối nào nên dẫn thêm paper cùng domain safety helmet/PPE;
> 3. biết câu nào có thể viết an toàn trong Methodology/Related Work;
> 4. tránh nhận nhầm một cơ chế đã tồn tại là đóng góp mới của nhóm;
> 5. có **link trực tiếp dưới từng reference IEEE** để kiểm tra nguồn.

---

# 1. Nguyên tắc dẫn chứng cho Rep-YOLO11s

Rep-YOLO11s nên được trình bày là **kiến trúc tích hợp do nhóm đề xuất**, không phải là việc nhóm phát minh từng module riêng lẻ.

Các thành phần như:

- CoordConv;
- FPN/PAN;
- BiFormer / Bi-Level Routing Attention;
- structural re-parameterization / RepConv;
- anchor-free decoupled head;
- Distribution Focal Loss;
- Task-Aligned Assigner;
- Focal-EIoU;
- NMS;

đều đã có nền tảng từ công trình trước.

Điểm nhóm có thể nhấn mạnh là:

> **Rep-YOLO11s đề xuất cách tích hợp và bố trí các cơ chế đã được chứng minh trong một kiến trúc YOLO11 chuyên biệt cho phát hiện mũ bảo hộ trong bối cảnh công trường, đồng thời hướng đến cân bằng độ chính xác, khả năng phát hiện mục tiêu nhỏ/che khuất và hiệu quả triển khai thời gian thực.**

Không nên viết:

> “We propose CoordConv.”

> “We introduce BiFormer.”

> “We propose structural re-parameterization.”

Nên viết:

> “We integrate CoordConv…”

> “We incorporate Bi-Level Routing Attention…”

> “We adopt structural re-parameterization…”

> “We design Rep-YOLO11s by integrating…”

---

# 2. Bảng ánh xạ nhanh: từng khối → reference nên dẫn

| Khối trong hình Rep-YOLO11s | Vai trò | Reference chính | Reference gần domain helmet | Mức độ bắt buộc |
|---|---|---|---|---|
| YOLO11 baseline | Nền tảng detector | Ultralytics YOLO11 | YOLO-DCRCF 2025 | Rất nên |
| CoordConv | Thêm tọa độ không gian | CoordConv, NeurIPS 2018 | YOLO-CBF 2025 | Bắt buộc |
| CBS | Conv + BN + SiLU | Ultralytics YOLO architecture | YOLO-CBF | Nên |
| C3k2 | Feature extraction của YOLO11 | Ultralytics YOLO11 | CSPNet chỉ là nền tảng CSP | Bắt buộc nếu mô tả C3k2 |
| SPPF | Mở rộng receptive field | Ultralytics YOLO + SPP-Net | — | Nên |
| C2PSA | Attention block của YOLO11 | Ultralytics YOLO11 | — | Bắt buộc nếu mô tả C2PSA |
| P3/P4/P5 | Multi-scale features | FPN, CVPR 2017 | Improved YOLOv8 helmet 2024 | Rất nên |
| PAN | Multi-scale fusion | PANet, CVPR 2018 | Improved YOLOv8 helmet 2024 | Bắt buộc |
| BiFormer / BRA | Sparse dynamic attention | BiFormer, CVPR 2023 | YOLO-CBF 2025 | Bắt buộc |
| RepConv | Train multi-branch, deploy single conv | RepVGG, CVPR 2021 | YOLOv7 / YOLO-CBF | Bắt buộc |
| Anchor-free head | Không dùng predefined anchors | YOLOX + Ultralytics | Improved YOLOv8 helmet | Nên |
| Decoupled head | Tách cls và reg | YOLOX | Ultralytics YOLO11 | Nên |
| DFL | Distribution-based box regression | Generalized Focal Loss, NeurIPS 2020 | Ultralytics YOLO11 | Bắt buộc |
| TAL | Gán mẫu theo alignment cls-reg | TOOD, ICCV 2021 | Ultralytics implementation | Bắt buộc |
| BCE | Classification loss | Loss chuẩn | Ultralytics implementation | Không cần paper riêng |
| Focal-EIoU | Box regression | Zhang et al., Neurocomputing 2022 | YOLO-CBF 2025 | Bắt buộc |
| NMS | Hậu xử lý | Soft-NMS paper giải thích NMS | — | Tùy mức mô tả |
| Total loss | BCE + Focal-EIoU + DFL | Dẫn từng loss thành phần | — | Bắt buộc dẫn component |

---

# 3. YOLO11 — nền tảng kiến trúc

## 3.1. Khối nào trong hình liên quan?

YOLO11 là nền tảng cho:

- backbone;
- C3k2;
- SPPF;
- C2PSA;
- feature maps P3/P4/P5;
- FPN/PAN-style neck;
- anchor-free decoupled Detect head;
- DFL regression.

## 3.2. Dẫn chứng chính

Đối với **chi tiết exact implementation của YOLO11**, nguồn đáng tin cậy nhất là **Ultralytics official documentation**, bởi vì YOLO11 không có một bài conference/journal chính thức riêng tương đương YOLOv7.

Tài liệu chính thức của Ultralytics mô tả rõ:

- YOLO11 dùng `C3k2`;
- có `SPPF`;
- chèn `C2PSA`;
- neck dùng FPN + PAN;
- detection head là anchor-free và decoupled;
- regression sử dụng DFL với `reg_max=16`.

## 3.3. Câu có thể dùng trong bài

> The proposed Rep-YOLO11s is developed from the Ultralytics YOLO11 architecture, which employs C3k2 feature extraction blocks, SPPF, C2PSA, multi-scale FPN/PAN fusion, and an anchor-free decoupled detection head.

Hoặc viết ngắn hơn:

> Rep-YOLO11s adopts YOLO11 as the baseline detector and modifies its feature extraction, multi-scale fusion, attention, and regression components for safety-helmet detection.

## 3.4. Reference theo IEEE style

**[R1]** G. Jocher and J. Qiu, “Ultralytics YOLO11,” Ultralytics, 2024. [Online]. Available: Ultralytics YOLO11 documentation.

**Link kiểm tra trực tiếp:**  
https://docs.ultralytics.com/models/yolo11/

**Link kiến trúc chi tiết:**  
https://docs.ultralytics.com/guides/yolo-architecture/

### Lưu ý

Không nên tự tạo một “YOLO11 paper” không tồn tại.  
Với C3k2/C2PSA/head exact implementation, hãy dẫn documentation chính thức.

---

# 4. CoordConv — Input + Spatial Coordinate Encoding

## 4.1. Khối trong hình

```text
Input image (RGB)
B × 3 × 640 × 640
        ↓
CoordConv
Concat [RGB, Cx, Cy]
        ↓
Spatial tensor
B × 5 × 640 × 640
```

## 4.2. Paper gốc quan trọng nhất

CoordConv được đề xuất trong:

**“An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution” — NeurIPS 2018.**

Ý tưởng cốt lõi:

- convolution thông thường không được cung cấp trực tiếp absolute coordinates;
- CoordConv bổ sung các coordinate channels;
- network có thể học mức độ translation dependence phù hợp;
- vẫn giữ được cách tính convolution quen thuộc.

## 4.3. Nó chứng minh khối nào trong hình?

Paper này chứng minh trực tiếp cho:

```text
RGB tensor
+
x-coordinate channel
+
y-coordinate channel
→ augmented spatial tensor
```

Đây chính là nền tảng khoa học mạnh nhất cho box `CoordConv` của bạn.

## 4.4. Tại sao phù hợp helmet detection?

Trong công trường có thể có:

- thùng nhựa vàng;
- biển cảnh báo;
- vật liệu xây dựng;
- đèn phản quang;
- vest vàng;

có màu/hình dáng gần giống helmet.

Việc có spatial coordinate information cho phép model học thêm ngữ cảnh vị trí thay vì chỉ nhìn appearance.

## 4.5. Reference cùng domain rất mạnh

YOLO-CBF 2025 cũng tích hợp CoordConv vào helmet detection.

Điều này rất giá trị vì:

- paper CoordConv gốc chứng minh **nguyên lý**;
- YOLO-CBF chứng minh **tiền lệ áp dụng vào helmet detection**.

## 4.6. Câu đề xuất cho Methodology

> Following Liu et al. [R2], the RGB input is augmented with normalized horizontal and vertical coordinate channels, enabling the network to explicitly encode spatial position instead of relying solely on translation-equivariant convolutional features.

Có thể bổ sung domain:

> The use of CoordConv in helmet detection is also supported by YOLO-CBF [R3], which incorporates coordinate information to enhance spatial perception under complex road scenes.

## 4.7. Reference IEEE

**[R2]** R. Liu, J. Lehman, P. Molino, F. P. Such, E. Frank, A. Sergeev, and J. Yosinski, “An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018.

**Link kiểm tra trực tiếp:**  
https://papers.nips.cc/paper/8169-an-intriguing-failing-of-convolutional-neural-networks-and-the-coordconv-solution

**PDF:**  
https://papers.nips.cc/paper/8169-an-intriguing-failing-of-convolutional-neural-networks-and-the-coordconv-solution.pdf

---

# 5. YOLO-CBF — reference domain hỗ trợ CoordConv + BiFormer + Focal-EIoU

Đây là một trong những reference **liên quan nhất với Rep-YOLO11s về mặt domain**, vì paper này cùng lúc sử dụng:

- CoordConv;
- BiFormer dynamic sparse attention;
- Focal-EIoU;
- YOLO-based helmet detection.

Nó không chứng minh toàn bộ Rep-YOLO11s đã tồn tại, nhưng là dẫn chứng rất mạnh rằng ba hướng cải tiến trên có cơ sở trong helmet detection.

## Reference IEEE

**[R3]** Z. Wu, J. Qin, X. Xiang, and Y. Tan, “YOLO-CBF: Optimized YOLOv7 Algorithm for Helmet Detection in Road Environments,” *Electronics*, vol. 14, no. 7, Art. no. 1413, 2025.

**Link bài báo:**  
https://www.mdpi.com/2079-9292/14/7/1413

**DOI:**  
https://doi.org/10.3390/electronics14071413

**PDF:**  
https://www.mdpi.com/2079-9292/14/7/1413/pdf

---

# 6. CBS — Conv + Batch Normalization + SiLU

## 6.1. Khối trong hình

```text
Stem CBS: 3×3 / s2
Conv + BN + SiLU
```

## 6.2. Dẫn chứng

Ultralytics mô tả `Conv` block cơ sở là chuỗi:

```text
Conv2d → BatchNorm → SiLU
```

Do đó `CBS` trong hình của bạn có cơ sở trực tiếp từ implementation family của Ultralytics YOLO.

## 6.3. Cách viết

> The stem follows the standard Ultralytics convolutional unit consisting of convolution, batch normalization, and SiLU activation [R1].

Không cần dẫn paper riêng cho BN hoặc SiLU nếu chúng không phải contribution.

---

# 7. C3k2 — Backbone feature extraction

## 7.1. Khối trong hình

```text
P2 stage: C3k2
P3 stage: C3k2
P4 stage: C3k2
```

## 7.2. Nguồn exact implementation

Ultralytics documentation mô tả:

- C3k2 là subclass/biến thể phát triển từ C2f;
- được dùng trong YOLO11;
- có thể sử dụng Bottleneck/C3k tùy cấu hình.

Do đó, **nguồn exact cho C3k2 là Ultralytics**, không phải CSPNet.

## 7.3. CSPNet dùng để làm gì?

CSPNet là **conceptual lineage**:

- chia và hợp nhất feature flows;
- giảm duplicate gradient information;
- tăng hiệu quả tính toán.

Nhưng không được viết:

> “CSPNet proposed C3k2.”

Câu này sai.

Nên viết:

> C3k2 belongs to the CSP-derived design lineage employed by modern Ultralytics YOLO architectures.

## 7.4. Reference CSPNet

**[R4]** C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H. Yeh, “CSPNet: A New Backbone That Can Enhance Learning Capability of CNN,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 390–391.

**Link bài báo:**  
https://openaccess.thecvf.com/content_CVPRW_2020/html/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.html

**PDF:**  
https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

---

# 8. SPPF — Spatial Pyramid Pooling Fast

## 8.1. Khối trong hình

```text
P5: SPPF + C2PSA
```

## 8.2. Hai mức reference cần phân biệt

### Mức 1 — Exact SPPF implementation

Ultralytics architecture documentation:

- SPPF sử dụng pooling lặp;
- nhằm mở rộng receptive field;
- là biến thể nhanh của SPP trong YOLO family.

Dẫn [R1].

### Mức 2 — Nguồn khái niệm spatial pyramid pooling

Paper SPP-Net của He et al.

Paper này là nền tảng học thuật cho tư tưởng spatial pyramid pooling.

## 8.3. Câu viết an toàn

> The deepest backbone stage employs SPPF to efficiently enlarge the effective receptive field, following the fast spatial-pyramid-pooling implementation used in Ultralytics YOLO architectures [R1], with its conceptual basis tracing back to spatial pyramid pooling [R5].

## 8.4. Reference IEEE

**[R5]** K. He, X. Zhang, S. Ren, and J. Sun, “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 37, no. 9, pp. 1904–1916, 2015.

**Link kiểm tra:**  
https://pubmed.ncbi.nlm.nih.gov/26353135/

**DOI:**  
https://doi.org/10.1109/TPAMI.2015.2389824

---

# 9. C2PSA

## 9.1. Khối trong hình

```text
SPPF + C2PSA
```

## 9.2. Nguồn nên dẫn

Exact `C2PSA` của YOLO11 được mô tả bởi Ultralytics.

Tài liệu architecture giải thích rằng:

- C2PSA được thêm sau SPPF;
- một nhánh feature đi qua PSABlock;
- sau đó concatenate/fuse.

Vì đây là implementation-specific module, không nên gán nó cho một paper không phải nguồn của C2PSA.

## 9.3. Câu viết

> The C2PSA block is retained from the YOLO11 backbone to enhance feature interaction at the deepest semantic stage [R1].

---

# 10. P3 / P4 / P5 và Feature Pyramid Network

## 10.1. Khối trong hình

```text
P3: stride 8
P4: stride 16
P5: stride 32
```

## 10.2. Paper nền tảng

**Feature Pyramid Networks for Object Detection — CVPR 2017.**

Paper FPN chứng minh ý tưởng:

- deep CNN có hierarchy tự nhiên;
- feature map các mức có resolution/semantic khác nhau;
- top-down pathway + lateral connections tạo feature pyramid mạnh cho multi-scale detection.

## 10.3. Vì sao rất quan trọng cho helmet?

Helmet thường:

- nhỏ;
- xa camera;
- bị giảm pixel;
- xuất hiện ở nhiều scale.

Do đó P3/P4/P5 là cơ sở để model xử lý scale variation.

## 10.4. Câu đề xuất

> Multi-scale feature maps P3, P4, and P5 are preserved to support objects at different spatial scales, following the feature-pyramid principle introduced by Lin et al. [R6].

## 10.5. Reference IEEE

**[R6]** T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2117–2125.

**Link bài báo:**  
https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

**PDF:**  
https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf

---

# 11. PAN / Multi-scale PAN Fusion

## 11.1. Khối trong hình

```text
P3
P4 → Multi-scale PAN fusion → N3/N4/N5
P5
```

## 11.2. Paper gốc

**Path Aggregation Network for Instance Segmentation — CVPR 2018.**

PANet đề xuất bottom-up path augmentation nhằm:

- tăng information flow;
- rút ngắn đường đi giữa low-level localization features và high-level semantic features;
- hỗ trợ feature aggregation hiệu quả hơn.

## 11.3. Cách liên hệ FPN + PAN

FPN:
```text
top-down semantic propagation
```

PAN:
```text
bottom-up path augmentation
```

Do đó neck của YOLO thường được mô tả theo FPN/PAN paradigm.

## 11.4. Câu đề xuất

> The neck follows an FPN/PAN-style feature aggregation strategy, combining the top-down multi-scale semantic hierarchy of FPN [R6] with the bottom-up path augmentation principle of PANet [R7].

## 11.5. Reference IEEE

**[R7]** S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, “Path Aggregation Network for Instance Segmentation,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 8759–8768.

**Link bài báo:**  
https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Path_Aggregation_Network_CVPR_2018_paper.html

**PDF:**  
https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Liu_Path_Aggregation_Network_CVPR_2018_paper.pdf

---

# 12. BiFormer / Bi-Level Routing Attention (BRA)

## 12.1. Khối trong hình

```text
Partition feature map
        ↓
Q, K, V
        ↓
Region-level Q^r, K^r
        ↓
A^r = Q^r(K^r)^T
        ↓
I^r = TopK(A^r, k)
        ↓
Gather routed K,V
        ↓
Fine token attention
```

## 12.2. Paper gốc bắt buộc dẫn

**BiFormer: Vision Transformer With Bi-Level Routing Attention — CVPR 2023.**

Paper này đề xuất:

- dynamic sparse attention;
- coarse region-level routing;
- loại key-value regions không liên quan;
- fine-grained token-to-token attention chỉ trên routed regions.

Đây là nguồn **trực tiếp nhất** cho toàn bộ flow BiFormer inset của bạn.

## 12.3. Ý nghĩa của từng box

### `Partition feature map`

Feature map được chia thành regions/windows.

### `Form Q,K,V`

Sinh query/key/value representations.

### `Region-level Q^r,K^r`

Tạo representations ở mức region.

### `A^r = Q^r(K^r)^T`

Tính region-to-region affinity.

### `I^r = TopK(A^r,k)`

Chọn các region liên quan nhất.

### `Gather routed K,V`

Gather K/V từ selected regions.

### `Fine token attention`

Chỉ tính attention trên routed candidate regions thay vì dense global pairs.

## 12.4. Câu cho Methodology

> Following Zhu et al. [R8], the feature map is partitioned into coarse regions, and region-level query-key affinity is computed to select the top-k relevant regions. Fine-grained token attention is subsequently evaluated only over the routed key-value pairs.

## 12.5. Caption nên dùng cho hình BiFormer

> **Simplified schematic of the Bi-Level Routing Attention routing mechanism used in Rep-YOLO11s, adapted from Zhu et al. [R8].**

Từ `adapted from` phù hợp hơn `reproduced from`, vì:
- layout hình là nhóm tự vẽ;
- bạn đang giản lược cơ chế để giải thích kiến trúc.

## 12.6. Reference IEEE

**[R8]** L. Zhu, X. Wang, Z. Ke, W. Zhang, and R. W. H. Lau, “BiFormer: Vision Transformer With Bi-Level Routing Attention,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 10323–10333.

**Link bài báo:**  
https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html

**PDF:**  
https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.pdf

---

# 13. RepConv / Structural Re-parameterization

## 13.1. Khối trong hình

Training topology:

```text
          Conv 3×3 + BN
         /
X ------ Conv 1×1 + BN ----> Σ → SiLU
         \
          Identity + BN
```

Deployment topology:

```text
X → Fused Conv 3×3 + bias → SiLU
```

## 13.2. Paper gốc quan trọng nhất

**RepVGG: Making VGG-Style ConvNets Great Again — CVPR 2021.**

Paper RepVGG đề xuất **structural re-parameterization**:

Training:
- multi-branch topology;
- 3×3 branch;
- 1×1 branch;
- identity branch khi phù hợp.

Inference:
- fuse các linear branches;
- quy về equivalent 3×3 convolution;
- chạy single-path inference.

## 13.3. Đây có phải cơ chế riêng của nhóm không?

Không.

**Hình là nhóm tự vẽ**, nhưng:
- nguyên lý structural re-parameterization không phải của nhóm;
- phải dẫn RepVGG.

Điểm riêng của nhóm nằm ở:
- cách đưa RepConv vào Rep-YOLO11s;
- vị trí module;
- cách kết hợp với PAN/BiFormer;
- lựa chọn activation phù hợp YOLO.

## 13.4. Câu Methodology

> RepConv follows the structural re-parameterization principle introduced by RepVGG [R9]. During training, parallel 3×3, 1×1, and identity branches provide multiple optimization paths. At deployment, the linear branches are algebraically fused into an equivalent 3×3 convolution kernel and bias.

## 13.5. Caption hình RepConv

> **Structural re-parameterization of the RepConv block used in Rep-YOLO11s, based on the RepVGG principle [R9].**

Không nên dùng:
> “our novel re-parameterization mechanism”

nếu toán học vẫn là structural re-parameterization chuẩn.

## 13.6. Reference IEEE

**[R9]** X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun, “RepVGG: Making VGG-Style ConvNets Great Again,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 13733–13742.

**Link bài báo:**  
https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html

**PDF:**  
https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf

---

# 14. YOLOv7 — supporting reference cho re-parameterization trong detector

YOLOv7 rất hữu ích để chứng minh structural re-parameterization không chỉ áp dụng cho classification backbone mà còn liên quan đến real-time object detection.

## Reference IEEE

**[R10]** C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 7464–7475.

**Link bài báo:**  
https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html

**PDF:**  
https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.pdf

---

# 15. Anchor-free decoupled detection head

## 15.1. Khối trong hình

```text
N3/P3
├── cv3 classification
└── cv2 regression

N4/P4
├── cv3 classification
└── cv2 regression

N5/P5
├── cv3 classification
└── cv2 regression
```

## 15.2. Reference học thuật rất phù hợp

**YOLOX: Exceeding YOLO Series in 2021.**

YOLOX nổi tiếng với:
- chuyển YOLO sang anchor-free;
- decoupled head;
- tách classification và regression.

## 15.3. Lưu ý

Exact YOLO11 head phải dẫn [R1].

YOLOX dùng để dẫn **conceptual precedent** cho anchor-free + decoupled design.

## 15.4. Câu viết

> The detector adopts an anchor-free decoupled head with separate classification and regression branches, consistent with the modern YOLO detection paradigm and the decoupled-head formulation popularized by YOLOX [R11].

## 15.5. Reference IEEE

**[R11]** Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, “YOLOX: Exceeding YOLO Series in 2021,” arXiv:2107.08430, 2021.

**Link:**  
https://arxiv.org/abs/2107.08430

**PDF:**  
https://arxiv.org/pdf/2107.08430

---

# 16. Box distributions + Distribution Focal Loss (DFL)

## 16.1. Khối trong hình

```text
cv2 regression
→ Box distributions
→ Box decoding
→ Decoded boxes
```

Training-only:

```text
P3/P4/P5 pre-decoding distributions
→ DFL
```

## 16.2. Paper bắt buộc dẫn

**Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection — NeurIPS 2020.**

Đây là reference chính cho Distribution Focal Loss.

## 16.3. Ý nghĩa

Thay vì chỉ xem bounding-box localization như một scalar regression đơn giản, DFL sử dụng **distribution representation** cho localization.

Điều này giải thích vì sao hình của bạn có:

- `Box distributions`;
- sau đó `Box decoding`;
- DFL nhận input từ distributions trước decoding.

## 16.4. Câu Methodology

> Bounding-box offsets are represented as discrete distributions and optimized using Distribution Focal Loss, following the distributed localization formulation introduced by Li et al. [R12].

## 16.5. Reference IEEE

**[R12]** X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020.

**Link bài báo:**  
https://papers.nips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html

**PDF:**  
https://papers.nips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf

---

# 17. Task-Aligned Assigner (TAL)

## 17.1. Khối trong hình

```text
Predictions
        \
         → Task-aligned assigner → assigned targets
        /
Ground truth
```

## 17.2. Paper gốc

**TOOD: Task-Aligned One-Stage Object Detection — ICCV 2021.**

TOOD giải quyết vấn đề:
- classification task;
- localization task;

có thể không spatially aligned trong one-stage detector.

Task Alignment Learning được đưa ra để:
- align hai nhiệm vụ;
- thiết kế sample assignment;
- dùng classification + localization quality trong quá trình training.

## 17.3. Vì sao TAL phải nằm training-only?

TAL dùng để:
- assign positive targets;
- tạo supervision;

không phải module chạy sau head lúc inference.

Hình hiện tại của bạn đặt TAL trong `Training-only supervision` là hợp lý.

## 17.4. Câu Methodology

> Positive training samples are assigned using a task-aligned strategy derived from TOOD [R13], which jointly considers classification confidence and localization quality to mitigate the misalignment between the two detection tasks.

## 17.5. Reference IEEE

**[R13]** C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, “TOOD: Task-Aligned One-Stage Object Detection,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 3510–3519.

**Link bài báo:**  
https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html

**PDF:**  
https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf

---

# 18. BCE Classification Loss

## 18.1. Khối

```text
Classification: BCE
L_BCE
```

## 18.2. Có cần reference riêng không?

Thông thường **không cần**.

BCE là loss nền tảng.

Nếu paper cần dẫn implementation:
- cite Ultralytics source/config;
- không cần thêm một paper chỉ để chứng minh công thức BCE.

## 18.3. Câu viết

> The classification branch is optimized using binary cross-entropy loss.

---

# 19. Focal-EIoU

## 19.1. Khối trong hình

```text
Box regression: Focal-EIoU
L_Focal-EIoU
```

## 19.2. Paper gốc bắt buộc

**Focal and Efficient IoU Loss for Accurate Bounding Box Regression — Neurocomputing 2022.**

Paper đề xuất:

### EIoU
Phân rã sai số box theo:
- overlap;
- center point distance;
- width difference;
- height difference.

### Focal-EIoU
Thêm focal weighting để điều chỉnh đóng góp của regression examples.

## 19.3. Metadata chính xác

Version journal chính thức:

- *Neurocomputing*
- vol. 506
- pp. 146–157
- 2022
- DOI: 10.1016/j.neucom.2022.07.042

## 19.4. Câu Methodology

> For bounding-box regression, Focal-EIoU [R14] is employed to explicitly model overlap, center-distance, and side-length discrepancies while introducing focal weighting into the regression objective.

## 19.5. Reference IEEE

**[R14]** Y.-F. Zhang, W. Ren, Z. Zhang, Z. Jia, L. Wang, and T. Tan, “Focal and Efficient IoU Loss for Accurate Bounding Box Regression,” *Neurocomputing*, vol. 506, pp. 146–157, 2022.

**Link bài báo:**  
https://www.sciencedirect.com/science/article/pii/S0925231222009018

**DOI:**  
https://doi.org/10.1016/j.neucom.2022.07.042

---

# 20. NMS — Post-processing

## 20.1. Khối trong hình

```text
Collect P3/P4/P5 predictions
        ↓
NMS
        ↓
Final detections
```

## 20.2. Reference

Soft-NMS paper có phần giải thích conventional NMS rất rõ.

Nó mô tả:
- sort detection boxes theo score;
- chọn box score cao nhất;
- suppress các overlapping boxes;
- lặp lại trên phần còn lại.

## 20.3. Cách dùng

Nếu NMS chỉ xuất hiện một box trong figure, không bắt buộc phải thêm reference riêng.

Nếu bạn viết một paragraph về NMS trong deployment/post-processing thì có thể dẫn.

## 20.4. Reference IEEE

**[R15]** N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS—Improving Object Detection With One Line of Code,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 5561–5569.

**Link bài báo:**  
https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html

**PDF:**  
https://openaccess.thecvf.com/content_iccv_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf

---

# 21. Total Training Objective

## 21.1. Công thức của Rep-YOLO11s

```text
L_total
=
λ_cls L_BCE
+
λ_box L_Focal-EIoU
+
λ_dfl L_DFL
```

## 21.2. Đây có phải cần một paper giống hệt không?

Không.

Bạn dẫn từng thành phần:

- BCE → standard;
- Focal-EIoU → [R14];
- DFL → [R12];
- TAL → [R13].

**Cách kết hợp chúng trong Rep-YOLO11s** có thể là formulation của mô hình bạn.

## 21.3. Câu Methodology

> The final multi-task objective combines BCE classification loss, Focal-EIoU regression loss [R14], and Distribution Focal Loss [R12], while the training targets are generated using task-aligned assignment [R13].

---

# 22. YOLO-DCRCF — reference rất gần bài của nhóm

## 22.1. Vì sao quan trọng?

YOLO-DCRCF:

- dựa trên YOLO11;
- safety helmet + glove;
- power grid operation;
- xử lý occlusion;
- multi-scale;
- feature pyramid;
- real-time safety monitoring.

Do đó đây là reference tốt để chứng minh:
- YOLO11 phù hợp PPE;
- domain có small/occluded targets;
- cần cải tiến multi-scale feature processing.

## 22.2. Cách dùng trong Related Work

> Zhao et al. proposed YOLO-DCRCF based on YOLO11 for helmet and glove detection in power-grid operation environments, integrating deformable convolution and recalibrated feature-pyramid processing to improve robustness under occlusion and scale variation [R16].

## 22.3. Reference IEEE

**[R16]** J. Zhao, Z. Yang, B. Li, and Y. Zhao, “YOLO-DCRCF: An Algorithm for Detecting the Wearing of Safety Helmets and Gloves in Power Grid Operation Environments,” *Journal of Imaging*, vol. 11, no. 9, Art. no. 320, 2025.

**Link bài báo:**  
https://www.mdpi.com/2313-433X/11/9/320

**DOI:**  
https://doi.org/10.3390/jimaging11090320

---

# 23. An Improved YOLOv8 Safety Helmet Wearing Detection Network

## 23.1. Vì sao liên quan?

Paper tập trung vào:
- safety helmet;
- complex industrial environment;
- small targets;
- distance variation;
- multi-scale processing;
- missed detection / false detection.

Paper mô tả YOLOv8 neck sử dụng:
- Path Aggregation Network;
- Feature Pyramid Network;
- multi-scale fusion;
- decoupled detection head.

Do đó đây là supporting reference tốt cho:
- PAN/FPN trong helmet detection;
- small-object problem;
- decoupled head trong helmet context.

## 23.2. Câu Related Work

> Song et al. improved YOLOv8 for safety-helmet detection under complex industrial backgrounds by strengthening multi-scale processing and small-object feature extraction, demonstrating the importance of multi-scale feature fusion for helmet targets with substantial distance variation [R17].

## 23.3. Reference IEEE

**[R17]** X. Song, T. Zhang, and W. Yi, “An improved YOLOv8 safety helmet wearing detection network,” *Scientific Reports*, vol. 14, Art. no. 17550, 2024.

**Link bài báo:**  
https://www.nature.com/articles/s41598-024-68446-z

**DOI:**  
https://doi.org/10.1038/s41598-024-68446-z

---

# 24. Vậy hình Overall Architecture có phải tự thiết kế không?

## Trả lời

**Có — ở cấp độ tổng thể.**

Không có một paper duy nhất mà kiến trúc của họ giống toàn bộ:

```text
CoordConv
→ YOLO11/C3k2 backbone
→ SPPF/C2PSA
→ PAN
→ BiFormer
→ RepConv
→ anchor-free decoupled head
→ TAL
→ BCE + Focal-EIoU + DFL
→ NMS
```

Do đó:

- sơ đồ overall = hình kiến trúc của nhóm;
- các module bên trong = phải cite công trình nguồn.

## Caption gợi ý

> **Overall architecture of the proposed Rep-YOLO11s framework. The model integrates coordinate-aware input encoding [R2], YOLO11 multi-scale feature extraction [R1], FPN/PAN-style aggregation [R6], [R7], Bi-Level Routing Attention [R8], RepVGG-style structural re-parameterization [R9], and task-aligned distribution-based detection supervision [R12]–[R14].**

---

# 25. Hình BiFormer có phải tự thiết kế không?

## Phần đồ họa

Có, layout của nhóm tự vẽ.

## Phần thuật toán

Không mới.

Cơ chế đến từ BiFormer [R8].

Vì vậy caption tốt nhất:

> **Simplified schematic of the Bi-Level Routing Attention routing mechanism used in Rep-YOLO11s, adapted from Zhu et al. [R8].**

---

# 26. Hình RepConv có phải tự thiết kế không?

## Phần đồ họa

Có, nhóm tự vẽ.

## Cơ chế

Dựa trên RepVGG structural re-parameterization [R9].

Caption:

> **Structural re-parameterization topology of the RepConv block used in Rep-YOLO11s, based on the RepVGG principle [R9].**

---

# 27. Cách cite trực tiếp trong Section III — Proposed Methodology

Dưới đây là một mẫu có thể dùng gần như trực tiếp.

## 27.1. Overall Architecture

> Rep-YOLO11s is developed from the Ultralytics YOLO11 detector [R1]. The proposed architecture integrates explicit coordinate encoding using CoordConv [R2], multi-scale FPN/PAN feature aggregation [R6], [R7], Bi-Level Routing Attention [R8], RepVGG-style structural re-parameterization [R9], and an anchor-free decoupled detection head supported by DFL-based regression [R12] and task-aligned assignment [R13]. Focal-EIoU [R14] is further employed for bounding-box refinement.

---

## 27.2. CoordConv

> Following Liu et al. [R2], normalized horizontal and vertical coordinate maps are concatenated with the RGB tensor to provide explicit spatial location information before backbone feature extraction. The use of coordinate convolution for helmet detection is further supported by YOLO-CBF [R3].

---

## 27.3. Multi-scale backbone and neck

> Multi-scale feature maps P3, P4, and P5 are retained to preserve information across different spatial resolutions, following the feature-pyramid principle [R6]. The neck employs PAN-style aggregation to strengthen bidirectional information flow between shallow localization features and deeper semantic features [R7].

---

## 27.4. BiFormer

> Bi-Level Routing Attention follows Zhu et al. [R8]. Coarse region-level query-key affinity is first used to identify the top-k relevant regions, after which fine-grained token attention is computed only over the routed key-value pairs.

---

## 27.5. RepConv

> The RepConv block follows the structural re-parameterization principle of RepVGG [R9]. Parallel 3×3, 1×1, and identity branches are jointly optimized during training and algebraically transformed into an equivalent single 3×3 convolution at deployment.

---

## 27.6. Decoupled Head

> The detection head adopts an anchor-free decoupled formulation consistent with modern YOLO detectors and YOLOX [R11], using separate classification and regression branches at P3, P4, and P5.

---

## 27.7. DFL + TAL + Focal-EIoU

> Localization distributions are supervised using Distribution Focal Loss [R12], whereas positive-sample assignment follows Task Alignment Learning from TOOD [R13]. In addition, Focal-EIoU [R14] is used to improve bounding-box regression by explicitly considering overlap, center-distance, and side-length discrepancies.

---

# 28. Related Work — cấu trúc citation nên dùng

## A. YOLO-based Safety Helmet Detection

Nên dẫn:
- [R16] YOLO-DCRCF;
- [R17] Improved YOLOv8 helmet;
- [R3] YOLO-CBF.

Mục tiêu:
- chứng minh helmet detection đã phát triển mạnh trên YOLO;
- chứng minh còn vấn đề small targets, occlusion, clutter, multi-scale;
- xác định research gap.

---

## B. Multi-scale Feature Fusion

Dẫn:
- [R6] FPN;
- [R7] PANet.

Mục tiêu:
- giải thích P3/P4/P5;
- giải thích top-down + bottom-up fusion;
- chứng minh multi-scale detection có lineage rõ ràng.

---

## C. Spatial Encoding and Sparse Attention

Dẫn:
- [R2] CoordConv;
- [R8] BiFormer;
- [R3] YOLO-CBF.

Mục tiêu:
- coordinate awareness;
- sparse routing;
- ứng dụng trong helmet domain.

---

## D. Structural Re-parameterization

Dẫn:
- [R9] RepVGG;
- [R10] YOLOv7.

Mục tiêu:
- multi-branch training;
- single-path deployment;
- real-time object detection context.

---

## E. Detection Head and Optimization

Dẫn:
- [R11] YOLOX;
- [R12] Generalized Focal Loss;
- [R13] TOOD;
- [R14] Focal-EIoU.

Mục tiêu:
- anchor-free;
- decoupled cls/reg;
- distribution regression;
- task alignment;
- improved IoU regression.

---

# 29. Reference nào mạnh nhất cho từng hình?

## Overall Architecture

Không dùng:
> “Adapted from X”

vì overall là combination của nhóm.

Nên cite trong caption:
- YOLO11;
- CoordConv;
- FPN/PAN;
- BiFormer;
- RepVGG;
- DFL;
- TOOD;
- Focal-EIoU.

---

## BiFormer Detail

Reference bắt buộc:
> [R8] BiFormer.

Reference domain bổ sung:
> [R3] YOLO-CBF.

---

## RepConv Detail

Reference bắt buộc:
> [R9] RepVGG.

Reference detector bổ sung:
> [R10] YOLOv7.

---

# 30. Reference nào KHÔNG nên dùng sai mục đích?

## CSPNet

Có thể hỗ trợ:
- CSP lineage.

Không dùng để khẳng định:
> CSPNet proposed C3k2.

---

## FPN

Có thể hỗ trợ:
- feature pyramid;
- multi-scale features;
- top-down pathway.

Không nên gọi PAN là FPN.

---

## PANet

Có thể hỗ trợ:
- bottom-up path augmentation.

Không nên nói toàn bộ YOLO PAN neck giống nguyên PANet architecture.

Nên viết:
> “PAN-style” hoặc “FPN/PAN-style”.

---

## RepVGG

Hỗ trợ:
- structural re-parameterization.

Không chứng minh:
- placement RepConv cụ thể của Rep-YOLO11s.

Placement là design của nhóm/code.

---

## BiFormer

Hỗ trợ:
- BRA.

Không chứng minh:
- model helmet của nhóm;
- placement trong neck.

Placement là design của Rep-YOLO11s.

---

# 31. Các vấn đề phải kiểm tra trước khi nộp

## 31.1. RepConv placement

Nếu hình nói:
> RepConv ở neck

nhưng text nói:
> replace C3k2 in backbone

thì phải sửa.

Cần kiểm tra implementation thực tế.

---

## 31.2. CoordConv placement

Nếu hình chỉ:
> input CoordConv

nhưng text nói:
> backbone and neck layers

thì cần kiểm tra lại code và thống nhất.

---

## 31.3. BiFormer notation

Nên dùng:

\[
Q^r,\quad K^r
\]

\[
A^r = Q^r(K^r)^T
\]

\[
I^r = \operatorname{TopK}(A^r,k)
\]

Không nên để:

```text
Qr, Kr, Ar, Ir
```

trong final publication.

---

## 31.4. Focal-EIoU metadata

Dùng version journal:

```text
Neurocomputing
vol. 506
pp. 146–157
2022
```

---

## 31.5. YOLO11

Không bịa research paper.

Dùng official Ultralytics documentation cho exact architecture.

---

# 32. Bộ reference tối thiểu nếu không muốn bibliography quá dài

Nếu chỉ chọn reference quan trọng nhất, dùng:

1. YOLO11 official — [R1]
2. CoordConv — [R2]
3. FPN — [R6]
4. PANet — [R7]
5. BiFormer — [R8]
6. RepVGG — [R9]
7. YOLOX — [R11]
8. Generalized Focal Loss / DFL — [R12]
9. TOOD — [R13]
10. Focal-EIoU — [R14]
11. YOLO-CBF — [R3]
12. YOLO-DCRCF — [R16]
13. Improved YOLOv8 helmet — [R17]

Bộ này đủ để bảo vệ gần như toàn bộ các block có ý nghĩa khoa học trong architecture.

---

# 33. Danh sách References IEEE đầy đủ + link kiểm tra

## [R1] YOLO11

G. Jocher and J. Qiu, “Ultralytics YOLO11,” Ultralytics, 2024. [Online].

**Link:**  
https://docs.ultralytics.com/models/yolo11/

**Architecture:**  
https://docs.ultralytics.com/guides/yolo-architecture/

---

## [R2] CoordConv

R. Liu, J. Lehman, P. Molino, F. P. Such, E. Frank, A. Sergeev, and J. Yosinski, “An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018.

**Link:**  
https://papers.nips.cc/paper/8169-an-intriguing-failing-of-convolutional-neural-networks-and-the-coordconv-solution

---

## [R3] YOLO-CBF

Z. Wu, J. Qin, X. Xiang, and Y. Tan, “YOLO-CBF: Optimized YOLOv7 Algorithm for Helmet Detection in Road Environments,” *Electronics*, vol. 14, no. 7, Art. no. 1413, 2025.

**Link:**  
https://www.mdpi.com/2079-9292/14/7/1413

**DOI:**  
https://doi.org/10.3390/electronics14071413

---

## [R4] CSPNet

C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H. Yeh, “CSPNet: A New Backbone That Can Enhance Learning Capability of CNN,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 390–391.

**Link:**  
https://openaccess.thecvf.com/content_CVPRW_2020/html/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.html

---

## [R5] Spatial Pyramid Pooling

K. He, X. Zhang, S. Ren, and J. Sun, “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 37, no. 9, pp. 1904–1916, 2015.

**Link:**  
https://pubmed.ncbi.nlm.nih.gov/26353135/

**DOI:**  
https://doi.org/10.1109/TPAMI.2015.2389824

---

## [R6] FPN

T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2117–2125.

**Link:**  
https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

---

## [R7] PANet

S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, “Path Aggregation Network for Instance Segmentation,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 8759–8768.

**Link:**  
https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Path_Aggregation_Network_CVPR_2018_paper.html

---

## [R8] BiFormer

L. Zhu, X. Wang, Z. Ke, W. Zhang, and R. W. H. Lau, “BiFormer: Vision Transformer With Bi-Level Routing Attention,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 10323–10333.

**Link:**  
https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html

---

## [R9] RepVGG

X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun, “RepVGG: Making VGG-Style ConvNets Great Again,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 13733–13742.

**Link:**  
https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html

---

## [R10] YOLOv7

C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 7464–7475.

**Link:**  
https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html

---

## [R11] YOLOX

Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, “YOLOX: Exceeding YOLO Series in 2021,” arXiv:2107.08430, 2021.

**Link:**  
https://arxiv.org/abs/2107.08430

---

## [R12] Generalized Focal Loss / DFL

X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020.

**Link:**  
https://papers.nips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html

---

## [R13] TOOD / TAL

C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, “TOOD: Task-Aligned One-Stage Object Detection,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 3510–3519.

**Link:**  
https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html

---

## [R14] Focal-EIoU

Y.-F. Zhang, W. Ren, Z. Zhang, Z. Jia, L. Wang, and T. Tan, “Focal and Efficient IoU Loss for Accurate Bounding Box Regression,” *Neurocomputing*, vol. 506, pp. 146–157, 2022.

**Link:**  
https://www.sciencedirect.com/science/article/pii/S0925231222009018

**DOI:**  
https://doi.org/10.1016/j.neucom.2022.07.042

---

## [R15] NMS / Soft-NMS

N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS—Improving Object Detection With One Line of Code,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 5561–5569.

**Link:**  
https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html

---

## [R16] YOLO-DCRCF

J. Zhao, Z. Yang, B. Li, and Y. Zhao, “YOLO-DCRCF: An Algorithm for Detecting the Wearing of Safety Helmets and Gloves in Power Grid Operation Environments,” *Journal of Imaging*, vol. 11, no. 9, Art. no. 320, 2025.

**Link:**  
https://www.mdpi.com/2313-433X/11/9/320

**DOI:**  
https://doi.org/10.3390/jimaging11090320

---

## [R17] Improved YOLOv8 Safety Helmet Detection

X. Song, T. Zhang, and W. Yi, “An improved YOLOv8 safety helmet wearing detection network,” *Scientific Reports*, vol. 14, Art. no. 17550, 2024.

**Link:**  
https://www.nature.com/articles/s41598-024-68446-z

**DOI:**  
https://doi.org/10.1038/s41598-024-68446-z

---

# 34. Kết luận nên dùng khi trình bày novelty

Một cách diễn đạt khoa học an toàn:

> **The novelty of Rep-YOLO11s does not lie in claiming each individual component as newly invented. Instead, the contribution is the task-specific architectural integration of explicit spatial coordinate encoding, multi-scale feature aggregation, content-aware sparse routing attention, structural re-parameterization, anchor-free decoupled detection, task-aligned assignment, and complementary localization objectives within a YOLO11-based safety-helmet detection framework.**

Dịch ý:

> **Tính mới của Rep-YOLO11s không nằm ở việc xem từng module riêng lẻ là phát minh mới. Đóng góp nằm ở cách nhóm thiết kế và tích hợp có chủ đích các cơ chế mã hóa tọa độ không gian, hợp nhất đặc trưng đa tỷ lệ, attention thưa theo nội dung, tái tham số hóa cấu trúc, head anchor-free tách nhánh, gán mẫu theo task alignment và các hàm loss định vị bổ trợ vào một framework YOLO11 dành riêng cho phát hiện mũ bảo hộ.**

Đây là cách framing dễ bảo vệ hơn trước reviewer vì:
- không overclaim;
- lineage của từng module rõ;
- novelty được gắn vào **architecture integration + task-specific design + experimental validation**.

---

# 35. Checklist cuối trước khi đưa reference vào paper

- [ ] YOLO11 exact implementation dẫn [R1].
- [ ] CoordConv dẫn [R2].
- [ ] Helmet-domain CoordConv/BiFormer/Focal-EIoU có thể dẫn thêm [R3].
- [ ] C3k2 không bị gán sai cho CSPNet.
- [ ] SPPF exact dẫn Ultralytics; concept SPP dẫn [R5].
- [ ] FPN dẫn [R6].
- [ ] PAN dẫn [R7].
- [ ] BiFormer/BRA dẫn [R8].
- [ ] Hình BiFormer dùng “adapted from” nếu cần.
- [ ] RepConv dẫn RepVGG [R9].
- [ ] YOLOv7 [R10] chỉ là supporting detector reference.
- [ ] Anchor-free/decoupled design có [R11] + [R1].
- [ ] DFL dẫn [R12].
- [ ] TAL dẫn [R13].
- [ ] Focal-EIoU dẫn [R14] với metadata 2022.
- [ ] NMS nằm ở post-processing.
- [ ] YOLO-DCRCF [R16] dùng cho related helmet/PPE literature.
- [ ] Improved YOLOv8 [R17] dùng cho multi-scale/small helmet motivation.
- [ ] RepConv placement phải thống nhất giữa code, figure và text.
- [ ] CoordConv placement phải thống nhất giữa code, figure và text.
- [ ] Reference numbering cuối cùng phải đánh lại theo thứ tự xuất hiện trong manuscript.

---

**Hết tài liệu.**
