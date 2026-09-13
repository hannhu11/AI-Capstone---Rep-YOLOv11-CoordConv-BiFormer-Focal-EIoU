# SHWD-YOLO11s — Hồ sơ thí nghiệm

Tài liệu tóm tắt các notebook, cấu hình và artifact dùng trong bài báo cho hai mô hình:

- **SHWD-YOLO11s-DA:** tối ưu tăng cường dữ liệu.
- **SHWD-YOLO11s-LO:** tối ưu loss weight và lịch học.

## Notebook chính

### DataAugmentation

**Link:** https://www.kaggle.com/code/tundng111/shwd-stage-3-master-research-pipeline-dataaugment

Notebook thực hiện:

- Chuẩn bị SHWD định dạng YOLO với hai lớp `hat` và `person`.
- Fine-tune YOLO11s ở độ phân giải 960 trên hai GPU.
- Chạy Grad-CAM, đánh giá năm partition và benchmark triển khai.
- Lưu checkpoint, `results.csv`, `args.yaml` và báo cáo inference.

**Cấu hình chính:** `box=7.5`, `mixup=0.0`, `cos_lr=False`.

### LossOptimize

**Link:** https://www.kaggle.com/code/tundng111/shwd-stage-3-master-research-pipeline-lossoptimize

Notebook thực hiện:

- Dùng cùng dataset, checkpoint nền và kiến trúc YOLO11s.
- Tăng `box` từ 7.5 lên 10.0.
- Bật `mixup=0.15` và `cos_lr=True`.
- Huấn luyện DDP, validation và benchmark triển khai.
- Lưu checkpoint cùng các artifact phục vụ báo cáo.

> LO thay đổi đồng thời nhiều thành phần nên kết quả hiện tại chứng minh hiệu quả của cấu hình kết hợp, chưa chứng minh riêng tác dụng của Focal-EIoU.

### Cross-Domain Generalization

**Link:** https://www.kaggle.com/code/tundng111/shwd-cross-domain-benchmark-shel5k-gduthwd-01

Notebook thực hiện:

- Load hai checkpoint DA và LO.
- Đánh giá trên VOC2028, GDUT-HWD, SHEL5K, Hard Hat Workers và SFCHD.
- Tổng hợp Precision, Recall, mAP50 và mAP50–95.
- Ghi rõ protocol `official`, `standardized` hoặc `deterministic holdout`.
- Tạo dữ liệu cho Table II trong bài báo.

## Các stage được đề cập trong bài báo

| Stage | Chức năng | Đầu ra chính |
|---|---|---|
| Grad-CAM | Trực quan hóa vùng mô hình tập trung | Hình heatmap so sánh |
| 5-partition evaluation | Đánh giá độ ổn định của một checkpoint trên năm partition | `KFOLD_REPORT.md`, `metrics_verified.json` |
| DataAugmentation | Fine-tune với chính sách tăng cường dữ liệu | `best.pt`, `results.csv`, `args.yaml` |
| LossOptimize | Fine-tune với box weight, MixUp và cosine LR | `best.pt`, `results.csv`, `args.yaml` |
| Knowledge Distillation | Pipeline teacher–student được đề cập trong nghiên cứu | Chưa có paired result đủ xác minh |
| Deployment | Benchmark FP32, FP16, ONNX CPU và CCTV | `inference_benchmark_report.csv` |
| Cross-domain | Đánh giá DA/LO trên năm dataset | Table II |

## Kết quả chính

| Model | Precision | Recall | mAP50 | mAP50–95 | Best epoch |
|---|---:|---:|---:|---:|---:|
| SHWD-YOLO11s-DA | 0.93241 | 0.92886 | 0.96111 | 0.63513 | 60 |
| SHWD-YOLO11s-LO | 0.94014 | 0.91938 | 0.96148 | 0.63760 | 62 |

LO tăng nhẹ Precision và mAP nhưng giảm Recall so với DA.

## Benchmark triển khai

| Model | GPU FP32 | GPU FP16 | ONNX CPU 4T | CCTV GPU |
|---|---:|---:|---:|---:|
| SHWD-YOLO11s-DA | 48.7 FPS | 48.8 FPS | 2.9 FPS | 64.2 FPS |
| SHWD-YOLO11s-LO | 47.0 FPS | 47.2 FPS | 3.3 FPS | 70.0 FPS |

Các phép đo dùng đầu vào 960 và batch 1. CCTV và core inference là hai code path riêng; không cộng trực tiếp latency giữa chúng.

## Cross-domain mAP50–95

| Dataset | Protocol | DA | LO |
|---|---|---:|---:|
| VOC2028 (SHWD) | Official | 0.7229 | 0.7244 |
| GDUT-HWD Standardized | Standardized | 0.3801 | 0.3911 |
| SHEL5K Standardized | Standardized | 0.2498 | 0.2436 |
| Hard Hat Workers | Deterministic holdout | 0.4222 | 0.4385 |
| SFCHD | Official | 0.3632 | 0.3735 |

LO tăng mAP50–95 trên 4/5 dataset; SHEL5K vẫn là miền khó nhất. Không gộp các protocol khác nhau thành một điểm trung bình chung.

## Artifact

- [Output DataAugmentation](../SHWD_Stage3_Outputs_Compact-DataAugment)
- [Output LossOptimize](../SHWD_Stage3_Outputs_Compact-LossOptimize)
- [Benchmark DataAugmentation](../SHWD_Stage3_Outputs_Compact-DataAugment/results/inference_benchmark_report.csv)
- [Benchmark LossOptimize](../SHWD_Stage3_Outputs_Compact-LossOptimize/results/inference_benchmark_report.csv)
- [Metric provenance](metric_provenance.md)
- [PDF nguồn](../RepYOLO11s%20%281%29.pdf)
- [Source bài báo](SHWD_YOLO11s_Paper.tex)
- [PDF bài báo](SHWD_YOLO11s_Paper.pdf)

## Lưu ý

- Đánh giá năm partition không phải huấn luyện K-Fold năm model độc lập.
- Knowledge Distillation và TensorRT/INT8 chưa có artifact kết quả đầy đủ.
- Cần ablation kiểm soát và xác minh DDP trước khi tuyên bố riêng về Focal-EIoU.
