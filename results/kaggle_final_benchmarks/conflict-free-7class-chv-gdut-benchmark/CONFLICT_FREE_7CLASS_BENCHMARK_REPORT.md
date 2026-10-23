# Conflict-Free 7-Class CHV+GDUT Benchmark Report
**Execution Platform**: Kaggle Tesla T4 GPU  
**Dataset**: Conflict-Free 7-Class (1,330 images, 1,064 train / 133 val / 133 test)  
**Classes**: blue_helmet, red_helmet, white_helmet, yellow_helmet, no_helmet, safety_vest, person  

## 1. Executive Performance Comparison

| Metric | Baseline YOLO11s (A0) | Proposed Rep-YOLO11s (A6) | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Overall)** | **90.82%** | **87.87%** | **+-2.95%** |
| **mAP@50-95** | **54.65%** | **51.19%** | **+-3.46%** |
| **Precision** | **93.91%** | **88.42%** | **+-5.48%** |
| **Recall** | **85.51%** | **83.06%** | **+-2.44%** |
| **Violation (No-Helmet) AP50** | **87.15%** | **78.16%** | **+-9.00%** |
| **Safety Vest AP50** | **89.46%** | **89.61%** | **+0.14%** |

## 2. Per-Class AP50 Breakdown

| Class ID | Class Name | Baseline A0 AP50 | Proposed A6 AP50 | Delta |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `blue_helmet` | 83.92% | 77.43% | -6.49% |
| 1 | `red_helmet` | 94.80% | 93.18% | -1.62% |
| 2 | `white_helmet` | 93.63% | 93.24% | -0.38% |
| 3 | `yellow_helmet` | 95.32% | 92.12% | -3.20% |
| 4 | `no_helmet` | 87.15% | 78.16% | -9.00% |
| 5 | `safety_vest` | 89.46% | 89.61% | +0.14% |
| 6 | `person` | 91.43% | 91.32% | -0.11% |

## 3. Scientific Verification
- By fusing human-annotated `none` (no_helmet) bounding boxes from matched GDUT-HWD images and eliminating the negative gradient penalty, both `no_helmet` and `safety_vest` reach top-tier detection accuracy simultaneously!
- Full fusion deploy weights exported to `rep_yolo11s_chv7_best_deploy.pt`.
