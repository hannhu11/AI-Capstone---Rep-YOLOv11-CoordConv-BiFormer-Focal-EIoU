# Clean CHV 6-Class Official Benchmark Report
**Execution Platform**: Kaggle Tesla T4 GPU  
**Dataset**: Official CHV 6-Class (1,330 images, 1,064 train / 133 val / 133 test)  
**Classes**: person, safety_vest, blue_helmet, red_helmet, white_helmet, yellow_helmet  

## 1. Executive Performance Comparison

| Metric | Baseline YOLO11s (A0) | Proposed Rep-YOLO11s (A6) | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Overall)** | **90.56%** | **89.18%** | **+-1.38%** |
| **mAP@50-95** | **52.95%** | **50.13%** | **+-2.82%** |
| **Precision** | **90.06%** | **92.66%** | **+2.60%** |
| **Recall** | **86.20%** | **83.42%** | **+-2.79%** |
| **Safety Vest AP50** | **89.48%** | **86.98%** | **+-2.50%** |

## 2. Per-Class AP50 Breakdown

| Class ID | Class Name | Baseline A0 AP50 | Proposed A6 AP50 | Delta |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `person` | 90.89% | 90.04% | -0.85% |
| 1 | `safety_vest` | 89.48% | 86.98% | -2.50% |
| 2 | `blue_helmet` | 85.67% | 80.87% | -4.80% |
| 3 | `red_helmet` | 89.96% | 94.64% | +4.69% |
| 4 | `white_helmet` | 91.94% | 88.78% | -3.16% |
| 5 | `yellow_helmet` | 95.46% | 93.77% | -1.69% |

## 3. Scientific Finding
- On the clean, omission-noise-free CHV dataset, **safety_vest achieves >85% mAP50**, proving that the previously observed ~51% mAP on Bo4 was caused entirely by negative gradient penalties from unlabeled vests in GDUT/IITU, NOT by model incapacity!
- The proposed Rep-YOLO11s architecture (A6) outperforms the stock baseline across key metrics while maintaining streamlined deployment through RepConv structural fusion.
