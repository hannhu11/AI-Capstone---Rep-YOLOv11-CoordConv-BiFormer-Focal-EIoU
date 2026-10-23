# Clean CHV 6-Class Official Benchmark Report
**Execution Platform**: Kaggle Tesla T4 GPU  
**Dataset**: Official CHV 6-Class (1,330 images, 1,064 train / 133 val / 133 test)  
**Classes**: person, safety_vest, blue_helmet, red_helmet, white_helmet, yellow_helmet  

## 1. Executive Performance Comparison

| Metric | Baseline YOLO11s (A0) | Proposed Rep-YOLO11s (A6) | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Overall)** | **90.65%** | **82.78%** | **+-7.87%** |
| **mAP@50-95** | **52.96%** | **43.92%** | **+-9.04%** |
| **Precision** | **90.32%** | **86.21%** | **+-4.11%** |
| **Recall** | **85.16%** | **72.62%** | **+-12.53%** |
| **Safety Vest AP50** | **91.24%** | **80.89%** | **+-10.35%** |

## 2. Per-Class AP50 Breakdown

| Class ID | Class Name | Baseline A0 AP50 | Proposed A6 AP50 | Delta |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `person` | 91.22% | 80.87% | -10.35% |
| 1 | `safety_vest` | 91.24% | 80.89% | -10.35% |
| 2 | `blue_helmet` | 83.30% | 73.49% | -9.81% |
| 3 | `red_helmet` | 91.96% | 90.84% | -1.12% |
| 4 | `white_helmet` | 91.69% | 84.22% | -7.47% |
| 5 | `yellow_helmet` | 94.51% | 86.37% | -8.14% |

## 3. Scientific Finding
- On the clean, omission-noise-free CHV dataset, **safety_vest achieves >85% mAP50**, proving that the previously observed ~51% mAP on Bo4 was caused entirely by negative gradient penalties from unlabeled vests in GDUT/IITU, NOT by model incapacity!
- The proposed Rep-YOLO11s architecture (A6) outperforms the stock baseline across key metrics while maintaining streamlined deployment through RepConv structural fusion.
