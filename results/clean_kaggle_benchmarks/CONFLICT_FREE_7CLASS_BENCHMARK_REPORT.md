# Conflict-Free 7-Class CHV+GDUT Benchmark Report
**Execution Platform**: Kaggle Tesla T4 GPU  
**Dataset**: Conflict-Free 7-Class (1,330 images, 1,064 train / 133 val / 133 test)  
**Classes**: blue_helmet, red_helmet, white_helmet, yellow_helmet, no_helmet, safety_vest, person  

## 1. Executive Performance Comparison

| Metric | Baseline YOLO11s (A0) | Proposed Rep-YOLO11s (A6) | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Overall)** | **78.30%** | **70.80%** | **+-7.50%** |
| **mAP@50-95** | **45.42%** | **37.82%** | **+-7.60%** |
| **Precision** | **81.57%** | **88.05%** | **+6.48%** |
| **Recall** | **75.05%** | **63.85%** | **+-11.20%** |
| **Violation (No-Helmet) AP50** | **5.46%** | **1.95%** | **+-3.51%** |
| **Safety Vest AP50** | **88.97%** | **83.32%** | **+-5.65%** |

## 2. Per-Class AP50 Breakdown

| Class ID | Class Name | Baseline A0 AP50 | Proposed A6 AP50 | Delta |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `blue_helmet` | 83.24% | 73.35% | -9.89% |
| 1 | `red_helmet` | 94.99% | 88.31% | -6.68% |
| 2 | `white_helmet` | 90.51% | 80.75% | -9.76% |
| 3 | `yellow_helmet` | 94.22% | 87.23% | -6.99% |
| 4 | `no_helmet` | 5.46% | 1.95% | -3.51% |
| 5 | `safety_vest` | 88.97% | 83.32% | -5.65% |
| 6 | `person` | 90.71% | 80.69% | -10.02% |

## 3. Scientific Verification & Failure Analysis of Heuristic 7-Class
- **Why `no_helmet` (Violation) collapsed to 1.95% - 5.46% AP50**:
  1. *Extreme Class Imbalance*: The test set contains only 62 violation instances (5.4% of all boxes) compared to 450 persons and 217 vests.
  2. *Spatial Misalignment from Heuristic Labels*: Synthesizing `no_helmet` boxes from the top 25% of `person` bounding boxes produces arbitrary, noisy boundaries that do not align with true anatomical head positions (e.g. crouching workers, workers seen from back, occluded heads).
  3. *Scientific Verdict*: Heuristically synthesizing `no_helmet` annotations fails peer-review quality standards. In real-world deployment, safety violation should be detected via **Spatial Relational Logic** (`Person` bounding box with no intersecting `Helmet` bounding box), preserving 100% ground-truth label integrity without contaminating the training dataset.
- Full fusion deploy weights exported to `rep_yolo11s_chv7_best_deploy.pt`.
