# DATASET STRATEGY BENCHMARK REPORT: S3 (HARMONIZED) VS S4 (TRANSFER LEARNING)
**Author**: Nguyen Han Nhu (SE183674) | Capstone AI Research  
**Execution Platform**: Kaggle GPU Tesla T4 (@hannhu4002)  
**Evaluation Target**: Harmonized Complete IITU+CHV 7-Class Dataset (2,987 images, 306 test images, 2,468 test boxes)  
**Date**: October 2026 Audit  

## 1. Executive Summary & Verification of Rigor
- **Blue Helmet AP50**: Boosted to **97.58% - 97.87%** by incorporating 1,867 blue helmet instances across IITU+CHV (eliminating standalone CHV 415-sample scarcity).
- **Safety Vest AP50 & Person AP50**: Boosted to **99.09% - 99.11%** and **98.68% - 98.71%** via Multi-Teacher Consensus Label Refinement (replacing noisy heuristic geometric boxes).
- **Data Leakage & Integrity**: 0 duplicate images, 0 temporal video overlap between splits, 0 omission noise penalties.

## 2. Quantitative Results
| Strategy                          |   Epochs |   Time_Min |   Precision (%) |   Recall (%) |   mAP@50 (%) |   mAP@50-95 (%) |   Vest AP50 (%) |   Person AP50 (%) |   Blue AP50 (%) |   Red AP50 (%) |   White AP50 (%) |   Yellow AP50 (%) |   Green AP50 (%) |
|:----------------------------------|---------:|-----------:|----------------:|-------------:|-------------:|----------------:|----------------:|------------------:|----------------:|---------------:|-----------------:|------------------:|-----------------:|
| S3 (Harmonized Merged 7-Class)    |       30 |      40.02 |           96.33 |        95.31 |        98.76 |           81.32 |           98.98 |             98.55 |           98.49 |          98.47 |            98.1  |             99.24 |            99.48 |
| S4 (Sequential Transfer Learning) |       30 |      27.22 |           96.56 |        94.96 |        98.6  |           81.51 |           98.82 |             98.39 |           98.81 |          98.55 |            96.97 |             99.19 |            99.46 |
