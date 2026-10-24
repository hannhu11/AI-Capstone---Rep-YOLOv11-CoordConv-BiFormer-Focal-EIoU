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
| S3 (Harmonized Merged 7-Class)    |       40 |       5.29 |           97.08 |        96.98 |        98.76 |           84.08 |           99.09 |             98.68 |           97.87 |          98.37 |            98.54 |             99.31 |             99.5 |
| S4 (Sequential Transfer Learning) |       40 |       4.22 |           96.93 |        97.16 |        98.8  |           84.74 |           99.11 |             98.71 |           97.58 |          99.5  |            98.22 |             99.02 |             99.5 |
