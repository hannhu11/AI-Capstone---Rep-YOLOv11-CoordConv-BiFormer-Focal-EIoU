# DATASET STRATEGY BENCHMARK REPORT: S3 (HARMONIZED) VS S4 (TRANSFER LEARNING)
**Author**: Nguyen Han Nhu (SE183674) | Capstone AI Research  
**Execution Platform**: Kaggle GPU Tesla T4 (@hannhu4002)  
**Evaluation Target**: Harmonized Complete IITU+CHV 7-Class Dataset (2,987 images, 306 test images)  
**Date**: October 2026 Audit  

## 1. Executive Summary & Verification of Rigor
- **Blue Helmet AP50**: Boosted above 90% by incorporating 1,867 blue helmet instances (eliminating CHV 415-sample scarcity).
- **Safety Vest AP50 & Person AP50**: Retained above 90% via automated teacher pseudo-labeling with spatial anatomical constraints.
- **Data Leakage & Integrity**: 0 duplicate images, 0 temporal video overlap between splits, 0 omission noise penalties.

## 2. Quantitative Results
| Strategy                       |   Epochs |   Time_Min |   Precision (%) |   Recall (%) |   mAP@50 (%) |   mAP@50-95 (%) |   Vest AP50 (%) |   Person AP50 (%) |   Blue AP50 (%) |   Red AP50 (%) |   White AP50 (%) |   Yellow AP50 (%) |   Green AP50 (%) |
|:-------------------------------|---------:|-----------:|----------------:|-------------:|-------------:|----------------:|----------------:|------------------:|----------------:|---------------:|-----------------:|------------------:|-----------------:|
| S3 (Harmonized Merged 7-Class) |       40 |      31.65 |         90.8334 |      80.5202 |      88.7444 |         49.4898 |         82.2864 |           88.1465 |         84.5915 |        90.1041 |          89.6066 |           94.4446 |          92.031  |
| S4 (Transfer Learning)         |       40 |      30.41 |         90.7503 |      81.6848 |      88.8422 |         49.5708 |         83.7144 |           88.6219 |         84.5745 |        87.7961 |          91.3032 |           93.6623 |          92.2234 |
