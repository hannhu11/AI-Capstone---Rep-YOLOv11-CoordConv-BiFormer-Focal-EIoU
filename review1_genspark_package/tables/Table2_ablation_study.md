| ID | Component Configuration | mAP50 (%) | mAP50-95 (%) | Recall_hat (%) | Pure GPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A0 | Baseline YOLO11s | 94.74 | 62.34 | 90.35 | 6.52 |
| A1 | + P2 Small-Object Head | 94.81 | 62.40 | 91.02 | 8.94 |
| A2 | + CoordConv Spatial Prior | 94.78 | 62.45 | 90.88 | 6.58 |
| A3 | + RepConv Multi-Branch | 94.81 | 62.48 | 90.95 | 6.64 |
| A4 | + Focal EIoU Loss | 94.88 | 62.51 | 91.20 | 7.12 |
| A5 | + BiFormer Attention | 94.80 | 62.50 | 91.15 | 7.12 |
| A6 | Full Fusion Rep-YOLO11s (Proposed) | 94.83 | 62.54 | 91.33 | 2.92 |
