| Hardware Target | Execution Engine | Resolution | Inference Latency (ms) | Throughput (FPS) | Operational Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NVIDIA Tesla T4 GPU | TensorRT 11.2 FP16 | 640x640 | 2.92 | 342.5 | Ultra Real-Time (>14x standard CCTV) |
| NVIDIA GeForce RTX 3050 Laptop | TensorRT 11.2 FP16 | 640x640 | 5.35 | 187.1 | High-Performance Edge AI |
| NVIDIA GeForce RTX 3050 Laptop | PyTorch Native FP32 | 640x640 | 12.43 | 80.4 | Native PyTorch Edge Inference |
| RTSP Video Pipeline (RTX 3050) | End-to-End Stream (Dec+Infer+NMS+UI) | 640x640 | 10.54-15.38 | 65.0-95.0 | Multi-Camera Industrial CCTV |
| NVIDIA Tesla T4 GPU | PyTorch FP32 | 960x960 | 20.53 | 48.7 | High-Resolution Surveillance |
| NVIDIA Tesla T4 GPU | CCTV Video Stream | 960x960 | 14.28 | 70.0 | High-Resolution RTSP Stream |
| NVIDIA GeForce MX230 (Edge Laptop) | PyTorch Native FP32 | 640x640 | 36.00 | 27.8 | Budget Edge Device (>24 FPS Real-Time) |
| Edge CPU (Intel Core 4-Cores) | ONNX Runtime INT8 | 960x960 | 303.00 | 3.3 | Constrained Embedded Low-Power |
