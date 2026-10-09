"""
Local Hardware Benchmark Audit Script: Rep-YOLO11s vs Baseline YOLO11s
Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
Execution Mode: Synchronized PyTorch FP32 and FP16 Inference (batch=1, 640x640)
"""

import time
from pathlib import Path
import torch
from ultralytics import YOLO


def benchmark_model(model_path: str, model_title: str, device: torch.device, n_warmup: int = 50, n_iters: int = 200):
    print(f"\n--- Benchmarking: {model_title} ({Path(model_path).name}) ---")
    if not Path(model_path).exists():
        print(f"  [ERROR] File not found: {model_path}")
        return None

    model = YOLO(model_path)
    model.to(device)

    # 1. PyTorch FP32 Benchmark
    dummy_fp32 = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)
    for _ in range(n_warmup):
        _ = model.model(dummy_fp32)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model.model(dummy_fp32)
        torch.cuda.synchronize()
    lat_fp32 = (time.perf_counter() - start) / n_iters * 1000.0
    fps_fp32 = 1000.0 / lat_fp32
    print(f"  PyTorch FP32 Latency: {lat_fp32:.2f} ms ({fps_fp32:.1f} FPS)")

    # 2. PyTorch FP16 Benchmark
    model.model.half()
    dummy_fp16 = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float16)
    for _ in range(n_warmup):
        _ = model.model(dummy_fp16)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model.model(dummy_fp16)
        torch.cuda.synchronize()
    lat_fp16 = (time.perf_counter() - start) / n_iters * 1000.0
    fps_fp16 = 1000.0 / lat_fp16
    print(f"  PyTorch FP16 Latency: {lat_fp16:.2f} ms ({fps_fp16:.1f} FPS)")

    return {
        "model": model_title,
        "fp32_ms": lat_fp32,
        "fp32_fps": fps_fp32,
        "fp16_ms": lat_fp16,
        "fp16_fps": fps_fp16,
    }


def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available on this system.")
        return

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f"LOCAL HARDWARE BENCHMARK AUDIT ON: {gpu_name}")
    print("=" * 70)

    # 1. Baseline YOLO11s
    res_base = benchmark_model("yolo11s.pt", "Baseline YOLO11s", device)

    # 2. Rep-YOLO11s Fused Deployment
    res_rep = benchmark_model("exported_engines/yolo11s_best_fused_deploy.pt", "Rep-YOLO11s (Fused Deploy)", device)

    print("\n" + "=" * 70)
    print("SUMMARY OF EMPIRICAL MEASUREMENTS (Table IV Verification)")
    print("=" * 70)
    print(f"{'Model Architecture':<30} | {'FP32 Latency':<15} | {'FP16 Latency':<15}")
    print("-" * 70)
    if res_base:
        print(f"{res_base['model']:<30} | {res_base['fp32_ms']:>6.2f} ms ({res_base['fp32_fps']:>5.1f} FPS) | {res_base['fp16_ms']:>6.2f} ms ({res_base['fp16_fps']:>5.1f} FPS)")
    if res_rep:
        print(f"{res_rep['model']:<30} | {res_rep['fp32_ms']:>6.2f} ms ({res_rep['fp32_fps']:>5.1f} FPS) | {res_rep['fp16_ms']:>6.2f} ms ({res_rep['fp16_fps']:>5.1f} FPS)")
    print("=" * 70)


if __name__ == "__main__":
    main()
