"""
Pure GPU TensorRT FP16 Latency & Throughput Benchmark.
Measures exact CUDA Forward Latency excluding Python wrapper overhead.
Fully reproducible on Local GPUs and Kaggle (NVIDIA Tesla T4 / P100 / RTX 3050).
"""

from __future__ import annotations

import argparse
import time
import torch
from ultralytics import YOLO

# Enable cuDNN benchmark and TensorFloat32 on NVIDIA GPUs
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def run_pure_gpu_benchmark(engine_path: str, iterations: int = 300, warmup: int = 50):
    print("=" * 70)
    print(f" 🎯 PURE GPU TENSORRT FP16 HARDWARE BENCHMARK")
    print(f" Model Engine : {engine_path}")
    print(f" GPU Hardware : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    model = YOLO(engine_path, task="detect")
    
    # Create input tensor directly on GPU VRAM
    x = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float32)

    # Initialize predictor and extract raw AutoBackend model
    model.predict(x, verbose=False)
    backend = model.predictor.model

    print(f"-> Warming up GPU CUDA Cores ({warmup} iterations)...")
    for _ in range(warmup):
        _ = backend(x)

    print(f"-> Measuring Pure GPU CUDA Kernel Latency ({iterations} iterations)...")
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iterations):
        _ = backend(x)
    end_evt.record()
    torch.cuda.synchronize()

    lat_ms = start_evt.elapsed_time(end_evt) / iterations
    fps = 1000.0 / lat_ms

    print("-" * 70)
    print(f"  [HARDWARE BENCHMARK RESULT]")
    print(f"  ⚡ Pure GPU CUDA Forward Latency : {lat_ms:.2f} ms / frame")
    print(f"  🚀 Peak TensorRT Throughput       : {fps:.1f} FPS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure GPU TensorRT Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.engine",
        help="Path to TensorRT engine",
    )
    args = parser.parse_args()
    run_pure_gpu_benchmark(args.model)
