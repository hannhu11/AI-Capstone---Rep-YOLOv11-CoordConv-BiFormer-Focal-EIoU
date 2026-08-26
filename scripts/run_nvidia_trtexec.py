"""
NVIDIA Native TensorRT Pure FP16 Hardware Benchmark Launcher.
Handles cross-platform (Windows / Linux / Kaggle) TensorRT FP16 Engine compilation
and measures exact CUDA Hardware Latency & Peak Throughput using FP16 VRAM tensors and torch.cuda.Event.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def run_pure_gpu_benchmark(engine_path: str, iterations: int = 300, warmup: int = 50):
    """Measure exact CUDA Hardware Latency on Native TensorRT Engine."""
    torch.cuda.empty_cache()
    model = YOLO(engine_path, task="detect")

    # Use FP16 CUDA input tensor directly to eliminate CPU/GPU casting overhead
    try:
        x = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
        model.predict(x, verbose=False)
        backend = model.predictor.model
        _ = backend(x)
    except Exception:
        x = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float32)
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

    print("=" * 75)
    print("  [NVIDIA TENSORRT FP16 HARDWARE BENCHMARK RESULT]")
    print(f"  ⚡ Pure GPU CUDA Forward Latency : {lat_ms:.2f} ms / frame")
    print(f"  🚀 Peak TensorRT Throughput       : {fps:.1f} FPS")
    print("=" * 75 + "\n")


def execute_trtexec_benchmark(onnx_path: str, engine_out: str):
    print("=" * 75)
    print(" 🚀 NVIDIA TENSORRT PURE FP16 HARDWARE BENCHMARK")
    print(f" Input Model   : {onnx_path}")
    print(f" Output Engine : {engine_out}")
    print(f" GPU Hardware  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 75)

    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA GPU is not active in this session!")
        print("--> On Kaggle Notebook: Click Settings menu (right panel) -> Accelerator -> Select 'GPU T4 x2'!\n")
        return

    # Check if target engine file exists or needs build
    target_engine = engine_out if os.path.exists(engine_out) else onnx_path.replace(".onnx", ".engine")

    if not os.path.exists(target_engine):
        pt_path = onnx_path.replace(".onnx", ".pt")
        source_model = pt_path if os.path.exists(pt_path) else onnx_path
        print(f"-> Exporting TensorRT FP16 Engine from {source_model}...")
        model = YOLO(source_model)
        target_engine = model.export(format="engine", device=0, half=True, imgsz=640)

    run_pure_gpu_benchmark(target_engine)


if __name__ == "__main__":
    onnx_file = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.onnx"
    engine_file = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_pure_fp16.engine"

    if len(sys.argv) > 1:
        onnx_file = sys.argv[1]
    if len(sys.argv) > 2:
        engine_file = sys.argv[2]

    execute_trtexec_benchmark(onnx_file, engine_file)
