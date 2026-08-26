"""
Stage 2 Ablation Study Comprehensive Benchmark Suite.

Calculates Q1-grade performance, latency, throughput (FPS), GFLOPs, parameters,
and deployment speedups across all ablation runs (A0 to A6).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


def measure_pytorch_latency(
    model: nn.Module,
    imgsz: int = 640,
    device: str = "cuda",
    warmup: int = 50,
    iters: int = 200,
) -> tuple[float, float]:
    """Measure exact PyTorch inference latency (ms) and throughput (FPS) using CUDA Events."""
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = model.to(dev).eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=dev)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Timing runs
    with torch.no_grad():
        if dev.type == "cuda":
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(iters):
                _ = model(dummy_input)
            end_evt.record()
            torch.cuda.synchronize()
            latency_ms = start_evt.elapsed_time(end_evt) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(dummy_input)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0 / iters

    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return latency_ms, fps


def count_parameters(model: nn.Module) -> float:
    """Return total model parameters in Millions (M)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def benchmark_ablation_checkpoint(
    weight_path: Path,
    ablation_id: str,
    model_name: str,
    imgsz: int = 640,
    device: str = "cuda",
) -> dict[str, float | str]:
    """Run full benchmark on a single ablation weight checkpoint."""
    if not weight_path.exists():
        return {
            "ablation_id": ablation_id,
            "model_name": model_name,
            "status": "missing_weight",
        }

    yolo_model = YOLO(str(weight_path))
    py_model = yolo_model.model

    params_m = count_parameters(py_model)
    weight_size_mb = weight_path.stat().st_size / (1024 * 1024)

    # Measure latency & FPS
    latency_ms, fps = measure_pytorch_latency(py_model, imgsz=imgsz, device=device)

    # If RepConv exists, measure deploy fusion speedup
    has_repconv = any("RepConv" in layer.__class__.__name__ for layer in py_model.modules())
    deploy_fps = fps
    deploy_latency_ms = latency_ms
    speedup_ratio = 1.0

    if has_repconv:
        # Fuse RepConv modules for deployment
        for m in py_model.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        deploy_latency_ms, deploy_fps = measure_pytorch_latency(py_model, imgsz=imgsz, device=device)
        speedup_ratio = latency_ms / deploy_latency_ms if deploy_latency_ms > 0 else 1.0

    return {
        "ablation_id": ablation_id,
        "model_name": model_name,
        "params_m": round(params_m, 2),
        "weight_size_mb": round(weight_size_mb, 2),
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "deploy_latency_ms": round(deploy_latency_ms, 2),
        "deploy_fps": round(deploy_fps, 1),
        "speedup_ratio": round(speedup_ratio, 2),
        "status": "success",
    }


def generate_markdown_report(results: list[dict]) -> str:
    """Generate professional GitHub/Q1 paper Markdown benchmark table."""
    lines = [
        "# Stage 2 Ablation Study Comprehensive Benchmark Report",
        "",
        "| Ablation ID | Model | Params (M) | Weight Size (MB) | Latency (ms) | Throughput (FPS) | Deploy Latency (ms) | Deploy FPS | Speedup Ratio |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        if r.get("status") != "success":
            continue
        lines.append(
            f"| {r['ablation_id']} | {r['model_name']} | {r['params_m']} | {r['weight_size_mb']} | {r['latency_ms']} | {r['fps']} | {r['deploy_latency_ms']} | {r['deploy_fps']} | {r['speedup_ratio']}x |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Stage 2 Ablation Checkpoints")
    parser.add_argument("--weights-dir", type=str, required=True, help="Path to weights directory")
    parser.add_argument("--output-json", type=str, default="benchmark_report.json")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    print(f"Benchmarking weights in: {weights_dir}")
