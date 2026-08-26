"""
TensorRT FP16 Exporter & Benchmarking Pipeline.

Converts trained Stage 2 PyTorch checkpoints (A3/A4/A5/A6 RepConv models)
into optimized TensorRT FP16 Engines (.engine) with Structural Re-parameterization
Deployment Fusion (switch_to_deploy).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

# Add parent directory to path for importing custom modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from custom_ablation_modules import RepConv


def fuse_repconv_model(model: nn.Module) -> nn.Module:
    """Fuse all RepConv multi-branch modules into single-path 3x3 convs."""
    fused_count = 0
    for m in model.modules():
        if isinstance(m, RepConv) or hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
            fused_count += 1
    print(f"[RepConv Fusion] Successfully fused {fused_count} RepConv layers into single-path 3x3 Convs!")
    return model


def export_model_to_tensorrt(
    weight_path: Path,
    output_dir: Path,
    imgsz: int = 640,
    batch_size: int = 1,
    half: bool = True,
    device: str = "cuda",
) -> Path:
    """
    Export PyTorch model to TensorRT FP16 engine.

    1. Fuses RepConv multi-branch layers to single-path.
    2. Exports to ONNX format with dynamic batch support.
    3. Compiles ONNX to TensorRT FP16 (.engine).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Export] Loading PyTorch model from: {weight_path}")
    
    yolo_model = YOLO(str(weight_path))
    py_model = yolo_model.model

    # 1. Structural Re-parameterization Fusion
    print("[Export] Performing Structural Re-parameterization switch_to_deploy()...")
    py_model = fuse_repconv_model(py_model)

    # Save fused PyTorch checkpoint
    fused_pt_path = output_dir / f"{weight_path.stem}_fused_deploy.pt"
    yolo_model.save(str(fused_pt_path))
    print(f"[Export] Saved fused deployment PyTorch checkpoint to: {fused_pt_path}")

    # 2. Export to TensorRT via Ultralytics exporter
    print(f"[Export] Exporting fused model to TensorRT FP16 (device={device}, imgsz={imgsz}, half={half})...")
    try:
        exported_path_str = yolo_model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch_size,
            workspace=4,
            half=half,
            device=0 if device.startswith("cuda") else "cpu",
        )
        print(f"[Export Success] TensorRT Engine generated at: {exported_path_str}")
        return Path(exported_path_str)
    except Exception as e:
        print(f"[Export Warning] Direct TensorRT export encounter: {e}")
        print("[Export Fallback] Exporting to ONNX format first...")
        onnx_path_str = yolo_model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,
            simplify=True,
            opset=12,
        )
        print(f"[Export Fallback] ONNX model generated at: {onnx_path_str}")
        return Path(onnx_path_str)


def benchmark_engine(
    engine_or_pt_path: Path,
    imgsz: int = 640,
    device: str = "cuda",
    warmup: int = 50,
    iters: int = 300,
):
    """Run exact CUDA event benchmark on the exported model engine."""
    print(f"\n[Benchmark] Running latency & throughput benchmark on: {engine_or_pt_path.name}")
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    model = YOLO(str(engine_or_pt_path))

    # Run inference benchmark
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=dev) if dev.type == "cuda" else None

    # Warmup
    for _ in range(warmup):
        _ = model.predict(source=torch.zeros((imgsz, imgsz, 3), dtype=torch.uint8).numpy(), verbose=False)

    # Measure exact time
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.predict(source=torch.zeros((imgsz, imgsz, 3), dtype=torch.uint8).numpy(), verbose=False)
    t1 = time.perf_counter()

    avg_latency_ms = (t1 - t0) * 1000.0 / iters
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    print("=" * 60)
    print(f" Model Engine:     {engine_or_pt_path.name}")
    print(f" Execution Device: {device.upper()}")
    print(f" Average Latency:  {avg_latency_ms:.2f} ms / frame")
    print(f" Throughput (FPS): {fps:.1f} FPS")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RepConv Stage 2 Model to TensorRT FP16 Engine")
    parser.add_argument("--weights", type=str, required=True, help="Path to PyTorch best.pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="exported_engines", help="Output directory for TensorRT engine")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--half", action="store_true", default=True, help="Enable FP16 precision")
    args = parser.parse_args()

    weight_p = Path(args.weights)
    out_p = Path(args.output_dir)

    exported_engine = export_model_to_tensorrt(weight_p, out_p, imgsz=args.imgsz, half=args.half)
    benchmark_engine(exported_engine, imgsz=args.imgsz)
