"""
=============================================================================
MATCHED-ENGINE HARDWARE BENCHMARK SUITE (LOCAL RTX 3050 LAPTOP GPU)
Compliant with IEEE AAIML 2027 Reviewer Rebuttal & Evaluation Parity
=============================================================================
This script benchmarks all baseline models (YOLOv8s, YOLOv10s, YOLO11s) and
the proposed Rep-YOLO11s under identical matched-engine conditions:
- Device: NVIDIA GeForce RTX 3050 Laptop GPU (Ampere, CUDA 12.4)
- Batch Size: 1
- Resolution: 640 x 640
- Precisions: PyTorch Native FP32 and PyTorch Native FP16 (Tensor Core)
- Synchronization: Strict torch.cuda.Event timing with torch.cuda.synchronize()
- Rejection of OS Jitter: 5% trimmed mean across 300 measurement iterations
=============================================================================
"""

import sys
import time
import json
import ctypes
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

class HardwarePowerMonitor:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.nvml_available = False
        self.handle = None
        self._sampling = False
        self._samples: List[float] = []
        self._thread: Optional[threading.Thread] = None

        try:
            self.nvml = ctypes.CDLL("nvml.dll")
            ret = self.nvml.nvmlInit_v2()
            if ret == 0:
                self.handle = ctypes.c_void_p()
                ret_dev = self.nvml.nvmlDeviceGetHandleByIndex_v2(device_index, ctypes.byref(self.handle))
                if ret_dev == 0:
                    self.nvml_available = True
        except Exception:
            self.nvml_available = False

    def get_instant_power_watts(self) -> float:
        if not self.nvml_available or self.handle is None:
            return 0.0
        pwr = ctypes.c_uint()
        ret = self.nvml.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(pwr))
        if ret == 0:
            return pwr.value / 1000.0
        return 0.0

    def start_sampling(self, interval_sec: float = 0.01):
        self._samples = []
        self._sampling = True

        def _worker():
            while self._sampling:
                pwr = self.get_instant_power_watts()
                if pwr > 0:
                    self._samples.append(pwr)
                time.sleep(interval_sec)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop_sampling(self) -> float:
        self._sampling = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)


def benchmark_model_precision(
    net: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    imgsz: int = 640,
    warmup_iters: int = 100,
    eval_iters: int = 300,
    power_monitor: Optional[HardwarePowerMonitor] = None,
) -> Dict[str, float]:
    net = net.to(device=device, dtype=dtype).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = net(dummy)
        torch.cuda.synchronize(device)

    if power_monitor and power_monitor.nvml_available:
        power_monitor.start_sampling(interval_sec=0.005)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(eval_iters):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            _ = net(dummy)
            end_evt.record()
            torch.cuda.synchronize(device)
            lat_ms = start_evt.elapsed_time(end_evt)
            latencies.append(lat_ms)

    avg_pwr = 0.0
    if power_monitor and power_monitor.nvml_available:
        avg_pwr = power_monitor.stop_sampling()

    latencies.sort()
    # 5% trimmed mean
    trim_k = max(1, int(len(latencies) * 0.05))
    clean = latencies[trim_k:-trim_k]
    mean_lat = sum(clean) / len(clean)
    median_lat = clean[len(clean) // 2]
    p95_lat = latencies[int(len(latencies) * 0.95)]
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 1),
        "power_w": round(avg_pwr, 2)
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting.")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"=================================================================")
    print(f"BENCHMARKING MATCHED ENGINES ON: {gpu_name}")
    print(f"=================================================================")

    power_monitor = HardwarePowerMonitor(device_index=0)

    model_paths = {
        "YOLOv8n": "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolov8n_best.pt",
        "YOLOv8s": "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolov8s_best.pt",
        "YOLOv10n": "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolov10n_best.pt",
        "YOLOv10s": "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolov10s_best.pt",
        "YOLO11n": "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolo11n_best.pt",
        "YOLO11s": "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolo11s_best.pt",
        "Rep-YOLO11s (Proposed, Fused)": "exported_engines/yolo11s_best_fused_deploy.pt"
    }

    results = {}

    for name, path in model_paths.items():
        print(f"\n---> Benchmarking {name} ({path})...")
        yolo_obj = YOLO(path)
        net = yolo_obj.model

        # 1. FP32
        print("     [1/2] PyTorch FP32...")
        fp32_res = benchmark_model_precision(net, torch.float32, device, imgsz=640, power_monitor=power_monitor)
        print(f"           FP32 Latency: {fp32_res['mean_latency_ms']:.2f} ms | FPS: {fp32_res['fps']:.1f}")

        # 2. FP16
        print("     [2/2] PyTorch FP16...")
        fp16_res = benchmark_model_precision(net, torch.float16, device, imgsz=640, power_monitor=power_monitor)
        print(f"           FP16 Latency: {fp16_res['mean_latency_ms']:.2f} ms | FPS: {fp16_res['fps']:.1f}")

        results[name] = {
            "fp32": fp32_res,
            "fp16": fp16_res
        }

    # Save to JSON
    out_dir = Path("results/matched_engine_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rtx3050_matched_engine_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=================================================================")
    print(f"SUMMARY OF MATCHED-ENGINE BENCHMARKS (RTX 3050 Laptop GPU, 640x640)")
    print(f"{'Model':<30} | {'FP32 Lat (ms)':<14} | {'FP32 FPS':<10} | {'FP16 Lat (ms)':<14} | {'FP16 FPS':<10}")
    print(f"-" * 88)
    for name, data in results.items():
        f32 = data["fp32"]
        f16 = data["fp16"]
        print(f"{name:<30} | {f32['mean_latency_ms']:<14.2f} | {f32['fps']:<10.1f} | {f16['mean_latency_ms']:<14.2f} | {f16['fps']:<10.1f}")
    print(f"=================================================================")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
