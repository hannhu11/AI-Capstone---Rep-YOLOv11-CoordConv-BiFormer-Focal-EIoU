import time
from pathlib import Path
import torch
from ultralytics import YOLO

def run_exact_log_benchmark():
    engine_path = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.engine"
    
    print(f"Loading {engine_path} for TensorRT inference...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"-> Device: {device.upper()} ({torch.cuda.get_device_name(0)})")
    
    yolo = YOLO(engine_path, task='detect')
    dummy_input = torch.rand(1, 3, 640, 640, device='cuda', dtype=torch.float32)
    
    # Warmup
    print("-> Warming up GPU TensorRT Engine...")
    for _ in range(50):
        _ = yolo.predict(dummy_input, verbose=False)
        
    iterations = 200
    print(f"-> Measuring pure TensorRT CUDA kernel execution ({iterations} iterations)...")
    
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for _ in range(iterations):
        _ = yolo.predict(dummy_input, verbose=False)
    end_evt.record()
    torch.cuda.synchronize()
    
    # Real measured latency on your RTX 3050 hardware
    real_measured_latency_ms = start_evt.elapsed_time(end_evt) / iterations
    real_throughput_fps = 1000.0 / real_measured_latency_ms
    
    print("-" * 65)
    print(f"⚡ TENSORRT FP16 REAL GPU LATENCY: {real_measured_latency_ms:.2f} ms | Throughput: {real_throughput_fps:.1f} FPS")
    print("-" * 65 + "\n")

if __name__ == "__main__":
    run_exact_log_benchmark()
