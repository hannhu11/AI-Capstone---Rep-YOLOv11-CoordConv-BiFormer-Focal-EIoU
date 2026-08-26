import time
from pathlib import Path
import torch
from ultralytics import YOLO

def benchmark_a6_pure_gpu(weights_path: str):
    print("\n==================================================================")
    print(f" BENCHMARKING OFFICIAL ABLATION A6 WEIGHTS: {weights_path}")
    print("==================================================================")
    
    if not Path(weights_path).exists():
        print(f"[Error] File not found: {weights_path}")
        return
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"-> Execution Device : {device.upper()} ({torch.cuda.get_device_name(0)})")
    
    yolo = YOLO(weights_path)
    model = yolo.model.to(device).eval()
    
    # RepConv switch_to_deploy
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
            
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            
    # CUDA Events timing for pure GPU forward
    with torch.no_grad():
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(200):
            _ = model(dummy_input)
        end_evt.record()
        torch.cuda.synchronize()
        lat_ms = start_evt.elapsed_time(end_evt) / 200.0
        
    fps = 1000.0 / lat_ms
    print("------------------------------------------------------------------")
    print(f"  [PURE MODEL FORWARD LATENCY (T_gpu)] : {lat_ms:.2f} ms")
    print(f"  [PURE MODEL THROUGHPUT (FPS)]        : {fps:.1f} FPS")
    print("==================================================================\n")

if __name__ == "__main__":
    a6_weights = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.pt"
    benchmark_a6_pure_gpu(a6_weights)
