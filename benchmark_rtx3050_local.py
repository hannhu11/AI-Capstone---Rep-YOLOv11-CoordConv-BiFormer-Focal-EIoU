import time
import torch
from ultralytics import YOLO

def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name}")
    
    # Load model
    model = YOLO("yolo11s.pt")
    model.to(device)
    
    # FP32 Benchmark
    dummy_input = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)
    # Warmup
    for _ in range(50):
        _ = model.model(dummy_input)
    torch.cuda.synchronize()
    
    # Timing FP32
    n_iters = 200
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model.model(dummy_input)
        torch.cuda.synchronize()
    lat_fp32 = (time.perf_counter() - start) / n_iters * 1000
    fps_fp32 = 1000.0 / lat_fp32
    print(f"PyTorch FP32 on {gpu_name}: {lat_fp32:.2f} ms ({fps_fp32:.1f} FPS)")
    
    # FP16 Benchmark
    model.model.half()
    dummy_fp16 = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float16)
    for _ in range(50):
        _ = model.model(dummy_fp16)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model.model(dummy_fp16)
        torch.cuda.synchronize()
    lat_fp16 = (time.perf_counter() - start) / n_iters * 1000
    fps_fp16 = 1000.0 / lat_fp16
    print(f"PyTorch FP16 on {gpu_name}: {lat_fp16:.2f} ms ({fps_fp16:.1f} FPS)")

if __name__ == "__main__":
    benchmark()
