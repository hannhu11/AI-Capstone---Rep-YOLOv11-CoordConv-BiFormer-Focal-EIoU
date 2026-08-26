import time
import torch
from ultralytics import YOLO

def measure_trt_pure_cuda_latency():
    engine_path = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.engine"
    print("\n==================================================================")
    print(f" DIRECT CUDA EVENT MEASUREMENT: {engine_path}")
    print("==================================================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"-> Device: {device.upper()} ({torch.cuda.get_device_name(0)})")
    
    yolo = YOLO(engine_path, task='detect')
    
    # Input tensor (1, 3, 640, 640)
    dummy_tensor = torch.zeros((640, 640, 3), dtype=torch.uint8).numpy()
    
    # Warmup
    for _ in range(30):
        _ = yolo.predict(dummy_tensor, verbose=False)
        
    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = yolo.predict(dummy_tensor, verbose=False)
    t1 = time.perf_counter()
    
    lat_ms = (t1 - t0) * 1000.0 / iterations
    fps = 1000.0 / lat_ms
    
    print("------------------------------------------------------------------")
    print(f"  [TENSORRT ENGINE PREDICT LATENCY] : {lat_ms:.2f} ms")
    print(f"  [TENSORRT ENGINE THROUGHPUT]      : {fps:.1f} FPS")
    print("==================================================================\n")

if __name__ == "__main__":
    measure_trt_pure_cuda_latency()
