import time
import torch
from ultralytics import YOLO

def measure_pure_trt_forward():
    engine_path = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.engine"
    print("\n==================================================================")
    print(f" MEASURING PURE GPU FORWARD TIME ON TENSORRT ENGINE: {engine_path}")
    print("==================================================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"-> Execution Device : {device.upper()} ({torch.cuda.get_device_name(0)})")
    
    yolo = YOLO(engine_path, task='detect')
    
    # Generate normalized float tensor strictly in [0.0, 1.0] to prevent auto-division warning
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
    
    total_time_ms = start_evt.elapsed_time(end_evt) / iterations
    fps = 1000.0 / total_time_ms
    
    print("------------------------------------------------------------------")
    print(f"  [RESULT] TensorRT Predict Latency : {total_time_ms:.2f} ms / frame")
    print(f"  [RESULT] Peak TensorRT Throughput : {fps:.1f} FPS")
    print("==================================================================\n")

if __name__ == "__main__":
    measure_pure_trt_forward()
