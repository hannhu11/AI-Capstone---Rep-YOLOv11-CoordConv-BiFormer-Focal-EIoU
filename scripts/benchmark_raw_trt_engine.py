import time
from pathlib import Path
import torch

def benchmark_raw_onnx_fp16(onnx_path: str, iterations=200, warmup=50):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Error] onnxruntime not installed.")
        return

    print(f"\n==================================================================")
    print(f" BENCHMARKING RAW ONNX / TENSORRT FP16: {onnx_path}")
    print(f"==================================================================")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Create dummy numpy array
    dummy_input = torch.randn(1, 3, 640, 640, device='cuda').cpu().numpy()
    
    # Warmup
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: dummy_input})
        
    # Measure
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = session.run([output_name], {input_name: dummy_input})
    t1 = time.perf_counter()
    
    lat_ms = (t1 - t0) * 1000.0 / iterations
    fps = 1000.0 / lat_ms
    print(f"  [PURE FORWARD LATENCY] : {lat_ms:.2f} ms")
    print(f"  [PURE THROUGHPUT]       : {fps:.1f} FPS")
    print("==================================================================\n")

if __name__ == "__main__":
    fp16_onnx = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.fp16.onnx"
    benchmark_raw_onnx_fp16(fp16_onnx)
