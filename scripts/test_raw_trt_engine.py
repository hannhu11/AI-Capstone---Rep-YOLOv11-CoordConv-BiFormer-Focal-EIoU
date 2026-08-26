import time
import torch

def test_tensorrt_direct_benchmark(engine_path: str, iterations=200, warmup=50):
    print("\n==================================================================")
    print(f" BENCHMARKING DIRECT TENSORRT ENGINE: {engine_path}")
    print("==================================================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"-> Execution Device : {device.upper()} ({torch.cuda.get_device_name(0)})")
    
    # Check tensorrt python bindings
    try:
        import tensorrt as trt
        print(f"-> TensorRT Python Version: {trt.__version__}")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        
        print(f"-> Engine successfully deserialized! Input/Output bindings ready.")
        
        # Prepare CUDA input/output memory buffers directly on GPU (Pure VRAM)
        # Input: (1, 3, 640, 640) float32 / float16
        input_shape = (1, 3, 640, 640)
        output_shape = (1, 6, 8400)
        
        d_input = torch.randn(input_shape, device='cuda', dtype=torch.float32)
        d_output = torch.empty(output_shape, device='cuda', dtype=torch.float32)
        
        bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]
        
        # Warmup GPU
        for _ in range(warmup):
            context.execute_v2(bindings)
            
        # CUDA Event precise timing
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        for _ in range(iterations):
            context.execute_v2(bindings)
        end_evt.record()
        torch.cuda.synchronize()
        
        lat_ms = start_evt.elapsed_time(end_evt) / iterations
        fps = 1000.0 / lat_ms
        
        print("------------------------------------------------------------------")
        print("  [PURE GPU TENSORRT FORWARD LATENCY]")
        print(f"  - Pure GPU Forward Latency (T_gpu) : {lat_ms:.2f} ms / frame")
        print(f"  - Peak TensorRT Throughput (FPS)    : {fps:.1f} FPS")
        print("------------------------------------------------------------------\n")
        return
        
    except Exception as e:
        print(f"[Notice] Direct TRT API notice: {e}")
        print("Falling back to Ultralytics raw CUDA predict tensor execution...")
        
    from ultralytics import YOLO
    model = YOLO(engine_path)
    
    dummy_input = torch.randn(1, 3, 640, 640, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)
        
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for _ in range(iterations):
        _ = model(dummy_input)
    end_evt.record()
    torch.cuda.synchronize()
    
    lat_ms = start_evt.elapsed_time(end_evt) / iterations
    fps = 1000.0 / lat_ms
    
    print("------------------------------------------------------------------")
    print("  [PURE GPU FORWARD LATENCY]")
    print(f"  - Pure GPU Forward Latency (T_gpu) : {lat_ms:.2f} ms / frame")
    print(f"  - Peak TensorRT Throughput (FPS)    : {fps:.1f} FPS")
    print("------------------------------------------------------------------\n")

if __name__ == "__main__":
    engine_path = "Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.engine"
    test_tensorrt_direct_benchmark(engine_path)
