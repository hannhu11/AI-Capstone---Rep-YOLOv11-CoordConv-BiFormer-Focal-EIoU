import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def benchmark_pure_gpu_latency(model_path: str, imgsz: int = 640, iterations: int = 200, warmup: int = 50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n==================================================================")
    print(f"  PHAN 1: DO PURE GPU MODEL INFERENCE LATENCY & PEAK FPS ({device.upper()})")
    print("==================================================================")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"-> Thiet bi GPU        : {gpu_name}")
    
    print(f"-> Nap mo hinh         : {model_path}")
    yolo = YOLO(model_path)
    model = yolo.model.to(device).eval()
    
    # 1. Thuc hien chuyen doi tai tham so hoa RepConv (neu co)
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
    
    # Tao Tensor dau vao gia lap (1, 3, 640, 640)
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    
    # 2. Chay khoi dong GPU (Warmup)
    print(f"-> Dang chay Warmup GPU ({warmup} vong)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    # 3. Do dac chinh xac bang CUDA Events (Chuan IEEE)
    print(f"-> Dang do do tre thuan GPU ({iterations} vong lap)...")
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(iterations):
                _ = model(dummy_input)
            end_evt.record()
            torch.cuda.synchronize()
            lat_ms = start_evt.elapsed_time(end_evt) / iterations
        else:
            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = model(dummy_input)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0 / iterations
            
    pure_fps = 1000.0 / lat_ms if lat_ms > 0 else 0.0
    print("------------------------------------------------------------------")
    print("  [KET QUA PURE GPU FORWARD]")
    print(f"  - Pure GPU Latency (T_gpu) : {lat_ms:.2f} ms / khung hinh")
    print(f"  - Pure Throughput (FPS)    : {pure_fps:.1f} FPS")
    print("------------------------------------------------------------------")
    return lat_ms, pure_fps

def benchmark_end_to_end_pipeline(model_path: str, video_path: str, frame_limit: int = 100):
    print("\n==================================================================")
    print("  PHAN 2: DO END-TO-END PIPELINE VIDEO THUC TE (REAL-WORLD RTSP)")
    print("==================================================================")
    
    if not Path(video_path).exists():
        print(f"[Loi] Khong tim thấy video: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Loi] Khong the mo luong video: {video_path}")
        return

    yolo = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("-> Dang xu ly cac khung hinh video hien trường thuc te...")
    frame_count = 0
    t_dec_list, t_prep_list, t_infer_list, t_rend_list, t_total_list = [], [], [], [], []
    
    while cap.isOpened() and frame_count < frame_limit:
        # Buc 1: Decode Frame
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            break
        t_dec = (t1 - t0) * 1000.0
        
        # Buc 2: Preprocess & Letterbox
        t0 = time.perf_counter()
        img_resized = cv2.resize(frame, (640, 640))
        t1 = time.perf_counter()
        t_prep = (t1 - t0) * 1000.0
        
        # Buc 3: Model Inference + NMS
        t0 = time.perf_counter()
        results = yolo.predict(img_resized, imgsz=640, device=device, verbose=False)
        t1 = time.perf_counter()
        t_infer = (t1 - t0) * 1000.0
        
        # Buc 4: Drawing Bounding Boxes & Render
        t0 = time.perf_counter()
        annotated = results[0].plot()
        t1 = time.perf_counter()
        t_rend = (t1 - t0) * 1000.0
        
        t_total = t_dec + t_prep + t_infer + t_rend
        
        # Bo qua 10 khung hinh dau
        if frame_count > 10:
            t_dec_list.append(t_dec)
            t_prep_list.append(t_prep)
            t_infer_list.append(t_infer)
            t_rend_list.append(t_rend)
            t_total_list.append(t_total)
            
        frame_count += 1
        
    cap.release()
    
    avg_dec = np.mean(t_dec_list)
    avg_prep = np.mean(t_prep_list)
    avg_infer = np.mean(t_infer_list)
    avg_rend = np.mean(t_rend_list)
    avg_total = np.mean(t_total_list)
    pipeline_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
    
    print("------------------------------------------------------------------")
    print(f"  BANG CHI SO DO TRE CHI TIET TUNG GIAI DOAN (Trung binh {len(t_total_list)} frames):")
    print(f"  1. Stream Decoding (T_dec)        : {avg_dec:.2f} ms")
    print(f"  2. Preprocessing (T_prep)         : {avg_prep:.2f} ms")
    print(f"  3. Model Forward + NMS (T_gpu+nms): {avg_infer:.2f} ms")
    print(f"  4. Rendering & Display (T_rend)   : {avg_rend:.2f} ms")
    print("  ----------------------------------------------------------------")
    print("  [KET QUA END-TO-END PIPELINE]")
    print(f"  - Total Pipeline Latency (T_total): {avg_total:.2f} ms / khung hinh")
    print(f"  - Real-World Pipeline Throughput  : {pipeline_fps:.1f} FPS")
    print("==================================================================\n")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    model_path = str(base_dir / "exported_engines" / "yolo11s_best_fused_deploy.pt")
    video_path = str(base_dir / "video_test" / "construction_site_workers_1080p.mp4")
    
    benchmark_pure_gpu_latency(model_path)
    benchmark_end_to_end_pipeline(model_path, video_path)
