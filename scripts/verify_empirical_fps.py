"""
Comprehensive Performance Benchmark Suite for Rep-YOLO11s (Unbuffered).
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def log(msg):
    print(msg, flush=True)

def run_pure_gpu_benchmark(model_path: str, imgsz: int = 640, iterations: int = 200, warmup: int = 50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"\n==================================================================")
    log(f"  PART 1: PURE GPU MODEL INFERENCE LATENCY BENCHMARK ({device.upper()})")
    log(f"==================================================================")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        log(f"GPU Device            : {gpu_name}")
    
    log(f"Loading Model Weights : {model_path}")
    yolo = YOLO(model_path)
    model = yolo.model.to(device).eval()
    
    # Fuse RepConv if available
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
    
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    
    log(f"Running Warmup ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    log(f"Measuring Pure Forward Latency ({iterations} iterations)...")
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
    log(f"------------------------------------------------------------------")
    log(f"  [RESULT] Pure GPU Forward Latency (T_gpu) : {lat_ms:.2f} ms")
    log(f"  [RESULT] Pure Model Throughput             : {pure_fps:.1f} FPS")
    log(f"------------------------------------------------------------------")
    return lat_ms, pure_fps

def run_end_to_end_pipeline_benchmark(model_path: str, video_path: str, frame_limit: int = 100):
    log(f"\n==================================================================")
    log(f"  PART 2: END-TO-END RTSP VIDEO PIPELINE BENCHMARK (REAL-WORLD)")
    log(f"==================================================================")
    
    if not Path(video_path).exists():
        log(f"[Error] Video file not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"[Error] Cannot open video stream: {video_path}")
        return

    yolo = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log(f"Processing video frames for real-world pipeline timing...")
    frame_count = 0
    t_dec_list, t_prep_list, t_infer_list, t_rend_list, t_total_list = [], [], [], [], []
    
    while cap.isOpened() and frame_count < frame_limit:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            break
        t_dec = (t1 - t0) * 1000.0
        
        t0 = time.perf_counter()
        img_resized = cv2.resize(frame, (640, 640))
        t1 = time.perf_counter()
        t_prep = (t1 - t0) * 1000.0
        
        t0 = time.perf_counter()
        results = yolo.predict(img_resized, imgsz=640, device=device, verbose=False)
        t1 = time.perf_counter()
        t_infer = (t1 - t0) * 1000.0
        
        t0 = time.perf_counter()
        annotated = results[0].plot()
        t1 = time.perf_counter()
        t_rend = (t1 - t0) * 1000.0
        
        t_total = t_dec + t_prep + t_infer + t_rend
        
        if frame_count > 10:  # Ignore initial warmup frames
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
    
    log(f"------------------------------------------------------------------")
    log(f"  STAGE LATENCY BREAKDOWN (Average over {len(t_total_list)} steady frames):")
    log(f"  1. Stream Decoding (T_dec)        : {avg_dec:.2f} ms")
    log(f"  2. Preprocessing (T_prep)         : {avg_prep:.2f} ms")
    log(f"  3. Model Inference + NMS (T_gpu)  : {avg_infer:.2f} ms")
    log(f"  4. Rendering & GUI Overlay (T_rend): {avg_rend:.2f} ms")
    log(f"  ----------------------------------------------------------------")
    log(f"  [RESULT] Total End-to-End Pipeline Latency (T_total): {avg_total:.2f} ms")
    log(f"  [RESULT] Real-World Pipeline Throughput            : {pipeline_fps:.1f} FPS")
    log(f"==================================================================\n")

if __name__ == "__main__":
    model_path = "exported_engines/yolo11s_best_fused_deploy.pt"
    video_path = "video_test/construction_site_workers_1080p.mp4"
    
    run_pure_gpu_benchmark(model_path)
    run_end_to_end_pipeline_benchmark(model_path, video_path)
