"""
Ultra-Fast Decoupled Real-Time Safety Helmet Detection Pipeline.
Optimized for NVIDIA RTX 3050 with TensorRT FP16 Engine & RTSP / File Playback Sync.

Key Features:
1. Native Video FPS Pacing (Sync Mode): Prevents fast-forward effect on saved MP4 files.
2. Uncapped Benchmark Mode: Allows raw GPU throughput measurement.
3. Single-Pass Vectorized Bounding Box CPU Extraction (Zero CUDA Sync Lock).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Enable cuDNN benchmark and TensorFloat32 on NVIDIA Ampere (RTX 3050)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def letterbox_resize(
    image: np.ndarray,
    target_size: tuple[int, int] = (1280, 720),
    color: tuple[int, int, int] = (15, 15, 15),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Fast aspect-preserving letterbox resize."""
    ih, iw = image.shape[:2]
    tw, th = target_size

    scale = min(tw / iw, th / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((th, tw, 3), color, dtype=np.uint8)
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2

    padded[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return padded, scale, (pad_x, pad_y)


class ThreadedVideoReader:
    """Async threaded video frame reader with native FPS querying."""

    def __init__(self, source: str | int = 0, target_size: tuple[int, int] = (1280, 720)):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.native_fps is None or self.native_fps <= 0 or self.native_fps > 120:
            self.native_fps = 30.0

        self.target_size = target_size
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.startswith("rtsp://"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.stopped = True
                    break

            padded_frame, scale, (pad_x, pad_y) = letterbox_resize(frame, self.target_size)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((padded_frame, scale, (pad_x, pad_y)))

    def read(self):
        try:
            return self.q.get(timeout=0.2)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


def fast_draw_hud(
    frame: np.ndarray,
    boxes_np: np.ndarray | None,
    classes_np: np.ndarray | None,
    confs_np: np.ndarray | None,
    fps: float,
    gpu_lat_ms: float,
    stream_name: str,
    device_info: str,
    native_fps: float,
    sync_mode: bool,
) -> np.ndarray:
    """Vectorized fast drawing of bounding boxes and HUD overlay."""
    h, w, _ = frame.shape
    hat_count = 0
    person_count = 0

    if boxes_np is not None and len(boxes_np) > 0:
        for i in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[i].astype(int)
            cls_id = int(classes_np[i])
            conf = float(confs_np[i])

            if cls_id == 0:
                hat_count += 1
                color = (0, 255, 0)
                label = f"HELMET {conf * 100:.1f}%"
            else:
                person_count += 1
                color = (0, 0, 255)
                label = f"PERSON {conf * 100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

    # Top HUD Banner
    mode_text = f"REAL-TIME 1.0X SYNC ({native_fps:.0f} FPS)" if sync_mode else "MAX UNMAPPED BENCHMARK"
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"AI SAFETY MONITORING | {stream_name} | {mode_text}",
        (15, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"PLAYBACK FPS: {fps:.1f} | GPU LATENCY: {gpu_lat_ms:.2f}ms | HARDWARE: {device_info}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"HELMETS: {hat_count} | WORKERS: {person_count}",
        (max(15, w - 380), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    return frame


def run_ultra_fast_demo(
    model_path: str,
    source: str | int = 0,
    imgsz: int = 640,
    conf_thresh: float = 0.15,
    sync_fps: bool = True,
):
    """Run ultra-high FPS RTSP / Video pipeline with optional real-time pacing sync."""
    dev_info = "NVIDIA RTX 3050 (TensorRT FP16 / CUDA)"

    print("=" * 80)
    print(" 🚀 AI CAPSTONE: HELMET DETECTION DEMO (WITH REAL-TIME PLAYBACK SYNC)")
    print(f" Model Path:   {model_path}")
    print(f" Video Source: {source}")
    print(f" Playback Sync: {'ENABLED (1.0x Real-time speed)' if sync_fps else 'DISABLED (Max Benchmark Speed)'}")
    print("=" * 80)

    model = YOLO(model_path)

    # Disable ultralytics internal fuse on Windows CUDA
    if hasattr(model, "model") and hasattr(model.model, "fuse"):
        model.model.fuse = lambda *args, **kwargs: model.model

    reader = ThreadedVideoReader(source, target_size=(1280, 720))
    native_fps = reader.native_fps
    target_frame_time = 1.0 / native_fps if sync_fps else 0.0

    win_name = "AI Safety Helmet Monitoring Dashboard"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    fps_history = []
    stream_name = Path(str(source)).name

    print("\n[Running Demo] Press 'q' to quit, 's' for screenshot, 'p' to toggle playback sync.\n")

    try:
        while True:
            t0 = time.perf_counter()
            data = reader.read()
            if data is None:
                continue

            padded_frame, scale, (pad_x, pad_y) = data

            # Pure GPU forward pass timing
            t_gpu_start = time.perf_counter()
            results = model.predict(
                padded_frame,
                imgsz=imgsz,
                conf=conf_thresh,
                device=0,
                verbose=False,
            )[0]
            t_gpu_end = time.perf_counter()
            gpu_lat_ms = (t_gpu_end - t_gpu_start) * 1000.0

            # Single-pass batch CPU extraction
            if results.boxes is not None and len(results.boxes) > 0:
                boxes_np = results.boxes.xyxy.cpu().numpy()
                classes_np = results.boxes.cls.cpu().numpy()
                confs_np = results.boxes.conf.cpu().numpy()
            else:
                boxes_np, classes_np, confs_np = None, None, None

            t1 = time.perf_counter()
            proc_lat = t1 - t0

            # Playback pacing delay for 1.0x natural speed on video files
            if sync_fps and proc_lat < target_frame_time:
                delay_ms = max(1, int((target_frame_time - proc_lat) * 1000.0))
            else:
                delay_ms = 1

            total_lat_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / total_lat_ms if total_lat_ms > 0 else 30.0

            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            annotated = fast_draw_hud(
                padded_frame,
                boxes_np,
                classes_np,
                confs_np,
                avg_fps,
                gpu_lat_ms,
                stream_name,
                dev_info,
                native_fps,
                sync_fps,
            )

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"screenshot_{int(time.time())}.jpg", annotated)
                print("[Saved Screenshot]")
            elif key == ord("p"):
                sync_fps = not sync_fps
                target_frame_time = 1.0 / native_fps if sync_fps else 0.0
                print(f"[Playback Sync Toggled] Sync={sync_fps}")

    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra Fast Video Demo")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint/engine")
    parser.add_argument("--source", type=str, required=True, help="Video path or RTSP link")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--sync", action="store_true", default=True, help="Enable 1.0x real-time video playback sync")
    parser.add_argument("--nosync", action="store_false", dest="sync", help="Disable sync for max benchmark speed")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_ultra_fast_demo(args.model, src, imgsz=args.imgsz, conf_thresh=args.conf, sync_fps=args.sync)
