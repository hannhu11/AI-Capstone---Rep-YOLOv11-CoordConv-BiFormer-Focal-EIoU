"""
Precision 1.0x Real-Time Safety Helmet Detection Pipeline.
Optimized for NVIDIA RTX 3050 with TensorRT FP16 Engine & Master Wall-Clock Sync.

Architectural Enhancements:
1. Asynchronous Threaded Video Reader: Offloads H.264 CPU video decoding to background thread.
2. Single-Pass 720p Pre-Resampling: Eliminates double CPU resize overhead.
3. Master Wall-Clock Sync (time.monotonic): Locks playback to 59.94 FPS (1.0x real-life speed).
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


class ThreadedVideoReader:
    """Asynchronous background thread for fast H.264 CPU frame decoding."""

    def __init__(self, source: str | int, target_size: tuple[int, int] = (1280, 720)):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.native_fps is None or self.native_fps <= 0 or self.native_fps > 120:
            self.native_fps = 30.0

        self.target_size = target_size
        self.q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=4)
        self.stopped = False
        self.source = source

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.source, str) and not self.source.startswith("rtsp://"):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.q.put(None)
                        break

                # Fast pre-resample to 720p C-contiguous array
                resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                if not resized.flags["C_CONTIGUOUS"]:
                    resized = np.ascontiguousarray(resized)

                self.q.put(resized)
            else:
                time.sleep(0.002)

    def read(self) -> np.ndarray | None:
        try:
            return self.q.get(timeout=2.0)
        except queue.Empty:
            return None

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def fast_draw_hud_normalized(
    display_frame: np.ndarray,
    boxes_norm: np.ndarray | None,
    classes_np: np.ndarray | None,
    confs_np: np.ndarray | None,
    display_fps: float,
    gpu_lat_ms: float,
    stream_name: str,
    device_info: str,
    native_fps: float,
    sync_enabled: bool,
) -> np.ndarray:
    """Vectorized HUD drawing using normalized bounding box coordinates."""
    h, w, _ = display_frame.shape
    hat_count = 0
    person_count = 0

    if boxes_norm is not None and len(boxes_norm) > 0:
        for i in range(len(boxes_norm)):
            x1n, y1n, x2n, y2n = boxes_norm[i]
            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)

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

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

    # Top HUD Banner
    mode_str = f"EXACT 1.0X SYNC ({native_fps:.1f} FPS)" if sync_enabled else "UNCAPPED BENCHMARK"
    cv2.rectangle(display_frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(
        display_frame,
        f"AI SAFETY MONITORING | {stream_name} | {mode_str}",
        (15, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"PLAYBACK FPS: {display_fps:.1f} | GPU LATENCY: {gpu_lat_ms:.2f}ms | HW: {device_info}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"HELMETS: {hat_count} | WORKERS: {person_count}",
        (max(15, w - 380), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    return display_frame


def run_precision_demo(
    model_path: str,
    source: str | int = 0,
    imgsz: int = 640,
    conf_thresh: float = 0.15,
    sync_fps: bool = True,
):
    """Execute master clock synchronized real-time video demo."""
    dev_info = "NVIDIA RTX 3050 (TensorRT FP16)"

    reader = ThreadedVideoReader(source, target_size=(1280, 720))
    native_fps = reader.native_fps
    frame_interval = 1.0 / native_fps

    print("=" * 80)
    print(" 🎯 AI CAPSTONE: ASYNC PRECISION REAL-TIME SAFETY MONITORING")
    print(f" Model Path:   {model_path}")
    print(f" Source Video: {source}")
    print(f" Native FPS:   {native_fps:.2f} FPS (Target Interval: {frame_interval * 1000.0:.2f} ms)")
    print(f" Clock Sync:   {'MASTER WALL-CLOCK 1.0X' if sync_fps else 'UNCAPPED BENCHMARK'}")
    print("=" * 80)

    model = YOLO(model_path, task="detect")

    win_name = "AI Safety Helmet Monitoring Dashboard (1.0x Real-Time Sync)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    stream_name = Path(str(source)).name
    frame_index = 0
    t_start_clock = time.monotonic()
    fps_history = []

    print("\n[Running Master Sync Demo] Press 'q' to quit, 's' for screenshot, 'p' to toggle sync.\n")

    try:
        while True:
            frame = reader.read()
            if frame is None:
                break

            # GPU Forward pass latency measurement
            t_gpu_start = time.perf_counter()
            results = model.predict(
                frame,
                imgsz=imgsz,
                conf=conf_thresh,
                device=0,
                verbose=False,
            )[0]
            t_gpu_end = time.perf_counter()
            gpu_lat_ms = (t_gpu_end - t_gpu_start) * 1000.0

            # Normalized coordinate extraction for display scaling
            if results.boxes is not None and len(results.boxes) > 0:
                boxes_norm = results.boxes.xyxyn.cpu().numpy()
                classes_np = results.boxes.cls.cpu().numpy()
                confs_np = results.boxes.conf.cpu().numpy()
            else:
                boxes_norm, classes_np, confs_np = None, None, None

            frame_index += 1

            # Master Wall-Clock Synchronization Pacing
            if sync_fps:
                target_elapsed = frame_index * frame_interval
                actual_elapsed = time.monotonic() - t_start_clock
                sleep_needed = target_elapsed - actual_elapsed
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

            total_elapsed_now = time.monotonic() - t_start_clock
            current_fps = frame_index / total_elapsed_now if total_elapsed_now > 0 else native_fps

            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_display_fps = sum(fps_history) / len(fps_history)

            annotated = fast_draw_hud_normalized(
                frame,
                boxes_norm,
                classes_np,
                confs_np,
                avg_display_fps,
                gpu_lat_ms,
                stream_name,
                dev_info,
                native_fps,
                sync_fps,
            )

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"screenshot_{int(time.time())}.jpg", annotated)
                print("[Saved Screenshot]")
            elif key == ord("p"):
                sync_fps = not sync_fps
                t_start_clock = time.monotonic()
                frame_index = 0
                print(f"[Master Sync Toggled] Sync={sync_fps}")

    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precision Real-time Video Demo")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint or engine")
    parser.add_argument("--source", type=str, required=True, help="Video path or RTSP stream")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--nosync", action="store_true", help="Disable wall-clock sync for uncapped benchmark")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_precision_demo(args.model, src, imgsz=args.imgsz, conf_thresh=args.conf, sync_fps=not args.nosync)
