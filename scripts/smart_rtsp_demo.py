"""
Smart RTSP / Video Demo Pipeline with Native CUDA FP16 Acceleration & Aspect Preservation.

Fixes:
1. Native FP16 (Half Precision) CUDA execution on NVIDIA RTX 3050 to resolve cuDNN stream mismatch crash.
2. Aspect-Preserving Letterboxing (Preserves Vertical / Portrait Videos).
3. Ultra-stable multi-frame playback loop.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def letterbox_resize(
    image: np.ndarray,
    target_size: tuple[int, int] = (1280, 720),
    color: tuple[int, int, int] = (15, 15, 15),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with letterboxing to preserve aspect ratio without stretching."""
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


class AspectPreservingStreamReader:
    """Multi-threaded video reader that applies aspect-preserving letterboxing."""

    def __init__(self, source: str | int = 0, display_size: tuple[int, int] = (1280, 720)):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.display_size = display_size

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

            padded_frame, scale, (pad_x, pad_y) = letterbox_resize(frame, self.display_size)

            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((padded_frame, frame, scale, (pad_x, pad_y)))

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


def draw_hud_and_detections(
    frame: np.ndarray,
    results,
    fps: float,
    latency_ms: float,
    stream_name: str,
    device_info: str,
) -> tuple[np.ndarray, int, int]:
    """Draw bounding boxes and clean HUD panel."""
    h, w, _ = frame.shape
    hat_count = 0
    person_count = 0

    boxes = results[0].boxes if len(results) > 0 else []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            label_name = results[0].names[cls_id]

            if label_name == "hat" or cls_id == 0:
                hat_count += 1
                color = (0, 230, 0)
                label_text = f"HELMET {conf*100:.1f}%"
            else:
                person_count += 1
                color = (0, 0, 230)
                label_text = f"PERSON {conf*100:.1f}%"

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            (tw, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (xyxy[0], max(0, xyxy[1] - th_text - 6)), (xyxy[0] + tw + 6, max(th_text + 6, xyxy[1])), color, -1)
            cv2.putText(frame, label_text, (xyxy[0] + 3, max(th_text, xyxy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Top Banner HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"AI SAFETY MONITORING | {stream_name}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Hardware: {device_info}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"HELMETS: {hat_count} | WORKERS: {person_count}", (max(15, w - 380), 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, hat_count, person_count


def run_smart_demo(
    model_path: str,
    source: str | int = 0,
    imgsz: int = 640,
    conf_thresh: float = 0.15,
    device: str = "cuda",
    save_output: str | None = None,
):
    """Execute real-time video demo with aspect preservation and PyTorch CUDA FP16 acceleration."""
    cuda_avail = torch.cuda.is_available()
    dev_target = 0 if cuda_avail else "cpu"
    use_half = cuda_avail
    dev_info = "NVIDIA RTX 3050 (CUDA FP16 Tensor Cores)" if cuda_avail else "CPU Execution"

    print("=" * 70)
    print(" AI CAPSTONE: SMART REAL-TIME HELMET DETECTION DEMO")
    print(f" Model Path:   {model_path}")
    print(f" Video Source: {source}")
    print(f" Execution:    {dev_info}")
    print("=" * 70)

    model = YOLO(model_path)
    if hasattr(model, "model") and hasattr(model.model, "fuse"):
        model.model.fuse = lambda *args, **kwargs: model.model

    reader = AspectPreservingStreamReader(source, display_size=(1280, 720))

    win_name = "Smart AI Safety Helmet Monitoring Dashboard"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, reader.fps, (1280, 720))

    fps_history = []
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()
            data = reader.read()
            if data is None:
                continue

            padded_frame, raw_frame, scale, (pad_x, pad_y) = data

            # Predict on letterboxed frame with CUDA FP16
            results = model.predict(
                padded_frame, imgsz=imgsz, conf=conf_thresh, device=dev_target, half=use_half, verbose=False
            )
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 30.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            annotated, hats, persons = draw_hud_and_detections(
                padded_frame, results, avg_fps, latency_ms, Path(str(source)).name, dev_info
            )

            if writer:
                writer.write(annotated)

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ss_name = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(ss_name, annotated)
                print(f"[Saved Screenshot]: {ss_name}")

            frame_count += 1
    finally:
        reader.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Real-Time Video Demo")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--source", type=str, required=True, help="Video path or RTSP link")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--save-output", type=str, default=None, help="Save MP4 path")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_smart_demo(args.model, src, imgsz=args.imgsz, conf_thresh=args.conf, device=args.device, save_output=args.save_output)
