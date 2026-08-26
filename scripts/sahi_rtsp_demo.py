"""
Slicing-Aided Tile Inference Pipeline for High-Resolution Construction CCTV Cameras.

Solves the Small Object Detection Bottleneck (distant safety helmets < 20x20 px)
by slicing 1080p/4K CCTV frames into overlapping 640x640 tiles before NMS fusion.
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


def torchvision_nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Non-Maximum Suppression (NMS) in Pure NumPy for fusing tiled bounding boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def predict_with_sahi_tiles(
    model,
    frame: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.25,
    conf_thresh: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice frame into overlapping tiles, predict on each tile, and map coordinates back to full frame.
    """
    h, w, _ = frame.shape
    stride = int(tile_size * (1.0 - overlap))

    all_boxes = []
    all_scores = []
    all_classes = []

    # Slide across height and width
    for y in range(0, max(1, h - tile_size + stride), stride):
        for x in range(0, max(1, w - tile_size + stride), stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)

            tile = frame[y1:y2, x1:x2]
            results = model.predict(tile, imgsz=tile_size, conf=conf_thresh, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())

                    # Map coordinates back to full frame
                    all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                    all_scores.append(score)
                    all_classes.append(cls_id)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)
    classes_arr = np.array(all_classes, dtype=np.int32)

    # Perform NMS fusion
    keep_indices = torchvision_nms_numpy(boxes_arr, scores_arr, iou_threshold=0.45)
    return boxes_arr[keep_indices], scores_arr[keep_indices], classes_arr[keep_indices]


def run_sahi_demo(
    model_path: str,
    source: str | int = 0,
    imgsz: int = 640,
    conf_thresh: float = 0.15,
    use_sahi: bool = True,
):
    """Run real-time video demo with SAHI tile inference for small object detection."""
    print("=" * 70)
    print(" AI CAPSTONE: SAHI SMALL OBJECT HELMET DETECTION PIPELINE")
    print(f" Model Engine:      {model_path}")
    print(f" SAHI Tile Mode:    {'ENABLED (Slicing 640x640)' if use_sahi else 'DISABLED'}")
    print(f" Confidence Cutoff: {conf_thresh}")
    print("=" * 70)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    win_name = "SAHI High-Precision Helmet Detection Dashboard"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    fps_history = []
    print("\n[Running Demo] Press 'q' to quit, 's' for screenshot.\n")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, w, _ = frame.shape

            if use_sahi and (w > 1280 or h > 720):
                boxes, scores, classes = predict_with_sahi_tiles(
                    model, frame, tile_size=imgsz, overlap=0.25, conf_thresh=conf_thresh
                )
            else:
                results = model.predict(frame, imgsz=imgsz, conf=conf_thresh, verbose=False)[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy().astype(int)
                else:
                    boxes, scores, classes = np.empty((0, 4)), np.empty((0,)), np.empty((0,))

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 30.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Draw Detections
            hat_count = 0
            person_count = 0
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes[i].astype(int)
                conf = scores[i]
                cls_id = classes[i]

                if cls_id == 0:
                    hat_count += 1
                    color = (0, 230, 0)
                    lbl = f"HELMET {conf*100:.1f}%"
                else:
                    person_count += 1
                    color = (0, 0, 230)
                    lbl = f"PERSON {conf*100:.1f}%"

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(th + 6, by1)), color, -1)
                cv2.putText(frame, lbl, (bx1 + 3, max(th, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Header Dashboard
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, f"SAHI SMALL OBJECT MONITORING | {Path(str(source)).name}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f} | Latency: {latency_ms:.1f}ms | Conf: {conf_thresh}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"HELMETS: {hat_count} | WORKERS: {person_count}", (max(15, w - 380), 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"sahi_screenshot_{int(time.time())}.jpg", frame)
                print("[Screenshot Saved]")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI Small Object Detection Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or RTSP stream")
    parser.add_argument("--imgsz", type=int, default=640, help="Tile size")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--no-sahi", action="store_true", help="Disable SAHI tile slicing")
    args = parser.parse_args()

    run_sahi_demo(args.model, args.source, imgsz=args.imgsz, conf_thresh=args.conf, use_sahi=not args.no_sahi)
