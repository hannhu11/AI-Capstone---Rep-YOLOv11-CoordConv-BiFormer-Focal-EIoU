"""
=============================================================================
INDUSTRIAL RTSP SAFETY SURVEILLANCE PIPELINE (IEEE AAIML 2027 COMPLIANT)
Task B3: Bipartite Person-Helmet Spatial Association & Anti-Poster Filter
Author: Nguyen Han Nhu (FPT University)
=============================================================================
This module implements the end-to-end RTSP surveillance pipeline featuring:
1. Multi-threaded aspect-preserving video/stream ingestion.
2. Rep-YOLO11s real-time inference with CUDA FP16 acceleration.
3. Bipartite Person-Helmet Spatial Association Filter (Eq. 24 in paper)
   that completely eliminates 2D safety poster false positives.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


# ============================================================================
# Task B3: Bipartite Spatial Constraint & Poster False Positive Filter
# ============================================================================
def compute_box_intersection_ratio(
    hat_box: Union[np.ndarray, List[float]],
    person_box: Union[np.ndarray, List[float]],
    use_upper_torso: bool = True,
    upper_torso_ratio: float = 0.60,
) -> float:
    r"""
    Compute intersection ratio |B_h \cap B_p^{upper}| / |B_h|.

    Args:
        hat_box: [x1, y1, x2, y2]
        person_box: [x1, y1, x2, y2]
        use_upper_torso: If True, restricts person box to upper body region
        upper_torso_ratio: Fraction of person height to consider as upper torso
    Returns:
        Intersection over Helmet Area ratio in [0.0, 1.0].
    """
    hx1, hy1, hx2, hy2 = hat_box[:4]
    px1, py1, px2, py2 = person_box[:4]

    hw = max(0.0, float(hx2 - hx1))
    hh = max(0.0, float(hy2 - hy1))
    h_area = hw * hh
    if h_area <= 1e-7:
        return 0.0

    # Person upper torso boundary
    if use_upper_torso:
        p_height = max(0.0, float(py2 - py1))
        py2_eff = float(py1) + p_height * upper_torso_ratio
    else:
        py2_eff = float(py2)

    # Intersection box
    ix1 = max(float(hx1), float(px1))
    iy1 = max(float(hy1), float(py1))
    ix2 = min(float(hx2), float(px2))
    iy2 = min(float(hy2), py2_eff)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    return inter_area / h_area


def filter_poster_false_positives(
    hat_boxes: Union[np.ndarray, List[List[float]], torch.Tensor],
    person_boxes: Union[np.ndarray, List[List[float]], torch.Tensor],
    overlap_thresh: float = 0.25,
    kappa: float = 0.15,
    conf_thresh: float = 0.25,
    use_upper_torso: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    r"""
    Bipartite Person-Helmet Association Filter (Eq. 24 in IEEE AAIML 2027 paper).

    Attenuates isolated helmet detections (e.g. from 2D warning posters, safety signs)
    that lack geometric spatial support from a detected human torso:
        Score(B_h) = P(h) if exists B_p : (|B_h \cap B_p^{upper}| / |B_h|) >= overlap_thresh
                     kappa * P(h) otherwise (kappa = 0.15)

    Args:
        hat_boxes: Array or tensor of shape (N, 4+) [x1, y1, x2, y2, conf, (optional cls_id)]
        person_boxes: Array or tensor of shape (M, 4+) [x1, y1, x2, y2, conf, (optional cls_id)]
        overlap_thresh: Minimum required overlap ratio (|B_h \cap B_p| / |B_h| >= 0.25)
        kappa: Attenuation factor for isolated helmet candidates (default 0.15)
        conf_thresh: Minimum reporting threshold after attenuation (default 0.25)
        use_upper_torso: Whether to check overlap against upper torso region (default True)

    Returns:
        valid_hat_boxes: numpy array of filtered/confirmed helmet detections [x1, y1, x2, y2, score, ...]
        associations: list of dicts describing association metadata for each candidate
    """
    # Convert inputs to standard numpy float32 arrays
    if isinstance(hat_boxes, torch.Tensor):
        hat_np = hat_boxes.detach().cpu().numpy().astype(np.float32)
    else:
        hat_np = np.asarray(hat_boxes, dtype=np.float32)

    if isinstance(person_boxes, torch.Tensor):
        person_np = person_boxes.detach().cpu().numpy().astype(np.float32)
    else:
        person_np = np.asarray(person_boxes, dtype=np.float32)

    # Handle empty cases
    if len(hat_np) == 0:
        return np.empty((0, 5), dtype=np.float32), []

    if hat_np.ndim == 1:
        hat_np = hat_np.reshape(1, -1)
    if person_np.ndim == 1 and len(person_np) >= 4:
        person_np = person_np.reshape(1, -1)
    elif len(person_np) == 0:
        person_np = np.empty((0, 4), dtype=np.float32)

    num_hats = len(hat_np)
    num_persons = len(person_np)

    valid_indices = []
    associations: List[Dict[str, Any]] = []

    for i in range(num_hats):
        h_box = hat_np[i]
        orig_score = float(h_box[4]) if len(h_box) > 4 else 1.0

        max_overlap = 0.0
        best_person_idx = -1

        if num_persons > 0:
            for j in range(num_persons):
                p_box = person_np[j]
                overlap = compute_box_intersection_ratio(
                    h_box[:4], p_box[:4], use_upper_torso=use_upper_torso
                )
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_person_idx = j

        # Check bipartite spatial condition
        is_supported = (max_overlap >= overlap_thresh)

        if is_supported:
            effective_score = orig_score
            is_poster = False
        else:
            effective_score = orig_score * kappa
            is_poster = True

        assoc_info = {
            "hat_index": i,
            "hat_box": h_box[:4].tolist(),
            "orig_score": orig_score,
            "effective_score": effective_score,
            "max_overlap": max_overlap,
            "best_person_index": best_person_idx,
            "is_supported": is_supported,
            "is_poster_rejected": is_poster and (effective_score < conf_thresh),
        }
        associations.append(assoc_info)

        # Retain candidate if effective score meets confidence threshold
        if effective_score >= conf_thresh:
            updated_box = h_box.copy()
            if len(updated_box) > 4:
                updated_box[4] = effective_score
            valid_indices.append(updated_box)

    if valid_indices:
        valid_hat_boxes = np.vstack(valid_indices).astype(np.float32)
    else:
        valid_hat_boxes = np.empty((0, hat_np.shape[1] if hat_np.ndim > 1 else 5), dtype=np.float32)

    return valid_hat_boxes, associations


# ============================================================================
# Multi-threaded Video Stream Reader with Aspect-Preserving Letterboxing
# ============================================================================
def letterbox_frame(
    image: np.ndarray,
    target_size: Tuple[int, int] = (1280, 720),
    fill_color: Tuple[int, int, int] = (18, 18, 18),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox resize keeping aspect ratio without stretching."""
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((th, tw, 3), fill_color, dtype=np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    padded[dy : dy + nh, dx : dx + nw] = resized
    return padded, scale, (dx, dy)


class AsyncStreamReader:
    """Threaded RTSP / Video capture with bounded queue to eliminate latency backlog."""
    def __init__(self, source: Union[str, int] = 0, display_size: Tuple[int, int] = (1280, 720)):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot connect to video source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.display_size = display_size
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.startswith("rtsp://"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.stopped = True
                    break
            padded, scale, pad = letterbox_frame(frame, self.display_size)
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put((padded, frame, scale, pad))

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]]:
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


# ============================================================================
# Complete RTSP Pipeline Engine
# ============================================================================
class RTSPIndustrialSurveillanceEngine:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        bipartite_thresh: float = 0.25,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.bipartite_thresh = bipartite_thresh
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.to(self.device).eval()
        if self.device.startswith("cuda"):
            self.model.half()

        print(f"✅ Surveillance Engine initialized on: {self.device.upper()} (FP16={self.device.startswith('cuda')})")

    def process_frame(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
    ) -> Dict[str, Any]:
        """Inference with bipartite poster false-positive elimination."""
        t0 = time.perf_counter()

        # Run model inference
        use_half = self.device.startswith("cuda")
        results = self.yolo.predict(
            frame,
            imgsz=imgsz,
            conf=self.conf_thresh * 0.5,  # pass slight margin for bipartite filter
            iou=self.iou_thresh,
            device=self.device,
            half=use_half,
            verbose=False,
        )

        t_infer = (time.perf_counter() - t0) * 1000.0

        hat_raw = []
        person_raw = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                row = xyxy + [conf, cls_id]
                name = results[0].names.get(cls_id, str(cls_id)).lower()
                if "hat" in name or "helmet" in name or (cls_id == 0 and "person" not in name):
                    hat_raw.append(row)
                else:
                    person_raw.append(row)

        hat_arr = np.array(hat_raw, dtype=np.float32) if hat_raw else np.empty((0, 6), dtype=np.float32)
        person_arr = np.array(person_raw, dtype=np.float32) if person_raw else np.empty((0, 6), dtype=np.float32)

        # Apply Task B3 Bipartite Filter
        filtered_hats, associations = filter_poster_false_positives(
            hat_arr,
            person_arr,
            overlap_thresh=self.bipartite_thresh,
            kappa=0.15,
            conf_thresh=self.conf_thresh,
            use_upper_torso=True,
        )

        t_total = (time.perf_counter() - t0) * 1000.0

        return {
            "hats": filtered_hats,
            "persons": person_arr,
            "raw_hats_count": len(hat_arr),
            "filtered_hats_count": len(filtered_hats),
            "persons_count": len(person_arr),
            "posters_rejected": len(hat_arr) - len(filtered_hats),
            "associations": associations,
            "inference_ms": t_infer,
            "total_ms": t_total,
            "fps": 1000.0 / t_total if t_total > 0 else 0.0,
        }


# ============================================================================
# Standalone Testing and Execution Entrypoint
# ============================================================================
def run_rtsp_demo(
    source: str,
    model_path: str = "exported_engines/yolo11s_best_fused_deploy.pt",
    imgsz: int = 640,
    save_output: Optional[str] = None,
):
    engine = RTSPIndustrialSurveillanceEngine(model_path=model_path)
    reader = AsyncStreamReader(source=source)

    win = "IEEE AAIML 2027: Industrial RTSP Surveillance Dashboard"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, reader.fps, (1280, 720))

    try:
        while True:
            data = reader.read()
            if data is None:
                continue
            padded, raw, scale, pad = data
            out = engine.process_frame(padded, imgsz=imgsz)

            # Draw detections
            vis = padded.copy()
            for p in out["persons"]:
                cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 240), 2)
                cv2.putText(vis, f"WORKER {p[4]*100:.0f}%", (int(p[0]), int(p[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 240), 1)

            for h in out["hats"]:
                cv2.rectangle(vis, (int(h[0]), int(h[1])), (int(h[2]), int(h[3])), (0, 240, 0), 2)
                cv2.putText(vis, f"HELMET {h[4]*100:.0f}%", (int(h[0]), int(h[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 0), 1)

            # HUD
            cv2.rectangle(vis, (0, 0), (1280, 45), (20, 20, 20), -1)
            hud = f"FPS: {out['fps']:.1f} | Latency: {out['total_ms']:.1f}ms | Helmets: {out['filtered_hats_count']} | Workers: {out['persons_count']} | Posters Rejected: {out['posters_rejected']}"
            cv2.putText(vis, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if writer:
                writer.write(vis)

            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial RTSP Surveillance Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source or RTSP URL")
    parser.add_argument("--model", type=str, default="exported_engines/yolo11s_best_fused_deploy.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_rtsp_demo(src, model_path=args.model, imgsz=args.imgsz, save_output=args.save)
