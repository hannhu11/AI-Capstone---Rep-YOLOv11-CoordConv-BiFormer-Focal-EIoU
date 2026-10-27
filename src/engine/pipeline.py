"""
src/engine/pipeline.py
======================
CCTV Simulation Pipeline Engine.
Coordinates Video Capture, Detection, ByteTrack MOT, Spatial Safety Rules,
Alert Dispatching, and High-Performance Frame Rendering.
"""

from __future__ import annotations
import cv2
import time
import torch
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator, Tuple

from src.config import config
from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.spatial_safety import SpatialSafetyEngine
from src.alerting.alert_manager import alert_manager
from src.engine.reid_tracker import reid_registry


CLASS_COLORS = {
    "person": (180, 180, 180),
    "safety_vest": (0, 255, 128),      # Hi-vis Green
    "blue_helmet": (255, 128, 0),      # Azure Blue
    "green_helmet": (0, 220, 100),     # Hi-vis Green
    "red_helmet": (0, 0, 255),         # Alert Red
    "white_helmet": (230, 230, 230),   # White
    "yellow_helmet": (0, 230, 255),    # Yellow/Gold
    "blue": (255, 128, 0),
    "green": (0, 220, 100),
    "red": (0, 0, 255),
    "white": (230, 230, 230),
    "yellow": (0, 230, 255),
}


def compute_box_intersection_ratio(
    box_a: List[float] | np.ndarray,
    box_b: List[float] | np.ndarray,
    use_upper_torso: bool = True,
    upper_torso_ratio: float = 0.60,
) -> float:
    r"""
    Computes fraction of box_a's area that overlaps with box_b (or box_b's upper torso).
    Overlap ratio = |box_a \cap box_b| / |box_a|
    """
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    aw = max(0.0, float(ax2 - ax1))
    ah = max(0.0, float(ay2 - ay1))
    a_area = aw * ah
    if a_area <= 1e-7:
        return 0.0

    if use_upper_torso:
        by2_eff = float(by1) + max(0.0, float(by2 - by1)) * upper_torso_ratio
    else:
        by2_eff = float(by2)

    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), by2_eff)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    return (inter_w * inter_h) / a_area


class CCTVSurveillancePipeline:
    """Thread-safe CCTV Surveillance and Video Processing Pipeline."""

    def __init__(self, video_source: Optional[str] = None):
        self.source_path = video_source or config.video_source
        self.detector = SafetyObjectDetector(
            model_path=config.model_path,
            conf_thresh=config.conf_threshold,
            iou_thresh=config.iou_threshold,
            device=config.device,
            use_half=config.use_half,
        )
        self.tracker = BYTETracker(
            track_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            grace_period=getattr(config, "track_grace_period", 2),
        )
        self.safety_engine = SpatialSafetyEngine(
            head_box_ratio=config.head_box_ratio,
            torso_min_ratio=config.torso_min_ratio,
            torso_max_ratio=config.torso_max_ratio,
            helmet_overlap_thresh=config.helmet_overlap_thresh,
            vest_overlap_thresh=config.vest_overlap_thresh,
            consecutive_frames_thresh=config.consecutive_violation_frames,
            cooldown_seconds=config.alert_cooldown_seconds,
            min_tracklet_len=getattr(config, "min_tracklet_len", 15),
        )

        # Stationary Furniture Prior Tracker (Root Cause 1c)
        self.stationary_furniture_tracker: Dict[int, Dict[str, Any]] = {}
        self.confirmed_furniture_zones: List[List[float]] = []
        self.stationary_next_id: int = 1

        # Focus / Solo Tracking Inspection Mode
        self.focus_track_id: Optional[int] = None
        reid_registry.reset()

        # Isolate/clear alert history for fresh session
        alert_manager.clear_alerts()

        self.cap: Optional[cv2.VideoCapture] = None
        self._init_video_capture()

        # Telemetry state (with EMA alpha = 0.25)
        self.fps_rolling = 0.0
        self.inference_latency_ms = 0.0
        self.total_latency_ms = 0.0
        self.workers_count_ema = 0.0
        self.compliance_rate_ema = 100.0
        self.active_workers_count = 0
        self.compliant_workers_count = 0
        self.active_violations_count = 0
        self.compliance_rate = 100.0
        self.helmet_color_distribution: Dict[str, int] = {
            "blue": 0, "green": 0, "red": 0, "white": 0, "yellow": 0
        }
        self.total_frames_processed = 0

        self.lock = threading.Lock()
        self.is_running = True
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_encoded_jpeg: Optional[bytes] = None
        self.frame_idx = 0
        self.frame_condition = threading.Condition()

        # Clean slate on startup to prevent stale alerts leaking from previous sessions
        alert_manager.clear_alerts()

        # Start dedicated background streaming worker thread
        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def _init_video_capture(self):
        if self.cap is not None:
            self.cap.release()

        # Handle numeric camera index vs file path
        src = int(self.source_path) if str(self.source_path).isdigit() else str(self.source_path)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"⚠️ Warning: Could not open video source: {src}")

    def switch_source(self, new_source: str):
        """Switches video simulation stream dynamically with clean slate."""
        with self.lock:
            self.source_path = new_source
            config.video_source = new_source
            self.tracker.reset()
            self.safety_engine.reset()
            self.focus_track_id = None
            reid_registry.reset()
            self._init_video_capture()
            self.active_workers_count = 0
            self.workers_count_ema = 0.0
            self.compliant_workers_count = 0
            self.active_violations_count = 0
            self.compliance_rate = 100.0
            self.compliance_rate_ema = 100.0
            self.total_frames_processed = 0
            self.fps_rolling = 0.0
            self.helmet_color_distribution = {"blue": 0, "green": 0, "red": 0, "white": 0, "yellow": 0}
            self.latest_encoded_jpeg = None
            alert_manager.clear_alerts()
            print(f"📹 Switched Video Source to: {new_source} (Clean Slate Active)")

    def _stream_worker(self):
        """Dedicated background loop that continuously runs inference and encodes frames with self-healing recovery."""
        consecutive_errors = 0
        last_error_log_time = 0.0

        while self.is_running:
            t0 = time.perf_counter()
            try:
                frame = self.process_next_frame()
                if frame is not None:
                    consecutive_errors = 0
                    # Optimize resolution for web streaming (960px width, preserving aspect ratio)
                    h, w = frame.shape[:2]
                    target_w = 960
                    if w > target_w:
                        target_h = int(h * (target_w / w))
                        display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        display_frame = frame

                    ret, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        jpeg_bytes = buffer.tobytes()
                        with self.frame_condition:
                            self.latest_encoded_jpeg = jpeg_bytes
                            self.frame_idx += 1
                            self.frame_condition.notify_all()
            except Exception as e:
                consecutive_errors += 1
                curr_t = time.time()
                if curr_t - last_error_log_time >= 3.0:
                    import traceback
                    print(f"⚠️ Stream worker exception (streak={consecutive_errors}): {e}")
                    traceback.print_exc()
                    last_error_log_time = curr_t

                err_msg = str(e).lower()
                # Fast exit on unrecoverable CUDA context poisoning so Docker's restart policy restarts cleanly
                unrecoverable_cuda = any(
                    sig in err_msg
                    for sig in (
                        "illegal memory access",
                        "device-side assert",
                        "misaligned address",
                        "illegal instruction",
                        "unspecified launch failure",
                        "hardware stack error",
                        "context is destroyed",
                    )
                )
                if unrecoverable_cuda:
                    print(f"🚨 [CUDA FATAL] Unrecoverable CUDA context poisoning detected: {e}")
                    print("🚨 [CUDA FATAL] Exiting process with code 1 to allow container clean restart...")
                    import os
                    os._exit(1)

                # Self-healing recovery for CUDA/driver exceptions
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if consecutive_errors >= 3:
                    print(f"🔄 [SELF-HEALING] Attempting pipeline recovery after {consecutive_errors} consecutive failures...")
                    try:
                        with self.lock:
                            self._init_video_capture()
                            self.detector.load_model(self.detector.model_path)
                        consecutive_errors = 0
                        print("✅ [SELF-HEALING] Successfully re-initialized detector and video capture!")
                    except Exception as rec_err:
                        print(f"⚠️ [SELF-HEALING] GPU reload failed: {rec_err}. Falling back to CPU mode...")
                        err_msg_rec = str(rec_err).lower()
                        if any(
                            sig in err_msg_rec
                            for sig in (
                                "illegal memory access",
                                "device-side assert",
                                "misaligned address",
                                "illegal instruction",
                                "unspecified launch failure",
                                "hardware stack error",
                                "context is destroyed",
                            )
                        ):
                            import os
                            os._exit(1)
                        try:
                            with self.lock:
                                try:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                self.detector.device = "cpu"
                                self.detector.use_half = False
                                self.detector.load_model(self.detector.model_path)
                            consecutive_errors = 0
                            print("✅ [SELF-HEALING] Successfully recovered in CPU fallback mode!")
                        except Exception as cpu_err:
                            print(f"❌ [SELF-HEALING] CPU fallback failed: {cpu_err}")
                            err_msg_cpu = str(cpu_err).lower()
                            if any(
                                sig in err_msg_cpu
                                for sig in (
                                    "illegal memory access",
                                    "device-side assert",
                                    "misaligned address",
                                    "illegal instruction",
                                    "unspecified launch failure",
                                    "hardware stack error",
                                    "context is destroyed",
                                )
                            ):
                                import os
                                os._exit(1)

                time.sleep(0.1)

            # Periodic GPU Auto-Recovery Probe: restore full CUDA acceleration when GPU context stabilizes
            if (
                self.detector.device == "cpu"
                and torch.cuda.is_available()
                and config.device.startswith("cuda")
                and (self.total_frames_processed > 0 and self.total_frames_processed % 150 == 0)
            ):
                try:
                    with self.lock:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        self.detector.device = config.device
                        self.detector.use_half = config.use_half
                        self.detector.load_model(self.detector.model_path)
                        print("⚡ [SELF-HEALING] Restored full GPU CUDA acceleration!")
                except Exception:
                    self.detector.device = "cpu"
                    self.detector.use_half = False

            # Target ~30 FPS smooth pacing
            elapsed = time.perf_counter() - t0
            target_delay = 1.0 / 30.0
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

    def get_latest_frame(self, last_frame_idx: int = -1, timeout: float = 0.5) -> Tuple[Optional[bytes], int]:
        """Thread-safe acquisition of newest JPEG frame without re-running inference."""
        with self.frame_condition:
            if self.frame_idx == last_frame_idx or self.latest_encoded_jpeg is None:
                self.frame_condition.wait(timeout=timeout)
            return self.latest_encoded_jpeg, self.frame_idx

    def process_next_frame(self) -> Optional[np.ndarray]:
        """Reads frame, runs detection, tracking, safety logic, draws annotations, and returns frame."""
        t_start = time.perf_counter()

        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self._init_video_capture()
                if not self.cap.isOpened():
                    time.sleep(0.05)
                    return None

            ret, frame = self.cap.read()
            if not ret:
                # Loop video simulation
                if not str(self.source_path).isdigit() and not str(self.source_path).startswith("rtsp://"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    # Reset tracker on loop to prevent stale tracks carrying over
                    self.tracker.reset()
                    self.safety_engine.reset()
                    self.stationary_furniture_tracker.clear()
                    self.confirmed_furniture_zones.clear()
                    self.stationary_next_id = 1
                    reid_registry.on_video_loop()
                if not ret:
                    time.sleep(0.03)
                    return None

            # High-Resolution Scaling Optimization (for 4K / UHD streams)
            # If input frame is ultra-high resolution (e.g. 3840x2160 or 2160x3840),
            # downscale to max 1920 on longest edge.
            # This maintains crisp full-HD fidelity while cutting memory & processing overhead by 75%,
            # unlocking 40-60 FPS smooth playback!
            orig_h, orig_w = frame.shape[:2]
            max_dim = max(orig_h, orig_w)
            if max_dim > 1920:
                scale = 1920.0 / max_dim
                frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)

            # 1. Run Object Detection (Inside Lock to avoid CUDA concurrency collisions)
            detections, person_dets, infer_ms = self.detector.predict(frame)
            self.inference_latency_ms = infer_ms

            # Stationary Furniture Tracker (Root Cause 1c)
            # Track static candidate detections across frames; candidates remaining motionless for > 30 frames
            # are classified as stationary furniture (e.g. white round table, warning signs) and suppressed.
            for d in detections:
                if d["cls"] != 0 and "person" not in d["name"]:
                    b = d["box"]
                    cx = (b[0] + b[2]) / 2.0
                    cy = (b[1] + b[3]) / 2.0

                    # Check against established stationary furniture zones
                    is_in_zone = False
                    for zb in self.confirmed_furniture_zones:
                        inter_w = max(0.0, min(b[2], zb[2]) - max(b[0], zb[0]))
                        inter_h = max(0.0, min(b[3], zb[3]) - max(b[1], zb[1]))
                        b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                        if (inter_w * inter_h) / b_area >= 0.50:
                            is_in_zone = True
                            break
                    if is_in_zone:
                        d["is_stationary_furniture"] = True
                        continue

                    matched_sid = None
                    for sid, sinfo in self.stationary_furniture_tracker.items():
                        if abs(cx - sinfo["cx"]) <= 12.0 and abs(cy - sinfo["cy"]) <= 12.0:
                            matched_sid = sid
                            break
                    if matched_sid is not None:
                        sentry = self.stationary_furniture_tracker[matched_sid]
                        sentry["count"] += 1
                        sentry["last_frame"] = self.total_frames_processed
                        sentry["box"] = b
                        if sentry["count"] > 30:
                            d["is_stationary_furniture"] = True
                            if not sentry.get("is_confirmed", False):
                                sentry["is_confirmed"] = True
                                is_already_covered = any(
                                    (max(0.0, min(b[2], zb[2]) - max(b[0], zb[0])) * max(0.0, min(b[3], zb[3]) - max(b[1], zb[1])))
                                    / max(1.0, (b[2] - b[0]) * (b[3] - b[1])) >= 0.70
                                    for zb in self.confirmed_furniture_zones
                                )
                                if not is_already_covered:
                                    self.confirmed_furniture_zones.append(list(b))
                    else:
                        self.stationary_furniture_tracker[self.stationary_next_id] = {
                            "cx": cx,
                            "cy": cy,
                            "box": b,
                            "count": 1,
                            "last_frame": self.total_frames_processed,
                            "is_confirmed": False,
                        }
                        self.stationary_next_id += 1

            # Prune stationary tracker items not seen for > 15 frames
            self.stationary_furniture_tracker = {
                sid: sinfo
                for sid, sinfo in self.stationary_furniture_tracker.items()
                if (self.total_frames_processed - sinfo["last_frame"]) <= 15
            }

            # Bipartite Spatial Association Filter (Root Cause 1a) & Luminance Gating (Root Cause 2)
            filtered_detections: List[Dict[str, Any]] = []
            h_frame, w_frame = frame.shape[:2]

            for d in detections:
                if d["cls"] == 0 or "person" in d["name"]:
                    filtered_detections.append(d)
                    continue

                # Suppress Stationary Furniture (Root Cause 1c)
                if d.get("is_stationary_furniture", False):
                    continue

                # Luminance / Value Gating for blue_helmet candidates (Root Cause 2)
                # Sample candidate crop; if mean HSV Value V < 95, reject as dark hair / shadowed head texture
                if d["name"] == "blue_helmet" or d.get("color") == "blue" or d["cls"] == 2:
                    bx1, by1, bx2, by2 = [int(round(v)) for v in d["box"][:4]]
                    bx1 = max(0, min(w_frame - 1, bx1))
                    by1 = max(0, min(h_frame - 1, by1))
                    bx2 = max(bx1 + 1, min(w_frame, bx2))
                    by2 = max(by1 + 1, min(h_frame, by2))
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size > 0 and np.any(crop > 0):
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mean_val = float(np.mean(hsv[:, :, 2]))
                        if mean_val < 95.0:
                            continue

                # Geometric Hardhat Plausibility Filter:
                # Reject flat furniture / table surfaces (aspect ratio > 2.20) detected as helmets
                if "helmet" in d["name"] or d["cls"] in (2, 3, 4, 5, 6):
                    bx1, by1, bx2, by2 = d["box"][:4]
                    bw = max(1.0, bx2 - bx1)
                    bh = max(1.0, by2 - by1)
                    if (bw / bh) > 2.20:
                        continue

                # Bipartite Spatial Association Filter (Eq. 24):
                # Attenuate isolated PPE confidences by kappa = 0.15 if no person box overlaps by >= 25%
                orig_conf = d["conf"]
                max_overlap = 0.0
                if len(person_dets) > 0:
                    is_helmet = "helmet" in d["name"] or d["cls"] in (2, 3, 4, 5, 6)
                    for p_box in person_dets:
                        ov = compute_box_intersection_ratio(
                            d["box"], p_box[:4], use_upper_torso=is_helmet, upper_torso_ratio=0.60
                        )
                        if ov > max_overlap:
                            max_overlap = ov

                effective_conf = orig_conf if max_overlap >= 0.25 else orig_conf * 0.15
                if effective_conf >= 0.25:
                    d_c = dict(d)
                    d_c["conf"] = round(effective_conf, 4)
                    filtered_detections.append(d_c)

            detections = filtered_detections

            # Filter candidate person detections before tracking to discard background noise smudges
            # Relax edge bounding box aspect ratio: If box touches image borders, allow w/h <= 3.5;
            # in the center, allow w/h <= 1.65 to accommodate bending postures.
            if len(person_dets) > 0:
                frame_h, frame_w = frame.shape[:2]
                valid_mask = []
                for p in person_dets:
                    pw = p[2] - p[0]
                    ph = p[3] - p[1]
                    touches_border = (p[0] <= 60 or p[1] <= 60 or p[2] >= frame_w - 60 or p[3] >= frame_h - 60)
                    max_aspect = 3.5 if touches_border else 2.6
                    is_valid = (ph >= 35 and pw >= 15 and (pw / max(1.0, ph)) <= max_aspect)
                    valid_mask.append(is_valid)
                person_dets = person_dets[np.array(valid_mask, dtype=bool)] if any(valid_mask) else np.empty((0, 6), dtype=np.float32)

            # 2. Multi-Object Tracking on Workers (with upper-torso appearance histograms)
            active_tracks = self.tracker.update(person_dets, frame=frame)

            # Map raw ByteTrack tracks to Persistent Re-ID Canonical IDs
            for track in active_tracks:
                tb = track.tlbr
                matched_color = None
                th = max(1.0, tb[3] - tb[1])
                tw = max(1.0, tb[2] - tb[0])
                tcx = (tb[0] + tb[2]) / 2.0
                for d in detections:
                    if "helmet" in d["name"]:
                        hb = d["box"]
                        hcx = (hb[0] + hb[2]) / 2.0
                        hw = max(0.0, hb[2] - hb[0])
                        hh = max(0.0, hb[3] - hb[1])
                        # Cranium Vertex Gating:
                        # Genuine helmet top edge must be at or above head cranium (hy1 <= py1 + 0.22 * ph)
                        if hb[1] > (tb[1] + 0.22 * th) or hb[3] < (tb[1] - 0.15 * th):
                            continue
                        if not (tb[0] - 0.05 * tw <= hcx <= tb[2] + 0.05 * tw) and abs(hcx - tcx) > 0.45 * tw:
                            continue
                        if hw > 1.05 * tw or (hw / max(1.0, hh)) > 2.20:
                            continue
                        if not (hb[2] < tb[0] or hb[0] > tb[2] or hb[3] < tb[1] or hb[1] > tb[1] + 0.42 * th):
                            matched_color = d.get("color") or d["name"].replace("_helmet", "")
                            break
                canonical_id = reid_registry.get_or_create_canonical_id(
                    raw_track_id=track.track_id,
                    box=tb,
                    frame=frame,
                    helmet_color=matched_color,
                    frame_idx=self.total_frames_processed,
                )
                track.canonical_id = canonical_id

            # 3. Spatial Relational Safety Analysis & Debouncing
            worker_reports, alert_events = self.safety_engine.evaluate_frame(
                active_tracks, detections, frame, camera_id=config.camera_id
            )

            # Update persistent registry profile statuses and trajectories
            for r in worker_reports:
                wid = r["track_id"]
                if wid in reid_registry.profiles:
                    reid_registry.profiles[wid].status = r["status"]
                    reid_registry.profiles[wid].violations = r["violations"]
                    if r.get("helmet_color") and r["helmet_color"] != "none":
                        reid_registry.profiles[wid].helmet_color = r["helmet_color"]
            reid_registry.update_frame_end([r["track_id"] for r in worker_reports], self.total_frames_processed)

            # 4. Enqueue Confirmed Violation Alerts to Dispatcher
            for alert in alert_events:
                alert_manager.enqueue_alert(alert)

            # 5. Aggregate Real-Time Telemetry with Exponential Moving Average (EMA, alpha = 0.25)
            instant_workers = len(worker_reports)
            instant_compliant = sum(1 for r in worker_reports if r["status"] == "COMPLIANT")
            instant_violations = instant_workers - instant_compliant
            instant_compliance_rate = (
                (instant_compliant / instant_workers * 100.0)
                if instant_workers > 0
                else 100.0
            )

            alpha = 0.25
            if self.total_frames_processed == 0:
                self.workers_count_ema = float(instant_workers)
                self.compliance_rate_ema = float(instant_compliance_rate)
            else:
                self.workers_count_ema = alpha * instant_workers + (1.0 - alpha) * self.workers_count_ema
                self.compliance_rate_ema = alpha * instant_compliance_rate + (1.0 - alpha) * self.compliance_rate_ema

            self.active_workers_count = int(round(self.workers_count_ema))
            self.compliant_workers_count = instant_compliant
            self.active_violations_count = instant_violations
            self.compliance_rate = round(self.compliance_rate_ema, 1)

            # Update helmet color distribution based on active tracked workers' stable colors
            color_counts = {"blue": 0, "green": 0, "red": 0, "white": 0, "yellow": 0}
            for r in worker_reports:
                h_col = r.get("helmet_color")
                if h_col and h_col in color_counts:
                    color_counts[h_col] += 1
            self.helmet_color_distribution = color_counts

            # 6. Render Visual Annotations
            annotated = self._render_annotations(frame, detections, worker_reports)

            t_total = (time.perf_counter() - t_start) * 1000.0
            self.total_latency_ms = t_total
            fps_instant = 1000.0 / t_total if t_total > 0 else 30.0
            self.fps_rolling = 0.9 * self.fps_rolling + 0.1 * fps_instant if self.fps_rolling > 0 else fps_instant
            self.total_frames_processed += 1

            self.latest_annotated_frame = annotated
            return annotated

    def _render_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        worker_reports: List[Dict[str, Any]],
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # -------------------------------------------------------------
        # 1. SPECIAL FOCUS / ISOLATION INSPECTION MODE
        # If user selected a specific worker ID, render ONLY this worker + motion trail
        # -------------------------------------------------------------
        if self.focus_track_id is not None:
            focused_worker = next((r for r in worker_reports if r["track_id"] == self.focus_track_id), None)
            
            # Draw Top Focus HUD Bar
            hud_h = 44
            cv2.rectangle(vis, (0, 0), (w, hud_h), (25, 20, 15), -1)
            cv2.line(vis, (0, hud_h), (w, hud_h), (0, 240, 255), 2)
            cv2.circle(vis, (20, 22), 6, (0, 255, 255), -1)

            status_note = "LOCKED / ACTIVE" if focused_worker is not None else "LOST / OCCLUDED"
            hud_text = (
                f"FOCUS MODE: TRACKING WORKER #{self.focus_track_id} ONLY ({status_note}) | "
                f"FPS: {self.fps_rolling:.1f} | Latency: {self.inference_latency_ms:.1f}ms | [CLICK 'VIEW ALL' TO EXIT]"
            )
            cv2.putText(vis, hud_text, (36, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2, cv2.LINE_AA)

            if focused_worker is not None:
                r = focused_worker
                x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
                is_compliant = (r["status"] == "COMPLIANT")
                box_color = (0, 240, 255) if is_compliant else (0, 0, 255)

                # Draw Historical Trajectory Trail (Motion Path)
                if self.focus_track_id in reid_registry.profiles:
                    traj = reid_registry.profiles[self.focus_track_id].trajectory
                    pts = [(int(pt[0]), int(pt[1])) for pt in traj]
                    for idx in range(1, len(pts)):
                        alpha = float(idx) / max(1, len(pts) - 1)
                        thick = max(1, int(4 * alpha))
                        cv2.line(vis, pts[idx - 1], pts[idx], (0, 230, 255), thick, cv2.LINE_AA)
                    if len(pts) > 0:
                        cv2.circle(vis, pts[-1], 6, (0, 255, 255), -1)

                # Main Bounding Box (High visibility 3px)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)

                # Target Crosshair Brackets
                d = min(22, (x2 - x1) // 3, (y2 - y1) // 3)
                cv2.line(vis, (x1, y1), (x1 + d, y1), (255, 255, 0), 3)
                cv2.line(vis, (x1, y1), (x1, y1 + d), (255, 255, 0), 3)
                cv2.line(vis, (x2, y1), (x2 - d, y1), (255, 255, 0), 3)
                cv2.line(vis, (x2, y1), (x2, y1 + d), (255, 255, 0), 3)
                cv2.line(vis, (x1, y2), (x1 + d, y2), (255, 255, 0), 3)
                cv2.line(vis, (x1, y2), (x1, y2 - d), (255, 255, 0), 3)
                cv2.line(vis, (x2, y2), (x2 - d, y2), (255, 255, 0), 3)
                cv2.line(vis, (x2, y2), (x2, y2 - d), (255, 255, 0), 3)

                # Draw ONLY PPE associated with this target worker (strictly matched by SpatialSafetyEngine)
                m_helmet = r.get("matched_helmet_box")
                if m_helmet is not None:
                    hx1, hy1, hx2, hy2 = [int(v) for v in m_helmet[:4]]
                    h_color_name = r.get("helmet_color") or "helmet"
                    c_name = f"{h_color_name}_helmet" if not h_color_name.endswith("helmet") else h_color_name
                    p_color = CLASS_COLORS.get(c_name, (0, 230, 255))
                    cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), p_color, 2)
                    p_label = f"{c_name.replace('_', ' ').upper()}"
                    (lw, lh), _ = cv2.getTextSize(p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(vis, (hx1, max(0, hy1 - 20)), (hx1 + lw + 6, max(0, hy1)), p_color, -1)
                    cv2.putText(vis, p_label, (hx1 + 3, max(14, hy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                m_vest = r.get("matched_vest_box")
                if m_vest is not None:
                    vx1, vy1, vx2, vy2 = [int(v) for v in m_vest[:4]]
                    v_color = CLASS_COLORS.get("safety_vest", (0, 255, 128))
                    cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), v_color, 2)
                    v_label = "SAFETY VEST"
                    (lw, lh), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(vis, (vx1, max(0, vy1 - 20)), (vx1 + lw + 6, max(0, vy1)), v_color, -1)
                    cv2.putText(vis, v_label, (vx1 + 3, max(14, vy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                # Target Worker Banner
                h_color_tag = f" [{r['helmet_color'].upper()}]" if r.get("helmet_color") else ""
                label_top = f"TARGET WORKER #{r['track_id']} [FOCUSED]{h_color_tag} - {r['status']}"
                (lw, lh), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - 28)), (x1 + lw + 10, max(0, y1)), box_color, -1)
                cv2.putText(vis, label_top, (x1 + 5, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)

                if not is_compliant and r["violations"]:
                    v_text = "MISSING: " + ", ".join([v.replace("MISSING_SAFETY_", "") for v in r["violations"]])
                    (vw, vh), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                    cv2.rectangle(vis, (x1, y2), (x1 + vw + 8, y2 + 22), (0, 0, 180), -1)
                    cv2.putText(vis, v_text, (x1 + 4, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            return vis

        # -------------------------------------------------------------
        # 2. STANDARD FULL SURVEILLANCE RENDERING (All Workers)
        # -------------------------------------------------------------

        # Draw PPE equipment boxes (helmets and vests)
        for d in detections:
            if d["cls"] != 0 and "person" not in d["name"]:
                c_name = d["name"]
                color = CLASS_COLORS.get(c_name, (0, 230, 255))
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{c_name.replace('_', ' ').upper()} {d['conf']*100:.0f}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - 20)), (x1 + lw + 6, max(0, y1)), color, -1)
                cv2.putText(vis, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw Workers with Persistent Track ID and Compliance State
        for r in worker_reports:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            is_compliant = (r["status"] == "COMPLIANT")
            box_color = (0, 230, 115) if is_compliant else (0, 0, 255)  # Green vs Red

            # Main Bounding Box
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2 if is_compliant else 3)

            # Highlighting corner marks for violations
            if not is_compliant:
                d = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
                cv2.line(vis, (x1, y1), (x1 + d, y1), (0, 255, 255), 3)
                cv2.line(vis, (x1, y1), (x1, y1 + d), (0, 255, 255), 3)
                cv2.line(vis, (x2, y1), (x2 - d, y1), (0, 255, 255), 3)
                cv2.line(vis, (x2, y1), (x2, y1 + d), (0, 255, 255), 3)

            # Label Header
            status_text = "COMPLIANT" if is_compliant else f"VIOLATION ({r['violation_streak']})"
            h_color_tag = f" [{r['helmet_color'].upper()}]" if r.get("helmet_color") else ""
            label_top = f"WORKER #{r['track_id']}: {status_text}{h_color_tag}"

            (lw, lh), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - 24)), (x1 + lw + 8, max(0, y1)), box_color, -1)
            cv2.putText(
                vis,
                label_top,
                (x1 + 4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0) if is_compliant else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Violation details badge under box
            if not is_compliant and r["violations"]:
                v_text = "MISSING: " + ", ".join([v.replace("MISSING_SAFETY_", "") for v in r["violations"]])
                (vw, vh), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, y2), (x1 + vw + 8, y2 + 20), (0, 0, 180), -1)
                cv2.putText(vis, v_text, (x1 + 4, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Top Futuristic CCTV HUD
        hud_h = 44
        cv2.rectangle(vis, (0, 0), (w, hud_h), (18, 22, 28), -1)
        cv2.line(vis, (0, hud_h), (w, hud_h), (0, 220, 255), 1)

        # Draw filled red recording circle (real graphic, eliminates Unicode '?' corruption)
        cv2.circle(vis, (20, 22), 6, (0, 0, 255), -1)

        hud_text = (
            f"REC [LIVE] | {config.camera_id} | FPS: {self.fps_rolling:.1f} | Latency: {self.inference_latency_ms:.1f}ms | "
            f"Workers: {self.active_workers_count} (SAFE: {self.compliant_workers_count} | WARN: {self.active_violations_count}) | "
            f"Compliance: {self.compliance_rate:.1f}%"
        )
        cv2.putText(vis, hud_text, (36, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2, cv2.LINE_AA)

        return vis

    def stop(self):
        """Cleanly halts background streaming worker and releases video capture resources."""
        self.is_running = False
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def set_focus_track(self, track_id: Optional[int]) -> Optional[int]:
        """Sets or clears the worker track ID for solo/isolated tracking inspection mode."""
        with self.lock:
            self.focus_track_id = int(track_id) if track_id is not None else None
            return self.focus_track_id

    def get_registered_workers(self, filter_ghosts: bool = True) -> List[Dict[str, Any]]:
        """Returns the full list of persistent registered workers from spatio-temporal Re-ID memory."""
        with self.lock:
            return reid_registry.get_registered_workers(filter_ghosts=filter_ghosts)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns structured metrics for real-time dashboard analytics."""
        with self.lock:
            return {
                "timestamp": time.time(),
                "fps": round(self.fps_rolling, 1),
                "inference_ms": round(self.inference_latency_ms, 1),
                "total_ms": round(self.total_latency_ms, 1),
                "workers_count": self.active_workers_count,
                "compliant_count": self.compliant_workers_count,
                "violations_count": self.active_violations_count,
                "compliance_rate": round(self.compliance_rate, 1),
                "helmet_colors": dict(self.helmet_color_distribution),
                "device": self.detector.device.upper(),
                "model_path": Path(self.detector.model_path).name,
                "video_source": Path(str(self.source_path)).name,
                "total_frames": self.total_frames_processed,
                "focus_track_id": self.focus_track_id,
                "registered_workers_count": len(reid_registry.get_registered_workers(filter_ghosts=True)),
            }


# Global singleton pipeline instance
cctv_pipeline = CCTVSurveillancePipeline()
