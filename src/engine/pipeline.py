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
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator

from src.config import config
from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.spatial_safety import SpatialSafetyEngine
from src.alerting.alert_manager import alert_manager


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
        )
        self.safety_engine = SpatialSafetyEngine(
            head_box_ratio=config.head_box_ratio,
            torso_min_ratio=config.torso_min_ratio,
            torso_max_ratio=config.torso_max_ratio,
            helmet_overlap_thresh=config.helmet_overlap_thresh,
            vest_overlap_thresh=config.vest_overlap_thresh,
            consecutive_frames_thresh=config.consecutive_violation_frames,
            cooldown_seconds=config.alert_cooldown_seconds,
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self._init_video_capture()

        # Telemetry state
        self.fps_rolling = 0.0
        self.inference_latency_ms = 0.0
        self.total_latency_ms = 0.0
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
            self._init_video_capture()
            self.active_workers_count = 0
            self.compliant_workers_count = 0
            self.active_violations_count = 0
            self.compliance_rate = 100.0
            self.total_frames_processed = 0
            alert_manager.clear_alerts()
            print(f"📹 Switched Video Source to: {new_source} (Clean Slate Active)")

    def _stream_worker(self):
        """Dedicated background loop that continuously runs inference and encodes frames."""
        while self.is_running:
            t0 = time.perf_counter()
            try:
                frame = self.process_next_frame()
                if frame is not None:
                    # Optimize resolution for web streaming (960px width, preserving aspect ratio)
                    h, w = frame.shape[:2]
                    target_w = 960
                    if w > target_w:
                        target_h = int(h * (target_w / w))
                        display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        display_frame = frame

                    ret, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        jpeg_bytes = buffer.tobytes()
                        with self.frame_condition:
                            self.latest_encoded_jpeg = jpeg_bytes
                            self.frame_idx += 1
                            self.frame_condition.notify_all()
            except Exception as e:
                print(f"⚠️ Stream worker exception: {e}")
                time.sleep(0.05)

            # Pace to ~30 FPS
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
                if not ret:
                    time.sleep(0.03)
                    return None

        # 1. Run Object Detection
        detections, person_dets, infer_ms = self.detector.predict(frame)
        self.inference_latency_ms = infer_ms

        # 2. Multi-Object Tracking on Workers
        active_tracks = self.tracker.update(person_dets)

        # 3. Spatial Relational Safety Analysis & Debouncing
        worker_reports, alert_events = self.safety_engine.evaluate_frame(
            active_tracks, detections, frame, camera_id=config.camera_id
        )

        # 4. Enqueue Confirmed Violation Alerts to Dispatcher
        for alert in alert_events:
            alert_manager.enqueue_alert(alert)

        # 5. Aggregate Real-Time Telemetry
        with self.lock:
            self.active_workers_count = len(worker_reports)
            self.compliant_workers_count = sum(1 for r in worker_reports if r["status"] == "COMPLIANT")
            self.active_violations_count = self.active_workers_count - self.compliant_workers_count
            self.compliance_rate = (
                (self.compliant_workers_count / self.active_workers_count * 100.0)
                if self.active_workers_count > 0
                else 100.0
            )

            # Update helmet color distribution
            color_counts = {"blue": 0, "green": 0, "red": 0, "white": 0, "yellow": 0}
            for d in detections:
                c_name = d["name"]
                for c in color_counts.keys():
                    if c in c_name:
                        color_counts[c] += 1
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
            }


# Global singleton pipeline instance
cctv_pipeline = CCTVSurveillancePipeline()
