"""
src/engine/spatial_safety.py
============================
Spatial Relational Safety Compliance & Debounce Engine.
Evaluates geometric containment and intersection between tracked workers,
helmets (with color identification), and safety vests.
Features:
- Sub-region decomposition: Head sub-box (top 35%) and Torso sub-box (15%-75%).
- Helmet color extraction (blue, green, red, white, yellow).
- Debounce filter: requires N consecutive violating frames before triggering alerts.
- Anti-spam cooldown: prevents repeated alerts for the same worker.
- Violation snapshot generation with visual overlay and metadata watermark.
"""

from __future__ import annotations
import cv2
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import config, SNAPSHOTS_DIR
from src.engine.bytetrack import STrack

# Helmet class mappings
HELMET_COLOR_MAP = {
    2: "blue",
    3: "green",
    4: "red",
    5: "white",
    6: "yellow",
}

# In case classes are string names
HELMET_NAME_TO_COLOR = {
    "blue_helmet": "blue",
    "green_helmet": "green",
    "red_helmet": "red",
    "white_helmet": "white",
    "yellow_helmet": "yellow",
    "blue": "blue",
    "green": "green",
    "red": "red",
    "white": "white",
    "yellow": "yellow",
}


def bbox_iou(b1: List[float] | np.ndarray, b2: List[float] | np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter + 1e-7
    return inter / union


# Explicit negative class indicators (must NEVER be treated as valid PPE)
NEGATIVE_HELMET_CLASSES = {
    "no_helmet", "none", "no-helmet", "head", "without_helmet", "no_hat", "head_without_helmet", "none_helmet"
}
NEGATIVE_VEST_CLASSES = {
    "no_vest", "none_vest", "no-vest", "without_vest", "no_safety_vest", "none_safety_vest"
}

def is_negative_helmet(name: str) -> bool:
    name_l = name.lower()
    return name_l in NEGATIVE_HELMET_CLASSES or name_l.startswith("no_helmet") or name_l.startswith("no-helmet") or name_l == "none"

def is_negative_vest(name: str) -> bool:
    name_l = name.lower()
    return name_l in NEGATIVE_VEST_CLASSES or name_l.startswith("no_vest") or name_l.startswith("no-vest")


class SpatialSafetyEngine:
    """Evaluates safety compliance for tracked workers in real time."""

    def __init__(
        self,
        head_box_ratio: float = 0.35,
        torso_min_ratio: float = 0.15,
        torso_max_ratio: float = 0.75,
        helmet_overlap_thresh: float = 0.15,
        vest_overlap_thresh: float = 0.20,
        consecutive_frames_thresh: int = 8,
        cooldown_seconds: float = 30.0,
    ):
        self.head_box_ratio = head_box_ratio
        self.torso_min_ratio = torso_min_ratio
        self.torso_max_ratio = torso_max_ratio
        self.helmet_overlap_thresh = helmet_overlap_thresh
        self.vest_overlap_thresh = vest_overlap_thresh
        self.consecutive_frames_thresh = consecutive_frames_thresh
        self.cooldown_seconds = cooldown_seconds

    def evaluate_frame(
        self,
        tracks: List[STrack],
        raw_detections: List[Dict[str, Any]],
        frame_bgr: np.ndarray,
        camera_id: str = "CCTV_GATE_01",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluate compliance for all active tracked workers.
        Returns:
            worker_reports: compliance state for all tracked workers
            alert_events: newly confirmed violation alert payloads (passed debounce and cooldown)
        """
        now = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Partition raw detections into Positive Helmets, Negative Helmets, Positive Vests, Negative Vests
        positive_helmets = []
        negative_helmets = []
        positive_vests = []
        negative_vests = []

        for d in raw_detections:
            cls_id = d.get("cls", -1)
            name = str(d.get("name", "")).lower().strip()
            box = d.get("box", [0, 0, 0, 0])
            conf = d.get("conf", 1.0)

            # Check for explicit negative indicators first
            if is_negative_helmet(name):
                negative_helmets.append({"box": box, "cls": cls_id, "name": name, "conf": conf})
                continue

            if is_negative_vest(name):
                negative_vests.append({"box": box, "cls": cls_id, "name": name, "conf": conf})
                continue

            # Check positive helmet
            is_pos_helmet = (
                cls_id in HELMET_COLOR_MAP
                or name in HELMET_NAME_TO_COLOR
                or any(c in name for c in ("blue", "green", "red", "white", "yellow"))
                or ("helmet" in name and not any(neg in name for neg in ("no_", "no-", "without", "none")))
            )
            if is_pos_helmet:
                color = HELMET_COLOR_MAP.get(cls_id) or HELMET_NAME_TO_COLOR.get(name)
                if not color:
                    for c in ("blue", "green", "red", "white", "yellow"):
                        if c in name:
                            color = c
                            break
                color = color or "unknown"
                positive_helmets.append({"box": box, "cls": cls_id, "name": name, "color": color, "conf": conf})
                continue

            # Check positive safety vest
            is_pos_vest = (
                cls_id == 1
                or ("vest" in name and not any(neg in name for neg in ("no_", "no-", "without", "none")))
            )
            if is_pos_vest:
                positive_vests.append({"box": box, "cls": cls_id, "name": name, "conf": conf})

        worker_reports: List[Dict[str, Any]] = []
        alert_events: List[Dict[str, Any]] = []

        for track in tracks:
            px1, py1, px2, py2 = [float(v) for v in track.tlbr]
            pw = max(0.0, px2 - px1)
            ph = max(0.0, py2 - py1)

            if pw <= 2.0 or ph <= 2.0:
                continue

            # 1. Sub-regions
            head_box = [px1, py1, px2, py1 + self.head_box_ratio * ph]
            torso_box = [px1, py1 + self.torso_min_ratio * ph, px2, py1 + self.torso_max_ratio * ph]

            # 2. Check for explicit Negative Helmet (no_helmet / none) on head
            has_explicit_no_helmet = False
            for nh in negative_helmets:
                hb = nh["box"]
                hcx = (hb[0] + hb[2]) / 2.0
                hcy = (hb[1] + hb[3]) / 2.0
                if (head_box[0] <= hcx <= head_box[2] and head_box[1] <= hcy <= head_box[3]) or bbox_iou(head_box, hb) >= 0.10:
                    has_explicit_no_helmet = True
                    break

            # 3. Helmet Match & Color
            matched_helmet = None
            if not has_explicit_no_helmet:
                for h in positive_helmets:
                    hb = h["box"]
                    hx1, hy1, hx2, hy2 = hb[:4]
                    hcx = (hx1 + hx2) / 2.0
                    hcy = (hy1 + hy2) / 2.0

                    if (head_box[0] <= hcx <= head_box[2]) and (head_box[1] <= hcy <= head_box[3]):
                        matched_helmet = h
                        break
                    elif bbox_iou(head_box, hb) >= self.helmet_overlap_thresh:
                        matched_helmet = h
                        break

            has_helmet = (matched_helmet is not None and not has_explicit_no_helmet)
            helmet_color = matched_helmet["color"] if has_helmet else None

            # 4. Check for explicit Negative Vest (no_vest) on torso
            has_explicit_no_vest = False
            for nv in negative_vests:
                vb = nv["box"]
                vcx = (vb[0] + vb[2]) / 2.0
                vcy = (vb[1] + vb[3]) / 2.0
                if (torso_box[0] <= vcx <= torso_box[2] and torso_box[1] <= vcy <= torso_box[3]) or bbox_iou(torso_box, vb) >= 0.10:
                    has_explicit_no_vest = True
                    break

            # 5. Safety Vest Match
            matched_vest = None
            if not has_explicit_no_vest:
                for v in positive_vests:
                    vb = v["box"]
                    if bbox_iou(torso_box, vb) >= self.vest_overlap_thresh or bbox_iou([px1, py1, px2, py2], vb) >= 0.25:
                        matched_vest = v
                        break

            has_vest = (matched_vest is not None and not has_explicit_no_vest)

            # 6. Violations classification
            violations = []
            if not has_helmet:
                violations.append("MISSING_SAFETY_HELMET")
            if not has_vest:
                violations.append("MISSING_SAFETY_VEST")

            is_compliant = (len(violations) == 0)
            status_str = "COMPLIANT" if is_compliant else "VIOLATION"

            # 7. Debounce streak update
            if is_compliant:
                track.compliance_streak += 1
                if track.compliance_streak >= 5:
                    track.violation_streak = 0
                    track.is_alerted = False
            else:
                track.violation_streak += 1
                track.compliance_streak = 0

            track.last_helmet_color = helmet_color
            track.last_violation_types = violations

            # Report data structure
            report = {
                "track_id": track.track_id,
                "bbox": [round(px1, 1), round(py1, 1), round(px2, 1), round(py2, 1)],
                "confidence": round(float(track.score), 3),
                "has_helmet": has_helmet,
                "helmet_color": helmet_color,
                "has_safety_vest": has_vest,
                "status": status_str,
                "violations": violations,
                "violation_streak": track.violation_streak,
                "compliance_streak": track.compliance_streak,
                "is_alerted": track.is_alerted,
            }
            worker_reports.append(report)

            # 8. Check if Alert should trigger (Debounce + Cooldown)
            should_alert = (
                not is_compliant
                and track.violation_streak >= self.consecutive_frames_thresh
                and (not track.is_alerted or (now - track.last_alert_timestamp) >= self.cooldown_seconds)
            )

            if should_alert:
                track.is_alerted = True
                track.last_alert_timestamp = now

                # Generate high-res snapshot with bounding box overlay and person crop
                snapshot_filename, snapshot_path, crop_filename, crop_path = self._generate_violation_snapshot(
                    frame_bgr, report, camera_id, now_str
                )

                alert_event = {
                    "alert_id": f"ALERT-W{track.track_id}-{int(now)}",
                    "camera_id": camera_id,
                    "worker_id": track.track_id,
                    "timestamp": now_str,
                    "violations": violations,
                    "has_helmet": has_helmet,
                    "helmet_color": helmet_color,
                    "has_vest": has_vest,
                    "confidence": report["confidence"],
                    "violation_duration_frames": track.violation_streak,
                    "snapshot_filename": snapshot_filename,
                    "snapshot_path": str(snapshot_path),
                    "crop_filename": crop_filename,
                    "crop_path": str(crop_path),
                    "bbox": report["bbox"],
                }
                alert_events.append(alert_event)

        return worker_reports, alert_events

    def _generate_violation_snapshot(
        self,
        frame_bgr: np.ndarray,
        report: Dict[str, Any],
        camera_id: str,
        timestamp_str: str,
    ) -> Tuple[str, Path, str, Path]:
        """Draws visual alert cues, extracts worker crop, embeds PiP inset, and saves snapshot + crop to disk."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in report["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # 1. Extract standalone person crop
        worker_crop = frame_bgr[y1:y2, x1:x2].copy()
        timestamp_ms = int(time.time() * 1000)
        crop_filename = f"crop_worker_{report['track_id']}_{timestamp_ms}.jpg"
        crop_path = SNAPSHOTS_DIR / crop_filename

        if worker_crop.size > 0 and worker_crop.shape[0] > 10 and worker_crop.shape[1] > 10:
            cv2.imwrite(str(crop_path), worker_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            # Fallback if box coordinates are degenerate
            worker_crop = vis.copy()
            cv2.imwrite(str(crop_path), worker_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 2. Draw red warning box around worker in main frame
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw Corner accents
        d = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        c_color = (0, 255, 255)
        # Top-left
        cv2.line(vis, (x1, y1), (x1 + d, y1), c_color, 4)
        cv2.line(vis, (x1, y1), (x1, y1 + d), c_color, 4)
        # Top-right
        cv2.line(vis, (x2, y1), (x2 - d, y1), c_color, 4)
        cv2.line(vis, (x2, y1), (x2, y1 + d), c_color, 4)
        # Bottom-left
        cv2.line(vis, (x1, y2), (x1 + d, y2), c_color, 4)
        cv2.line(vis, (x1, y2), (x1, y2 - d), c_color, 4)
        # Bottom-right
        cv2.line(vis, (x2, y2), (x2 - d, y2), c_color, 4)
        cv2.line(vis, (x2, y2), (x2, y2 - d), c_color, 4)

        # 3. Embed Picture-in-Picture (PiP) Zoom-In Person Crop in Top-Right
        if worker_crop.size > 0 and worker_crop.shape[0] > 10 and worker_crop.shape[1] > 10:
            pip_target_h = min(220, max(120, h // 4))
            aspect = worker_crop.shape[1] / max(1, worker_crop.shape[0])
            pip_target_w = int(pip_target_h * aspect)
            pip_target_w = max(100, min(pip_target_w, w // 4))

            resized_crop = cv2.resize(worker_crop, (pip_target_w, pip_target_h), interpolation=cv2.INTER_LINEAR)
            
            # Position at top-right with margin
            margin = 15
            pip_x1 = w - pip_target_w - margin
            pip_y1 = margin + 10
            pip_x2 = pip_x1 + pip_target_w
            pip_y2 = pip_y1 + pip_target_h

            if pip_x1 > 0 and pip_y1 > 0 and pip_x2 < w and pip_y2 < h:
                # Background container and shadow
                cv2.rectangle(vis, (pip_x1 - 3, pip_y1 - 22), (pip_x2 + 3, pip_y2 + 3), (15, 15, 25), -1)
                cv2.rectangle(vis, (pip_x1 - 3, pip_y1 - 22), (pip_x2 + 3, pip_y2 + 3), (0, 0, 255), 2)
                # Inset image
                vis[pip_y1:pip_y2, pip_x1:pip_x2] = resized_crop
                # PiP Title Tag
                pip_tag = f"ZOOM CROP: #{report['track_id']}"
                cv2.putText(vis, pip_tag, (pip_x1 + 4, pip_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        # 4. Overlay HUD banner at bottom
        banner_h = 75
        cv2.rectangle(vis, (0, h - banner_h), (w, h), (15, 15, 25), -1)
        cv2.line(vis, (0, h - banner_h), (w, h - banner_h), (0, 0, 255), 2)

        violations_text = " | ".join([v.replace("_", " ") for v in report["violations"]])
        line1 = f"CRITICAL SAFETY VIOLATION | Camera: {camera_id} | {timestamp_str}"
        line2 = f"Worker ID: #{report['track_id']} (Conf: {report['confidence']*100:.1f}%) | Violations: {violations_text}"

        cv2.putText(vis, line1, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line2, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

        # 5. Save snapshot file
        filename = f"violation_worker_{report['track_id']}_{timestamp_ms}.jpg"
        save_path = SNAPSHOTS_DIR / filename
        cv2.imwrite(str(save_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return filename, save_path, crop_filename, crop_path
