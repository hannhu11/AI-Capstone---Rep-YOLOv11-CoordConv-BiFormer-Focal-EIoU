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
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import config, SNAPSHOTS_DIR
from src.engine.bytetrack import STrack, TrackState

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
        border_margin_px: int = 60,
        min_tracklet_len: int = 0,
        min_worker_width: float = 28.0,
        min_worker_height: float = 50.0,
        min_worker_area: float = 1600.0,
    ):
        self.head_box_ratio = head_box_ratio
        self.torso_min_ratio = torso_min_ratio
        self.torso_max_ratio = torso_max_ratio
        self.helmet_overlap_thresh = helmet_overlap_thresh
        self.vest_overlap_thresh = vest_overlap_thresh
        self.consecutive_frames_thresh = consecutive_frames_thresh
        self.cooldown_seconds = cooldown_seconds
        self.border_margin_px = border_margin_px
        self.min_tracklet_len = min_tracklet_len
        self.min_worker_width = min_worker_width
        self.min_worker_height = min_worker_height
        self.min_worker_area = min_worker_area

        # Spatial-Centroid Alert Debouncing (suppress duplicates within radius if track ID switches)
        self.recent_alert_centroids: List[Tuple[float, float, float]] = []  # (norm_cx, norm_cy, timestamp)
        self.spatial_debounce_distance: float = 0.08

        # Background ThreadPoolExecutor for Non-blocking Snapshot Disk I/O
        self.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="snapshot_io")

    def reset(self):
        """Clean-slate reset on video switch or camera reload."""
        self.recent_alert_centroids.clear()

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

        # Prune expired alert centroids older than cooldown_seconds
        self.recent_alert_centroids = [
            (cx, cy, ts) for (cx, cy, ts) in self.recent_alert_centroids
            if (now - ts) < self.cooldown_seconds
        ]

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

                # Luminance / Value Gating on blue_helmet candidates (Root Cause 2):
                # Bare dark hair under cool fluorescent lighting is misclassified as blue_helmet,
                # but has very low luminance (V in [65, 82] out of 255) vs real blue hardhats (V >= 110).
                # If mean HSV Value V < 95, reject it as dark hair / shadowed head texture!
                if color == "blue" and frame_bgr is not None and frame_bgr.size > 0:
                    bx1, by1, bx2, by2 = [int(round(v)) for v in box[:4]]
                    h_img, w_img = frame_bgr.shape[:2]
                    bx1 = max(0, min(w_img - 1, bx1))
                    by1 = max(0, min(h_img - 1, by1))
                    bx2 = max(bx1 + 1, min(w_img, bx2))
                    by2 = max(by1 + 1, min(h_img, by2))
                    crop = frame_bgr[by1:by2, bx1:bx2]
                    if crop.size > 0 and np.any(crop > 0):
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mean_val = float(np.mean(hsv[:, :, 2]))
                        if mean_val < 95.0:
                            continue

                positive_helmets.append({"box": box, "cls": cls_id, "name": name, "color": color, "conf": conf, "is_stationary_furniture": d.get("is_stationary_furniture", False)})
                continue

            # Check positive safety vest (Must NEVER classify helmet colors like 'white' as vest)
            is_pos_vest = (
                ("vest" in name and not any(neg in name for neg in ("no_", "no-", "without", "none")))
                or (cls_id == 1 and not any(hc in name for hc in ("white", "blue", "yellow", "red", "green", "helmet", "none")))
            )
            if is_pos_vest:
                positive_vests.append({"box": box, "cls": cls_id, "name": name, "conf": conf})

        worker_reports: List[Dict[str, Any]] = []
        alert_events: List[Dict[str, Any]] = []
        h_frame, w_frame = frame_bgr.shape[:2]

        # -------------------------------------------------------------
        # HELMET VIOLATION MODE (e.g. GDUT 5-class helmet-only dataset)
        # When no worker body tracks exist (len(tracks) == 0) but helmet / head detections are present:
        # Directly evaluate head-level safety:
        # - Positive helmets (blue, white, yellow, red) -> COMPLIANT
        # - Negative helmets ('none' / no_helmet) -> DIRECT VIOLATION (MISSING_SAFETY_HELMET)
        # -------------------------------------------------------------
        if len(tracks) == 0 and (len(positive_helmets) > 0 or len(negative_helmets) > 0):
            # 1. Process positive helmets as compliant heads
            for idx, h in enumerate(positive_helmets):
                hb = h["box"]
                h_col = h.get("color", "unknown")
                h_conf = float(h.get("conf", 1.0))
                worker_reports.append({
                    "track_id": f"HEAD_P_{idx+1}",
                    "bbox": [round(float(v), 1) for v in hb[:4]],
                    "confidence": round(h_conf, 3),
                    "has_helmet": True,
                    "helmet_color": h_col,
                    "stable_helmet_color": h_col,
                    "has_safety_vest": True,  # Safety vest not part of helmet-only taxonomy
                    "status": "COMPLIANT",
                    "violations": [],
                    "violation_streak": 0,
                    "compliance_streak": 1,
                    "is_alerted": False,
                    "is_at_border": False,
                    "tracklet_len": 1,
                    "matched_helmet_box": hb,
                    "matched_vest_box": None,
                })

            # 2. Process negative helmets ('none' = bare head) as direct violations
            for idx, nh in enumerate(negative_helmets):
                nb = nh["box"]
                n_conf = float(nh.get("conf", 1.0))
                rep_id = f"HEAD_V_{idx+1}"
                v_report = {
                    "track_id": rep_id,
                    "bbox": [round(float(v), 1) for v in nb[:4]],
                    "confidence": round(n_conf, 3),
                    "has_helmet": False,
                    "helmet_color": "none",
                    "stable_helmet_color": "none",
                    "has_safety_vest": True,
                    "status": "VIOLATION",
                    "violations": ["MISSING_SAFETY_HELMET"],
                    "violation_streak": 1,
                    "compliance_streak": 0,
                    "is_alerted": True,
                    "is_at_border": False,
                    "tracklet_len": 1,
                    "matched_helmet_box": None,
                    "matched_vest_box": None,
                }
                worker_reports.append(v_report)

                # Spatial-Centroid Alert Debouncing (prevent duplicate alerts within cooldown & radius)
                nx1, ny1, nx2, ny2 = [float(v) for v in nb[:4]]
                norm_cx = (nx1 + nx2) / (2.0 * max(1, w_frame))
                norm_cy = (ny1 + ny2) / (2.0 * max(1, h_frame))
                is_spatially_debounced = any(
                    np.hypot(norm_cx - rcx, norm_cy - rcy) < self.spatial_debounce_distance
                    for rcx, rcy, _ in self.recent_alert_centroids
                )

                if not is_spatially_debounced and n_conf >= 0.25:
                    self.recent_alert_centroids.append((norm_cx, norm_cy, now))
                    snapshot_filename, snapshot_path, crop_filename, crop_path = self._generate_violation_snapshot(
                        frame_bgr, v_report, camera_id, now_str
                    )
                    alert_events.append({
                        "alert_id": f"ALERT-HEAD-{int(now * 1000) % 1000000}-{idx+1}",
                        "camera_id": camera_id,
                        "worker_id": rep_id,
                        "timestamp": now_str,
                        "violations": ["MISSING_SAFETY_HELMET"],
                        "has_helmet": False,
                        "helmet_color": "none",
                        "has_vest": True,
                        "confidence": v_report["confidence"],
                        "violation_duration_frames": 1,
                        "snapshot_filename": snapshot_filename,
                        "snapshot_path": str(snapshot_path),
                        "crop_filename": crop_filename,
                        "crop_path": str(crop_path),
                        "bbox": v_report["bbox"],
                    })

            return worker_reports, alert_events

        for track in tracks:
            # 1. State check: Reject removed tracks and tracks with time_since_update > 2
            if hasattr(track, "state") and track.state == TrackState.Removed:
                continue
            if getattr(track, "time_since_update", 0) > 2:
                continue
            if getattr(track, "score", 1.0) < 0.20:
                continue

            px1, py1, px2, py2 = [float(v) for v in track.tlbr]
            pw = max(0.0, px2 - px1)
            ph = max(0.0, py2 - py1)

            # Dynamic resolution-aware border margin: max(20, 5% of min dimension)
            border_margin = max(20, int(0.05 * min(h_frame, w_frame)))
            is_at_border = (
                px1 <= border_margin
                or py1 <= border_margin
                or px2 >= (w_frame - border_margin)
                or py2 >= (h_frame - border_margin)
            )

            # 2. Geometry check: Worker box must meet minimum dimensions and valid aspect ratio
            # Relax edge bounding box aspect ratio: If box touches image borders, allow w/h <= 3.5;
            # in the center, allow w/h <= 1.65 to accommodate bending postures.
            if pw < self.min_worker_width or ph < self.min_worker_height or (pw * ph) < self.min_worker_area:
                continue
            max_aspect = 3.5 if is_at_border else 2.6
            if (pw / ph) > max_aspect:
                continue

            is_predicted_frame = getattr(track, "is_predicted", False) or getattr(track, "time_since_update", 0) > 0

            if is_predicted_frame:
                # Maintain previous compliance state during brief prediction grace period
                violations = list(getattr(track, "last_violation_types", []))
                has_helmet = (not any("HELMET" in v for v in violations))
                has_vest = (not any("VEST" in v for v in violations))
                helmet_color = getattr(track, "last_helmet_color", None)
                matched_helmet_box = getattr(track, "last_matched_helmet_box", None)
                matched_vest_box = getattr(track, "last_matched_vest_box", None)
                is_compliant = (len(violations) == 0)
                status_str = "COMPLIANT" if is_compliant else "VIOLATION"
            else:
                # 1. Sub-regions (restored full-width head cranium and torso)
                pcx = (px1 + px2) / 2.0
                head_box = [px1 - 0.05 * pw, py1 - 0.08 * ph, px2 + 0.05 * pw, py1 + self.head_box_ratio * ph]
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
                        if h.get("is_stationary_furniture", False):
                            continue
                        hb = h["box"]
                        hx1, hy1, hx2, hy2 = hb[:4]
                        hcx = (hx1 + hx2) / 2.0
                        hcy = (hy1 + hy2) / 2.0

                        # Cranium Vertex Gating:
                        # Allow hy1 <= py1 + 0.22 * ph (rejecting waist-level table at > 0.35 * ph while preserving real bending heads).
                        if hy1 > (py1 + 0.22 * ph):
                            continue
                        if hy2 < (py1 - 0.15 * ph):
                            continue

                        # Horizontal Head Centering:
                        # Relax cranium horizontal alignment to px1 - 0.05 * pw <= hcx <= px2 + 0.05 * pw (or abs(hcx - pcx) <= 0.45 * pw).
                        if not (px1 - 0.05 * pw <= hcx <= px2 + 0.05 * pw) and abs(hcx - pcx) > 0.45 * pw:
                            continue

                        # Geometric Hardhat Plausibility:
                        # Real helmet width on human head never exceeds person width.
                        # Relax helmet aspect ratio to hw / max(1.0, hh) <= 2.20 for wide-brim hardhats.
                        hw = max(0.0, float(hx2 - hx1))
                        hh = max(0.0, float(hy2 - hy1))
                        if hw > 1.05 * pw:
                            continue
                        if (hw / max(1.0, hh)) > 2.20:
                            continue

                        # Compute intersection over helmet area
                        inter_x = max(0.0, min(head_box[2], hx2) - max(head_box[0], hx1))
                        inter_y = max(0.0, min(head_box[3], hy2) - max(head_box[1], hy1))
                        inter_area = inter_x * inter_y
                        hb_area = max(1.0, (hx2 - hx1) * (hy2 - hy1))
                        io_helmet = inter_area / hb_area

                        if (head_box[0] <= hcx <= head_box[2]) and (head_box[1] <= hcy <= head_box[3]):
                            matched_helmet = h
                            break
                        elif bbox_iou(head_box, hb) >= self.helmet_overlap_thresh or io_helmet >= 0.25:
                            matched_helmet = h
                            break
                        elif (px1 <= hcx <= px2) and (py1 - 0.10 * ph <= hcy <= py1 + 0.42 * ph):
                            matched_helmet = h
                            break

                has_helmet = (matched_helmet is not None and not has_explicit_no_helmet)
                helmet_color = matched_helmet["color"] if has_helmet else None
                matched_helmet_box = matched_helmet["box"] if has_helmet else None
                track.last_matched_helmet_box = matched_helmet_box

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
                        vx1, vy1, vx2, vy2 = vb[:4]
                        inter_vx = max(0.0, min(torso_box[2], vx2) - max(torso_box[0], vx1))
                        inter_vy = max(0.0, min(torso_box[3], vy2) - max(torso_box[1], vy1))
                        inter_varea = inter_vx * inter_vy
                        vb_area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
                        io_vest = inter_varea / vb_area

                        if bbox_iou(torso_box, vb) >= self.vest_overlap_thresh or bbox_iou([px1, py1, px2, py2], vb) >= 0.20 or io_vest >= 0.25:
                            matched_vest = v
                            break

                has_vest = (matched_vest is not None and not has_explicit_no_vest)
                matched_vest_box = matched_vest["box"] if has_vest else None
                track.last_matched_vest_box = matched_vest_box

                # 6. Violations classification
                violations = []
                if not has_helmet:
                    violations.append("MISSING_SAFETY_HELMET")
                if not has_vest:
                    violations.append("MISSING_SAFETY_VEST")

                # Boundary Truncation Protection (Root Cause 3 Fix):
                # 1. Exiting worker protection: If worker was PREVIOUSLY compliant (compliance_streak >= 2) and touches edge
                # 2. Entering worker protection: For new tracklets (tracklet_len <= 8) touching edge
                is_entering = (getattr(track, "tracklet_len", 0) <= 8)
                was_compliant = (getattr(track, "compliance_streak", 0) >= 2)

                if is_at_border and (was_compliant or is_entering):
                    edge_cut_thresh = 5.0
                    # Suppress MISSING_SAFETY_VEST if the TORSO region itself is physically cut by any frame border
                    torso_cut = (
                        torso_box[1] <= edge_cut_thresh
                        or torso_box[3] >= (h_frame - edge_cut_thresh)
                        or py2 >= (h_frame - edge_cut_thresh)
                        or px1 <= edge_cut_thresh
                        or px2 >= (w_frame - edge_cut_thresh)
                    )
                    if "MISSING_SAFETY_VEST" in violations and torso_cut and not has_explicit_no_vest:
                        violations.remove("MISSING_SAFETY_VEST")

                    # Suppress MISSING_SAFETY_HELMET if the HEAD region itself is physically cut by any frame border
                    head_cut = (
                        py1 <= edge_cut_thresh
                        or head_box[3] >= (h_frame - edge_cut_thresh)
                        or px1 <= edge_cut_thresh
                        or px2 >= (w_frame - edge_cut_thresh)
                    )
                    if "MISSING_SAFETY_HELMET" in violations and head_cut and not has_explicit_no_helmet:
                        violations.remove("MISSING_SAFETY_HELMET")

                is_compliant = (len(violations) == 0)
                status_str = "COMPLIANT" if is_compliant else "VIOLATION"

                track.last_helmet_color = helmet_color
                if hasattr(track, "color_history") and has_helmet and helmet_color:
                    track.color_history.append(helmet_color)
                track.last_violation_types = violations

            # 7. Debounce streak update
            if is_predicted_frame:
                # During grace period prediction, do not increment violation streak
                pass
            elif is_compliant:
                track.compliance_streak += 1
                if track.compliance_streak >= 2:
                    track.violation_streak = 0
                    # Note: We do NOT reset track.is_alerted here to prevent immediate re-alerting
                    # when head orientation or temporary occlusion causes a brief PPE flicker.
            else:
                track.violation_streak += 1
                track.compliance_streak = 0

            # Report data structure (using temporally stable helmet color only when helmet is present)
            stable_col = getattr(track, "stable_helmet_color", None) or helmet_color
            effective_color = stable_col if has_helmet else None
            effective_track_id = getattr(track, "canonical_id", track.track_id)
            report = {
                "track_id": effective_track_id,
                "bbox": [round(px1, 1), round(py1, 1), round(px2, 1), round(py2, 1)],
                "confidence": round(float(track.score), 3),
                "has_helmet": has_helmet,
                "helmet_color": effective_color,
                "stable_helmet_color": effective_color,
                "has_safety_vest": has_vest,
                "status": status_str,
                "violations": violations,
                "violation_streak": track.violation_streak,
                "compliance_streak": track.compliance_streak,
                "is_alerted": track.is_alerted,
                "is_at_border": is_at_border,
                "tracklet_len": getattr(track, "tracklet_len", 0),
                "matched_helmet_box": matched_helmet_box,
                "matched_vest_box": matched_vest_box,
            }
            worker_reports.append(report)

            # 8. Check if Alert should trigger (Debounce + Cooldown + Maturity + Spatial Centroid Debouncing)
            is_mature_track = (self.min_tracklet_len <= 0) or (getattr(track, "tracklet_len", 0) >= self.min_tracklet_len)
            strict_cooldown_passed = (not track.is_alerted) or ((now - track.last_alert_timestamp) >= self.cooldown_seconds)

            # Spatial-Centroid Alert Debouncing: suppress duplicates if another alert recently fired nearby
            norm_cx = (px1 + px2) / (2.0 * max(1, w_frame))
            norm_cy = (py1 + py2) / (2.0 * max(1, h_frame))
            is_spatially_debounced = any(
                np.hypot(norm_cx - rcx, norm_cy - rcy) < self.spatial_debounce_distance
                for rcx, rcy, _ in self.recent_alert_centroids
            )

            should_alert = (
                not is_predicted_frame
                and not is_compliant
                and is_mature_track
                and track.violation_streak >= self.consecutive_frames_thresh
                and strict_cooldown_passed
                and not is_spatially_debounced
            )

            if should_alert:
                track.is_alerted = True
                track.last_alert_timestamp = now
                self.recent_alert_centroids.append((norm_cx, norm_cy, now))

                # Generate high-res snapshot with bounding box overlay and person crop
                snapshot_filename, snapshot_path, crop_filename, crop_path = self._generate_violation_snapshot(
                    frame_bgr, report, camera_id, now_str
                )

                alert_event = {
                    "alert_id": f"ALERT-W{effective_track_id}-{int(now)}",
                    "camera_id": camera_id,
                    "worker_id": effective_track_id,
                    "timestamp": now_str,
                    "violations": violations,
                    "has_helmet": has_helmet,
                    "helmet_color": effective_color,
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

        if not (worker_crop.size > 0 and worker_crop.shape[0] > 10 and worker_crop.shape[1] > 10):
            # Fallback if box coordinates are degenerate
            worker_crop = vis.copy()

        # Submit worker crop write asynchronously to avoid blocking inference
        fut_crop = self.io_executor.submit(
            cv2.imwrite, str(crop_path), worker_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )

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

        # 5. Save snapshot file asynchronously via background thread pool
        filename = f"violation_worker_{report['track_id']}_{timestamp_ms}.jpg"
        save_path = SNAPSHOTS_DIR / filename
        fut_vis = self.io_executor.submit(
            cv2.imwrite, str(save_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )

        # Await quick flush (up to 25ms) for low-latency frame tests without blocking heavy 4K streams
        try:
            fut_crop.result(timeout=0.025)
            fut_vis.result(timeout=0.025)
        except (concurrent.futures.TimeoutError, Exception):
            pass

        return filename, save_path, crop_filename, crop_path
