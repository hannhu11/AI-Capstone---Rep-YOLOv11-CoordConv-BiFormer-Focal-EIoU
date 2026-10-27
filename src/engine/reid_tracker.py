"""
src/engine/reid_tracker.py
==========================
Persistent Worker Re-Identification & Spatio-Temporal Appearance Memory.
Guarantees consistent, non-exploding worker IDs across occlusions, postures,
and infinite video playback loops.
"""

from __future__ import annotations
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple


def compute_appearance_hist(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Computes a normalized 2D HSV color histogram (8x8 bins) from a bounding box crop."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(x1 + 1, min(w_img, x2))
    y2 = max(y1 + 1, min(h_img, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return np.zeros(64, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 8 bins for H [0, 180], 8 bins for S [0, 256]
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def hist_correlation(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    """Calculates histogram correlation similarity in range [-1.0, 1.0]."""
    if h1 is None or h2 is None or len(h1) == 0 or len(h2) == 0:
        return 0.0
    if np.all(h1 == 0) or np.all(h2 == 0):
        return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


class PersistentWorkerProfile:
    """Persistent identity and historical profile of a tracked worker."""

    def __init__(
        self,
        worker_id: int,
        box: np.ndarray,
        helmet_color: Optional[str] = None,
        hist: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ):
        self.worker_id = worker_id
        self.initial_box = np.asarray(box[:4], dtype=np.float32).copy()
        self.last_box = np.asarray(box[:4], dtype=np.float32).copy()
        self.helmet_color = helmet_color
        self.hist = hist
        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.total_frames_seen = 1
        self.is_active = True
        self.status = "COMPLIANT"
        self.violations: List[str] = []
        
        # Historical motion trajectory (bottom-center coordinates)
        bx_center = float((box[0] + box[2]) / 2.0)
        by_bottom = float(box[3])
        self.trajectory: deque[Tuple[float, float]] = deque(maxlen=90)
        self.trajectory.append((bx_center, by_bottom))

    def update(
        self,
        box: np.ndarray,
        frame_idx: int,
        helmet_color: Optional[str] = None,
        hist: Optional[np.ndarray] = None,
        status: str = "COMPLIANT",
        violations: Optional[List[str]] = None,
    ):
        self.last_box = np.asarray(box[:4], dtype=np.float32).copy()
        self.last_seen_frame = frame_idx
        self.total_frames_seen += 1
        self.is_active = True
        self.status = status
        self.violations = violations or []
        if helmet_color:
            self.helmet_color = helmet_color
        if hist is not None and not np.all(hist == 0):
            if self.hist is None:
                self.hist = hist
            else:
                # Exponential Moving Average for appearance signature
                self.hist = 0.80 * self.hist + 0.20 * hist

        bx_center = float((box[0] + box[2]) / 2.0)
        by_bottom = float(box[3])
        self.trajectory.append((bx_center, by_bottom))


class PersistentTrackRegistry:
    """
    Spatio-Temporal Appearance Memory & Re-Identification Engine.
    - Prevents ID fragmentation by re-associating occluded workers with existing profiles.
    - Enforces Baseline Loop Anchors so worker IDs are preserved across video loops.
    """

    def __init__(self, reid_affinity_thresh: float = 0.52):
        self.profiles: Dict[int, PersistentWorkerProfile] = {}
        self.next_available_id = 1
        self.reid_affinity_thresh = reid_affinity_thresh
        
        # Frame 0-15 Baseline Anchors to lock IDs on video loop
        self.baseline_anchors: Dict[int, Dict[str, Any]] = {}
        self.has_recorded_baseline = False
        self.is_looping_phase = False
        self.loop_cooldown_frames = 0

    def reset(self):
        """Cleans registry for a brand-new video source."""
        self.profiles.clear()
        self.baseline_anchors.clear()
        self.next_available_id = 1
        self.has_recorded_baseline = False
        self.is_looping_phase = False
        self.loop_cooldown_frames = 0

    def on_video_loop(self):
        """Called when video loops back to frame 0."""
        self.is_looping_phase = True
        self.loop_cooldown_frames = 20
        # Mark all current profiles as temporarily inactive until re-anchored at frame 0
        for p in self.profiles.values():
            p.is_active = False

    def get_or_create_canonical_id(
        self,
        raw_track_id: int,
        box: np.ndarray,
        frame: np.ndarray,
        helmet_color: Optional[str],
        frame_idx: int,
    ) -> int:
        """
        Maps a ByteTrack tracklet to a persistent Canonical Worker ID.
        If track already has a profile, updates it.
        If track is new, searches inactive profiles for Re-ID match before minting a new ID.
        """
        # 1. If we already know this raw_track_id in our profiles, keep it
        if raw_track_id in self.profiles:
            profile = self.profiles[raw_track_id]
            hist = compute_appearance_hist(frame, box)
            profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist)
            return raw_track_id

        hist = compute_appearance_hist(frame, box)
        h_img, w_img = frame.shape[:2]
        diag = np.sqrt(h_img**2 + w_img**2) + 1e-5

        # 2. Check Loop Baseline Anchors first if in loop recovery phase
        if self.is_looping_phase and self.baseline_anchors:
            best_anchor_id = None
            best_anchor_score = -1.0
            for aid, anchor in self.baseline_anchors.items():
                if self.profiles.get(aid) and self.profiles[aid].is_active:
                    continue  # already occupied in current frame
                abox = anchor["box"]
                # IoU or center distance with anchor
                inter_x1 = max(box[0], abox[0])
                inter_y1 = max(box[1], abox[1])
                inter_x2 = min(box[2], abox[2])
                inter_y2 = min(box[3], abox[3])
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = (box[2]-box[0])*(box[3]-box[1]) + (abox[2]-abox[0])*(abox[3]-abox[1]) - inter_area + 1e-5
                iou = inter_area / union_area

                dist_center = np.sqrt(((box[0]+box[2])/2 - (abox[0]+abox[2])/2)**2 + ((box[1]+box[3])/2 - (abox[1]+abox[3])/2)**2) / diag
                sim_pos = np.exp(- (dist_center**2) / (2 * 0.10**2))
                sim_col = hist_correlation(hist, anchor.get("hist"))
                sim_helm = 1.0 if (helmet_color and anchor.get("helmet_color") and helmet_color == anchor["helmet_color"]) else 0.0

                score = 0.40 * iou + 0.30 * sim_pos + 0.20 * max(0.0, sim_col) + 0.10 * sim_helm
                if score > best_anchor_score and score >= 0.45:
                    best_anchor_score = score
                    best_anchor_id = aid

            if best_anchor_id is not None:
                profile = self.profiles.get(best_anchor_id)
                if profile:
                    profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist)
                    return best_anchor_id

        # 3. Check for Re-ID among currently INACTIVE worker profiles
        best_match_id = None
        best_affinity = -1.0

        for wid, p in self.profiles.items():
            if p.is_active:
                continue  # cannot re-assign to someone already active in this frame

            # Distance between centers normalized to image diagonal
            p_cx = (p.last_box[0] + p.last_box[2]) / 2.0
            p_cy = (p.last_box[1] + p.last_box[3]) / 2.0
            d_cx = (box[0] + box[2]) / 2.0
            d_cy = (box[1] + box[3]) / 2.0
            dist_norm = np.sqrt((p_cx - d_cx)**2 + (p_cy - d_cy)**2) / diag

            # Spatial affinity: worker shouldn't jump across the entire screen
            sim_pos = np.exp(- (dist_norm**2) / (2 * 0.15**2))

            # Appearance affinity via color histogram
            sim_color = max(0.0, hist_correlation(p.hist, hist))

            # Helmet color consistency
            helmet_match = 1.0 if (helmet_color and p.helmet_color and helmet_color == p.helmet_color) else 0.0
            helmet_conflict = 1.0 if (helmet_color and p.helmet_color and helmet_color != p.helmet_color) else 0.0

            # Scale similarity
            ph = max(1.0, p.last_box[3] - p.last_box[1])
            dh = max(1.0, box[3] - box[1])
            sim_scale = min(ph, dh) / max(ph, dh)

            affinity = (
                0.35 * sim_pos
                + 0.35 * sim_color
                + 0.20 * helmet_match
                - 0.25 * helmet_conflict
                + 0.10 * sim_scale
            )

            # Boost affinity if within reasonable spatial distance
            if dist_norm <= 0.10 and (helmet_match or sim_color >= 0.50):
                affinity += 0.15

            if affinity > best_affinity and affinity >= self.reid_affinity_thresh:
                best_affinity = affinity
                best_match_id = wid

        if best_match_id is not None:
            # Re-associate with existing worker profile!
            profile = self.profiles[best_match_id]
            profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist)
            return best_match_id

        # 4. If no Re-ID match, mint a new Canonical ID
        canonical_id = self.next_available_id
        self.next_available_id += 1

        new_profile = PersistentWorkerProfile(
            worker_id=canonical_id,
            box=box,
            helmet_color=helmet_color,
            hist=hist,
            frame_idx=frame_idx,
        )
        self.profiles[canonical_id] = new_profile

        # Record baseline anchor for video loop if during initial frames (frame_idx < 30)
        if frame_idx < 30 and canonical_id not in self.baseline_anchors:
            self.baseline_anchors[canonical_id] = {
                "box": box[:4].copy(),
                "hist": hist.copy() if hist is not None else None,
                "helmet_color": helmet_color,
            }

        return canonical_id

    def update_frame_end(self, active_worker_ids: List[int], frame_idx: int):
        """Updates active states of all registered profiles at the end of each frame."""
        active_set = set(active_worker_ids)
        for wid, p in self.profiles.items():
            if wid in active_set:
                p.is_active = True
                p.last_seen_frame = frame_idx
            else:
                p.is_active = False

        if self.loop_cooldown_frames > 0:
            self.loop_cooldown_frames -= 1
            if self.loop_cooldown_frames == 0:
                self.is_looping_phase = False

    def get_registered_workers(self) -> List[Dict[str, Any]]:
        """Returns structured metadata for all registered workers."""
        workers = []
        for wid in sorted(self.profiles.keys()):
            p = self.profiles[wid]
            workers.append({
                "track_id": p.worker_id,
                "helmet_color": p.helmet_color,
                "status": p.status,
                "is_active": p.is_active,
                "first_frame": p.first_frame,
                "last_seen_frame": p.last_seen_frame,
                "total_frames_seen": p.total_frames_seen,
                "violations": list(p.violations),
                "trajectory_len": len(p.trajectory),
            })
        return workers


# Global singleton registry instance
reid_registry = PersistentTrackRegistry()
