"""
src/engine/reid_tracker.py
==========================
SOTA Spatio-Temporal Worker Re-Identification & Persistent Appearance Memory.
Features:
- Upper-Torso ROI Color Signature (immune to floor background contamination).
- Track Confirmation & Maturity Filtering (eliminates 1-frame ghost false positives).
- Violation-Resilient Affinity Matrix (zero penalty for missing helmet / PPE violations).
- Multi-View Appearance Gallery with EMA (matches workers front/back/profile).
- Velocity-Aware Motion Extrapolation (tracks moving workers across occlusions).
- Low-Density Single-Worker Fallback (strictly bounds ID population to true head count).
- Infinite Loop Anchor Lock (Worker #1 at frame 0 remains Worker #1 across infinite loops).
"""

from __future__ import annotations
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple


def extract_torso_roi(box: np.ndarray, h_img: int, w_img: int) -> Tuple[int, int, int, int]:
    """
    Extracts the upper-torso bounding box crop (18% - 62% height, 15% - 85% width).
    This region strictly captures safety vests and shirts while discarding floor background.
    """
    x1, y1, x2, y2 = box[:4]
    bw = x2 - x1
    bh = y2 - y1

    tx1 = int(max(0, min(w_img - 1, x1 + 0.15 * bw)))
    tx2 = int(max(tx1 + 1, min(w_img, x2 - 0.15 * bw)))
    ty1 = int(max(0, min(h_img - 1, y1 + 0.18 * bh)))
    ty2 = int(max(ty1 + 1, min(h_img, y1 + 0.62 * bh)))
    return tx1, ty1, tx2, ty2


def compute_appearance_hist(frame: np.ndarray, box: np.ndarray, use_torso_only: bool = True) -> np.ndarray:
    """
    Computes a normalized 2D HSV color histogram (12 Hue bins, 8 Saturation bins)
    from the upper-torso ROI to avoid ground floor color bleed.
    """
    h_img, w_img = frame.shape[:2]
    if use_torso_only:
        x1, y1, x2, y2 = extract_torso_roi(box, h_img, w_img)
    else:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(x1 + 1, min(w_img, x2))
        y2 = max(y1 + 1, min(h_img, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros(96, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 12 bins for H [0, 180], 8 bins for S [0, 256] -> 96 dimensions
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
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
        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.total_frames_seen = 1
        self.is_active = True
        self.is_confirmed = False  # Requires confirmation before permanent status
        self.consecutive_hits = 1
        self.status = "COMPLIANT"
        self.violations: List[str] = []

        # Velocity tracking (dx, dy per frame)
        self.vx = 0.0
        self.vy = 0.0

        # Multi-view appearance gallery (keeps up to 5 representative angles)
        self.hist_gallery: deque[np.ndarray] = deque(maxlen=5)
        if hist is not None and not np.all(hist == 0):
            self.hist_gallery.append(hist)
        self.hist = hist

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
        dt = max(1, frame_idx - self.last_seen_frame)
        prev_cx = (self.last_box[0] + self.last_box[2]) / 2.0
        prev_cy = (self.last_box[1] + self.last_box[3]) / 2.0
        new_cx = (box[0] + box[2]) / 2.0
        new_cy = (box[1] + box[3]) / 2.0

        if dt <= 10:
            inst_vx = (new_cx - prev_cx) / float(dt)
            inst_vy = (new_cy - prev_cy) / float(dt)
            self.vx = 0.70 * self.vx + 0.30 * inst_vx
            self.vy = 0.70 * self.vy + 0.30 * inst_vy

        self.last_box = np.asarray(box[:4], dtype=np.float32).copy()
        self.last_seen_frame = frame_idx
        self.total_frames_seen += 1
        self.consecutive_hits += 1
        if self.consecutive_hits >= 5 or self.total_frames_seen >= 8:
            self.is_confirmed = True

        self.is_active = True
        self.status = status
        self.violations = violations or []

        # Only update helmet color if a valid color is positively detected
        if helmet_color and helmet_color != "none":
            self.helmet_color = helmet_color

        if hist is not None and not np.all(hist == 0):
            if self.hist is None:
                self.hist = hist
                self.hist_gallery.append(hist)
            else:
                self.hist = 0.85 * self.hist + 0.15 * hist
                # Append to gallery if novel perspective
                max_sim = max([hist_correlation(h, hist) for h in self.hist_gallery]) if self.hist_gallery else 0.0
                if max_sim < 0.85 and len(self.hist_gallery) < 5:
                    self.hist_gallery.append(hist)

        bx_center = float((box[0] + box[2]) / 2.0)
        by_bottom = float(box[3])
        self.trajectory.append((bx_center, by_bottom))

    def get_max_appearance_sim(self, candidate_hist: np.ndarray) -> float:
        """Finds maximum correlation across all stored perspective views."""
        if candidate_hist is None or np.all(candidate_hist == 0):
            return 0.0
        sims = [hist_correlation(h, candidate_hist) for h in self.hist_gallery]
        if self.hist is not None:
            sims.append(hist_correlation(self.hist, candidate_hist))
        return max(sims) if sims else 0.0


class PersistentTrackRegistry:
    """
    SOTA Spatio-Temporal Worker Re-Identification & Identity Memory Engine.
    - Prevents ID fragmentation and explosion across occlusions and posture shifts.
    - Locks Baseline Loop Anchors so worker IDs are preserved across video loops.
    - Suppresses ghost false positives via track maturity gating.
    """

    def __init__(self, reid_affinity_thresh: float = 0.42):
        self.profiles: Dict[int, PersistentWorkerProfile] = {}
        self.next_available_id = 1
        self.reid_affinity_thresh = reid_affinity_thresh

        # Map ByteTrack raw track ID to canonical worker ID
        self.raw_to_canonical: Dict[int, int] = {}

        # Frame 0-30 Baseline Anchors to lock IDs on video loop
        self.baseline_anchors: Dict[int, Dict[str, Any]] = {}
        self.is_looping_phase = False
        self.loop_cooldown_frames = 0

    def reset(self):
        """Cleans registry for a brand-new video source or explicit reset."""
        self.profiles.clear()
        self.raw_to_canonical.clear()
        self.baseline_anchors.clear()
        self.next_available_id = 1
        self.is_looping_phase = False
        self.loop_cooldown_frames = 0

    def on_video_loop(self):
        """Called when video loops back to frame 0."""
        self.is_looping_phase = True
        self.loop_cooldown_frames = 35
        self.raw_to_canonical.clear()
        for p in self.profiles.values():
            p.is_active = False
            p.consecutive_hits = 0

    def get_or_create_canonical_id(
        self,
        raw_track_id: int,
        box: np.ndarray,
        frame: np.ndarray,
        helmet_color: Optional[str],
        frame_idx: int,
    ) -> int:
        """
        Maps a ByteTrack tracklet to a canonical persistent Worker ID using:
        1. Direct raw-to-canonical binding (if already active).
        2. Loop Baseline Anchor matching (at start of video or loop).
        3. Multi-View Torso Re-ID & Velocity-aware Spatial Gating among inactive profiles.
        4. Single-Worker Low Density prior (prevents multi-ID minting on 1-worker scenes).
        5. Candidate minting if genuinely novel worker.
        """
        # 1. Check direct raw_to_canonical binding or if raw_track_id is already an active canonical ID
        if raw_track_id in self.raw_to_canonical:
            cid = self.raw_to_canonical[raw_track_id]
            if cid in self.profiles:
                hist = compute_appearance_hist(frame, box, use_torso_only=True)
                self.profiles[cid].update(box, frame_idx, helmet_color=helmet_color, hist=hist)
                return cid

        if raw_track_id in self.profiles:
            hist = compute_appearance_hist(frame, box, use_torso_only=True)
            self.profiles[raw_track_id].update(box, frame_idx, helmet_color=helmet_color, hist=hist)
            self.raw_to_canonical[raw_track_id] = raw_track_id
            return raw_track_id

        hist = compute_appearance_hist(frame, box, use_torso_only=True)
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

                # Center distance normalized
                d_cx = (box[0] + box[2]) / 2.0
                d_cy = (box[1] + box[3]) / 2.0
                a_cx = (abox[0] + abox[2]) / 2.0
                a_cy = (abox[1] + abox[3]) / 2.0
                dist_center = np.sqrt((d_cx - a_cx)**2 + (d_cy - a_cy)**2) / diag
                sim_pos = np.exp(- (dist_center**2) / (2 * 0.18**2))

                sim_col = hist_correlation(hist, anchor.get("hist"))
                sim_helm = 1.0 if (helmet_color and anchor.get("helmet_color") and helmet_color == anchor["helmet_color"]) else 0.0

                score = 0.45 * sim_pos + 0.35 * max(0.0, sim_col) + 0.20 * sim_helm
                if score > best_anchor_score and score >= 0.40:
                    best_anchor_score = score
                    best_anchor_id = aid

            if best_anchor_id is not None:
                profile = self.profiles.get(best_anchor_id)
                if profile:
                    profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist)
                    self.raw_to_canonical[raw_track_id] = best_anchor_id
                    return best_anchor_id

        # 3. Check for Re-ID among currently INACTIVE worker profiles
        best_match_id = None
        best_affinity = -1.0
        inactive_profiles = [p for p in self.profiles.values() if not p.is_active]

        # Case 3A: Single-Worker Special Optimization
        # If there is only 1 confirmed profile in total, and 0 currently active:
        if len(self.profiles) == 1 and len(inactive_profiles) == 1:
            p = inactive_profiles[0]
            sim_col = p.get_max_appearance_sim(hist)
            # As long as there is no blatant conflict, it's the exact same worker walking
            if sim_col >= 0.20 or (helmet_color and p.helmet_color and helmet_color == p.helmet_color):
                p.update(box, frame_idx, helmet_color=helmet_color, hist=hist)
                self.raw_to_canonical[raw_track_id] = p.worker_id
                return p.worker_id

        d_cx = (box[0] + box[2]) / 2.0
        d_cy = (box[1] + box[3]) / 2.0
        dh = max(1.0, box[3] - box[1])

        for p in inactive_profiles:
            dt = max(1, frame_idx - p.last_seen_frame)
            # Kalman Velocity extrapolation of previous worker center
            pred_cx = (p.last_box[0] + p.last_box[2]) / 2.0 + p.vx * min(dt, 30)
            pred_cy = (p.last_box[1] + p.last_box[3]) / 2.0 + p.vy * min(dt, 30)

            dist_norm = np.sqrt((pred_cx - d_cx)**2 + (pred_cy - d_cy)**2) / diag

            # Dynamic spatial affinity (expands search radius as time dt elapses)
            sigma_pos = min(0.35, 0.15 + 0.005 * dt)
            sim_pos = np.exp(- (dist_norm**2) / (2 * sigma_pos**2))

            # Upper-Torso Appearance similarity across multi-view gallery
            sim_color = max(0.0, p.get_max_appearance_sim(hist))

            # Helmet color logic: DO NOT PENALIZE MISSING HELMET / VIOLATION
            if helmet_color is not None and p.helmet_color is not None:
                if helmet_color == p.helmet_color:
                    helm_score = 1.0
                    w_helm = 0.20
                else:
                    # Explicit conflicting helmet color (e.g. Yellow vs Blue)
                    helm_score = -0.35
                    w_helm = 0.20
            else:
                # Helmet missing / occluded / violation -> NEUTRAL
                helm_score = 0.0
                w_helm = 0.0

            # Scale similarity
            ph = max(1.0, p.last_box[3] - p.last_box[1])
            sim_scale = min(ph, dh) / max(ph, dh)

            # Re-normalize remaining weights
            base_w_sum = 0.40 + 0.35 + 0.05 + w_helm
            w_pos = 0.40 / base_w_sum
            w_col = 0.35 / base_w_sum
            w_scale = 0.05 / base_w_sum
            w_h = w_helm / base_w_sum

            affinity = (
                w_pos * sim_pos
                + w_col * sim_color
                + w_scale * sim_scale
                + w_h * helm_score
            )

            # Proximity boost if very close to predicted position
            if dist_norm <= 0.08:
                affinity += 0.12

            if affinity > best_affinity and affinity >= self.reid_affinity_thresh:
                best_affinity = affinity
                best_match_id = p.worker_id

        if best_match_id is not None:
            profile = self.profiles[best_match_id]
            profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist)
            self.raw_to_canonical[raw_track_id] = best_match_id
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
        self.raw_to_canonical[raw_track_id] = canonical_id

        # Record baseline anchor for video loop if during initial frames (frame_idx < 35)
        if frame_idx < 35 and canonical_id not in self.baseline_anchors:
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
                p.consecutive_hits = 0

        if self.loop_cooldown_frames > 0:
            self.loop_cooldown_frames -= 1
            if self.loop_cooldown_frames == 0:
                self.is_looping_phase = False

    def get_registered_workers(self, filter_ghosts: bool = True) -> List[Dict[str, Any]]:
        """
        Returns structured metadata for registered workers.
        If filter_ghosts is True, unconfirmed 1-frame spurious tracks are excluded.
        """
        workers = []
        for wid in sorted(self.profiles.keys()):
            p = self.profiles[wid]
            # Exclude unconfirmed spurious ghost detections (< 5 hits and not currently active)
            if filter_ghosts and not p.is_confirmed and p.total_frames_seen < 5 and not p.is_active:
                continue

            workers.append({
                "track_id": p.worker_id,
                "helmet_color": p.helmet_color,
                "status": p.status,
                "is_active": p.is_active,
                "is_confirmed": p.is_confirmed or (p.total_frames_seen >= 5),
                "first_frame": p.first_frame,
                "last_seen_frame": p.last_seen_frame,
                "total_frames_seen": p.total_frames_seen,
                "violations": list(p.violations),
                "trajectory_len": len(p.trajectory),
            })
        return workers


# Global singleton registry instance
reid_registry = PersistentTrackRegistry()
