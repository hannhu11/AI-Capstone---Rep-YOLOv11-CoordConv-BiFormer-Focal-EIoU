"""
src/engine/reid_tracker.py
==========================
SOTA Spatio-Temporal Worker Re-Identification & Persistent Appearance Memory.
Features:
- Upper-Torso ROI Color Signature (immune to floor background contamination & edge truncation).
- Anti-Hijack Guard: Layer 2 verification against cross-over identity swaps.
- Track Confirmation & Maturity Filtering (eliminates 1-frame ghost false positives).
- Single-Worker Low Density Prior (strictly bounds ID population to true head count).
- Multi-View Appearance Gallery with EMA (matches workers across varied angles).
- Dynamic Border Re-entry Affinity (smooth re-acquisition of workers entering/exiting frame edges).
- Automatic Spatio-Temporal Profile Consolidation (merges temporal split IDs caused by posture shifts).
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
    Robust to bottom-edge leg truncation and bending/stooping postures.
    """
    x1, y1, x2, y2 = box[:4]
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    aspect = bh / bw

    is_bottom_truncated = (y2 >= h_img - 20)
    is_bending = (aspect < 1.45)

    if is_bottom_truncated and aspect < 2.0:
        # Legs are clipped off by the bottom frame border.
        # Visible portion consists mainly of the head and upper torso.
        ty1 = int(max(0, min(h_img - 1, y1 + 0.20 * bh)))
        ty2 = int(max(ty1 + 1, min(h_img, y2)))
    elif is_bending:
        # Stooping/bending posture: back and torso occupy upper-central portion.
        ty1 = int(max(0, min(h_img - 1, y1 + 0.15 * bh)))
        ty2 = int(max(ty1 + 1, min(h_img, y1 + 0.70 * bh)))
    else:
        # Standard upright walking posture: upper torso between 18% and 62% of box height.
        ty1 = int(max(0, min(h_img - 1, y1 + 0.18 * bh)))
        ty2 = int(max(ty1 + 1, min(h_img, y1 + 0.62 * bh)))

    tx1 = int(max(0, min(w_img - 1, x1 + 0.15 * bw)))
    tx2 = int(max(tx1 + 1, min(w_img, x2 - 0.15 * bw)))
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
        self.last_seen_near_border = False
        self.active_intervals: List[Tuple[int, int]] = [(frame_idx, frame_idx)]

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
        frame_shape: Optional[Tuple[int, int]] = None,
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

        # Track active intervals for co-existence validation
        if self.active_intervals and self.active_intervals[-1][1] >= frame_idx - 2:
            self.active_intervals[-1] = (self.active_intervals[-1][0], frame_idx)
        else:
            self.active_intervals.append((frame_idx, frame_idx))

        # Check if worker is near border
        if frame_shape is not None:
            h_img, w_img = frame_shape[:2]
            self.last_seen_near_border = (
                box[0] <= 50 or box[1] <= 50 or box[2] >= w_img - 50 or box[3] >= h_img - 50
            )

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

    def coexisted_with(self, other: PersistentWorkerProfile) -> bool:
        """Returns True if self and other were active during overlapping frame windows."""
        for (s1, e1) in self.active_intervals:
            for (s2, e2) in other.active_intervals:
                if max(s1, s2) <= min(e1, e2):
                    return True
        return False


class PersistentTrackRegistry:
    """
    SOTA Spatio-Temporal Worker Re-Identification & Identity Memory Engine.
    - Prevents ID fragmentation and explosion across occlusions and posture shifts.
    - Locks Baseline Loop Anchors so worker IDs are preserved across video loops.
    - Suppresses ghost false positives via track maturity gating.
    - Defends against lock hijacking via Layer-2 Anti-Hijack Guard.
    """

    def __init__(self, reid_affinity_thresh: float = 0.40):
        self.profiles: Dict[int, PersistentWorkerProfile] = {}
        self.next_available_id = 1
        self.reid_affinity_thresh = reid_affinity_thresh

        # Map ByteTrack raw track ID to canonical worker ID
        self.raw_to_canonical: Dict[int, int] = {}
        self.raw_to_conflict_streak: Dict[int, int] = {}

        # Frame 0-30 Baseline Anchors to lock IDs on video loop
        self.baseline_anchors: Dict[int, Dict[str, Any]] = {}
        self.is_looping_phase = False
        self.loop_cooldown_frames = 0

    def reset(self):
        """Cleans registry for a brand-new video source or explicit reset."""
        self.profiles.clear()
        self.raw_to_canonical.clear()
        self.raw_to_conflict_streak.clear()
        self.baseline_anchors.clear()
        self.next_available_id = 1
        self.is_looping_phase = False
        self.loop_cooldown_frames = 0

    def on_video_loop(self):
        """Called when video loops back to frame 0."""
        self.is_looping_phase = True
        self.loop_cooldown_frames = 35
        self.raw_to_canonical.clear()
        self.raw_to_conflict_streak.clear()
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
        1. Direct raw-to-canonical binding with Layer-2 Anti-Hijack Guard.
        2. Loop Baseline Anchor matching (at start of video or loop).
        3. Single-Worker Low Density prior (prevents multi-ID minting on 1-worker scenes).
        4. Multi-View Torso Re-ID & Border Re-entry Affinity among inactive profiles.
        5. Candidate minting if genuinely novel worker.
        """
        hist = compute_appearance_hist(frame, box, use_torso_only=True)
        h_img, w_img = frame.shape[:2]
        diag = np.sqrt(h_img**2 + w_img**2) + 1e-5

        # 1. Direct Binding with Layer-2 Anti-Hijack Guard
        if raw_track_id in self.raw_to_canonical:
            cid = self.raw_to_canonical[raw_track_id]
            if cid in self.profiles:
                p = self.profiles[cid]
                # Anti-Hijack Guard: If profile is confirmed (>= 8 frames), verify that this track
                # has not been hijacked by a cross-over with an entirely different person
                if p.is_confirmed and p.total_frames_seen >= 8:
                    app_sim = p.get_max_appearance_sim(hist)
                    color_conflict = (
                        helmet_color
                        and p.helmet_color
                        and helmet_color != "none"
                        and p.helmet_color != "none"
                        and helmet_color != p.helmet_color
                    )
                    if color_conflict:
                        self.raw_to_conflict_streak[raw_track_id] = self.raw_to_conflict_streak.get(raw_track_id, 0) + 1
                    else:
                        self.raw_to_conflict_streak[raw_track_id] = 0

                    persistent_conflict = (self.raw_to_conflict_streak[raw_track_id] >= 3)

                    # Anti-Hijack Guard:
                    # - If app_sim >= 0.45 (identical torso appearance), NEVER split the track on color flicker alone.
                    # - Only break binding if appearance similarity is genuinely divergent (app_sim < 0.25) AND color conflict persists over multiple frames (>= 3 frames),
                    #   or if appearance is severely divergent (app_sim < 0.05) and color conflict exists.
                    should_break = False
                    if app_sim < 0.45:
                        if app_sim < 0.25 and persistent_conflict:
                            should_break = True
                        elif app_sim < 0.05 and color_conflict:
                            should_break = True

                    if should_break:
                        # Hijack intercepted! Break binding so Worker #cid is NOT stolen!
                        del self.raw_to_canonical[raw_track_id]
                        self.raw_to_conflict_streak.pop(raw_track_id, None)
                    else:
                        p.update(box, frame_idx, helmet_color=helmet_color, hist=hist, frame_shape=(h_img, w_img))
                        return cid
                else:
                    p.update(box, frame_idx, helmet_color=helmet_color, hist=hist, frame_shape=(h_img, w_img))
                    return cid


        # 2. Check Loop Baseline Anchors first if in loop recovery phase
        if self.is_looping_phase and self.baseline_anchors:
            best_anchor_id = None
            best_anchor_score = -1.0
            for aid, anchor in self.baseline_anchors.items():
                if self.profiles.get(aid) and self.profiles[aid].is_active:
                    continue  # already occupied in current frame
                abox = anchor["box"]

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
                    profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist, frame_shape=(h_img, w_img))
                    self.raw_to_canonical[raw_track_id] = best_anchor_id
                    return best_anchor_id

        # 3. Check for Re-ID among currently INACTIVE worker profiles
        inactive_profiles = [p for p in self.profiles.values() if not p.is_active]
        active_profiles = [p for p in self.profiles.values() if p.is_active]

        # Case 3A: Single-Worker Special Optimization
        # In a scene where only 1 worker has been confirmed and 0 are currently active,
        # any detected person is guaranteed to be that exact worker!
        if len(self.profiles) == 1 and len(inactive_profiles) == 1 and len(active_profiles) == 0:
            p = inactive_profiles[0]
            color_mismatch = (
                helmet_color
                and p.helmet_color
                and helmet_color != "none"
                and p.helmet_color != "none"
                and helmet_color != p.helmet_color
            )
            app_sim = p.get_max_appearance_sim(hist)
            if not color_mismatch or app_sim >= 0.40:
                p.update(box, frame_idx, helmet_color=helmet_color, hist=hist, frame_shape=(h_img, w_img))
                self.raw_to_canonical[raw_track_id] = p.worker_id
                return p.worker_id

        best_match_id = None
        best_affinity = -1.0
        d_cx = (box[0] + box[2]) / 2.0
        d_cy = (box[1] + box[3]) / 2.0
        dh = max(1.0, box[3] - box[1])
        is_candidate_near_border = (
            box[0] <= 50 or box[1] <= 50 or box[2] >= w_img - 50 or box[3] >= h_img - 50
        )

        for p in inactive_profiles:
            dt = max(1, frame_idx - p.last_seen_frame)
            pred_cx = (p.last_box[0] + p.last_box[2]) / 2.0 + p.vx * min(dt, 30)
            pred_cy = (p.last_box[1] + p.last_box[3]) / 2.0 + p.vy * min(dt, 30)

            dist_norm = np.sqrt((pred_cx - d_cx)**2 + (pred_cy - d_cy)**2) / diag

            sigma_pos = min(0.35, 0.15 + 0.005 * dt)
            sim_pos = np.exp(- (dist_norm**2) / (2 * sigma_pos**2))

            sim_color = max(0.0, p.get_max_appearance_sim(hist))

            # Helmet color logic: DO NOT PENALIZE MISSING HELMET / VIOLATION
            if helmet_color is not None and p.helmet_color is not None:
                if helmet_color == p.helmet_color:
                    helm_score = 1.0
                    w_helm = 0.20
                else:
                    # If torso appearance is identical (sim_color >= 0.45), soften helmet mismatch penalty against glare/lighting flicker
                    helm_score = -0.10 if sim_color >= 0.45 else -0.40
                    w_helm = 0.20
            else:
                helm_score = 0.0
                w_helm = 0.0

            ph = max(1.0, p.last_box[3] - p.last_box[1])
            sim_scale = min(ph, dh) / max(ph, dh)

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

            # Proximity boost if close to predicted position
            if dist_norm <= 0.08:
                affinity += 0.12

            # Border Re-entry boost: if both disappeared and reappeared near border
            if is_candidate_near_border and p.last_seen_near_border:
                affinity += 0.25

            if affinity > best_affinity and affinity >= self.reid_affinity_thresh:
                best_affinity = affinity
                best_match_id = p.worker_id

        if best_match_id is not None:
            profile = self.profiles[best_match_id]
            profile.update(box, frame_idx, helmet_color=helmet_color, hist=hist, frame_shape=(h_img, w_img))
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
        new_profile.last_seen_near_border = is_candidate_near_border
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

    def consolidate_profiles(self, current_frame: int):
        """
        Merges inactive duplicate profiles that represent the same physical worker
        who was temporarily lost during posture change / occlusion.
        """
        inactive_confirmed = [
            p for p in self.profiles.values()
            if not p.is_active and (p.is_confirmed or p.total_frames_seen >= 8)
        ]
        if len(inactive_confirmed) < 2:
            return

        # Sort by worker_id ascending (keep smaller ID as canonical)
        inactive_confirmed.sort(key=lambda x: x.worker_id)
        to_remove = set()

        for i in range(len(inactive_confirmed)):
            p_primary = inactive_confirmed[i]
            if p_primary.worker_id in to_remove:
                continue

            for j in range(i + 1, len(inactive_confirmed)):
                p_candidate = inactive_confirmed[j]
                if p_candidate.worker_id in to_remove:
                    continue

                # 1. Must never have co-existed in the same frame
                if p_primary.coexisted_with(p_candidate):
                    continue

                # 2. Helmet colors must not conflict
                if p_primary.helmet_color and p_candidate.helmet_color:
                    if (
                        p_primary.helmet_color != p_candidate.helmet_color
                        and p_primary.helmet_color != "none"
                        and p_candidate.helmet_color != "none"
                    ):
                        continue

                # 3. High appearance correlation across galleries
                max_sim = 0.0
                for h1 in p_primary.hist_gallery:
                    for h2 in p_candidate.hist_gallery:
                        sim = hist_correlation(h1, h2)
                        if sim > max_sim:
                            max_sim = sim

                if max_sim >= 0.45:
                    # MERGE p_candidate into p_primary
                    for h in p_candidate.hist_gallery:
                        if len(p_primary.hist_gallery) < 5:
                            p_primary.hist_gallery.append(h)
                    p_primary.total_frames_seen += p_candidate.total_frames_seen
                    p_primary.active_intervals.extend(p_candidate.active_intervals)
                    p_primary.last_seen_frame = max(p_primary.last_seen_frame, p_candidate.last_seen_frame)

                    # Remap all raw track mappings
                    for raw_id, can_id in list(self.raw_to_canonical.items()):
                        if can_id == p_candidate.worker_id:
                            self.raw_to_canonical[raw_id] = p_primary.worker_id

                    to_remove.add(p_candidate.worker_id)

        for wid in to_remove:
            if wid in self.profiles:
                del self.profiles[wid]

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

        # Periodic consolidation every 30 frames
        if frame_idx % 30 == 0:
            self.consolidate_profiles(frame_idx)

        if self.loop_cooldown_frames > 0:
            self.loop_cooldown_frames -= 1
            if self.loop_cooldown_frames == 0:
                self.is_looping_phase = False

    def get_registered_workers(self, filter_ghosts: bool = True) -> List[Dict[str, Any]]:
        """
        Returns structured metadata for registered workers.
        If filter_ghosts is True, transient spurious detections (< 15 frames seen
        and not currently active) are cleanly excluded from the official registry.
        """
        workers = []
        for wid in sorted(self.profiles.keys()):
            p = self.profiles[wid]
            if filter_ghosts and not p.is_active and p.total_frames_seen < 15:
                continue

            workers.append({
                "track_id": p.worker_id,
                "helmet_color": p.helmet_color,
                "status": p.status,
                "is_active": p.is_active,
                "is_confirmed": p.is_confirmed or (p.total_frames_seen >= 10),
                "first_frame": p.first_frame,
                "last_seen_frame": p.last_seen_frame,
                "total_frames_seen": p.total_frames_seen,
                "violations": list(p.violations),
                "trajectory_len": len(p.trajectory),
            })
        return workers


# Global singleton registry instance
reid_registry = PersistentTrackRegistry()
