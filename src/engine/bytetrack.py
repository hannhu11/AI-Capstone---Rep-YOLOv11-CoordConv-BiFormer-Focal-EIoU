"""
src/engine/bytetrack.py
=======================
High-Performance ByteTrack Multi-Object Tracking (MOT) Engine.
Tailored for Industrial Construction Worker Tracking (Persistent Person IDs).
Features:
- Two-stage association: high-confidence detections first, then low-confidence detections to recover occlusions.
- Kalman Filter motion prediction (constant velocity model).
- Scipy linear sum assignment (Hungarian algorithm) for 100% platform portability.
- Re-ID appearance persistence across temporary occlusions.
"""

from __future__ import annotations
import numpy as np
from collections import Counter, deque
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional, Dict, Any

from src.engine.kalman_tracker import KalmanFilter


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Individual Worker Tracklet with Kalman state and persistent metadata."""
    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float, cls_id: int = 0):
        # tlwh: [top, left, width, height]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated = False

        self.score = float(score)
        self.cls_id = int(cls_id)
        self.tracklet_len = 0
        self.state = TrackState.New
        self.time_since_update = 0
        self.is_predicted = False

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

        # Safety & Violation History
        self.violation_streak = 0
        self.compliance_streak = 0
        self.is_alerted = False
        self.last_alert_timestamp = 0.0
        self.last_violation_types: List[str] = []
        self.last_helmet_color: Optional[str] = None
        self.color_history: deque[Optional[str]] = deque(maxlen=15)
        self.unconfirmed_misses = 0
        self.history_centers: List[Tuple[float, float]] = []
        self.hist: Optional[np.ndarray] = None

    @property
    def stable_helmet_color(self) -> Optional[str]:
        """Temporally smoothed helmet color via majority voting over recent frames."""
        if not self.color_history:
            return self.last_helmet_color
        valid_colors = [c for c in self.color_history if c and c != "unknown"]
        if valid_colors:
            return Counter(valid_colors).most_common(1)[0][0]
        non_none = [c for c in self.color_history if c is not None]
        if non_none:
            return Counter(non_none).most_common(1)[0][0]
        return self.last_helmet_color

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    def activate(self, kalman_filter: KalmanFilter, frame_id: int, track_thresh: float = 0.45):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1 or self.score >= track_thresh:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.is_predicted = False
        self._record_center()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Re-activate a lost tracklet with newly associated detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
        self.is_predicted = False
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        if new_track.hist is not None:
            self.hist = new_track.hist if self.hist is None else (0.85 * self.hist + 0.15 * new_track.hist)
        self._record_center()

    def update(self, new_track: STrack, frame_id: int):
        """Update a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        self.is_predicted = False

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        if new_track.hist is not None:
            self.hist = new_track.hist if self.hist is None else (0.85 * self.hist + 0.15 * new_track.hist)
        self._record_center()

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        h = max(1.0, float(mean_state[3]))
        # Clamp velocity components to physical human limits
        mean_state[4] = np.clip(mean_state[4], -0.25 * h, 0.25 * h)
        mean_state[5] = np.clip(mean_state[5], -0.25 * h, 0.25 * h)
        mean_state[6] = np.clip(mean_state[6], -0.05, 0.05)
        mean_state[7] = np.clip(mean_state[7], -0.15 * h, 0.15 * h)
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        if np.isnan(self.mean).any() or np.isinf(self.mean).any():
            self.mean = mean_state
        if np.isnan(self.covariance).any() or np.isinf(self.covariance).any():
            self.covariance = np.eye(8, dtype=np.float32) * 10.0
        self.time_since_update += 1

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def _record_center(self):
        tlwh = self.tlwh
        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        self.history_centers.append((cx, cy))
        if len(self.history_centers) > 30:
            self.history_centers.pop(0)

    @property
    def tlwh(self) -> np.ndarray:
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None or np.isnan(self.mean).any() or np.isinf(self.mean).any():
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        h = max(float(ret[3]), 1.0)
        a = float(ret[2])
        if np.isnan(a) or np.isinf(a):
            a = 0.40
        a = max(0.05, min(a, 4.0))
        w = max(a * h, 1.0)
        ret[2] = w
        ret[3] = h
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Convert bounding box to `(min x, min y, max x, max y)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box to `(center x, center y, aspect ratio, height)`."""
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        if np.isnan(ret).any() or np.isinf(ret).any():
            return np.array([0.0, 0.0, 0.40, 100.0], dtype=np.float32)
        ret[:2] += ret[2:] / 2
        h = max(float(ret[3]), 1e-4)
        w = max(float(ret[2]), 1e-4)
        ret[2] = w / h
        ret[3] = h
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        if np.isnan(ret).any() or np.isinf(ret).any():
            return np.array([0.0, 0.0, 40.0, 100.0], dtype=np.float32)
        ret[2:] -= ret[:2]
        ret[2] = max(float(ret[2]), 1.0)
        ret[3] = max(float(ret[3]), 1.0)
        return ret


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes (tlbr format)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a_x1, a_y1, a_x2, a_y2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, a_x2 - a_x1) * np.maximum(0.0, a_y2 - a_y1)
    area_b = np.maximum(0.0, b_x2 - b_x1) * np.maximum(0.0, b_y2 - b_y1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area + 1e-7

    return inter_area / union_area


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching using scipy.optimize.linear_sum_assignment."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, np.asarray(unmatched_a, dtype=int), np.asarray(unmatched_b, dtype=int)


def compute_matching_cost(tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
    """
    Computes cost matrix between tracks and detections based on IoU distance (1 - IoU)
    with Centroid Distance Fallback in Hungarian matching:
    If IoU distance > 0.70, calculate normalized centroid distance d_center.
    If d_center <= 0.08 and Upper-Body IoU >= 0.20, accept the match by capping cost <= 0.65.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    boxes_a = np.array([s.tlbr for s in tracks], dtype=np.float32)
    boxes_b = np.array([d.tlbr for d in detections], dtype=np.float32)

    full_iou = box_iou_batch(boxes_a, boxes_b)
    dists = 1.0 - full_iou

    max_w = max(1920.0, float(np.max(boxes_a[:, 2])), float(np.max(boxes_b[:, 2])))
    max_h = max(1080.0, float(np.max(boxes_a[:, 3])), float(np.max(boxes_b[:, 3])))
    diag = float(np.hypot(max_w, max_h))

    # Upper-body boxes (top 50% of bounding box height)
    ub_a = boxes_a.copy()
    ub_a[:, 3] = ub_a[:, 1] + 0.50 * np.maximum(1.0, ub_a[:, 3] - ub_a[:, 1])

    ub_b = boxes_b.copy()
    ub_b[:, 3] = ub_b[:, 1] + 0.50 * np.maximum(1.0, ub_b[:, 3] - ub_b[:, 1])

    upper_iou = box_iou_batch(ub_a, ub_b)

    # Full-body centroids
    c_ax = (boxes_a[:, 0] + boxes_a[:, 2]) / 2.0
    c_ay = (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0
    c_bx = (boxes_b[:, 0] + boxes_b[:, 2]) / 2.0
    c_by = (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0

    # Upper-body centroids (for robust association when worker bends)
    ub_c_ax = (ub_a[:, 0] + ub_a[:, 2]) / 2.0
    ub_c_ay = (ub_a[:, 1] + ub_a[:, 3]) / 2.0
    ub_c_bx = (ub_b[:, 0] + ub_b[:, 2]) / 2.0
    ub_c_by = (ub_b[:, 1] + ub_b[:, 3]) / 2.0

    for i in range(len(tracks)):
        for j in range(len(detections)):
            d_full = float(np.hypot(c_ax[i] - c_bx[j], c_ay[i] - c_by[j])) / diag
            d_ub = float(np.hypot(ub_c_ax[i] - ub_c_bx[j], ub_c_ay[i] - ub_c_by[j])) / diag
            d_center = min(d_full, d_ub)
            u_iou = float(upper_iou[i, j])

            if dists[i, j] > 0.70:
                if d_center <= 0.08 and u_iou >= 0.20:
                    # Accept fallback match: scale into [0.45, 0.65] <= match_thresh (0.70)
                    fallback_dist = 0.65 - 0.20 * min(1.0, (u_iou - 0.20) / 0.80)
                    dists[i, j] = min(dists[i, j], fallback_dist)

            # Appearance Feature Affinity Gating: Prevents identity swap during cross-over
            if tracks[i].hist is not None and detections[j].hist is not None:
                from src.engine.reid_tracker import hist_correlation
                sim_app = hist_correlation(tracks[i].hist, detections[j].hist)
                if sim_app < 0.15:
                    # Distinct appearance conflict (e.g. Yellow Vest vs Grey Shirt)
                    # Add heavy penalty to prevent Hungarian algorithm from swapping them!
                    dists[i, j] += 0.80
                elif sim_app >= 0.40 and (full_iou[i, j] > 0.05 or d_center <= 0.15):
                    # Appearance similarity reward strictly conditioned on spatial proximity
                    dists[i, j] = 0.60 * dists[i, j] + 0.40 * max(0.0, 1.0 - sim_app)

    return dists


class BYTETracker:
    """
    SOTA ByteTrack Tracker for CCTV Safety Surveillance.
    Associates high-confidence detections first, then low-confidence detections to recover occluded workers.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.50,
        track_buffer: int = 60,
        match_thresh: float = 0.70,
        frame_rate: int = 30,
        grace_period: int = 0,
    ):
        self.track_thresh = track_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.grace_period = grace_period

        self.kalman_filter = KalmanFilter()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.max_time_lost = int(track_buffer)

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_counter()

    def update(self, dets: np.ndarray, frame: Optional[np.ndarray] = None) -> List[STrack]:
        """
        dets: array of shape (N, 5 or 6) with format: [x1, y1, x2, y2, score, (cls_id)]
        frame: optional BGR image frame used for computing upper-torso appearance histograms.
        Returns: list of currently confirmed, active tracked STrack objects.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) > 0:
            scores = dets[:, 4]
            bboxes = dets[:, :4]
            classes = dets[:, 5] if dets.shape[1] > 5 else np.zeros(len(dets), dtype=int)

            # Split into High-Score and Low-Score detections
            remain_inds = scores >= self.track_thresh
            inds_low = (scores >= self.track_low_thresh) & (scores < self.track_thresh)

            dets_first = bboxes[remain_inds]
            scores_first = scores[remain_inds]
            classes_first = classes[remain_inds]

            dets_second = bboxes[inds_low]
            scores_second = scores[inds_low]
            classes_second = classes[inds_low]
        else:
            dets_first = np.empty((0, 4), dtype=np.float32)
            scores_first = np.empty((0,), dtype=np.float32)
            classes_first = np.empty((0,), dtype=int)
            dets_second = np.empty((0, 4), dtype=np.float32)
            scores_second = np.empty((0,), dtype=np.float32)
            classes_second = np.empty((0,), dtype=int)

        # 1. Instantiate STrack candidate objects for first association
        detections = []
        for tlbr, s, c in zip(dets_first, scores_first, classes_first):
            trk = STrack(STrack.tlbr_to_tlwh(tlbr), s, c)
            if frame is not None:
                from src.engine.reid_tracker import compute_appearance_hist
                trk.hist = compute_appearance_hist(frame, tlbr, use_torso_only=True)
            detections.append(trk)

        # 2. Divide tracks into tracked pool and lost pool
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Predict Kalman states for all tracked (active + unconfirmed) and lost tracks
        for strack in self.tracked_stracks + self.lost_stracks:
            strack.predict()

        # Combine confirmed tracked and lost tracks for first association pool
        strack_pool = tracked_stracks + self.lost_stracks

        # 3. First Association: High-score detections with strack_pool
        dists = compute_matching_cost(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 4. Second Association: Low-score detections with remaining unmatched tracks
        detections_second = []
        for tlbr, s, c in zip(dets_second, scores_second, classes_second):
            trk = STrack(STrack.tlbr_to_tlwh(tlbr), s, c)
            if frame is not None:
                from src.engine.reid_tracker import compute_appearance_hist
                trk.hist = compute_appearance_hist(frame, tlbr, use_torso_only=True)
            detections_second.append(trk)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state in (TrackState.Tracked, TrackState.Lost)]

        dists_second = compute_matching_cost(r_tracked_stracks, detections_second)
        matches_second, u_track_second, _ = linear_assignment(dists_second, thresh=self.match_thresh)

        for itracked, idet in matches_second:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Tracks still unmatched after second association are marked lost
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 5. Deal with unconfirmed tracks (matches with remaining detections)
        detections_rem = [detections[i] for i in u_detection]
        dists_unconf = 1.0 - box_iou_batch(
            np.array([s.tlbr for s in unconfirmed], dtype=np.float32),
            np.array([d.tlbr for d in detections_rem], dtype=np.float32),
        )
        matches_unconf, u_unconfirmed, u_detection_final = linear_assignment(dists_unconf, thresh=0.7)

        for itracked, idet in matches_unconf:
            unconfirmed[itracked].update(detections_rem[idet], self.frame_id)
            unconfirmed[itracked].unconfirmed_misses = 0
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.unconfirmed_misses = getattr(track, "unconfirmed_misses", 0) + 1
            if track.unconfirmed_misses >= 3:
                track.mark_removed()
                removed_stracks.append(track)
            else:
                # Grace period for unconfirmed tracklets: keep in pool to prevent ID destruction on 1-frame dip
                activated_stracks.append(track)

        # 6. Initialize new tracks for unmatched high-confidence detections
        for inew in u_detection_final:
            track = detections_rem[inew]
            if track.score >= self.new_track_thresh:
                track.activate(self.kalman_filter, self.frame_id, self.track_thresh)
                activated_stracks.append(track)

        # 7. Update state lists
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._merge_tracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._merge_tracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks = self._merge_tracks(self.lost_stracks, lost_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.tracked_stracks]

        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 50:
            self.removed_stracks = self.removed_stracks[-50:]

        # Return active tracked objects
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        for t in output_stracks:
            t.is_predicted = False

        # Grace Period Rendering: Activated tracks with time_since_update <= grace_period
        # remain in output_stracks using Kalman predicted coordinates (with is_predicted = True)
        if self.grace_period > 0:
            for track in self.lost_stracks:
                if track.is_activated and 1 <= track.time_since_update <= self.grace_period:
                    # GHOST BOX ELIMINATION:
                    # 1. Require activated tracklet, eliminating unconfirmed spurious detections
                    if not track.is_activated:
                        continue
                    # 2. Kalman filter validity: no NaN or Inf
                    if track.mean is None or np.isnan(track.mean).any() or np.isinf(track.mean).any():
                        continue
                    # 3. Suppress tracks with excessive velocity or diverged covariance
                    if np.hypot(track.mean[4], track.mean[5]) > 60.0:
                        continue
                    if track.covariance is not None and np.trace(track.covariance[:2, :2]) > 5000.0:
                        continue
                    b = track.tlbr
                    bw = b[2] - b[0]
                    bh = b[3] - b[1]
                    if bw < 10 or bh < 25 or (bw / max(1.0, bh)) > 3.5:
                        continue
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        if b[2] < -50 or b[0] > fw + 50 or b[3] < -50 or b[1] > fh + 50:
                            continue
                    track.is_predicted = True
                    output_stracks.append(track)

        return output_stracks

    @staticmethod
    def _merge_tracks(track_list: List[STrack], new_tracks: List[STrack]) -> List[STrack]:
        track_dict = {t.track_id: t for t in track_list}
        for t in new_tracks:
            track_dict[t.track_id] = t
        return list(track_dict.values())
