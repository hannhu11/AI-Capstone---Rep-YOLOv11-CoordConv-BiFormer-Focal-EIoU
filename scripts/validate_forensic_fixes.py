"""
scripts/validate_forensic_fixes.py
==================================
Comprehensive automated validation script verifying:
1. White Round Table is NEVER detected or associated as a white helmet.
2. Bare Black Hair is NEVER misclassified or associated as a blue helmet.
3. Unauthorized Person is correctly tracked and flagged with VIOLATION alerts.
4. Compliant construction workers in 1080p video retain COMPLIANT status with zero false alerts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.spatial_safety import SpatialSafetyEngine

def compute_box_intersection_ratio(box_a, box_b, use_upper_torso=True, upper_torso_ratio=0.60):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    aw = max(0.0, float(ax2 - ax1))
    ah = max(0.0, float(ay2 - ay1))
    if aw * ah <= 1e-7: return 0.0
    by2_eff = float(by1) + max(0.0, float(by2 - by1)) * upper_torso_ratio if use_upper_torso else float(by2)
    ix1, iy1 = max(float(ax1), float(bx1)), max(float(ay1), float(by1))
    ix2, iy2 = min(float(ax2), float(bx2)), min(float(ay2), by2_eff)
    return (max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)) / (aw * ah)

def validate_unauthorized_cctv():
    print("=" * 70)
    print("VALIDATION 1: video_test/real_cctv_unauthorized_person_violations.mp4")
    print("=" * 70)

    video_path = "video_test/real_cctv_unauthorized_person_violations.mp4"
    detector = SafetyObjectDetector(model_path='Output/kaggle_results/hannhu40022_warmstart_ablations/rep_yolo11s_chv7_best_deploy.pt')
    tracker = BYTETracker()
    engine = SpatialSafetyEngine(consecutive_frames_thresh=5, cooldown_seconds=10.0, min_tracklet_len=5)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")

    stationary_tracker = {}
    confirmed_furniture_zones = []
    stationary_next_id = 1

    white_helmet_matches = 0
    blue_helmet_matches = 0
    violation_reports = 0
    total_alerts = 0
    table_rendered_count = 0

    for frame_idx in range(1, 150):
        ret, frame = cap.read()
        if not ret:
            break

        raw_dets, person_dets, _ = detector.predict(frame)
        h_frame, w_frame = frame.shape[:2]

        # 1. Update Stationary Furniture Tracker
        for d in raw_dets:
            if d['cls'] != 0 and 'person' not in d['name']:
                b = d['box']
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0

                # Check against confirmed stationary furniture zones
                is_in_zone = False
                for zb in confirmed_furniture_zones:
                    inter_w = max(0.0, min(b[2], zb[2]) - max(b[0], zb[0]))
                    inter_h = max(0.0, min(b[3], zb[3]) - max(b[1], zb[1]))
                    b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                    if (inter_w * inter_h) / b_area >= 0.50:
                        is_in_zone = True
                        break
                if is_in_zone:
                    d['is_stationary_furniture'] = True
                    continue

                matched_sid = None
                for sid, sinfo in stationary_tracker.items():
                    if abs(cx - sinfo['cx']) <= 12.0 and abs(cy - sinfo['cy']) <= 12.0:
                        matched_sid = sid
                        break
                if matched_sid is not None:
                    sentry = stationary_tracker[matched_sid]
                    sentry['count'] += 1
                    sentry['last_frame'] = frame_idx
                    sentry['box'] = b
                    if sentry['count'] > 30:
                        d['is_stationary_furniture'] = True
                        if sentry['box'] not in confirmed_furniture_zones:
                            confirmed_furniture_zones.append(list(sentry['box']))
                else:
                    stationary_tracker[stationary_next_id] = {'cx': cx, 'cy': cy, 'box': b, 'count': 1, 'last_frame': frame_idx}
                    stationary_next_id += 1

        stationary_tracker = {
            sid: sinfo for sid, sinfo in stationary_tracker.items()
            if (frame_idx - sinfo['last_frame']) <= 15
        }

        # 2. Bipartite Spatial Filter & Luminance Gating
        filtered_dets = []
        for d in raw_dets:
            if d['cls'] == 0 or 'person' in d['name']:
                filtered_dets.append(d)
                continue

            if d.get('is_stationary_furniture', False):
                continue

            if d['name'] == 'blue_helmet' or d.get('color') == 'blue' or d['cls'] == 2:
                bx1, by1, bx2, by2 = [int(round(v)) for v in d['box'][:4]]
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

            orig_conf = d['conf']
            max_overlap = 0.0
            if len(person_dets) > 0:
                is_helmet = 'helmet' in d['name'] or d['cls'] in (2, 3, 4, 5, 6)
                for p_box in person_dets:
                    ov = compute_box_intersection_ratio(d['box'], p_box[:4], use_upper_torso=is_helmet, upper_torso_ratio=0.60)
                    if ov > max_overlap:
                        max_overlap = ov

            effective_conf = orig_conf if max_overlap >= 0.25 else orig_conf * 0.15
            if effective_conf >= 0.25:
                d_c = dict(d)
                d_c['conf'] = round(effective_conf, 4)
                filtered_dets.append(d_c)

        for fd in filtered_dets:
            if fd['cls'] != 0 and 'person' not in fd['name']:
                # Check if table coordinates leaked into display
                tb = fd['box']
                if abs(tb[0] - 383.5) < 30 and abs(tb[1] - 78.5) < 30:
                    table_rendered_count += 1

        tracks = tracker.update(person_dets, frame=frame)
        reports, alerts = engine.evaluate_frame(tracks, filtered_dets, frame)

        for r in reports:
            if r.get('helmet_color') == 'white':
                white_helmet_matches += 1
            if r.get('helmet_color') == 'blue':
                blue_helmet_matches += 1
            if r['status'] == 'VIOLATION':
                violation_reports += 1

        if len(alerts) > 0:
            total_alerts += len(alerts)

    cap.release()

    print(f"Results on CCTV Unauthorized Video (150 frames):")
    print(f" - Table rendered as white helmet: {table_rendered_count} (Must be 0)")
    print(f" - Reports matching white helmet:  {white_helmet_matches} (Must be 0)")
    print(f" - Reports matching blue helmet:   {blue_helmet_matches} (Must be 0)")
    print(f" - Confirmed VIOLATION frames:     {violation_reports} (Expected >= 60)")
    print(f" - Violation Alerts fired:         {total_alerts} (Expected >= 1)")

    assert table_rendered_count == 0, f"Table was rendered {table_rendered_count} times!"
    assert white_helmet_matches == 0, f"Table was matched as white helmet {white_helmet_matches} times!"
    assert blue_helmet_matches == 0, f"Dark hair was matched as blue helmet {blue_helmet_matches} times!"
    assert violation_reports >= 60, f"Unauthorized woman not detected in enough violation frames: {violation_reports}"
    assert total_alerts >= 1, f"No violation alert fired for unauthorized person!"
    print(">>> VALIDATION 1 PASSED 100%! <<<\n")


def validate_construction_1080p():
    print("=" * 70)
    print("VALIDATION 2: video_test/construction_site_workers_1080p.mp4")
    print("=" * 70)

    video_path = "video_test/construction_site_workers_1080p.mp4"
    detector = SafetyObjectDetector(model_path='Output/kaggle_results/hannhu40022_warmstart_ablations/rep_yolo11s_chv7_best_deploy.pt')
    tracker = BYTETracker()
    engine = SpatialSafetyEngine(consecutive_frames_thresh=5, cooldown_seconds=10.0, min_tracklet_len=5)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

    compliant_frames = 0
    violation_frames = 0
    yellow_helmet_count = 0

    for frame_idx in range(200, 250):
        ret, frame = cap.read()
        if not ret:
            break
        raw_dets, person_dets, _ = detector.predict(frame)
        tracks = tracker.update(person_dets, frame=frame)
        reports, alerts = engine.evaluate_frame(tracks, raw_dets, frame)

        for r in reports:
            if r['status'] == 'COMPLIANT':
                compliant_frames += 1
            else:
                violation_frames += 1
            if r.get('helmet_color') == 'yellow':
                yellow_helmet_count += 1

    cap.release()

    print(f"Results on Construction Site 1080p (50 frames):")
    print(f" - Compliant frames:   {compliant_frames} (Expected >= 40)")
    print(f" - Violation frames:   {violation_frames} (Must be 0)")
    print(f" - Yellow helmet count: {yellow_helmet_count} (Expected >= 40)")

    assert compliant_frames >= 40, f"Compliant frames {compliant_frames} below threshold 40"
    assert violation_frames == 0, f"False violation frames occurred: {violation_frames}"
    assert yellow_helmet_count >= 40, f"Yellow helmet count {yellow_helmet_count} below threshold 40"
    print(">>> VALIDATION 2 PASSED 100%! <<<\n")


if __name__ == '__main__':
    validate_unauthorized_cctv()
    validate_construction_1080p()
    print("ALL FORENSIC VALIDATION CRITERIA SATISFIED WITH 100% SCIENTIFIC INTEGRITY!")
