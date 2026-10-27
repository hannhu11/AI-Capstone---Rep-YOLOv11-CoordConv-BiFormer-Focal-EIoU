import cv2
import numpy as np
from src.config import config
from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.spatial_safety import SpatialSafetyEngine
from src.engine.reid_tracker import PersistentTrackRegistry

video_path = "video_test/13921040-uhd_3840_2160_30fps.mp4"
detector = SafetyObjectDetector(config.model_path, device=config.device, use_half=False)
tracker = BYTETracker(track_thresh=0.30, track_low_thresh=0.15, new_track_thresh=0.30)
safety_engine = SpatialSafetyEngine()
registry = PersistentTrackRegistry(reid_affinity_thresh=0.48)

cap = cv2.VideoCapture(video_path)

focus_target_id = 3

for f in range(60):
    ret, frame = cap.read()
    if not ret: break

    orig_h, orig_w = frame.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim > 1920:
        scale = 1920.0 / max_dim
        frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)

    dets, person_dets, _ = detector.predict(frame)
    if len(person_dets) > 0:
        frame_h, frame_w = frame.shape[:2]
        valid_mask = []
        for p in person_dets:
            pw, ph = p[2]-p[0], p[3]-p[1]
            touches = (p[0]<=60 or p[1]<=60 or p[2]>=frame_w-60 or p[3]>=frame_h-60)
            valid_mask.append(ph>=35 and pw>=15 and (pw/max(1.0,ph)<=(3.5 if touches else 1.65)))
        person_dets = person_dets[np.array(valid_mask, dtype=bool)] if any(valid_mask) else np.empty((0,6), dtype=np.float32)

    active_tracks = tracker.update(person_dets)
    for track in active_tracks:
        tb = track.tlbr
        matched_color = None
        for d in dets:
            if "helmet" in d["name"]:
                hb = d["box"]
                if not (hb[2] < tb[0] or hb[0] > tb[2] or hb[3] < tb[1] or hb[1] > tb[1] + 0.40*(tb[3]-tb[1])):
                    matched_color = d.get("color") or d["name"].replace("_helmet", "")
                    break
        canonical_id = registry.get_or_create_canonical_id(
            raw_track_id=track.track_id, box=tb, frame=frame, helmet_color=matched_color, frame_idx=f
        )
        track.track_id = canonical_id

    reports, _ = safety_engine.evaluate_frame(active_tracks, dets, frame)
    registry.update_frame_end([r["track_id"] for r in reports], f)

    if f == 55:
        # Render in focus mode
        vis = frame.copy()
        # Find target worker
        target_r = next((r for r in reports if r["track_id"] == focus_target_id), None)
        if target_r:
            tb = target_r["bbox"]
            x1, y1, x2, y2 = [int(v) for v in tb]
            # Draw glowing cyan box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 240, 255), 3)
            # Corner targeting brackets
            d = 16
            cv2.line(vis, (x1, y1), (x1 + d, y1), (255, 255, 0), 3)
            cv2.line(vis, (x1, y1), (x1, y1 + d), (255, 255, 0), 3)
            cv2.line(vis, (x2, y1), (x2 - d, y1), (255, 255, 0), 3)
            cv2.line(vis, (x2, y1), (x2, y1 + d), (255, 255, 0), 3)
            cv2.line(vis, (x1, y2), (x1 + d, y2), (255, 255, 0), 3)
            cv2.line(vis, (x1, y2), (x1, y2 - d), (255, 255, 0), 3)
            cv2.line(vis, (x2, y2), (x2 - d, y2), (255, 255, 0), 3)
            cv2.line(vis, (x2, y2), (x2, y2 - d), (255, 255, 0), 3)

            # Motion trajectory trail
            traj = registry.profiles[focus_target_id].trajectory
            pts = [(int(pt[0]), int(pt[1])) for pt in traj]
            for i in range(1, len(pts)):
                alpha = float(i) / max(1, len(pts) - 1)
                thick = max(1, int(4 * alpha))
                cv2.line(vis, pts[i - 1], pts[i], (0, 230, 255), thick, cv2.LINE_AA)
            if len(pts) > 0:
                cv2.circle(vis, pts[-1], 6, (0, 255, 255), -1)

            # Target Banner
            label = f"TARGET WORKER #{focus_target_id} [{target_r.get('helmet_color','').upper()}] - {target_r['status']}"
            cv2.putText(vis, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)

            # Top HUD in Focus Mode
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (20, 20, 30), -1)
            hud = f"FOCUS MODE ACTIVE: TRACKING WORKER #{focus_target_id} ONLY (TENACITY AUDIT)"
            cv2.putText(vis, hud, (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)

            cv2.imwrite("snapshots/verification_evidence/focus_mode_worker3_sample.jpg", vis)
            print(f"Saved focus mode sample to snapshots/verification_evidence/focus_mode_worker3_sample.jpg with {len(traj)} trajectory points!")

cap.release()
