import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.reid_tracker import PersistentTrackRegistry
from src.engine.spatial_safety import SpatialSafetyEngine
from src.config import config

def test_single_worker_video_integrity():
    video_path = "video_test/construction_site_workers_1080p.mp4"
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"

    detector = SafetyObjectDetector(
        model_path=config.model_path,
        conf_thresh=0.45,
        iou_thresh=0.45,
        device=config.device,
        use_half=False,
    )
    tracker = BYTETracker(track_thresh=0.45, track_low_thresh=0.15, match_thresh=0.70)
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.40)
    safety = SpatialSafetyEngine()

    print("=== Testing Single-Worker Video Tracking (construction_site_workers_1080p.mp4) ===")
    print("Processing 800 frames including edge transitions...")

    t0 = time.time()
    for f in range(1, 801):
        ret, frame = cap.read()
        if not ret:
            break

        dets, person_dets, _ = detector.predict(frame)

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

        tracks = tracker.update(person_dets, frame=frame)

        active_canonical = []
        for track in tracks:
            tb = track.tlbr
            matched_color = None
            for d in dets:
                if "helmet" in d["name"]:
                    hb = d["box"]
                    if not (hb[2] < tb[0] or hb[0] > tb[2] or hb[3] < tb[1] or hb[1] > tb[1] + 0.40 * (tb[3] - tb[1])):
                        matched_color = d.get("color") or d["name"].replace("_helmet", "")
                        break

            cid = registry.get_or_create_canonical_id(
                raw_track_id=track.track_id,
                box=tb,
                frame=frame,
                helmet_color=matched_color,
                frame_idx=f,
            )
            track.canonical_id = cid
            active_canonical.append(cid)

        # Run safety evaluation to verify no false violations at borders
        worker_reports, alert_events = safety.evaluate_frame(tracks, dets, frame)
        for r in worker_reports:
            wid = r["track_id"]
            if wid in registry.profiles:
                registry.profiles[wid].status = r["status"]
                registry.profiles[wid].violations = r["violations"]

        registry.update_frame_end(active_canonical, f)

        if f % 100 == 0:
            workers = registry.get_registered_workers(filter_ghosts=True)
            active = [w["track_id"] for w in workers if w["is_active"]]
            all_ids = [w["track_id"] for w in workers]
            print(f"Frame {f:3d}: Active={active}, Registered Total={len(workers)}, IDs={all_ids}")

    cap.release()
    elapsed = time.time() - t0
    final_workers = registry.get_registered_workers(filter_ghosts=True)
    final_ids = [w["track_id"] for w in final_workers]
    print(f"\nFinal Result: Processed 800 frames in {elapsed:.1f}s ({800/elapsed:.1f} FPS)")
    print(f"Registered Workers: {len(final_ids)} | IDs: {final_ids}")

    assert len(final_ids) == 1, f"Expected exactly 1 worker, got {len(final_ids)}: {final_ids}"
    assert final_ids[0] == 1, f"Expected Worker ID #1, got #{final_ids[0]}"
    assert final_workers[0]["status"] == "COMPLIANT", f"Expected COMPLIANT, got {final_workers[0]['status']}"

    print(">>> SUCCESS: Single worker video has exactly 1 persistent ID (Worker #1) with COMPLIANT status!")

if __name__ == "__main__":
    test_single_worker_video_integrity()
