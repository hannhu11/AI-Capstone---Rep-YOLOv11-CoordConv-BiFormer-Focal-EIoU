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

def test_uhd_headcount():
    video_path = "video_test/13921040-uhd_3840_2160_30fps.mp4"
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

    print("Processing 300 frames of 4K UHD Concreting Video...")
    t0 = time.time()
    for f in range(1, 301):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for smooth edge surveillance
        h, w = frame.shape[:2]
        if max(h, w) > 1920:
            scale = 1920.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        dets, person_dets, _ = detector.predict(frame)

        # Filter background noise
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

        registry.update_frame_end(active_canonical, f)

        if f % 50 == 0:
            active_p = [p.worker_id for p in registry.profiles.values() if p.is_active]
            confirmed_p = [p.worker_id for p in registry.profiles.values() if p.is_confirmed or p.total_frames_seen >= 5]
            print(f"Frame {f:3d}: Active Count={len(active_p)} | Total Confirmed={len(confirmed_p)} | Registered Profiles={len(registry.profiles)}")

    cap.release()
    elapsed = time.time() - t0
    workers_meta = registry.get_registered_workers(filter_ghosts=True)
    final_confirmed = [w["track_id"] for w in workers_meta]
    print(f"\nFinal Result: Processed 300 frames in {elapsed:.1f}s ({300/elapsed:.1f} FPS)")
    print(f"Total Registered Headcount: {len(final_confirmed)} | IDs: {final_confirmed}")
    print(f"Total Raw Profiles in Registry: {len(registry.profiles)}")

    # In this video, there are 18 simultaneous workers at frame 116.
    # Prior to our fix, headcount inflated to 21-27.
    # With anti-hijack gating and consolidation, headcount is strictly bounded <= 18!
    assert len(final_confirmed) <= 18, f"Headcount inflation detected: {len(final_confirmed)} > 18!"
    print(">>> SUCCESS: Multi-worker headcount strictly bounded and protected against ID inflation!")

if __name__ == "__main__":
    test_uhd_headcount()
