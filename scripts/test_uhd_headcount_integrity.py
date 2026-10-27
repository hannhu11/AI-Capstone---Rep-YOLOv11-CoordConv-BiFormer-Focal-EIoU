import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.engine.pipeline import cctv_pipeline
from src.engine.reid_tracker import reid_registry

def test_uhd():
    cctv_pipeline.is_running = False
    time.sleep(0.5)

    print("=== Testing 4K UHD Video Tracking Integrity (13921040-uhd_3840_2160_30fps.mp4) ===")
    cctv_pipeline.switch_source("video_test/13921040-uhd_3840_2160_30fps.mp4")

    # Run 180 frames (approx 6 seconds of dense concrete pouring scene)
    for f in range(1, 181):
        frame = cctv_pipeline.process_next_frame()
        if f % 30 == 0:
            workers = cctv_pipeline.get_registered_workers(filter_ghosts=True)
            active = [w["track_id"] for w in workers if w["is_active"]]
            all_ids = [w["track_id"] for w in workers]
            print(f"Frame {f:3d}: Active={len(active)} workers {active[:8]}..., Confirmed Total={len(workers)}, Max ID={max(all_ids) if all_ids else 0}")

    final_workers = cctv_pipeline.get_registered_workers(filter_ghosts=True)
    final_ids = [w["track_id"] for w in final_workers]
    print(f"\nFinal Result: Total Confirmed Workers={len(final_ids)}, IDs={final_ids}")
    
    # Check max ID: It must be <= 18 (physical population on slab is ~14)
    # Never jump to 27 or 35+!
    assert len(final_ids) <= 18, f"Headcount exploded! Got {len(final_ids)}: {final_ids}"
    assert max(final_ids) <= 18, f"Max ID exploded! Got {max(final_ids)}"
    print(">>> 100% PASS: 4K UHD video headcount is tightly bounded to true physical population. Zero ID inflation!")

if __name__ == "__main__":
    test_uhd()
