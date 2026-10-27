import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.engine.pipeline import cctv_pipeline
from src.engine.reid_tracker import reid_registry

def test_clean():
    # Stop the background thread so the test has exclusive control over frames
    cctv_pipeline.is_running = False
    time.sleep(0.5)

    print("=== Testing Single Worker Frame-by-Frame Exclusively ===")
    cctv_pipeline.switch_source("video_test/construction_site_workers_1080p.mp4")

    # Now run 500 frames exclusively
    for f in range(1, 501):
        frame = cctv_pipeline.process_next_frame()
        if f % 50 == 0:
            workers = cctv_pipeline.get_registered_workers(filter_ghosts=True)
            active = [w["track_id"] for w in workers if w["is_active"]]
            all_ids = [w["track_id"] for w in workers]
            print(f"Frame {f:3d}: Active={active}, Total Confirmed={len(workers)}, IDs={all_ids}")

    final_workers = cctv_pipeline.get_registered_workers(filter_ghosts=True)
    final_ids = [w["track_id"] for w in final_workers]
    print(f"\nFinal Result: Confirmed Workers Count = {len(final_ids)}, IDs = {final_ids}")
    assert len(final_ids) == 1, f"Expected exactly 1 worker, got {len(final_ids)}: {final_ids}"
    assert final_ids[0] == 1, f"Expected Worker #1, got #{final_ids[0]}"
    print(">>> 100% PASS: Single-worker video maintains exactly 1 persistent ID (Worker #1) throughout 500 frames!")

if __name__ == "__main__":
    test_clean()
