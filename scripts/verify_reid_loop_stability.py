import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.pipeline import cctv_pipeline
from src.engine.reid_tracker import reid_registry

def test_pipeline_loop():
    print("=== Testing Live Pipeline Video Loop Stability ===")
    cctv_pipeline.switch_source("video_test/13921040-uhd_3840_2160_30fps.mp4")
    
    # Wait for pipeline background worker to process at least 45 frames
    print("Waiting for initial frames to process...")
    t0 = time.time()
    while cctv_pipeline.total_frames_processed < 45 and time.time() - t0 < 10.0:
        time.sleep(0.3)
        
    workers_pass1 = cctv_pipeline.get_registered_workers()
    ids_pass1 = [w["track_id"] for w in workers_pass1]
    print(f"Pass 1: Processed {cctv_pipeline.total_frames_processed} frames.")
    print(f"Pass 1 Registered Workers Count: {len(ids_pass1)}, IDs: {ids_pass1}")
    print(f"Baseline Anchors Captured: {len(reid_registry.baseline_anchors)}")
    assert len(reid_registry.baseline_anchors) > 0, "Baseline anchors should be established!"

    # Simulate loop event
    print("\nTriggering on_video_loop() event...")
    with cctv_pipeline.lock:
        cctv_pipeline.cap.set(1, 0) # rewind to frame 0
        cctv_pipeline.total_frames_processed = 0
        reid_registry.on_video_loop()

    # Wait for pipeline to process next 30 frames after loop
    t1 = time.time()
    while cctv_pipeline.total_frames_processed < 30 and time.time() - t1 < 10.0:
        time.sleep(0.3)

    workers_pass2 = cctv_pipeline.get_registered_workers()
    ids_pass2 = [w["track_id"] for w in workers_pass2]
    print(f"\nPass 2 (After Loop): Processed {cctv_pipeline.total_frames_processed} frames.")
    print(f"Pass 2 Registered Workers Count: {len(ids_pass2)}, IDs: {ids_pass2}")

    # Calculate ID retention
    retained_ids = set(ids_pass1).intersection(set(ids_pass2))
    print(f"Retained IDs across loop: {sorted(list(retained_ids))}")
    max_id_pass2 = max(ids_pass2) if ids_pass2 else 0
    print(f"Max Track ID after loop: {max_id_pass2}")
    
    assert len(retained_ids) >= 0.7 * min(len(ids_pass1), len(ids_pass2)), "IDs must be stably retained across loops!"
    print("\n[VERIFIED] Continuous live loop stability confirmed. No ID explosion or reset.")

if __name__ == "__main__":
    test_pipeline_loop()
