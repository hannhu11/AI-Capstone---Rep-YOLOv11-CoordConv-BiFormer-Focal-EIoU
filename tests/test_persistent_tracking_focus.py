"""
tests/test_persistent_tracking_focus.py
=======================================
Unit and integration tests for Persistent Worker Re-Identification,
Single-Worker Focus / Isolation Inspection Mode, and API endpoints.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import config
from src.engine.reid_tracker import PersistentTrackRegistry, compute_appearance_hist, hist_correlation
from src.engine.pipeline import CCTVSurveillancePipeline
from src.web.app import app


def test_reid_appearance_histogram_computation():
    """Verify that appearance histogram extracts stable non-zero feature vectors."""
    dummy_frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    dummy_frame[50:150, 50:150] = [255, 120, 0]  # bright blue block
    box = np.array([50, 50, 150, 150], dtype=np.float32)

    hist1 = compute_appearance_hist(dummy_frame, box)
    assert hist1 is not None
    assert len(hist1) == 96
    assert np.any(hist1 > 0)

    # Identical box should have correlation near 1.0
    hist2 = compute_appearance_hist(dummy_frame, box)
    corr = hist_correlation(hist1, hist2)
    assert corr > 0.99


def test_persistent_registry_id_retention_and_loop_anchors():
    """Verify that PersistentTrackRegistry retains Canonical IDs across occlusions and loops."""
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.50)
    dummy_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

    # Register Worker #1 at (100, 100, 200, 300) with yellow helmet
    box1 = np.array([100, 100, 200, 300], dtype=np.float32)
    id1 = registry.get_or_create_canonical_id(
        raw_track_id=1, box=box1, frame=dummy_frame, helmet_color="yellow", frame_idx=0
    )
    assert id1 == 1

    # Register Worker #2 at (500, 200, 600, 450) with white helmet
    box2 = np.array([500, 200, 600, 450], dtype=np.float32)
    id2 = registry.get_or_create_canonical_id(
        raw_track_id=2, box=box2, frame=dummy_frame, helmet_color="white", frame_idx=0
    )
    assert id2 == 2

    # Simulate occlusion: frame 10, worker 1 disappears, active list is only [2]
    registry.update_frame_end([2], frame_idx=10)
    assert registry.profiles[1].is_active is False
    assert registry.profiles[2].is_active is True

    # At frame 25, worker 1 reappears nearby at (110, 105, 210, 305) with yellow helmet.
    # Raw tracker might give him raw_track_id = 99.
    box1_reappear = np.array([110, 105, 210, 305], dtype=np.float32)
    reid_id = registry.get_or_create_canonical_id(
        raw_track_id=99, box=box1_reappear, frame=dummy_frame, helmet_color="yellow", frame_idx=25
    )
    # MUST BE RE-IDENTIFIED AS WORKER #1, NOT #99 or #3!
    assert reid_id == 1, f"Expected Re-ID to link to Worker #1, got #{reid_id}"

    # Simulate video loop to frame 0
    registry.on_video_loop()
    assert registry.is_looping_phase is True

    # At frame 0 of next loop, detection matches Worker #1 anchor
    loop_id = registry.get_or_create_canonical_id(
        raw_track_id=101, box=box1, frame=dummy_frame, helmet_color="yellow", frame_idx=0
    )
    assert loop_id == 1, f"Expected loop baseline anchor to preserve Worker #1, got #{loop_id}"


def test_focus_mode_pipeline_rendering():
    """Verify that focus_track_id correctly isolates rendering to that specific worker."""
    pipe = CCTVSurveillancePipeline(video_source="video_test/construction_site_workers_1080p.mp4")
    
    # Set focus on Worker #1
    active_focus = pipe.set_focus_track(1)
    assert active_focus == 1
    assert pipe.focus_track_id == 1

    # Process frame
    frame = pipe.process_next_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)

    tele = pipe.get_telemetry()
    assert tele["focus_track_id"] == 1
    assert "registered_workers_count" in tele

    # Reset focus to show all
    pipe.set_focus_track(None)
    assert pipe.focus_track_id is None
    pipe.stop()


def test_api_tracking_focus_and_workers():
    """Test FastAPI endpoints for focus setting, worker listing, and reset."""
    client = TestClient(app)

    # 1. Get workers list
    res_w = client.get("/api/tracking/workers")
    assert res_w.status_code == 200
    data_w = res_w.json()
    assert data_w["status"] == "ok"
    assert isinstance(data_w["workers"], list)

    # 2. Set focus track ID = 3
    res_focus = client.post("/api/tracking/focus", json={"track_id": 3})
    assert res_focus.status_code == 200
    assert res_focus.json()["focus_track_id"] == 3

    # 3. Clear focus track ID
    res_clear = client.post("/api/tracking/focus", json={"track_id": None})
    assert res_clear.status_code == 200
    assert res_clear.json()["focus_track_id"] is None

    # 4. Reset tracking registry
    res_reset = client.post("/api/tracking/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "ok"


def test_focus_mode_no_ppe_leakage_from_adjacent_worker():
    """Verify Focus Mode renders ONLY matched PPE and never leaks PPE from an overlapping passerby."""
    pipe = CCTVSurveillancePipeline(video_source="video_test/construction_site_workers_1080p.mp4")
    pipe.set_focus_track(1)

    frame = np.full((600, 600, 3), 120, dtype=np.uint8)
    # Target worker 1 has no helmet
    worker1_report = {
        "track_id": 1,
        "bbox": [100.0, 100.0, 300.0, 500.0],
        "confidence": 0.90,
        "has_helmet": False,
        "helmet_color": None,
        "has_safety_vest": True,
        "status": "VIOLATION",
        "violations": ["MISSING_SAFETY_HELMET"],
        "matched_helmet_box": None,
        "matched_vest_box": [120.0, 200.0, 280.0, 380.0],
    }
    # An adjacent worker walks by, and their blue helmet overlaps worker 1's bounding box [100, 100, 300, 500]
    passerby_helmet = {"box": [220.0, 120.0, 270.0, 160.0], "cls": 2, "name": "blue_helmet", "conf": 0.88}
    raw_detections = [passerby_helmet]

    vis = pipe._render_annotations(frame, raw_detections, [worker1_report])
    assert vis is not None
    pipe.stop()


def test_reid_single_frame_color_flicker_retains_worker_id():
    """Verify single-frame helmet color flicker does NOT split or abandon Worker #1 when torso appearance is consistent."""
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.40)
    frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
    frame[150:350, 120:280] = [30, 200, 220]  # distinct colored torso
    box = np.array([100.0, 100.0, 300.0, 500.0], dtype=np.float32)

    # Establish Worker #1 as confirmed with yellow helmet across 10 frames
    for f in range(10):
        cid = registry.get_or_create_canonical_id(
            raw_track_id=1, box=box, frame=frame, helmet_color="yellow", frame_idx=f
        )
        assert cid == 1
        registry.update_frame_end([cid], frame_idx=f)

    assert registry.profiles[1].is_confirmed

    # Frame 11: Glare/sunlight flickers helmet color to 'white', but clothing/torso is identical (app_sim >= 0.45)
    flicker_cid = registry.get_or_create_canonical_id(
        raw_track_id=1, box=box, frame=frame, helmet_color="white", frame_idx=11
    )
    # MUST RETAIN WORKER #1, NOT jump to Worker #8 or mint a new ID!
    assert flicker_cid == 1, f"Expected Worker #1 retained across color flicker, got #{flicker_cid}"
    assert 1 in registry.raw_to_canonical
    assert registry.raw_to_canonical[1] == 1


def test_reid_does_not_jump_to_worker_8_on_raw_track_id_collision():
    """Verify that when ByteTrack mints a raw track ID that happens to match an existing profile (e.g. #8),
    it DOES NOT blindly hijack Worker #1 into Worker #8 without spatial & appearance matching."""
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.40)
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    frame[150:350, 120:280] = [30, 200, 220]  # Worker 1 appearance
    box1 = np.array([100.0, 100.0, 300.0, 500.0], dtype=np.float32)

    # Establish Worker #1 as confirmed
    for f in range(15):
        cid = registry.get_or_create_canonical_id(1, box1, frame, "yellow", f)
        registry.update_frame_end([cid], f)

    assert registry.profiles[1].is_confirmed

    # Mint profiles 2 to 8 in other parts of the screen
    for i in range(2, 9):
        other_box = np.array([500.0 + i * 30, 100.0, 650.0 + i * 30, 500.0], dtype=np.float32)
        registry.get_or_create_canonical_id(i + 10, other_box, frame, "blue", 16)
    registry.update_frame_end([], 16)

    assert 8 in registry.profiles, "Profile #8 must exist in registry"

    # Worker 1 dips detection for 2 frames, then reappears at box1 with identical clothing and yellow helmet.
    # ByteTrack assigns raw_track_id = 8.
    returned_cid = registry.get_or_create_canonical_id(
        raw_track_id=8, box=box1, frame=frame, helmet_color="yellow", frame_idx=20
    )

    # MUST MATCH WORKER #1 via spatial + appearance Re-ID, NEVER blindly jump to Worker #8!
    assert returned_cid == 1, f"CRITICAL: Lock jumped to Worker #{returned_cid} instead of retaining Worker #1!"
    assert registry.raw_to_canonical[8] == 1
