import pytest
import numpy as np
import cv2
from src.engine.reid_tracker import (
    PersistentTrackRegistry,
    PersistentWorkerProfile,
    extract_torso_roi,
    compute_appearance_hist,
    hist_correlation,
)


def test_extract_torso_roi_edge_and_bending():
    """Verify that torso extraction adapts properly to border truncation and bending."""
    h_img, w_img = 1080, 1920

    # 1. Normal upright box
    box_normal = np.array([200, 200, 300, 500])
    tx1, ty1, tx2, ty2 = extract_torso_roi(box_normal, h_img, w_img)
    assert ty1 >= 200 + int(0.18 * 300)
    assert ty2 <= 200 + int(0.62 * 300)

    # 2. Bottom-edge truncated box (legs off screen, y2 = 1075)
    box_bottom = np.array([200, 900, 300, 1075])  # bh = 175, bw = 100
    tx1, ty1, tx2, ty2 = extract_torso_roi(box_bottom, h_img, w_img)
    # Visible portion is head and torso, so ty2 extends down to image bottom
    assert ty2 == 1075
    assert ty1 == 900 + int(0.20 * 175)

    # 3. Stooping/bending box (aspect < 1.45)
    box_bending = np.array([400, 400, 550, 550])  # bw = 150, bh = 150, aspect = 1.0
    tx1, ty1, tx2, ty2 = extract_torso_roi(box_bending, h_img, w_img)
    assert ty1 == 400 + int(0.15 * 150)
    assert ty2 == 400 + int(0.70 * 150)


def test_anti_hijack_guard_prevents_lock_stealing():
    """Verify Layer-2 Anti-Hijack Guard breaks raw binding when a distinct worker hijacks the track."""
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.40)

    # Synthetic frames
    frame_yellow_vest = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_yellow_vest[:, :] = (30, 220, 220)  # Bright yellow/green

    frame_grey_shirt = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_grey_shirt[:, :] = (90, 90, 90)  # Dark grey

    box_a = np.array([300, 300, 400, 600])

    # Establish Worker #2 as a confirmed yellow vest worker
    for f in range(1, 12):
        cid = registry.get_or_create_canonical_id(
            raw_track_id=2,
            box=box_a,
            frame=frame_yellow_vest,
            helmet_color="yellow",
            frame_idx=f,
        )
        assert cid == 1  # First worker gets canonical ID 1 (or 2 depending on index)
        registry.update_frame_end([cid], f)

    assert registry.profiles[cid].is_confirmed
    worker_can_id = cid

    # Cross-over event: ByteTrack assigns raw track 2 to the grey shirt worker!
    box_b = np.array([350, 320, 450, 620])
    # The grey shirt worker has a blue helmet and dark grey clothing
    new_cid = registry.get_or_create_canonical_id(
        raw_track_id=2,
        box=box_b,
        frame=frame_grey_shirt,
        helmet_color="blue",
        frame_idx=15,
    )

    # ANTI-HIJACK GUARANTEE:
    # The grey shirt worker MUST NOT be given Worker #1's ID!
    # A new ID must be created, and Worker #1 remains intact in memory!
    assert new_cid != worker_can_id, "Anti-hijack guard failed! Lock was hijacked by grey shirt worker!"
    assert worker_can_id in registry.profiles, "Original worker profile was corrupted!"
    assert registry.profiles[worker_can_id].helmet_color == "yellow"


def test_single_worker_scene_prior_zero_inflation():
    """Verify single-worker video never mints Worker #2 even across long occlusions / border exits."""
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.40)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :] = (40, 200, 200)

    # Worker #1 active in first 20 frames
    box = np.array([100, 100, 200, 400])
    for f in range(1, 21):
        cid = registry.get_or_create_canonical_id(
            raw_track_id=1,
            box=box,
            frame=frame,
            helmet_color="white",
            frame_idx=f,
        )
        assert cid == 1
        registry.update_frame_end([cid], f)

    # Worker walks off screen for 300 frames
    for f in range(21, 321):
        registry.update_frame_end([], f)

    assert len([p for p in registry.profiles.values() if p.is_active]) == 0
    assert len(registry.profiles) == 1

    # Worker re-enters from bottom border (y2 = 1070) with new raw track ID 88
    box_reenter = np.array([500, 850, 620, 1070])
    new_cid = registry.get_or_create_canonical_id(
        raw_track_id=88,
        box=box_reenter,
        frame=frame,
        helmet_color="white",
        frame_idx=322,
    )

    # Single-worker prior MUST match Worker #1!
    assert new_cid == 1, f"Expected Worker #1, but minted #{new_cid}!"
    assert len(registry.profiles) == 1, f"Expected 1 total profile, found {len(registry.profiles)}!"


def test_spatio_temporal_profile_consolidation():
    """Verify automatic consolidation merges split duplicate IDs from posture changes."""
    registry = PersistentTrackRegistry(reid_affinity_thresh=0.40)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :] = (50, 180, 180)

    # Profile #3 active from frame 1 to 30
    box_3 = np.array([500, 500, 600, 800])
    for f in range(1, 31):
        cid = registry.get_or_create_canonical_id(
            raw_track_id=3,
            box=box_3,
            frame=frame,
            helmet_color="yellow",
            frame_idx=f,
        )
        registry.update_frame_end([cid], f)

    # Frame 31 to 50: worker temporarily lost
    for f in range(31, 51):
        registry.update_frame_end([], f)

    # Frame 51 to 80: worker re-detected with raw ID 17
    box_17 = np.array([520, 510, 620, 810])
    for f in range(51, 81):
        cid = registry.get_or_create_canonical_id(
            raw_track_id=17,
            box=box_17,
            frame=frame,
            helmet_color="yellow",
            frame_idx=f,
        )
        registry.update_frame_end([cid], f)

    # Both profiles become inactive at frame 100
    for f in range(81, 121):
        registry.update_frame_end([], f)

    # Run consolidation at frame 120
    registry.consolidate_profiles(current_frame=120)

    # Duplicate Profile #2 (minted by raw 17) should be merged into Profile #1 (canonical)!
    assert len(registry.profiles) == 1, f"Consolidation failed! Remaining profiles: {list(registry.profiles.keys())}"
    assert 1 in registry.profiles
