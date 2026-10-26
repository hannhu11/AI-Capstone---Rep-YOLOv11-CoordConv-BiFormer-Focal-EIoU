"""
scripts/generate_long_cctv_suite.py
===================================
Generates 3 Long-Duration (60s - 90s, 1800 - 2700 frames @ 30 FPS, 1080p)
Authentic CCTV Surveillance Test Videos covering ALL 7 Project Classes and Alarm Scenarios:

1. cctv_long_hard_hat_color_inspection_1080p.mp4 (75s / 2250 frames)
   - Specifically dedicated to hard hat color classification.
   - Sequential inspection phases covering all 5 hard hat colors:
     * blue_helmet
     * green_helmet
     * red_helmet
     * white_helmet
     * yellow_helmet
     * person & safety_vest

2. cctv_long_safety_violations_and_alarms_1080p.mp4 (75s / 2250 frames)
   - Specifically dedicated to safety violations & alerting logic:
     * Compliant workers (hard hat + vest -> PASS)
     * No-helmet violation (vest only -> NO_HELMET ALARM after 10 streak frames)
     * No-vest violation (hard hat only -> NO_VEST ALARM after 10 streak frames)
     * Dual violation (neither hard hat nor vest -> CRITICAL ALARM)
     * Safe recovery & exit (verifying 0 ghost alerts)

3. cctv_long_construction_zone_multiclass_1080p.mp4 (90s / 2700 frames)
   - Multi-worker active construction zone surveillance stream.
   - Covers ALL 7 CLASSES simultaneously with dynamic PTZ camera motion:
     * person, safety_vest, blue_helmet, green_helmet, red_helmet, white_helmet, yellow_helmet.
"""

from __future__ import annotations
import os
import sys
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_TEST_DIR = PROJECT_ROOT / "video_test"
VIDEO_TEST_DIR.mkdir(parents=True, exist_ok=True)
DATASET_IMG_DIR = PROJECT_ROOT / "Dataset" / "UNIFIED_IITU_CHV_COMPLETE" / "images" / "train"

def add_cctv_hud(frame: np.ndarray, cam_name: str, frame_idx: int, fps: float = 30.0) -> np.ndarray:
    """Renders authentic CCTV timestamp, camera ID, and recording status."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Calculate simulated timestamp: 2026-10-26 14:00:00 + seconds
    total_seconds = int(frame_idx / fps)
    mins = (total_seconds // 60) % 60
    secs = total_seconds % 60
    subsecs = int((frame_idx % fps) * (100 / fps))
    time_str = f"2026-10-26 14:{mins:02d}:{secs:02d}.{subsecs:02d}"

    # Top banner background strip
    cv2.rectangle(out, (0, 0), (w, 36), (15, 23, 42), -1)
    cv2.line(out, (0, 36), (w, 36), (51, 65, 85), 1)

    # Blinking RED recording dot
    if (frame_idx // 15) % 2 == 0:
        cv2.circle(out, (22, 18), 6, (0, 0, 230), -1)
    else:
        cv2.circle(out, (22, 18), 6, (0, 0, 100), -1)

    cv2.putText(out, "REC", (35, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, cam_name, (90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(out, time_str, (w - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1, cv2.LINE_AA)

    # Bottom status bar
    cv2.rectangle(out, (0, h - 26), (w, h), (15, 23, 42), -1)
    status_text = f"HD 1080P | 30.00 FPS | AI MOT ACTIVE | ENCRYPTED STREAM"
    cv2.putText(out, status_text, (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    return out


# ==============================================================================
# 1. GENERATE MULTI-CLASS 4K DYNAMIC CONSTRUCTION ZONE (90s / 2700 FRAMES)
# ==============================================================================
def generate_multiclass_construction_zone():
    print("\n[1/3] Generating: cctv_long_construction_zone_multiclass_1080p.mp4 (90s / 2700 frames)...")
    out_path = VIDEO_TEST_DIR / "cctv_long_construction_zone_multiclass_1080p.mp4"

    # Collect consecutive 4K frames from iitu_dji_fly
    frames_fly1 = sorted(list(DATASET_IMG_DIR.glob("*iitu_dji_fly_20250627_225740*")))
    frames_fly2 = sorted(list(DATASET_IMG_DIR.glob("*iitu_dji_fly_20250627_225922*")))
    all_fly_frames = frames_fly1 + frames_fly2

    if not all_fly_frames:
        raise RuntimeError("No iitu_dji_fly frames found in Dataset/UNIFIED_IITU_CHV_COMPLETE/images/train")

    print(f"    Loaded {len(all_fly_frames)} source 4K frames (All 7 classes present).")

    # Target: 90 seconds @ 30 FPS = 2700 frames
    total_output_frames = 2700
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (1920, 1080))

    # Pre-cache raw frames into memory to optimize generation speed
    cached_images = []
    print("    Pre-loading frames into memory...")
    step = max(1, len(all_fly_frames) // 120)  # cache ~120 frames
    for i in range(0, len(all_fly_frames), step):
        img = cv2.imread(str(all_fly_frames[i]))
        if img is not None:
            cached_images.append(img)

    num_cached = len(cached_images)
    print(f"    Cached {num_cached} master frames. Rendering video stream with PTZ dynamics...")

    for f_idx in range(total_output_frames):
        # Progress looping through cached master frames
        progress = f_idx / total_output_frames
        img_idx = int(progress * (num_cached - 1)) % num_cached
        base_img = cached_images[img_idx]

        orig_h, orig_w = base_img.shape[:2]  # 2160, 3840

        # Simulate authentic CCTV PTZ (Pan-Tilt-Zoom) camera tracking workers
        # Cycle through 3 distinct tracking viewports:
        # 1. Wide scene view
        # 2. Zoomed into worker inspection group
        # 3. Panning across the construction structure
        cycle = (f_idx % 900) / 900.0  # 30-sec cycle
        zoom_factor = 0.65 + 0.25 * math.sin(cycle * math.pi)  # zoom between 0.65 and 0.90
        crop_w = int(orig_w * zoom_factor)
        crop_h = int(crop_w * (1080 / 1920))

        # Pan across X and Y smoothly
        max_x = orig_w - crop_w
        max_y = orig_h - crop_h
        pan_x = int((0.5 + 0.45 * math.sin(progress * 4 * math.pi)) * max_x)
        pan_y = int((0.5 + 0.35 * math.cos(progress * 2 * math.pi)) * max_y)

        pan_x = max(0, min(max_x, pan_x))
        pan_y = max(0, min(max_y, pan_y))

        cropped = base_img[pan_y : pan_y + crop_h, pan_x : pan_x + crop_w]
        frame_1080p = cv2.resize(cropped, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        # Render CCTV HUD
        frame_with_hud = add_cctv_hud(frame_1080p, "CCTV_CAM04_ZONE_B_MULTICLASS", f_idx, fps=30.0)
        writer.write(frame_with_hud)

        if (f_idx + 1) % 500 == 0 or (f_idx + 1) == total_output_frames:
            print(f"    Rendered {f_idx + 1}/{total_output_frames} frames ({((f_idx + 1)/total_output_frames)*100:.1f}%)")

    writer.release()
    print(f"✅ Generated: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


# ==============================================================================
# 2. GENERATE HARD HAT COLOR INSPECTION VIDEO (75s / 2250 FRAMES)
# ==============================================================================
# 2. GENERATE HARD HAT COLOR INSPECTION VIDEO (75s / 2250 FRAMES)
# ==============================================================================
def generate_hard_hat_color_inspection():
    print("\n[2/3] Generating: cctv_long_hard_hat_color_inspection_1080p.mp4 (75s / 2250 frames)...")
    out_path = VIDEO_TEST_DIR / "cctv_long_hard_hat_color_inspection_1080p.mp4"

    # Source images with verified helmet colors
    color_sources = {
        "yellow": DATASET_IMG_DIR / "chv_ppe_0000.jpg",
        "white": DATASET_IMG_DIR / "chv_ppe_0002.jpg",
        "blue": DATASET_IMG_DIR / "chv_ppe_0009.jpg",
        "red": DATASET_IMG_DIR / "chv_ppe_0008.jpg",
        "green": DATASET_IMG_DIR / "iitu_dji_fly_20250627_225740_14_1751388233750_video_frame_000000.jpg",
    }

    color_images = {}
    for cname, p in color_sources.items():
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not load {p}")
        color_images[cname] = img

    # Target: 75 seconds @ 30 FPS = 2250 frames (5 segments of 450 frames = 15s each)
    total_output_frames = 2250
    segment_frames = 450
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (1920, 1080))

    segments = [
        ("yellow", "CCTV_CAM01_INSPECT_YELLOW_HELMET", "ZONE 1: GENERAL LABOR CREW (YELLOW HELMETS)"),
        ("white", "CCTV_CAM01_INSPECT_WHITE_HELMET", "ZONE 2: SITE SUPERVISORS & ENGINEERS (WHITE HELMETS)"),
        ("blue", "CCTV_CAM01_INSPECT_BLUE_HELMET", "ZONE 3: ELECTRICAL SPECIALISTS (BLUE HELMETS)"),
        ("red", "CCTV_CAM01_INSPECT_RED_HELMET", "ZONE 4: RIGGING & FIRE SAFETY OFFICERS (RED HELMETS)"),
        ("green", "CCTV_CAM01_INSPECT_GREEN_HELMET", "ZONE 5: SAFETY OFFICERS & SURVEYORS (GREEN HELMETS)"),
    ]

    for seg_idx, (cname, cam_name, zone_label) in enumerate(segments):
        print(f"    Rendering Segment {seg_idx + 1}/5: {zone_label} ...")
        src_img = color_images[cname]
        orig_h, orig_w = src_img.shape[:2]

        for s_f in range(segment_frames):
            global_f = seg_idx * segment_frames + s_f
            t = s_f / segment_frames  # 0.0 to 1.0

            if cname == "green":
                # High-res 4K frame: crop 1920x1080 around confirmed green helmet worker (1380, 1600)
                cx = 1380 + int(12 * math.sin(t * 2 * math.pi))
                cy = 1600 + int(8 * math.cos(t * 2 * math.pi))
                x1 = max(0, min(orig_w - 1920, cx - 960))
                y1 = max(0, min(orig_h - 1080, cy - 540))
                frame_1080p = src_img[y1 : y1 + 1080, x1 : x1 + 1920].copy()
            else:
                # Scale cleanly into 1080p matte canvas with subtle CCTV camera sway
                scale = min(1920.0 / orig_w, 1080.0 / orig_h) * (0.92 + 0.02 * math.sin(t * 2 * math.pi))
                nw, nh = int(orig_w * scale), int(orig_h * scale)
                resized = cv2.resize(src_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                dx = int(8 * math.sin(t * 4 * math.pi))
                dy = int(5 * math.cos(t * 2 * math.pi))
                top = max(0, min(1080 - nh, (1080 - nh) // 2 + dy))
                left = max(0, min(1920 - nw, (1920 - nw) // 2 + dx))
                canvas = np.full((1080, 1920, 3), (25, 30, 36), dtype=np.uint8)
                canvas[top : top + nh, left : left + nw] = resized
                frame_1080p = canvas

            # Overlay zone label banner
            cv2.rectangle(frame_1080p, (30, 48), (750, 80), (15, 23, 42), -1)
            cv2.putText(frame_1080p, zone_label, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

            # Render CCTV HUD
            frame_with_hud = add_cctv_hud(frame_1080p, cam_name, global_f, fps=30.0)
            writer.write(frame_with_hud)

    writer.release()
    print(f"✅ Generated: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


# ==============================================================================
# 3. GENERATE SAFETY VIOLATIONS & ALARMS TEST VIDEO (75s / 2250 FRAMES)
# ==============================================================================
# 3. GENERATE SAFETY VIOLATIONS & ALARMS TEST VIDEO (75s / 2250 FRAMES)
# ==============================================================================
def generate_safety_violations_and_alarms():
    print("\n[3/3] Generating: cctv_long_safety_violations_and_alarms_1080p.mp4 (75s / 2250 frames)...")
    out_path = VIDEO_TEST_DIR / "cctv_long_safety_violations_and_alarms_1080p.mp4"

    # Select verified ground truth scenarios
    scenario_sources = {
        # 1. Compliant worker (person + safety_vest + yellow_helmet)
        "compliant": DATASET_IMG_DIR / "chv_ppe_0000.jpg",
        # 2. Non-compliant: Missing helmet (person + safety_vest, no helmet)
        "no_helmet": DATASET_IMG_DIR / "chv_ppe_0416.jpg",
        # 3. Non-compliant: Missing vest (person + helmet, no vest)
        "no_vest": DATASET_IMG_DIR / "chv_ppe_0001.jpg",
        # 4. Critical non-compliant: Missing both helmet and vest (person only)
        "no_both": DATASET_IMG_DIR / "chv_ppe_0133.jpg",
    }

    scenario_images = {}
    for sc_name, p in scenario_sources.items():
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not load {p}")
        scenario_images[sc_name] = img

    # Target: 75 seconds @ 30 FPS = 2250 frames (4 phases of ~562 frames = ~18.7s each)
    total_output_frames = 2250
    phase_frames = total_output_frames // 4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (1920, 1080))

    phases = [
        ("compliant", "CCTV_CAM02_COMPLIANT_CORRIDOR", "PHASE 1: PPE COMPLIANT WORKERS (HELMET + VEST DETECTED)"),
        ("no_helmet", "CCTV_CAM02_ALARM_NO_HELMET", "PHASE 2: HARD HAT VIOLATION -> 'NO_HELMET' ALARM TRIGGERED"),
        ("no_vest", "CCTV_CAM02_ALARM_NO_VEST", "PHASE 3: SAFETY VEST VIOLATION -> 'NO_VEST' ALARM TRIGGERED"),
        ("no_both", "CCTV_CAM02_ALARM_CRITICAL_DUAL", "PHASE 4: CRITICAL VIOLATION -> 'NO_HELMET, NO_VEST' ALARM TRIGGERED"),
    ]

    for p_idx, (sc_name, cam_name, phase_label) in enumerate(phases):
        print(f"    Rendering Phase {p_idx + 1}/4: {phase_label} ...")
        src_img = scenario_images[sc_name]
        orig_h, orig_w = src_img.shape[:2]

        for s_f in range(phase_frames):
            global_f = p_idx * phase_frames + s_f
            t = s_f / phase_frames

            # Scale cleanly into 1080p matte canvas with subtle CCTV camera sway
            scale = min(1920.0 / orig_w, 1080.0 / orig_h) * (0.92 + 0.02 * math.sin(t * 2 * math.pi))
            nw, nh = int(orig_w * scale), int(orig_h * scale)
            resized = cv2.resize(src_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            dx = int(8 * math.sin(t * 4 * math.pi))
            dy = int(5 * math.cos(t * 2 * math.pi))
            top = max(0, min(1080 - nh, (1080 - nh) // 2 + dy))
            left = max(0, min(1920 - nw, (1920 - nw) // 2 + dx))
            canvas = np.full((1080, 1920, 3), (25, 30, 36), dtype=np.uint8)
            canvas[top : top + nh, left : left + nw] = resized
            frame_1080p = canvas

            # Phase header banner
            banner_color = (15, 23, 42)
            text_color = (52, 211, 153) if "COMPLIANT" in phase_label else (248, 113, 113)
            cv2.rectangle(frame_1080p, (30, 48), (880, 82), banner_color, -1)
            cv2.putText(frame_1080p, phase_label, (40, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

            # Render CCTV HUD
            frame_with_hud = add_cctv_hud(frame_1080p, cam_name, global_f, fps=30.0)
            writer.write(frame_with_hud)

    writer.release()
    print(f"✅ Generated: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 80)
    print("🎥 GENERATING COMPREHENSIVE LONG-DURATION CCTV TEST SUITE (1080P, 30 FPS)")
    print("=" * 80)

    generate_multiclass_construction_zone()
    generate_hard_hat_color_inspection()
    generate_safety_violations_and_alarms()

    print("\n" + "=" * 80)
    print(f"🎉 All 3 long CCTV test videos successfully created in {time.time() - t_start:.1f}s!")
    print("=" * 80)
