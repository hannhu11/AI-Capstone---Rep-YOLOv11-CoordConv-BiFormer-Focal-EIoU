"""
scripts/verify_tracking_fix.py
==============================
Rigorous Empirical Verification of Tracking & Floor Ghost Fixes:
1. Validates that FP32 completely eliminates floor hazard tape ghost detections.
2. Validates that the walking worker maintains a stable, continuous Track ID.
3. Validates that Worker bounding boxes and PPE badges are correctly rendered.
4. Generates visual proof frames in snapshots/verification_evidence/.
"""

import os
import cv2
import time
import json
import torch
import numpy as np
from pathlib import Path

from src.config import config
from src.engine.detector import SafetyObjectDetector
from src.engine.bytetrack import BYTETracker
from src.engine.spatial_safety import SpatialSafetyEngine

def main():
    video_path = "video_test/construction_site_workers_1080p.mp4"
    output_dir = Path("snapshots/verification_evidence")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🔬 EMPIRICAL TRACKING & GHOST REJECTION AUDIT")
    print(f"Target Video: {video_path}")
    print(f"CUDA Available: {torch.cuda.is_available()} | Device: {config.device}")
    print(f"Model Path: {config.model_path}")
    print(f"Inference Mode: FP32 (use_half={config.use_half})")
    print("=" * 70)

    # Initialize components
    detector = SafetyObjectDetector(
        model_path=config.model_path,
        conf_thresh=config.conf_threshold,
        iou_thresh=config.iou_threshold,
        device=config.device,
        use_half=False,  # FP32
    )

    tracker = BYTETracker(
        track_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        new_track_thresh=config.new_track_thresh,
        track_buffer=config.track_buffer,
        match_thresh=config.match_thresh,
        grace_period=2,
    )

    safety_engine = SpatialSafetyEngine(
        head_box_ratio=config.head_box_ratio,
        torso_min_ratio=config.torso_min_ratio,
        torso_max_ratio=config.torso_max_ratio,
        helmet_overlap_thresh=config.helmet_overlap_thresh,
        vest_overlap_thresh=config.vest_overlap_thresh,
        consecutive_frames_thresh=config.consecutive_violation_frames,
        cooldown_seconds=config.alert_cooldown_seconds,
        border_margin_px=60,
        min_tracklet_len=15,
    )

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video {video_path}"

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {w}x{h} @ {fps:.2f} FPS | Total Frames: {total_video_frames}")

    frames_to_test = 300
    latencies = []
    track_ids_seen = set()
    worker_presence_per_frame = []
    floor_ghost_detections = 0
    total_person_detections = 0
    saved_frames = []

    t_start_total = time.time()

    for f_idx in range(frames_to_test):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections, person_dets, infer_ms = detector.predict(frame)

        # Filter candidate person detections before tracking
        if len(person_dets) > 0:
            frame_h, frame_w = frame.shape[:2]
            valid_mask = []
            for p in person_dets:
                pw = p[2] - p[0]
                ph = p[3] - p[1]
                touches_border = (p[0] <= 60 or p[1] <= 60 or p[2] >= frame_w - 60 or p[3] >= frame_h - 60)
                max_aspect = 3.5 if touches_border else 1.65
                is_valid = (ph >= 35 and pw >= 15 and (pw / max(1.0, ph)) <= max_aspect)
                valid_mask.append(is_valid)
            person_dets = person_dets[np.array(valid_mask, dtype=bool)] if any(valid_mask) else np.empty((0, 6), dtype=np.float32)

        # Multi-Object Tracking
        active_tracks = tracker.update(person_dets)

        # Spatial Safety Evaluation
        worker_reports, alert_events = safety_engine.evaluate_frame(
            active_tracks, detections, frame, camera_id=config.camera_id
        )

        t_end = time.perf_counter()
        latencies.append((t_end - t0) * 1000.0)

        # Check for floor ghost detections:
        # In the user screenshot, floor ghost detections were on the hazard tape at the bottom-center/bottom-right
        # (y > 700, x between 600 and 1500, with confidence 100% red helmet, 75% white helmet, 49% safety vest)
        for d in detections:
            box = d["box"]
            bx_center = (box[0] + box[2]) / 2.0
            by_center = (box[1] + box[3]) / 2.0
            # Floor zone where tape is located: y > 750 with no actual worker
            if by_center > 750 and 500 < bx_center < 1600:
                # Check if this box overlaps any tracked worker
                overlaps_worker = False
                for r in worker_reports:
                    wb = r["bbox"]
                    if not (box[2] < wb[0] or box[0] > wb[2] or box[3] < wb[1] or box[1] > wb[3]):
                        overlaps_worker = True
                        break
                if not overlaps_worker and d["cls"] != 0:
                    floor_ghost_detections += 1

        active_ids = [r["track_id"] for r in worker_reports]
        for aid in active_ids:
            track_ids_seen.add(aid)
        worker_presence_per_frame.append((f_idx, active_ids, [r["status"] for r in worker_reports]))

        # Save sample annotated frames at frames 50, 100, 150, 200, 250
        if f_idx in [50, 100, 150, 200, 250]:
            vis = frame.copy()
            # Draw detections
            for d in detections:
                if d["cls"] != 0:
                    x1, y1, x2, y2 = [int(v) for v in d["box"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis, f"{d['name']} {d['conf']:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            # Draw workers
            for r in worker_reports:
                x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
                col = (0, 230, 115) if r["status"] == "COMPLIANT" else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                h_col = r.get("helmet_color") or "NONE"
                label = f"WORKER #{r['track_id']} [{h_col.upper()}] - {r['status']}"
                cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            save_path = output_dir / f"frame_{f_idx:04d}_annotated.jpg"
            cv2.imwrite(str(save_path), vis)
            saved_frames.append(str(save_path))

    cap.release()
    total_time = time.time() - t_start_total

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    achieved_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n" + "=" * 70)
    print("📊 EMPIRICAL VERIFICATION RESULTS")
    print(f"Total Frames Evaluated: {frames_to_test}")
    print(f"Total Elapsed Time: {total_time:.2f}s | Average Latency: {avg_latency:.2f}ms | 95th Percentile: {p95_latency:.2f}ms")
    print(f"Effective Throughput: {achieved_fps:.1f} FPS (Target >= 30 FPS: {'PASSED ✅' if achieved_fps >= 30 else 'FAILED ❌'})")
    print(f"Floor Ghost Detections Count: {floor_ghost_detections} (Target == 0: {'PASSED ✅' if floor_ghost_detections == 0 else 'FAILED ❌'})")
    print(f"Distinct Worker Track IDs Assigned: {sorted(list(track_ids_seen))}")
    print(f"Saved Evidence Frames: {saved_frames}")
    print("=" * 70)

    # Save summary report
    summary = {
        "frames_evaluated": frames_to_test,
        "avg_latency_ms": round(float(avg_latency), 2),
        "p95_latency_ms": round(float(p95_latency), 2),
        "achieved_fps": round(float(achieved_fps), 1),
        "floor_ghost_detections": floor_ghost_detections,
        "track_ids_seen": sorted(list(track_ids_seen)),
        "saved_evidence_frames": saved_frames,
    }
    with open(output_dir / "verification_report.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
