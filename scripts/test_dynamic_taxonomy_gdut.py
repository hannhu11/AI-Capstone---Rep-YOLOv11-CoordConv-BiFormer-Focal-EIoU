"""
scripts/test_dynamic_taxonomy_gdut.py
=====================================
Unit test verifying:
1. Pure isolated loading of GDUT A6 deploy checkpoint.
2. Dynamic Taxonomy Adapter correctly detects GDUT 5-classes:
   - has_person_class == False
   - is_helmet_only == True
3. Predict does NOT mistake cls == 0 ('blue') for a person.
4. SpatialSafetyEngine does NOT mistake cls == 1 ('white') for a vest.
5. In Helmet Violation mode, cls == 4 ('none') triggers direct head-level violation alert.
6. ByteTrack handles empty person_arr without crashing.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.detector import SafetyObjectDetector
from src.engine.spatial_safety import SpatialSafetyEngine
from src.engine.bytetrack import BYTETracker


def main():
    print("==================================================")
    print("🧪 Step 1: Loading GDUT A6 YOLO11s deploy checkpoint")
    deploy_11s = PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "deploy_weights" / "rep_yolo11s_gdut_a6_deploy.pt"
    assert deploy_11s.exists(), f"File not found: {deploy_11s}"

    detector = SafetyObjectDetector(model_path=str(deploy_11s))
    print(f"   [+] Loaded names: {detector.class_names}")
    print(f"   [+] has_person_class: {detector.has_person_class}")
    print(f"   [+] is_helmet_only: {detector.is_helmet_only}")
    assert not detector.has_person_class, "GDUT must have has_person_class == False"
    assert detector.is_helmet_only, "GDUT must have is_helmet_only == True"

    print("\n==================================================")
    print("🧪 Step 2: Testing Predict & Person Array Filtering")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    all_dets, person_arr, latency = detector.predict(dummy)
    print(f"   [+] person_arr shape: {person_arr.shape}")
    assert person_arr.shape[0] == 0, f"Expected 0 person detections, got {person_arr.shape[0]}"

    print("\n==================================================")
    print("🧪 Step 3: Testing ByteTrack with 0 Person Detections")
    tracker = BYTETracker()
    tracks = tracker.update(person_arr, dummy)
    print(f"   [+] Active tracks: {len(tracks)}")
    assert len(tracks) == 0, "Tracks must be 0"

    print("\n==================================================")
    print("🧪 Step 4: Testing SpatialSafetyEngine Helmet Violation Mode")
    safety = SpatialSafetyEngine()
    mock_detections = [
        {"box": [100.0, 100.0, 180.0, 180.0], "cls": 0, "name": "blue", "conf": 0.92},
        {"box": [250.0, 100.0, 330.0, 180.0], "cls": 1, "name": "white", "conf": 0.89},
        {"box": [400.0, 100.0, 480.0, 180.0], "cls": 4, "name": "none", "conf": 0.85},
    ]

    reports, alerts = safety.evaluate_frame(tracks, mock_detections, dummy, camera_id="TEST_CAM")
    print(f"   [+] Total Worker Reports: {len(reports)}")
    for r in reports:
        print(f"       - {r['track_id']}: status={r['status']}, helmet={r['helmet_color']}, violations={r['violations']}")

    print(f"   [+] Alert Events Triggered: {len(alerts)}")
    for a in alerts:
        print(f"       - {a['alert_id']}: worker={a['worker_id']}, violations={a['violations']}, snapshot={a['snapshot_filename']}")

    assert len(reports) == 3, f"Expected 3 reports, got {len(reports)}"
    assert sum(1 for r in reports if r["status"] == "COMPLIANT") == 2, "Expected 2 compliant helmets"
    assert sum(1 for r in reports if r["status"] == "VIOLATION") == 1, "Expected 1 violation"
    assert len(alerts) == 1, f"Expected 1 alert event, got {len(alerts)}"
    assert alerts[0]["violations"] == ["MISSING_SAFETY_HELMET"], f"Unexpected violations: {alerts[0]['violations']}"

    print("\n==================================================")
    print("🧪 Step 5: Loading GDUT A6 YOLOv8s deploy checkpoint")
    deploy_8s = PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "deploy_weights" / "rep_yolov8s_gdut_a6_deploy.pt"
    assert deploy_8s.exists(), f"File not found: {deploy_8s}"

    detector_8s = SafetyObjectDetector(model_path=str(deploy_8s))
    print(f"   [+] Loaded names: {detector_8s.class_names}")
    assert not detector_8s.has_person_class
    assert detector_8s.is_helmet_only

    print("\n🎉 ALL DYNAMIC TAXONOMY & HELMET VIOLATION ENGINE TESTS PASSED!")


if __name__ == "__main__":
    main()
