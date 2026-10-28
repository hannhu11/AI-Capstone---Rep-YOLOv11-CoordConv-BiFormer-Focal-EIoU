"""
tests/test_stationary_worker_ppe.py
===================================
Rigorous verification for:
1. GDUT Yellow Helmet Color Resolution (Fix 1).
2. Stationary Worker PPE Preservation over 100 consecutive motionless frames (Fix 2).
3. Switch Source & Model Clean Slate Reset for stationary zones (Fix 2b).
"""

from __future__ import annotations
import numpy as np
import pytest

from src.engine.bytetrack import STrack
from src.engine.spatial_safety import SpatialSafetyEngine


class TestStationaryWorkerPPE:

    def test_gdut_yellow_helmet_resolves_to_yellow_not_blue(self):
        """GDUT model cls=2 with name='yellow' must resolve to 'yellow', not 'blue'."""
        safety = SpatialSafetyEngine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # GDUT-style detections: cls=2, name='yellow'
        detections = [
            {"box": [100.0, 100.0, 180.0, 180.0], "cls": 2, "name": "yellow", "conf": 0.95},
            {"box": [200.0, 100.0, 280.0, 180.0], "cls": 0, "name": "blue", "conf": 0.92},
            {"box": [300.0, 100.0, 380.0, 180.0], "cls": 1, "name": "white", "conf": 0.91},
            {"box": [400.0, 100.0, 480.0, 180.0], "cls": 3, "name": "red", "conf": 0.93},
        ]

        reports, alerts = safety.evaluate_frame([], detections, dummy_frame, is_helmet_only=True)
        assert len(reports) == 4
        assert len(alerts) == 0

        # Assert correct color mappings
        color_by_x = {r["bbox"][0]: r["helmet_color"] for r in reports}
        assert color_by_x[100.0] == "yellow", f"Expected 'yellow' for cls=2 ('yellow'), got {color_by_x[100.0]}"
        assert color_by_x[200.0] == "blue"
        assert color_by_x[300.0] == "white"
        assert color_by_x[400.0] == "red"

    def test_stationary_worker_chv_100_frames_remains_compliant(self):
        """
        A worker standing completely motionless for 100 frames (3+ seconds at 30fps)
        must NEVER lose their helmet or vest, must NEVER be suppressed as stationary furniture,
        and must maintain 100% compliance with 0 false alerts.
        """
        safety = SpatialSafetyEngine(consecutive_frames_thresh=8)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create persistent motionless worker track
        person_box = np.array([200.0, 100.0, 400.0, 600.0], dtype=np.float32)
        track = STrack(STrack.tlbr_to_tlwh(person_box), score=0.92, cls_id=0)
        track.track_id = 7
        track.is_activated = True

        # Static PPE detections at exact same coordinates for 100 frames
        static_detections = [
            {"box": [260.0, 100.0, 340.0, 180.0], "cls": 6, "name": "yellow_helmet", "conf": 0.94},
            {"box": [230.0, 220.0, 370.0, 460.0], "cls": 1, "name": "safety_vest", "conf": 0.91},
        ]

        for frame_idx in range(1, 101):
            reports, alerts = safety.evaluate_frame([track], static_detections, dummy_frame, is_helmet_only=False)

            assert len(reports) == 1, f"Frame {frame_idx}: Expected 1 worker report"
            rep = reports[0]
            assert rep["status"] == "COMPLIANT", f"Frame {frame_idx}: Worker falsely flagged as {rep['status']}!"
            assert rep["has_helmet"] is True, f"Frame {frame_idx}: Helmet lost on motionless worker!"
            assert rep["helmet_color"] == "yellow"
            assert rep["has_safety_vest"] is True, f"Frame {frame_idx}: Vest lost on motionless worker!"
            assert len(rep["violations"]) == 0, f"Frame {frame_idx}: Violations falsely reported: {rep['violations']}"
            assert len(alerts) == 0, f"Frame {frame_idx}: False alert fired: {alerts}"

        assert track.compliance_streak >= 100, f"Compliance streak should be >= 100, got {track.compliance_streak}"
        assert track.violation_streak == 0, f"Violation streak must be 0, got {track.violation_streak}"

    def test_stationary_helmet_gdut_100_frames_remains_active_and_compliant(self):
        """
        In GDUT helmet-only mode, a motionless helmet for 100 frames must NEVER be suppressed
        or disappear from active worker reports.
        """
        safety = SpatialSafetyEngine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        static_helmet = [
            {"box": [250.0, 100.0, 330.0, 180.0], "cls": 2, "name": "yellow", "conf": 0.95}
        ]

        for frame_idx in range(1, 101):
            reports, alerts = safety.evaluate_frame([], static_helmet, dummy_frame, is_helmet_only=True)
            assert len(reports) == 1, f"Frame {frame_idx}: Helmet vanished (suppressed as furniture)!"
            assert reports[0]["status"] == "COMPLIANT"
            assert reports[0]["helmet_color"] == "yellow"
            assert len(alerts) == 0

    def test_pipeline_switch_clears_stationary_structures(self):
        """Switching source or model must cleanly purge all stationary trackers and zones."""
        from unittest.mock import MagicMock
        from src.engine.pipeline import CCTVSurveillancePipeline

        # Mock minimal pipeline instance
        pipeline = object.__new__(CCTVSurveillancePipeline)
        import threading
        pipeline.lock = threading.Lock()
        pipeline.source_path = "mock.mp4"
        pipeline.tracker = MagicMock()
        pipeline.safety_engine = MagicMock()
        pipeline.detector = MagicMock()
        pipeline._init_video_capture = MagicMock()
        pipeline.stationary_furniture_tracker = {1: {"cx": 100, "cy": 100, "count": 50}}
        pipeline.confirmed_furniture_zones = [[50.0, 50.0, 150.0, 150.0]]
        pipeline.stationary_next_id = 42

        # Test switch_source
        pipeline.switch_source("new_video.mp4")
        assert len(pipeline.stationary_furniture_tracker) == 0, "switch_source must clear stationary_furniture_tracker"
        assert len(pipeline.confirmed_furniture_zones) == 0, "switch_source must clear confirmed_furniture_zones"
        assert pipeline.stationary_next_id == 1

        # Populate again and test switch_model
        pipeline.stationary_furniture_tracker = {2: {"cx": 200, "cy": 200, "count": 60}}
        pipeline.confirmed_furniture_zones = [[150.0, 150.0, 250.0, 250.0]]
        pipeline.stationary_next_id = 99

        pipeline.switch_model("new_model.pt")
        assert len(pipeline.stationary_furniture_tracker) == 0, "switch_model must clear stationary_furniture_tracker"
        assert len(pipeline.confirmed_furniture_zones) == 0, "switch_model must clear confirmed_furniture_zones"
        assert pipeline.stationary_next_id == 1
