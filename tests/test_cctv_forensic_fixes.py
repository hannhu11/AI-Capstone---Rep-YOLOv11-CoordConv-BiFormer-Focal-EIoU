"""
tests/test_cctv_forensic_fixes.py
=================================
Unit tests for CCTV Forensic Root-Cause Fixes:
1. Cranium Vertex Gating (Root Cause 1b)
2. Stationary Furniture Prior (Root Cause 1c)
3. Luminance / Value Gating on dark hair vs blue helmet (Root Cause 2)
4. Border Truncation & Violation Debounce Recovery (Root Cause 3)
5. Bipartite Spatial Association Filter in Pipeline (Root Cause 1a)
"""

from __future__ import annotations
import numpy as np
import pytest
import cv2

from src.engine.bytetrack import STrack
from src.engine.spatial_safety import SpatialSafetyEngine
from src.engine.pipeline import compute_box_intersection_ratio


class TestCCTVForensicFixes:

    def test_cranium_vertex_gating_valid_helmet_accepted(self):
        """Genuine helmet worn on head satisfies hy1 <= py1 + 0.18 * ph."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        # Person: [100, 100, 200, 400] -> py1=100, ph=300, cranium threshold py1 + 0.18*ph = 154
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 200.0, 400.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)
        # Helmet on head: hy1 = 100 <= 154
        helmet_det = [{"box": [120.0, 100.0, 180.0, 150.0], "cls": 5, "name": "white_helmet", "conf": 0.85}]
        reports, _ = engine.evaluate_frame([track], helmet_det, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is True
        assert reports[0]["helmet_color"] == "white"

    def test_cranium_vertex_gating_table_at_waist_rejected(self):
        """White table surface sitting at waist level (hy1 > py1 + 0.18 * ph) is rejected."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        # Person: [100, 100, 200, 400] -> py1=100, ph=300, max allowed hy1 = 154
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 200.0, 400.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)
        # Table top sitting at y=200 (waist level)
        table_det = [{"box": [110.0, 200.0, 190.0, 270.0], "cls": 5, "name": "white_helmet", "conf": 0.88}]
        reports, _ = engine.evaluate_frame([track], table_det, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is False, "Table at waist level must NOT be matched as helmet!"
        assert reports[0]["helmet_color"] is None

    def test_blue_helmet_luminance_gating_dark_hair_rejected(self):
        """Bare dark hair (Mean HSV Value V < 95) is rejected as a blue_helmet."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 200.0, 400.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        # Create a frame with dark hair color in the candidate region (BGR: ~60, 50, 45 -> V ~ 60 < 95)
        frame = np.full((500, 500, 3), 55, dtype=np.uint8)
        frame[90:150, 110:190] = [65, 52, 48]  # BGR dark hair tone

        hair_det = [{"box": [110.0, 95.0, 190.0, 145.0], "cls": 2, "name": "blue_helmet", "conf": 0.82}]
        reports, _ = engine.evaluate_frame([track], hair_det, frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is False, "Bare dark hair must NOT be matched as blue helmet!"

    def test_blue_helmet_luminance_gating_glossy_hardhat_accepted(self):
        """Glossy blue hardhat (Mean HSV Value V >= 110 >= 95) is accepted."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 200.0, 400.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        # Create frame with bright blue hardhat (BGR: 220, 130, 20 -> V ~ 220 >= 95)
        frame = np.full((500, 500, 3), 100, dtype=np.uint8)
        frame[90:150, 110:190] = [220, 130, 20]

        hardhat_det = [{"box": [110.0, 95.0, 190.0, 145.0], "cls": 2, "name": "blue_helmet", "conf": 0.88}]
        reports, _ = engine.evaluate_frame([track], hardhat_det, frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is True
        assert reports[0]["helmet_color"] == "blue"

    def test_unauthorized_intruder_violations_not_suppressed_at_border(self):
        """An unauthorized intruder entering frame has compliance_streak=0; violations must NOT be suppressed!"""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=3)
        # Person entering near border: px1=15, py1=15 in 432p frame (border_margin=21)
        track = STrack(STrack.tlbr_to_tlwh(np.array([15.0, 15.0, 200.0, 400.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True
        track.compliance_streak = 0
        track.violation_streak = 0

        dummy_frame = np.zeros((432, 768, 3), dtype=np.uint8)

        # Frame 1: Violations must be recorded (not wiped by boundary bug!)
        reports1, alerts1 = engine.evaluate_frame([track], [], dummy_frame)
        assert reports1[0]["status"] == "VIOLATION"
        assert "MISSING_SAFETY_HELMET" in reports1[0]["violations"]
        assert "MISSING_SAFETY_VEST" in reports1[0]["violations"]
        assert track.violation_streak == 1, "Violation streak must increment even at border!"

        # Frame 2
        reports2, alerts2 = engine.evaluate_frame([track], [], dummy_frame)
        assert track.violation_streak == 2

        # Frame 3: reaches threshold 3 -> Alert must fire!
        reports3, alerts3 = engine.evaluate_frame([track], [], dummy_frame)
        assert track.violation_streak == 3
        assert len(alerts3) == 1, "Violation alert MUST fire for unauthorized person even at border!"

    def test_compliant_worker_torso_border_clipping_protected(self):
        """Previously compliant worker (compliance_streak >= 2) with torso cut off by border is protected."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=3)
        # Frame height = 400. Worker walking out of bottom of frame: py1=200, py2=500
        # torso_box bottom = 200 + 0.75*300 = 425 >= (400 - 10) = 390 (torso physically cut!)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 200.0, 200.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 2
        track.is_activated = True
        track.compliance_streak = 5  # Established compliant worker

        dummy_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Worker still has helmet at head cranium (hy1=200 <= 200 + 0.18*300 = 254)
        helmet_only = [{"box": [120.0, 200.0, 180.0, 250.0], "cls": 6, "name": "yellow_helmet", "conf": 0.90}]

        reports, alerts = engine.evaluate_frame([track], helmet_only, dummy_frame)
        assert len(reports) == 1
        assert reports[0]["status"] == "COMPLIANT", "Boundary clipping of vest must NOT flag compliant worker!"
        assert len(alerts) == 0

    def test_bipartite_spatial_overlap_calculation(self):
        """Intersection over helmet area calculation matches Eq. 24 specifications."""
        person_box = [100.0, 100.0, 300.0, 500.0]  # h=400, upper torso: 100..340
        helmet_on_head = [150.0, 100.0, 250.0, 200.0]  # Fully inside upper torso
        overlap = compute_box_intersection_ratio(helmet_on_head, person_box, use_upper_torso=True)
        assert overlap == pytest.approx(1.0, rel=1e-3)

        table_at_knees = [150.0, 380.0, 250.0, 460.0]  # Lower body
        lower_overlap = compute_box_intersection_ratio(table_at_knees, person_box, use_upper_torso=True)
        assert lower_overlap == 0.0, "Lower body detection must have 0.0 upper torso overlap!"

    def test_helmet_aspect_ratio_table_surface_rejected(self):
        """Flat horizontal surface (aspect ratio hw/hh >= 1.70) is rejected as non-helmet."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 300.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)
        # Flat tabletop at head height: hw = 140, hh = 70 -> aspect ratio 2.0 > 1.70
        flat_table_det = [{"box": [130.0, 100.0, 270.0, 170.0], "cls": 5, "name": "white_helmet", "conf": 0.88}]
        reports, _ = engine.evaluate_frame([track], flat_table_det, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is False, "Flat table surface (aspect ratio 2.0) must be rejected!"

    def test_horizontal_head_centering_shoulder_detection_rejected(self):
        """Object out on shoulder or elbow (|hcx - pcx| > 0.20 * pw) is rejected as head helmet."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        # Person: [100, 100, 300, 500] -> pcx=200, pw=200, allowed hcx in [160, 240]
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 300.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)
        # Object at far shoulder/arm: x in [260, 300] -> hcx = 280 (|hcx - pcx| = 80 = 0.40 * pw > 0.20 * pw)
        shoulder_det = [{"box": [260.0, 110.0, 300.0, 150.0], "cls": 5, "name": "white_helmet", "conf": 0.85}]
        reports, _ = engine.evaluate_frame([track], shoulder_det, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is False, "Object out on shoulder must NOT match as head helmet!"
