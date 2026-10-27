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
        """Flat horizontal surface (aspect ratio hw/hh > 2.20) is rejected as non-helmet."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 300.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)
        # Flat tabletop at head height: hw = 160, hh = 65 -> aspect ratio 2.46 > 2.20
        flat_table_det = [{"box": [120.0, 100.0, 280.0, 165.0], "cls": 5, "name": "white_helmet", "conf": 0.88}]
        reports, _ = engine.evaluate_frame([track], flat_table_det, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is False, "Flat table surface (aspect ratio > 2.20) must be rejected!"

    def test_horizontal_head_centering_shoulder_detection_rejected(self):
        """Object out on shoulder or elbow (|hcx - pcx| > 0.45 * pw) is rejected as head helmet."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        # Person: [100, 100, 300, 500] -> pcx=200, pw=200, allowed hcx in [110, 290] (0.45*pw)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 300.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)
        # Object far out on arm/elbow: x in [300, 340] -> hcx = 320 (|hcx - pcx| = 120 = 0.60 * pw > 0.45 * pw)
        shoulder_det = [{"box": [300.0, 110.0, 340.0, 150.0], "cls": 5, "name": "white_helmet", "conf": 0.85}]
        reports, _ = engine.evaluate_frame([track], shoulder_det, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is False, "Object out on shoulder must NOT match as head helmet!"

    def test_side_stride_and_wide_brim_helmet_accepted(self):
        """Worker walking in profile/stride with head offset 0.35*pw and wide-brim helmet (aspect 1.85) is accepted."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        # Person: [100, 100, 300, 500] -> pcx=200, pw=200, py1=100, ph=400
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 300.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)
        # Head shifted in stride: hcx = 200 + 0.35 * 200 = 270 (x in [233, 307], hw = 74, hh = 40 -> aspect 1.85)
        # hy1 = 100 + 0.15 * 400 = 160 (<= py1 + 0.22*ph = 188)
        stride_helmet = [{"box": [233.0, 120.0, 307.0, 160.0], "cls": 6, "name": "yellow_helmet", "conf": 0.89}]
        reports, _ = engine.evaluate_frame([track], stride_helmet, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["has_helmet"] is True, "Side-stride worker helmet must be accepted!"
        assert reports[0]["helmet_color"] == "yellow"
        assert reports[0]["matched_helmet_box"] == [233.0, 120.0, 307.0, 160.0]

    def test_matched_ppe_boxes_stored_in_report(self):
        """Worker report must store matched_helmet_box and matched_vest_box directly."""
        engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
        track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 300.0, 500.0])), score=0.90, cls_id=0)
        track.track_id = 1
        track.is_activated = True

        dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)
        h_box = [170.0, 100.0, 230.0, 150.0]
        v_box = [130.0, 180.0, 270.0, 380.0]
        dets = [
            {"box": h_box, "cls": 5, "name": "white_helmet", "conf": 0.92},
            {"box": v_box, "cls": 1, "name": "safety_vest", "conf": 0.88},
        ]
        reports, _ = engine.evaluate_frame([track], dets, dummy_frame)

        assert len(reports) == 1
        assert reports[0]["matched_helmet_box"] == h_box
        assert reports[0]["matched_vest_box"] == v_box
        assert reports[0]["status"] == "COMPLIANT"
