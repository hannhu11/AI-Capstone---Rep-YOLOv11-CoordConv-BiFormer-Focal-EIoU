"""
tests/test_tracking_and_alert_engine.py
=======================================
Comprehensive unit and integration test suite for:
1. Kalman Filter & ByteTrack MOT Engine (Low-confidence association, ID persistence across occlusion).
2. Spatial Relational Safety Engine (Helmet color extraction, vest association, debouncing).
3. Telegram & Zalo Alert Dispatchers.
4. FastAPI CCTV Surveillance Endpoints.
"""

from __future__ import annotations
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from src.engine.kalman_tracker import KalmanFilter
from src.engine.bytetrack import BYTETracker, STrack
from src.engine.spatial_safety import SpatialSafetyEngine, bbox_iou
from src.alerting.telegram_bot import TelegramAlertBot
from src.alerting.zalo_bot import ZaloAlertBot
from src.web.app import app


# ==============================================================================
# 1. Kalman Filter Tests
# ==============================================================================
def test_kalman_filter_init_predict_update():
    kf = KalmanFilter()
    measurement = np.array([100.0, 200.0, 0.5, 80.0], dtype=np.float32)
    mean, cov = kf.initiate(measurement)

    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    assert np.allclose(mean[:4], measurement)

    # Predict
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)
    assert np.all(np.diag(pred_cov) > 0)

    # Update with slightly shifted measurement (worker moved 2 pixels)
    meas2 = np.array([102.0, 201.0, 0.5, 80.0], dtype=np.float32)
    upd_mean, upd_cov = kf.update(pred_mean, pred_cov, meas2)
    assert upd_mean.shape == (8,)
    # Center should be shifted towards new measurement
    assert upd_mean[0] > 100.0


# ==============================================================================
# 2. ByteTrack MOT Tests (Persistent ID & Occlusion Handling)
# ==============================================================================
def test_bytetrack_id_persistence_and_occlusion():
    tracker = BYTETracker(track_thresh=0.45, track_low_thresh=0.15, track_buffer=30)
    tracker.reset()

    # Frame 1: Worker appears at [100, 100, 150, 250] with high score 0.90
    det_f1 = np.array([[100.0, 100.0, 150.0, 250.0, 0.90, 0]], dtype=np.float32)
    tracks_f1 = tracker.update(det_f1)
    assert len(tracks_f1) == 1
    worker_id = tracks_f1[0].track_id
    assert worker_id > 0

    # Frame 2: Worker moves slightly to [104, 102, 154, 252] with score 0.88
    det_f2 = np.array([[104.0, 102.0, 154.0, 252.0, 0.88, 0]], dtype=np.float32)
    tracks_f2 = tracker.update(det_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == worker_id  # SAME ID!

    # Frames 3..5: Simulated Occlusion (worker walks behind beam -> 0 detections)
    for _ in range(3):
        empty_tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert len(empty_tracks) == 0

    # Frame 6: Worker re-emerges near predicted location [116, 108, 166, 258]
    # Even with lower confidence (0.40, below track_thresh 0.45), ByteTrack second-stage association recovers it!
    det_f6 = np.array([[116.0, 108.0, 166.0, 258.0, 0.40, 0]], dtype=np.float32)
    tracks_f6 = tracker.update(det_f6)
    assert len(tracks_f6) == 1
    assert tracks_f6[0].track_id == worker_id  # EXACT SAME ID PERSISTED!


# ==============================================================================
# 3. Spatial Relational Safety Engine Tests
# ==============================================================================
def test_spatial_safety_compliance_and_debouncing():
    engine = SpatialSafetyEngine(consecutive_frames_thresh=3, cooldown_seconds=5.0)

    # Mock Tracklet at [100, 100, 200, 300] (w=100, h=200)
    # Head box: [100, 100, 200, 170]
    # Torso box: [100, 130, 200, 250]
    track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 200.0, 300.0])), score=0.92, cls_id=0)
    track.track_id = 42
    track.is_activated = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Case A: Fully Compliant Worker (Has Blue Helmet & Vest)
    compliant_detections = [
        {"box": [120.0, 105.0, 180.0, 160.0], "cls": 2, "name": "blue_helmet", "conf": 0.89},  # Head box
        {"box": [110.0, 140.0, 190.0, 240.0], "cls": 1, "name": "safety_vest", "conf": 0.94},  # Torso box
    ]
    reports_a, alerts_a = engine.evaluate_frame([track], compliant_detections, dummy_frame)
    assert len(reports_a) == 1
    assert reports_a[0]["status"] == "COMPLIANT"
    assert reports_a[0]["has_helmet"] is True
    assert reports_a[0]["helmet_color"] == "blue"
    assert reports_a[0]["has_safety_vest"] is True
    assert len(alerts_a) == 0  # No alert

    # Case B: Violation - Missing Helmet & Vest (Frames 1..2 under debounce threshold)
    violation_detections: list = []  # No helmet, no vest
    reports_b1, alerts_b1 = engine.evaluate_frame([track], violation_detections, dummy_frame)
    assert reports_b1[0]["status"] == "VIOLATION"
    assert len(alerts_b1) == 0  # Streak is 1 < 3, NO alert yet (debounced!)

    reports_b2, alerts_b2 = engine.evaluate_frame([track], violation_detections, dummy_frame)
    assert len(alerts_b2) == 0  # Streak is 2 < 3, NO alert yet

    # Frame 3: Streak reaches 3 -> ALERT FIRES!
    reports_b3, alerts_b3 = engine.evaluate_frame([track], violation_detections, dummy_frame)
    assert len(alerts_b3) == 1
    alert = alerts_b3[0]
    assert alert["worker_id"] == 42
    assert "MISSING_SAFETY_HELMET" in alert["violations"]
    assert "MISSING_SAFETY_VEST" in alert["violations"]
    assert Path(alert["snapshot_path"]).exists()  # Snapshot was generated!
    assert Path(alert["crop_path"]).exists()      # Person crop was generated!

    # Frame 4: Next immediate frame within cooldown -> NO duplicate alert!
    reports_b4, alerts_b4 = engine.evaluate_frame([track], violation_detections, dummy_frame)
    assert len(alerts_b4) == 0  # Suppressed by cooldown!


def test_negative_labels_suppression_and_crossing_tracks():
    # 1. Test negative label suppression
    engine = SpatialSafetyEngine(consecutive_frames_thresh=1)
    track = STrack(STrack.tlbr_to_tlwh(np.array([100.0, 100.0, 200.0, 300.0])), score=0.90, cls_id=0)
    track.track_id = 77
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Detections include an explicit 'no_helmet' and 'no_vest'
    neg_detections = [
        {"box": [110.0, 105.0, 190.0, 160.0], "cls": 7, "name": "no_helmet", "conf": 0.88},
        {"box": [110.0, 140.0, 190.0, 240.0], "cls": 8, "name": "no_vest", "conf": 0.85},
    ]
    reports, alerts = engine.evaluate_frame([track], neg_detections, dummy_frame)
    assert len(reports) == 1
    assert reports[0]["status"] == "VIOLATION"
    assert reports[0]["has_helmet"] is False
    assert reports[0]["has_safety_vest"] is False
    assert "MISSING_SAFETY_HELMET" in reports[0]["violations"]
    assert "MISSING_SAFETY_VEST" in reports[0]["violations"]

    # 2. Test two workers crossing trajectories with realistic continuous movement
    tracker = BYTETracker(track_thresh=0.45, track_low_thresh=0.15, track_buffer=30)
    tracker.reset()

    # Track 1 moves left-to-right (100 -> 160). Track 2 moves right-to-left (160 -> 100)
    for step in range(11):
        x1 = 100.0 + step * 6.0
        x2 = 160.0 - step * 6.0
        frame_dets = np.array([
            [x1, 100.0, x1 + 50.0, 250.0, 0.90, 0],
            [x2, 100.0, x2 + 50.0, 250.0, 0.90, 0],
        ], dtype=np.float32)
        active_t = tracker.update(frame_dets)
        assert len(active_t) == 2
        ids = {tr.track_id for tr in active_t}
        assert ids == {1, 2}  # Both worker IDs persisted through the crossing!


# ==============================================================================
# 4. Telegram & Zalo Bot Dispatcher Tests
# ==============================================================================
def test_telegram_and_zalo_bot_connectivity():
    # Test Telegram getMe ping
    tg = TelegramAlertBot(token="8931795673:AAElznwj0coP6ef1NCtPNQj3veGaWRMvnNU")
    ok, res = tg.test_connection()
    assert ok is True
    assert res.get("username") == "safety_helmet_cctv_bot"

    # Test Zalo getMe ping
    zalo = ZaloAlertBot(token="4263098262366049360:ZeYiCYoZHuZVprjkhdzjapYQJssZWVdBpBtUTMuHhWGAPHamYqYdPoiYTmQlMlzo")
    z_ok, z_res = zalo.test_connection()
    assert z_ok is True
    assert z_res.get("display_name") == "Bot safety helmet CCTV"


# ==============================================================================
# 5. FastAPI Endpoints Integration Tests
# ==============================================================================
def test_fastapi_web_endpoints():
    client = TestClient(app)

    # 1. Dashboard Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "INDUSTRIAL CCTV SURVEILLANCE" in r_root.text

    # 2. System Status API
    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    status_data = r_status.json()
    assert status_data["status"] == "online"
    assert "device" in status_data

    # 3. Telemetry Metrics API
    r_metrics = client.get("/api/metrics")
    assert r_metrics.status_code == 200
    metrics = r_metrics.json()
    assert "fps" in metrics
    assert "compliance_rate" in metrics
    assert "workers_count" in metrics

    # 4. Alerts API
    r_alerts = client.get("/api/alerts")
    assert r_alerts.status_code == 200
    assert isinstance(r_alerts.json(), list)

    # 5. Settings Update API
    r_set = client.post("/api/settings", json={"conf_threshold": 0.30, "consecutive_violation_frames": 10})
    assert r_set.status_code == 200
    assert r_set.json()["config"]["conf_threshold"] == 0.30
