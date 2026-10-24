"""
zalo_alert_dispatcher.py
=========================
Production-grade Spatial Relational PPE Compliance Engine & Zalo Alert Dispatcher.
1. Evaluates spatial relationships between Persons, Helmets (with color), and Safety Vests.
2. Identifies non-compliant workers (missing helmet, missing safety vest).
3. Formats structured Zalo notification messages for construction site safety officers.
4. Supports dry-run simulation and live Webhook HTTP POST dispatch.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Class index maps:
# Legacy 6-Class: 0: person, 1: safety_vest, 2: blue_helmet, 3: red_helmet, 4: white_helmet, 5: yellow_helmet
HELMET_CLASSES_6 = {2: "blue", 3: "red", 4: "white", 5: "yellow"}

# Unified 7-Class (Champion): 0: person, 1: safety_vest, 2: blue_helmet, 3: green_helmet, 4: red_helmet, 5: white_helmet, 6: yellow_helmet
HELMET_CLASSES_7 = {2: "blue", 3: "green", 4: "red", 5: "white", 6: "yellow"}

HELMET_CLASSES = HELMET_CLASSES_7
VEST_CLASS = 1
PERSON_CLASS = 0

def bbox_iou(b1: List[float], b2: List[float]) -> float:
    """Computes Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-7
    return inter / union

def evaluate_worker_compliance(detections: List[Dict[str, Any]], nc: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Evaluates spatial relational compliance for each detected worker.
    detections: list of dicts with 'box': [x1, y1, x2, y2], 'cls': int, 'conf': float
    nc: optional number of dataset classes (6 or 7). If None, auto-detected from detections.
    Returns a list of worker compliance reports.
    """
    if nc == 6:
        helmet_map = HELMET_CLASSES_6
    elif nc == 7:
        helmet_map = HELMET_CLASSES_7
    else:
        # Auto-detect: if any class is 6, must be 7-class; default to 7-class
        max_cls = max([d.get("cls", 0) for d in detections], default=0)
        helmet_map = HELMET_CLASSES_6 if max_cls == 5 else HELMET_CLASSES_7

    persons = [d for d in detections if d.get("cls") == PERSON_CLASS]
    vests = [d for d in detections if d.get("cls") == VEST_CLASS]
    helmets = [d for d in detections if d.get("cls") in helmet_map]

    worker_reports = []

    for idx, p in enumerate(persons):
        box = p.get("box", [0, 0, 0, 0])
        px1, py1, px2, py2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        pw = max(0.0, px2 - px1)
        ph = max(0.0, py2 - py1)

        # Skip degenerate boxes
        if pw <= 1.0 or ph <= 1.0:
            continue

        # Sub-regions:
        # Head sub-region: upper 35% of person bounding box
        head_box = [px1, py1, px2, py1 + 0.35 * ph]
        # Torso sub-region: middle 55% of person bounding box
        torso_box = [px1, py1 + 0.15 * ph, px2, py1 + 0.70 * ph]

        # 1. Helmet Match Check
        matched_helmet = None
        for h in helmets:
            hb = h.get("box", [0, 0, 0, 0])
            hx1, hy1, hx2, hy2 = float(hb[0]), float(hb[1]), float(hb[2]), float(hb[3])
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            # Direct containment of helmet center inside head box
            if (head_box[0] <= hcx <= head_box[2]) and (head_box[1] <= hcy <= head_box[3]):
                matched_helmet = h
                break
            elif bbox_iou(head_box, hb) > 0.15:
                matched_helmet = h
                break

        has_helmet = (matched_helmet is not None)
        helmet_color = helmet_map.get(matched_helmet["cls"]) if has_helmet else None

        # 2. Safety Vest Match Check
        matched_vest = None
        for v in vests:
            vb = v.get("box", [0, 0, 0, 0])
            if bbox_iou(torso_box, vb) > 0.20 or bbox_iou([px1, py1, px2, py2], vb) > 0.25:
                matched_vest = v
                break

        has_vest = (matched_vest is not None)

        # 3. Violations & Status
        violations = []
        if not has_helmet:
            violations.append("MISSING_SAFETY_HELMET")
        if not has_vest:
            violations.append("MISSING_SAFETY_VEST")

        status = "COMPLIANT" if len(violations) == 0 else "NON_COMPLIANT"

        report = {
            "person_id": idx + 1,
            "person_box": [round(px1, 1), round(py1, 1), round(px2, 1), round(py2, 1)],
            "confidence": round(float(p.get("conf", 1.0)), 3),
            "has_helmet": has_helmet,
            "helmet_color": helmet_color,
            "has_safety_vest": has_vest,
            "status": status,
            "violations": violations
        }
        worker_reports.append(report)

    return worker_reports

def format_zalo_alert(
    image_name: str,
    camera_id: str,
    worker_report: Dict[str, Any],
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """Formats a structured Zalo notification message."""
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    violations_str = ", ".join([v.replace("_", " ") for v in worker_report.get("violations", [])])
    
    helmet_desc = f"CO ({worker_report.get('helmet_color', 'khong ro')})" if worker_report.get("has_helmet") else "KHONG CO"
    vest_desc = "CO" if worker_report.get("has_safety_vest") else "KHONG CO"

    zalo_text = (
        f"[CANH BAO AN TOAN PPE - ZALO NOTIFICATION]\n"
        f"Camera: {camera_id} | Anh: {image_name}\n"
        f"Thoi gian: {ts}\n"
        f"Doi tuong: Cong nhan #{worker_report.get('person_id')} (Do tin cay: {worker_report.get('confidence', 0)*100:.1f}%)\n"
        f"Trang thai: VI PHAM AN TOAN LAO DONG!\n"
        f"Loi vi pham: {violations_str}\n"
        f"Kiem tra: Mu bao ho: {helmet_desc} | Ao phan quang: {vest_desc}\n"
        f"-> De nghi An toan vien cong truong lap tuc nhac nho va xu ly!"
    )

    alert_id = f"ZALO-ALERT-{Path(image_name).stem}-{worker_report.get('person_id')}"
    return {
        "alert_id": alert_id,
        "camera_id": camera_id,
        "timestamp": ts,
        "image_file": str(image_name),
        "worker_data": worker_report,
        "zalo_message": zalo_text
    }

def dispatch_zalo_alert(
    alert_payload: Dict[str, Any],
    webhook_url: Optional[str] = None,
    dry_run: bool = True
) -> bool:
    """Dispatches alert to Zalo webhook URL or simulates dispatch in dry-run mode."""
    if dry_run or not webhook_url:
        print(f"[ZALO DISPATCH (DRY-RUN)] Alert {alert_payload['alert_id']} queued successfully:")
        print(f"  Camera: {alert_payload['camera_id']} | Image: {alert_payload['image_file']}")
        print(f"  Worker #{alert_payload['worker_data']['person_id']} -> Violations: {alert_payload['worker_data']['violations']}")
        return True

    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(webhook_url, json=alert_payload, headers=headers, timeout=5.0)
        if resp.status_code in [200, 201, 204]:
            print(f"[ZALO DISPATCH (LIVE)] Alert {alert_payload['alert_id']} sent successfully (Status {resp.status_code}).")
            return True
        else:
            print(f"[ZALO DISPATCH (ERROR)] Failed with status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[ZALO DISPATCH (EXCEPTION)] Failed to send alert {alert_payload['alert_id']}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("ZALO ALERT DISPATCHER TEST HARNESS")
    print("=" * 80)

    # Sample detection list
    sample_dets = [
        {"box": [150.0, 50.0, 350.0, 500.0], "cls": 0, "conf": 0.88},  # Person
        {"box": [190.0, 48.0, 310.0, 160.0], "cls": 2, "conf": 0.91},   # Blue Helmet
        # Missing safety vest
    ]

    reports = evaluate_worker_compliance(sample_dets)
    for r in reports:
        print("Worker Report:", json.dumps(r, indent=2))
        if r["status"] == "NON_COMPLIANT":
            alert = format_zalo_alert("test_sample_01.jpg", "CAM_GATE_01", r)
            dispatch_zalo_alert(alert, dry_run=True)
            print("Formatted Zalo Message:\n" + alert["zalo_message"])
