import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.zalo_alert_dispatcher import evaluate_worker_compliance, HELMET_CLASSES_6, HELMET_CLASSES_7

print("Testing edge cases for Zalo Compliance Rule Engine...")

# Edge Case 1: Empty detections
assert evaluate_worker_compliance([]) == [], "Failed on empty detections"

# Edge Case 2: Degenerate zero-area box
r_deg = evaluate_worker_compliance([{"box": [10, 10, 10, 10], "cls": 0, "conf": 0.9}])
assert len(r_deg) == 0, "Failed on zero-area box"

# Edge Case 3: Compliant worker in 6-class (white helmet = class 4)
detections_6class = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95}, # Person h=300
    {"box": [120, 100, 180, 160], "cls": 4, "conf": 0.92}, # White helmet in 6-class
    {"box": [110, 160, 190, 300], "cls": 1, "conf": 0.88}, # Safety vest
]
r_6 = evaluate_worker_compliance(detections_6class, nc=6)
assert len(r_6) == 1
assert r_6[0]["status"] == "COMPLIANT"
assert r_6[0]["helmet_color"] == "white"
assert r_6[0]["has_safety_vest"] is True

# Edge Case 4: Compliant worker in 7-class with YELLOW helmet (Class 6)
detections_7class_yellow = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95}, # Person
    {"box": [120, 100, 180, 160], "cls": 6, "conf": 0.93}, # Yellow helmet in 7-class
    {"box": [110, 160, 190, 300], "cls": 1, "conf": 0.91}, # Safety vest
]
r_7_yellow = evaluate_worker_compliance(detections_7class_yellow, nc=7)
assert len(r_7_yellow) == 1
assert r_7_yellow[0]["status"] == "COMPLIANT", f"Expected COMPLIANT for yellow helmet, got {r_7_yellow[0]['status']}"
assert r_7_yellow[0]["helmet_color"] == "yellow"
assert r_7_yellow[0]["has_safety_vest"] is True

# Edge Case 5: Compliant worker in 7-class with GREEN helmet (Class 3)
detections_7class_green = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95}, # Person
    {"box": [120, 100, 180, 160], "cls": 3, "conf": 0.93}, # Green helmet in 7-class
    {"box": [110, 160, 190, 300], "cls": 1, "conf": 0.91}, # Safety vest
]
r_7_green = evaluate_worker_compliance(detections_7class_green, nc=7)
assert len(r_7_green) == 1
assert r_7_green[0]["status"] == "COMPLIANT"
assert r_7_green[0]["helmet_color"] == "green"
assert r_7_green[0]["has_safety_vest"] is True

# Edge Case 6: Non-compliant worker (no helmet, no vest)
detections_viol = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95},
]
r_viol = evaluate_worker_compliance(detections_viol, nc=7)
assert len(r_viol) == 1
assert r_viol[0]["status"] == "NON_COMPLIANT"
assert "MISSING_SAFETY_HELMET" in r_viol[0]["violations"]
assert "MISSING_SAFETY_VEST" in r_viol[0]["violations"]

# Edge Case 7: Helmet only, missing safety vest
detections_no_vest = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95},
    {"box": [120, 100, 180, 160], "cls": 2, "conf": 0.91}, # Blue helmet
]
r_no_vest = evaluate_worker_compliance(detections_no_vest, nc=7)
assert len(r_no_vest) == 1
assert r_no_vest[0]["status"] == "NON_COMPLIANT"
assert r_no_vest[0]["has_helmet"] is True
assert r_no_vest[0]["helmet_color"] == "blue"
assert "MISSING_SAFETY_VEST" in r_no_vest[0]["violations"]
assert "MISSING_SAFETY_HELMET" not in r_no_vest[0]["violations"]

print("All 7 Zalo Compliance Rule Engine edge case tests PASSED successfully!")
