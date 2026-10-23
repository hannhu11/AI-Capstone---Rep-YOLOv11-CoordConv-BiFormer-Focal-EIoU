import sys
from pathlib import Path

# Extract rule engine from build_research_kernels or implement test
HELMET_CLASSES = {2: "blue", 3: "red", 4: "white", 5: "yellow"}
VEST_CLASS = 1
PERSON_CLASS = 0

def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-7
    return inter / union

def evaluate_worker_compliance(detections):
    persons = [d for d in detections if d["cls"] == PERSON_CLASS]
    vests = [d for d in detections if d["cls"] == VEST_CLASS]
    helmets = [d for d in detections if d["cls"] in HELMET_CLASSES]

    worker_reports = []

    for idx, p in enumerate(persons):
        px1, py1, px2, py2 = p["box"]
        pw = max(0.0, px2 - px1)
        ph = max(0.0, py2 - py1)
        
        if pw <= 0 or ph <= 0:
            continue

        head_box = [px1, py1, px2, py1 + 0.35 * ph]
        torso_box = [px1, py1 + 0.15 * ph, px2, py1 + 0.70 * ph]

        matched_helmet = None
        for h in helmets:
            hx1, hy1, hx2, hy2 = h["box"]
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            if (head_box[0] <= hcx <= head_box[2]) and (head_box[1] <= hcy <= head_box[3]):
                matched_helmet = h
                break
            elif bbox_iou(head_box, h["box"]) > 0.15:
                matched_helmet = h
                break

        has_helmet = (matched_helmet is not None)
        helmet_color = HELMET_CLASSES.get(matched_helmet["cls"]) if has_helmet else None

        matched_vest = None
        for v in vests:
            if bbox_iou(torso_box, v["box"]) > 0.20 or bbox_iou(p["box"], v["box"]) > 0.25:
                matched_vest = v
                break

        has_vest = (matched_vest is not None)

        violations = []
        if not has_helmet:
            violations.append("MISSING_SAFETY_HELMET")
        if not has_vest:
            violations.append("MISSING_SAFETY_VEST")

        status = "COMPLIANT" if len(violations) == 0 else "NON_COMPLIANT"

        report = {
            "person_id": idx + 1,
            "person_box": [round(float(c), 1) for c in p["box"]],
            "confidence": round(float(p.get("conf", 1.0)), 3),
            "has_helmet": has_helmet,
            "helmet_color": helmet_color,
            "has_safety_vest": has_vest,
            "status": status,
            "violations": violations
        }
        worker_reports.append(report)

    return worker_reports

print("Testing edge cases for Rule Engine...")
# Edge Case 1: Empty
assert evaluate_worker_compliance([]) == [], "Failed on empty detections"

# Edge Case 2: Degenerate box
r_deg = evaluate_worker_compliance([{"box": [10, 10, 10, 10], "cls": 0, "conf": 0.9}])
assert len(r_deg) == 0, "Failed on zero-area box"

# Edge Case 3: Compliant worker
detections_ok = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95}, # Person h=300
    {"box": [120, 100, 180, 160], "cls": 4, "conf": 0.92}, # White helmet
    {"box": [110, 160, 190, 300], "cls": 1, "conf": 0.88}, # Safety vest
]
r_ok = evaluate_worker_compliance(detections_ok)
assert len(r_ok) == 1
assert r_ok[0]["status"] == "COMPLIANT"
assert r_ok[0]["helmet_color"] == "white"
assert r_ok[0]["has_safety_vest"] is True

# Edge Case 4: Non-compliant worker (no helmet, no vest)
detections_viol = [
    {"box": [100, 100, 200, 400], "cls": 0, "conf": 0.95},
]
r_viol = evaluate_worker_compliance(detections_viol)
assert len(r_viol) == 1
assert r_viol[0]["status"] == "NON_COMPLIANT"
assert "MISSING_SAFETY_HELMET" in r_viol[0]["violations"]
assert "MISSING_SAFETY_VEST" in r_viol[0]["violations"]

print("All Rule Engine edge case tests PASSED successfully!")
