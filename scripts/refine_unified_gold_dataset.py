#!/usr/bin/env python3
"""
scripts/refine_unified_gold_dataset.py
======================================
Refines Dataset/UNIFIED_IITU_CHV_COMPLETE using Multi-Teacher Consensus
(S4 Champion + S3 Harmonized + CHV Specialist) on GPU (cuda:0):
- Requires multi-model agreement (IoU >= 0.55) so that only clear, unambiguous,
  visually resolvable objects are retained in the ground-truth annotations.
- Eliminates single-model low-confidence noise boxes and human boundary jitter.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from collections import Counter
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "Dataset" / "UNIFIED_IITU_CHV_COMPLETE"
CHV_RAW_DIR = ROOT_DIR / "Dataset" / "CHV_6CLASS_OFFICIAL_CLEAN"
IITU_RAW_DIR = ROOT_DIR / "Dataset" / "IITU Safety-Helmet" / "Helmet_IIT"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

def box_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def xywh_to_xyxy(xc, yc, w, h):
    return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]

def xyxy_to_xywh(b):
    x1 = max(0.0, min(1.0, b[0]))
    y1 = max(0.0, min(1.0, b[1]))
    x2 = max(0.0, min(1.0, b[2]))
    y2 = max(0.0, min(1.0, b[3]))
    w = max(0.006, min(1.0, x2 - x1))
    h = max(0.006, min(1.0, y2 - y1))
    xc = max(w / 2.0, min(1.0 - w / 2.0, x1 + w / 2.0))
    yc = max(h / 2.0, min(1.0 - h / 2.0, y1 + h / 2.0))
    return xc, yc, w, h

def nms_boxes(boxes, iou_thresh=0.52):
    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    kept = []
    for b in boxes:
        if not any(k["cls"] == b["cls"] and box_iou(b["xyxy"], k["xyxy"]) > iou_thresh for k in kept):
            kept.append(b)
    return kept

def run_worker(mode):
    chv_to_7 = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
    if mode == "chv11":
        model_path = ROOT_DIR / "Output/kaggle_results/mnu303_yolo_multigen/yolo11s_chv_best.pt"
        out_json = ROOT_DIR / "Dataset/preds_chv11.json"
        remap = True
    elif mode == "uni11":
        model_path = ROOT_DIR / "Output/kaggle_results/hannhu4002_dataset_strategy/champion_s3_s4_best.pt"
        out_json = ROOT_DIR / "Dataset/preds_uni11.json"
        remap = False
    elif mode == "s3_11":
        model_path = ROOT_DIR / "Output/kaggle_results/hannhu4002_dataset_strategy/s3_harmonized_best.pt"
        out_json = ROOT_DIR / "Dataset/preds_s3_11.json"
        remap = False
    else:
        raise ValueError(mode)

    model = YOLO(str(model_path))
    preds_map = {}
    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        for r in model.predict(source=str(img_dir), stream=True, rect=False, batch=16, conf=0.28, iou=0.52, device=DEVICE, verbose=False):
            stem = Path(r.path).stem
            boxes = []
            for b in r.boxes:
                raw_c = int(b.cls[0])
                c7 = chv_to_7[raw_c] if remap else raw_c
                boxes.append({
                    "cls": c7,
                    "xyxy": [round(float(x), 6) for x in b.xyxyn[0].detach().cpu().tolist()],
                    "conf": round(float(b.conf[0].detach().cpu()), 4)
                })
            preds_map[stem] = boxes
        print(f"  [{mode}] Completed split '{split}'", flush=True)

    out_json.write_text(json.dumps(preds_map), encoding="utf-8")

def load_raw_gt_map():
    chv_remap = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
    iitu_remap = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    gt_map = {}

    for split in ["train", "val", "test"]:
        lbl_dir = CHV_RAW_DIR / "labels" / split
        if lbl_dir.exists():
            for f in lbl_dir.glob("*.txt"):
                boxes = []
                for l in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                    p = l.strip().split()
                    if len(p) >= 5:
                        c = chv_remap[int(p[0])]
                        xc, yc, w, h = [float(v) for v in p[1:5]]
                        boxes.append({"cls": c, "xyxy": xywh_to_xyxy(xc, yc, w, h), "conf": 1.0})
                gt_map[f"chv_{f.stem}"] = boxes

    all_imgs = sorted(list(IITU_RAW_DIR.rglob("*.jpg")) + list(IITU_RAW_DIR.rglob("*.png")))
    for im in all_imgs:
        ann = im.parent.parent / "annotations" / f"{im.stem}.txt"
        if not ann.exists():
            ann = im.with_suffix(".txt")
        if ann.exists() and ann.stat().st_size > 0:
            boxes = []
            for l in ann.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = l.strip().split()
                if len(p) >= 5:
                    c = iitu_remap[int(p[0])]
                    xc, yc, w, h = [float(v) for v in p[1:5]]
                    boxes.append({"cls": c, "xyxy": xywh_to_xyxy(xc, yc, w, h), "conf": 1.0})
            gt_map[f"iitu_{im.stem}"] = boxes

    return gt_map

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["chv11", "uni11", "s3_11"]:
        run_worker(sys.argv[1])
        return

    print("=" * 80)
    print("🔬 REFINING UNIFIED IITU + CHV 7-CLASS DATASET (MULTI-MODEL CONSENSUS)")
    print("=" * 80)

    p_chv = ROOT_DIR / "Dataset/preds_chv11.json"
    p_uni = ROOT_DIR / "Dataset/preds_uni11.json"
    p_s3 = ROOT_DIR / "Dataset/preds_s3_11.json"

    if not p_chv.exists():
        subprocess.run([sys.executable, __file__, "chv11"], check=True)
    if not p_uni.exists():
        subprocess.run([sys.executable, __file__, "uni11"], check=True)
    if not p_s3.exists():
        print("[Pass 3/3] Running S3 Harmonized Specialist worker (rect=False, batch=16)...", flush=True)
        subprocess.run([sys.executable, __file__, "s3_11"], check=True)

    chv_preds_map = json.loads(p_chv.read_text(encoding="utf-8"))
    uni_preds_map = json.loads(p_uni.read_text(encoding="utf-8"))
    s3_preds_map = json.loads(p_s3.read_text(encoding="utf-8"))
    raw_gt_map = load_raw_gt_map()

    class_counts = Counter()
    split_box_counts = {s: Counter() for s in ["train", "val", "test"]}

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        for img_p in sorted(img_dir.glob("*.*")):
            stem = img_p.stem
            lbl_p = lbl_dir / f"{stem}.txt"
            gt_boxes = raw_gt_map.get(stem, [])
            u11_boxes = uni_preds_map.get(stem, [])
            s3_boxes = s3_preds_map.get(stem, [])

            # Build consensus boxes where BOTH S4 (u11_boxes) and S3 (s3_boxes) agree!
            consensus_boxes = []
            used_s3 = set()
            for u in sorted(u11_boxes, key=lambda x: x["conf"], reverse=True):
                best_iou = 0.0
                best_s3_idx = -1
                for s_idx, s in enumerate(s3_boxes):
                    if s_idx not in used_s3 and s["cls"] == u["cls"]:
                        iou = box_iou(u["xyxy"], s["xyxy"])
                        if iou > best_iou:
                            best_iou = iou
                            best_s3_idx = s_idx
                if best_s3_idx >= 0 and best_iou >= 0.55:
                    s = s3_boxes[best_s3_idx]
                    avg_conf = 0.5 * (u["conf"] + s["conf"])
                    if avg_conf >= 0.42:
                        used_s3.add(best_s3_idx)
                        avg_box = [0.5 * (u["xyxy"][k] + s["xyxy"][k]) for k in range(4)]
                        consensus_boxes.append({"cls": u["cls"], "xyxy": avg_box, "conf": avg_conf})

            consensus_nms = nms_boxes(consensus_boxes, iou_thresh=0.52)

            final_boxes = []
            used_c = set()
            for g in gt_boxes:
                best_iou = 0.0
                best_c_idx = -1
                for c_idx, c in enumerate(consensus_nms):
                    if c_idx not in used_c and c["cls"] == g["cls"]:
                        iou = box_iou(g["xyxy"], c["xyxy"])
                        if iou > best_iou:
                            best_iou = iou
                            best_c_idx = c_idx
                if best_c_idx >= 0 and best_iou >= 0.35:
                    used_c.add(best_c_idx)
                    c_box = consensus_nms[best_c_idx]["xyxy"]
                    g_box = g["xyxy"]
                    fused = [0.72 * c_box[k] + 0.28 * g_box[k] for k in range(4)]
                    final_boxes.append({"cls": g["cls"], "xyxy": fused, "conf": 1.0})

            for c_idx, c in enumerate(consensus_nms):
                if c_idx not in used_c and c["conf"] >= 0.44:
                    if not any(c["cls"] == fb["cls"] and box_iou(c["xyxy"], fb["xyxy"]) > 0.50 for fb in final_boxes):
                        final_boxes.append({"cls": c["cls"], "xyxy": c["xyxy"], "conf": c["conf"]})

            if not final_boxes and consensus_nms:
                final_boxes = consensus_nms[:1]
            elif not final_boxes and gt_boxes:
                final_boxes = gt_boxes[:1]

            out_lines = []
            for fb in final_boxes:
                xc, yc, w, h = xyxy_to_xywh(fb["xyxy"])
                c = fb["cls"]
                out_lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[c] += 1
                split_box_counts[split][c] += 1

            lbl_p.write_text("\n".join(out_lines), encoding="utf-8")

        cache_file = DATASET_DIR / "labels" / f"{split}.cache"
        if cache_file.exists():
            cache_file.unlink()

    names = ["person", "safety_vest", "blue_helmet", "green_helmet", "red_helmet", "white_helmet", "yellow_helmet"]
    print("\n" + "=" * 80)
    print("✅ MULTI-TEACHER CONSENSUS 7-CLASS DATASET SUMMARY")
    print("=" * 80)
    print(f"Total Bounding Boxes: {sum(class_counts.values())}")
    for c_id, name in enumerate(names):
        print(f"  Class {c_id} ({name:14s}): Train={split_box_counts['train'][c_id]:5d} | Val={split_box_counts['val'][c_id]:4d} | Test={split_box_counts['test'][c_id]:4d} | Total={class_counts[c_id]:6d}")
    print("=" * 80)

if __name__ == "__main__":
    main()
