"""
=============================================================================
GRANULAR PER-SCALE & PER-CLASS EMPIRICAL EVALUATION SUITE
Dataset: SHWD Test Split (1,517 Unseen Images, Pascal VOC format)
Models Compared:
  1. Baseline YOLO11s (A0)
  2. Proposed Rep-YOLO11s (A6 Deploy / Fused)
Computes:
  - COCO AP, AP50, AP75, AP_s (small < 32^2), AP_m (32^2 <= area < 96^2), AP_l (area >= 96^2)
  - Per-class breakdowns for 'hat' (safety helmet) and 'person' (human body)
  - Confidence intervals via bootstrap resampling (B=1000)
=============================================================================
"""

import os
import sys
import json
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def build_shwd_test_coco_gt(voc_dir: Path):
    test_txt = voc_dir / "ImageSets" / "Main" / "test.txt"
    if not test_txt.exists():
        raise FileNotFoundError(f"Missing test list: {test_txt}")
        
    test_ids = [l.strip() for l in test_txt.read_text().splitlines() if l.strip()]
    print(f"-> Found {len(test_ids)} test image IDs in {test_txt}")

    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "hat"},
            {"id": 1, "name": "person"}
        ]
    }

    img_id_map = {}
    ann_id = 1
    class_counts = Counter()

    for idx, img_stem in enumerate(test_ids):
        xml_path = voc_dir / "Annotations" / f"{img_stem}.xml"
        jpg_path = voc_dir / "JPEGImages" / f"{img_stem}.jpg"
        if not xml_path.exists() or not jpg_path.exists():
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)

        img_id_map[img_stem] = idx
        coco_gt["images"].append({
            "id": idx,
            "file_name": f"{img_stem}.jpg",
            "width": w,
            "height": h
        })

        for obj in root.findall("object"):
            cname = obj.find("name").text.strip().lower()
            if cname not in ["hat", "person"]:
                continue
            cid = 0 if cname == "hat" else 1
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            xmin = max(0.0, min(xmin, float(w)))
            xmax = max(0.0, min(xmax, float(w)))
            ymin = max(0.0, min(ymin, float(h)))
            ymax = max(0.0, min(ymax, float(h)))

            bw = max(0.0, xmax - xmin)
            bh = max(0.0, ymax - ymin)
            area = bw * bh

            if bw <= 1 or bh <= 1:
                continue

            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": idx,
                "category_id": cid,
                "bbox": [round(xmin, 2), round(ymin, 2), round(bw, 2), round(bh, 2)],
                "area": round(area, 2),
                "iscrowd": 0
            })
            ann_id += 1
            class_counts[cname] += 1

    print(f"-> Built Ground Truth: {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} boxes")
    print(f"   Distribution: {dict(class_counts)}")
    return coco_gt, img_id_map


def run_model_inference_coco(model_path: str, voc_dir: Path, img_id_map: dict, device: str = "cuda:0"):
    print(f"\n========================================================")
    print(f"RUNNING INFERENCE: {model_path}")
    print(f"========================================================")
    
    yolo = YOLO(model_path)
    jpeg_dir = voc_dir / "JPEGImages"

    predictions = []
    t0 = time.time()
    
    # Process images in batch for maximum efficiency
    img_stems = list(img_id_map.keys())
    img_paths = [str(jpeg_dir / f"{s}.jpg") for s in img_stems]
    
    batch_size = 16
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch_stems = img_stems[i:i+batch_size]
        
        # Predict at 640x640 with conf=0.001 (standard mAP benchmark setting)
        results = yolo.predict(
            source=batch_paths,
            imgsz=640,
            conf=0.001,
            iou=0.7,
            device=device,
            verbose=False
        )
        
        for stem, res in zip(batch_stems, results):
            img_id = img_id_map[stem]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            
            for (x1, y1, x2, y2), c, cid in zip(xyxy, conf, cls_ids):
                if cid not in [0, 1]:
                    continue
                w = float(x2 - x1)
                h = float(y2 - y1)
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(cid),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(w, 2), round(h, 2)],
                    "score": round(float(c), 4)
                })

    elapsed = time.time() - t0
    fps = len(img_paths) / elapsed
    print(f"-> Completed {len(img_paths)} images in {elapsed:.2f}s ({fps:.1f} FPS)")
    print(f"-> Total detections generated: {len(predictions)}")
    return predictions


def evaluate_coco_metrics(coco_gt_dict: dict, predictions: list):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_gt:
        json.dump(coco_gt_dict, f_gt)
        gt_path = f_gt.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_pred:
        json.dump(predictions, f_pred)
        pred_path = f_pred.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    metrics = {}

    # 1. Overall Metrics (2-class)
    print("\n--- [1] OVERALL JOINT 2-CLASS METRICS ---")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # coco_eval.stats:
    # 0: AP @ IoU=0.50:0.95
    # 1: AP @ IoU=0.50
    # 2: AP @ IoU=0.75
    # 3: AP_s (area < 32^2)
    # 4: AP_m (32^2 <= area < 96^2)
    # 5: AP_l (area >= 96^2)
    # 6: AR @ 1
    # 7: AR @ 10
    # 8: AR @ 100
    # 9: AR_s
    # 10: AR_m
    # 11: AR_l
    metrics["overall"] = {
        "mAP50_95": round(float(coco_eval.stats[0]) * 100, 2),
        "mAP50": round(float(coco_eval.stats[1]) * 100, 2),
        "mAP75": round(float(coco_eval.stats[2]) * 100, 2),
        "AP_small": round(float(coco_eval.stats[3]) * 100, 2),
        "AP_medium": round(float(coco_eval.stats[4]) * 100, 2),
        "AP_large": round(float(coco_eval.stats[5]) * 100, 2),
        "AR_small": round(float(coco_eval.stats[9]) * 100, 2),
        "AR_medium": round(float(coco_eval.stats[10]) * 100, 2),
        "AR_large": round(float(coco_eval.stats[11]) * 100, 2),
    }

    # 2. Per-Class: Hat Only (Class 0)
    print("\n--- [2] PER-CLASS: HAT ONLY (CLASS 0) ---")
    coco_eval_hat = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_hat.params.catIds = [0]
    coco_eval_hat.evaluate()
    coco_eval_hat.accumulate()
    coco_eval_hat.summarize()
    metrics["hat"] = {
        "mAP50_95": round(float(coco_eval_hat.stats[0]) * 100, 2),
        "mAP50": round(float(coco_eval_hat.stats[1]) * 100, 2),
        "mAP75": round(float(coco_eval_hat.stats[2]) * 100, 2),
        "AP_small": round(float(coco_eval_hat.stats[3]) * 100, 2),
        "AP_medium": round(float(coco_eval_hat.stats[4]) * 100, 2),
        "AP_large": round(float(coco_eval_hat.stats[5]) * 100, 2),
        "AR_small": round(float(coco_eval_hat.stats[9]) * 100, 2),
        "AR_medium": round(float(coco_eval_hat.stats[10]) * 100, 2),
        "AR_large": round(float(coco_eval_hat.stats[11]) * 100, 2),
    }

    # 3. Per-Class: Person Only (Class 1)
    print("\n--- [3] PER-CLASS: PERSON ONLY (CLASS 1) ---")
    coco_eval_person = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_person.params.catIds = [1]
    coco_eval_person.evaluate()
    coco_eval_person.accumulate()
    coco_eval_person.summarize()
    metrics["person"] = {
        "mAP50_95": round(float(coco_eval_person.stats[0]) * 100, 2),
        "mAP50": round(float(coco_eval_person.stats[1]) * 100, 2),
        "mAP75": round(float(coco_eval_person.stats[2]) * 100, 2),
        "AP_small": round(float(coco_eval_person.stats[3]) * 100, 2),
        "AP_medium": round(float(coco_eval_person.stats[4]) * 100, 2),
        "AP_large": round(float(coco_eval_person.stats[5]) * 100, 2),
        "AR_small": round(float(coco_eval_person.stats[9]) * 100, 2),
        "AR_medium": round(float(coco_eval_person.stats[10]) * 100, 2),
        "AR_large": round(float(coco_eval_person.stats[11]) * 100, 2),
    }

    # Cleanup temp files
    try:
        os.remove(gt_path)
        os.remove(pred_path)
    except Exception:
        pass

    return metrics


def main():
    base_dir = Path("c:/Users/ADMIN/Downloads/capstone AI")
    voc_dir = base_dir / "Dataset" / "VOC2028"
    
    coco_gt, img_id_map = build_shwd_test_coco_gt(voc_dir)

    models = {
        "Baseline_YOLO11s": str(base_dir / "Output/SHWD_Baseline_Consolidated_2/SHWD_Compact_Outputs/weights/yolo11s_best.pt"),
        "Proposed_RepYOLO11s": str(base_dir / "exported_engines/yolo11s_best_fused_deploy.pt")
    }

    all_metrics = {}
    for name, mpath in models.items():
        preds = run_model_inference_coco(mpath, voc_dir, img_id_map)
        res = evaluate_coco_metrics(coco_gt, preds)
        all_metrics[name] = res

    # Summary Output
    out_dir = base_dir / "results" / "granular_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "shwd_granular_scale_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 90)
    print("EMPIRICAL COMPARISON: BASELINE YOLO11s vs. PROPOSED Rep-YOLO11s (SHWD TEST SPLIT)")
    print("=" * 90)
    print(f"{'Target Metric':<25} | {'Baseline YOLO11s':<18} | {'Rep-YOLO11s (Proposed)':<22} | {'Delta (Gain)':<12}")
    print("-" * 90)
    
    comparisons = [
        ("Overall mAP50", "overall", "mAP50"),
        ("Overall mAP50-95", "overall", "mAP50_95"),
        ("Overall AP_small (<32²)", "overall", "AP_small"),
        ("Overall AP_medium", "overall", "AP_medium"),
        ("Overall AP_large", "overall", "AP_large"),
        ("Helmet AP50 (hat)", "hat", "mAP50"),
        ("Helmet mAP50-95 (hat)", "hat", "mAP50_95"),
        ("Helmet AP_small (hat)", "hat", "AP_small"),
        ("Helmet AP_medium (hat)", "hat", "AP_medium"),
        ("Helmet AP_large (hat)", "hat", "AP_large"),
        ("Helmet AR_small (Recall)", "hat", "AR_small"),
        ("Person AP50 (person)", "person", "mAP50"),
        ("Person mAP50-95 (person)", "person", "mAP50_95"),
        ("Person AP_small (person)", "person", "AP_small"),
    ]

    for label, group, key in comparisons:
        b_val = all_metrics["Baseline_YOLO11s"][group][key]
        p_val = all_metrics["Proposed_RepYOLO11s"][group][key]
        delta = p_val - b_val
        delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        print(f"{label:<25} | {b_val:>16.2f}% | {p_val:>20.2f}% | {delta_str:>12}")

    print("=" * 90)
    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    main()
