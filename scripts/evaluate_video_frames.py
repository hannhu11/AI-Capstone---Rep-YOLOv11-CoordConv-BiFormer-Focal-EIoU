import os
import glob
import time
import cv2
import torch
from ultralytics import YOLO

def main():
    model_path = "exported_engines/yolo11s_best_fused_deploy.pt"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found!")
        return

    print("=" * 80)
    print("[EVALUATION] KIỂM THỬ MÔ HÌNH REP-YOLO11s TRÊN KHUNG HÌNH VIDEO THỰC ĐỊA")
    print(f"Checkpoint: {model_path}")
    print("=" * 80)

    model = YOLO(model_path)
    
    frame_paths = sorted(glob.glob("video_test/sample_frames/*.jpg"))
    print(f"Tổng số khung hình kiểm thử: {len(frame_paths)}")

    out_dir = "video_test/detection_results"
    os.makedirs(out_dir, exist_ok=True)

    results_summary = []

    for f_path in frame_paths:
        base_name = os.path.basename(f_path)
        img = cv2.imread(f_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        t0 = time.time()
        results = model.predict(source=img, conf=0.25, iou=0.45, verbose=False)
        t_infer = (time.time() - t0) * 1000 # ms

        res = results[0]
        boxes = res.boxes
        num_dets = len(boxes)
        
        hat_count = 0
        person_count = 0
        conf_sum = 0.0

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            conf_sum += conf
            if cls_id == 0:
                hat_count += 1
            elif cls_id == 1:
                person_count += 1

        avg_conf = (conf_sum / num_dets * 100) if num_dets > 0 else 0.0
        
        # Save annotated image
        annotated_img = res.plot()
        cv2.imwrite(os.path.join(out_dir, f"detected_{base_name}"), annotated_img)

        results_summary.append({
            "frame": base_name,
            "resolution": f"{w}x{h}",
            "latency_ms": t_infer,
            "fps": 1000.0 / t_infer if t_infer > 0 else 0,
            "detections": num_dets,
            "hats": hat_count,
            "persons": person_count,
            "avg_conf": avg_conf
        })

        print(f"[{base_name:<42}] {w}x{h} | Detections: {num_dets} (Hat: {hat_count}, Person: {person_count}) | Conf: {avg_conf:.1f}% | Latency: {t_infer:.1f}ms ({1000.0/t_infer:.1f} FPS)")

    print("\n" + "=" * 80)
    print("[SUMMARY] BẢNG TỔNG HỢP KIỂM THỬ TRÊN KHUNG HÌNH VIDEO")
    print("=" * 80)
    print(f"{'Frame Name':<42} | {'Res':<10} | {'Dets':<5} | {'Hat':<4} | {'Person':<6} | {'AvgConf':<7} | {'FPS':<6}")
    print("-" * 88)
    for r in results_summary:
        print(f"{r['frame']:<42} | {r['resolution']:<10} | {r['detections']:<5} | {r['hats']:<4} | {r['persons']:<6} | {r['avg_conf']:.1f}%   | {r['fps']:.1f}")

if __name__ == "__main__":
    main()
