import os
import sys
import time
import cv2
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import custom_ablation_modules as cam
import ultralytics.nn.modules as un_mod
import ultralytics.nn.tasks as un_tasks

# Register custom classes for torch unpickler compatibility
from scripts.ablation_factory import BiFormerSequentialWrapper
setattr(sys.modules['__main__'], 'BiFormerSequentialWrapper', BiFormerSequentialWrapper)
setattr(un_mod, 'BiFormerSequentialWrapper', BiFormerSequentialWrapper)
setattr(un_tasks, 'BiFormerSequentialWrapper', BiFormerSequentialWrapper)

for name in dir(cam):
    obj = getattr(cam, name)
    if isinstance(obj, type):
        setattr(sys.modules['__main__'], name, obj)
        setattr(un_mod, name, obj)
        setattr(un_tasks, name, obj)

from ultralytics import YOLO

def main():
    candidate_weights = [
        Path("Output/kaggle_results/hannhu40022_conflict_free_7class/rep_yolo11s_chv7_best_deploy.pt"),
        Path("Output/kaggle_results/hannhu4002_clean_chv6/rep_yolo11s_chv6_fused_deploy.pt"),
        Path("exported_engines/yolo11s_best_fused_deploy.pt"),
    ]
    
    model_path = None
    for p in candidate_weights:
        if p.exists():
            model_path = p
            break
            
    if model_path is None:
        model_path = candidate_weights[-1] # fallback

    video_path = "video_test/10810476-hd_1920_1080_30fps.mp4"
    output_path = "video_test/annotated_cctv_construction_demo.mp4"

    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    print("=" * 80)
    print("[VIDEO DEMO] MULTI-CLASS PPE CCTV AUDIT VIDEO ANNOTATION")
    print(f"Checkpoint: {model_path}")
    print(f"Input:      {video_path}")
    print(f"Output:     {output_path}")
    print("=" * 80)

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    max_frames = 150 # 5 seconds of 30 FPS footage for demo
    total_time = 0.0

    CLASS_COLORS = {
        'blue_helmet': (255, 128, 0),     # Azure blue
        'red_helmet': (0, 0, 255),        # Red
        'white_helmet': (220, 220, 220),  # White
        'yellow_helmet': (0, 230, 255),   # Yellow
        'no_helmet': (0, 0, 255),         # Warning Red
        'safety_vest': (0, 255, 128),     # Hi-vis green
        'person': (180, 180, 180),        # Gray
        'hat': (0, 230, 255),
        'vest': (0, 255, 128)
    }

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model.predict(source=frame, conf=0.25, iou=0.45, verbose=False)
        infer_time = time.time() - t0
        total_time += infer_time
        frame_fps = 1.0 / infer_time if infer_time > 0 else 0

        annotated = frame.copy()
        res = results[0]
        boxes = res.boxes

        violation_count = 0
        vest_count = 0
        helmet_count = 0

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id)).lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if "no_helmet" in cls_name or cls_name == "none":
                    violation_count += 1
                    color = (0, 0, 255)
                    label = f"VIOLATION: NO HELMET ({conf:.2f})"
                elif "vest" in cls_name:
                    vest_count += 1
                    color = (0, 255, 128)
                    label = f"SAFETY VEST ({conf:.2f})"
                elif "helmet" in cls_name or cls_name == "hat":
                    helmet_count += 1
                    color = CLASS_COLORS.get(cls_name, (0, 230, 255))
                    label = f"{cls_name.upper()} ({conf:.2f})"
                else:
                    color = (180, 180, 180)
                    label = f"{cls_name} ({conf:.2f})"

                # Draw bounding box
                thickness = 3 if "no_helmet" in cls_name else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Draw label tag
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + lw + 6, max(0, y1)), color, -1)
                text_color = (0, 0, 0) if color != (0, 0, 255) else (255, 255, 255)
                cv2.putText(annotated, label, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

        # Overlay Head-Up Display (HUD)
        hud_h = 75
        cv2.rectangle(annotated, (20, 20), (1200, 20 + hud_h), (25, 25, 25), -1)
        cv2.rectangle(annotated, (20, 20), (1200, 20 + hud_h), (60, 60, 60), 1)

        hud_title = f"Rep-YOLO11s Real-Time Edge Audit | Latency: {infer_time*1000:.1f}ms | FPS: {frame_fps:.1f}"
        cv2.putText(annotated, hud_title, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 128), 2, cv2.LINE_AA)

        hud_stats = f"Helmets: {helmet_count} | Vests: {vest_count} | Violations: {violation_count}"
        stat_color = (0, 200, 255) if violation_count == 0 else (0, 0, 255)
        cv2.putText(annotated, hud_stats, (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.70, stat_color, 2, cv2.LINE_AA)

        out.write(annotated)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames ({frame_count/max_frames*100:.0f}%) | Avg FPS: {frame_count/total_time:.1f}")

    cap.release()
    out.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n[STATUS: COMPLETED] Exported CCTV demo video: {output_path}")
    print(f"Frames: {frame_count} | Avg FPS: {avg_fps:.1f} | Size: {os.path.getsize(output_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
