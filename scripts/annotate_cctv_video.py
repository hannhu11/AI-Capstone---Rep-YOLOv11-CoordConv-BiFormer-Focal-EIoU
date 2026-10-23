import cv2
import time
import os
from ultralytics import YOLO

def main():
    model_path = "exported_engines/yolo11s_best_fused_deploy.pt"
    video_path = "video_test/10810476-hd_1920_1080_30fps.mp4"
    output_path = "video_test/annotated_cctv_construction_demo.mp4"

    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("Model or video path does not exist.")
        return

    print("=" * 80)
    print("[VIDEO DEMO] XUẤT VIDEO MINH HỌA GIÁM SÁT AN TOÀN LAO ĐỘNG THỜI GIAN THỰC")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    max_frames = 150 # Process 5 seconds of 30 FPS footage for demo
    total_time = 0.0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model.predict(source=frame, conf=0.3, iou=0.45, verbose=False)
        infer_time = time.time() - t0
        total_time += infer_time
        frame_fps = 1.0 / infer_time if infer_time > 0 else 0

        annotated = results[0].plot()

        # Add professional overlay HUD
        hud_text = f"Rep-YOLO11s Real-Time PPE Audit | FPS: {frame_fps:.1f} | Latency: {infer_time*1000:.1f}ms | Detections: {len(results[0].boxes)}"
        cv2.rectangle(annotated, (20, 20), (1050, 70), (20, 20, 20), -1)
        cv2.putText(annotated, hud_text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2, cv2.LINE_AA)

        out.write(annotated)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames ({frame_count/max_frames*100:.0f}%) | Avg FPS: {frame_count/total_time:.1f}")

    cap.release()
    out.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n[STATUS: COMPLETED] Đã tạo thành công video demo: {output_path}")
    print(f"Tổng khung hình: {frame_count} | Tốc độ xử lý trung bình: {avg_fps:.1f} FPS | Kích thước: {os.path.getsize(output_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
