"""
scripts/test_pipeline_both_videos.py
====================================
End-to-end verification of CCTV surveillance pipeline on both benchmark videos:
1. construction_site_workers_1080p.mp4 (60 FPS, Stride 2)
2. 13921040-uhd_3840_2160_30fps.mp4 (30 FPS UHD, Stride 1)
"""

import sys
import time
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.engine.pipeline import CCTVSurveillancePipeline
from src.alerting.alert_manager import alert_manager

def main():
    print("=================================================================")
    print("TEST 1: construction_site_workers_1080p.mp4 (60 FPS Stream)")
    print("=================================================================")
    pipe1 = CCTVSurveillancePipeline("video_test/construction_site_workers_1080p.mp4")
    print(f"Detected Stream FPS: {pipe1.stream_fps:.2f} | Configured Frame Stride: {pipe1.frame_stride}")
    assert pipe1.frame_stride == 2, f"Expected stride 2 for 60 FPS stream, got {pipe1.frame_stride}"

    t_start = time.perf_counter()
    processed_count = 0
    t_start = time.perf_counter()
    last_idx = -1
    processed_count = 0
    while processed_count < 100:
        jpeg, cur_idx = pipe1.get_latest_frame(last_frame_idx=last_idx, timeout=1.0)
        if cur_idx != last_idx and jpeg is not None:
            last_idx = cur_idx
            processed_count += 1
            if processed_count % 25 == 0:
                tele = pipe1.get_telemetry()
                print(f"Streamed Frame {cur_idx:3d}: FPS={tele['fps']:4.1f} | Infer Latency={tele['inference_ms']:5.1f}ms | Workers={tele['workers_count']} | Violations={tele['violations_count']}")
        else:
            time.sleep(0.01)

    elapsed = time.perf_counter() - t_start
    pipe1.stop()
    print(f"Video 1 Complete: {processed_count} frames streamed in {elapsed:.2f}s ({processed_count / elapsed:.1f} FPS)")
    alerts1 = alert_manager.get_recent_alerts(limit=10)
    print(f"Alerts triggered during Video 1: {len(alerts1)}")

    print("\n=================================================================")
    print("TEST 2: 13921040-uhd_3840_2160_30fps.mp4 (30 FPS 4K Stream)")
    print("=================================================================")
    pipe2 = CCTVSurveillancePipeline("video_test/13921040-uhd_3840_2160_30fps.mp4")
    print(f"Detected Stream FPS: {pipe2.stream_fps:.2f} | Configured Frame Stride: {pipe2.frame_stride}")
    assert pipe2.frame_stride == 1, f"Expected stride 1 for 30 FPS stream, got {pipe2.frame_stride}"

    t_start = time.perf_counter()
    last_idx = -1
    processed_count = 0
    while processed_count < 60:
        jpeg, cur_idx = pipe2.get_latest_frame(last_frame_idx=last_idx, timeout=1.0)
        if cur_idx != last_idx and jpeg is not None:
            last_idx = cur_idx
            processed_count += 1
            if processed_count % 20 == 0:
                tele = pipe2.get_telemetry()
                print(f"Streamed Frame {cur_idx:3d}: FPS={tele['fps']:4.1f} | Infer Latency={tele['inference_ms']:5.1f}ms | Workers={tele['workers_count']} | Violations={tele['violations_count']}")
        else:
            time.sleep(0.01)

    elapsed = time.perf_counter() - t_start
    pipe2.stop()
    print(f"Video 2 Complete: {processed_count} frames streamed in {elapsed:.2f}s ({processed_count / elapsed:.1f} FPS)")

    elapsed = time.perf_counter() - t_start
    pipe2.stop()
    print(f"Video 2 Complete: {processed_count} frames processed in {elapsed:.2f}s ({processed_count / elapsed:.1f} FPS)")

    print("\n=================================================================")
    print("VERIFICATION RESULT: ALL VIDEO PIPELINE TESTS PASSED!")
    print("=================================================================")

if __name__ == "__main__":
    main()
