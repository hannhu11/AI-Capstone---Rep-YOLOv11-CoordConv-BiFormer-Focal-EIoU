"""
Automatic Video Downloader & Test Suite Generator for Q1 Real-World CCTV Evaluation.

Downloads sample public domain construction site CCTV video sequences into video_test/ folder.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# Sample direct public video streams & sample CCTV footage URLs
SAMPLE_VIDEOS = {
    "construction_cctv_far_range.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    "construction_site_workers_1080p.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/worker-zone-detection.mp4",
}


def download_sample_test_videos(output_dir: Path):
    """Download public CCTV video sequences for Q1 paper dynamic evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Downloader] Output directory: {output_dir}")

    for name, url in SAMPLE_VIDEOS.items():
        dst_path = output_dir / name
        if dst_path.exists() and dst_path.stat().st_size > 0:
            print(f"[Downloader] Existing video found: {dst_path.name} ({dst_path.stat().st_size / (1024*1024):.2f} MB)")
            continue

        print(f"[Downloader] Downloading {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, str(dst_path))
            print(f"[Success] Downloaded {name} ({dst_path.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[Error] Failed to download {name}: {e}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "video_test"
    download_sample_test_videos(out_dir)
