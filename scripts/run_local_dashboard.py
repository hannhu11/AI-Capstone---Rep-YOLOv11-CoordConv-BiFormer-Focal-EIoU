"""
scripts/run_local_dashboard.py
==============================
Local GPU-Accelerated Launcher for Industrial CCTV Safety Surveillance Dashboard.
Executes the FastAPI server with CUDA FP16 inference on RTX 3050 Laptop GPU.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
from src.config import config


def main():
    print("=" * 80)
    print("🚀 LAUNCHING INDUSTRIAL CCTV SAFETY SURVEILLANCE DASHBOARD (LOCAL GPU)")
    print(f"📍 Video Source: {config.video_source}")
    print(f"🧠 Model Weights: {config.model_path}")
    print(f"⚡ Device:       {config.device.upper()} (FP16={config.use_half})")
    print(f"🌐 Web Server:   http://{config.web_host}:{config.web_port}")
    print(f"🤖 Telegram Bot: {config.telegram_bot_token[:10]}... (@safety_helmet_cctv_bot)")
    print(f"💬 Zalo Bot:     {config.zalo_bot_token[:10]}...")
    print("=" * 80)

    from src.web.app import app
    uvicorn.run(
        app,
        host=config.web_host,
        port=config.web_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
