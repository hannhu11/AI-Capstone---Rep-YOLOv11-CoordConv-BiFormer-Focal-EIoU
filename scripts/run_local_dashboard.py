"""
scripts/run_local_dashboard.py
==============================
Local GPU-Accelerated Launcher for Industrial CCTV Safety Surveillance Dashboard.
Executes the FastAPI server with CUDA FP16 inference on RTX 3050 Laptop GPU.
Automatically opens the browser to http://localhost:8000 to prevent Windows ERR_ADDRESS_INVALID.
"""

from __future__ import annotations
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
from src.config import config


def _auto_open_browser():
    """Waits for server socket to listen and automatically opens browser."""
    time.sleep(1.5)
    url = f"http://localhost:{config.web_port}"
    print(f"\n🌐 [AUTO-LAUNCH] Opening Dashboard in your browser: {url} ...\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"⚠️ Could not auto-launch browser: {e}")


def main():
    print("=" * 82)
    print("🚀 HỆ THỐNG GIÁM SÁT AN TOÀN LAO ĐỘNG CCTV THỜI GIAN THỰC (EDGE SIMULATION)")
    print("=" * 82)
    print(f"📍 Video Nguồn:       {config.video_source}")
    print(f"🧠 Trọng số Mô hình:  {config.model_path}")
    print(f"⚡ Thiết bị Suy luận: {config.device.upper()} (FP16={config.use_half})")
    print(f"🤖 Telegram Bot:      @safety_helmet_cctv_bot (Chat ID: {config.telegram_chat_id or 'Auto-Detecting'})")
    print(f"💬 Zalo Bot:          {config.zalo_bot_token[:10]}... (Configured={bool(config.zalo_chat_id or config.zalo_webhook_url)})")
    print("-" * 82)
    print("🌐 ĐỊA CHỈ TRUY CẬP TRÊN TRÌNH DUYỆT (CHROME / EDGE / COC COC):")
    print(f"   👉 http://localhost:{config.web_port}")
    print(f"   👉 http://127.0.0.1:{config.web_port}")
    print("⚠️  LƯU Ý: KHÔNG gõ http://0.0.0.0:8000 (Windows Chrome sẽ báo lỗi ERR_ADDRESS_INVALID)!")
    print("=" * 82)

    # Spawn auto-open thread
    threading.Thread(target=_auto_open_browser, daemon=True).start()

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
