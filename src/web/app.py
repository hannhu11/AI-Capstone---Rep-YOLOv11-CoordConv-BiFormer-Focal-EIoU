"""
src/web/app.py
==============
FastAPI Industrial Surveillance Server & CCTV Streaming Backend.
Features:
- High-performance asynchronous REST endpoints.
- Real-time MJPEG video stream with bounding boxes, tracking IDs, and safety status.
- Live WebSocket & REST telemetry feed (FPS, Latency, Worker Count, Compliance Rate).
- Hot-swappable model weights & video sources.
- Telegram & Zalo Alert testing and auto-discovery.
"""

from __future__ import annotations
import os
import cv2
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import config, SNAPSHOTS_DIR, PROJECT_ROOT, update_env_file
from src.engine.pipeline import cctv_pipeline
from src.alerting.alert_manager import alert_manager


app = FastAPI(
    title="Industrial CCTV Safety Surveillance & Alerting System",
    description="Rep-YOLO11s Spatial Relational ByteTrack MOT Surveillance Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount snapshots directory to serve violation snapshot images
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# Pydantic Request Models
class SettingsPayload(BaseModel):
    conf_threshold: Optional[float] = None
    consecutive_violation_frames: Optional[int] = None
    telegram_chat_id: Optional[str] = None
    zalo_chat_id: Optional[str] = None
    zalo_webhook_url: Optional[str] = None


class SwitchSourcePayload(BaseModel):
    video_source: Optional[str] = None
    source: Optional[str] = None

    @property
    def target_source(self) -> str:
        return self.video_source or self.source or ""


class SwitchModelPayload(BaseModel):
    model_path: str


class TelegramTestPayload(BaseModel):
    chat_id: Optional[str] = None


class ZaloTestPayload(BaseModel):
    webhook_url: Optional[str] = None
    chat_id: Optional[str] = None


# 1. Main Dashboard UI
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = TEMPLATES_DIR / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# 2. MJPEG Video Stream Generator (Decoupled, zero duplicate GPU inference)
def mjpeg_frame_generator():
    last_idx = -1
    while cctv_pipeline.is_running:
        jpeg_bytes, current_idx = cctv_pipeline.get_latest_frame(last_frame_idx=last_idx, timeout=0.5)
        if jpeg_bytes is None or current_idx == last_idx:
            time.sleep(0.01)
            continue
        last_idx = current_idx

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# 3. Telemetry & Metrics API
@app.get("/api/metrics")
async def get_metrics():
    return JSONResponse(content=cctv_pipeline.get_telemetry())


@app.get("/api/status")
async def get_status():
    import torch
    return {
        "status": "online",
        "device": cctv_pipeline.detector.device.upper(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "model_path": cctv_pipeline.detector.model_path,
        "video_source": str(cctv_pipeline.source_path),
        "classes": cctv_pipeline.detector.class_names,
        "config": config.to_dict(),
    }


# 4. Alerts API
@app.get("/api/alerts")
async def get_alerts():
    return JSONResponse(content=alert_manager.get_recent_alerts(limit=50))


@app.post("/api/alerts/clear")
async def clear_alerts():
    alert_manager.clear_alerts()
    return {"message": "Alert history cleared"}


# 5. Live Settings Update API
@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    env_updates = {}
    if payload.conf_threshold is not None:
        config.conf_threshold = payload.conf_threshold
        cctv_pipeline.detector.conf_thresh = payload.conf_threshold
        env_updates["CONF_THRESHOLD"] = str(payload.conf_threshold)
    if payload.consecutive_violation_frames is not None:
        config.consecutive_violation_frames = payload.consecutive_violation_frames
        cctv_pipeline.safety_engine.consecutive_frames_thresh = payload.consecutive_violation_frames
        env_updates["CONSECUTIVE_VIOLATION_FRAMES"] = str(payload.consecutive_violation_frames)
    if payload.telegram_chat_id is not None:
        config.telegram_chat_id = payload.telegram_chat_id
        alert_manager.telegram.chat_id = payload.telegram_chat_id
        env_updates["TELEGRAM_CHAT_ID"] = payload.telegram_chat_id
    if payload.zalo_chat_id is not None:
        config.zalo_chat_id = payload.zalo_chat_id
        alert_manager.zalo.chat_id = payload.zalo_chat_id
        env_updates["ZALO_CHAT_ID"] = payload.zalo_chat_id
    if payload.zalo_webhook_url is not None:
        val = payload.zalo_webhook_url.strip()
        if val.startswith("http://") or val.startswith("https://"):
            config.zalo_webhook_url = val
            alert_manager.zalo.webhook_url = val
            env_updates["ZALO_WEBHOOK_URL"] = val
        elif val:
            config.zalo_chat_id = val
            alert_manager.zalo.chat_id = val
            env_updates["ZALO_CHAT_ID"] = val

    if env_updates:
        update_env_file(env_updates)

    return {"message": "Settings updated and saved to .env successfully", "config": config.to_dict()}


@app.get("/api/videos")
@app.get("/api/sources")
async def list_available_videos():
    video_dir = PROJECT_ROOT / "video_test"
    videos = []
    if video_dir.exists():
        for vp in sorted(video_dir.glob("*.mp4")):
            videos.append({
                "filename": vp.name,
                "path": f"video_test/{vp.name}",
                "size_mb": round(vp.stat().st_size / (1024 * 1024), 2),
            })
    return {"videos": videos, "active_source": str(cctv_pipeline.source_path)}


# 6. Video Source Controls & Upload
@app.post("/api/switch_source")
@app.post("/api/controls/source")
async def switch_video_source(payload: SwitchSourcePayload):
    target = payload.target_source
    if not target:
        raise HTTPException(status_code=400, detail="Missing 'video_source' or 'source' field in request body")
    cctv_pipeline.switch_source(target)
    alert_manager.clear_alerts()
    return {"message": f"Switched video source to {target}", "alerts_cleared": True}


@app.post("/api/webhook/zalo")
async def zalo_webhook_receiver(request: Request):
    """Local webhook endpoint to ingest Zalo Bot Platform webhook payloads."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    return {"status": "received", "data": data}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    save_dir = PROJECT_ROOT / "video_test"
    save_dir.mkdir(exist_ok=True)
    dest_path = save_dir / file.filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cctv_pipeline.switch_source(str(dest_path))
    alert_manager.clear_alerts()
    return {"message": "Uploaded and loaded video successfully", "filename": file.filename}


# 7. Model Checkpoint Hot-Swap API
@app.post("/api/switch_model")
async def switch_model(payload: SwitchModelPayload):
    try:
        cctv_pipeline.detector.load_model(payload.model_path)
        config.model_path = payload.model_path
        return {"message": f"Successfully loaded model: {payload.model_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")


# 8. Bot Integration Test Endpoints
@app.get("/api/telegram/discover_chat_id")
async def discover_tg_chat_id():
    chat_id = alert_manager.telegram.discover_latest_chat_id()
    return {"chat_id": chat_id, "found": bool(chat_id)}


@app.get("/api/zalo/discover_chat_id")
async def discover_zalo_chat_id():
    chat_id = alert_manager.zalo.discover_latest_chat_id()
    return {"chat_id": chat_id, "found": bool(chat_id)}


@app.post("/api/alerts/test-telegram")
@app.post("/api/test_alert/telegram")
async def test_telegram_alert(payload: Optional[TelegramTestPayload] = None):
    chat_id = payload.chat_id if payload and payload.chat_id else None
    if not chat_id:
        chat_id = config.telegram_chat_id or alert_manager.telegram.discover_latest_chat_id()

    test_msg = (
        "🔔 *[TEST ALERT] CCTV Safety Surveillance System*\n\n"
        "This is an automated test ping from Rep-YOLO11s CCTV Safety Engine.\n"
        "Connection to Telegram Bot verified successfully! ✅"
    )
    ok, msg = alert_manager.telegram.send_message(test_msg, chat_id=chat_id)
    return {"success": ok, "message": msg, "chat_id": chat_id}


@app.post("/api/alerts/test-zalo")
@app.post("/api/test_alert/zalo")
async def test_zalo_alert(payload: Optional[ZaloTestPayload] = None):
    if payload:
        if payload.webhook_url:
            val = payload.webhook_url.strip()
            if val.startswith("http://") or val.startswith("https://"):
                alert_manager.zalo.webhook_url = val
            else:
                alert_manager.zalo.chat_id = val
        if payload.chat_id:
            alert_manager.zalo.chat_id = payload.chat_id

    test_msg = (
        "[TEST ALERT - ZALO BOT PLATFORM]\n"
        "He thong giam sat an toan CCTV Rep-YOLO11s da ket noi thanh cong den Zalo Bot Platform! ✅"
    )
    ok, msg = alert_manager.zalo.send_message(test_msg)
    return {"success": ok, "message": msg}


def run_server():
    import uvicorn
    uvicorn.run(
        app,
        host=config.web_host,
        port=config.web_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
