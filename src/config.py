"""
src/config.py
=============
Centralized configuration for the Industrial CCTV Safety Surveillance & Alerting System.
Decouples all model weights, video sources, thresholds, and bot tokens via .env / environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Decoupled Model Weights (Default to Champion Rep-YOLO11s checkpoint)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Output" / "kaggle_results" / "hannhu40022_warmstart_ablations" / "rep_yolo11s_chv7_best_deploy.pt"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "yolo11s.pt"

# Video simulation source
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "video_test" / "construction_site_workers_1080p.mp4"

# Snapshots directory
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Alerts log directory
ALERTS_LOG_DIR = PROJECT_ROOT / "Output" / "alerts"
ALERTS_LOG_DIR.mkdir(parents=True, exist_ok=True)


class SystemConfig:
    """Runtime Configuration with live mutable parameters."""

    def __init__(self):
        # 1. Model & Inference Settings
        self.model_path: str = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else FALLBACK_MODEL_PATH))
        self.conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
        self.iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
        self.imgsz: int = int(os.getenv("IMGSZ", "640"))
        self.device: str = os.getenv("DEVICE", "cuda:0")
        self.use_half: bool = os.getenv("USE_HALF", "True").lower() in ("true", "1", "yes")

        # 2. Tracking Settings (ByteTrack)
        self.track_high_thresh: float = float(os.getenv("TRACK_HIGH_THRESH", "0.45"))
        self.track_low_thresh: float = float(os.getenv("TRACK_LOW_THRESH", "0.15"))
        self.new_track_thresh: float = float(os.getenv("NEW_TRACK_THRESH", "0.50"))
        self.track_buffer: int = int(os.getenv("TRACK_BUFFER", "60"))  # frames to retain lost tracks (~2 sec)
        self.match_thresh: float = float(os.getenv("MATCH_THRESH", "0.70"))  # IoU distance threshold

        # 3. Spatial Relational Safety Rules
        self.head_box_ratio: float = float(os.getenv("HEAD_BOX_RATIO", "0.35"))  # upper 35% of person height
        self.torso_box_min_ratio: float = float(os.getenv("TORSO_MIN_RATIO", "0.15"))
        self.torso_box_max_ratio: float = float(os.getenv("TORSO_MAX_RATIO", "0.75"))
        self.torso_min_ratio: float = self.torso_box_min_ratio
        self.torso_max_ratio: float = self.torso_box_max_ratio
        self.helmet_overlap_thresh: float = float(os.getenv("HELMET_OVERLAP_THRESH", "0.15"))
        self.vest_overlap_thresh: float = float(os.getenv("VEST_OVERLAP_THRESH", "0.20"))

        # 4. Debounce & Anti-Spam Cooldown Settings
        self.consecutive_violation_frames: int = int(os.getenv("CONSECUTIVE_VIOLATION_FRAMES", "8"))  # ~0.3-0.5s at 20-30 FPS
        self.alert_cooldown_seconds: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "30.0"))

        # 5. Telegram Bot Credentials
        self.telegram_bot_token: str = os.getenv(
            "TELEGRAM_BOT_TOKEN", "8931795673:AAElznwj0coP6ef1NCtPNQj3veGaWRMvnNU"
        )
        self.telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
        self.telegram_enabled: bool = os.getenv("TELEGRAM_ENABLED", "True").lower() in ("true", "1", "yes")

        # 6. Zalo Bot Credentials
        self.zalo_bot_token: str = os.getenv(
            "ZALO_BOT_TOKEN", "4263098262366049360:ZeYiCYoZHuZVprjkhdzjapYQJssZWVdBpBtUTMuHhWGAPHamYqYdPoiYTmQlMlzo"
        )
        self.zalo_api_base: str = os.getenv("ZALO_API_BASE", "https://bot-api.zaloplatforms.com")
        self.zalo_chat_id: str = os.getenv("ZALO_CHAT_ID", "")
        self.zalo_webhook_url: str = os.getenv("ZALO_WEBHOOK_URL", "")
        self.zalo_enabled: bool = os.getenv("ZALO_ENABLED", "True").lower() in ("true", "1", "yes")

        # 7. Video Simulation Source & Web Service
        self.video_source: str = os.getenv("VIDEO_SOURCE", str(DEFAULT_VIDEO_PATH))
        self.camera_id: str = os.getenv("CAMERA_ID", "CCTV_GATE_A1_1080P")
        self.web_host: str = os.getenv("WEB_HOST", "0.0.0.0")
        self.web_port: int = int(os.getenv("WEB_PORT", "8000"))

    def to_dict(self) -> Dict[str, any]:
        return {
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "use_half": self.use_half,
            "consecutive_violation_frames": self.consecutive_violation_frames,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "telegram_configured": bool(self.telegram_bot_token and self.telegram_chat_id),
            "zalo_configured": bool(self.zalo_bot_token and (self.zalo_chat_id or self.zalo_webhook_url)),
            "video_source": self.video_source,
            "camera_id": self.camera_id,
        }



def update_env_file(updates: Dict[str, str]) -> bool:
    """Updates key-value pairs in the .env file while preserving comments and structure."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        example_path = PROJECT_ROOT / ".env.example"
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
        else:
            env_path.touch()

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        keys_found = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                new_lines.append(line)
                continue
            k, _ = stripped.split("=", 1)
            k = k.strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                keys_found.add(k)
            else:
                new_lines.append(line)

        for k, v in updates.items():
            if k not in keys_found:
                new_lines.append(f"{k}={v}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        print(f"⚠️ Failed to update .env: {e}")
        return False


# Global singleton configuration
config = SystemConfig()
