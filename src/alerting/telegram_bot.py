"""
src/alerting/telegram_bot.py
============================
Telegram Bot Alert Dispatcher for Construction Safety Violations.
Bot Token: 8931795673:AAElznwj0coP6ef1NCtPNQj3veGaWRMvnNU (@safety_helmet_cctv_bot)
Features:
- Markdown formatted safety violation messages.
- Photo attachment with bounding box overlays.
- Dynamic chat_id discovery from /getUpdates.
- Connection testing and diagnostic logging.
"""

from __future__ import annotations
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.config import config


class TelegramAlertBot:
    """Dispatches real-time alerts to Telegram."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or config.telegram_bot_token
        self.chat_id = chat_id or config.telegram_chat_id
        self.api_base = f"https://api.telegram.org/bot{self.token}"

    def test_connection(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify bot token validity with getMe."""
        url = f"{self.api_base}/getMe"
        try:
            r = requests.get(url, timeout=5.0)
            data = r.json()
            if r.status_code == 200 and data.get("ok"):
                return True, data.get("result", {})
            return False, {"error": data.get("description", "Unknown error"), "code": r.status_code}
        except Exception as e:
            return False, {"error": str(e)}

    def discover_latest_chat_id(self) -> Optional[str]:
        """Auto-discovers the latest chat_id by querying getUpdates."""
        url = f"{self.api_base}/getUpdates"
        try:
            r = requests.get(url, timeout=5.0)
            data = r.json()
            if data.get("ok") and data.get("result"):
                # Take the most recent message's chat id
                for update in reversed(data["result"]):
                    msg = update.get("message") or update.get("channel_post") or update.get("my_chat_member")
                    if msg:
                        chat = msg.get("chat", {})
                        c_id = str(chat.get("id"))
                        if c_id:
                            self.chat_id = c_id
                            config.telegram_chat_id = c_id
                            print(f"✅ Discovered Telegram Chat ID: {c_id} (Username: {chat.get('username')})")
                            return c_id
        except Exception as e:
            print(f"⚠️ Telegram getUpdates error: {e}")
        return None

    def send_message(self, text: str, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """Sends a text message with Markdown formatting."""
        target_chat = chat_id or self.chat_id
        if not target_chat:
            target_chat = self.discover_latest_chat_id()
        if not target_chat:
            return False, "No Telegram Chat ID configured. Please open @safety_helmet_cctv_bot and press Start."

        url = f"{self.api_base}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            r = requests.post(url, json=payload, timeout=8.0)
            data = r.json()
            if r.status_code == 200 and data.get("ok"):
                return True, "Message sent successfully"
            return False, f"Telegram error {r.status_code}: {data.get('description', '')}"
        except Exception as e:
            return False, f"Network exception: {str(e)}"

    def send_photo(self, photo_path: str | Path, caption: str, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """Uploads a photo snapshot along with caption."""
        target_chat = chat_id or self.chat_id
        if not target_chat:
            target_chat = self.discover_latest_chat_id()
        if not target_chat:
            return False, "No Telegram Chat ID configured. Please open @safety_helmet_cctv_bot and press Start."

        url = f"{self.api_base}/sendPhoto"
        file_path = Path(photo_path)
        if not file_path.exists():
            return False, f"Snapshot file not found: {photo_path}"

        try:
            with open(file_path, "rb") as f:
                files = {"photo": (file_path.name, f, "image/jpeg")}
                data = {
                    "chat_id": target_chat,
                    "caption": caption,
                    "parse_mode": "Markdown",
                }
                r = requests.post(url, data=data, files=files, timeout=12.0)
                res = r.json()
                if r.status_code == 200 and res.get("ok"):
                    return True, "Photo sent successfully"
                return False, f"Telegram error {r.status_code}: {res.get('description', '')}"
        except Exception as e:
            return False, f"Network exception: {str(e)}"

    def dispatch_alert(self, alert_event: Dict[str, Any]) -> Tuple[bool, str]:
        """Formats and dispatches safety violation alert."""
        v_list = [v.replace("_", " ") for v in alert_event.get("violations", [])]
        v_str = "\n".join([f"  • *{v}*" for v in v_list])
        helmet_info = (
            f"✅ YES ({alert_event.get('helmet_color', 'unknown')})"
            if alert_event.get("has_helmet")
            else "❌ NO"
        )
        vest_info = "✅ YES" if alert_event.get("has_vest") else "❌ NO"

        caption = (
            f"🚨 *CRITICAL SAFETY VIOLATION DETECTED* 🚨\n\n"
            f"📍 *Camera:* `{alert_event.get('camera_id')}`\n"
            f"🕒 *Timestamp:* `{alert_event.get('timestamp')}`\n"
            f"👷 *Worker ID:* `#{alert_event.get('worker_id')}` (Confidence: {alert_event.get('confidence', 0)*100:.1f}%)\n"
            f"⚠️ *Violations:*\n{v_str}\n\n"
            f"🔍 *Check Summary:*\n"
            f"  - Hard Hat: {helmet_info}\n"
            f"  - Hi-Vis Vest: {vest_info}\n"
            f"  - Duration: {alert_event.get('violation_duration_frames', 0)} frames\n\n"
            f"⚡ *Action Required:* Safety officer please inspect worker immediately!"
        )

        snapshot_path = alert_event.get("snapshot_path")
        if snapshot_path and Path(snapshot_path).exists():
            return self.send_photo(snapshot_path, caption)
        else:
            return self.send_message(caption)
