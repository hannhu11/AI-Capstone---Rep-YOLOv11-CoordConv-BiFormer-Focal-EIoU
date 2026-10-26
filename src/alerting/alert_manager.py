"""
src/alerting/alert_manager.py
=============================
Unified Alert Manager & Asynchronous Dispatch Queue.
Dispatches confirmed safety violation alerts to:
1. Telegram Bot (@safety_helmet_cctv_bot)
2. Zalo Bot Platform
3. Local JSON Audit Log (for reproducible records)
4. In-Memory Alert Buffer (for Web Dashboard UI)
Runs asynchronously on a background worker thread to prevent network latency from choking the CCTV stream.
"""

from __future__ import annotations
import json
import time
import queue
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import config, ALERTS_LOG_DIR
from src.alerting.telegram_bot import TelegramAlertBot
from src.alerting.zalo_bot import ZaloAlertBot


class AlertManager:
    """Thread-safe Asynchronous Alert Dispatcher."""

    def __init__(self):
        self.telegram = TelegramAlertBot()
        self.zalo = ZaloAlertBot()

        self.alert_queue: queue.Queue = queue.Queue(maxsize=100)
        self.recent_alerts: List[Dict[str, Any]] = []
        self.audit_log_path = ALERTS_LOG_DIR / "alerts_audit_log.json"

        self._load_existing_audit_log()

        # Background worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _load_existing_audit_log(self):
        if self.audit_log_path.exists():
            try:
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    self.recent_alerts = json.load(f)
                    if not isinstance(self.recent_alerts, list):
                        self.recent_alerts = []
            except Exception:
                self.recent_alerts = []

    def _save_audit_log(self):
        try:
            with open(self.audit_log_path, "w", encoding="utf-8") as f:
                json.dump(self.recent_alerts[-200:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to write alerts audit log: {e}")

    def enqueue_alert(self, alert_event: Dict[str, Any]):
        """Put alert into asynchronous processing queue."""
        try:
            self.alert_queue.put_nowait(alert_event)
        except queue.Full:
            print("⚠️ Alert queue is full, dropping oldest alert")
            try:
                self.alert_queue.get_nowait()
                self.alert_queue.put_nowait(alert_event)
            except Exception:
                pass

    def _worker_loop(self):
        while self.running:
            try:
                alert_event = self.alert_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Record in-memory
            alert_record = dict(alert_event)
            dispatch_results = {}

            # 1. Dispatch to Telegram
            if config.telegram_enabled:
                tg_ok, tg_msg = self.telegram.dispatch_alert(alert_event)
                dispatch_results["telegram"] = {"success": tg_ok, "detail": tg_msg}
                print(f"📢 [TELEGRAM] Worker #{alert_event['worker_id']} alert: {'OK' if tg_ok else 'FAILED'} ({tg_msg})")

            # 2. Dispatch to Zalo
            if config.zalo_enabled:
                zalo_ok, zalo_msg = self.zalo.dispatch_alert(alert_event)
                dispatch_results["zalo"] = {"success": zalo_ok, "detail": zalo_msg}
                print(f"📢 [ZALO] Worker #{alert_event['worker_id']} alert: {'OK' if zalo_ok else 'FAILED'} ({zalo_msg})")

            alert_record["dispatch_results"] = dispatch_results
            self.recent_alerts.insert(0, alert_record)
            if len(self.recent_alerts) > 200:
                self.recent_alerts = self.recent_alerts[:200]

            self._save_audit_log()
            self.alert_queue.task_done()

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent alerts for web UI consumption."""
        return self.recent_alerts[:limit]

    def clear_alerts(self):
        self.recent_alerts = []
        self._save_audit_log()


# Global singleton instance
alert_manager = AlertManager()
