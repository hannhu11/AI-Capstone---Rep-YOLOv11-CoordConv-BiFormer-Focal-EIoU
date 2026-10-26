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
        self._last_tg_warn_time = 0.0
        self._last_zalo_warn_time = 0.0

        # Background worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Background Telegram chat auto-discovery loop
        self.discovery_thread = threading.Thread(target=self._telegram_discovery_loop, daemon=True)
        self.discovery_thread.start()

        # Background Zalo chat auto-discovery loop
        self.zalo_discovery_thread = threading.Thread(target=self._zalo_discovery_loop, daemon=True)
        self.zalo_discovery_thread.start()

    def _telegram_discovery_loop(self):
        """Polls Telegram getUpdates every 3 seconds if chat_id is not yet bound."""
        while self.running:
            try:
                if not config.telegram_chat_id:
                    discovered = self.telegram.discover_latest_chat_id()
                    if discovered:
                        welcome_msg = (
                            "🤖 *Rep-YOLO11s CCTV Safety Surveillance Connected!*\n\n"
                            "Real-time safety violation alerts are now active for this chat.\n"
                            "• Camera: `CCTV_GATE_A1_1080P`\n"
                            "• Mode: Live Real-Time Monitoring\n"
                            "• Monitored Violations: Missing Helmet / Missing Vest\n"
                            "• Status: ACTIVE ✅"
                        )
                        self.telegram.send_message(welcome_msg, chat_id=discovered)
            except Exception as e:
                pass
            time.sleep(3.0)

    def _zalo_discovery_loop(self):
        """Polls Zalo getUpdates every 3 seconds if chat_id is not yet bound."""
        while self.running:
            try:
                if not config.zalo_chat_id:
                    discovered = self.zalo.discover_latest_chat_id()
                    if discovered:
                        welcome_msg = (
                            "[CANH BAO AN TOAN PPE - ZALO BOT PLATFORM]\n"
                            "He thong giam sat an toan CCTV Rep-YOLO11s da ket noi thanh cong den Zalo Chat!\n"
                            "• Camera: CCTV_GATE_A1_1080P\n"
                            "• Trang thai: DANG HOAT DONG ✅"
                        )
                        self.zalo.send_message(welcome_msg, chat_id=discovered)
            except Exception:
                pass
            time.sleep(3.0)

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
                if tg_ok:
                    print(f"📢 [TELEGRAM] Worker #{alert_event['worker_id']} alert: SENT ({tg_msg})")
                else:
                    now_t = time.time()
                    if now_t - self._last_tg_warn_time > 30.0:
                        self._last_tg_warn_time = now_t
                        print(f"ℹ️ [TELEGRAM READY] Alerts stored in local audit stream. Send /start to @safety_helmet_cctv_bot to auto-link push alerts.")

            # 2. Dispatch to Zalo
            if config.zalo_enabled:
                zalo_ok, zalo_msg = self.zalo.dispatch_alert(alert_event)
                dispatch_results["zalo"] = {"success": zalo_ok, "detail": zalo_msg}
                if zalo_ok and "stored" not in zalo_msg.lower():
                    print(f"📢 [ZALO] Worker #{alert_event['worker_id']} alert: SENT ({zalo_msg})")
                elif not zalo_ok:
                    now_t = time.time()
                    if now_t - self._last_zalo_warn_time > 30.0:
                        self._last_zalo_warn_time = now_t
                        print(f"ℹ️ [ZALO AUDIT ACTIVE] Alerts logged to audit trail. Set Zalo Chat ID or Webhook in Settings to receive live messages.")

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
        while not self.alert_queue.empty():
            try:
                self.alert_queue.get_nowait()
                self.alert_queue.task_done()
            except Exception:
                break
        self._save_audit_log()


# Global singleton instance
alert_manager = AlertManager()
