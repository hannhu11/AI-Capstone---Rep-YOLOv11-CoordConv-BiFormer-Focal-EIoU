"""
src/engine/detector.py
======================
Decoupled YOLO / Rep-YOLO11s Inference Engine.
- Injects custom modules (CoordConv, RepConv, BiFormerSequentialWrapper, ConvBNAct) into namespace.
- Supports CUDA FP16 acceleration for true real-time performance.
- Seamlessly hot-swaps model checkpoints without restarting the web service.
"""

from __future__ import annotations
import sys
import time
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Silence ultralytics FP16 deprecation warning flood
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
try:
    from ultralytics.utils import LOGGER
    import logging
    class UltralyticsHalfFilter(logging.Filter):
        def filter(self, record):
            return "is deprecated" not in record.getMessage()
    LOGGER.addFilter(UltralyticsHalfFilter())
except Exception:
    pass

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Register custom research modules for PyTorch unpickler compatibility
import custom_ablation_modules as cam
import ultralytics.nn.modules as un_mod
import ultralytics.nn.tasks as un_tasks
from scripts.ablation_factory import BiFormerSequentialWrapper

# Set custom module references across all necessary module scopes
setattr(sys.modules["__main__"], "BiFormerSequentialWrapper", BiFormerSequentialWrapper)
setattr(un_mod, "BiFormerSequentialWrapper", BiFormerSequentialWrapper)
setattr(un_tasks, "BiFormerSequentialWrapper", BiFormerSequentialWrapper)

for name in dir(cam):
    obj = getattr(cam, name)
    if isinstance(obj, type):
        setattr(sys.modules["__main__"], name, obj)
        setattr(un_mod, name, obj)
        setattr(un_tasks, name, obj)

from ultralytics import YOLO
from src.config import config


class SafetyObjectDetector:
    """YOLO Detection Wrapper with decoupled model loading and FP16 inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
        use_half: bool = False,
    ):
        self.model_path = model_path or config.model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Auto-detect CUDA device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        self.use_half = use_half and self.device.startswith("cuda")
        self.yolo: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.load_model(self.model_path)

    def load_model(self, model_path: str):
        """Loads or hot-swaps model weights cleanly."""
        target_path = Path(model_path)
        if not target_path.exists():
            # Check relative to root
            target_path = root_dir / model_path

        if not target_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        print(f"🔄 Loading Detection Model: {target_path}")
        self.yolo = YOLO(str(target_path))
        self.class_names = self.yolo.names

        # Disable FP16 for models containing custom modules (CoordConv, BiFormerBlockLite)
        # to eliminate numerical overflow that produces ghost clusters on the floor
        has_custom_modules = False
        if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "modules"):
            for m in self.yolo.model.modules():
                m_type = type(m).__name__
                if m_type in ("CoordConv", "BiFormerBlockLite", "BiFormerBlock", "BiFormerSequentialWrapper"):
                    has_custom_modules = True
                    break
        if has_custom_modules and self.use_half:
            print("⚠️ Disabling FP16 (enforcing FP32) to prevent numerical overflow in custom CoordConv/BiFormer modules.")
            self.use_half = False

        # Warm-up pass on GPU
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(
            dummy_img,
            device=self.device,
            half=self.use_half,
            conf=self.conf_thresh,
            verbose=False,
        )
        self.model_path = str(target_path)
        print(f"✅ Model loaded successfully on {self.device.upper()} (FP16={self.use_half}) | Classes: {self.class_names}")

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: int = 640,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, float]:
        """
        Runs inference on an image / video frame.
        Returns:
            all_detections: list of dicts for all classes (filtered at conf_val >= 0.25 for display)
            person_boxes: numpy array (N, 6) [x1, y1, x2, y2, conf, cls_id] for ByteTrack (down to track_low_thresh 0.15)
            latency_ms: pure inference latency in milliseconds
        """
        conf_val = conf if conf is not None else self.conf_thresh
        iou_val = iou if iou is not None else self.iou_thresh
        infer_conf = min(conf_val, getattr(config, "track_low_thresh", 0.15))

        t0 = time.perf_counter()
        predict_kwargs = {
            "conf": infer_conf,
            "iou": iou_val,
            "imgsz": imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.use_half and "cuda" in str(self.device).lower():
            predict_kwargs["half"] = True

        with torch.inference_mode():
            if self.use_half and "cuda" in str(self.device).lower():
                with torch.amp.autocast("cuda"):
                    results = self.yolo.predict(frame_bgr, **predict_kwargs)
            else:
                results = self.yolo.predict(frame_bgr, **predict_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        all_detections: List[Dict[str, Any]] = []
        person_raw: List[List[float]] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                c_name = self.class_names.get(cls_id, str(cls_id)).lower()

                det_dict = {
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "cls": cls_id,
                    "conf": round(conf_score, 4),
                    "name": c_name,
                }

                # Keep all_detections (equipment display, etc.) at conf_val (e.g. 0.25)
                if conf_score >= conf_val:
                    all_detections.append(det_dict)

                # Collect person detections specifically for tracking (down to track_low_thresh 0.15)
                has_person_class = any("person" in n.lower() or "worker" in n.lower() or "human" in n.lower() for n in self.class_names.values())
                is_person = ("person" in c_name or "worker" in c_name or "human" in c_name) if has_person_class else (cls_id == 0)
                if is_person:
                    person_raw.append([x1, y1, x2, y2, conf_score, cls_id])

        person_arr = np.array(person_raw, dtype=np.float32) if person_raw else np.empty((0, 6), dtype=np.float32)
        return all_detections, person_arr, latency_ms
