"""
scripts/export_gdut_a6_deploy.py
================================
Exports clean, standalone deployment checkpoints for GDUT A6 models:
- Rep-YOLO11s A6 -> Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt
- Rep-YOLOv8s A6  -> Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolov8s_gdut_a6_deploy.pt

Eliminates all unpickler dependencies on research_trainer and model_code by:
1. Re-binding ResearchDetectionModel to standard ultralytics.nn.tasks.DetectionModel
2. Fusing RepConv multi-branch blocks via switch_to_deploy() to pure Conv2d
3. Re-binding SPPFWithBiFormer to custom_ablation_modules.SPPFWithBiFormer
4. Converting parameters to FP32 for robust CPU/GPU runtime portability
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add GDUT model_code temporarily to load raw checkpoints
MODEL_CODE_DIR = PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "GDUT_A6_CLEAN_VAL_RESULTS" / "model_code"
if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_CODE_DIR))

import custom_ablation_modules as cam
import research_trainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

RAW_CHECKPOINTS = {
    "YOLO11s": {
        "raw": PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "GDUT_A6_CLEAN_VAL_RESULTS" / "YOLO11s" / "weights" / "GDUT_A6_YOLO11s_best.pt",
        "deploy": PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "deploy_weights" / "rep_yolo11s_gdut_a6_deploy.pt",
    },
    "YOLOv8s": {
        "raw": PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "GDUT_A6_CLEAN_VAL_RESULTS" / "YOLOv8s" / "weights" / "GDUT_A6_YOLOv8s_best.pt",
        "deploy": PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "deploy_weights" / "rep_yolov8s_gdut_a6_deploy.pt",
    },
}


def export_checkpoint(name: str, raw_path: Path, deploy_path: Path):
    print(f"\n=======================================================")
    print(f"📦 Processing {name}: {raw_path}")
    assert raw_path.exists(), f"Raw checkpoint not found: {raw_path}"

    deploy_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load raw checkpoint
    raw_ckpt = torch.load(raw_path, map_location="cpu")
    model = raw_ckpt["model"].float()

    # 2. Re-bind ResearchDetectionModel -> DetectionModel
    model.__class__ = DetectionModel

    # 3. Fuse RepConv layers
    fused_count = 0
    for m in model.modules():
        if isinstance(m, cam.RepConv) and hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
            fused_count += 1
    print(f"   [+] Fused {fused_count} RepConv layers into single Conv2d")

    # 4. Handle SPPFWithBiFormer for YOLOv8s if present
    for idx, layer in enumerate(model.model):
        layer_cls_name = type(layer).__name__
        if layer_cls_name == "SPPFWithBiFormer":
            # Reconstruct with custom_ablation_modules.SPPFWithBiFormer
            sppf = layer.sppf
            attention = layer.attention
            new_layer = cam.SPPFWithBiFormer(sppf, attention)
            new_layer.i = getattr(layer, "i", idx)
            new_layer.f = getattr(layer, "f", -1)
            new_layer.type = "SPPFWithBiFormer"
            new_layer.np = getattr(layer, "np", sum(p.numel() for p in new_layer.parameters()))
            model.model[idx] = new_layer
            print(f"   [+] Layer {idx} re-bound to custom_ablation_modules.SPPFWithBiFormer")

    # 5. Sanity check: Ensure no reference to research_trainer remains
    for name_mod, mod in model.named_modules():
        mod_cls_module = getattr(type(mod), "__module__", "")
        if "research_trainer" in mod_cls_module:
            raise RuntimeError(f"Module {name_mod} still has reference to {mod_cls_module}!")

    # 6. Ensure names dict is properly formatted
    expected_names = {0: "blue", 1: "white", 2: "yellow", 3: "red", 4: "none"}
    if not hasattr(model, "names") or not model.names:
        model.names = expected_names
    else:
        # Normalize keys to int
        model.names = {int(k): str(v) for k, v in model.names.items()}

    # 7. Package deploy checkpoint
    deploy_ckpt = {
        "model": model,
        "date": raw_ckpt.get("date"),
        "version": raw_ckpt.get("version", "8.4.115"),
        "train_args": raw_ckpt.get("train_args", {}),
    }

    torch.save(deploy_ckpt, deploy_path)
    file_size_mb = deploy_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved clean deploy checkpoint to {deploy_path} ({file_size_mb:.2f} MB)")
    print(f"   Classes: {model.names}")


def verify_clean_load():
    print(f"\n=======================================================")
    print("🧪 Verifying Standalone Loading in Isolation...")

    # Remove model_code from sys.path and sys.modules
    for mod_name in list(sys.modules.keys()):
        if "research_trainer" in mod_name or "model_code" in mod_name:
            del sys.modules[mod_name]
    if str(MODEL_CODE_DIR) in sys.path:
        sys.path.remove(str(MODEL_CODE_DIR))

    for name, paths in RAW_CHECKPOINTS.items():
        deploy_path = paths["deploy"]
        print(f"\nTesting isolated load for {name}: {deploy_path}")
        yolo_model = YOLO(str(deploy_path))
        print(f"   [+] Successfully loaded with ultralytics.YOLO!")
        print(f"   [+] Names: {yolo_model.names}")

        # Run dummy inference
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = yolo_model.predict(dummy_img, imgsz=640, verbose=False)
        print(f"   [+] Dummy inference passed! Boxes output: {len(results[0].boxes)}")

    print("\n✅ All GDUT A6 deploy checkpoints verified successfully without research_trainer!")


if __name__ == "__main__":
    for name, paths in RAW_CHECKPOINTS.items():
        export_checkpoint(name, paths["raw"], paths["deploy"])
    verify_clean_load()
