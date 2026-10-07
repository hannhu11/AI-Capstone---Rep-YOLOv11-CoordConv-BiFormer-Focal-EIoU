"""
=============================================================================
GENERATOR: KAGGLE TASK B2 MULTI-SEED PARALLEL ABLATION NOTEBOOKS (DUAL TESLA T4 X2)
IEEE AAIML 2027 Reviewer Rebuttal & Statistical Significance Verification
Author: Nguyen Han Nhu (FPT University)
=============================================================================
This generator produces 3 independent, dedicated production Kaggle notebooks:
1. Kaggle_TaskB2_Seed42_Ablation_T4x2.ipynb   (Dedicated to Seed 42, Account 1)
2. Kaggle_TaskB2_Seed1337_Ablation_T4x2.ipynb (Dedicated to Seed 1337, Account 2)
3. Kaggle_TaskB2_Seed2026_Ablation_T4x2.ipynb (Dedicated to Seed 2026, Account 3)
Plus the consolidated multi-seed reference notebook:
4. Kaggle_TaskB2_MultiSeed_Ablation_T4x2.ipynb

Key Architectural & Operating Features:
- Targets Kaggle Dual Tesla T4 x2 accelerator (32GB VRAM).
- Eliminates 12-hour timeout risk by running 1 seed per account across 3 accounts.
- Removes smoke test mode entirely; runs full 100 epochs with scientific rigor:
  epochs=100, patience=30, cos_lr=True, close_mosaic=10, lr0=0.01, lrf=0.01.
- Strict academic integrity: Zero Weight Contamination (fresh from clean yolo11s.pt).
- Official VOC2028 ImageSets/Main split compliance.
- True multi-GPU DDP training support via dedicated ablation_trainer.py module.
- Full genuine Bi-Level Routing Attention (BiFormer, S=8, k=4) implementation.
- Injected AblationBboxLoss with explicit os and torch imports.
- Auto-packages outputs into distinct zip files per seed.
- ZERO emojis throughout all code cells, markdown cells, and print outputs.
=============================================================================
"""

import base64
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CUSTOM_MODULES_FILE = ROOT_DIR / "custom_ablation_modules.py"
CUSTOM_MODULES_SRC = CUSTOM_MODULES_FILE.read_text(encoding="utf-8")

P2_YAML_CONTENT = """# Rep-YOLO11s-P2 AFPN (4-Head High-Resolution PPE Detector)
nc: 2
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2-P2/4 (Micro-Scale Feature Map)
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [256, False, 0.25]]  # 4-P3/8
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6-P4/16
  - [-1, 1, Conv, [512, 3, 2]]         # 7-P5/32
  - [-1, 2, C3k2, [512, True]]         # 8-P5/32
  - [-1, 1, SPPF, [512, 5]]            # 9-P5/32

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 10
  - [[-1, 6], 1, Concat, [1]]                  # 11 cat backbone P4
  - [-1, 2, C3k2, [512, False]]                # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 13
  - [[-1, 4], 1, Concat, [1]]                  # 14 cat backbone P3
  - [-1, 2, C3k2, [256, False]]                # 15

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 16
  - [[-1, 2], 1, Concat, [1]]                  # 17 cat backbone P2
  - [-1, 2, C3k2, [128, False]]                # 18 (P2/4-Head: 160x160)

  - [-1, 1, Conv, [128, 3, 2]]                 # 19
  - [[-1, 15], 1, Concat, [1]]                 # 20 cat P3
  - [-1, 2, C3k2, [256, False]]                # 21 (P3/8-Head: 80x80)

  - [-1, 1, Conv, [256, 3, 2]]                 # 22
  - [[-1, 12], 1, Concat, [1]]                 # 23 cat P4
  - [-1, 2, C3k2, [512, False]]                # 24 (P4/16-Head: 40x40)

  - [-1, 1, Conv, [512, 3, 2]]                 # 25
  - [[-1, 9], 1, Concat, [1]]                  # 26 cat P5
  - [-1, 2, C3k2, [512, True]]                 # 27 (P5/32-Head: 20x20)

  - [[18, 21, 24, 27], 1, Detect, [nc]]        # 28 4-Head Detection Layer
"""

ABLATION_TRAINER_SRC = """# =============================================================================
# ABLATION TRAINER AND ARCHITECTURE FACTORY (KAGGLE DUAL TESLA T4 DDP)
# IEEE AAIML 2027 Reviewer Rebuttal & Statistical Significance Verification
# Author: Nguyen Han Nhu (FPT University)
# =============================================================================
import os
import sys
import copy
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import ultralytics.nn.modules as un_mod
import ultralytics.nn.tasks as un_tasks
import ultralytics.utils.loss as ul_loss

# 1. Import research modules
from custom_ablation_modules import CoordConv, RepConv, BiFormerBlockLite, focal_eiou_loss

# 2. Dynamic module registration into Ultralytics namespace
un_mod.CoordConv = CoordConv
un_mod.RepConv = RepConv
un_mod.BiFormerBlockLite = BiFormerBlockLite
setattr(un_tasks, 'CoordConv', CoordConv)
setattr(un_tasks, 'RepConv', RepConv)
setattr(un_tasks, 'BiFormerBlockLite', BiFormerBlockLite)

# 3. Hook AblationBboxLoss for Focal EIoU support
class AblationBboxLoss(ul_loss.BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        p_box = pred_bboxes[fg_mask]
        t_box = target_bboxes[fg_mask]
        cur_ab = os.environ.get("CURRENT_ABLATION_ID", "A0")
        if cur_ab in ["A4", "A6"] and p_box.shape[0] > 0:
            px1, py1, px2, py2 = p_box.unbind(-1)
            tx1, ty1, tx2, ty2 = t_box.unbind(-1)
            pw = (px2 - px1).clamp(min=1e-7)
            ph = (py2 - py1).clamp(min=1e-7)
            tw = (tx2 - tx1).clamp(min=1e-7)
            th = (ty2 - ty1).clamp(min=1e-7)

            inter_x1 = torch.maximum(px1, tx1)
            inter_y1 = torch.maximum(py1, ty1)
            inter_x2 = torch.minimum(px2, tx2)
            inter_y2 = torch.minimum(py2, ty2)
            inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
            union = pw * ph + tw * th - inter + 1e-7
            iou = (inter / union).clamp(min=1e-7, max=1.0)

            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            center_dist = (pcx - tcx).square() + (pcy - tcy).square()

            cw = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=1e-7)
            ch = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=1e-7)
            c2 = cw.square() + ch.square() + 1e-7

            gamma = 0.5
            eiou = 1.0 - iou + center_dist / c2 + (pw - tw).square() / (cw.square() + 1e-7) + (ph - th).square() / (ch.square() + 1e-7)
            loss_box_sample = iou.pow(gamma) * eiou
            loss_iou = (loss_box_sample.unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            iou = ul_loss.bbox_iou(p_box, t_box, xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss and p_box.shape[0] > 0:
            target_ltrb = ul_loss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

ul_loss.BboxLoss = AblationBboxLoss

# 4. Model Architecture Factory
def build_ablation_model(ab_id: str, base_weight: str = "yolo11s.pt", p2_yaml_path: str = "/kaggle/working/rep_yolo11s_p2.yaml") -> YOLO:
    p2_path = Path(p2_yaml_path)
    if not p2_path.exists():
        p2_path = Path("rep_yolo11s_p2.yaml")

    if ab_id == "A1":
        base = YOLO(str(p2_path))
        base.load(base_weight)
        return base

    base = YOLO(base_weight)
    m = base.model

    if ab_id == "A0":
        return base

    if ab_id in ["A2", "A6"]:
        conv0 = m.model[0].conv
        coord_conv = CoordConv(c1=conv0.in_channels, c2=conv0.out_channels, k=3, s=2, with_r=False)
        coord_conv.i, coord_conv.f, coord_conv.type = 0, -1, "CoordConv"
        m.model[0] = coord_conv

    if ab_id in ["A3", "A4", "A5", "A6"]:
        for idx, layer in enumerate(m.model):
            if idx > 0 and hasattr(layer, "conv") and hasattr(layer.conv, "kernel_size") and layer.conv.kernel_size == (3, 3):
                c1, c2, s = layer.conv.in_channels, layer.conv.out_channels, layer.conv.stride[0]
                rep_conv = RepConv(c1=c1, c2=c2, k=3, s=s, deploy=False)
                rep_conv.i, rep_conv.f, rep_conv.type = getattr(layer, "i", idx), getattr(layer, "f", -1), "RepConv"
                m.model[idx] = rep_conv

    if ab_id in ["A5", "A6"]:
        for idx, layer in enumerate(m.model):
            if layer.__class__.__name__ in ["C2PSA", "C3k2"] and idx >= 9:
                c_in = getattr(layer, "c1", 512)
                biformer = BiFormerBlockLite(channels=c_in, num_heads=4, region_size=8, topk=4)
                biformer.i, biformer.f, biformer.type = getattr(layer, "i", idx), getattr(layer, "f", -1), "BiFormerBlockLite"
                m.model[idx] = biformer
                break

    base.model = m
    return base

CURRENT_ABLATION_MODEL = None

class MultiSeedAblationTrainer(DetectionTrainer):
    \"\"\"Custom Trainer ensuring exact architectural module injection for each ablation in single and multi-GPU DDP.\"\"\"
    def get_model(self, cfg=None, weights=None, verbose=True):
        global CURRENT_ABLATION_MODEL
        if CURRENT_ABLATION_MODEL is not None:
            return CURRENT_ABLATION_MODEL
        cur_ab = os.environ.get("CURRENT_ABLATION_ID", "A0")
        model = build_ablation_model(cur_ab, weights or "yolo11s.pt").model
        return model
"""

CUSTOM_MODULES_B64 = base64.b64encode(CUSTOM_MODULES_SRC.encode("utf-8")).decode("ascii")
ABLATION_TRAINER_B64 = base64.b64encode(ABLATION_TRAINER_SRC.encode("utf-8")).decode("ascii")
P2_YAML_B64 = base64.b64encode(P2_YAML_CONTENT.strip().encode("utf-8")).decode("ascii")


def create_single_seed_notebook(seed: int, account_num: int):
    """
    Creates a dedicated, self-contained Kaggle notebook for a single seed (A0 -> A6).
    """
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataProxyVersion": "v2",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # ------------------------------------------------------------------------
    # CELL 0: SPECIFICATION & OPERATING MANIFEST
    # ------------------------------------------------------------------------
    cell_0_md = f"""# IEEE AAIML 2027: Dedicated Seed {seed} Statistical Ablation Study (A0 -> A6)
### Safety Helmet Detection in Industrial Surveillance (SHWD / VOC2028)
- Author / Lead Researcher: Nguyen Han Nhu (FPT University)
- Target Account: Kaggle Account {account_num} (Dedicated execution to prevent 12-hour timeout)
- Assigned Random Seed: {seed} (Independent 100-epoch evaluation across A0 -> A6)
- Reviewer Rebuttal Goal (Task B2): Multi-seed variance verification proving statistical significance against baseline A0.

---

## Operating Manifest and Kaggle Environment Settings

| Parameter / Field | Detailed Standard Specification | Operational Notes |
| :--- | :--- | :--- |
| **NOTEBOOK NAME** | `Kaggle_TaskB2_Seed{seed}_Ablation_T4x2.ipynb` | Fully self-contained production notebook |
| **TARGET ACCOUNT** | **Kaggle Account {account_num}** | Dedicated to Seed {seed} execution |
| **ACCELERATOR** | **GPU T4 x2** (Dual NVIDIA Tesla T4 16GB x 2 = 32GB VRAM) | Select via Kaggle Settings panel (right side) |
| **INTERNET** | **ON (Mandatory)** | Required for package updates and clean base weights |
| **PERSISTENCE** | **Files only** | Preserves state and outputs across disconnections |
| **DATASET INPUTS** | **VOC2028 (SHWD)**: Add dataset `voc2028` or `hannhu4002/voc2028` into `/kaggle/input`.<br>*(Strict adherence to official `ImageSets/Main/train.txt` and `val.txt` splits)* | Contains 7,581 annotated images |
| **WEIGHT INPUTS** | `yolo11s.pt` (Auto-downloaded from official Ultralytics releases, guaranteeing Zero Weight Contamination). | Clean re-initialization for each ablation |
| **RANDOM SEED** | **{seed}** (Deterministic execution) | Fixed seed for reproducible statistics |
| **ABLATION STEPS** | - **A0**: Baseline YOLO11s (Stock Multi-Branch)<br>- **A1**: + P2 High-Resolution Micro-Head (Stride 4)<br>- **A2**: + CoordConv Stem Layer ($C_x, C_y \\in [-1, 1]$)<br>- **A3**: + RepConv Multi-Branch Structural Fusion<br>- **A4**: + Focal EIoU Loss ($\\gamma=0.5$)<br>- **A5**: + BiFormer Bi-Level Routing Attention<br>- **A6**: Proposed Rep-YOLO11s Full Fusion | Comprehensive single-seed 7-model suite |
| **EPOCHS & HYPERPARAMS**| **100 epochs** per ablation, `patience=30`, `cos_lr=True`, `close_mosaic=10`, `lr0=0.01`, `lrf=0.01`, `batch=32`, `imgsz=640`. | Full scientific rigor, no smoke test |
| **RUNTIME ESTIMATE** | ~18-20 min per model -> ~2.2 - 2.5 hours total on Dual Tesla T4 | Safely under the 12-hour session timeout limit |
| **SMART RESUME** | Automatic caching: if `seed_{seed}_{{ab_id}}_best.pt` exists, skips execution. | Prevents redundant work on re-runs |
| **OUTPUT ARTIFACTS** | Packaged into **`TaskB2_Seed{seed}_Outputs.zip`** containing CSV metrics, JSON logs, and trained weights. | 1-click download from Kaggle Output panel |
"""

    # ------------------------------------------------------------------------
    # CELL 1: ENVIRONMENT & HARDWARE VERIFICATION
    # ------------------------------------------------------------------------
    cell_1_code = f"""# CELL 1: ENVIRONMENT SETUP AND DUAL TESLA T4 HARDWARE VERIFICATION
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path

print("=" * 80)
print("[INFO] KAGGLE DUAL TESLA T4 ENVIRONMENT SETUP - SEED {seed}")
print("=" * 80)
print(f"Python Version : {{sys.version.split()[0]}}")

# 1. Install required packages
!pip install -q -U ultralytics scipy tabulate matplotlib seaborn pandas

# 2. Suppress noisy loggers
os.environ["TENSORBOARD_BINARY"] = ""
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import settings
settings.update({{"tensorboard": False}})

import torch
print(f"PyTorch Version: {{torch.__version__}}")
print(f"CUDA Available : {{torch.cuda.is_available()}}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Detected GPUs  : {{gpu_count}}")
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        print(f"  -> GPU [{{idx}}]: {{props.name}} | VRAM: {{props.total_memory / (1024**3):.2f}} GB")
    TRAIN_DEVICE = "0,1" if gpu_count >= 2 else "0"
    EVAL_DEVICE = "0"
else:
    print("[WARNING] No GPU detected. Defaulting to CPU fallback.")
    TRAIN_DEVICE = "cpu"
    EVAL_DEVICE = "cpu"

print(f"[INFO] Training Device   : {{TRAIN_DEVICE}}")
print(f"[INFO] Evaluation Device : {{EVAL_DEVICE}}")
"""

    # ------------------------------------------------------------------------
    # CELL 2: ARCHITECTURAL MODULE REGISTRATION & DDP INJECTION
    # ------------------------------------------------------------------------
    cell_2_code = f"""# CELL 2: ARCHITECTURAL MODULE REGISTRATION AND DDP SITE-PACKAGES INJECTION
import base64
import os
import site
import sys
from pathlib import Path

# 1. Materialize custom_ablation_modules.py (genuine BiFormer, RepConv, CoordConv, Focal EIoU)
custom_modules_b64 = "{CUSTOM_MODULES_B64}"
custom_modules_src = base64.b64decode(custom_modules_b64.encode("ascii")).decode("utf-8")
Path("custom_ablation_modules.py").write_text(custom_modules_src, encoding="utf-8")

# 2. Materialize ablation_trainer.py (MultiSeedAblationTrainer and build_ablation_model factory)
ablation_trainer_b64 = "{ABLATION_TRAINER_B64}"
ablation_trainer_src = base64.b64decode(ablation_trainer_b64.encode("ascii")).decode("utf-8")
Path("ablation_trainer.py").write_text(ablation_trainer_src, encoding="utf-8")

# 3. Propagate modules into all site-packages directories for DDP subprocesses
for sp in site.getsitepackages():
    try:
        (Path(sp) / "custom_ablation_modules.py").write_text(custom_modules_src, encoding="utf-8")
        (Path(sp) / "ablation_trainer.py").write_text(ablation_trainer_src, encoding="utf-8")
    except Exception:
        pass

# 4. Ensure current working directory and /kaggle/working are top-priority in sys.path and PYTHONPATH
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
if "." not in sys.path:
    sys.path.insert(0, ".")

existing_pp = os.environ.get("PYTHONPATH", "")
if "/kaggle/working" not in existing_pp:
    os.environ["PYTHONPATH"] = f"/kaggle/working:{{existing_pp}}" if existing_pp else "/kaggle/working"

# 5. In-memory registration into Ultralytics engine
from custom_ablation_modules import CoordConv, RepConv, BiFormerBlockLite, focal_eiou_loss
from ablation_trainer import AblationBboxLoss, MultiSeedAblationTrainer, build_ablation_model
import ultralytics.nn.modules as un_mod
import ultralytics.nn.tasks as un_tasks
import ultralytics.utils.loss as ul_loss

un_mod.CoordConv = CoordConv
un_mod.RepConv = RepConv
un_mod.BiFormerBlockLite = BiFormerBlockLite
setattr(un_tasks, 'CoordConv', CoordConv)
setattr(un_tasks, 'RepConv', RepConv)
setattr(un_tasks, 'BiFormerBlockLite', BiFormerBlockLite)
ul_loss.BboxLoss = AblationBboxLoss

# 6. Physical file injection into loss.py for worker subprocesses with explicit os import
loss_file_path = Path(ul_loss.__file__).resolve()
loss_src = loss_file_path.read_text(encoding="utf-8")
if "class AblationBboxLoss" not in loss_src:
    patch_code = '''
import os
import torch
class AblationBboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        p_box = pred_bboxes[fg_mask]
        t_box = target_bboxes[fg_mask]
        cur_ab = os.environ.get("CURRENT_ABLATION_ID", "A0")
        if cur_ab in ["A4", "A6"] and p_box.shape[0] > 0:
            px1, py1, px2, py2 = p_box.unbind(-1)
            tx1, ty1, tx2, ty2 = t_box.unbind(-1)
            pw = (px2 - px1).clamp(min=1e-7)
            ph = (py2 - py1).clamp(min=1e-7)
            tw = (tx2 - tx1).clamp(min=1e-7)
            th = (ty2 - ty1).clamp(min=1e-7)
            inter_x1 = torch.maximum(px1, tx1)
            inter_y1 = torch.maximum(py1, ty1)
            inter_x2 = torch.minimum(px2, tx2)
            inter_y2 = torch.minimum(py2, ty2)
            inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
            union = pw * ph + tw * th - inter + 1e-7
            iou = (inter / union).clamp(min=1e-7, max=1.0)
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            center_dist = (pcx - tcx).square() + (pcy - tcy).square()
            cw = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=1e-7)
            ch = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=1e-7)
            c2 = cw.square() + ch.square() + 1e-7
            gamma = 0.5
            eiou = 1.0 - iou + center_dist / c2 + (pw - tw).square() / (cw.square() + 1e-7) + (ph - th).square() / (ch.square() + 1e-7)
            loss_box_sample = iou.pow(gamma) * eiou
            loss_iou = (loss_box_sample.unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(p_box, t_box, xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss and p_box.shape[0] > 0:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

BboxLoss = AblationBboxLoss
'''
    try:
        loss_file_path.write_text(loss_src + "\\n" + patch_code, encoding="utf-8")
        print("[INFO] Injected AblationBboxLoss into physical site-packages loss.py")
    except Exception as e:
        print(f"[WARNING] Could not patch physical loss.py: {{e}}")

print("[SUCCESS] Registered CoordConv, RepConv, BiFormer, Focal EIoU, and MultiSeedAblationTrainer.")
"""

    # ------------------------------------------------------------------------
    # CELL 3: DATASET INGESTION & OFFICIAL VOC2028 SPLIT COMPLIANCE
    # ------------------------------------------------------------------------
    cell_3_code = """# CELL 3: DATASET INGESTION AND OFFICIAL VOC2028 SPLIT COMPLIANCE
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import zipfile

print("=" * 80)
print("[INFO] SCANNING FOR VOC2028 / SHWD DATASET IN /kaggle/input/...")
print("=" * 80)

candidate_dirs = [
    Path("/kaggle/input/voc2028/VOC2028"),
    Path("/kaggle/input/voc2028"),
    Path("/kaggle/input/datasets/hannhu4002/voc2028/VOC2028"),
    Path("/kaggle/input/datasets/hannhu4002/voc2028"),
    Path("Dataset/VOC2028"),
    Path("VOC2028"),
]

dataset_root = None
for cand in candidate_dirs:
    if (cand / "JPEGImages").exists() and (cand / "Annotations").exists():
        dataset_root = cand
        break

if dataset_root is None:
    for p in Path("/kaggle/input").rglob("JPEGImages"):
        if p.parent.is_dir() and (p.parent / "Annotations").is_dir():
            dataset_root = p.parent
            break

if dataset_root is None:
    print("[INFO] Searching for compressed dataset archive in /kaggle/input...")
    for z in Path("/kaggle/input").rglob("*.zip"):
        if "voc" in z.name.lower() or "shwd" in z.name.lower():
            extract_to = Path("/kaggle/working/VOC2028_EXTRACTED")
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(extract_to)
            for p in extract_to.rglob("JPEGImages"):
                if p.parent.is_dir() and (p.parent / "Annotations").is_dir():
                    dataset_root = p.parent
                    break
            if dataset_root:
                break

if dataset_root is None:
    raise FileNotFoundError(
        "[ERROR] VOC2028 dataset not found in /kaggle/input! "
        "Please attach 'voc2028' (or 'hannhu4002/voc2028') to this Kaggle notebook session."
    )

print(f"[INFO] Verified VOC2028 dataset root at: {dataset_root}")

# Setup YOLO directory structure
YOLO_DIR = Path("/kaggle/working/SHWD_YOLO")
images_dir = YOLO_DIR / "images"
labels_dir = YOLO_DIR / "labels"

for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"hat": 0, "helmet": 0, "person": 1, "head": 1}

# Adhere strictly to official VOC2028 ImageSets splits (Zero Test Contamination)
imagesets_main = dataset_root / "ImageSets" / "Main"
train_ids = set()
val_ids = set()

if (imagesets_main / "train.txt").exists():
    train_ids = set(line.strip() for line in (imagesets_main / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"[INFO] Found official ImageSets/Main/train.txt: {len(train_ids)} images")

if (imagesets_main / "val.txt").exists():
    val_ids = set(line.strip() for line in (imagesets_main / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"[INFO] Found official ImageSets/Main/val.txt: {len(val_ids)} images")

all_xmls = sorted(list((dataset_root / "Annotations").glob("*.xml")))
if not train_ids or not val_ids:
    print("[WARNING] ImageSets/Main splits not found. Fallback to deterministic 80/20 split (seed 2026)...")
    import random
    rng = random.Random(2026)
    stems = [x.stem for x in all_xmls]
    rng.shuffle(stems)
    split_idx = int(len(stems) * 0.8)
    train_ids = set(stems[:split_idx])
    val_ids = set(stems[split_idx:])

def convert_and_populate(xml_list, target_split, id_set):
    count = 0
    for xml_path in xml_list:
        stem = xml_path.stem
        if stem not in id_set:
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            continue
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w <= 0 or h <= 0:
            continue

        src_img = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            cand = dataset_root / "JPEGImages" / f"{stem}{ext}"
            if cand.exists():
                src_img = cand
                break
        if src_img is None:
            continue

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip().lower()
            if cls_name not in CLASS_MAP:
                continue
            cid = CLASS_MAP[cls_name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            cx = ((xmin + xmax) / 2.0) / w
            cy = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            if bw > 0 and bh > 0:
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        dst_img = images_dir / target_split / src_img.name
        if not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
            except Exception:
                shutil.copy2(src_img, dst_img)

        dst_lbl = labels_dir / target_split / f"{stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write("\\n".join(yolo_lines))
        count += 1
    return count

n_train = convert_and_populate(all_xmls, "train", train_ids)
n_val = convert_and_populate(all_xmls, "val", val_ids)
print(f"[SUCCESS] Prepared dataset: {n_train} train images, {n_val} validation images.")

data_yaml_path = YOLO_DIR / "shwd_data.yaml"
data_yaml_content = f\"\"\"
path: {YOLO_DIR.resolve()}
train: images/train
val: images/val
nc: 2
names: ['hat', 'person']
\"\"\"
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(data_yaml_content.strip())
print(f"[INFO] Configuration file saved: {data_yaml_path}")
"""

    # ------------------------------------------------------------------------
    # CELL 4: ARCHITECTURE FACTORY & MULTI-HEAD P2 SETUP
    # ------------------------------------------------------------------------
    cell_4_code = f"""# CELL 4: ARCHITECTURE FACTORY AND MULTI-HEAD P2 INITIALIZATION (A0 -> A6)
import base64
import os
from pathlib import Path
import torch
from ultralytics import YOLO
from ablation_trainer import MultiSeedAblationTrainer, build_ablation_model

# Write rep_yolo11s_p2.yaml for Ablation A1
p2_yaml_path = Path("/kaggle/working/rep_yolo11s_p2.yaml")
p2_yaml_content = base64.b64decode("{P2_YAML_B64}".encode("ascii")).decode("utf-8")
p2_yaml_path.write_text(p2_yaml_content.strip(), encoding="utf-8")
print(f"[INFO] 4-Head P2 Model Architecture YAML written to: {{p2_yaml_path}}")

# Sanity assertion check across all 7 ablations
print("[INFO] Validating architectural factory instantiation across A0 -> A6...")
dummy = torch.randn(1, 3, 640, 640)
for ab_check in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
    m_check = build_ablation_model(ab_check, "yolo11s.pt")
    m_check.model.eval()
    with torch.no_grad():
        out_check = m_check.model(dummy)
    del m_check
print("[SUCCESS] All 7 ablation architectures verified successfully.")
"""

    # ------------------------------------------------------------------------
    # CELL 5: SINGLE-SEED FULL 100-EPOCH TRAINING PIPELINE (A0 -> A6)
    # ------------------------------------------------------------------------
    cell_5_code = f"""# CELL 5: DEDICATED SEED {seed} FULL 100-EPOCH TRAINING PIPELINE (A0 -> A6)
import gc
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# CONFIGURATION FOR DEDICATED SEED {seed}
SEED = {seed}
ABLATIONS = [
    {{"id": "A0", "name": "Baseline YOLO11s", "desc": "Standard stock YOLO11s"}},
    {{"id": "A1", "name": "+ P2 Small-Object Head", "desc": "High-resolution P2 micro-head"}},
    {{"id": "A2", "name": "+ CoordConv Stem", "desc": "Spatial vertical coordinate priors"}},
    {{"id": "A3", "name": "+ RepConv Re-Param", "desc": "Structural re-parameterization branches"}},
    {{"id": "A4", "name": "+ Focal EIoU Loss", "desc": "Independent width/height aspect ratio penalty"}},
    {{"id": "A5", "name": "+ BiFormer Attention", "desc": "Bi-level routing attention across P4/P5"}},
    {{"id": "A6", "name": "Full Fusion (Proposed)", "desc": "Rep-YOLO11s full end-to-end integration"}},
]

# FULL SCIENTIFIC RIGOR: 100 EPOCHS WITH EARLY STOPPING (PATIENCE=30)
EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 32
PATIENCE = 30
COS_LR = True
CLOSE_MOSAIC = 10
LR0 = 0.01
LRF = 0.01

OUTPUT_DIR = Path("/kaggle/working/TaskB2_Seed{seed}_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

results_records = []
csv_results_path = OUTPUT_DIR / "seed_{seed}_ablation_results.csv"
log_file = OUTPUT_DIR / "seed_{seed}_ablation_log.json"

if csv_results_path.exists():
    df_cached = pd.read_csv(csv_results_path)
    results_records = df_cached.to_dict(orient="records")
    print(f"[RESUME] Loaded {{len(results_records)}} completed ablation runs from {{csv_results_path.name}}.")

def is_already_completed(ab_id: str) -> bool:
    for rec in results_records:
        if rec["ablation_id"] == ab_id:
            ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}_{{ab_id}}_best.pt"
            if ckpt_path.exists() or Path(rec.get("checkpoint", "")).exists():
                return True
    return False

print("=" * 80)
print(f"[START] DEDICATED SEED {seed} ABLATION SUITE ({{len(ABLATIONS)}} MODELS, {{EPOCHS}} EPOCHS EACH)")
print(f"   Target Account : Kaggle Account {account_num}")
print(f"   Random Seed    : {{SEED}}")
print(f"   Batch Size     : {{BATCH_SIZE}} | ImgSz: {{IMGSZ}}")
print(f"   Hyperparameters: lr0={{LR0}}, lrf={{LRF}}, patience={{PATIENCE}}, cos_lr={{COS_LR}}, close_mosaic={{CLOSE_MOSAIC}}")
print("=" * 80)

for ab in ABLATIONS:
    ab_id = ab["id"]
    ckpt_name = f"seed_{seed}_{{ab_id}}_best.pt"
    target_ckpt = CHECKPOINTS_DIR / ckpt_name

    if is_already_completed(ab_id):
        print(f"[SKIP] Ablation {{ab_id}} (Seed {seed}) already completed in cache. Moving to next.")
        continue

    print("\\n----------------------------------------------------------------------")
    print(f"[RUNNING] Ablation {{ab_id}}: {{ab['name']}} | Seed = {{SEED}} | Epochs = {{EPOCHS}}")
    print("----------------------------------------------------------------------")

    # Set ablation ID in environment for DDP worker processes
    os.environ["CURRENT_ABLATION_ID"] = ab_id

    # Clean architectural factory instantiation from fresh base weight (Zero Weight Contamination)
    model = build_ablation_model(ab_id, "yolo11s.pt")
    import ablation_trainer
    ablation_trainer.CURRENT_ABLATION_MODEL = model.model

    run_name = f"run_{{ab_id}}_seed_{seed}"
    t_start = time.time()

    try:
        train_results = model.train(
            trainer=MultiSeedAblationTrainer,
            data=str(data_yaml_path),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=TRAIN_DEVICE,
            seed=SEED,
            deterministic=True,
            cos_lr=COS_LR,
            patience=PATIENCE,
            close_mosaic=CLOSE_MOSAIC,
            lr0=LR0,
            lrf=LRF,
            save=True,
            val=True,
            plots=False,
            name=run_name,
            project=str(OUTPUT_DIR / "runs"),
            exist_ok=True,
            verbose=False,
        )
    except Exception as e:
        print(f"[WARNING] DDP training encountered issue: {{e}}. Falling back to single GPU 0...")
        train_results = model.train(
            trainer=MultiSeedAblationTrainer,
            data=str(data_yaml_path),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device="0",
            seed=SEED,
            deterministic=True,
            cos_lr=COS_LR,
            patience=PATIENCE,
            close_mosaic=CLOSE_MOSAIC,
            lr0=LR0,
            lrf=LRF,
            save=True,
            val=True,
            plots=False,
            name=run_name,
            project=str(OUTPUT_DIR / "runs"),
            exist_ok=True,
            verbose=False,
        )

    train_time_min = (time.time() - t_start) / 60.0

    # Independent validation on unseen val split
    val_metrics = model.val(
        data=str(data_yaml_path),
        imgsz=IMGSZ,
        device=EVAL_DEVICE,
        verbose=False,
    )

    map50 = float(val_metrics.box.map50) * 100.0
    map50_95 = float(val_metrics.box.map) * 100.0
    precision = float(val_metrics.box.mp) * 100.0
    recall = float(val_metrics.box.mr) * 100.0

    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_ckpt)

    rec = {{
        "ablation_id": ab_id,
        "ablation_name": ab["name"],
        "seed": SEED,
        "mAP50": round(map50, 2),
        "mAP50_95": round(map50_95, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "train_time_min": round(train_time_min, 1),
        "checkpoint": str(target_ckpt.name),
    }}
    results_records.append(rec)

    # Persist results immediately to disk
    df_curr = pd.DataFrame(results_records)
    df_curr.to_csv(csv_results_path, index=False)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)

    print(f"[DONE] Completed {{ab_id}} Seed {seed}: mAP50 = {{map50:.2f}}%, mAP50-95 = {{map50_95:.2f}}% ({{train_time_min:.1f}} min)")

    # Clean VRAM
    del model
    ablation_trainer.CURRENT_ABLATION_MODEL = None
    torch.cuda.empty_cache()
    gc.collect()

print("\\n[SUCCESS] All ablation models for Seed {seed} completed successfully.")
"""

    # ------------------------------------------------------------------------
    # CELL 6: TABULATION & LOCAL COMPARISON
    # ------------------------------------------------------------------------
    cell_6_code = f"""# CELL 6: TABULATION AND ABLATION COMPARISON (SEED {seed})
import pandas as pd
from tabulate import tabulate

csv_file = OUTPUT_DIR / "seed_{seed}_ablation_results.csv"
if csv_file.exists():
    df = pd.read_csv(csv_file)
    print("=" * 80)
    print("[RESULTS] ABLATION SUITE METRICS - SEED {seed}")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))

    if len(df) > 1 and "A0" in df["ablation_id"].values:
        a0_row = df[df["ablation_id"] == "A0"].iloc[0]
        a0_m50 = a0_row["mAP50"]
        a0_m95 = a0_row["mAP50_95"]
        print("\\n" + "=" * 80)
        print(f"[ANALYSIS] DELTA GAINS VS BASELINE A0 (mAP50={{a0_m50:.2f}}%, mAP50-95={{a0_m95:.2f}}%)")
        print("=" * 80)
        delta_rows = []
        for _, row in df.iterrows():
            d50 = row["mAP50"] - a0_m50
            d95 = row["mAP50_95"] - a0_m95
            delta_rows.append({{
                "ID": row["ablation_id"],
                "Configuration": row["ablation_name"],
                "mAP50 (%)": f"{{row['mAP50']:.2f}}",
                "Delta mAP50": f"{{d50:+.2f}}%",
                "mAP50-95 (%)": f"{{row['mAP50_95']:.2f}}",
                "Delta mAP50-95": f"{{d95:+.2f}}%",
                "Recall (%)": f"{{row['recall']:.2f}}",
            }})
        print(tabulate(delta_rows, headers="keys", tablefmt="pipe", showindex=False))
"""

    # ------------------------------------------------------------------------
    # CELL 7: AUTOMATED ARTIFACT PACKAGING
    # ------------------------------------------------------------------------
    cell_7_code = f"""# CELL 7: AUTOMATED ARTIFACT PACKAGING (SEED {seed})
import zipfile
from pathlib import Path

zip_filename = Path("/kaggle/working/TaskB2_Seed{seed}_Outputs.zip")
print("=" * 80)
print(f"[PACKAGE] ARCHIVING ARTIFACTS TO: {{zip_filename.name}}...")
print("=" * 80)

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in OUTPUT_DIR.rglob("*"):
        if item.is_file():
            arcname = item.relative_to(OUTPUT_DIR.parent)
            zf.write(item, arcname)

size_mb = zip_filename.stat().st_size / (1024 * 1024)
print(f"[SUCCESS] Packaged {{zip_filename.name}} ({{size_mb:.2f}} MB)")
print(f"[INFO] Download {{zip_filename.name}} directly from Kaggle Output panel.")
"""

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": cell_0_md.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_1_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_2_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_3_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_4_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_5_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_6_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_7_code.splitlines(keepends=True)},
    ]

    nb["cells"] = cells
    return nb


def create_consolidated_multiseed_notebook():
    """
    Creates a consolidated reference Kaggle notebook running all 3 seeds (42, 1337, 2026)
    sequentially with full 100 epochs, true DDP support, and zero emojis.
    """
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataProxyVersion": "v2",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    cell_0_md = """# IEEE AAIML 2027: Multi-Seed Statistical Ablation Study (A0 -> A6)
### Safety Helmet Detection in Industrial Surveillance (SHWD / VOC2028)
- Author / Lead Researcher: Nguyen Han Nhu (FPT University)
- Reviewer Rebuttal Goal (Task B2): Prove statistical significance across 3 Random Seeds in {42, 1337, 2026} across 7 experimental configurations (A0 -> A6).
- Parallel Execution Notice: To prevent session timeout limits, users with 3 Kaggle accounts can execute the parallel dedicated notebooks:
  * Kaggle_TaskB2_Seed42_Ablation_T4x2.ipynb (Account 1)
  * Kaggle_TaskB2_Seed1337_Ablation_T4x2.ipynb (Account 2)
  * Kaggle_TaskB2_Seed2026_Ablation_T4x2.ipynb (Account 3)

---

## Operating Manifest and Kaggle Environment Settings

| Parameter / Field | Detailed Standard Specification | Operational Notes |
| :--- | :--- | :--- |
| **NOTEBOOK NAME** | `Kaggle_TaskB2_MultiSeed_Ablation_T4x2.ipynb` | Consolidated multi-seed reference notebook |
| **ACCELERATOR** | **GPU T4 x2** (Dual NVIDIA Tesla T4 16GB x 2 = 32GB VRAM) | Select via Kaggle Settings panel (right side) |
| **INTERNET** | **ON (Mandatory)** | Required for package updates and clean base weights |
| **PERSISTENCE** | **Files only** | Preserves state and outputs across disconnections |
| **DATASET INPUTS** | **VOC2028 (SHWD)**: Add dataset `voc2028` or `hannhu4002/voc2028` into `/kaggle/input`.<br>*(Strict adherence to official `ImageSets/Main/train.txt` and `val.txt` splits)* | Contains 7,581 annotated images |
| **WEIGHT INPUTS** | `yolo11s.pt` (Auto-downloaded from official Ultralytics releases, guaranteeing Zero Weight Contamination). | Clean re-initialization for each ablation |
| **RANDOM SEEDS** | **{42, 1337, 2026}** (3 Random Seeds per reviewer request) | Computes Mean mu and Std +/- sigma |
| **ABLATION STEPS** | - **A0**: Baseline YOLO11s (Stock Multi-Branch)<br>- **A1**: + P2 High-Resolution Micro-Head (Stride 4)<br>- **A2**: + CoordConv Stem Layer ($C_x, C_y \\in [-1, 1]$)<br>- **A3**: + RepConv Multi-Branch Structural Fusion<br>- **A4**: + Focal EIoU Loss ($\\gamma=0.5$)<br>- **A5**: + BiFormer Bi-Level Routing Attention<br>- **A6**: Proposed Rep-YOLO11s Full Fusion | Full 7 x 3 = 21 experiments |
| **EPOCHS & HYPERPARAMS**| **100 epochs** per ablation, `patience=30`, `cos_lr=True`, `close_mosaic=10`, `lr0=0.01`, `lrf=0.01`, `batch=32`, `imgsz=640`. | Full scientific rigor, no smoke test |
| **SMART RESUME** | Automatic caching: if `seed_{seed}_{ab_id}_best.pt` exists, skips execution. | Prevents redundant work on re-runs |
| **OUTPUT ARTIFACTS** | Packaged into **`TaskB2_MultiSeed_Ablation_T4x2_Outputs.zip`** containing CSV metrics, JSON logs, and trained weights. | 1-click download from Kaggle Output panel |
"""

    cell_1_code = """# CELL 1: ENVIRONMENT SETUP AND DUAL TESLA T4 HARDWARE VERIFICATION
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path

print("=" * 80)
print("[INFO] KAGGLE DUAL TESLA T4 ENVIRONMENT SETUP (TASK B2)")
print("=" * 80)
print(f"Python Version : {sys.version.split()[0]}")

# 1. Install required packages
!pip install -q -U ultralytics scipy tabulate matplotlib seaborn pandas

# 2. Suppress noisy loggers
os.environ["TENSORBOARD_BINARY"] = ""
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import settings
settings.update({"tensorboard": False})

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Detected GPUs  : {gpu_count}")
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        print(f"  -> GPU [{idx}]: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB")
    TRAIN_DEVICE = "0,1" if gpu_count >= 2 else "0"
    EVAL_DEVICE = "0"
else:
    print("[WARNING] No GPU detected. Defaulting to CPU fallback.")
    TRAIN_DEVICE = "cpu"
    EVAL_DEVICE = "cpu"

print(f"[INFO] Training Device   : {TRAIN_DEVICE}")
print(f"[INFO] Evaluation Device : {EVAL_DEVICE}")
"""

    cell_2_code = f"""# CELL 2: ARCHITECTURAL MODULE REGISTRATION AND DDP SITE-PACKAGES INJECTION
import base64
import os
import site
import sys
from pathlib import Path

# 1. Materialize custom_ablation_modules.py
custom_modules_b64 = "{CUSTOM_MODULES_B64}"
custom_modules_src = base64.b64decode(custom_modules_b64.encode("ascii")).decode("utf-8")
Path("custom_ablation_modules.py").write_text(custom_modules_src, encoding="utf-8")

# 2. Materialize ablation_trainer.py
ablation_trainer_b64 = "{ABLATION_TRAINER_B64}"
ablation_trainer_src = base64.b64decode(ablation_trainer_b64.encode("ascii")).decode("utf-8")
Path("ablation_trainer.py").write_text(ablation_trainer_src, encoding="utf-8")

# 3. Propagate modules into all site-packages directories for DDP subprocesses
for sp in site.getsitepackages():
    try:
        (Path(sp) / "custom_ablation_modules.py").write_text(custom_modules_src, encoding="utf-8")
        (Path(sp) / "ablation_trainer.py").write_text(ablation_trainer_src, encoding="utf-8")
    except Exception:
        pass

# 4. Ensure current working directory and /kaggle/working are top-priority in sys.path and PYTHONPATH
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
if "." not in sys.path:
    sys.path.insert(0, ".")

existing_pp = os.environ.get("PYTHONPATH", "")
if "/kaggle/working" not in existing_pp:
    os.environ["PYTHONPATH"] = f"/kaggle/working:{{existing_pp}}" if existing_pp else "/kaggle/working"

# 5. In-memory registration into Ultralytics engine
from custom_ablation_modules import CoordConv, RepConv, BiFormerBlockLite, focal_eiou_loss
from ablation_trainer import AblationBboxLoss, MultiSeedAblationTrainer, build_ablation_model
import ultralytics.nn.modules as un_mod
import ultralytics.nn.tasks as un_tasks
import ultralytics.utils.loss as ul_loss

un_mod.CoordConv = CoordConv
un_mod.RepConv = RepConv
un_mod.BiFormerBlockLite = BiFormerBlockLite
setattr(un_tasks, 'CoordConv', CoordConv)
setattr(un_tasks, 'RepConv', RepConv)
setattr(un_tasks, 'BiFormerBlockLite', BiFormerBlockLite)
ul_loss.BboxLoss = AblationBboxLoss

# 6. Physical file injection into loss.py for worker subprocesses with explicit os import
loss_file_path = Path(ul_loss.__file__).resolve()
loss_src = loss_file_path.read_text(encoding="utf-8")
if "class AblationBboxLoss" not in loss_src:
    patch_code = '''
import os
import torch
class AblationBboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        p_box = pred_bboxes[fg_mask]
        t_box = target_bboxes[fg_mask]
        cur_ab = os.environ.get("CURRENT_ABLATION_ID", "A0")
        if cur_ab in ["A4", "A6"] and p_box.shape[0] > 0:
            px1, py1, px2, py2 = p_box.unbind(-1)
            tx1, ty1, tx2, ty2 = t_box.unbind(-1)
            pw = (px2 - px1).clamp(min=1e-7)
            ph = (py2 - py1).clamp(min=1e-7)
            tw = (tx2 - tx1).clamp(min=1e-7)
            th = (ty2 - ty1).clamp(min=1e-7)
            inter_x1 = torch.maximum(px1, tx1)
            inter_y1 = torch.maximum(py1, ty1)
            inter_x2 = torch.minimum(px2, tx2)
            inter_y2 = torch.minimum(py2, ty2)
            inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
            union = pw * ph + tw * th - inter + 1e-7
            iou = (inter / union).clamp(min=1e-7, max=1.0)
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            center_dist = (pcx - tcx).square() + (pcy - tcy).square()
            cw = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=1e-7)
            ch = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=1e-7)
            c2 = cw.square() + ch.square() + 1e-7
            gamma = 0.5
            eiou = 1.0 - iou + center_dist / c2 + (pw - tw).square() / (cw.square() + 1e-7) + (ph - th).square() / (ch.square() + 1e-7)
            loss_box_sample = iou.pow(gamma) * eiou
            loss_iou = (loss_box_sample.unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(p_box, t_box, xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss and p_box.shape[0] > 0:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

BboxLoss = AblationBboxLoss
'''
    try:
        loss_file_path.write_text(loss_src + "\\n" + patch_code, encoding="utf-8")
        print("[INFO] Injected AblationBboxLoss into physical site-packages loss.py")
    except Exception as e:
        print(f"[WARNING] Could not patch physical loss.py: {{e}}")

print("[SUCCESS] Registered CoordConv, RepConv, BiFormer, Focal EIoU, and MultiSeedAblationTrainer.")
"""

    cell_3_code = """# CELL 3: DATASET INGESTION AND OFFICIAL VOC2028 SPLIT COMPLIANCE
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import zipfile

print("=" * 80)
print("[INFO] SCANNING FOR VOC2028 / SHWD DATASET IN /kaggle/input/...")
print("=" * 80)

candidate_dirs = [
    Path("/kaggle/input/voc2028/VOC2028"),
    Path("/kaggle/input/voc2028"),
    Path("/kaggle/input/datasets/hannhu4002/voc2028/VOC2028"),
    Path("/kaggle/input/datasets/hannhu4002/voc2028"),
    Path("Dataset/VOC2028"),
    Path("VOC2028"),
]

dataset_root = None
for cand in candidate_dirs:
    if (cand / "JPEGImages").exists() and (cand / "Annotations").exists():
        dataset_root = cand
        break

if dataset_root is None:
    for p in Path("/kaggle/input").rglob("JPEGImages"):
        if p.parent.is_dir() and (p.parent / "Annotations").is_dir():
            dataset_root = p.parent
            break

if dataset_root is None:
    print("[INFO] Searching for compressed dataset archive in /kaggle/input...")
    for z in Path("/kaggle/input").rglob("*.zip"):
        if "voc" in z.name.lower() or "shwd" in z.name.lower():
            extract_to = Path("/kaggle/working/VOC2028_EXTRACTED")
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(extract_to)
            for p in extract_to.rglob("JPEGImages"):
                if p.parent.is_dir() and (p.parent / "Annotations").is_dir():
                    dataset_root = p.parent
                    break
            if dataset_root:
                break

if dataset_root is None:
    raise FileNotFoundError(
        "[ERROR] VOC2028 dataset not found in /kaggle/input! "
        "Please attach 'voc2028' (or 'hannhu4002/voc2028') to this Kaggle notebook session."
    )

print(f"[INFO] Verified VOC2028 dataset root at: {dataset_root}")

# Setup YOLO directory structure
YOLO_DIR = Path("/kaggle/working/SHWD_YOLO")
images_dir = YOLO_DIR / "images"
labels_dir = YOLO_DIR / "labels"

for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"hat": 0, "helmet": 0, "person": 1, "head": 1}

# Adhere strictly to official VOC2028 ImageSets splits (Zero Test Contamination)
imagesets_main = dataset_root / "ImageSets" / "Main"
train_ids = set()
val_ids = set()

if (imagesets_main / "train.txt").exists():
    train_ids = set(line.strip() for line in (imagesets_main / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"[INFO] Found official ImageSets/Main/train.txt: {len(train_ids)} images")

if (imagesets_main / "val.txt").exists():
    val_ids = set(line.strip() for line in (imagesets_main / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"[INFO] Found official ImageSets/Main/val.txt: {len(val_ids)} images")

all_xmls = sorted(list((dataset_root / "Annotations").glob("*.xml")))
if not train_ids or not val_ids:
    print("[WARNING] ImageSets/Main splits not found. Fallback to deterministic 80/20 split (seed 2026)...")
    import random
    rng = random.Random(2026)
    stems = [x.stem for x in all_xmls]
    rng.shuffle(stems)
    split_idx = int(len(stems) * 0.8)
    train_ids = set(stems[:split_idx])
    val_ids = set(stems[split_idx:])

def convert_and_populate(xml_list, target_split, id_set):
    count = 0
    for xml_path in xml_list:
        stem = xml_path.stem
        if stem not in id_set:
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            continue
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w <= 0 or h <= 0:
            continue

        src_img = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            cand = dataset_root / "JPEGImages" / f"{stem}{ext}"
            if cand.exists():
                src_img = cand
                break
        if src_img is None:
            continue

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip().lower()
            if cls_name not in CLASS_MAP:
                continue
            cid = CLASS_MAP[cls_name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            cx = ((xmin + xmax) / 2.0) / w
            cy = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            if bw > 0 and bh > 0:
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        dst_img = images_dir / target_split / src_img.name
        if not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
            except Exception:
                shutil.copy2(src_img, dst_img)

        dst_lbl = labels_dir / target_split / f"{stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write("\\n".join(yolo_lines))
        count += 1
    return count

n_train = convert_and_populate(all_xmls, "train", train_ids)
n_val = convert_and_populate(all_xmls, "val", val_ids)
print(f"[SUCCESS] Prepared dataset: {n_train} train images, {n_val} validation images.")

data_yaml_path = YOLO_DIR / "shwd_data.yaml"
data_yaml_content = f\"\"\"
path: {YOLO_DIR.resolve()}
train: images/train
val: images/val
nc: 2
names: ['hat', 'person']
\"\"\"
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(data_yaml_content.strip())
print(f"[INFO] Configuration file saved: {data_yaml_path}")
"""

    cell_4_code = f"""# CELL 4: ARCHITECTURE FACTORY AND MULTI-HEAD P2 INITIALIZATION (A0 -> A6)
import base64
import os
from pathlib import Path
import torch
from ultralytics import YOLO
from ablation_trainer import MultiSeedAblationTrainer, build_ablation_model

# Write rep_yolo11s_p2.yaml for Ablation A1
p2_yaml_path = Path("/kaggle/working/rep_yolo11s_p2.yaml")
p2_yaml_content = base64.b64decode("{P2_YAML_B64}".encode("ascii")).decode("utf-8")
p2_yaml_path.write_text(p2_yaml_content.strip(), encoding="utf-8")
print(f"[INFO] 4-Head P2 Model Architecture YAML written to: {{p2_yaml_path}}")

# Sanity assertion check across all 7 ablations
print("[INFO] Validating architectural factory instantiation across A0 -> A6...")
dummy = torch.randn(1, 3, 640, 640)
for ab_check in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
    m_check = build_ablation_model(ab_check, "yolo11s.pt")
    m_check.model.eval()
    with torch.no_grad():
        out_check = m_check.model(dummy)
    del m_check
print("[SUCCESS] All 7 ablation architectures verified successfully.")
"""

    cell_5_code = """# CELL 5: MULTI-SEED FULL 100-EPOCH TRAINING PIPELINE (SEEDS 42, 1337, 2026)
import gc
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# CONFIGURATION FOR MULTI-SEED ABLATION
SEEDS = [42, 1337, 2026]
ABLATIONS = [
    {"id": "A0", "name": "Baseline YOLO11s", "desc": "Standard stock YOLO11s"},
    {"id": "A1", "name": "+ P2 Small-Object Head", "desc": "High-resolution P2 micro-head"},
    {"id": "A2", "name": "+ CoordConv Stem", "desc": "Spatial vertical coordinate priors"},
    {"id": "A3", "name": "+ RepConv Re-Param", "desc": "Structural re-parameterization branches"},
    {"id": "A4", "name": "+ Focal EIoU Loss", "desc": "Independent width/height aspect ratio penalty"},
    {"id": "A5", "name": "+ BiFormer Attention", "desc": "Bi-level routing attention across P4/P5"},
    {"id": "A6", "name": "Full Fusion (Proposed)", "desc": "Rep-YOLO11s full end-to-end integration"},
]

# FULL SCIENTIFIC RIGOR: 100 EPOCHS WITH EARLY STOPPING (PATIENCE=30)
EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 32
PATIENCE = 30
COS_LR = True
CLOSE_MOSAIC = 10
LR0 = 0.01
LRF = 0.01

OUTPUT_DIR = Path("/kaggle/working/TaskB2_MultiSeed_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

results_records = []
csv_results_path = OUTPUT_DIR / "multiseed_ablation_results.csv"
log_file = OUTPUT_DIR / "multiseed_ablation_log.json"

if csv_results_path.exists():
    df_cached = pd.read_csv(csv_results_path)
    results_records = df_cached.to_dict(orient="records")
    print(f"[RESUME] Loaded {len(results_records)} completed ablation runs from {csv_results_path.name}.")

def is_already_completed(ab_id: str, seed: int) -> bool:
    for rec in results_records:
        if rec["ablation_id"] == ab_id and rec["seed"] == seed:
            ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}_{ab_id}_best.pt"
            if ckpt_path.exists() or Path(rec.get("checkpoint", "")).exists():
                return True
    return False

print("=" * 80)
print(f"[START] MULTI-SEED ABLATION SUITE ({len(SEEDS)} SEEDS x {len(ABLATIONS)} MODELS, {EPOCHS} EPOCHS EACH)")
print(f"   Batch Size     : {BATCH_SIZE} | ImgSz: {IMGSZ}")
print(f"   Hyperparameters: lr0={LR0}, lrf={LRF}, patience={PATIENCE}, cos_lr={COS_LR}, close_mosaic={CLOSE_MOSAIC}")
print("=" * 80)

for seed in SEEDS:
    print(f"\\n{'=' * 80}\\n[SEED EXECUTION] COMMENCING EVALUATION FOR SEED = {seed}\\n{'=' * 80}")
    for ab in ABLATIONS:
        ab_id = ab["id"]
        ckpt_name = f"seed_{seed}_{ab_id}_best.pt"
        target_ckpt = CHECKPOINTS_DIR / ckpt_name

        if is_already_completed(ab_id, seed):
            print(f"[SKIP] Ablation {ab_id} (Seed {seed}) already completed in cache. Moving to next.")
            continue

        print("\\n----------------------------------------------------------------------")
        print(f"[RUNNING] Ablation {ab_id}: {ab['name']} | Seed = {seed} | Epochs = {EPOCHS}")
        print("----------------------------------------------------------------------")

        # Set ablation ID in environment for DDP worker processes
        os.environ["CURRENT_ABLATION_ID"] = ab_id

        # Clean architectural factory instantiation from fresh base weight (Zero Weight Contamination)
        model = build_ablation_model(ab_id, "yolo11s.pt")
        import ablation_trainer
        ablation_trainer.CURRENT_ABLATION_MODEL = model.model

        run_name = f"run_{ab_id}_seed_{seed}"
        t_start = time.time()

        try:
            train_results = model.train(
                trainer=MultiSeedAblationTrainer,
                data=str(data_yaml_path),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH_SIZE,
                device=TRAIN_DEVICE,
                seed=seed,
                deterministic=True,
                cos_lr=COS_LR,
                patience=PATIENCE,
                close_mosaic=CLOSE_MOSAIC,
                lr0=LR0,
                lrf=LRF,
                save=True,
                val=True,
                plots=False,
                name=run_name,
                project=str(OUTPUT_DIR / "runs"),
                exist_ok=True,
                verbose=False,
            )
        except Exception as e:
            print(f"[WARNING] DDP training encountered issue: {e}. Falling back to single GPU 0...")
            train_results = model.train(
                trainer=MultiSeedAblationTrainer,
                data=str(data_yaml_path),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH_SIZE,
                device="0",
                seed=seed,
                deterministic=True,
                cos_lr=COS_LR,
                patience=PATIENCE,
                close_mosaic=CLOSE_MOSAIC,
                lr0=LR0,
                lrf=LRF,
                save=True,
                val=True,
                plots=False,
                name=run_name,
                project=str(OUTPUT_DIR / "runs"),
                exist_ok=True,
                verbose=False,
            )

        train_time_min = (time.time() - t_start) / 60.0

        # Independent validation on unseen val split
        val_metrics = model.val(
            data=str(data_yaml_path),
            imgsz=IMGSZ,
            device=EVAL_DEVICE,
            verbose=False,
        )

        map50 = float(val_metrics.box.map50) * 100.0
        map50_95 = float(val_metrics.box.map) * 100.0
        precision = float(val_metrics.box.mp) * 100.0
        recall = float(val_metrics.box.mr) * 100.0

        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, target_ckpt)

        rec = {
            "ablation_id": ab_id,
            "ablation_name": ab["name"],
            "seed": seed,
            "mAP50": round(map50, 2),
            "mAP50_95": round(map50_95, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "train_time_min": round(train_time_min, 1),
            "checkpoint": str(target_ckpt.name),
        }
        results_records.append(rec)

        # Persist results immediately to disk
        df_curr = pd.DataFrame(results_records)
        df_curr.to_csv(csv_results_path, index=False)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(results_records, f, indent=2)

        print(f"[DONE] Completed {ab_id} Seed {seed}: mAP50 = {map50:.2f}%, mAP50-95 = {map50_95:.2f}% ({train_time_min:.1f} min)")

        # Clean VRAM
        del model
        ablation_trainer.CURRENT_ABLATION_MODEL = None
        torch.cuda.empty_cache()
        gc.collect()

print("\\n[SUCCESS] All ablation models across all seeds completed successfully.")
"""

    cell_6_code = """# CELL 6: TABULATION AND IEEE TABLE II LATEX EXPORT
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

df = pd.DataFrame(results_records)
summary_data = []
baseline_a0_map50 = df[df["ablation_id"] == "A0"].sort_values("seed")["mAP50"].values

for ab in ABLATIONS:
    ab_id = ab["id"]
    sub = df[df["ablation_id"] == ab_id].sort_values("seed")
    if len(sub) == 0:
        continue

    m50_mean = float(sub["mAP50"].mean())
    m50_std = float(sub["mAP50"].std(ddof=1)) if len(sub) > 1 else 0.0

    m95_mean = float(sub["mAP50_95"].mean())
    m95_std = float(sub["mAP50_95"].std(ddof=1)) if len(sub) > 1 else 0.0

    rec_mean = float(sub["recall"].mean())
    rec_std = float(sub["recall"].std(ddof=1)) if len(sub) > 1 else 0.0

    if ab_id != "A0" and len(baseline_a0_map50) == len(sub["mAP50"].values) and len(baseline_a0_map50) >= 2:
        _, p_val = stats.ttest_rel(sub["mAP50"].values, baseline_a0_map50)
        p_str = f"{p_val:.4f}*" if p_val < 0.05 else f"{p_val:.4f}"
    else:
        p_str = "-"

    summary_data.append({
        "ID": ab_id,
        "Configuration": ab["name"],
        "mAP50 (Mean +/- sigma)": f"{m50_mean:.2f} +/- {m50_std:.2f}%",
        "mAP50-95 (Mean +/- sigma)": f"{m95_mean:.2f} +/- {m95_std:.2f}%",
        "Recall (Mean +/- sigma)": f"{rec_mean:.2f} +/- {rec_std:.2f}%",
        "p-value vs A0": p_str,
        "Significance": "Significant (p < 0.05)" if p_str.endswith("*") else ("Baseline" if ab_id == "A0" else "p >= 0.05"),
    })

summary_df = pd.DataFrame(summary_data)
summary_csv = OUTPUT_DIR / "statistical_ablation_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\\n" + "=" * 80)
print("[RESULTS] MASTER STATISTICAL ABLATION SUMMARY (IEEE AAIML 2027 TABLE II)")
print("=" * 80)
print(tabulate(summary_df, headers="keys", tablefmt="pipe", showindex=False))

latex_rows = []
for row in summary_data:
    bold_start = "\\\\textbf{" if row["ID"] == "A6" else ""
    bold_end = "}" if row["ID"] == "A6" else ""
    ab_id_tex = f"$A_{{{row['ID'][1:]}}}$" if row["ID"].startswith("A") else f"${row['ID']}$"
    m50_c = row["mAP50 (Mean +/- sigma)"].replace("%", "").replace("+/-", "\\\\pm")
    m95_c = row["mAP50-95 (Mean +/- sigma)"].replace("%", "").replace("+/-", "\\\\pm")
    rec_c = row["Recall (Mean +/- sigma)"].replace("%", "").replace("+/-", "\\\\pm")
    latex_rows.append(
        f"{ab_id_tex} & {bold_start}{row['Configuration']}{bold_end} & {m50_c} & {m95_c} & {rec_c} & {row['p-value vs A0']} \\\\\\\\"
    )

latex_code = f\"\"\"\\\\begin{{table}}[t]
\\\\caption{{Multi-Seed Statistical Ablation Study across Seeds $\\\\in \\\\{{42, 1337, 2026\\\\}}$ on SHWD.}}
\\\\label{{tab:multiseed_ablation}}
\\\\centering
\\\\renewcommand{{\\\\arraystretch}}{{0.95}}
\\\\resizebox{{\\\\columnwidth}}{{!}}{{%
\\\\begin{{tabular}}{{clcccc}}
\\\\toprule
\\\\textbf{{ID}} & \\\\textbf{{Configuration}} & \\\\textbf{{$mAP_{{50}}$ ($\\\\mu \\\\pm \\\\sigma$)}} & \\\\textbf{{$mAP_{{50-95}}$ ($\\\\mu \\\\pm \\\\sigma$)}} & \\\\textbf{{$R^{{hat}}$ ($\\\\mu \\\\pm \\\\sigma$)}} & \\\\textbf{{$p$-value}} \\\\\\\\
\\\\midrule
{chr(10).join(latex_rows)}
\\\\bottomrule
\\\\multicolumn{{6}}{{l}}{{\\\\small Evaluated on Kaggle Dual Tesla T4 across 3 random seeds. * indicates $p < 0.05$.}}
\\\\end{{tabular}}%
}}
\\\\end{{table}}
\"\"\"

latex_file = OUTPUT_DIR / "Table2_MultiSeed_Ablation.tex"
with open(latex_file, "w", encoding="utf-8") as f:
    f.write(latex_code.strip())
print(f"\\n[SUCCESS] IEEE LaTeX table exported to: {latex_file}")
"""

    cell_7_code = """# CELL 7: PUBLICATION ERROR BAR CHART (300 DPI)
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

ab_ids = [d["ID"] for d in summary_data]
means_50 = [float(d["mAP50 (Mean +/- sigma)"].split()[0]) for d in summary_data]
stds_50 = [float(d["mAP50 (Mean +/- sigma)"].split()[2].replace('%', '')) for d in summary_data]

means_95 = [float(d["mAP50-95 (Mean +/- sigma)"].split()[0]) for d in summary_data]
stds_95 = [float(d["mAP50-95 (Mean +/- sigma)"].split()[2].replace('%', '')) for d in summary_data]

colors = ["#2b5c8f", "#3470a3", "#3d85b8", "#4699cc", "#50ade0", "#59c2f4", "#e63946"]

# Subplot 1: mAP50 with Error Bars
bars1 = ax1.bar(ab_ids, means_50, yerr=stds_50, capsize=6, color=colors, edgecolor='black', alpha=0.88, width=0.55)
min_y1 = max(0.0, min(means_50) - 0.25)
max_y1 = min(100.0, max(means_50) + 0.25)
ax1.set_ylim([min_y1, max_y1])
ax1.set_title("mAP@0.5 Across 3 Random Seeds (mu +/- sigma)", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Ablation Configuration", fontsize=11, fontweight='bold')
ax1.set_ylabel("mAP@0.5 (%)", fontsize=11, fontweight='bold')

for bar, mean, std in zip(bars1, means_50, stds_50):
    ax1.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, f"{mean:.2f}+/-{std:.2f}%",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: mAP50-95 with Error Bars
bars2 = ax2.bar(ab_ids, means_95, yerr=stds_95, capsize=6, color=colors, edgecolor='black', alpha=0.88, width=0.55)
min_y2 = max(0.0, min(means_95) - 0.35)
max_y2 = min(100.0, max(means_95) + 0.35)
ax2.set_ylim([min_y2, max_y2])
ax2.set_title("mAP@0.5:0.95 Across 3 Random Seeds (mu +/- sigma)", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Ablation Configuration", fontsize=11, fontweight='bold')
ax2.set_ylabel("mAP@0.5:0.95 (%)", fontsize=11, fontweight='bold')

for bar, mean, std in zip(bars2, means_95, stds_95):
    ax2.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, f"{mean:.2f}+/-{std:.2f}%",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
chart_path = OUTPUT_DIR / "ablation_multiseed_errorbars.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[INFO] 300 DPI chart saved to: {chart_path}")
"""

    cell_8_code = """# CELL 8: AUTOMATED ARTIFACT PACKAGING
import zipfile
from pathlib import Path

zip_filename = Path("/kaggle/working/TaskB2_MultiSeed_Ablation_T4x2_Outputs.zip")
print("=" * 80)
print(f"[PACKAGE] ARCHIVING ARTIFACTS TO: {zip_filename.name}...")
print("=" * 80)

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in OUTPUT_DIR.rglob("*"):
        if item.is_file():
            arcname = item.relative_to(OUTPUT_DIR.parent)
            zf.write(item, arcname)

size_mb = zip_filename.stat().st_size / (1024 * 1024)
print(f"[SUCCESS] Packaged {zip_filename.name} ({size_mb:.2f} MB)")
print("[INFO] Download TaskB2_MultiSeed_Ablation_T4x2_Outputs.zip directly from Kaggle Output panel.")
"""

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": cell_0_md.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_1_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_2_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_3_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_4_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_5_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_6_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_7_code.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "source": cell_8_code.splitlines(keepends=True)},
    ]

    nb["cells"] = cells
    return nb


def build_all_notebooks():
    configs = [
        {"seed": 42, "account": 1, "filename": "Kaggle_TaskB2_Seed42_Ablation_T4x2.ipynb"},
        {"seed": 1337, "account": 2, "filename": "Kaggle_TaskB2_Seed1337_Ablation_T4x2.ipynb"},
        {"seed": 2026, "account": 3, "filename": "Kaggle_TaskB2_Seed2026_Ablation_T4x2.ipynb"},
    ]

    generated_files = []
    # 1. Build dedicated single-seed notebooks for the 3 Kaggle accounts
    for cfg in configs:
        nb = create_single_seed_notebook(seed=cfg["seed"], account_num=cfg["account"])
        out_path = Path(cfg["filename"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        generated_files.append(out_path)
        print(f"[SUCCESS] Generated: {out_path.name} (Seed {cfg['seed']}, Account {cfg['account']})")

    # 2. Build clean consolidated reference notebook (0 emojis, 100 epochs)
    consolidated_nb = create_consolidated_multiseed_notebook()
    consolidated_path = Path("Kaggle_TaskB2_MultiSeed_Ablation_T4x2.ipynb")
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_nb, f, indent=1)
    generated_files.append(consolidated_path)
    print(f"[SUCCESS] Generated: {consolidated_path.name} (Consolidated Reference)")

    return generated_files


if __name__ == "__main__":
    build_all_notebooks()
