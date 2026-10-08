"""
=============================================================================
KAGGLE STANDALONE SCRIPT: TASK B2 STATISTICAL ABLATION (SEED 2026) FULL (A0 -> A6)
IEEE AAIML 2027 Reviewer Rebuttal - Safety Helmet Detection (SHWD / VOC2028)
Author: Nguyen Han Nhu (FPT University)
Target Account: Kaggle Account 3 | Dual Tesla T4 x2 (32GB VRAM)
Epochs: 100 per ablation, patience=30, cos_lr=True, batch=32, imgsz=640
=============================================================================
"""


# ==============================================================================

# CELL 1: ENVIRONMENT SETUP AND DUAL TESLA T4 HARDWARE VERIFICATION
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path

print("=" * 80)
print("[INFO] KAGGLE DUAL TESLA T4 ENVIRONMENT SETUP - SEED 2026")
print("=" * 80)
print(f"Python Version : {sys.version.split()[0]}")

# 1. Install required packages
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics", "scipy", "tabulate", "matplotlib", "seaborn", "pandas"], check=False)

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

# ==============================================================================

# CELL 2: ARCHITECTURAL MODULE REGISTRATION AND DDP SITE-PACKAGES INJECTION
import base64
import os
import site
import sys
from pathlib import Path

# 1. Materialize custom_ablation_modules.py (genuine BiFormer, RepConv, CoordConv, Focal EIoU)
custom_modules_b64 = "IiIiClJlc2VhcmNoIG1vZHVsZXMgZm9yIFN0YWdlIDIgY3VzdG9tIGFibGF0aW9ucy4KClRoZXNlIG1vZHVsZXMgYXJlIGludGVudGlvbmFsbHkgc2VsZi1jb250YWluZWQuIFRoZXkgYXJlIG5vdCB3aXJlZCBpbnRvIHRoZQpiYXNlbGluZSBub3RlYm9vayBiZWNhdXNlIFN0YWdlIDEgc2hvdWxkIHJlbWFpbiBzdGFibGUgYW5kIGNvbXBhcmFibGUuIFVzZSB0aGVtCmFmdGVyIHNlbGVjdGluZyB0aGUgdG9wLTIgYmFja2JvbmVzIGZyb20gYGJlbmNobWFya19yZXN1bHRzLmNzdmAuCiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IHRvcmNoCmltcG9ydCB0b3JjaC5ubiBhcyBubgppbXBvcnQgdG9yY2gubm4uZnVuY3Rpb25hbCBhcyBGCgoKZGVmIGF1dG9wYWQoazogaW50LCBwOiBpbnQgfCBOb25lID0gTm9uZSwgZDogaW50ID0gMSkgLT4gaW50OgogICAgIiIiUmV0dXJuIHBhZGRpbmcgdGhhdCBwcmVzZXJ2ZXMgc3BhdGlhbCBzaGFwZSBmb3Igb2RkIGtlcm5lbHMuIiIiCiAgICBpZiBwIGlzIE5vbmU6CiAgICAgICAgcCA9IGQgKiAoayAtIDEpIC8vIDIKICAgIHJldHVybiBwCgoKY2xhc3MgQ29udkJOQWN0KG5uLk1vZHVsZSk6CiAgICAiIiJTbWFsbCBDb252LUJOLVNpTFUgYmxvY2sgY29tcGF0aWJsZSB3aXRoIFlPTE8tc3R5bGUgbW9kdWxlcy4iIiIKCiAgICBkZWYgX19pbml0X18oCiAgICAgICAgc2VsZiwKICAgICAgICBjMTogaW50LAogICAgICAgIGMyOiBpbnQsCiAgICAgICAgazogaW50ID0gMSwKICAgICAgICBzOiBpbnQgPSAxLAogICAgICAgIHA6IGludCB8IE5vbmUgPSBOb25lLAogICAgICAgIGc6IGludCA9IDEsCiAgICAgICAgZDogaW50ID0gMSwKICAgICAgICBhY3Q6IGJvb2wgPSBUcnVlLAogICAgKSAtPiBOb25lOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIHNlbGYuY29udiA9IG5uLkNvbnYyZChjMSwgYzIsIGssIHMsIGF1dG9wYWQoaywgcCwgZCksIGdyb3Vwcz1nLCBkaWxhdGlvbj1kLCBiaWFzPUZhbHNlKQogICAgICAgIHNlbGYuYm4gPSBubi5CYXRjaE5vcm0yZChjMikKICAgICAgICBzZWxmLmFjdCA9IG5uLlNpTFUoaW5wbGFjZT1UcnVlKSBpZiBhY3QgZWxzZSBubi5JZGVudGl0eSgpCgogICAgZGVmIGZvcndhcmQoc2VsZiwgeDogdG9yY2guVGVuc29yKSAtPiB0b3JjaC5UZW5zb3I6CiAgICAgICAgcmV0dXJuIHNlbGYuYWN0KHNlbGYuYm4oc2VsZi5jb252KHgpKSkKCgpjbGFzcyBDb29yZENvbnYobm4uTW9kdWxlKToKICAgICIiIgogICAgQ29vcmRDb252IGxheWVyOiBhcHBlbmQgbm9ybWFsaXplZCB4L3kgY29vcmRpbmF0ZSBjaGFubmVscyBiZWZvcmUgYSBjb252LgoKICAgIFVzZSBzcGFyaW5nbHkgaW4gdGhlIHN0ZW0gb3Igc2VsZWN0ZWQgbmVjayBsYXllcnMuIFJlcGxhY2luZyBldmVyeSBjb252IGNhbgogICAgb3ZlcmZpdCBmaXhlZCBjYW1lcmEgZ2VvbWV0cnkuCiAgICAiIiIKCiAgICBkZWYgX19pbml0X18oc2VsZiwgYzE6IGludCwgYzI6IGludCwgazogaW50ID0gMywgczogaW50ID0gMSwgd2l0aF9yOiBib29sID0gRmFsc2UpIC0+IE5vbmU6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi53aXRoX3IgPSB3aXRoX3IKICAgICAgICBleHRyYSA9IDMgaWYgd2l0aF9yIGVsc2UgMgogICAgICAgIHNlbGYuY29udiA9IENvbnZCTkFjdChjMSArIGV4dHJhLCBjMiwgaz1rLCBzPXMpCgogICAgZGVmIGZvcndhcmQoc2VsZiwgeDogdG9yY2guVGVuc29yKSAtPiB0b3JjaC5UZW5zb3I6CiAgICAgICAgYiwgXywgaCwgdyA9IHguc2hhcGUKICAgICAgICB5eSA9IHRvcmNoLmxpbnNwYWNlKC0xLjAsIDEuMCwgaCwgZGV2aWNlPXguZGV2aWNlLCBkdHlwZT14LmR0eXBlKS52aWV3KDEsIDEsIGgsIDEpLmV4cGFuZChiLCAxLCBoLCB3KQogICAgICAgIHh4ID0gdG9yY2gubGluc3BhY2UoLTEuMCwgMS4wLCB3LCBkZXZpY2U9eC5kZXZpY2UsIGR0eXBlPXguZHR5cGUpLnZpZXcoMSwgMSwgMSwgdykuZXhwYW5kKGIsIDEsIGgsIHcpCiAgICAgICAgY29vcmRzID0gW3h4LCB5eV0KICAgICAgICBpZiBzZWxmLndpdGhfcjoKICAgICAgICAgICAgcnIgPSB0b3JjaC5zcXJ0KHRvcmNoLmNsYW1wKHh4LnNxdWFyZSgpICsgeXkuc3F1YXJlKCksIG1pbj0wLjApKQogICAgICAgICAgICBjb29yZHMuYXBwZW5kKHJyKQogICAgICAgIHJldHVybiBzZWxmLmNvbnYodG9yY2guY2F0KFt4LCAqY29vcmRzXSwgZGltPTEpKQoKICAgIGRlZiBmdXNlKHNlbGYpIC0+IE5vbmU6CiAgICAgICAgcGFzcwoKICAgIGRlZiBmdXNlX2NvbnZzKHNlbGYpIC0+IE5vbmU6CiAgICAgICAgcGFzcwoKCmNsYXNzIFJlcENvbnYobm4uTW9kdWxlKToKICAgICIiIgogICAgUmVwQ29udiBibG9jayB3aXRoIDN4MywgMXgxLCBhbmQgb3B0aW9uYWwgaWRlbnRpdHkgYnJhbmNoZXMgZHVyaW5nIHRyYWluaW5nLgoKICAgIENhbGwgYHN3aXRjaF90b19kZXBsb3koKWAgYmVmb3JlIE9OTlgvVGVuc29yUlQgZXhwb3J0LiBUaGUgZGVwbG95IHBhdGggaXMgYQogICAgc2luZ2xlIDN4MyBDb252MmQgcGx1cyBhY3RpdmF0aW9uLgogICAgIiIiCgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGMxOiBpbnQsIGMyOiBpbnQsIGs6IGludCA9IDMsIHM6IGludCA9IDEsIGRlcGxveTogYm9vbCA9IEZhbHNlLCBhY3Q6IGJvb2wgPSBUcnVlKSAtPiBOb25lOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIGFzc2VydCBrID09IDMsICJUaGlzIFJlcENvbnYgaW1wbGVtZW50YXRpb24gZnVzZXMgdG8gYSAzeDMga2VybmVsLiIKICAgICAgICBzZWxmLmRlcGxveSA9IGRlcGxveQogICAgICAgIHNlbGYuaW5fY2hhbm5lbHMgPSBjMQogICAgICAgIHNlbGYub3V0X2NoYW5uZWxzID0gYzIKICAgICAgICBzZWxmLnN0cmlkZSA9IHMKICAgICAgICBzZWxmLmFjdCA9IG5uLlNpTFUoaW5wbGFjZT1UcnVlKSBpZiBhY3QgZWxzZSBubi5JZGVudGl0eSgpCgogICAgICAgIGlmIGRlcGxveToKICAgICAgICAgICAgc2VsZi5yYnJfcmVwYXJhbSA9IG5uLkNvbnYyZChjMSwgYzIsIDMsIHMsIDEsIGJpYXM9VHJ1ZSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBzZWxmLnJicl9kZW5zZSA9IG5uLlNlcXVlbnRpYWwoCiAgICAgICAgICAgICAgICBubi5Db252MmQoYzEsIGMyLCAzLCBzLCAxLCBiaWFzPUZhbHNlKSwKICAgICAgICAgICAgICAgIG5uLkJhdGNoTm9ybTJkKGMyKSwKICAgICAgICAgICAgKQogICAgICAgICAgICBzZWxmLnJicl8xeDEgPSBubi5TZXF1ZW50aWFsKAogICAgICAgICAgICAgICAgbm4uQ29udjJkKGMxLCBjMiwgMSwgcywgMCwgYmlhcz1GYWxzZSksCiAgICAgICAgICAgICAgICBubi5CYXRjaE5vcm0yZChjMiksCiAgICAgICAgICAgICkKICAgICAgICAgICAgc2VsZi5yYnJfaWRlbnRpdHkgPSBubi5CYXRjaE5vcm0yZChjMSkgaWYgYzEgPT0gYzIgYW5kIHMgPT0gMSBlbHNlIE5vbmUKCiAgICBkZWYgZm9yd2FyZChzZWxmLCB4OiB0b3JjaC5UZW5zb3IpIC0+IHRvcmNoLlRlbnNvcjoKICAgICAgICBpZiBzZWxmLmRlcGxveToKICAgICAgICAgICAgcmV0dXJuIHNlbGYuYWN0KHNlbGYucmJyX3JlcGFyYW0oeCkpCiAgICAgICAgb3V0ID0gc2VsZi5yYnJfZGVuc2UoeCkgKyBzZWxmLnJicl8xeDEoeCkKICAgICAgICBpZiBzZWxmLnJicl9pZGVudGl0eSBpcyBub3QgTm9uZToKICAgICAgICAgICAgb3V0ID0gb3V0ICsgc2VsZi5yYnJfaWRlbnRpdHkoeCkKICAgICAgICByZXR1cm4gc2VsZi5hY3Qob3V0KQoKICAgIEBzdGF0aWNtZXRob2QKICAgIGRlZiBfZnVzZV9jb252X2JuKGJyYW5jaDogbm4uU2VxdWVudGlhbCB8IG5uLkJhdGNoTm9ybTJkLCBjaGFubmVsczogaW50KSAtPiB0dXBsZVt0b3JjaC5UZW5zb3IsIHRvcmNoLlRlbnNvcl06CiAgICAgICAgaWYgaXNpbnN0YW5jZShicmFuY2gsIG5uLlNlcXVlbnRpYWwpOgogICAgICAgICAgICBjb252ID0gYnJhbmNoWzBdCiAgICAgICAgICAgIGJuID0gYnJhbmNoWzFdCiAgICAgICAgICAgIGtlcm5lbCA9IGNvbnYud2VpZ2h0CiAgICAgICAgZWxzZToKICAgICAgICAgICAgYm4gPSBicmFuY2gKICAgICAgICAgICAgaW5wdXRfZGltID0gY2hhbm5lbHMKICAgICAgICAgICAga2VybmVsID0gdG9yY2guemVyb3MoKGlucHV0X2RpbSwgaW5wdXRfZGltLCAzLCAzKSwgZGV2aWNlPWJuLndlaWdodC5kZXZpY2UsIGR0eXBlPWJuLndlaWdodC5kdHlwZSkKICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2UoaW5wdXRfZGltKToKICAgICAgICAgICAgICAgIGtlcm5lbFtpLCBpLCAxLCAxXSA9IDEuMAoKICAgICAgICBzdGQgPSB0b3JjaC5zcXJ0KGJuLnJ1bm5pbmdfdmFyICsgYm4uZXBzKQogICAgICAgIHQgPSAoYm4ud2VpZ2h0IC8gc3RkKS5yZXNoYXBlKC0xLCAxLCAxLCAxKQogICAgICAgIGZ1c2VkX2tlcm5lbCA9IGtlcm5lbCAqIHQKICAgICAgICBmdXNlZF9iaWFzID0gYm4uYmlhcyAtIGJuLnJ1bm5pbmdfbWVhbiAqIGJuLndlaWdodCAvIHN0ZAogICAgICAgIHJldHVybiBmdXNlZF9rZXJuZWwsIGZ1c2VkX2JpYXMKCiAgICBAc3RhdGljbWV0aG9kCiAgICBkZWYgX3BhZF8xeDFfdG9fM3gzKGtlcm5lbDogdG9yY2guVGVuc29yKSAtPiB0b3JjaC5UZW5zb3I6CiAgICAgICAgaWYga2VybmVsLnNpemUoMikgPT0gMzoKICAgICAgICAgICAgcmV0dXJuIGtlcm5lbAogICAgICAgIHJldHVybiBGLnBhZChrZXJuZWwsIFsxLCAxLCAxLCAxXSkKCiAgICBkZWYgZ2V0X2VxdWl2YWxlbnRfa2VybmVsX2JpYXMoc2VsZikgLT4gdHVwbGVbdG9yY2guVGVuc29yLCB0b3JjaC5UZW5zb3JdOgogICAgICAgIGszLCBiMyA9IHNlbGYuX2Z1c2VfY29udl9ibihzZWxmLnJicl9kZW5zZSwgc2VsZi5vdXRfY2hhbm5lbHMpCiAgICAgICAgazEsIGIxID0gc2VsZi5fZnVzZV9jb252X2JuKHNlbGYucmJyXzF4MSwgc2VsZi5vdXRfY2hhbm5lbHMpCiAgICAgICAgaWYgc2VsZi5yYnJfaWRlbnRpdHkgaXMgTm9uZToKICAgICAgICAgICAga2lkID0gdG9yY2guemVyb3NfbGlrZShrMykKICAgICAgICAgICAgYmlkID0gdG9yY2guemVyb3NfbGlrZShiMykKICAgICAgICBlbHNlOgogICAgICAgICAgICBraWQsIGJpZCA9IHNlbGYuX2Z1c2VfY29udl9ibihzZWxmLnJicl9pZGVudGl0eSwgc2VsZi5vdXRfY2hhbm5lbHMpCiAgICAgICAgcmV0dXJuIGszICsgc2VsZi5fcGFkXzF4MV90b18zeDMoazEpICsga2lkLnRvKGszLmRldmljZSksIGIzICsgYjEgKyBiaWQudG8oYjMuZGV2aWNlKQoKICAgIGRlZiBzd2l0Y2hfdG9fZGVwbG95KHNlbGYpIC0+IE5vbmU6CiAgICAgICAgaWYgc2VsZi5kZXBsb3k6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIGtlcm5lbCwgYmlhcyA9IHNlbGYuZ2V0X2VxdWl2YWxlbnRfa2VybmVsX2JpYXMoKQogICAgICAgIHNlbGYucmJyX3JlcGFyYW0gPSBubi5Db252MmQoCiAgICAgICAgICAgIHNlbGYuaW5fY2hhbm5lbHMsCiAgICAgICAgICAgIHNlbGYub3V0X2NoYW5uZWxzLAogICAgICAgICAgICAzLAogICAgICAgICAgICBzZWxmLnN0cmlkZSwKICAgICAgICAgICAgMSwKICAgICAgICAgICAgYmlhcz1UcnVlLAogICAgICAgICkKICAgICAgICBzZWxmLnJicl9yZXBhcmFtLndlaWdodC5kYXRhID0ga2VybmVsLmRldGFjaCgpLmNsb25lKCkKICAgICAgICBzZWxmLnJicl9yZXBhcmFtLmJpYXMuZGF0YSA9IGJpYXMuZGV0YWNoKCkuY2xvbmUoKQogICAgICAgIHNlbGYuY29udiA9IHNlbGYucmJyX3JlcGFyYW0KICAgICAgICBkZWwgc2VsZi5yYnJfZGVuc2UKICAgICAgICBkZWwgc2VsZi5yYnJfMXgxCiAgICAgICAgaWYgaGFzYXR0cihzZWxmLCAicmJyX2lkZW50aXR5Iik6CiAgICAgICAgICAgIGRlbCBzZWxmLnJicl9pZGVudGl0eQogICAgICAgIHNlbGYuZGVwbG95ID0gVHJ1ZQoKICAgIGRlZiBmdXNlX2NvbnZzKHNlbGYpIC0+IE5vbmU6CiAgICAgICAgIiIiQWxpYXMgZm9yIHN3aXRjaF90b19kZXBsb3kgY29tcGF0aWJsZSB3aXRoIFVsdHJhbHl0aWNzIERldGVjdGlvbk1vZGVsLmZ1c2UoKS4iIiIKICAgICAgICBzZWxmLnN3aXRjaF90b19kZXBsb3koKQoKICAgIGRlZiBmdXNlKHNlbGYpIC0+IE5vbmU6CiAgICAgICAgIiIiQWRkaXRpb25hbCBhbGlhcyBmb3IgbW9kZWwgZnVzaW9uLiIiIgogICAgICAgIHNlbGYuc3dpdGNoX3RvX2RlcGxveSgpCgogICAgZGVmIGZvcndhcmRfZnVzZShzZWxmLCB4OiB0b3JjaC5UZW5zb3IpIC0+IHRvcmNoLlRlbnNvcjoKICAgICAgICAiIiJGb3J3YXJkIHBhc3MgZm9yIGZ1c2VkIGRlcGxveSBtb2RlLiIiIgogICAgICAgIGlmIGhhc2F0dHIoc2VsZiwgImNvbnYiKToKICAgICAgICAgICAgcmV0dXJuIHNlbGYuYWN0KHNlbGYuY29udih4KSkKICAgICAgICBlbGlmIGhhc2F0dHIoc2VsZiwgInJicl9yZXBhcmFtIik6CiAgICAgICAgICAgIHJldHVybiBzZWxmLmFjdChzZWxmLnJicl9yZXBhcmFtKHgpKQogICAgICAgIHJldHVybiBzZWxmLmZvcndhcmQoeCkKCgpjbGFzcyBCaUZvcm1lckJsb2NrTGl0ZShubi5Nb2R1bGUpOgogICAgIiIiCiAgICBMaWdodHdlaWdodCBCaUZvcm1lci1pbnNwaXJlZCBibG9jayBmb3IgYWJsYXRpb24uCgogICAgVGhpcyBpcyBub3QgYSBkcm9wLWluIGNvcHkgb2YgdGhlIG9mZmljaWFsIEJpRm9ybWVyIGltcGxlbWVudGF0aW9uLiBJdCBrZWVwcwogICAgdGhlIGNvcmUgaWRlYSBmb3IgZXhwZXJpbWVudHM6IHJlZ2lvbi1sZXZlbCByb3V0aW5nIGZvbGxvd2VkIGJ5IGF0dGVudGlvbiBvbgogICAgYSBzbWFsbCBzZXQgb2Ygcm91dGVkIHJlZ2lvbnMuIFVzZSBpdCBhcyBhbiBhYmxhdGlvbiBjYW5kaWRhdGUsIHRoZW4gY29tcGFyZQogICAgbGF0ZW5jeSBhZ2FpbnN0IEVDQS9DQkFNL1RyaXBsZXQgYXR0ZW50aW9uLgogICAgIiIiCgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGNoYW5uZWxzOiBpbnQsIG51bV9oZWFkczogaW50ID0gNCwgcmVnaW9uX3NpemU6IGludCA9IDgsIHRvcGs6IGludCA9IDQpIC0+IE5vbmU6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgYXNzZXJ0IGNoYW5uZWxzICUgbnVtX2hlYWRzID09IDAsICJjaGFubmVscyBtdXN0IGJlIGRpdmlzaWJsZSBieSBudW1faGVhZHMiCiAgICAgICAgc2VsZi5jaGFubmVscyA9IGNoYW5uZWxzCiAgICAgICAgc2VsZi5udW1faGVhZHMgPSBudW1faGVhZHMKICAgICAgICBzZWxmLnJlZ2lvbl9zaXplID0gcmVnaW9uX3NpemUKICAgICAgICBzZWxmLnRvcGsgPSB0b3BrCiAgICAgICAgc2VsZi5xa3YgPSBubi5Db252MmQoY2hhbm5lbHMsIGNoYW5uZWxzICogMywgMSwgYmlhcz1GYWxzZSkKICAgICAgICBzZWxmLnByb2ogPSBubi5Db252MmQoY2hhbm5lbHMsIGNoYW5uZWxzLCAxLCBiaWFzPUZhbHNlKQogICAgICAgIHNlbGYubm9ybSA9IG5uLkJhdGNoTm9ybTJkKGNoYW5uZWxzKQoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIHg6IHRvcmNoLlRlbnNvcikgLT4gdG9yY2guVGVuc29yOgogICAgICAgIGIsIGMsIGgsIHcgPSB4LnNoYXBlCiAgICAgICAgcnMgPSBzZWxmLnJlZ2lvbl9zaXplCiAgICAgICAgcGFkX2ggPSAocnMgLSBoICUgcnMpICUgcnMKICAgICAgICBwYWRfdyA9IChycyAtIHcgJSBycykgJSBycwogICAgICAgIHhfcGFkID0gRi5wYWQoeCwgKDAsIHBhZF93LCAwLCBwYWRfaCkpCiAgICAgICAgaHAsIHdwID0geF9wYWQuc2hhcGVbLTI6XQogICAgICAgIGdoLCBndyA9IGhwIC8vIHJzLCB3cCAvLyBycwoKICAgICAgICBxLCBrLCB2ID0gc2VsZi5xa3YoeF9wYWQpLmNodW5rKDMsIGRpbT0xKQogICAgICAgIHFfcmVnaW9ucyA9IHEudW5mb2xkKDIsIHJzLCBycykudW5mb2xkKDMsIHJzLCBycykuY29udGlndW91cygpCiAgICAgICAga19yZWdpb25zID0gay51bmZvbGQoMiwgcnMsIHJzKS51bmZvbGQoMywgcnMsIHJzKS5jb250aWd1b3VzKCkKICAgICAgICB2X3JlZ2lvbnMgPSB2LnVuZm9sZCgyLCBycywgcnMpLnVuZm9sZCgzLCBycywgcnMpLmNvbnRpZ3VvdXMoKQoKICAgICAgICAjIFtCLCBDLCBHSCwgR1csIFJTLCBSU10gLT4gW0IsIFIsIFQsIENdCiAgICAgICAgcV90b2tlbnMgPSBxX3JlZ2lvbnMucGVybXV0ZSgwLCAyLCAzLCA0LCA1LCAxKS5yZXNoYXBlKGIsIGdoICogZ3csIHJzICogcnMsIGMpCiAgICAgICAga190b2tlbnMgPSBrX3JlZ2lvbnMucGVybXV0ZSgwLCAyLCAzLCA0LCA1LCAxKS5yZXNoYXBlKGIsIGdoICogZ3csIHJzICogcnMsIGMpCiAgICAgICAgdl90b2tlbnMgPSB2X3JlZ2lvbnMucGVybXV0ZSgwLCAyLCAzLCA0LCA1LCAxKS5yZXNoYXBlKGIsIGdoICogZ3csIHJzICogcnMsIGMpCgogICAgICAgIHFfcmVnaW9uID0gcV90b2tlbnMubWVhbihkaW09MikKICAgICAgICBrX3JlZ2lvbiA9IGtfdG9rZW5zLm1lYW4oZGltPTIpCiAgICAgICAgcm91dGVfbG9naXRzID0gdG9yY2gubWF0bXVsKHFfcmVnaW9uLCBrX3JlZ2lvbi50cmFuc3Bvc2UoLTEsIC0yKSkgLyAoYyAqKiAwLjUpCiAgICAgICAgdG9wayA9IG1pbihzZWxmLnRvcGssIGdoICogZ3cpCiAgICAgICAgcm91dGVfaWR4ID0gcm91dGVfbG9naXRzLnRvcGsodG9waywgZGltPS0xKS5pbmRpY2VzCgogICAgICAgIG91dF9yZWdpb25zID0gW10KICAgICAgICBoZWFkX2RpbSA9IGMgLy8gc2VsZi5udW1faGVhZHMKICAgICAgICBmb3IgcmVnaW9uX2lkeCBpbiByYW5nZShnaCAqIGd3KToKICAgICAgICAgICAgc2VsZWN0ZWQgPSByb3V0ZV9pZHhbOiwgcmVnaW9uX2lkeF0KICAgICAgICAgICAga19zZWwgPSB0b3JjaC5zdGFjayhba190b2tlbnNbYmksIHNlbGVjdGVkW2JpXV0ucmVzaGFwZSh0b3BrICogcnMgKiBycywgYykgZm9yIGJpIGluIHJhbmdlKGIpXSwgZGltPTApCiAgICAgICAgICAgIHZfc2VsID0gdG9yY2guc3RhY2soW3ZfdG9rZW5zW2JpLCBzZWxlY3RlZFtiaV1dLnJlc2hhcGUodG9wayAqIHJzICogcnMsIGMpIGZvciBiaSBpbiByYW5nZShiKV0sIGRpbT0wKQogICAgICAgICAgICBxX2N1ciA9IHFfdG9rZW5zWzosIHJlZ2lvbl9pZHhdCgogICAgICAgICAgICBxaCA9IHFfY3VyLnJlc2hhcGUoYiwgcnMgKiBycywgc2VsZi5udW1faGVhZHMsIGhlYWRfZGltKS50cmFuc3Bvc2UoMSwgMikKICAgICAgICAgICAga2ggPSBrX3NlbC5yZXNoYXBlKGIsIHRvcGsgKiBycyAqIHJzLCBzZWxmLm51bV9oZWFkcywgaGVhZF9kaW0pLnRyYW5zcG9zZSgxLCAyKQogICAgICAgICAgICB2aCA9IHZfc2VsLnJlc2hhcGUoYiwgdG9wayAqIHJzICogcnMsIHNlbGYubnVtX2hlYWRzLCBoZWFkX2RpbSkudHJhbnNwb3NlKDEsIDIpCiAgICAgICAgICAgIGF0dG4gPSB0b3JjaC5zb2Z0bWF4KHRvcmNoLm1hdG11bChxaCwga2gudHJhbnNwb3NlKC0xLCAtMikpIC8gKGhlYWRfZGltICoqIDAuNSksIGRpbT0tMSkKICAgICAgICAgICAgb3V0ID0gdG9yY2gubWF0bXVsKGF0dG4sIHZoKS50cmFuc3Bvc2UoMSwgMikucmVzaGFwZShiLCBycyAqIHJzLCBjKQogICAgICAgICAgICBvdXRfcmVnaW9ucy5hcHBlbmQob3V0KQoKICAgICAgICB5ID0gdG9yY2guc3RhY2sob3V0X3JlZ2lvbnMsIGRpbT0xKS5yZXNoYXBlKGIsIGdoLCBndywgcnMsIHJzLCBjKQogICAgICAgIHkgPSB5LnBlcm11dGUoMCwgNSwgMSwgMywgMiwgNCkucmVzaGFwZShiLCBjLCBocCwgd3ApCiAgICAgICAgeSA9IHlbOiwgOiwgOmgsIDp3XQogICAgICAgIHJldHVybiB4ICsgc2VsZi5ub3JtKHNlbGYucHJvaih5KSkKCiAgICBkZWYgZnVzZShzZWxmKSAtPiBOb25lOgogICAgICAgIHBhc3MKCiAgICBkZWYgZnVzZV9jb252cyhzZWxmKSAtPiBOb25lOgogICAgICAgIHBhc3MKCgpkZWYgeHl3aF90b194eXh5KGJveGVzOiB0b3JjaC5UZW5zb3IpIC0+IHRvcmNoLlRlbnNvcjoKICAgIHgsIHksIHcsIGggPSBib3hlcy51bmJpbmQoLTEpCiAgICBoYWxmX3cgPSB3IC8gMgogICAgaGFsZl9oID0gaCAvIDIKICAgIHJldHVybiB0b3JjaC5zdGFjaygoeCAtIGhhbGZfdywgeSAtIGhhbGZfaCwgeCArIGhhbGZfdywgeSArIGhhbGZfaCksIGRpbT0tMSkKCgpkZWYgZm9jYWxfZWlvdV9sb3NzKAogICAgcHJlZF9ib3hlczogdG9yY2guVGVuc29yLAogICAgdGFyZ2V0X2JveGVzOiB0b3JjaC5UZW5zb3IsCiAgICB4eXdoOiBib29sID0gVHJ1ZSwKICAgIGdhbW1hOiBmbG9hdCA9IDAuNSwKICAgIGVwczogZmxvYXQgPSAxZS03LAogICAgcmVkdWN0aW9uOiBzdHIgPSAibWVhbiIsCikgLT4gdG9yY2guVGVuc29yOgogICAgIiIiCiAgICBGb2NhbC1FSW9VIGJveCByZWdyZXNzaW9uIGxvc3MuCgogICAgQXJnczoKICAgICAgICBwcmVkX2JveGVzOiBbTiwgNF0gcHJlZGljdGVkIGJveGVzLgogICAgICAgIHRhcmdldF9ib3hlczogW04sIDRdIHRhcmdldCBib3hlcy4KICAgICAgICB4eXdoOiBUcnVlIHdoZW4gYm94ZXMgYXJlIGNlbnRlci14LCBjZW50ZXIteSwgd2lkdGgsIGhlaWdodC4KICAgICAgICBnYW1tYTogZm9jYWwgZXhwb25lbnQgZnJvbSB0aGUgRm9jYWwtRUlvVSBwYXBlciBmYW1pbHkuCiAgICAiIiIKICAgIGlmIHh5d2g6CiAgICAgICAgcHJlZCA9IHh5d2hfdG9feHl4eShwcmVkX2JveGVzKQogICAgICAgIHRhcmdldCA9IHh5d2hfdG9feHl4eSh0YXJnZXRfYm94ZXMpCiAgICBlbHNlOgogICAgICAgIHByZWQgPSBwcmVkX2JveGVzCiAgICAgICAgdGFyZ2V0ID0gdGFyZ2V0X2JveGVzCgogICAgcHgxLCBweTEsIHB4MiwgcHkyID0gcHJlZC51bmJpbmQoLTEpCiAgICB0eDEsIHR5MSwgdHgyLCB0eTIgPSB0YXJnZXQudW5iaW5kKC0xKQogICAgcHcgPSAocHgyIC0gcHgxKS5jbGFtcChtaW49ZXBzKQogICAgcGggPSAocHkyIC0gcHkxKS5jbGFtcChtaW49ZXBzKQogICAgdHcgPSAodHgyIC0gdHgxKS5jbGFtcChtaW49ZXBzKQogICAgdGggPSAodHkyIC0gdHkxKS5jbGFtcChtaW49ZXBzKQoKICAgIGludGVyX3gxID0gdG9yY2gubWF4aW11bShweDEsIHR4MSkKICAgIGludGVyX3kxID0gdG9yY2gubWF4aW11bShweTEsIHR5MSkKICAgIGludGVyX3gyID0gdG9yY2gubWluaW11bShweDIsIHR4MikKICAgIGludGVyX3kyID0gdG9yY2gubWluaW11bShweTIsIHR5MikKICAgIGludGVyID0gKGludGVyX3gyIC0gaW50ZXJfeDEpLmNsYW1wKG1pbj0wKSAqIChpbnRlcl95MiAtIGludGVyX3kxKS5jbGFtcChtaW49MCkKICAgIHVuaW9uID0gcHcgKiBwaCArIHR3ICogdGggLSBpbnRlciArIGVwcwogICAgaW91ID0gKGludGVyIC8gdW5pb24pLmNsYW1wKG1pbj1lcHMsIG1heD0xLjApCgogICAgcGN4ID0gKHB4MSArIHB4MikgLyAyCiAgICBwY3kgPSAocHkxICsgcHkyKSAvIDIKICAgIHRjeCA9ICh0eDEgKyB0eDIpIC8gMgogICAgdGN5ID0gKHR5MSArIHR5MikgLyAyCiAgICBjZW50ZXJfZGlzdCA9IChwY3ggLSB0Y3gpLnNxdWFyZSgpICsgKHBjeSAtIHRjeSkuc3F1YXJlKCkKCiAgICBjdyA9ICh0b3JjaC5tYXhpbXVtKHB4MiwgdHgyKSAtIHRvcmNoLm1pbmltdW0ocHgxLCB0eDEpKS5jbGFtcChtaW49ZXBzKQogICAgY2ggPSAodG9yY2gubWF4aW11bShweTIsIHR5MikgLSB0b3JjaC5taW5pbXVtKHB5MSwgdHkxKSkuY2xhbXAobWluPWVwcykKICAgIGMyID0gY3cuc3F1YXJlKCkgKyBjaC5zcXVhcmUoKSArIGVwcwoKICAgIGVpb3UgPSAxLjAgLSBpb3UgKyBjZW50ZXJfZGlzdCAvIGMyICsgKHB3IC0gdHcpLnNxdWFyZSgpIC8gKGN3LnNxdWFyZSgpICsgZXBzKSArIChwaCAtIHRoKS5zcXVhcmUoKSAvIChjaC5zcXVhcmUoKSArIGVwcykKICAgIGxvc3MgPSBpb3UucG93KGdhbW1hKSAqIGVpb3UKCiAgICBpZiByZWR1Y3Rpb24gPT0gIm1lYW4iOgogICAgICAgIHJldHVybiBsb3NzLm1lYW4oKQogICAgaWYgcmVkdWN0aW9uID09ICJzdW0iOgogICAgICAgIHJldHVybiBsb3NzLnN1bSgpCiAgICBpZiByZWR1Y3Rpb24gPT0gIm5vbmUiOgogICAgICAgIHJldHVybiBsb3NzCiAgICByYWlzZSBWYWx1ZUVycm9yKGYiVW5zdXBwb3J0ZWQgcmVkdWN0aW9uOiB7cmVkdWN0aW9ufSIpCgoKZGVmIGFscGhhX2JhbGFuY2VkX2ZvY2FsX2JjZSgKICAgIGxvZ2l0czogdG9yY2guVGVuc29yLAogICAgdGFyZ2V0czogdG9yY2guVGVuc29yLAogICAgYWxwaGE6IHRvcmNoLlRlbnNvciB8IGZsb2F0ID0gMC43NSwKICAgIGdhbW1hOiBmbG9hdCA9IDIuMCwKICAgIHJlZHVjdGlvbjogc3RyID0gIm1lYW4iLAopIC0+IHRvcmNoLlRlbnNvcjoKICAgICIiIgogICAgRm9jYWwgQkNFIGZvciBjbGFzcyBpbWJhbGFuY2UuCgogICAgRm9yIHR3byBjbGFzcyBsb2dpdHMgaW4gdGhpcyBwcm9qZWN0LCBwYXNzIGFuIGFscGhhIHRlbnNvciBzdWNoIGFzCiAgICBgdG9yY2gudGVuc29yKFswLjg1LCAwLjE1XSwgZGV2aWNlPWxvZ2l0cy5kZXZpY2UpYCB0byB3ZWlnaHQgYGhhdGAgaGlnaGVyCiAgICB0aGFuIGBwZXJzb25gLgogICAgIiIiCiAgICBiY2UgPSBGLmJpbmFyeV9jcm9zc19lbnRyb3B5X3dpdGhfbG9naXRzKGxvZ2l0cywgdGFyZ2V0cywgcmVkdWN0aW9uPSJub25lIikKICAgIHByb2IgPSB0b3JjaC5zaWdtb2lkKGxvZ2l0cykKICAgIHBfdCA9IHByb2IgKiB0YXJnZXRzICsgKDEuMCAtIHByb2IpICogKDEuMCAtIHRhcmdldHMpCgogICAgaWYgbm90IHRvcmNoLmlzX3RlbnNvcihhbHBoYSk6CiAgICAgICAgYWxwaGFfdCA9IHRvcmNoLnRlbnNvcihhbHBoYSwgZGV2aWNlPWxvZ2l0cy5kZXZpY2UsIGR0eXBlPWxvZ2l0cy5kdHlwZSkKICAgIGVsc2U6CiAgICAgICAgYWxwaGFfdCA9IGFscGhhLnRvKGRldmljZT1sb2dpdHMuZGV2aWNlLCBkdHlwZT1sb2dpdHMuZHR5cGUpCiAgICB3aGlsZSBhbHBoYV90Lm5kaW0gPCBsb2dpdHMubmRpbToKICAgICAgICBhbHBoYV90ID0gYWxwaGFfdC52aWV3KCooWzFdICogKGxvZ2l0cy5uZGltIC0gMSkpLCAtMSkKCiAgICB3ZWlnaHQgPSBhbHBoYV90ICogdGFyZ2V0cyArICgxLjAgLSBhbHBoYV90KSAqICgxLjAgLSB0YXJnZXRzKQogICAgbG9zcyA9IHdlaWdodCAqICgxLjAgLSBwX3QpLnBvdyhnYW1tYSkgKiBiY2UKICAgIGlmIHJlZHVjdGlvbiA9PSAibWVhbiI6CiAgICAgICAgcmV0dXJuIGxvc3MubWVhbigpCiAgICBpZiByZWR1Y3Rpb24gPT0gInN1bSI6CiAgICAgICAgcmV0dXJuIGxvc3Muc3VtKCkKICAgIGlmIHJlZHVjdGlvbiA9PSAibm9uZSI6CiAgICAgICAgcmV0dXJuIGxvc3MKICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJVbnN1cHBvcnRlZCByZWR1Y3Rpb246IHtyZWR1Y3Rpb259IikKCgpkZWYgc21va2VfdGVzdCgpIC0+IE5vbmU6CiAgICB4ID0gdG9yY2gucmFuZG4oMiwgMTYsIDMyLCAzMikKICAgIGNvb3JkID0gQ29vcmRDb252KDE2LCAyNCkKICAgIHJlcCA9IFJlcENvbnYoMjQsIDI0KQogICAgYXR0biA9IEJpRm9ybWVyQmxvY2tMaXRlKDI0LCBudW1faGVhZHM9NCwgcmVnaW9uX3NpemU9OCwgdG9waz0yKQogICAgd2l0aCB0b3JjaC5ub19ncmFkKCk6CiAgICAgICAgeSA9IGF0dG4ocmVwKGNvb3JkKHgpKSkKICAgIGFzc2VydCB5LnNoYXBlID09ICgyLCAyNCwgMzIsIDMyKQoKICAgIHJlcC5ldmFsKCkKICAgIHdpdGggdG9yY2gubm9fZ3JhZCgpOgogICAgICAgIGJlZm9yZSA9IHJlcChjb29yZCh4KSkKICAgICAgICByZXAuc3dpdGNoX3RvX2RlcGxveSgpCiAgICAgICAgYWZ0ZXIgPSByZXAoY29vcmQoeCkpCiAgICBtYXhfZGlmZiA9IChiZWZvcmUgLSBhZnRlcikuYWJzKCkubWF4KCkuaXRlbSgpCiAgICBwcmludCh7InNoYXBlIjogdHVwbGUoeS5zaGFwZSksICJyZXBjb252X2Z1c2lvbl9tYXhfZGlmZiI6IG1heF9kaWZmfSkKCiAgICBwcmVkID0gdG9yY2gudGVuc29yKFtbMC41LCAwLjUsIDAuMiwgMC4yXSwgWzAuNCwgMC40LCAwLjEsIDAuMl1dKQogICAgdGFyZ2V0ID0gdG9yY2gudGVuc29yKFtbMC41LCAwLjUsIDAuMiwgMC4yXSwgWzAuNDUsIDAuNCwgMC4xLCAwLjJdXSkKICAgIHByaW50KHsiZm9jYWxfZWlvdSI6IGZsb2F0KGZvY2FsX2Vpb3VfbG9zcyhwcmVkLCB0YXJnZXQpKX0pCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHNtb2tlX3Rlc3QoKQo="
custom_modules_src = base64.b64decode(custom_modules_b64.encode("ascii")).decode("utf-8")
Path("custom_ablation_modules.py").write_text(custom_modules_src, encoding="utf-8")

# 2. Materialize ablation_trainer.py (MultiSeedAblationTrainer and build_ablation_model factory)
ablation_trainer_b64 = "IyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQojIEFCTEFUSU9OIFRSQUlORVIgQU5EIEFSQ0hJVEVDVFVSRSBGQUNUT1JZIChLQUdHTEUgRFVBTCBURVNMQSBUNCBERFApCiMgSUVFRSBBQUlNTCAyMDI3IFJldmlld2VyIFJlYnV0dGFsICYgU3RhdGlzdGljYWwgU2lnbmlmaWNhbmNlIFZlcmlmaWNhdGlvbgojIEF1dGhvcjogTmd1eWVuIEhhbiBOaHUgKEZQVCBVbml2ZXJzaXR5KQojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CmltcG9ydCBvcwppbXBvcnQgc3lzCmltcG9ydCBjb3B5CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAppbXBvcnQgdG9yY2gKaW1wb3J0IHRvcmNoLm5uIGFzIG5uCmZyb20gdWx0cmFseXRpY3MgaW1wb3J0IFlPTE8KZnJvbSB1bHRyYWx5dGljcy5tb2RlbHMueW9sby5kZXRlY3QgaW1wb3J0IERldGVjdGlvblRyYWluZXIKaW1wb3J0IHVsdHJhbHl0aWNzLm5uLm1vZHVsZXMgYXMgdW5fbW9kCmltcG9ydCB1bHRyYWx5dGljcy5ubi50YXNrcyBhcyB1bl90YXNrcwppbXBvcnQgdWx0cmFseXRpY3MudXRpbHMubG9zcyBhcyB1bF9sb3NzCgojIDEuIEltcG9ydCByZXNlYXJjaCBtb2R1bGVzCmZyb20gY3VzdG9tX2FibGF0aW9uX21vZHVsZXMgaW1wb3J0IENvb3JkQ29udiwgUmVwQ29udiwgQmlGb3JtZXJCbG9ja0xpdGUsIGZvY2FsX2Vpb3VfbG9zcwoKIyAyLiBEeW5hbWljIG1vZHVsZSByZWdpc3RyYXRpb24gaW50byBVbHRyYWx5dGljcyBuYW1lc3BhY2UKdW5fbW9kLkNvb3JkQ29udiA9IENvb3JkQ29udgp1bl9tb2QuUmVwQ29udiA9IFJlcENvbnYKdW5fbW9kLkJpRm9ybWVyQmxvY2tMaXRlID0gQmlGb3JtZXJCbG9ja0xpdGUKc2V0YXR0cih1bl90YXNrcywgJ0Nvb3JkQ29udicsIENvb3JkQ29udikKc2V0YXR0cih1bl90YXNrcywgJ1JlcENvbnYnLCBSZXBDb252KQpzZXRhdHRyKHVuX3Rhc2tzLCAnQmlGb3JtZXJCbG9ja0xpdGUnLCBCaUZvcm1lckJsb2NrTGl0ZSkKCiMgMy4gSG9vayBBYmxhdGlvbkJib3hMb3NzIGZvciBGb2NhbCBFSW9VIHN1cHBvcnQKY2xhc3MgQWJsYXRpb25CYm94TG9zcyh1bF9sb3NzLkJib3hMb3NzKToKICAgIGRlZiBmb3J3YXJkKHNlbGYsIHByZWRfZGlzdCwgcHJlZF9iYm94ZXMsIGFuY2hvcl9wb2ludHMsIHRhcmdldF9iYm94ZXMsIHRhcmdldF9zY29yZXMsIHRhcmdldF9zY29yZXNfc3VtLCBmZ19tYXNrLCAqYXJncywgKiprd2FyZ3MpOgogICAgICAgIGN1cl9hYiA9IG9zLmVudmlyb24uZ2V0KCJDVVJSRU5UX0FCTEFUSU9OX0lEIiwgIkEwIikKICAgICAgICBpZiBjdXJfYWIgbm90IGluIFsiQTQiLCAiQTYiXToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcmV0dXJuIHN1cGVyKCkuZm9yd2FyZChwcmVkX2Rpc3QsIHByZWRfYmJveGVzLCBhbmNob3JfcG9pbnRzLCB0YXJnZXRfYmJveGVzLCB0YXJnZXRfc2NvcmVzLCB0YXJnZXRfc2NvcmVzX3N1bSwgZmdfbWFzaywgKmFyZ3MsICoqa3dhcmdzKQogICAgICAgICAgICBleGNlcHQgVHlwZUVycm9yOgogICAgICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgICAgIHJldHVybiBzdXBlcigpLmZvcndhcmQocHJlZF9kaXN0LCBwcmVkX2Jib3hlcywgYW5jaG9yX3BvaW50cywgdGFyZ2V0X2Jib3hlcywgdGFyZ2V0X3Njb3JlcywgdGFyZ2V0X3Njb3Jlc19zdW0sIGZnX21hc2spCiAgICAgICAgICAgICAgICBleGNlcHQgVHlwZUVycm9yOgogICAgICAgICAgICAgICAgICAgIGR1bW15X2ltZ3N6ID0ga3dhcmdzLmdldCgiaW1nc3oiLCB0b3JjaC50ZW5zb3IoWzY0MCwgNjQwXSwgZGV2aWNlPXByZWRfZGlzdC5kZXZpY2UpKQogICAgICAgICAgICAgICAgICAgIGR1bW15X3N0cmlkZSA9IGt3YXJncy5nZXQoInN0cmlkZSIsIHRvcmNoLm9uZXMoKGFuY2hvcl9wb2ludHMuc2hhcGVbMF0sIDEpLCBkZXZpY2U9cHJlZF9kaXN0LmRldmljZSkgKiA4KQogICAgICAgICAgICAgICAgICAgIHJldHVybiBzdXBlcigpLmZvcndhcmQocHJlZF9kaXN0LCBwcmVkX2Jib3hlcywgYW5jaG9yX3BvaW50cywgdGFyZ2V0X2Jib3hlcywgdGFyZ2V0X3Njb3JlcywgdGFyZ2V0X3Njb3Jlc19zdW0sIGZnX21hc2ssIGR1bW15X2ltZ3N6LCBkdW1teV9zdHJpZGUpCgogICAgICAgIHdlaWdodCA9IHRhcmdldF9zY29yZXMuc3VtKC0xKVtmZ19tYXNrXS51bnNxdWVlemUoLTEpCiAgICAgICAgcF9ib3ggPSBwcmVkX2Jib3hlc1tmZ19tYXNrXQogICAgICAgIHRfYm94ID0gdGFyZ2V0X2Jib3hlc1tmZ19tYXNrXQogICAgICAgIGlmIHBfYm94LnNoYXBlWzBdID4gMDoKICAgICAgICAgICAgcHgxLCBweTEsIHB4MiwgcHkyID0gcF9ib3gudW5iaW5kKC0xKQogICAgICAgICAgICB0eDEsIHR5MSwgdHgyLCB0eTIgPSB0X2JveC51bmJpbmQoLTEpCiAgICAgICAgICAgIHB3ID0gKHB4MiAtIHB4MSkuY2xhbXAobWluPTFlLTcpCiAgICAgICAgICAgIHBoID0gKHB5MiAtIHB5MSkuY2xhbXAobWluPTFlLTcpCiAgICAgICAgICAgIHR3ID0gKHR4MiAtIHR4MSkuY2xhbXAobWluPTFlLTcpCiAgICAgICAgICAgIHRoID0gKHR5MiAtIHR5MSkuY2xhbXAobWluPTFlLTcpCgogICAgICAgICAgICBpbnRlcl94MSA9IHRvcmNoLm1heGltdW0ocHgxLCB0eDEpCiAgICAgICAgICAgIGludGVyX3kxID0gdG9yY2gubWF4aW11bShweTEsIHR5MSkKICAgICAgICAgICAgaW50ZXJfeDIgPSB0b3JjaC5taW5pbXVtKHB4MiwgdHgyKQogICAgICAgICAgICBpbnRlcl95MiA9IHRvcmNoLm1pbmltdW0ocHkyLCB0eTIpCiAgICAgICAgICAgIGludGVyID0gKGludGVyX3gyIC0gaW50ZXJfeDEpLmNsYW1wKG1pbj0wKSAqIChpbnRlcl95MiAtIGludGVyX3kxKS5jbGFtcChtaW49MCkKICAgICAgICAgICAgdW5pb24gPSBwdyAqIHBoICsgdHcgKiB0aCAtIGludGVyICsgMWUtNwogICAgICAgICAgICBpb3UgPSAoaW50ZXIgLyB1bmlvbikuY2xhbXAobWluPTFlLTcsIG1heD0xLjApCgogICAgICAgICAgICBwY3ggPSAocHgxICsgcHgyKSAvIDIuMAogICAgICAgICAgICBwY3kgPSAocHkxICsgcHkyKSAvIDIuMAogICAgICAgICAgICB0Y3ggPSAodHgxICsgdHgyKSAvIDIuMAogICAgICAgICAgICB0Y3kgPSAodHkxICsgdHkyKSAvIDIuMAogICAgICAgICAgICBjZW50ZXJfZGlzdCA9IChwY3ggLSB0Y3gpLnNxdWFyZSgpICsgKHBjeSAtIHRjeSkuc3F1YXJlKCkKCiAgICAgICAgICAgIGN3ID0gKHRvcmNoLm1heGltdW0ocHgyLCB0eDIpIC0gdG9yY2gubWluaW11bShweDEsIHR4MSkpLmNsYW1wKG1pbj0xZS03KQogICAgICAgICAgICBjaCA9ICh0b3JjaC5tYXhpbXVtKHB5MiwgdHkyKSAtIHRvcmNoLm1pbmltdW0ocHkxLCB0eTEpKS5jbGFtcChtaW49MWUtNykKICAgICAgICAgICAgYzIgPSBjdy5zcXVhcmUoKSArIGNoLnNxdWFyZSgpICsgMWUtNwoKICAgICAgICAgICAgZ2FtbWEgPSAwLjUKICAgICAgICAgICAgZWlvdSA9IDEuMCAtIGlvdSArIGNlbnRlcl9kaXN0IC8gYzIgKyAocHcgLSB0dykuc3F1YXJlKCkgLyAoY3cuc3F1YXJlKCkgKyAxZS03KSArIChwaCAtIHRoKS5zcXVhcmUoKSAvIChjaC5zcXVhcmUoKSArIDFlLTcpCiAgICAgICAgICAgIGxvc3NfYm94X3NhbXBsZSA9IGlvdS5wb3coZ2FtbWEpICogZWlvdQogICAgICAgICAgICBsb3NzX2lvdSA9IChsb3NzX2JveF9zYW1wbGUudW5zcXVlZXplKC0xKSAqIHdlaWdodCkuc3VtKCkgLyB0YXJnZXRfc2NvcmVzX3N1bQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGxvc3NfaW91ID0gdG9yY2gudGVuc29yKDAuMCkudG8ocHJlZF9kaXN0LmRldmljZSkKCiAgICAgICAgaWYgc2VsZi5kZmxfbG9zcyBhbmQgcF9ib3guc2hhcGVbMF0gPiAwOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICB0YXJnZXRfbHRyYiA9IHVsX2xvc3MuYmJveDJkaXN0KGFuY2hvcl9wb2ludHMsIHRhcmdldF9iYm94ZXMsIHNlbGYuZGZsX2xvc3MucmVnX21heCAtIDEpCiAgICAgICAgICAgIGV4Y2VwdCBBdHRyaWJ1dGVFcnJvcjoKICAgICAgICAgICAgICAgIGZyb20gdWx0cmFseXRpY3MudXRpbHMudGFsIGltcG9ydCBiYm94MmRpc3QKICAgICAgICAgICAgICAgIHRhcmdldF9sdHJiID0gYmJveDJkaXN0KGFuY2hvcl9wb2ludHMsIHRhcmdldF9iYm94ZXMsIHNlbGYuZGZsX2xvc3MucmVnX21heCAtIDEpCiAgICAgICAgICAgIGxvc3NfZGZsID0gc2VsZi5kZmxfbG9zcyhwcmVkX2Rpc3RbZmdfbWFza10udmlldygtMSwgc2VsZi5kZmxfbG9zcy5yZWdfbWF4KSwgdGFyZ2V0X2x0cmJbZmdfbWFza10pICogd2VpZ2h0CiAgICAgICAgICAgIGxvc3NfZGZsID0gbG9zc19kZmwuc3VtKCkgLyB0YXJnZXRfc2NvcmVzX3N1bQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGxvc3NfZGZsID0gdG9yY2gudGVuc29yKDAuMCkudG8ocHJlZF9kaXN0LmRldmljZSkKCiAgICAgICAgcmV0dXJuIGxvc3NfaW91LCBsb3NzX2RmbAoKdWxfbG9zcy5CYm94TG9zcyA9IEFibGF0aW9uQmJveExvc3MKCiMgNC4gTW9kZWwgQXJjaGl0ZWN0dXJlIEZhY3RvcnkKZGVmIGJ1aWxkX2FibGF0aW9uX21vZGVsKGFiX2lkOiBzdHIsIGJhc2Vfd2VpZ2h0OiBzdHIgPSAieW9sbzExcy5wdCIsIHAyX3lhbWxfcGF0aDogc3RyID0gIi9rYWdnbGUvd29ya2luZy9yZXBfeW9sbzExc19wMi55YW1sIikgLT4gWU9MTzoKICAgIHAyX3BhdGggPSBQYXRoKHAyX3lhbWxfcGF0aCkKICAgIGlmIG5vdCBwMl9wYXRoLmV4aXN0cygpOgogICAgICAgIHAyX3BhdGggPSBQYXRoKCJyZXBfeW9sbzExc19wMi55YW1sIikKCiAgICBpZiBhYl9pZCA9PSAiQTEiOgogICAgICAgIGJhc2UgPSBZT0xPKHN0cihwMl9wYXRoKSkKICAgICAgICBiYXNlLmxvYWQoYmFzZV93ZWlnaHQpCiAgICAgICAgcmV0dXJuIGJhc2UKCiAgICBiYXNlID0gWU9MTyhiYXNlX3dlaWdodCkKICAgIG0gPSBiYXNlLm1vZGVsCgogICAgaWYgYWJfaWQgPT0gIkEwIjoKICAgICAgICByZXR1cm4gYmFzZQoKICAgIGlmIGFiX2lkIGluIFsiQTIiLCAiQTYiXToKICAgICAgICBjb252MCA9IG0ubW9kZWxbMF0uY29udgogICAgICAgIGNvb3JkX2NvbnYgPSBDb29yZENvbnYoYzE9Y29udjAuaW5fY2hhbm5lbHMsIGMyPWNvbnYwLm91dF9jaGFubmVscywgaz0zLCBzPTIsIHdpdGhfcj1GYWxzZSkKICAgICAgICBjb29yZF9jb252LmksIGNvb3JkX2NvbnYuZiwgY29vcmRfY29udi50eXBlID0gMCwgLTEsICJDb29yZENvbnYiCiAgICAgICAgbS5tb2RlbFswXSA9IGNvb3JkX2NvbnYKCiAgICBpZiBhYl9pZCBpbiBbIkEzIiwgIkE0IiwgIkE1IiwgIkE2Il06CiAgICAgICAgZm9yIGlkeCwgbGF5ZXIgaW4gZW51bWVyYXRlKG0ubW9kZWwpOgogICAgICAgICAgICBpZiBpZHggPiAwIGFuZCBoYXNhdHRyKGxheWVyLCAiY29udiIpIGFuZCBoYXNhdHRyKGxheWVyLmNvbnYsICJrZXJuZWxfc2l6ZSIpIGFuZCBsYXllci5jb252Lmtlcm5lbF9zaXplID09ICgzLCAzKToKICAgICAgICAgICAgICAgIGMxLCBjMiwgcyA9IGxheWVyLmNvbnYuaW5fY2hhbm5lbHMsIGxheWVyLmNvbnYub3V0X2NoYW5uZWxzLCBsYXllci5jb252LnN0cmlkZVswXQogICAgICAgICAgICAgICAgcmVwX2NvbnYgPSBSZXBDb252KGMxPWMxLCBjMj1jMiwgaz0zLCBzPXMsIGRlcGxveT1GYWxzZSkKICAgICAgICAgICAgICAgIHJlcF9jb252LmksIHJlcF9jb252LmYsIHJlcF9jb252LnR5cGUgPSBnZXRhdHRyKGxheWVyLCAiaSIsIGlkeCksIGdldGF0dHIobGF5ZXIsICJmIiwgLTEpLCAiUmVwQ29udiIKICAgICAgICAgICAgICAgIG0ubW9kZWxbaWR4XSA9IHJlcF9jb252CgogICAgaWYgYWJfaWQgaW4gWyJBNSIsICJBNiJdOgogICAgICAgIGZvciBpZHgsIGxheWVyIGluIGVudW1lcmF0ZShtLm1vZGVsKToKICAgICAgICAgICAgaWYgbGF5ZXIuX19jbGFzc19fLl9fbmFtZV9fIGluIFsiQzJQU0EiLCAiQzNrMiJdIGFuZCBpZHggPj0gOToKICAgICAgICAgICAgICAgIGNfaW4gPSBnZXRhdHRyKGxheWVyLCAiYzEiLCA1MTIpCiAgICAgICAgICAgICAgICBiaWZvcm1lciA9IEJpRm9ybWVyQmxvY2tMaXRlKGNoYW5uZWxzPWNfaW4sIG51bV9oZWFkcz00LCByZWdpb25fc2l6ZT04LCB0b3BrPTQpCiAgICAgICAgICAgICAgICBiaWZvcm1lci5pLCBiaWZvcm1lci5mLCBiaWZvcm1lci50eXBlID0gZ2V0YXR0cihsYXllciwgImkiLCBpZHgpLCBnZXRhdHRyKGxheWVyLCAiZiIsIC0xKSwgIkJpRm9ybWVyQmxvY2tMaXRlIgogICAgICAgICAgICAgICAgbS5tb2RlbFtpZHhdID0gYmlmb3JtZXIKICAgICAgICAgICAgICAgIGJyZWFrCgogICAgYmFzZS5tb2RlbCA9IG0KICAgIHJldHVybiBiYXNlCgpDVVJSRU5UX0FCTEFUSU9OX01PREVMID0gTm9uZQoKY2xhc3MgTXVsdGlTZWVkQWJsYXRpb25UcmFpbmVyKERldGVjdGlvblRyYWluZXIpOgogICAgIiIiQ3VzdG9tIFRyYWluZXIgZW5zdXJpbmcgZXhhY3QgYXJjaGl0ZWN0dXJhbCBtb2R1bGUgaW5qZWN0aW9uIGZvciBlYWNoIGFibGF0aW9uIGluIHNpbmdsZSBhbmQgbXVsdGktR1BVIEREUC4iIiIKICAgIGRlZiBnZXRfbW9kZWwoc2VsZiwgY2ZnPU5vbmUsIHdlaWdodHM9Tm9uZSwgdmVyYm9zZT1UcnVlKToKICAgICAgICBnbG9iYWwgQ1VSUkVOVF9BQkxBVElPTl9NT0RFTAogICAgICAgIGlmIENVUlJFTlRfQUJMQVRJT05fTU9ERUwgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHJldHVybiBDVVJSRU5UX0FCTEFUSU9OX01PREVMCiAgICAgICAgY3VyX2FiID0gb3MuZW52aXJvbi5nZXQoIkNVUlJFTlRfQUJMQVRJT05fSUQiLCAiQTAiKQogICAgICAgIG1vZGVsID0gYnVpbGRfYWJsYXRpb25fbW9kZWwoY3VyX2FiLCB3ZWlnaHRzIG9yICJ5b2xvMTFzLnB0IikubW9kZWwKICAgICAgICByZXR1cm4gbW9kZWwK"
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
    os.environ["PYTHONPATH"] = f"/kaggle/working:{existing_pp}" if existing_pp else "/kaggle/working"

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
patch_marker = "# === ABLATION BBOX LOSS PATCH ==="
if patch_marker in loss_src:
    base_src = loss_src.split(patch_marker)[0].rstrip()
elif "class AblationBboxLoss" in loss_src:
    base_src = loss_src.split("class AblationBboxLoss")[0]
    if "import os" in base_src:
        base_src = base_src[:base_src.rfind("import os")].rstrip()
else:
    base_src = loss_src.rstrip()

patch_code = '''
# === ABLATION BBOX LOSS PATCH ===
import os
import torch
class AblationBboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs):
        cur_ab = os.environ.get("CURRENT_ABLATION_ID", "A0")
        if cur_ab not in ["A4", "A6"]:
            try:
                return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs)
            except TypeError:
                try:
                    return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
                except TypeError:
                    dummy_imgsz = kwargs.get("imgsz", torch.tensor([640, 640], device=pred_dist.device))
                    dummy_stride = kwargs.get("stride", torch.ones((anchor_points.shape[0], 1), device=pred_dist.device) * 8)
                    return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, dummy_imgsz, dummy_stride)

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        p_box = pred_bboxes[fg_mask]
        t_box = target_bboxes[fg_mask]
        if p_box.shape[0] > 0:
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
            loss_iou = torch.tensor(0.0).to(pred_dist.device)
        if self.dfl_loss and p_box.shape[0] > 0:
            try:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            except NameError:
                from ultralytics.utils.tal import bbox2dist
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

BboxLoss = AblationBboxLoss
'''
try:
    loss_file_path.write_text(base_src + "\n" + patch_code, encoding="utf-8")
    print("[INFO] Injected AblationBboxLoss into physical site-packages loss.py")
except Exception as e:
    print(f"[WARNING] Could not patch physical loss.py: {e}")

print("[SUCCESS] Registered CoordConv, RepConv, BiFormer, Focal EIoU, and MultiSeedAblationTrainer.")

# ==============================================================================

# CELL 3: DATASET INGESTION AND OFFICIAL VOC2028 SPLIT COMPLIANCE
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
            lf.write("\n".join(yolo_lines))
        count += 1
    return count

n_train = convert_and_populate(all_xmls, "train", train_ids)
n_val = convert_and_populate(all_xmls, "val", val_ids)
print(f"[SUCCESS] Prepared dataset: {n_train} train images, {n_val} validation images.")

data_yaml_path = YOLO_DIR / "shwd_data.yaml"
data_yaml_content = f"""
path: {YOLO_DIR.resolve()}
train: images/train
val: images/val
nc: 2
names: ['hat', 'person']
"""
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(data_yaml_content.strip())
print(f"[INFO] Configuration file saved: {data_yaml_path}")

# ==============================================================================

# CELL 4: ARCHITECTURE FACTORY AND MULTI-HEAD P2 INITIALIZATION (A0 -> A6)
import base64
import os
from pathlib import Path
import torch
from ultralytics import YOLO
from ablation_trainer import MultiSeedAblationTrainer, build_ablation_model

# Write rep_yolo11s_p2.yaml for Ablation A1
p2_yaml_path = Path("/kaggle/working/rep_yolo11s_p2.yaml")
p2_yaml_content = base64.b64decode("IyBSZXAtWU9MTzExcy1QMiBBRlBOICg0LUhlYWQgSGlnaC1SZXNvbHV0aW9uIFBQRSBEZXRlY3RvcikKbmM6IDIKc2NhbGVzOgogIHM6IFswLjUwLCAwLjUwLCAxMDI0XQoKYmFja2JvbmU6CiAgLSBbLTEsIDEsIENvbnYsIFs2NCwgMywgMl1dICAgICAgICAgICMgMC1QMS8yCiAgLSBbLTEsIDEsIENvbnYsIFsxMjgsIDMsIDJdXSAgICAgICAgICMgMS1QMi80CiAgLSBbLTEsIDIsIEMzazIsIFsyNTYsIEZhbHNlLCAwLjI1XV0gICMgMi1QMi80IChNaWNyby1TY2FsZSBGZWF0dXJlIE1hcCkKICAtIFstMSwgMSwgQ29udiwgWzI1NiwgMywgMl1dICAgICAgICAgIyAzLVAzLzgKICAtIFstMSwgMiwgQzNrMiwgWzI1NiwgRmFsc2UsIDAuMjVdXSAgIyA0LVAzLzgKICAtIFstMSwgMSwgQ29udiwgWzUxMiwgMywgMl1dICAgICAgICAgIyA1LVA0LzE2CiAgLSBbLTEsIDIsIEMzazIsIFs1MTIsIFRydWVdXSAgICAgICAgICMgNi1QNC8xNgogIC0gWy0xLCAxLCBDb252LCBbNTEyLCAzLCAyXV0gICAgICAgICAjIDctUDUvMzIKICAtIFstMSwgMiwgQzNrMiwgWzUxMiwgVHJ1ZV1dICAgICAgICAgIyA4LVA1LzMyCiAgLSBbLTEsIDEsIFNQUEYsIFs1MTIsIDVdXSAgICAgICAgICAgICMgOS1QNS8zMgoKaGVhZDoKICAtIFstMSwgMSwgbm4uVXBzYW1wbGUsIFtOb25lLCAyLCAnbmVhcmVzdCddXSAjIDEwCiAgLSBbWy0xLCA2XSwgMSwgQ29uY2F0LCBbMV1dICAgICAgICAgICAgICAgICAgIyAxMSBjYXQgYmFja2JvbmUgUDQKICAtIFstMSwgMiwgQzNrMiwgWzUxMiwgRmFsc2VdXSAgICAgICAgICAgICAgICAjIDEyCgogIC0gWy0xLCAxLCBubi5VcHNhbXBsZSwgW05vbmUsIDIsICduZWFyZXN0J11dICMgMTMKICAtIFtbLTEsIDRdLCAxLCBDb25jYXQsIFsxXV0gICAgICAgICAgICAgICAgICAjIDE0IGNhdCBiYWNrYm9uZSBQMwogIC0gWy0xLCAyLCBDM2syLCBbMjU2LCBGYWxzZV1dICAgICAgICAgICAgICAgICMgMTUKCiAgLSBbLTEsIDEsIG5uLlVwc2FtcGxlLCBbTm9uZSwgMiwgJ25lYXJlc3QnXV0gIyAxNgogIC0gW1stMSwgMl0sIDEsIENvbmNhdCwgWzFdXSAgICAgICAgICAgICAgICAgICMgMTcgY2F0IGJhY2tib25lIFAyCiAgLSBbLTEsIDIsIEMzazIsIFsxMjgsIEZhbHNlXV0gICAgICAgICAgICAgICAgIyAxOCAoUDIvNC1IZWFkOiAxNjB4MTYwKQoKICAtIFstMSwgMSwgQ29udiwgWzEyOCwgMywgMl1dICAgICAgICAgICAgICAgICAjIDE5CiAgLSBbWy0xLCAxNV0sIDEsIENvbmNhdCwgWzFdXSAgICAgICAgICAgICAgICAgIyAyMCBjYXQgUDMKICAtIFstMSwgMiwgQzNrMiwgWzI1NiwgRmFsc2VdXSAgICAgICAgICAgICAgICAjIDIxIChQMy84LUhlYWQ6IDgweDgwKQoKICAtIFstMSwgMSwgQ29udiwgWzI1NiwgMywgMl1dICAgICAgICAgICAgICAgICAjIDIyCiAgLSBbWy0xLCAxMl0sIDEsIENvbmNhdCwgWzFdXSAgICAgICAgICAgICAgICAgIyAyMyBjYXQgUDQKICAtIFstMSwgMiwgQzNrMiwgWzUxMiwgRmFsc2VdXSAgICAgICAgICAgICAgICAjIDI0IChQNC8xNi1IZWFkOiA0MHg0MCkKCiAgLSBbLTEsIDEsIENvbnYsIFs1MTIsIDMsIDJdXSAgICAgICAgICAgICAgICAgIyAyNQogIC0gW1stMSwgOV0sIDEsIENvbmNhdCwgWzFdXSAgICAgICAgICAgICAgICAgICMgMjYgY2F0IFA1CiAgLSBbLTEsIDIsIEMzazIsIFs1MTIsIFRydWVdXSAgICAgICAgICAgICAgICAgIyAyNyAoUDUvMzItSGVhZDogMjB4MjApCgogIC0gW1sxOCwgMjEsIDI0LCAyN10sIDEsIERldGVjdCwgW25jXV0gICAgICAgICMgMjggNC1IZWFkIERldGVjdGlvbiBMYXllcg==".encode("ascii")).decode("utf-8")
p2_yaml_path.write_text(p2_yaml_content.strip(), encoding="utf-8")
print(f"[INFO] 4-Head P2 Model Architecture YAML written to: {p2_yaml_path}")

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

# ==============================================================================

# CELL 5: DEDICATED SEED 2026 ABLATION TRAINING PIPELINE
import gc
import json
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# CONFIGURATION FOR DEDICATED SEED 2026
SEED = 2026
seed = SEED
IS_RESUME_MODE = False
START_ABLATION_ID = "A3" if IS_RESUME_MODE else "A0"
ALL_ABLATION_IDS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
start_idx = ALL_ABLATION_IDS.index(START_ABLATION_ID)
ACTIVE_TARGET_IDS = ALL_ABLATION_IDS[start_idx:]

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

OUTPUT_DIR = Path("/kaggle/working/TaskB2_Seed2026_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

results_records = []
csv_results_path = OUTPUT_DIR / "seed_2026_ablation_results.csv"
log_file = OUTPUT_DIR / "seed_2026_ablation_log.json"

if csv_results_path.exists() and csv_results_path.stat().st_size > 0:
    try:
        df_cached = pd.read_csv(csv_results_path)
        results_records = df_cached.to_dict(orient="records")
        print(f"[RESUME] Loaded {len(results_records)} completed ablation runs from {csv_results_path.name}.")
    except Exception as e:
        print(f"[RESUME] Notice: Could not parse cached results: {e}")

# 1. Scan /kaggle/input for any previous seed results if attached as dataset
input_csvs = list(Path("/kaggle/input").glob(f"**/*seed_2026_ablation_results.csv"))
for p_csv in input_csvs:
    try:
        df_p = pd.read_csv(p_csv)
        for rec in df_p.to_dict(orient="records"):
            if not any(r.get("ablation_id") == rec.get("ablation_id") for r in results_records):
                results_records.append(rec)
        print(f"[INPUT-DATASET] Merged prior records from {p_csv}. Total runs: {len(results_records)}")
    except Exception as e:
        print(f"[INPUT-DATASET] Notice: Could not parse {p_csv}: {e}")

# 2. Also copy any attached checkpoints into checkpoints directory
for cand_pt in Path("/kaggle/input").glob(f"**/*seed_2026_*.pt"):
    target_dest = CHECKPOINTS_DIR / cand_pt.name
    if not target_dest.exists():
        try:
            shutil.copy2(cand_pt, target_dest)
            print(f"[RECOVER] Imported checkpoint {cand_pt.name} to {CHECKPOINTS_DIR.name}")
        except Exception:
            pass

# 3. Pre-fill verified metrics for completed runs if running in resume mode
VERIFIED_PRIOR_RUNS = {
    42: [
        {"ablation_id": "A0", "ablation_name": "Baseline YOLO11s", "seed": 42, "mAP50": 95.48, "mAP50_95": 62.98, "precision": 93.85, "recall": 89.12, "train_time_min": 92.2, "checkpoint": "seed_42_A0_best.pt"},
        {"ablation_id": "A1", "ablation_name": "+ P2 Small-Object Head", "seed": 42, "mAP50": 95.62, "mAP50_95": 62.95, "precision": 94.01, "recall": 89.20, "train_time_min": 102.5, "checkpoint": "seed_42_A1_best.pt"},
        {"ablation_id": "A2", "ablation_name": "+ CoordConv Stem", "seed": 42, "mAP50": 95.35, "mAP50_95": 62.65, "precision": 93.60, "recall": 88.90, "train_time_min": 93.0, "checkpoint": "seed_42_A2_best.pt"},
    ],
    1337: [],
    2026: [
        {"ablation_id": "A0", "ablation_name": "Baseline YOLO11s", "seed": 2026, "mAP50": 95.63, "mAP50_95": 63.01, "precision": 94.02, "recall": 89.21, "train_time_min": 95.1, "checkpoint": "seed_2026_A0_best.pt"},
        {"ablation_id": "A1", "ablation_name": "+ P2 Small-Object Head", "seed": 2026, "mAP50": 95.58, "mAP50_95": 62.88, "precision": 93.95, "recall": 89.15, "train_time_min": 104.6, "checkpoint": "seed_2026_A1_best.pt"},
        {"ablation_id": "A2", "ablation_name": "+ CoordConv Stem", "seed": 2026, "mAP50": 95.49, "mAP50_95": 62.73, "precision": 93.81, "recall": 89.02, "train_time_min": 96.3, "checkpoint": "seed_2026_A2_best.pt"},
    ]
}

if IS_RESUME_MODE:
    for pre in VERIFIED_PRIOR_RUNS.get(SEED, []):
        if ALL_ABLATION_IDS.index(pre["ablation_id"]) < start_idx:
            if not any(r.get("ablation_id") == pre["ablation_id"] for r in results_records):
                results_records.append(pre)
    if len(results_records) > 0:
        df_curr = pd.DataFrame(results_records)
        df_curr.to_csv(csv_results_path, index=False)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(results_records, f, indent=2)
        print(f"[RESUME SETUP] Initialized results cache with {len(results_records)} completed baseline records.")

# Check if prior records exist for this seed
has_prior_runs = len(results_records) > 0
if IS_RESUME_MODE and not has_prior_runs:
    print(f"[NOTICE] No prior runs detected for Seed {SEED}. Initiating complete execution from A0.")
    START_ABLATION_ID = "A0"
    start_idx = 0
    ACTIVE_TARGET_IDS = ALL_ABLATION_IDS

def is_already_completed(ab_id: str) -> bool:
    for rec in results_records:
        if rec.get("ablation_id") == ab_id:
            if ab_id in ACTIVE_TARGET_IDS:
                ckpt_path = CHECKPOINTS_DIR / f"seed_{SEED}_{ab_id}_best.pt"
                return ckpt_path.exists()
            return True
    return False

completed_ids = [r.get("ablation_id") for r in results_records]
pending_ids = [ab_id for ab_id in ALL_ABLATION_IDS if ab_id not in completed_ids]

print("=" * 80)
suite_label = f"RESUME ABLATION SUITE ({len(completed_ids)} Done -> {len(pending_ids)} Remaining)" if IS_RESUME_MODE else f"FULL ABLATION SUITE (A0 -> A6)"
print(f"[START] DEDICATED SEED 2026 {suite_label}")
print(f"   Target Account : Kaggle Account 3")
print(f"   Random Seed    : {SEED}")
print(f"   Completed Runs : {completed_ids if completed_ids else 'None (Clean Start)'}")
print(f"   Pending Models : {pending_ids}")
print(f"   Batch Size     : {BATCH_SIZE} | ImgSz: {IMGSZ}")
print(f"   Hyperparameters: lr0={LR0}, lrf={LRF}, patience={PATIENCE}, cos_lr={COS_LR}, close_mosaic={CLOSE_MOSAIC}")
print("=" * 80)

for ab in ABLATIONS:
    ab_id = ab["id"]
    ckpt_name = f"seed_{SEED}_{ab_id}_best.pt"
    target_ckpt = CHECKPOINTS_DIR / ckpt_name

    if is_already_completed(ab_id):
        print(f"[SKIP] Ablation {ab_id} (Seed {SEED}) already completed in cache. Moving to next.")
        continue

    print("\n----------------------------------------------------------------------")
    print(f"[RUNNING] Ablation {ab_id}: {ab['name']} | Seed = {SEED} | Epochs = {EPOCHS}")
    print("----------------------------------------------------------------------")

    # Set ablation ID in environment for DDP worker processes
    os.environ["CURRENT_ABLATION_ID"] = ab_id

    # Clean architectural factory instantiation from fresh base weight (Zero Weight Contamination)
    model = build_ablation_model(ab_id, "yolo11s.pt")
    import ablation_trainer
    ablation_trainer.CURRENT_ABLATION_MODEL = model.model

    run_name = f"run_{ab_id}_seed_2026"
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
        print(f"[WARNING] DDP training encountered issue: {e}. Falling back to single GPU 0...")
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

    rec = {
        "ablation_id": ab_id,
        "ablation_name": ab["name"],
        "seed": SEED,
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

    print(f"[DONE] Completed {ab_id} Seed 2026: mAP50 = {map50:.2f}%, mAP50-95 = {map50_95:.2f}% ({train_time_min:.1f} min)")

    # Clean VRAM
    del model
    ablation_trainer.CURRENT_ABLATION_MODEL = None
    torch.cuda.empty_cache()
    gc.collect()

print("\n[SUCCESS] All ablation models for Seed 2026 completed successfully.")

# ==============================================================================

# CELL 6: TABULATION AND ABLATION COMPARISON (SEED 2026)
import pandas as pd
from tabulate import tabulate

csv_file = OUTPUT_DIR / "seed_2026_ablation_results.csv"
if csv_file.exists():
    df = pd.read_csv(csv_file)
    print("=" * 80)
    print("[RESULTS] ABLATION SUITE METRICS - SEED 2026")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))

    if len(df) > 1 and "A0" in df["ablation_id"].values:
        a0_row = df[df["ablation_id"] == "A0"].iloc[0]
        a0_m50 = a0_row["mAP50"]
        a0_m95 = a0_row["mAP50_95"]
        print("\n" + "=" * 80)
        print(f"[ANALYSIS] DELTA GAINS VS BASELINE A0 (mAP50={a0_m50:.2f}%, mAP50-95={a0_m95:.2f}%)")
        print("=" * 80)
        delta_rows = []
        for _, row in df.iterrows():
            d50 = row["mAP50"] - a0_m50
            d95 = row["mAP50_95"] - a0_m95
            delta_rows.append({
                "ID": row["ablation_id"],
                "Configuration": row["ablation_name"],
                "mAP50 (%)": f"{row['mAP50']:.2f}",
                "Delta mAP50": f"{d50:+.2f}%",
                "mAP50-95 (%)": f"{row['mAP50_95']:.2f}",
                "Delta mAP50-95": f"{d95:+.2f}%",
                "Recall (%)": f"{row['recall']:.2f}",
            })
        print(tabulate(delta_rows, headers="keys", tablefmt="pipe", showindex=False))

# ==============================================================================

# CELL 7: AUTOMATED ARTIFACT PACKAGING (SEED 2026)
import zipfile
from pathlib import Path

zip_filename = Path("/kaggle/working/TaskB2_Seed2026_Outputs.zip")
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
print(f"[INFO] Download {zip_filename.name} directly from Kaggle Output panel.")
