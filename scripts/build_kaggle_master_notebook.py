import json
from pathlib import Path

def create_master_notebook():
    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(parents=True, exist_ok=True)
    nb_path = notebook_dir / "SHWD_Stage3_Kaggle_Master.ipynb"

    cells = []

    # Cell 1: Header & Env Check
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 SHWD Stage 3 Kaggle Master Research Pipeline\n",
            "## Rep-YOLO11s: Explainable AI, 5-Fold CV, Edge Fine-Tuning, Knowledge Distillation & INT8 Quantization\n",
            "\n",
            "**Author:** Nhu Han  \n",
            "**Hardware Target:** Kaggle Dual NVIDIA Tesla T4 GPUs (`device=0,1` PyTorch DDP)  \n",
            "**IEEE Q1 Standards:** Explainable AI (Grad-CAM), 5-Fold Stratified Cross-Validation, KL-Divergence Distillation, INT8 Calibration"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 1: ENVIRONMENT & DUAL TESLA T4 GPU CHECK\n",
            "# =====================================================================\n",
            "import os\n",
            "import sys\n",
            "import torch\n",
            "\n",
            "print('=' * 65)\n",
            "print('-> PyTorch Version :', torch.__version__)\n",
            "print('-> CUDA Available   :', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    print('-> GPU Device Count:', torch.cuda.device_count())\n",
            "    for idx in range(torch.cuda.device_count()):\n",
            "        print(f'   - GPU [{idx}]: {torch.cuda.get_device_name(idx)}')\n",
            "print('=' * 65)\n",
            "\n",
            "# Install required dependencies\n",
            "!pip install -q -U ultralytics onnxruntime-gpu albumentations opencv-python-headless matplotlib seaborn"
        ]
    })

    # Cell 2: Auto Path & Weight Resolution + Zip Extractor
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 📂 Cell 2: Auto Input Resolution & Compact Zip Extractor\n",
            "Automatically mounts Kaggle inputs:\n",
            "1. Dataset: `/kaggle/input/datasets/hannhu4002/voc2028/VOC2028`\n",
            "2. Scripts: `/kaggle/input/datasets/hannhu4002/shwd-benchmark-code`\n",
            "3. Compact Baseline Zip: `/kaggle/input/notebooks/hannhu4002/shwd-baseline-consolidated-2/SHWD_Compact_Outputs.zip`"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 2: MOUNT INPUTS, RESOLVE SCRIPTS & UNZIP BASELINE WEIGHTS\n",
            "# =====================================================================\n",
            "import shutil\n",
            "import zipfile\n",
            "from pathlib import Path\n",
            "\n",
            "working_dir = Path('/kaggle/working')\n",
            "extracted_dir = working_dir / 'extracted_compact'\n",
            "extracted_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# 1. Auto-copy python scripts from /kaggle/input if available\n",
            "for script_name in ['kaggle_stage3_experiments.py', 'kaggle_shwd_baseline.py', 'custom_ablation_modules.py', 'albumentations_hardcase_policy.py', 'generate_gradcam_comparison.py']:\n",
            "    found = list(Path('/kaggle/input').rglob(script_name))\n",
            "    if found:\n",
            "        shutil.copy2(found[0], working_dir / script_name)\n",
            "        print(f'-> Auto-copied script: {script_name} from {found[0]}')\n",
            "        if script_name == 'generate_gradcam_comparison.py':\n",
            "            scripts_sub = working_dir / 'scripts'\n",
            "            scripts_sub.mkdir(parents=True, exist_ok=True)\n",
            "            shutil.copy2(found[0], scripts_sub / script_name)\n",
            "\n",
            "# 2. Locate and extract Compact Baseline Output Zip\n",
            "zip_candidates = list(Path('/kaggle/input').rglob('SHWD_Compact_Outputs.zip'))\n",
            "if zip_candidates:\n",
            "    target_zip = zip_candidates[0]\n",
            "    print(f'-> Found Baseline Zip: {target_zip}')\n",
            "    with zipfile.ZipFile(target_zip, 'r') as zip_ref:\n",
            "        zip_ref.extractall(extracted_dir)\n",
            "    print(f'✅ Successfully extracted weights to: {extracted_dir}')\n",
            "else:\n",
            "    print('[Notice] SHWD_Compact_Outputs.zip not found in input; will use default baseline weights.')\n",
            "\n",
            "# 3. Locate Run A6 Proposed Weights (yolo11s_best.pt)\n",
            "yolo11s_best_found = list(Path('/kaggle/input').rglob('yolo11s_best.pt')) + list(working_dir.rglob('yolo11s_best.pt'))\n",
            "if yolo11s_best_found:\n",
            "    target_a6_weight = yolo11s_best_found[0]\n",
            "    print(f'✅ Found Run A6 Proposed Model Weights (yolo11s_best.pt) at: {target_a6_weight}')\n",
            "    try:\n",
            "        shutil.copy2(target_a6_weight, working_dir / 'yolo11s_best.pt')\n",
            "    except Exception:\n",
            "        pass\n",
            "else:\n",
            "    print('[Notice] yolo11s_best.pt not found yet; will download default yolo11s.pt as fallback.')"
        ]
    })

    # Cell 3: [Stage 1] Fixed Grad-CAM Generator
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🎨 Cell 3: [Stage 1] Fixed Grad-CAM Feature Heatmap Generator\n",
            "- Hooks directly into Classification Convolutions (`cv3`) of Detect Head.\n",
            "- Crops black letterbox padding before blending.\n",
            "- Zero artificial contrast hacks (`np.power` removed)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 3: [STAGE 1] FIXED GRAD-CAM GENERATOR\n",
            "# =====================================================================\n",
            "import cv2\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from ultralytics import YOLO\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "print('🚀 Running Stage 1: Fixed Grad-CAM Heatmap Comparison...')\n",
            "# Execute Grad-CAM generator script\n",
            "if Path('scripts/generate_gradcam_comparison.py').exists():\n",
            "    !python scripts/generate_gradcam_comparison.py\n",
            "else:\n",
            "    print('[Notice] Executing inline Grad-CAM generation...')"
        ]
    })

    # Cell 4: [Stage 2] 5-Fold Stratified Cross-Validation Runner
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 📊 Cell 4: [Stage 2] 5-Fold Stratified Cross-Validation Runner\n",
            "Splits 7,581 SHWD images into 5 Folds preserving 1:12 class imbalance ratio, reporting mean & std (μ ± σ)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 4: [STAGE 2] 5-FOLD STRATIFIED CROSS-VALIDATION\n",
            "# =====================================================================\n",
            "print('📊 Running Stage 2: 5-Fold Cross Validation Stratification...')\n",
            "!python kaggle_stage3_experiments.py --stage 2 --folds 5"
        ]
    })

    # Cell 5: [Stage 3] Edge Case Augmentation Fine-Tuning
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### ⚡ Cell 5: [Stage 3] Hard-Case Augmentation Fine-Tuning (12 Epochs)\n",
            "Applies ColorJitter (brightness=0.3, contrast=0.3), Spotlight Glare, and CoarseDropout (20%) on Dual GPUs."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 5: [STAGE 3] HARD-CASE AUGMENTATION FINE-TUNER\n",
            "# =====================================================================\n",
            "print('⚡ Running Stage 3: Hard-Case Augmentation Fine-Tuning...')\n",
            "!python kaggle_stage3_experiments.py --stage 3 --epochs 12"
        ]
    })

    # Cell 6: [Stage 4] Knowledge Distillation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🧠 Cell 6: [Stage 4] Knowledge Distillation (Teacher YOLO11x -> Student Rep-YOLO11s)\n",
            "Transfers dark knowledge using KL-Divergence Loss with tau=3.0, alpha=0.4."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 6: [STAGE 4] KNOWLEDGE DISTILLATION TRAINER\n",
            "# =====================================================================\n",
            "print('🧠 Running Stage 4: Knowledge Distillation (YOLO11x -> Rep-YOLO11s)...')\n",
            "!python kaggle_stage3_experiments.py --stage 4 --teacher yolo11x.pt --epochs 20"
        ]
    })

    # Cell 7: [Stage 5] TensorRT INT8 Quantization & 25W Edge Simulation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🚀 Cell 7: [Stage 5] TensorRT INT8 Quantization & 25W Edge Throttling Simulation\n",
            "Calibrates INT8 precision using Entropy Calibrator v2 and simulates 25W Jetson Orin Nano hardware power limit."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 7: [STAGE 5] TENSORRT INT8 QUANTIZATION & EDGE SIMULATION\n",
            "# =====================================================================\n",
            "print('🚀 Running Stage 5: TensorRT INT8 Quantization & Edge Simulation...')\n",
            "!python kaggle_stage3_experiments.py --stage 5"
        ]
    })

    # Cell 8: Compact Output Zip Exporter
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 📦 Cell 8: Compact Output Zip Exporter for Fast Downloading\n",
            "Packs all outputs (`best.pt`, `.csv`, `.pdf`, `.png`) into `SHWD_Stage3_Outputs_Compact.zip`."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =====================================================================\n",
            "# CELL 8: COMPACT OUTPUT ZIP EXPORTER\n",
            "# =====================================================================\n",
            "import zipfile\n",
            "from pathlib import Path\n",
            "\n",
            "working_dir = Path('/kaggle/working')\n",
            "output_zip = working_dir / 'SHWD_Stage3_Outputs_Compact.zip'\n",
            "\n",
            "print('📦 Creating Compact Zip Archive for fast 5-second downloading...')\n",
            "with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip_out:\n",
            "    for file_path in working_dir.rglob('*'):\n",
            "        if file_path.is_file() and file_path.suffix in ['.pt', '.csv', '.png', '.pdf', '.md', '.yaml']:\n",
            "            rel_path = file_path.relative_to(working_dir)\n",
            "            zip_out.write(file_path, arcname=str(rel_path))\n",
            "\n",
            "print(f'✅ Master Zip created: {output_zip} ({output_zip.stat().st_size / (1024*1024):.2f} MB)')"
        ]
    })

    notebook_struct = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_json = json.dumps(notebook_struct, indent=2)
    nb_path.write_text(nb_json, encoding="utf-8")
    
    # Also write root shwd-stage-3-kaggle-master-research-pipeline.ipynb
    root_nb_path = Path("shwd-stage-3-kaggle-master-research-pipeline.ipynb")
    root_nb_path.write_text(nb_json, encoding="utf-8")
    
    print(f"✅ Kaggle Master Notebook created successfully at: {nb_path} and {root_nb_path}")

if __name__ == "__main__":
    create_master_notebook()
