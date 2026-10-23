#!/usr/bin/env python3
"""
scripts/upload_dataset_to_kaggle.py
===================================
Uploads the complete unified dataset (Dataset/UNIFIED_IITU_CHV_COMPLETE)
to Kaggle as a public dataset: hannhu4002/unified-iitu-chv-ppe-complete.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "Dataset" / "UNIFIED_IITU_CHV_COMPLETE"

def main():
    print("=" * 80)
    print("🚀 PACKAGING AND UPLOADING DATASET TO KAGGLE")
    print("=" * 80)
    
    if not DATASET_DIR.exists():
        print(f"❌ Error: Dataset directory {DATASET_DIR} does not exist!")
        return False

    # Check images counts
    n_train = len(list((DATASET_DIR / "images" / "train").glob("*")))
    n_val = len(list((DATASET_DIR / "images" / "val").glob("*")))
    n_test = len(list((DATASET_DIR / "images" / "test").glob("*")))
    total_imgs = n_train + n_val + n_test
    print(f"📊 Dataset verification: Total {total_imgs} images (Train={n_train}, Val={n_val}, Test={n_test})")

    # Create dataset-metadata.json
    metadata = {
        "title": "Unified IITU CHV PPE Complete 7-Class Dataset",
        "id": "hannhu4002/unified-iitu-chv-ppe-complete",
        "licenses": [
            {
                "name": "CC-BY-4.0"
            }
        ]
    }
    meta_path = DATASET_DIR / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"✅ Created metadata at {meta_path}")

    # Check if dataset already exists on Kaggle
    print("\nChecking if dataset already exists on Kaggle...")
    chk = subprocess.run(["kaggle", "datasets", "status", "hannhu4002/unified-iitu-chv-ppe-complete"], capture_output=True, text=True)
    
    if "ready" in chk.stdout.lower() or chk.returncode == 0:
        print("Dataset already exists on Kaggle. Creating new version...")
        cmd = ["kaggle", "datasets", "version", "-p", str(DATASET_DIR), "-m", "100% complete leak-free 7-class annotations with verified person and safety_vest boxes", "--dir-mode", "zip"]
    else:
        print("Creating new public dataset on Kaggle...")
        cmd = ["kaggle", "datasets", "create", "-p", str(DATASET_DIR), "--public", "--dir-mode", "zip"]

    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"STDOUT: {res.stdout}")
    print(f"STDERR: {res.stderr}")

    if res.returncode == 0:
        print("✅ Dataset upload initiated successfully!")
        # Poll status
        for _ in range(30):
            time.sleep(10)
            st = subprocess.run(["kaggle", "datasets", "status", "hannhu4002/unified-iitu-chv-ppe-complete"], capture_output=True, text=True)
            print(f"Current status: {st.stdout.strip()}")
            if "ready" in st.stdout.lower():
                print("🟢 Dataset is ready on Kaggle!")
                return True
        return True
    else:
        print("❌ Dataset upload failed!")
        return False

if __name__ == "__main__":
    main()
