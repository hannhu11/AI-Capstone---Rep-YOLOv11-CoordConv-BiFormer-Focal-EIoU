#!/usr/bin/env python3
"""
scripts/upload_unified_dataset_to_kaggle.py
============================================
Creates a zip archive of Dataset/UNIFIED_IITU_CHV_COMPLETE and uploads it
to Kaggle as a PUBLIC dataset under @hannhu4002 via Kaggle REST API with Bearer token authentication.
"""

import os
import sys
import time
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path("Dataset/UNIFIED_IITU_CHV_COMPLETE").resolve()
ZIP_PATH = Path("Dataset/unified_iitu_chv_ppe_complete.zip").resolve()

TOKEN = "KGAT_609b5046b49fec2fc48bc45ec30839b7"
OWNER = "hannhu4002"
SLUG = "unified-iitu-chv-ppe-clean-7class"
TITLE = "Unified IITU CHV PPE Clean 7-Class Dataset"

def make_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    
    print(f"Creating clean zip archive {ZIP_PATH} from {DATASET_DIR}...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in DATASET_DIR.rglob('*'):
            if f.is_file():
                # Strict exclusion of weights, checkpoints, caches, and temp files
                if f.suffix in ['.pt', '.tmp', '.cache']:
                    continue
                if 'weights' in f.parts or '__pycache__' in f.parts:
                    continue
                if f.name in ['dataset-metadata.json', 'data_local.yaml']:
                    continue
                rel_path = f.relative_to(DATASET_DIR)
                zf.write(f, arcname=str(rel_path))
    elapsed = time.time() - t0
    print(f"Created {ZIP_PATH} in {elapsed:.1f}s ({ZIP_PATH.stat().st_size / (1024**2):.2f} MB)")
    return ZIP_PATH

def upload_dataset():
    zip_file = make_zip()
    file_size = zip_file.stat().st_size
    file_name = zip_file.name

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    print("\n[Step 1] Requesting blob upload token from Kaggle...")
    blob_req = {
        "type": "dataset",
        "name": file_name,
        "contentLength": file_size,
        "lastModifiedEpochSeconds": int(zip_file.stat().st_mtime)
    }
    r = requests.post("https://www.kaggle.com/api/v1/blobs/upload", headers=headers, json=blob_req)
    if r.status_code != 200:
        print(f"Blob upload request failed: {r.status_code} {r.text}")
        return False

    blob_data = r.json()
    token = blob_data.get("token")
    create_url = blob_data.get("createUrl")
    print(f"Token acquired: {token[:30]}...")

    print(f"\n[Step 2] Uploading {file_size / (1024**2):.2f} MB to Google Cloud Storage...")
    class ProgressReader:
        def __init__(self, file_path, pbar):
            self.file_obj = open(file_path, "rb")
            self.pbar = pbar

        def read(self, size=-1):
            chunk = self.file_obj.read(size)
            if chunk:
                self.pbar.update(len(chunk))
            return chunk

        def close(self):
            self.file_obj.close()

    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Uploading") as pbar:
        reader = ProgressReader(zip_file, pbar)
        upload_resp = requests.put(create_url, data=reader)
        reader.close()

    print(f"\nUpload response status: {upload_resp.status_code}")
    if upload_resp.status_code not in [200, 201]:
        print(f"Upload failed: {upload_resp.text}")
        return False

    print("\n[Step 3] Registering PUBLIC dataset on Kaggle...")
    create_payload = {
        "title": TITLE,
        "slug": SLUG,
        "ownerSlug": OWNER,
        "licenseName": "CC0-1.0",
        "isPrivate": False,
        "files": [
            {
                "token": token,
                "description": "Unified 7-Class IITU+CHV PPE Dataset (2,987 images, 26,360 clean boxes, 0 duplicates, with README.md)"
            }
        ]
    }
    r_create = requests.post("https://www.kaggle.com/api/v1/datasets/create/new", headers=headers, json=create_payload)
    print("Dataset creation response:", r_create.status_code, r_create.text)
    if r_create.status_code == 200:
        data = r_create.json()
        if data.get("status") == "Error" and "already exists" in str(data.get("error")).lower():
            ver_payload = {
                "versionNotes": "Clean 7-class annotations with zero weights or temp files",
                "deleteOldVersions": True,
                "files": [{"token": token}]
            }
            r_ver = requests.post(f"https://www.kaggle.com/api/v1/datasets/create/version/{OWNER}/{SLUG}", headers=headers, json=ver_payload)
            print("Version response:", r_ver.status_code, r_ver.text)
        return True
    return False

if __name__ == "__main__":
    success = upload_dataset()
    sys.exit(0 if success else 1)

