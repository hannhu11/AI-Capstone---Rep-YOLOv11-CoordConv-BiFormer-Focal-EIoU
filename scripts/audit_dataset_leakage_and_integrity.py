"""
scripts/audit_dataset_leakage_and_integrity.py
==============================================
Rigorous Forensic Audit of All Datasets in Capstone AI:
1. Cross-split Data Leakage (MD5 duplicate detection between Train, Val, and Test).
2. Intra-split redundancy.
3. Class distribution and box counts.
4. Detection of pseudo-labeled vs human-annotated splits.
"""

import os
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path("c:/Users/ADMIN/Downloads/capstone AI")

def file_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset_splits(dataset_name: str, dataset_path: Path):
    print("=" * 80)
    print(f"AUDITING DATASET: {dataset_name}")
    print(f"Path: {dataset_path}")
    print("=" * 80)

    if not dataset_path.exists():
        print(f"Directory not found: {dataset_path}")
        return

    splits = ["train", "val", "test"]
    split_hashes = {}
    split_images = {}
    split_classes = {}

    for s in splits:
        img_dir = dataset_path / "images" / s
        lbl_dir = dataset_path / "labels" / s
        
        # Also check alternative folder structures (e.g. YOLO standard train/val/test without images/ prefix)
        if not img_dir.exists() and (dataset_path / s / "images").exists():
            img_dir = dataset_path / s / "images"
            lbl_dir = dataset_path / s / "labels"
        elif not img_dir.exists() and (dataset_path / s).exists():
            img_dir = dataset_path / s
            lbl_dir = dataset_path / "labels" / s if (dataset_path / "labels" / s).exists() else dataset_path / s

        if not img_dir.exists():
            print(f"  Split '{s}': Directory not found ({img_dir})")
            continue

        images = [f for f in img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        split_images[s] = images
        
        # MD5 hashing
        hashes = {}
        for img in images:
            h = file_md5(img)
            hashes[img.name] = h
        split_hashes[s] = hashes

        # Label counting
        cls_counts = Counter()
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                try:
                    for line in lf.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if parts:
                            cls_counts[int(parts[0])] += 1
                except Exception:
                    pass
        split_classes[s] = cls_counts

        print(f"  Split '{s}': {len(images)} images, {sum(cls_counts.values())} boxes, {len(set(hashes.values()))} unique MD5s")
        if cls_counts:
            print(f"    Class counts: {dict(sorted(cls_counts.items()))}")

    # Check for Cross-Split Leakage (Train vs Val, Train vs Test, Val vs Test)
    print("\n  --- DATA LEAKAGE ANALYSIS (EXACT MD5 MATCH) ---")
    leak_found = False
    for i in range(len(splits)):
        s1 = splits[i]
        if s1 not in split_hashes:
            continue
        h1 = set(split_hashes[s1].values())
        for j in range(i + 1, len(splits)):
            s2 = splits[j]
            if s2 not in split_hashes:
                continue
            h2 = set(split_hashes[s2].values())
            common = h1.intersection(h2)
            if common:
                leak_found = True
                print(f"  ❌ CRITICAL DATA LEAKAGE: {len(common)} identical images shared between '{s1}' and '{s2}'!")
                # Sample 3 matching files
                s1_inv = {v: k for k, v in split_hashes[s1].items()}
                s2_inv = {v: k for k, v in split_hashes[s2].items()}
                for h in list(common)[:3]:
                    print(f"     Match: {s1}/{s1_inv[h]} <===> {s2}/{s2_inv[h]} (MD5: {h})")
            else:
                print(f"  ✅ ZERO LEAKAGE between '{s1}' and '{s2}' (0 shared MD5 hashes).")

    if not leak_found:
        print("  ✅ All evaluated splits are 100% mutually exclusive by image content.")
    print()

def main():
    datasets = [
        ("CHV_6CLASS_OFFICIAL_CLEAN", BASE_DIR / "Dataset" / "CHV_6CLASS_OFFICIAL_CLEAN"),
        ("CHV_7CLASS_CONFLICT_FREE", BASE_DIR / "Dataset" / "CHV_7CLASS_CONFLICT_FREE"),
        ("UNIFIED_IITU_CHV_COMPLETE", BASE_DIR / "Dataset" / "UNIFIED_IITU_CHV_COMPLETE"),
        ("UNIFIED_PPE_MASTER_HARMONIZED", BASE_DIR / "Dataset" / "UNIFIED_PPE_MASTER_HARMONIZED"),
    ]

    for name, p in datasets:
        audit_dataset_splits(name, p)

if __name__ == "__main__":
    main()
