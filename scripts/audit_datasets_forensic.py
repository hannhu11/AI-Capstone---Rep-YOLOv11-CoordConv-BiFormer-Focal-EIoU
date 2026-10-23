import os
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path("c:/Users/ADMIN/Downloads/capstone AI")

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    print("=" * 80)
    print("FORENSIC DATASET AUDIT")
    print("=" * 80)

    # 1. Inspect CHV_6CLASS_OFFICIAL_CLEAN
    p6 = BASE_DIR / "Dataset" / "CHV_6CLASS_OFFICIAL_CLEAN"
    if p6.exists():
        print("\n--- CHV_6CLASS_OFFICIAL_CLEAN ---")
        for s in ["train", "val", "test"]:
            imgs = list((p6 / "images" / s).glob("*.*"))
            c = Counter()
            for f in (p6 / "labels" / s).glob("*.txt"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        c[int(line.split()[0])] += 1
            print(f"  {s}: {len(imgs)} imgs, bboxes: {dict(sorted(c.items()))}")

    # 2. Inspect CHV_7CLASS_CONFLICT_FREE
    p7 = BASE_DIR / "Dataset" / "CHV_7CLASS_CONFLICT_FREE"
    if p7.exists():
        print("\n--- CHV_7CLASS_CONFLICT_FREE ---")
        for s in ["train", "val", "test"]:
            imgs = list((p7 / "images" / s).glob("*.*"))
            c = Counter()
            for f in (p7 / "labels" / s).glob("*.txt"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        c[int(line.split()[0])] += 1
            print(f"  {s}: {len(imgs)} imgs, bboxes: {dict(sorted(c.items()))}")

    # 3. Inspect UNIFIED_PPE_MASTER_HARMONIZED (Bo4)
    p_bo4 = BASE_DIR / "Dataset" / "UNIFIED_PPE_MASTER_HARMONIZED"
    if p_bo4.exists():
        print("\n--- UNIFIED_PPE_MASTER_HARMONIZED (Bo4) ---")
        for s in ["train", "val", "test"]:
            imgs = list((p_bo4 / "images" / s).glob("*.*"))
            c = Counter()
            for f in (p_bo4 / "labels" / s).glob("*.txt"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        c[int(line.split()[0])] += 1
            print(f"  {s}: {len(imgs)} imgs, bboxes: {dict(sorted(c.items()))}")

    # 4. Check GDUT-HWD
    p_gdut = BASE_DIR / "Dataset" / "GDUT-HWD"
    if p_gdut.exists():
        print("\n--- GDUT-HWD ---")
        imgs = list((p_gdut / "JPEGImages").glob("*.*"))
        xmls = list((p_gdut / "Annotations").glob("*.xml"))
        print(f"  Images: {len(imgs)}, XML annotations: {len(xmls)}")

    # 5. Check IITU
    p_iitu = BASE_DIR / "Dataset" / "IITU Safety-Helmet"
    if p_iitu.exists():
        print("\n--- IITU Safety-Helmet ---")
        imgs = list((p_iitu / "images").glob("*.*")) if (p_iitu / "images").exists() else []
        print(f"  Images: {len(imgs)}")

    # 6. Check CHV_dataset
    p_chv = BASE_DIR / "Dataset" / "CHV_dataset"
    if p_chv.exists():
        print("\n--- CHV_dataset (Raw) ---")
        imgs = list((p_chv / "images").glob("*.*"))
        lbls = list((p_chv / "annotations").glob("*.txt"))
        print(f"  Images: {len(imgs)}, Labels: {len(lbls)}")

if __name__ == "__main__":
    main()
