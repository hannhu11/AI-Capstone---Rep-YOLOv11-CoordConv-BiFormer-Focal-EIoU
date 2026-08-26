# =====================================================================
# STANDARDIZE & ZERO-SHOT CROSS-DOMAIN BENCHMARK ENGINE
# Evaluates In-Domain (SHWD) vs. Cross-Domain (GDUT-HWD & SHEL5K)
# Project: Rep-YOLO11s-P2 AFPN (IEEE TPAMI / Q1 Target)
# =====================================================================
import os
import sys
import time
import zipfile
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

# Canonical Class Taxonomy
CANONICAL_CLASSES = {0: 'hat', 1: 'person'}

def standardize_gdut_hwd(base_dir: Path):
    """
    Standardizes GDUT-HWD dataset from Roboflow YOLO11 zip.
    Original classes: ['head', 'helmet', 'person']
    Canonical Mapping:
      - 'helmet' (1) -> 0 ('hat')
      - 'head'   (0) -> 1 ('person')
      - 'person' (2) -> ignored (full body overlap)
    """
    print("\n" + "=" * 70)
    print("🔄 STANDARDIZING GDUT-HWD DATASET (Roboflow YOLO Format)")
    print("=" * 70)

    zip_path = base_dir / "Dataset" / "GDUT-HWD" / "GDUT-HWD.v1i.yolov11.zip"
    out_dir = base_dir / "Dataset" / "STANDARDIZED" / "GDUT_HWD"
    
    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return None

    out_images_dir = out_dir / "images"
    out_labels_dir = out_dir / "labels"
    for split in ["train", "valid", "test"]:
        (out_images_dir / split).mkdir(parents=True, exist_ok=True)
        (out_labels_dir / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        for name in namelist:
            # Process label files
            if name.endswith('.txt') and not name.startswith('README') and not name.endswith('data.yaml'):
                split = "train" if "train" in name else ("valid" if "valid" in name else "test")
                raw_lines = z.read(name).decode('utf-8', errors='ignore').strip().split('\n')
                
                converted_lines = []
                for line in raw_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        raw_cls = int(parts[0])
                        coords = " ".join(parts[1:5])
                        
                        if raw_cls == 1:  # helmet -> 0 (hat)
                            converted_lines.append(f"0 {coords}")
                            stats["helmet_mapped_to_hat"] += 1
                        elif raw_cls == 0:  # head -> 1 (person)
                            converted_lines.append(f"1 {coords}")
                            stats["head_mapped_to_person"] += 1
                        elif raw_cls == 2:  # full person -> drop
                            stats["person_dropped"] += 1

                label_filename = Path(name).name
                out_label_path = out_labels_dir / split / label_filename
                out_label_path.write_text("\n".join(converted_lines), encoding='utf-8')

            # Process image files
            elif name.lower().endswith(('.jpg', '.jpeg', '.png')) and ('train' in name or 'valid' in name or 'test' in name):
                split = "train" if "train" in name else ("valid" if "valid" in name else "test")
                img_filename = Path(name).name
                out_img_path = out_images_dir / split / img_filename
                if not out_img_path.exists():
                    out_img_path.write_bytes(z.read(name))
                    stats[f"{split}_images_extracted"] += 1

    yaml_content = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/valid
test: images/test

nc: 2
names: ['hat', 'person']
"""
    yaml_path = out_dir / "gdut_hwd.yaml"
    yaml_path.write_text(yaml_content, encoding='utf-8')

    print(f"✅ GDUT-HWD Standardized Successfully!")
    print(f"   -> Train Images: {stats['train_images_extracted']} | Valid Images: {stats['valid_images_extracted']} | Test Images: {stats['test_images_extracted']}")
    print(f"   -> Mapped helmet -> hat (0): {stats['helmet_mapped_to_hat']}")
    print(f"   -> Mapped head -> person (1): {stats['head_mapped_to_person']}")
    print(f"   -> Filtered Full Body Boxes : {stats['person_dropped']}")
    print(f"   -> YAML Config: {yaml_path.resolve()}")
    return yaml_path

def standardize_shel5k(base_dir: Path):
    """
    Standardizes SHEL5K dataset from Mendeley Pascal VOC XML zip.
    Original classes: ['helmet', 'head_with_helmet', 'face', 'person_with_helmet', 'head', 'person_no_helmet']
    Canonical Mapping:
      - 'helmet', 'head_with_helmet' -> 0 ('hat')
      - 'head', 'person_no_helmet'   -> 1 ('person')
      - 'face', 'person_with_helmet' -> ignored
    """
    print("\n" + "=" * 70)
    print("🔄 STANDARDIZING SHEL5K DATASET (Pascal VOC XML -> YOLO)")
    print("=" * 70)

    zip_path = base_dir / "Dataset" / "SHEL5K" / "9rcv8mm682-4.zip"
    out_dir = base_dir / "Dataset" / "STANDARDIZED" / "SHEL5K"
    
    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return None

    out_images_dir = out_dir / "images"
    out_labels_dir = out_dir / "labels"
    for split in ["train", "val"]:
        (out_images_dir / split).mkdir(parents=True, exist_ok=True)
        (out_labels_dir / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        xml_files = [n for n in namelist if n.endswith('.xml')]
        print(f"-> Total XML Annotations in SHEL5K: {len(xml_files)}")

        for idx, xml_name in enumerate(xml_files):
            split = "val" if (idx % 5 == 0) else "train"
            
            xml_bytes = z.read(xml_name)
            tree = ET.fromstring(xml_bytes)
            
            size_elem = tree.find("size")
            width = float(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 0
            height = float(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 0
            
            base_stem = Path(xml_name).stem
            img_candidates = [n for n in namelist if Path(n).stem == base_stem and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not img_candidates:
                continue
            
            img_zip_name = img_candidates[0]
            img_bytes = z.read(img_zip_name)
            
            if width <= 0 or height <= 0:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    continue

            converted_lines = []
            for obj in tree.findall("object"):
                name = obj.find("name").text.strip().lower()
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                xmin = max(0.0, min(xmin, width))
                xmax = max(0.0, min(xmax, width))
                ymin = max(0.0, min(ymin, height))
                ymax = max(0.0, min(ymax, height))

                bw = xmax - xmin
                bh = ymax - ymin
                if bw <= 1 or bh <= 1:
                    continue

                x_center = (xmin + bw / 2.0) / width
                y_center = (ymin + bh / 2.0) / height
                norm_w = bw / width
                norm_h = bh / height

                if name in ["helmet", "head_with_helmet", "hat"]:
                    converted_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    stats["helmet_mapped_to_hat"] += 1
                elif name in ["head", "person_no_helmet", "no_helmet", "person"]:
                    converted_lines.append(f"1 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    stats["head_mapped_to_person"] += 1
                else:
                    stats["other_ignored"] += 1

            out_img_path = out_images_dir / split / f"{base_stem}.jpg"
            out_img_path.write_bytes(img_bytes)
            
            out_label_path = out_labels_dir / split / f"{base_stem}.txt"
            out_label_path.write_text("\n".join(converted_lines), encoding='utf-8')
            stats[f"{split}_images_saved"] += 1

    yaml_content = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/val

nc: 2
names: ['hat', 'person']
"""
    yaml_path = out_dir / "shel5k.yaml"
    yaml_path.write_text(yaml_content, encoding='utf-8')

    print(f"✅ SHEL5K Standardized Successfully!")
    print(f"   -> Train Images: {stats['train_images_saved']} | Val Images: {stats['val_images_saved']}")
    print(f"   -> Mapped helmet -> hat (0): {stats['helmet_mapped_to_hat']}")
    print(f"   -> Mapped head -> person (1): {stats['head_mapped_to_person']}")
    print(f"   -> Filtered Complex PPE / Face: {stats['other_ignored']}")
    print(f"   -> YAML Config: {yaml_path.resolve()}")
    return yaml_path

if __name__ == '__main__':
    base_dir = Path('.').resolve()
    standardize_gdut_hwd(base_dir)
    standardize_shel5k(base_dir)

