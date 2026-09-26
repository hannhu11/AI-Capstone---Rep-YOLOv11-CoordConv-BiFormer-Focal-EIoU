"""
Standardize CHV (Color Helmet and Vest) Dataset to 3 Canonical Classes:
  - Class 0: 'hat'    (Aggregated from 2: blue, 3: red, 4: white, 5: yellow helmet)
  - Class 1: 'person' (From 0: person)
  - Class 2: 'vest'   (From 1: vest)

Dataset Reference:
  Zijian Wang et al., "Fast Personal Protective Equipment Detection for Real Construction Sites
  Using Deep Learning Approaches", Sensors 2021, 21(10), 3478.
"""

import os
import zipfile
from collections import Counter
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = BASE_DIR / "dataset" / "CHV.zip"
OUT_DIR = BASE_DIR / "dataset" / "STANDARDIZED_CHV"

CLASS_MAP = {
    0: 1,  # person -> 1 (person)
    1: 2,  # vest   -> 2 (vest)
    2: 0,  # blue   -> 0 (hat)
    3: 0,  # red    -> 0 (hat)
    4: 0,  # white  -> 0 (hat)
    5: 0,  # yellow -> 0 (hat)
}

TARGET_NAMES = ['hat', 'person', 'vest']


def standardize():
    print("=" * 70)
    print("🔄 STANDARDIZING CHV DATASET TO 3 CLASSES: ['hat', 'person', 'vest']")
    print("=" * 70)
    print(f"Zip archive: {ZIP_PATH}")
    print(f"Output directory: {OUT_DIR}")

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing {ZIP_PATH}")

    # Prepare output directories
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # 1. Read splits
        # Note: In CHV zip, paths inside train.txt look like: 'CHV_dataset/images/ppe_1106.jpg'
        def get_stems_from_split(split_name):
            split_txt = f"CHV_dataset/data split/{split_name}.txt"
            content = z.read(split_txt).decode('utf-8', errors='ignore').splitlines()
            stems = set()
            for line in content:
                line = line.strip()
                if line:
                    stems.add(Path(line).stem)
            return stems

        train_stems = get_stems_from_split("train")
        valid_stems = get_stems_from_split("valid")
        test_stems = get_stems_from_split("test")

        print(f"Split counts: Train={len(train_stems)}, Valid={len(valid_stems)}, Test={len(test_stems)}")

        stats = Counter()
        class_instance_counts = Counter()

        # 2. Extract and convert annotations and images
        all_names = z.namelist()
        img_names = [n for n in all_names if 'images/' in n and n.lower().endswith(('.jpg', '.jpeg', '.png')) and not n.startswith('__MACOSX')]

        for img_path_in_zip in img_names:
            stem = Path(img_path_in_zip).stem
            
            # Determine split
            if stem in train_stems:
                split = "train"
            elif stem in valid_stems:
                split = "val"
            elif stem in test_stems:
                split = "test"
            else:
                # Default to train if not explicitly listed
                split = "train"

            # Corresponding label in zip
            anno_path_in_zip = f"CHV_dataset/annotations/{stem}.txt"
            
            converted_lines = []
            if anno_path_in_zip in all_names:
                raw_lines = z.read(anno_path_in_zip).decode('utf-8', errors='ignore').splitlines()
                for line in raw_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cls = int(parts[0])
                        coords = " ".join(parts[1:5])
                        if orig_cls in CLASS_MAP:
                            target_cls = CLASS_MAP[orig_cls]
                            converted_lines.append(f"{target_cls} {coords}")
                            stats[f"orig_cls_{orig_cls}"] += 1
                            class_instance_counts[TARGET_NAMES[target_cls]] += 1

            # Write label file
            out_label = OUT_DIR / "labels" / split / f"{stem}.txt"
            out_label.write_text("\n".join(converted_lines), encoding='utf-8')

            # Write image file
            out_img = OUT_DIR / "images" / split / f"{stem}.jpg"
            out_img.write_bytes(z.read(img_path_in_zip))
            stats[f"{split}_images"] += 1

    # Write data yaml
    yaml_content = f"""path: {OUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['hat', 'person', 'vest']
"""
    yaml_path = OUT_DIR / "chv_3class.yaml"
    yaml_path.write_text(yaml_content, encoding='utf-8')

    print("\n✅ STANDARDIZATION COMPLETE!")
    print(f"   Images: Train={stats['train_images']}, Val={stats['val_images']}, Test={stats['test_images']}")
    print(f"   Total Instances mapped:")
    for cname in TARGET_NAMES:
        print(f"     - {cname}: {class_instance_counts[cname]}")
    print(f"   Config YAML: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    standardize()
