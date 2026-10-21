"""
Converts Safety_Vests_Adil (_annotations.csv) to standard YOLO txt format.
Classes:
  0: Safety Vest
  1: NO-Safety Vest
"""
import os
import pandas as pd
from pathlib import Path
from PIL import Image

BASE = Path(r"C:\Users\ADMIN\Downloads\capstone AI\Dataset\DOWNLOADED_CANDIDATES\Safety_Vests_Adil")
CLASS_MAP = {
    "Safety Vest": 0,
    "NO-Safety Vest": 1
}

def convert():
    print("=" * 70)
    print("🔄 CONVERTING SAFETY_VESTS_ADIL CSV TO YOLO TXT FORMAT")
    print("=" * 70)
    
    for split in ["train", "valid", "test"]:
        csv_file = BASE / split / split / "_annotations.csv"
        img_dir = BASE / split / split
        labels_dir = BASE / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        if not csv_file.exists():
            print(f"⚠️ CSV not found: {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        print(f"Processing '{split}': {len(df)} annotations across images...")
        
        # Group by filename
        grouped = df.groupby("filename")
        written = 0
        for fname, rows in grouped:
            stem = Path(fname).stem
            txt_path = labels_dir / f"{stem}.txt"
            lines = []
            for _, r in rows.iterrows():
                cname = str(r["class"]).strip()
                if cname not in CLASS_MAP:
                    continue
                cid = CLASS_MAP[cname]
                w = float(r["width"])
                h = float(r["height"])
                xmin = float(r["xmin"])
                ymin = float(r["ymin"])
                xmax = float(r["xmax"])
                ymax = float(r["ymax"])
                
                # Bounding box sanity
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                if bw <= 1 or bh <= 1 or w <= 0 or h <= 0:
                    continue
                xc = (xmin + bw / 2.0) / w
                yc = (ymin + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                
            if lines:
                txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                written += 1
                
        print(f"  -> Generated {written} YOLO txt label files in {labels_dir}")

    # Generate data.yaml
    yaml_content = f"""# Safety Vests Adil - YOLO Format
path: {BASE.as_posix()}
train: train/train
val: valid/valid
test: test/test

nc: 2
names: ['Safety Vest', 'NO-Safety Vest']
"""
    (BASE / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"✅ Generated YAML: {BASE / 'data.yaml'}")

if __name__ == "__main__":
    convert()
