"""
Visualizes random samples from UNIFIED_PPE_MASTER_HARMONIZED
specifically verifying fused images (GDUT helmet + CHV vest).
"""
import cv2
import random
from pathlib import Path

DATA_DIR = Path(r"C:\Users\ADMIN\Downloads\capstone AI\Dataset\UNIFIED_PPE_MASTER_HARMONIZED")
OUT_DIR = DATA_DIR / "sample_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'blue_helmet',   # 0 - Blue
    'green_helmet',  # 1 - Green
    'red_helmet',    # 2 - Red
    'white_helmet',  # 3 - White / Grey
    'yellow_helmet', # 4 - Yellow
    'no_helmet',     # 5 - Purple (Violation)
    'safety_vest'    # 6 - Orange / Yellow-Green
]

COLORS = [
    (255, 0, 0),     # 0: Blue
    (0, 255, 0),     # 1: Green
    (0, 0, 255),     # 2: Red
    (220, 220, 220), # 3: White
    (0, 255, 255),   # 4: Yellow
    (255, 0, 255),   # 5: Magenta (Violation)
    (0, 165, 255)    # 6: Orange (Safety Vest)
]

def draw_samples():
    img_dir = DATA_DIR / "images" / "train"
    lbl_dir = DATA_DIR / "labels" / "train"
    
    # Pick 5 fused samples and 5 general samples
    fused_imgs = list(img_dir.glob("fused_*.jpg"))
    other_imgs = [p for p in img_dir.glob("*.jpg") if not p.name.startswith("fused_")]
    
    selected = random.sample(fused_imgs, min(5, len(fused_imgs))) + random.sample(other_imgs, min(5, len(other_imgs)))
    
    for img_p in selected:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        im = cv2.imread(str(img_p))
        if im is None or not lbl_p.exists():
            continue
        h, w = im.shape[:2]
        
        lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                xmin = int((xc - bw / 2.0) * w)
                xmax = int((xc + bw / 2.0) * w)
                ymin = int((yc - bh / 2.0) * h)
                ymax = int((yc + bh / 2.0) * h)
                
                color = COLORS[cid] if cid < len(COLORS) else (0, 255, 0)
                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(im, cname, (xmin, max(15, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        out_p = OUT_DIR / f"vis_{img_p.name}"
        cv2.imwrite(str(out_p), im)
        print(f"✅ Saved visualization: {out_p.name}")

if __name__ == "__main__":
    draw_samples()
