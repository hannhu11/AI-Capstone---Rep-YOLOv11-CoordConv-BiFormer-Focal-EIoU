import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import hashlib

chv_root = Path('Dataset/CHV_dataset')
gdut_root = Path('Dataset/GDUT-HWD')

# 1. Parse all GDUT annotations
gdut_data = {}
for xml_p in (gdut_root / 'Annotations').glob('*.xml'):
    tree = ET.parse(xml_p)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        continue
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    if w <= 0 or h <= 0:
        continue
    
    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip().lower()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append((cls_name, xc, yc, bw, bh))
    gdut_data[xml_p.stem] = (w, h, boxes)

print(f"Loaded {len(gdut_data)} GDUT annotations.")

# 2. Parse all CHV annotations
# CHV classes: 0: person, 1: vest, 2: blue_helmet, 3: red_helmet, 4: white_helmet, 5: yellow_helmet
chv_class_map = {0: 'person', 1: 'vest', 2: 'blue', 3: 'red', 4: 'white', 5: 'yellow'}
chv_data = {}
for txt_p in (chv_root / 'annotations').glob('*.txt'):
    boxes = []
    for line in txt_p.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            boxes.append((chv_class_map[cid], xc, yc, bw, bh))
    chv_data[txt_p.stem] = boxes

print(f"Loaded {len(chv_data)} CHV annotations.")

# 3. Match CHV to GDUT using helmet coordinates
# Helmet classes in GDUT: blue, red, white, yellow, none
# Helmet classes in CHV: blue, red, white, yellow
matches = {}
for chv_stem, chv_boxes in chv_data.items():
    chv_helmets = [(c, xc, yc, bw, bh) for (c, xc, yc, bw, bh) in chv_boxes if c in ['blue', 'red', 'white', 'yellow']]
    if not chv_helmets:
        continue
    
    best_match = None
    best_score = 0.0
    for gdut_stem, (gw, gh, gdut_boxes) in gdut_data.items():
        gdut_colored = [(c, xc, yc, bw, bh) for (c, xc, yc, bw, bh) in gdut_boxes if c in ['blue', 'red', 'white', 'yellow']]
        if len(gdut_colored) != len(chv_helmets):
            continue
        # Compare boxes
        all_matched = True
        for (cc, c_xc, c_yc, c_bw, c_bh) in chv_helmets:
            # find matching box in gdut
            found = False
            for (gc, g_xc, g_yc, g_bw, g_bh) in gdut_colored:
                if cc == gc and abs(c_xc - g_xc) < 0.05 and abs(c_yc - g_yc) < 0.05:
                    found = True
                    break
            if not found:
                all_matched = False
                break
        if all_matched:
            best_match = gdut_stem
            break
    if best_match:
        matches[chv_stem] = best_match

print(f"Matched {len(matches)} / {len(chv_data)} CHV images to GDUT images using exact helmet coordinate fingerprints.")

# Count none (no_helmet) boxes in the matched GDUT images
none_boxes_count = 0
none_images_count = 0
for chv_stem, gdut_stem in matches.items():
    gw, gh, gdut_boxes = gdut_data[gdut_stem]
    nones = [b for b in gdut_boxes if b[0] == 'none']
    if nones:
        none_images_count += 1
        none_boxes_count += len(nones)

print(f"In matched GDUT images: {none_images_count} images have {none_boxes_count} 'none' (no_helmet) boxes!")
