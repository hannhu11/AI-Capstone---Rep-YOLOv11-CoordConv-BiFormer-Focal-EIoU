import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import scripts.ablation_factory

CLASS_NAMES = {
    0: 'person',
    1: 'safety_vest',
    2: 'blue_helmet',
    3: 'green_helmet',
    4: 'red_helmet',
    5: 'white_helmet',
    6: 'yellow_helmet'
}

paths = list(Path('.').rglob('*rep_yolo11s_chv7_best_deploy.pt'))
print(f"Found {len(paths)} weight files to update.")
for p in paths:
    ckpt = torch.load(p, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        old_names = getattr(ckpt['model'], 'names', None)
        print(f"Updating {p}: old names={old_names}")
        ckpt['model'].names = CLASS_NAMES
        if 'ema' in ckpt and ckpt['ema'] is not None and hasattr(ckpt['ema'], 'names'):
            ckpt['ema'].names = CLASS_NAMES
        torch.save(ckpt, p)
        print(f"Successfully saved {p} with 7-class human-readable names.")

print("All deploy checkpoints updated.")
