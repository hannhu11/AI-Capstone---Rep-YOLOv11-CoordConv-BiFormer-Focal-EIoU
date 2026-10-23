from pathlib import Path
from collections import Counter

classes = ['person', 'safety_vest', 'blue_helmet', 'red_helmet', 'white_helmet', 'yellow_helmet']
counts = {s: Counter() for s in ['train', 'val', 'test']}

for s in ['train', 'val', 'test']:
    p = Path('c:/Users/ADMIN/Downloads/capstone AI/Dataset/CHV_6CLASS_OFFICIAL_CLEAN/labels') / s
    for f in p.glob('*.txt'):
        for l in f.read_text().splitlines():
            if l.strip():
                counts[s][int(l.split()[0])] += 1

print("=" * 80)
print("EXACT CLASS DISTRIBUTION IN CHV_6CLASS_OFFICIAL_CLEAN:")
print("=" * 80)
for i, name in enumerate(classes):
    tr = counts['train'][i]
    va = counts['val'][i]
    te = counts['test'][i]
    tot = tr + va + te
    print(f"{i}: {name:15s} | Train: {tr:5d} | Val: {va:5d} | Test: {te:5d} | Total: {tot:5d}")
print("=" * 80)
print(f"Total BBoxes: {sum(sum(c.values()) for c in counts.values())}")
