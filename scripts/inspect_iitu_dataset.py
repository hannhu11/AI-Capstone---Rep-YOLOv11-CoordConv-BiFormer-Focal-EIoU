from pathlib import Path
from collections import Counter

iitu_dir = Path("c:/Users/ADMIN/Downloads/capstone AI/Dataset/IITU Safety-Helmet")
print("IITU Directory structure:")
subdirs = [d.name for d in iitu_dir.iterdir() if d.is_dir()]
print("Subdirs:", subdirs)

# Check classes and counts in IITU
# Check if there is data.yaml
yaml_files = list(iitu_dir.glob("*.yaml")) + list(iitu_dir.glob("*.yml"))
print("YAML files:", yaml_files)

# Count annotations
ann_files = list(iitu_dir.rglob("*.txt"))
print(f"Total txt files in IITU: {len(ann_files)}")

# Check class IDs in txt
iitu_counts = Counter()
for f in ann_files:
    if f.name in ["data.yaml", "README.txt", "classes.txt"]:
        continue
    for l in f.read_text(errors="ignore").splitlines():
        if l.strip():
            parts = l.strip().split()
            if len(parts) >= 5:
                try:
                    c = int(parts[0])
                    iitu_counts[c] += 1
                except ValueError:
                    pass

print("IITU Class Counts by ID:")
for c, cnt in sorted(iitu_counts.items()):
    print(f"Class {c}: {cnt}")
