from pathlib import Path

p = Path("Dataset/UNIFIED_IITU_CHV_COMPLETE")
train_cnt = len(list((p / "images" / "train").glob("*")))
val_cnt = len(list((p / "images" / "val").glob("*")))
test_cnt = len(list((p / "images" / "test").glob("*")))
print(f"Train: {train_cnt}, Val: {val_cnt}, Test: {test_cnt}")

lbl_dir = p / "labels" / "test"
cls_counts = {}
for lf in lbl_dir.glob("*.txt"):
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                cls_counts[cid] = cls_counts.get(cid, 0) + 1

names = ["person", "safety_vest", "blue_helmet", "green_helmet", "red_helmet", "white_helmet", "yellow_helmet"]
print("Test instances per class:")
for cid in sorted(cls_counts):
    print(f"  {cid}: {names[cid]} = {cls_counts[cid]}")
