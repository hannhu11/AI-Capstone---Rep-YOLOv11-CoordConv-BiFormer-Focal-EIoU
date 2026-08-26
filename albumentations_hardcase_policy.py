"""
Albumentations hard-case augmentation policy for Stage 2 experiments.

This file is intentionally separate from the stable baseline pipeline. Stage 1
uses stock Ultralytics augmentation knobs for fair comparison. Stage 2 can use
this policy in a custom Dataset/Trainer when testing hard-case robustness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def build_hardcase_transform(image_size: int = 640):
    """
    Build an Albumentations transform for helmet hard cases.

    Targeted effects:
    - RandomShadow: cloud/scaffolding shadows.
    - RandomBrightnessContrast: harsh sunlight and low light.
    - HueSaturationValue: color robustness against yellow buckets/signs.
    - CoarseDropout: partial occlusion from scaffolding/grid structures.

    Bboxes use YOLO format: class_id x_center y_center width height.
    """
    import albumentations as A

    coarse_kwargs = {
        "num_holes_range": (1, 8),
        "hole_height_range": (0.03, 0.18),
        "hole_width_range": (0.03, 0.18),
        "fill": 0,
        "p": 0.35,
    }

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, fill=114, p=1.0),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), shadow_dimension=5, p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.45),
            A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=45, val_shift_limit=35, p=0.55),
            A.MotionBlur(blur_limit=5, p=0.12),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.18),
            A.CoarseDropout(**coarse_kwargs),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.20),
    )


def read_yolo_label(label_path: Path) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    if not label_path.exists():
        return boxes, labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append(int(float(parts[0])))
        boxes.append([float(value) for value in parts[1:]])
    return boxes, labels


def write_yolo_label(label_path: Path, boxes: Iterable[Iterable[float]], labels: Iterable[int]) -> None:
    lines = []
    for label, box in zip(labels, boxes):
        x, y, w, h = box
        x = min(1.0, max(0.0, float(x)))
        y = min(1.0, max(0.0, float(y)))
        w = min(1.0, max(0.0, float(w)))
        h = min(1.0, max(0.0, float(h)))
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{int(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def augment_one_yolo_sample(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    image_size: int = 640,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, labels = read_yolo_label(label_path)
    transform = build_hardcase_transform(image_size=image_size)
    augmented = transform(image=image, bboxes=boxes, class_labels=labels)
    out_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), out_image)
    write_yolo_label(output_label_path, augmented["bboxes"], augmented["class_labels"])
    return True


if __name__ == "__main__":
    print("Import this module from a Stage 2 custom Dataset/Trainer.")
