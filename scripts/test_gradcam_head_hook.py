import torch
import cv2
import numpy as np
from ultralytics import YOLO

def inspect_yolo_head():
    yolo = YOLO("yolo11s.pt")
    model = yolo.model
    print("Model structure last layers:")
    detect_head = model.model[-1]
    print(f"Detect head type: {type(detect_head)}")
    if hasattr(detect_head, "cv3"):
        print(f"cv3 type: {type(detect_head.cv3)}")
        for idx, m in enumerate(detect_head.cv3):
            print(f"  cv3[{idx}]: {m}")

if __name__ == "__main__":
    inspect_yolo_head()
