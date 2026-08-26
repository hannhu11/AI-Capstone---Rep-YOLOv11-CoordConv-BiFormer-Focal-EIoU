"""
Fixed Grad-CAM & Feature Heatmap Comparison Generator for IEEE Q1 Paper.

100% Real Dataset Execution:
1. Dynamic Dataset Image Resolver: Automatically locates and loads real images from VOC2028 / SHWD dataset.
2. Target Layer Hooking: Hooks directly into the Classification Convolutions (cv3) of Ultralytics Detect Head.
3. Letterbox Padding Crop: Automatically un-pads and crops the black letterbox border before overlaying heatmap.
4. Raw Normalization: Zero artificial contrast hacks (np.power removed completely).
5. Loss Target: Specific confidence score backpropagation targeting Class 0 (hat).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


def find_real_dataset_samples() -> list[str]:
    """
    Search for authentic dataset images across Kaggle and local filesystems.
    Prioritizes specific images demonstrating the 3 paper scenarios.
    """
    search_roots = [
        Path("/kaggle/input/datasets/hannhu4002/voc2028/VOC2028/JPEGImages"),
        Path("/kaggle/input/voc2028/VOC2028/JPEGImages"),
        Path("VOC2028/JPEGImages"),
        Path("../VOC2028/JPEGImages"),
        Path("JPEGImages"),
        Path("test_screenshots"),
    ]

    # Additional recursive search for any JPEGImages folder
    for candidate_root in [Path("/kaggle/input"), Path(".")]:
        if candidate_root.exists():
            for p in candidate_root.rglob("JPEGImages"):
                if p.is_dir() and p not in search_roots:
                    search_roots.append(p)

    target_scenarios = [
        ["000008.jpg", "000063.jpg", "000054.jpg", "000030.jpg", "test_construction_site_workers_1080p.jpg"],
        ["000055.jpg", "000095.jpg", "000115.jpg", "000286.jpg", "test_13921040-uhd_3840_2160_30fps.jpg"],
        ["000128.jpg", "000255.jpg", "000382.jpg", "000459.jpg", "test_14842935_2160_3840_30fps.jpg"],
    ]

    resolved_samples: list[str] = []
    
    for row_idx, candidates in enumerate(target_scenarios):
        found_path = None
        for root in search_roots:
            if not root.exists():
                continue
            for cand_name in candidates:
                p = root / cand_name
                if p.exists() and p.is_file():
                    found_path = str(p)
                    break
            if found_path:
                break
        
        if found_path:
            resolved_samples.append(found_path)
            print(f"-> [Scenario {row_idx+1}] Located real image: {found_path}")

    # Fallback if specific filenames are not found: pick any 3 real images from found directories
    if len(resolved_samples) < 3:
        all_found_images = []
        for root in search_roots:
            if root.exists() and root.is_dir():
                for img_p in root.glob("*.jpg"):
                    if str(img_p) not in resolved_samples:
                        all_found_images.append(str(img_p))
                    if len(all_found_images) >= 10:
                        break
            if len(all_found_images) >= 10:
                break
        
        while len(resolved_samples) < 3 and all_found_images:
            pick = all_found_images.pop(0)
            resolved_samples.append(pick)
            print(f"-> [Scenario {len(resolved_samples)}] Fallback selected real image: {pick}")

    if not resolved_samples:
        raise FileNotFoundError(
            "CRITICAL: Could not find any real images in VOC2028/JPEGImages. "
            "Please ensure the VOC2028 dataset is mounted in /kaggle/input or locally."
        )

    return resolved_samples[:3]


def letterbox_image(image: np.ndarray, target_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad image preserving aspect ratio. Returns (padded_image, scale, (pad_x, pad_y))."""
    ih, iw = image.shape[:2]
    tw, th = target_shape
    scale = min(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((th, tw, 3), dtype=np.uint8)  # Black padding
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    padded[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return padded, scale, (pad_x, pad_y)


class FixedGradCAMYOLO:
    """Fixed YOLO Grad-CAM targeting Detect Head Classification Convolutions (cv3)."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        # Locate Detect head
        detect_head = None
        if hasattr(self.model, "model"):
            for m in reversed(self.model.model):
                if m.__class__.__name__ == "Detect":
                    detect_head = m
                    break
        
        if detect_head is not None and hasattr(detect_head, "cv3"):
            # Hook into P3 and P4 classification heads
            target_modules = [detect_head.cv3[0], detect_head.cv3[1]]
        else:
            # Fallback to layer 22
            target_modules = [self.model.model[22]]

        for mod in target_modules:
            def get_fwd_hook():
                def hook(module, input, output):
                    act = output[0] if isinstance(output, (tuple, list)) else output
                    self.activations.append(act)
                return hook

            def get_bwd_hook():
                def hook(module, grad_in, grad_out):
                    g = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
                    self.gradients.append(g)
                return hook

            mod.register_forward_hook(get_fwd_hook())
            mod.register_full_backward_hook(get_bwd_hook())

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        self.activations.clear()
        self.gradients.clear()
        self.model.zero_grad()
        input_tensor.requires_grad_(True)

        preds = self.model(input_tensor)

        if isinstance(preds, (tuple, list)):
            pred_tensor = preds[0]
        else:
            pred_tensor = preds

        # Target class 0 (hat) score backprop
        if pred_tensor.ndim == 3:
            if pred_tensor.shape[1] > 4 + class_idx:
                score = pred_tensor[0, 4 + class_idx, :].sum()
            else:
                score = pred_tensor[0, class_idx, :].sum()
        else:
            score = pred_tensor.sum()

        score.backward(retain_graph=True)

        cams = []
        for act, grad in zip(self.activations, self.gradients):
            if grad is None or act is None:
                continue
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1).squeeze(0)
            cam = F.relu(cam)
            cam_np = cam.detach().cpu().numpy()
            cam_resized = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))
            cams.append(cam_resized)

        if cams:
            combined_cam = np.mean(cams, axis=0)
        else:
            combined_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Min-Max Normalization (Zero artificial power scaling hacks)
        cam_min, cam_max = combined_cam.min(), combined_cam.max()
        if cam_max > cam_min:
            combined_cam = (combined_cam - cam_min) / (cam_max - cam_min)
        else:
            combined_cam = np.zeros_like(combined_cam)

        return combined_cam


def overlay_cropped_heatmap(
    img_bgr: np.ndarray,
    heatmap_640: np.ndarray,
    pad_info: tuple[int, int],
    target_shape: tuple[int, int] = (640, 640),
    alpha: float = 0.50,
) -> np.ndarray:
    """Crop padding region from 640x640 heatmap before blending with original image."""
    ih, iw = img_bgr.shape[:2]
    pad_x, pad_y = pad_info
    tw, th = target_shape
    nw = tw - 2 * pad_x
    nh = th - 2 * pad_y

    # Crop out black letterbox padding area from 640x640 heatmap
    cropped_heatmap = heatmap_640[pad_y : pad_y + nh, pad_x : pad_x + nw]
    
    # Resize cropped heatmap back to original image dimensions (iw, ih)
    heatmap_resized = cv2.resize(cropped_heatmap, (iw, ih), interpolation=cv2.INTER_LINEAR)
    
    # Convert heatmap to JET colormap
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original BGR image
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def run_gradcam_comparison(
    baseline_path: str,
    proposed_path: str,
    image_paths: list[str],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Loading Baseline model from: {baseline_path}")
    yolo_base = YOLO(baseline_path)

    print(f"-> Loading Proposed Rep-YOLO11s model from: {proposed_path}")
    yolo_prop = YOLO(proposed_path)

    gradcam_base = FixedGradCAMYOLO(yolo_base.model, device=device)
    gradcam_prop = FixedGradCAMYOLO(yolo_prop.model, device=device)

    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4 * len(image_paths)), dpi=300)
    if len(image_paths) == 1:
        axes = np.expand_dims(axes, axis=0)

    scenario_titles = [
        "Scenario 1: Yellow Distractors (Buckets / Signs)",
        "Scenario 2: Distant Tiny Targets (<20x20 px)",
        "Scenario 3: Partial Occlusion & Scaffolding",
    ]

    col_headers = [
        "(a) Original Input",
        "(b) Baseline YOLO11s\n(Diffused Attention)",
        "(c) Rep-YOLO11s (Ours)\n(Focused Saliency)",
        "(d) Final Detections\n(Proposed Model)",
    ]

    for row_idx, img_path_str in enumerate(image_paths):
        img_path = Path(img_path_str)
        if not img_path.exists():
            raise FileNotFoundError(f"Real dataset image not found: {img_path}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess with letterboxing and save padding info
        padded_bgr, scale, pad_info = letterbox_image(img_bgr, (640, 640))
        input_tensor = torch.from_numpy(padded_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        # Generate 640x640 Grad-CAM heatmaps
        cam_base = gradcam_base.generate(input_tensor, class_idx=0)
        cam_prop = gradcam_prop.generate(input_tensor, class_idx=0)

        # Overlay cropped heatmaps (un-padded)
        heatmap_base_rgb = overlay_cropped_heatmap(img_bgr, cam_base, pad_info, alpha=0.50)
        heatmap_prop_rgb = overlay_cropped_heatmap(img_bgr, cam_prop, pad_info, alpha=0.50)

        # Predict Column 4 Detections
        results = yolo_prop.predict(img_bgr, imgsz=640, conf=0.25, verbose=False)
        det_img_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        # Plot 4 Columns
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 1].imshow(heatmap_base_rgb)
        axes[row_idx, 2].imshow(heatmap_prop_rgb)
        axes[row_idx, 3].imshow(det_img_rgb)

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=11, fontweight="bold", pad=10)

        title_label = scenario_titles[row_idx] if row_idx < len(scenario_titles) else f"Scenario {row_idx+1}"
        axes[row_idx, 0].set_ylabel(title_label, fontsize=10, fontweight="bold", labelpad=10)

    plt.tight_layout()
    pdf_path = output_dir / "shwd_gradcam_comparison.pdf"
    png_path = output_dir / "shwd_gradcam_comparison.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()

    print("\n" + "=" * 65)
    print("✅ FIXED GRAD-CAM COMPARISON FIGURE GENERATED SUCCESSFULLY FROM REAL IMAGES!")
    print(f"   - PDF Artifact : {pdf_path}")
    print(f"   - PNG Artifact : {png_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fixed Grad-CAM Comparison for IEEE Q1 Paper")
    parser.add_argument("--baseline", type=str, default="yolo11s.pt", help="Baseline weights")
    parser.add_argument(
        "--proposed",
        type=str,
        default="Output/shwd-stage2-ablation-setup-full-train-run-a6/weights/yolo11s_best.pt",
        help="Proposed Rep-YOLO11s weights",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_overleaf/figures"), help="Output directory"
    )
    args = parser.parse_args()

    # Dynamic search for proposed weights
    proposed_weight = args.proposed
    if not Path(proposed_weight).exists():
        found_weights = list(Path(".").rglob("yolo11s_best.pt"))
        if found_weights:
            proposed_weight = str(found_weights[0])
            print(f"-> Auto-resolved Proposed weights path to: {proposed_weight}")

    # Dynamically find 3 real dataset images
    sample_images = find_real_dataset_samples()

    run_gradcam_comparison(args.baseline, proposed_weight, sample_images, args.output_dir)
