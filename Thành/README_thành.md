# Rep-YOLO11s: Safety Helmet Detection & Cross-Domain Benchmark Pipeline

This repository contains the complete research and evaluation pipeline for **Rep-YOLO11s**, a structurally re-parameterized YOLO model optimized for real-time safety helmet detection in complex industrial construction environments. 

The project is structured into two primary phases: **(1) The Master Training Pipeline** and **(2) The Cross-Domain Generalization Benchmark**.

---

## 1. Master Research Pipeline (Model Training)

This phase covers the end-to-end training of the Rep-YOLO11s architecture on the source domain (SHWD / VOC2028). The pipeline integrates advanced architectural modifications, including CoordConv spatial priors, BiFormer dynamic routing attention, and Focal EIoU Loss, to address challenges like tiny distant targets and heavy occlusion.

### 🔗 Kaggle Notebooks
* **Link:** [shwd-stage-3-kaggle-master-research-pipeline-4](https://www.kaggle.com/code/nvthanh2004/shwd-stage-3-kaggle-master-research-pipeline-4)
  * **Version 4:** Configured for standard spatial resolution (`imgsz=640`). Matches the local file: `shwd-stage-3-kaggle-master-research-pipeline-4_imgsz=640/shwd-stage-3-kaggle-master-research-pipeline-4.ipynb`.
  * **Version 5:** Configured for high-resolution input (`imgsz=960`) to breach the theoretical bottleneck for tiny drone/CCTV targets. Matches the local file: `shwd-stage-3-kaggle-master-research-pipeline-4_imgsz=960/shwd-stage-3-kaggle-master-research-pipeline-4.ipynb`.

### 🛠️ Implementation Details
- **Environment:** PyTorch on Dual NVIDIA Tesla T4 GPUs.
- **Dataset:** Safety Helmet Wearing Dataset (SHWD) featuring $7,581$ high-resolution images with extreme class imbalance ($1:12$ ratio of hats to persons).
- **Hyperparameters:** SGD optimizer ($\eta_0=0.01$), 100 epochs, batch size of 32 (scaled dynamically for 960px).

---

## 2. Cross-Domain Generalization Benchmark

To rigorously evaluate the robustness of Rep-YOLO11s on unobserved data distributions without any fine-tuning (zero-shot evaluation), the trained weights from Phase 1 are tested against multiple external datasets.

### 🔗 Kaggle Notebook
* **Link:** [shwd-cross-domain-benchmark](https://www.kaggle.com/code/nguyenvanthanh232/shwd-cross-domain-benchmark)
* **Local File:** `shwd_cross_domain_benchmark.ipynb`

### 🛠️ Methodology & Detailed Deployment
This notebook automates the downloading, parsing, and strict evaluation of 5 distinct datasets (VOC2028, GDUT-HWD, SHEL5K, Hard Hat Workers, Safety Helmet Det). To achieve high-fidelity scientific metrics, the following strategies were deployed:

1. **Dataset Standardization & Parsing:**
   The notebook dynamically extracts `.zip` datasets and converts diverse annotation formats (XML, raw TXT) into a unified YOLO YAML structure. 

2. **Label Harmonization (Hat-Only Evaluation):**
   * **The Problem:** External datasets employ highly divergent annotation guidelines for the `person` class. For instance, the "Hard Hat Workers" dataset draws bounding boxes exclusively around the *head*, while the source domain (VOC2028) draws boxes around the *entire body*. Evaluating full-body predictions against head-only ground truths results in mathematically degraded $mAP$ scores, despite the model detecting the person perfectly.
   * **The Solution:** The parsing script aggressively filters the datasets to map annotations exclusively to the `hat` (safety helmet) class (`nc=1`). This isolates the evaluation to the primary objective of the research—detecting safety helmets—ensuring metrics reflect true model capability rather than annotation noise.

3. **Strict Metric Post-Processing:**
   To push evaluation metrics ($mAP_{50}$ and $mAP_{50-95}$) to their maximum theoretical limits, the validation function (`model.val`) is heavily tuned:
   * **FP32 Precision (`half=False`):** Bounding box coordinates are calculated in 32-bit floating-point precision. This is critical for maximizing $mAP_{50-95}$, which requires pixel-perfect overlap with ground truth boxes.
   * **Aspect Ratio Preservation (`rect=False`):** Disables rectangular inference padding, forcing the model to evaluate on strict square tensors. This prevents edge objects (helmets at the border of the image) from being distorted or cropped.
   * **NMS Tuning:** Non-Maximum Suppression is configured with a standard Intersection over Union (`iou=0.6`) and an extremely low confidence threshold (`conf=0.001`) to capture every possible bounding box proposal, maximizing theoretical Recall during PR-curve generation.
