# Stage 2 Next Plan and Kaggle Upload Guide

## Current State From Stage 1

Stage 1 stable baseline is complete. The consolidated outputs are in:

```text
C:\Users\ADMIN\Downloads\capstone AI\SHWD_Baseline_Consolidated_2
```

The compact Kaggle-downloadable package is:

```text
C:\Users\ADMIN\Downloads\capstone AI\SHWD_Baseline_Consolidated_2\SHWD_Compact_Outputs
```

Important confirmed files:

```text
SHWD_Compact_Outputs/master_benchmark_results.csv
SHWD_Compact_Outputs/MASTER_BENCHMARK_SUMMARY.md
SHWD_Compact_Outputs/weights/yolo11s_best.pt
SHWD_Compact_Outputs/weights/yolov8s_best.pt
SHWD_Compact_Outputs/eval_plots/
SHWD_Compact_Outputs/plots/
```

## Top-2 Models Locked For Stage 2

The Stage 2 ablation should use only these two backbones:

| Role | Model | Reason |
|---|---|---|
| Accuracy Leader | `yolo11s.pt` / `yolo11s_best.pt` | Highest `mAP@0.5:0.95 = 0.6254`, highest `AP_hat = 0.7426` |
| Balance/Recall Leader | `yolov8s.pt` / `yolov8s_best.pt` | Highest `mAP@0.5 = 0.9489`, best `recall_hat = 0.9062`, best `F1_hat = 0.9120` |

Do not spend the next Kaggle session rerunning all six baselines. Use Stage 1 as the control group.

## Key Stage 1 Interpretation

- The target `mAP@0.5 >= 95%` was not fully reached; best is `yolov8s.pt = 94.89%`.
- The target `mAP@0.5:0.95 >= 70%` was not reached; best is `yolo11s.pt = 62.54%`.
- The main Stage 2 optimization target should be localization quality and hard cases, not only mAP@0.5.
- `hard_negative_status = not_available`, so yellow bucket / traffic cone / sign false-positive testing is still missing.
- The ONNX latency numbers in the consolidated CSV are not deployment-grade. Official speed should be measured later on the PC with TensorRT FP16.

## Stage 2 Ablation Strategy

Run a controlled ablation, not one giant combined model first.

Recommended order:

1. `A0`: Re-evaluate `yolo11s_best.pt` and `yolov8s_best.pt` as frozen controls on the same test split.
2. `A1`: Add hard-case augmentation only: RandomShadow, HSV shift, brightness/contrast, cutout/CoarseDropout.
3. `A2`: Add CoordConv in the stem or early neck only.
4. `A3`: Add RepConv/RepC3 and verify `switch_to_deploy()` before ONNX/TensorRT export.
5. `A4`: Add Focal-EIoU for box regression and alpha-balanced focal classification for `hat`.
6. `A5`: Add BiFormer/BRA-style attention only if A1-A4 still miss occluded/tiny helmets.
7. `A6`: Train the best combined candidate.

Promotion rule:

- Keep a module only if it improves at least one of:
  - `mAP@0.5:0.95`
  - `AP_hat`
  - `recall_hat`
  - yellow-object hard-negative false-positive rate
- Reject a module if it adds large latency without improving hard cases.

## What To Upload To Kaggle Next

Create or attach three Kaggle datasets:

### 1. Dataset: `voc2028`

Upload folder:

```text
C:\Users\ADMIN\Downloads\capstone AI\VOC2028
```

Expected Kaggle content:

```text
/kaggle/input/<voc2028-slug>/VOC2028/Annotations
/kaggle/input/<voc2028-slug>/VOC2028/JPEGImages
/kaggle/input/<voc2028-slug>/VOC2028/ImageSets/Main/trainval.txt
/kaggle/input/<voc2028-slug>/VOC2028/ImageSets/Main/test.txt
```

### 2. Dataset: `shwd-baseline-compact-outputs`

Upload folder:

```text
C:\Users\ADMIN\Downloads\capstone AI\SHWD_Baseline_Consolidated_2\SHWD_Compact_Outputs
```

This is required so Stage 2 can start from:

```text
weights/yolo11s_best.pt
weights/yolov8s_best.pt
master_benchmark_results.csv
MASTER_BENCHMARK_SUMMARY.md
```

### 3. Optional Dataset: `hard-negatives`

Upload hard-negative images if you have them:

```text
hard-negatives/
  yellow_bucket/
  traffic_cone/
  safety_sign/
  low_light/
```

If this dataset is missing, Stage 2 can still train, but false-positive claims on yellow objects remain unproven.

## Notebook To Upload / Use

Use this new notebook for the next Kaggle session:

```text
SHWD_Stage2_Ablation_Setup.ipynb
```

Also upload these support files:

```text
kaggle_shwd_baseline.py
custom_ablation_modules.py
albumentations_hardcase_policy.py
requirements_kaggle.txt
CUSTOM_ABLATION_BLUEPRINT.md
STAGE2_NEXT_PLAN_AND_KAGGLE_UPLOAD.md
```

Do not use `structural-re-parameterized-yolo-architecture2.ipynb` for Stage 2. That notebook already served the baseline rerun for `yolo11n/yolo11s`.

Use `SHWD_Benchmark_Aggregator_and_Selection.ipynb` only if you need to regenerate the consolidated summary/charts.

## Kaggle Settings

Use:

```text
Accelerator: GPU T4 x2
Internet: On
Persistence: Save Version after important runs
```

Start Stage 2 with smaller smoke checks:

```text
epochs=1 or 3
imgsz=640
device=0,1
batch=8 or 16
```

Then run full ablations:

```text
epochs=100
imgsz=640
device=0,1
batch=16
```

## Stop Conditions

Stop and inspect before continuing if:

- `mAP@0.5:0.95` drops below the baseline by more than 0.5 percentage points.
- `recall_hat` drops below the baseline.
- Training becomes unstable after adding a custom loss.
- ONNX export fails after RepConv fusion.
- Hard-negative false positives increase.

## Immediate Next Task

Run `SHWD_Stage2_Ablation_Setup.ipynb` on Kaggle first. It should verify:

- VOC2028 dataset path is found.
- `SHWD_Compact_Outputs` is found.
- `yolo11s_best.pt` and `yolov8s_best.pt` are found.
- conversion to YOLO labels still produces `6064` train and `1517` test images.
- custom module smoke test passes under Kaggle PyTorch.

After that, implement the first true custom ablation trainer/run for `A1` and `A2`.
