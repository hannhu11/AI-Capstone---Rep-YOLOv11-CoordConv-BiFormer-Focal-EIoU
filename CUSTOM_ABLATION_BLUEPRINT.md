# Custom Ablation Blueprint: CoordConv + BiFormer + RepConv + Focal-EIoU

Use this only after Stage 1 baseline results exist in:

```text
/kaggle/working/SHWD_YOLO/benchmark_results.csv
```

Stage 1 stays stable and comparable. Stage 2 modifies architecture/loss only for the top-2 baselines.

## Inputs

- Dataset config: `/kaggle/working/SHWD_YOLO/shwd.yaml`
- Baseline results: `/kaggle/working/SHWD_YOLO/benchmark_results.csv`
- Custom modules: `custom_ablation_modules.py`
- Hard-case augmentation policy: `albumentations_hardcase_policy.py`

## Select Top-2 Backbones

Rank models by:

1. `map50_95`
2. `map50`
3. `ap_hat` or `recall_hat`
4. `onnx_latency_mean_ms`
5. hard-negative `hat` false-positive rate

Example selection code:

```python
import pandas as pd

df = pd.read_csv("/kaggle/working/SHWD_YOLO/benchmark_results.csv")
sort_cols = [c for c in ["map50_95", "map50", "ap_hat", "onnx_fps_mean"] if c in df.columns]
top2 = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(2)
display(top2[["model", *sort_cols, "best_pt"]])
```

## Ablation Matrix

Run the same training settings for every row:

- `epochs=100`
- `imgsz=640`
- `device=0,1`
- `seed=3407`
- `train=trainval.txt`, `val/test=test.txt`
- keep only `best.pt`
- export ONNX on Kaggle
- export TensorRT FP16 only on the PC deployment machine

| ID | Change | Keep If |
|---|---|---|
| A0 | Top-2 baseline | Control |
| A1 | +P2 or CARAFE/P2-style path | Improves tiny/distant `hat` recall |
| A2 | +CoordConv in stem or selected neck conv | Reduces yellow-object false positives or improves `hat` AP |
| A3 | +ECA/Triplet/CBAM lightweight attention | Improves mAP/F1 with minor latency cost |
| A4 | +BiFormerBlockLite in neck/head only | Beats lightweight attention on hard cases |
| A5 | +RepConv/RepC3 | Preserves accuracy and improves deploy latency after fusion |
| A6 | +Focal-EIoU box loss | Improves mAP@0.5:0.95 or localization failures |
| A7 | +alpha-balanced focal classification | Improves `hat` recall without large `person` collapse |
| A8 | Best combined | Final candidate |

Promotion rule: keep a module only if it improves `hat` recall/AP, hard-negative false positives, or mAP@0.5:0.95 without violating latency.

## Class Imbalance Policy

SHWD/VOC2028 class counts are:

- `hat=9044`
- `person=111514`
- ratio about `1:12`

For custom classification loss, start with:

```python
import torch
from custom_ablation_modules import alpha_balanced_focal_bce

alpha = torch.tensor([0.85, 0.15], device=logits.device)
loss_cls = alpha_balanced_focal_bce(logits, targets, alpha=alpha, gamma=2.0)
```

Do not judge this only by global mAP. Track `hat` recall, `hat` AP50, `hat` AP50:95, and false positives on yellow distractors.

## Focal-EIoU Box Loss Snippet

```python
from custom_ablation_modules import focal_eiou_loss

loss_box = focal_eiou_loss(pred_boxes, target_boxes, xywh=True, gamma=0.5)
```

Integration note:

- Ultralytics internals change across versions. For a serious Stage 2 run, copy the installed `ultralytics` package into `/kaggle/working/vendor/ultralytics`, then patch the detection loss in that vendored copy.
- Keep Stage 1 untouched. Do not patch the global site-packages version used by baselines.

## CoordConv / RepConv / BiFormer Snippets

```python
import torch
from custom_ablation_modules import CoordConv, RepConv, BiFormerBlockLite

x = torch.randn(2, 64, 80, 80).cuda()
coord = CoordConv(64, 64).cuda()
rep = RepConv(64, 64).cuda()
attn = BiFormerBlockLite(64, num_heads=4, region_size=8, topk=4).cuda()

y = attn(rep(coord(x)))
```

Before export:

```python
for module in model.modules():
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()
```

Verify fusion:

```bash
python custom_ablation_modules.py
```

The printed `repconv_fusion_max_diff` should be very small when the block is in `eval()` mode before fusion.

## Albumentations Hard-Case Policy

Use `albumentations_hardcase_policy.py` for a custom Dataset/Trainer in Stage 2:

```python
from albumentations_hardcase_policy import build_hardcase_transform

transform = build_hardcase_transform(image_size=640)
```

This policy includes:

- Random shadow for cloud/scaffolding shadows.
- Random brightness/contrast for harsh light and low light.
- HSV hue/saturation/value shift to avoid yellow-color overfitting.
- Coarse dropout for occlusion from scaffolding/grid structures.
- Motion blur and noise for CCTV quality degradation.

Do not generate a fully duplicated offline augmented dataset by default. It can waste `/kaggle/working` storage. Use online transforms in the custom trainer.

## Multi-GPU Launch Pattern

For Ultralytics baselines, this is enough:

```bash
yolo detect train model=yolo11n.pt data=/kaggle/working/SHWD_YOLO/shwd.yaml epochs=100 imgsz=640 device=0,1
```

For custom trainers/losses, prefer explicit distributed launch:

```bash
python -m torch.distributed.run --nproc_per_node=2 train_custom_ablation.py \
  --data /kaggle/working/SHWD_YOLO/shwd.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --device 0,1 \
  --modules coordconv,biformer,repconv \
  --box-loss focal_eiou \
  --cls-loss alpha_focal \
  --alpha-hat 0.85 \
  --alpha-person 0.15
```

## Hard-Negative Audit

Put yellow-object distractor images here:

```text
/kaggle/input/hard-negatives
```

Suggested folders:

```text
/kaggle/input/hard-negatives/yellow_bucket
/kaggle/input/hard-negatives/traffic_cone
/kaggle/input/hard-negatives/safety_sign
/kaggle/input/hard-negatives/low_light
```

The baseline script will report:

- `hard_negative_hat_fp_boxes`
- `hard_negative_hat_fp_image_rate`
- `hard_negative_total_boxes`

## TensorRT PC Export

Run this only on the target PC/RTX environment:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
model.export(format="engine", imgsz=640, half=True, device=0)
```

Benchmark the full RTSP path separately:

- decode
- letterbox/resize
- inference
- postprocess/NMS
- draw/alert

The official deployment claim should use PC TensorRT FP16 numbers, not Kaggle ONNX numbers.
