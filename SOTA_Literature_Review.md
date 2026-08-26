# SOTA Literature Review for Real-Time Safety Helmet Detection

This document synthesizes the local reference papers and recent related work for an open exploration phase on real-time safety helmet detection under construction, road, mine, and CCTV-like conditions. The goal is not to lock one YOLO family immediately, but to create a technically defensible benchmark and ablation path.

## Claude / Agent Handoff Prompt

Act as a senior computer vision research engineer. Read this review, then read the local PDFs in `Refference/` before proposing architecture changes. Keep the project in open exploration mode: benchmark YOLOv8, YOLOv10, YOLO11 first, then add YOLOv7/YOLOv9 and custom ablations only after the top-2 stable baselines are known. Prioritize the target metrics `mAP@0.5`, `mAP@0.5:0.95`, FPS, latency, RTSP readiness, per-class recall for `hat`, and false positives on yellow buckets, traffic cones, and safety signs. Treat SHWD class imbalance (`hat=9044`, `person=111514`) as a central risk.

## Executive Research Position

The most promising direction is not "YOLOv11 plus every module". The safer research path is:

1. Establish clean baselines with reproducible data conversion and per-class metrics.
2. Select the top-2 backbones by accuracy-speed tradeoff.
3. Add modules one at a time: P2/small-object head, lightweight attention, CoordConv, RepConv/RepC3, then Focal-EIoU/class-weighted loss.
4. Promote only modules that improve `hat` recall and hard-negative false positives without breaking latency.

The literature strongly supports four hypotheses:

- Spatial cues matter: CoordConv and coordinate/position attention can help when helmet-like objects are confused with signs, buckets, or background objects.
- Small-object restoration matters: P2 heads, CARAFE, large-kernel/reparameterized blocks, and feature fusion are repeatedly useful for tiny or distant helmets.
- Occlusion needs targeted learning: SEAM-like occlusion-aware attention, Triplet/CBAM/GAM/ECA-style attention, and hard-example losses can reduce missed detections.
- Real-time claims must be hardware-specific: Kaggle Dual T4 is useful for training, but official deployment speed should be measured on the target PC with ONNX/TensorRT, fixed image size, batch size, precision, and warmup.

## Dataset and Project Risk Notes

Local `VOC2028` inspection shows the expected SHWD-style structure:

- Images: 7,581 `.jpg`
- Annotations: 7,581 `.xml`
- Splits: `train=5457`, `val=607`, `trainval=6064`, `test=1517`
- Labels found: `person=111514`, `hat=9044`, `dog=3`
- The 3 `dog` labels appear in `000377.xml` and should be ignored/logged as label noise.

The severe `hat:person` imbalance is approximately `1:12`. A global mAP can hide poor helmet recall. Every benchmark must therefore report per-class AP, precision, recall, F1, confusion matrix, and failure images.

## Summary Table of Related Work

| Paper / Source | Year | Backbone / Base | Key Modules | Dataset(s) | Reported Metrics | Strengths | Weaknesses / Failure Cases |
|---|---:|---|---|---|---|---|---|
| EC-YOLOv8: An Algorithm for Safety Helmet Detection Based on Improved YOLOv8 | 2026 in PDF / Applied Sciences | YOLOv8n baseline | ECA attention, CARAFE upsampling, EIoU loss | SHWD, split 7:2:1 | mAP@0.5 95.7%; mAP@0.5:0.95 74.6%; latency 5.8 ms; 172 FPS; 3.48M params; 9.2 GFLOPs on RTX 2060 SUPER | Strong small-target restoration; good speed/accuracy tradeoff; explicitly discusses construction clutter, blur, dense targets, and different lighting | Adds overhead versus YOLOv8n; still vulnerable to visually similar objects, extreme occlusion, and domain shift; not a dedicated yellow bucket/sign hard-negative benchmark |
| YOLO-CBF: Optimized YOLOv7 Algorithm for Helmet Detection in Road Environments | 2025 in PDF / Electronics | YOLOv7 | CoordConv, BiFormer/BRA, Focal-EIoU | Custom electric-vehicle road helmet dataset, VOC style, 962 images, split 7:1.5:1.5 | mAP@0.5 95.6%; P 94.7%; R 99%; +4.0 mAP over YOLOv7 | Directly matches the proposed CoordConv + BiFormer + Focal-EIoU hypothesis; strong occlusion and complex-road motivation | Dataset is small and road-domain, not construction SHWD; high compute cost noted; low light and severe occlusion remain limited |
| YOLOv8n-FADS: Enhancing Miners' Helmet Detection Accuracy in Complex Underground Environments | 2024 / Sensors | YOLOv8n | ASF-P2 head, Dilated Reparam Block, RepNCSPELAN, SEAMHead, Triplet attention, Focaler-IoU | Custom underground mine helmet dataset | Final mAP around 79.7% in paper table; +4.9 points accuracy claim; 36.7% parameter reduction; 29% memory compression | Strong small/occluded object emphasis; P2 head is highly relevant to distant helmets; reparameterized large/dilated kernels are deployable | Metrics are lower than SHWD papers due to harder mine setting; dataset not public in paper; lighting/headlamp domain differs from construction CCTV |
| RepVGG: Making VGG-style ConvNets Great Again | 2021 / CVPR | VGG-style ConvNet | Structural reparameterization: 3x3 branch, 1x1 branch, identity branch fused for inference | ImageNet and downstream tasks | Over 80% ImageNet top-1; faster than ResNet-50/101 on 1080Ti in paper | Provides the mathematical basis for RepConv/RepC3: multi-branch training, single Conv inference | Classification paper, not helmet detection; fusion must be implemented and verified carefully for custom YOLO modules |
| Focal and Efficient IoU Loss | 2021 / Neurocomputing / arXiv | Loss function, detector-agnostic | EIoU decomposes overlap, center distance, width, height; focal term reweights regression examples | Synthetic + COCO with multiple detectors | Reports improved convergence and localization over IoU/GIoU/DIoU/CIoU variants | Directly relevant to occluded boxes and aspect-ratio errors; reduces dominance of poor-quality anchors/examples | It is a regression loss, not a class-imbalance fix by itself; classification imbalance still needs focal/class weighting |
| CoordConv | 2018 / NeurIPS | Layer replacement / augmentation | Adds x/y coordinate channels before convolution | Toy coordinate transform, detection evidence, GAN/RL experiments | Solves coordinate transform toy task with far fewer params and faster learning; improves detection localization in paper experiments | Useful when location priors matter, such as helmet position relative to head/person and construction scene geometry | Can overfit dataset camera geometry if used too early or too widely; does not solve color confusion alone |
| BiFormer | 2023 / CVPR | Hierarchical vision transformer | Bi-Level Routing Attention (BRA), dynamic sparse attention | ImageNet, COCO detection, ADE20K segmentation | Strong dense-prediction results with content-aware sparse attention | Relevant for dense, occluded scenes where local windows are insufficient | Attention can raise latency and memory; needs careful placement in neck/head, not everywhere |
| YOLOv7 | 2022 arXiv / 2023 CVPR | E-ELAN family | Trainable bag-of-freebies, planned reparameterized conv, model scaling | COCO | 56.8 AP among real-time detectors at release; strong speed/accuracy range | Good basis for CBF-style ablation and RepConv ideas | Separate repo/training stack; less convenient than Ultralytics for Kaggle baselines |
| YOLOv9 | 2024 | GELAN | PGI (Programmable Gradient Information), GELAN | COCO | Improves parameter utilization and train-from-scratch behavior | Interesting for preserving information through deep networks, small object features | Integration/training is repo-specific; not first notebook priority |
| YOLOv10 | 2024 | YOLO family | Consistent dual assignments, NMS-free end-to-end inference, efficiency-accuracy design | COCO | Lower latency at comparable AP; YOLOv10-S reported faster than RT-DETR-R18 at similar AP | Strong candidate for low-latency target and RTSP | NMS-free behavior must be validated on crowded helmets; Ultralytics support/version must be pinned |
| YOLO11 | 2024 Ultralytics | Ultralytics YOLO family | Updated architecture/training recipes for detection/segmentation/pose/OBB | COCO and Ultralytics tasks | Official docs emphasize accuracy-speed-efficiency gains | Best stable first-pass baseline in current Ultralytics ecosystem | Not specifically helmet-focused; custom loss/module integration requires care |
| YOLO-LHD | 2023 / Frontiers in Built Environment | YOLOv8 | Coordinate attention, Focal loss, high-resolution features, large-scale detection head | Industrial helmet environments | Paper reports improved lightweight helmet detection; exact metrics should be copied from paper table during final writeup | Supports coordinate attention + focal loss + small-object head direction | Coordinate attention is not CoordConv; focal classification does not replace better box loss |
| FB-YOLOv7 | 2024 / PLOS ONE | YOLOv7-tiny family | FasterNet-style/lightweight modules, BiFPN-like feature fusion in reported method | Safety helmet detection dataset | Paper reports improved lightweight detection over YOLOv7-tiny | Useful for edge deployment and lightweight neck design | Tiny-model optimization may sacrifice hard-case recall if overcompressed |
| URD-YOLOv8 | 2025 / Scientific Reports / PMC | YOLOv8 | Improved upsampling/feature enhancement direction for helmet detection | Construction safety helmet data | Reports improved safety helmet detection in complex construction environments | Relevant for construction clutter and unclear target identification | Must verify exact module names/metrics from article tables before adopting |
| YOLOv8-CGS | 2025 / PLOS ONE / PMC | YOLOv8 | CBAM, GAM, SLOU-style loss | Construction helmet detection | Reports improved robustness in complex scenarios | Attention/loss combination is relevant as an alternative to ECA/CARAFE/Focal-EIoU | More modules may increase latency; loss compatibility with Ultralytics must be tested |

## Deep Analysis of Key Papers

### EC-YOLOv8

The EC-YOLOv8 paper is one of the closest SHWD references because it reports on the same benchmark family and provides both accuracy and runtime. Its final model combines:

- ECA attention after feature fusion / before detection heads, to recalibrate channels with low parameter overhead.
- CARAFE replacing standard upsampling, to recover small helmet details that simple nearest/bilinear upsampling may lose.
- EIoU loss replacing CIoU/DIoU/SIoU alternatives for better bounding-box localization.

The paper reports:

- YOLOv8n baseline: mAP@0.5 0.932, latency 5.2 ms, 192 FPS, 3.15M params, 8.7 GFLOPs.
- EC-YOLOv8: mAP@0.5 0.957, mAP@0.5:0.95 0.746, latency 5.8 ms, 172 FPS, 3.48M params, 9.2 GFLOPs.
- EIoU outperforms CIoU, DIoU, and SIoU in their ablation on the same improved network.

Project interpretation:

- CARAFE and a P2/small-object path should be tested before heavier global attention, because tiny helmets dominate many SHWD failure cases.
- ECA is a strong lightweight attention baseline. BiFormer should beat ECA on hard cases to justify its extra latency.
- Their paper explicitly acknowledges residual weaknesses: similar-looking yellow objects, extreme occlusion, domain shift, and deployment constraints on weaker edge hardware. These are exactly the project's hard-negative audit areas.

### YOLO-CBF

YOLO-CBF directly matches the proposed hybrid hypothesis:

- CoordConv improves spatial-position awareness.
- BiFormer/BRA enables dynamic sparse attention over relevant regions.
- Focal-EIoU improves bounding-box regression.

Reported ablation:

- YOLOv7 baseline: mAP 91.6, P 94.0, R 99.
- +CoordConv: mAP 93.2.
- +CoordConv + BiFormer: mAP 95.4.
- +CoordConv + BiFormer + Focal-EIoU: mAP 95.6, P 94.7, R 99.

Project interpretation:

- CoordConv is not optional in the ablation if yellow bucket/sign false positives matter. The paper shows a meaningful +1.6 mAP jump before attention.
- BiFormer delivers the largest incremental jump in the CBF ablation (+2.2 mAP), but this must be checked against FPS and memory on T4/PC.
- Focal-EIoU contributed only +0.2 mAP in their reported cumulative ablation, so in our project it should be tested with per-class `hat` recall and localization quality, not judged only by global mAP.
- The dataset is only 962 road images. Results may not transfer to construction CCTV without hard-negative and domain-diverse validation.

### YOLOv8n-FADS

YOLOv8n-FADS is useful because the underground mine domain stresses poor lighting, occlusion, and small targets. Its ingredients are:

- ASF-P2 detection head for higher-resolution small-object detection.
- Dilated Reparam Block and RepNCSPELAN-style backbone changes.
- SEAMHead for occlusion-aware feature handling.
- Triplet attention and Focaler-IoU loss.

Project interpretation:

- P2 detection is one of the strongest practical additions for tiny helmets and distant CCTV scenes.
- Reparameterized/dilated blocks align with the project's deployment goal: rich training representation, simpler inference path.
- SEAMHead-like occlusion handling should be compared against BiFormer. BiFormer may capture long-range context, while SEAM focuses on occlusion-aware local/head features.
- The paper's mAP values are not directly comparable with SHWD because the underground dataset is harder and apparently not the same public benchmark.

### RepVGG and RepConv Mechanics

RepVGG separates the training graph and inference graph. At training time, a block can include:

- A 3x3 Conv + BN branch.
- A 1x1 Conv + BN branch.
- An identity + BN branch when input/output channels and stride permit.

At inference, each branch is algebraically fused into an equivalent 3x3 convolution:

1. Fuse Conv + BN into a single kernel and bias:
   - `W_fused = gamma / sqrt(var + eps) * W`
   - `b_fused = beta - gamma * mean / sqrt(var + eps) + gamma / sqrt(var + eps) * b`
2. Pad the 1x1 kernel into the center of a 3x3 kernel.
3. Convert identity into a 3x3 Dirac kernel, then fuse its BN.
4. Sum all branch kernels and biases.

Why it matters for helmet detection:

- During training, the model can learn richer contours and local variations from multi-branch paths.
- During inference, the branch overhead disappears, preserving FPS.

Implementation caution:

- `switch_to_deploy()` must be called before ONNX/TensorRT benchmarking.
- Fused and unfused outputs should be compared numerically on a fixed random tensor before trusting exported speed.

### Focal-EIoU and Class Imbalance

EIoU decomposes bounding-box mismatch into:

- Overlap quality.
- Center distance.
- Width difference.
- Height difference.

This is more direct than CIoU's aspect-ratio term when helmets are partially visible, tiny, or oddly cropped. Focal-EIoU then reweights regression examples using IoU so training focuses more on useful localization examples and suppresses poor/outlier boxes.

Important limitation:

- Focal-EIoU is a box regression loss, not a complete solution to `hat/person` class imbalance.
- The project also needs classification-level balancing: focal classification loss, class weights, positive sampling, or hard-case mining with higher emphasis on `hat`.

Recommended loss policy:

- Baseline stage: use stock Ultralytics loss for comparability.
- Custom stage: add Focal-EIoU for box regression and focal/class-weighted classification for `hat`.
- Report whether the change improves `hat` recall and hard-negative false positives, not only global mAP.

### CoordConv for Helmet False Positives

Standard convolution is translation equivariant by design and may struggle when absolute/relative position matters. CoordConv appends coordinate channels to the feature map, commonly normalized x/y maps in `[-1, 1]`.

Why it can help:

- Helmets have contextual position: above the body/head region, with geometry related to the person.
- Yellow buckets, signs, and cones may share color/shape fragments but appear in different scene context and scale.
- Coordinate cues help the network decide when translation invariance is harmful.

Risk:

- Overusing CoordConv may overfit fixed camera layouts or dataset-specific viewpoints.
- Test it in early backbone/stem and neck positions separately. Do not blindly replace every convolution.

### BiFormer / Bi-Level Routing Attention

BiFormer uses Bi-Level Routing Attention:

1. Partition the feature map into coarse regions.
2. Route each query region to a small set of relevant key-value regions.
3. Apply token-level attention only inside selected routed regions.

Why it can help:

- Dense construction scenes contain many irrelevant tokens: scaffolding, equipment, signs, workers, shadows.
- Dynamic sparse attention can focus on relevant helmet/head contours without global full attention cost.
- It is more flexible than fixed local windows.

Risk:

- Even sparse attention can be slower than channel attention on T4/PC.
- Placement matters. Best candidates: neck/head high-level fusion layers, not every backbone stage.
- It should be compared against ECA/CBAM/Triplet attention using identical training settings.

## Benchmark Design Implications

### Stage 1: Stable Baseline Battle

Models:

- `yolov8n.pt`
- `yolov8s.pt`
- `yolov10n.pt`
- `yolov10s.pt`
- `yolo11n.pt`
- `yolo11s.pt`

Training:

- Dataset: converted VOC2028 from `trainval.txt` and `test.txt`.
- Epochs: 100 for official runs; 1 epoch for smoke test.
- Image size: 640.
- Device: `0,1` on Kaggle Dual T4.
- Save policy: keep only `best.pt` and essential logs.

Metrics:

- Overall `mAP@0.5`, `mAP@0.5:0.95`, precision, recall.
- Per-class AP/P/R/F1 for `hat` and `person`.
- Params, FLOPs, model size.
- PyTorch latency/FPS.
- ONNX Runtime CUDA latency/FPS on Kaggle.
- TensorRT FP16 latency/FPS on PC only.
- Hard-negative false positives on yellow distractors when a folder is available.

### Stage 2: Custom Ablation Study

Only run custom modules on the top-2 baselines from Stage 1.

Minimum ablation matrix:

| Experiment | Purpose |
|---|---|
| Baseline top-2 | Control |
| +P2 or CARAFE/P2-style small-object path | Recover tiny/distant helmets |
| +CoordConv | Reduce spatial/contextual false positives |
| +Lightweight attention (ECA/Triplet/CBAM) | Cheap feature recalibration baseline |
| +BiFormer/BRA | Stronger dynamic context modeling |
| +RepConv/RepC3 | Training-time richness, inference-time fusion |
| +Focal-EIoU | Better regression on occluded/extreme boxes |
| +Focal/class-weighted classification | Address `hat/person` imbalance |
| Combined best | Final candidate |

Promotion rule:

- A module is kept only if it improves either `hat` AP/recall, hard-negative FP rate, or mAP@0.5:0.95 without unacceptable latency cost.

## Failure Case Taxonomy

Every validation run should save qualitative examples in these buckets:

- False negative `hat`: tiny helmet, motion blur, shadow, headlamp glare, partial occlusion.
- False positive `hat`: yellow bucket, yellow traffic cone, yellow warning sign, reflective vest, machinery part.
- Class confusion: `person` head predicted as `hat`, or helmet predicted as `person`.
- Localization error: correct class but poor box due to occlusion or extreme aspect ratio.
- Domain shift: road scenes, construction scenes, mine scenes, low-light CCTV, rainy/foggy/weather-degraded frames.

## Kaggle and Deployment Notes

Kaggle:

- Use ONNX export and ONNX Runtime CUDA benchmarking.
- Avoid depending on TensorRT `.engine` export inside Kaggle because version mismatches between TensorRT, CUDA, Python bindings, and PyCUDA are common.
- Use symlinks for images to avoid duplicating the dataset into `/kaggle/working`.
- Keep `/kaggle/working` below 20GB by pruning `last.pt`, intermediate weights, cached predictions, and temporary exports.

PC deployment:

- Export TensorRT FP16 `.engine` on the actual PC GPU.
- Benchmark with warmup, fixed batch size, fixed image size, and the same preprocessing/postprocessing path expected for RTSP.
- Record total pipeline latency, not only model forward time: decode, resize/letterbox, inference, NMS/postprocess, draw/alert.

## Recommended First Deliverables

1. `SOTA_Literature_Review.md` for research grounding and Claude/agent handoff.
2. `kaggle_shwd_baseline.py` for reproducible conversion, training, validation, ONNX export, hard-negative audit, and cleanup.
3. `SHWD_Kaggle_Benchmark.ipynb` as the Kaggle-friendly notebook wrapper.
4. `CUSTOM_ABLATION_BLUEPRINT.md` after Stage 1 results exist, using the top-2 baselines only.

## Source Links

- EC-YOLOv8: https://www.mdpi.com/2076-3417/16/10/4613
- YOLO-CBF: https://www.mdpi.com/2079-9292/14/7/1413
- BiFormer: https://arxiv.org/abs/2303.08810
- CoordConv: https://arxiv.org/abs/1807.03247
- RepVGG: https://arxiv.org/abs/2101.03697
- Focal-EIoU: https://arxiv.org/abs/2101.08158
- YOLOv10: https://arxiv.org/abs/2405.14458
- YOLOv9: https://arxiv.org/abs/2402.13616
- YOLOv7: https://arxiv.org/abs/2207.02696
- YOLO11 docs: https://docs.ultralytics.com/models/yolo11
- Ultralytics training docs: https://docs.ultralytics.com/modes/train
- Ultralytics ONNX export docs: https://docs.ultralytics.com/integrations/onnx
- YOLO-LHD: https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2023.1288445/full
- FB-YOLOv7: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303866
- URD-YOLOv8: https://pmc.ncbi.nlm.nih.gov/articles/PMC12234873/
- YOLOv8-CGS: https://pmc.ncbi.nlm.nih.gov/articles/PMC12091821/
