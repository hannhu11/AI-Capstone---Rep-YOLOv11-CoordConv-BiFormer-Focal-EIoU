# Kaggle Runbook for SHWD / VOC2028 Baselines

Upload these files to Kaggle together:

- `SHWD_Kaggle_Benchmark.ipynb`
- `kaggle_shwd_baseline.py`
- `requirements_kaggle.txt`
- `custom_ablation_modules.py`
- `albumentations_hardcase_policy.py`
- `CUSTOM_ABLATION_BLUEPRINT.md`

Attach the dataset so this path exists:

```text
/kaggle/input/datasets/hannhu4002/voc2028/VOC2028
```

## Recommended Run Order

1. Run the install cell:

```python
%pip install -q -U -r requirements_kaggle.txt
```

If `requirements_kaggle.txt` is not available in the notebook working directory, use:

```python
%pip install -q -U ultralytics onnx onnxruntime-gpu pandas
```

2. Run VOC-to-YOLO conversion:

```bash
python kaggle_shwd_baseline.py --mode convert --overwrite
```

Expected conversion summary:

- train images from `trainval.txt`: `6064`
- test images from `test.txt`: `1517`
- class counts: `hat=9044`, `person=111514`
- ignored labels: `dog=3`

3. Run the 1-epoch smoke test:

```bash
python kaggle_shwd_baseline.py --mode smoke --epochs 1 --device 0,1
```

4. Run the full baseline tournament:

```bash
python kaggle_shwd_baseline.py --mode all --epochs 100 --device 0,1
```

## Outputs

Main output folder:

```text
/kaggle/working/SHWD_YOLO
```

Important files:

- `shwd.yaml`
- `conversion_report.json`
- `ignored_labels.csv`
- `invalid_annotations.csv`
- `benchmark_results.csv`
- `BENCHMARK_RESULTS.md`
- `runs/<model>/weights/best.pt`
- `CUSTOM_ABLATION_BLUEPRINT.md` for Stage 2 custom modules after top-2 baselines are known

## Notes

- Kaggle speed checks use ONNX Runtime CUDA when available.
- TensorRT FP16 `.engine` export should be run on the target PC/RTX environment, not Kaggle.
- The script prunes `last.pt`, `epoch*.pt`, and cache files after each run to protect the `/kaggle/working` 20GB limit.
- Put yellow-bucket/sign/cone hard-negative images under `/kaggle/input/hard-negatives` to activate the false-positive audit.
- Stage 1 uses stable Ultralytics augmentation knobs. Stage 2 should use `albumentations_hardcase_policy.py` in a custom Dataset/Trainer for RandomShadow, illumination shift, HSV, and cutout-style occlusion.
