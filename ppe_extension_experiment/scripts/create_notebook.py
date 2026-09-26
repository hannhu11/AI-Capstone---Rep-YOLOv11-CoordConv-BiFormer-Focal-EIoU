import json
from pathlib import Path

nb = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 🛡️ Rep-YOLO11s Extension: 3-Class Safety Helmet & PPE (Vest) Detection\n',
                '**Project**: Real-Time Safety Helmet & Personal Protective Equipment Detection  \n',
                '**Author**: Nguyen Han Nhu (Như)  \n',
                '**Objective**: Evaluate Hướng 2 (Teacher Huy\'s feedback) - Adding Personal Protective Equipment (`vest`) to existing `hat` & `person` detector.  \n',
                '**Architecture**: Rep-YOLO11s (CoordConv + RepConv + BiFormer + Focal-EIoU)  \n',
                '**Dataset**: CHV (Color Helmet and Vest - Wang et al., Sensors 2021)  \n',
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Environment & Hardware Verification\n',
                'Check PyTorch, GPU availability, and Ultralytics library.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import torch\n',
                'import sys\n',
                'print(f"Python Version: {sys.version}")\n',
                'print(f"PyTorch Version: {torch.__version__}")\n',
                'print(f"CUDA Available: {torch.cuda.is_available()}")\n',
                'if torch.cuda.is_available():\n',
                '    print(f"GPU Device: {torch.cuda.get_device_name(0)}")\n',
                '    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")\n',
                '\n',
                'from ultralytics import YOLO\n',
                'print("Ultralytics YOLO loaded successfully!")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Dataset Setup & Canonical 3-Class Mapping\n',
                'The CHV benchmark dataset is standardized into 3 non-conflicting classes:\n',
                '- Class 0: `hat` (aggregated from blue, red, white, yellow helmets)\n',
                '- Class 1: `person` (from person)\n',
                '- Class 2: `vest` (safety vest / protective equipment)\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'from pathlib import Path\n',
                'import yaml\n',
                '\n',
                '# Configure paths\n',
                'DATA_YAML = Path("../dataset/STANDARDIZED_CHV/chv_3class.yaml").resolve()\n',
                'if not DATA_YAML.exists():\n',
                '    DATA_YAML = Path("ppe_extension_experiment/dataset/STANDARDIZED_CHV/chv_3class.yaml").resolve()\n',
                '\n',
                'print(f"Dataset YAML: {DATA_YAML}")\n',
                'with open(DATA_YAML, "r") as f:\n',
                '    config = yaml.safe_load(f)\n',
                'print("YAML Configuration:")\n',
                'print(yaml.dump(config, default_flow_style=False))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Warm-Start Transfer Learning\n',
                'Load the best trained weights (`yolo11s_best.pt` from Stage 2/3 SHWD) to transfer all Backbone and Neck spatial representations. Ultralytics automatically rebuilds the Detect head for 3 classes.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Find base checkpoint\n',
                'ckpt_candidates = [\n',
                '    Path("Output/shwd-stage-3-kaggle-master-research-pipeline-4/yolo11s_best.pt"),\n',
                '    Path("../Output/shwd-stage-3-kaggle-master-research-pipeline-4/yolo11s_best.pt"),\n',
                '    Path("yolo11s_best.pt")\n',
                ']\n',
                'base_ckpt = next((c for c in ckpt_candidates if c.exists()), None)\n',
                'if base_ckpt is None:\n',
                '    for pt in Path(".").glob("**/yolo11s_best.pt"):\n',
                '        base_ckpt = pt\n',
                '        break\n',
                '\n',
                'print(f"Loaded Base Checkpoint: {base_ckpt}")\n',
                'model = YOLO(str(base_ckpt))\n',
                'print("Model architecture ready for 3-class adaptation.")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. Fine-Tuning Execution\n',
                'Train for 25-30 epochs on the 3-class dataset with cosine annealing learning rate scheduler.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'EPOCHS = 25\n',
                'IMGSZ = 640\n',
                'BATCH = 16\n',
                'DEVICE = 0 if torch.cuda.is_available() else "cpu"\n',
                '\n',
                'train_results = model.train(\n',
                '    data=str(DATA_YAML),\n',
                '    epochs=EPOCHS,\n',
                '    imgsz=IMGSZ,\n',
                '    batch=BATCH,\n',
                '    device=DEVICE,\n',
                '    project="ppe_results",\n',
                '    name="chv_3class_finetune",\n',
                '    exist_ok=True,\n',
                '    save=True,\n',
                '    plots=True,\n',
                '    workers=4,\n',
                '    lr0=0.003,\n',
                '    lrf=0.01,\n',
                '    seed=3407,\n',
                '    deterministic=True\n',
                ')\n',
                'print("Fine-tuning complete!")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Comprehensive 3-Class Test Split Evaluation\n',
                'Evaluate the best fine-tuned model on the independent test split and inspect per-class metrics ($mAP_{50}$, Precision, Recall, F1).\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'best_model_path = Path("ppe_results/chv_3class_finetune/weights/best.pt")\n',
                'eval_model = YOLO(str(best_model_path))\n',
                '\n',
                'metrics = eval_model.val(\n',
                '    data=str(DATA_YAML),\n',
                '    split="test",\n',
                '    imgsz=IMGSZ,\n',
                '    device=DEVICE,\n',
                '    plots=True,\n',
                '    project="ppe_results",\n',
                '    name="test_eval",\n',
                '    exist_ok=True\n',
                ')\n',
                '\n',
                'class_names = ["hat", "person", "vest"]\n',
                'print("\\n=================== PER-CLASS METRICS ===================")\n',
                'for idx, name in enumerate(class_names):\n',
                '    ap50 = metrics.box.ap50[idx] * 100\n',
                '    ap = metrics.box.maps[idx] * 100\n',
                '    p = metrics.box.p[idx] * 100\n',
                '    r = metrics.box.r[idx] * 100\n',
                '    f1 = metrics.box.f1[idx]\n',
                '    print(f"[{name.upper():<6}] mAP50: {ap50:.2f}% | mAP50-95: {ap:.2f}% | Recall: {r:.2f}% | Precision: {p:.2f}% | F1: {f1:.4f}")\n',
                '\n',
                'print(f"\\nMEAN OVER ALL CLASSES:")\n',
                'print(f"mAP50: {metrics.box.map50 * 100:.2f}%")\n',
                'print(f"mAP50-95: {metrics.box.map * 100:.2f}%")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 6. Visualizing Sample Predictions (Hat, Person, Vest)\n',
                'Display sample inference predictions showing simultaneous detection of helmets and safety vests.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import matplotlib.pyplot as plt\n',
                'from PIL import Image\n',
                'import glob\n',
                '\n',
                'val_images = list(Path("../dataset/STANDARDIZED_CHV/images/test").glob("*.jpg"))\n',
                'if not val_images:\n',
                '    val_images = list(Path("ppe_extension_experiment/dataset/STANDARDIZED_CHV/images/test").glob("*.jpg"))\n',
                '\n',
                'if val_images:\n',
                '    sample_img = str(val_images[0])\n',
                '    results = eval_model.predict(sample_img, imgsz=640, conf=0.35)\n',
                '    res_plotted = results[0].plot()\n',
                '    plt.figure(figsize=(10, 10))\n',
                '    plt.imshow(res_plotted[:, :, ::-1])\n',
                '    plt.axis("off")\n',
                '    plt.title("Rep-YOLO11s PPE Detection Sample (Hat, Person, Vest)")\n',
                '    plt.show()\n'
            ]
        }
    ],
    'metadata': {
        'language_info': {'name': 'python'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

out_nb = Path('ppe_extension_experiment/notebooks/SHWD_PPE_3Class_Finetune_and_Benchmark.ipynb')
out_nb.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Successfully built notebook: {out_nb}')
