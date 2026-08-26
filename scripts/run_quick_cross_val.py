import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

base_dir = Path('.').resolve()
shwd_yaml = base_dir / 'SHWD_YOLO' / 'shwd.yaml'
gdut_yaml = base_dir / 'Dataset' / 'STANDARDIZED' / 'GDUT_HWD' / 'gdut_hwd.yaml'
shel_yaml = base_dir / 'Dataset' / 'STANDARDIZED' / 'SHEL5K' / 'shel5k.yaml'

ckpt_a6 = base_dir / 'Output' / 'shwd-stage2-ablation-setup-full-train-run-a6' / 'weights' / 'yolo11s_best.pt'
ckpt_fix4 = base_dir / 'Output' / 'shwd-stage-3-kaggle-master-research-pipeline-4' / 'runs' / 'detect' / 'runs' / 'detect' / 'stage3_hard_augment_finetune' / 'weights' / 'best.pt'
ckpt_fix5 = base_dir / 'Output' / 'shwd-stage-3-kaggle-master-research-pipeline-fix-5' / 'runs' / 'detect' / 'runs' / 'detect' / 'stage3_hard_augment_finetune' / 'weights' / 'best.pt'

checkpoints = {
    'A6_YOLO11s_Baseline': ckpt_a6,
    'Stage3_Fix4_P2': ckpt_fix4,
    'Stage3_Fix5_P2_1024': ckpt_fix5,
}

benchmarks = [
    ('In-Domain (SHWD Val)', str(shwd_yaml.resolve()), 'val'),
    ('Zero-Shot Cross-Domain (GDUT-HWD Test)', str(gdut_yaml.resolve()), 'test'),
    ('Zero-Shot Cross-Domain (SHEL5K Val)', str(shel_yaml.resolve()), 'val')
]

records = []
for m_name, ckpt_p in checkpoints.items():
    if not ckpt_p.exists():
        print(f'Skipping missing: {ckpt_p}')
        continue
    print(f'\n======================================================')
    print(f'⚡ Evaluating {m_name} ({ckpt_p.name})...')
    print(f'======================================================')
    try:
        model = YOLO(str(ckpt_p.resolve()))
    except Exception as e:
        print(f'Error loading {m_name}: {e}')
        continue
        
    for d_name, d_yaml, split_mode in benchmarks:
        print(f'  -> Testing on {d_name} (split={split_mode})...')
        try:
            res = model.val(data=d_yaml, split=split_mode, imgsz=640, batch=8, workers=0, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)
            m50 = res.results_dict.get('metrics/mAP50(B)', 0.0) * 100
            m5095 = res.results_dict.get('metrics/mAP50-95(B)', 0.0) * 100
            p = res.results_dict.get('metrics/precision(B)', 0.0) * 100
            r = res.results_dict.get('metrics/recall(B)', 0.0) * 100
            records.append({
                'Model': m_name,
                'Benchmark Dataset': d_name,
                'mAP@0.50': f'{m50:.2f}%',
                'mAP@0.50:0.95': f'{m5095:.2f}%',
                'Precision': f'{p:.2f}%',
                'Recall': f'{r:.2f}%'
            })
            print(f'     ✅ {d_name}: mAP50 = {m50:.2f}%, mAP50-95 = {m5095:.2f}%, P = {p:.2f}%, R = {r:.2f}%')
        except Exception as e:
            print(f'     ❌ Error: {e}')

df = pd.DataFrame(records)
print('\n' + '='*70)
print('FINAL CROSS-DOMAIN BENCHMARK AUDIT TABLE:')
print('='*70)
print(df.to_string(index=False))
results_dir = Path('results')
results_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(results_dir / 'cross_domain_benchmark_empirical_audit.csv', index=False)
print(f'\n✅ Saved to {results_dir / "cross_domain_benchmark_empirical_audit.csv"}')

