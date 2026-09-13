import pandas as pd
from pathlib import Path

print('=== 1. BASELINE CONSOLIDATED ===')
p_base = Path('Output/SHWD_Baseline_Consolidated_2/master_benchmark_results.csv')
if p_base.exists():
    df = pd.read_csv(p_base)
    print("Columns:", list(df.columns))
    print(df.to_string())

print('\n=== 2. ABLATION RUNS A0 - A6 ===')
ablations = [
    ('A0 Control', 'Output/SHWD_Stage2_Ablation_Setup_full_train_RUN_A1/csv_results/A0_control_results.csv'),
    ('A1 HardCaseAug', 'Output/SHWD_Stage2_Ablation_Setup_full_train_RUN_A1/csv_results/A1_hardcase_aug_train_yolo11s_results.csv'),
    ('A2 CoordConv', 'Output/shwd-stage2-ablation-setup-full-train-run-a2/csv_results/A2_coordconv_train_yolo11s_results.csv'),
    ('A3 RepConv', 'Output/shwd-stage2-ablation-setup-full-train-run-a3/csv_results/A3_repconv_train_yolo11s_results.csv'),
    ('A4 FocalEIoU', 'Output/shwd-stage2-ablation-setup-full-train-run-a4/csv_results/A4_focal_eiou_train_yolo11s_results.csv'),
    ('A5 BiFormer', 'Output/shwd-stage2-ablation-setup-full-train-run-a5/csv_results/A5_biformer_train_yolo11s_results.csv'),
    ('A6 FullFusion', 'Output/shwd-stage2-ablation-setup-full-train-run-a6/csv_results/A6_full_fusion_train_yolo11s_results.csv'),
]
for tag, p in ablations:
    path = Path(p)
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        col_map50 = [c for c in df.columns if 'map50' in c.lower() and '95' not in c][0]
        col_map5095 = [c for c in df.columns if '50-95' in c or '0.5:0.95' in c][0]
        col_p = [c for c in df.columns if 'precision' in c.lower()][0]
        col_r = [c for c in df.columns if 'recall' in c.lower()][0]
        idx = df[col_map5095].idxmax()
        row = df.loc[idx]
        ep = row.get('epoch', idx)
        print(f'{tag:15s}: Best Epoch={ep:2} | P={row[col_p]*100:.2f}% | R={row[col_r]*100:.2f}% | mAP50={row[col_map50]*100:.2f}% | mAP50-95={row[col_map5095]*100:.2f}%')

print('\n=== 3. STAGE 3 RUNS ===')
s3_runs = [
    ('Stage 3 Fix-4', 'Output/shwd-stage-3-kaggle-master-research-pipeline_fix_4'),
    ('Stage 3 Fix-5', 'Output/shwd-stage-3-kaggle-master-research-pipeline-fix-5'),
    ('Stage 3 Fix-6', 'Output/shwd-stage-3-kaggle-master-research-pipeline-fix-6'),
]
for tag, folder in s3_runs:
    p_csv = Path(folder) / 'runs/detect/runs/detect/stage3_hard_augment_finetune/results.csv'
    if not p_csv.exists():
        p_csv = Path(folder) / 'runs/detect/stage3_hard_augment_finetune/results.csv'
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        df.columns = [c.strip() for c in df.columns]
        col_map50 = [c for c in df.columns if 'map50' in c.lower() and '95' not in c][0]
        col_map5095 = [c for c in df.columns if '50-95' in c or '0.5:0.95' in c][0]
        col_p = [c for c in df.columns if 'precision' in c.lower()][0]
        col_r = [c for c in df.columns if 'recall' in c.lower()][0]
        idx = df[col_map5095].idxmax()
        row = df.loc[idx]
        ep = row.get('epoch', idx)
        print(f'{tag:15s} (Finetune): Best Epoch={ep:2} | P={row[col_p]*100:.2f}% | R={row[col_r]*100:.2f}% | mAP50={row[col_map50]*100:.2f}% | mAP50-95={row[col_map5095]*100:.2f}%')
    
    p_kf = Path(folder) / 'SHWD_YOLO_KFOLD/kfold_statistical_report.csv'
    if p_kf.exists():
        df_kf = pd.read_csv(p_kf)
        print(f'{tag} 5-Fold summary:\n', df_kf[['Fold', 'mAP50', 'mAP50-95', 'Precision', 'Recall']].to_string())