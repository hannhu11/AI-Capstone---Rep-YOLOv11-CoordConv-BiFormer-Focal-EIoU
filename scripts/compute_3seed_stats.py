import json
import numpy as np

def get_results(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        for out in c.get('outputs', []):
            text = ''.join(out.get('text', []))
            if '| ablation_id' in text:
                lines = text.strip().split('\n')
                data = {}
                for line in lines:
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) >= 8 and parts[0] in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
                        ab_id = parts[0]
                        name = parts[1]
                        seed = int(parts[2])
                        map50 = float(parts[3])
                        map50_95 = float(parts[4])
                        prec = float(parts[5])
                        rec = float(parts[6])
                        data[ab_id] = {'name': name, 'seed': seed, 'mAP50': map50, 'mAP50_95': map50_95, 'prec': prec, 'rec': rec}
                return data
    return None

s42 = get_results('kaggle-taskb2-resume-seed42-ablation-6.ipynb')
s1337 = get_results('kaggle-taskb2-resume-seed1337-ablation-6.ipynb')
s2026 = get_results('kaggle-taskb2-resume-seed2026-ablation-6.ipynb')

print("All seeds loaded successfully:")
for ab in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
    v42_50, v42_95 = s42[ab]['mAP50'], s42[ab]['mAP50_95']
    v1337_50, v1337_95 = s1337[ab]['mAP50'], s1337[ab]['mAP50_95']
    v2026_50, v2026_95 = s2026[ab]['mAP50'], s2026[ab]['mAP50_95']
    
    m50 = [v42_50, v1337_50, v2026_50]
    m95 = [v42_95, v1337_95, v2026_95]
    
    mean50, std50 = np.mean(m50), np.std(m50, ddof=1)
    mean95, std95 = np.mean(m95), np.std(m95, ddof=1)
    name = s42[ab]['name']
    print(f"{ab:2s} | {name:25s} | 42: {v42_50:5.2f}/{v42_95:5.2f} | 1337: {v1337_50:5.2f}/{v1337_95:5.2f} | 2026: {v2026_50:5.2f}/{v2026_95:5.2f} | Mean: {mean50:5.2f}+/-{std50:4.2f} / {mean95:5.2f}+/-{std95:4.2f}")
