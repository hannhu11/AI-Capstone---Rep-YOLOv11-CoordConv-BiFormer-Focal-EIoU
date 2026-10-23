import json
from pathlib import Path

targets = [
    'Dataset_Strategy_Transfer_Learning_Benchmark.ipynb',
    'Multi_Generation_YOLO_Baseline_Selection.ipynb',
    'Rep_YOLO11s_WarmStart_Ablations_And_RuleEngine.ipynb'
]

for t in targets:
    p = Path('notebooks') / t
    data = json.loads(p.read_text(encoding='utf-8'))
    for i, c in enumerate(data['cells']):
        if c['cell_type'] == 'code':
            code_str = ''.join(c['source'])
            compile(code_str, f'{t}_cell_{i}', 'exec')
    print(f"PASSED: {t} ({len(data['cells'])} cells)")
