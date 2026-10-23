with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if '"""' in line:
        print(f'{idx+1}: {line.strip()[:60]}')
