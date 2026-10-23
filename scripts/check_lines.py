with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(895, 908):
    print(i+1, repr(lines[i]))
