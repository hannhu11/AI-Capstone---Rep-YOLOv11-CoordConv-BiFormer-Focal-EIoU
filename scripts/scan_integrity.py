import os
import re

targets = [
    'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.tex',
    'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027_DoubleBlind.tex',
    'paper_overleaf/main.tex',
    'AAIML 2027/conference-latex-template_10-17-19/references.bib',
    'paper_overleaf/references.bib'
]

keywords = ['jetson', 'orin', '909', 'xavier', 'coral', 'npu', 'embedded device', 'embedded devices']

for target in targets:
    if os.path.exists(target):
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f'=== Scanning {target} ===')
        for kw in keywords:
            matches = [m.start() for m in re.finditer(re.escape(kw), content, re.IGNORECASE)]
            if matches:
                print(f'  [FOUND] keyword "{kw}": {len(matches)} times')
                for m in matches:
                    start = max(0, m - 40)
                    end = min(len(content), m + 40)
                    snippet = content[start:end].replace('\n', ' ')
                    print(f'     ...{snippet}...')
            else:
                print(f'  [CLEAN] keyword "{kw}": 0 occurrences')
