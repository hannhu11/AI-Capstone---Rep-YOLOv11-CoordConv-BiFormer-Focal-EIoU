import re

with open('AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# find all sentences containing \cite{...}
sentences = re.split(r'(?<=[.!?])\s+', text)
for s in sentences:
    if '\\cite{' in s:
        # extract keys
        keys = re.findall(r'\\cite\{([^}]+)\}', s)
        flat_keys = []
        for k in keys:
            flat_keys.extend([x.strip() for x in k.split(',')])
        clean_s = ' '.join(s.strip().split())
        print(f"Keys: {flat_keys}")
        print(f"Context: {clean_s}\n")
