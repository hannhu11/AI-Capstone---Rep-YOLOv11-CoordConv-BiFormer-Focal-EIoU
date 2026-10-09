import re

with open('AAIML 2027/conference-latex-template_10-17-19/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),(.*?)\n\}', text, re.DOTALL)
entry_dict = {key: (t, body) for t, key, body in entries}

with open('AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

citations = []
for c in re.findall(r'\\cite\{([^}]+)\}', tex):
    for k in c.split(','):
        if k.strip() not in citations:
            citations.append(k.strip())

print(f'Total cited entries in order: {len(citations)}')
for idx, key in enumerate(citations):
    entry_type, body = entry_dict[key]
    fields = {}
    for line in body.split('\n'):
        m = re.match(r'([a-zA-Z0-9_-]+)\s*=\s*[\{"](.*)[\}"]', line.strip())
        if m:
            fields[m.group(1).lower()] = m.group(2).rstrip('}",')
    print(f"[{idx+1}] Key: {key} ({entry_type})")
    print(f"     Title: {fields.get('title')}")
    print(f"     Author: {fields.get('author')}")
    venue = fields.get('journal') or fields.get('booktitle') or fields.get('howpublished')
    print(f"     Venue: {venue}")
    print(f"     Year: {fields.get('year')}")
    if 'doi' in fields:
        print(f"     DOI: {fields.get('doi')}")
