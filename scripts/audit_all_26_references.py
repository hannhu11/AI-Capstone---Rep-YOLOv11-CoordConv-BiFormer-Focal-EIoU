import re
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET

def check_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={'User-Agent': 'RefAudit/1.0 (mailto:audit@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['message']
            title = data.get('title', [''])[0]
            authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in data.get('author', [])]
            return True, f"Crossref Match: Title='{title}', Authors='{'; '.join(authors[:3])}...'"
    except Exception as e:
        return False, f"Crossref Error: {e}"

def check_arxiv(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'RefAudit/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            if entry is not None:
                title_el = entry.find('{http://www.w3.org/2005/Atom}title')
                authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
                title = title_el.text.strip().replace('\n', ' ') if title_el is not None else ''
                return True, f"Arxiv Match: Title='{title}', Authors='{'; '.join(authors[:3])}...'"
            return False, "Arxiv entry not found"
    except Exception as e:
        return False, f"Arxiv Error: {e}"

def check_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, f"URL OK: HTTP {resp.status}"
    except Exception as e:
        return False, f"URL Error: {e}"

with open('AAIML 2027/conference-latex-template_10-17-19/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),(.*?)\n\}', text, re.DOTALL)
print(f"Total bib entries: {len(entries)}")

for entry_type, key, body in entries:
    fields = {}
    for line in body.split('\n'):
        line = line.strip()
        m = re.match(r'([a-zA-Z0-9_-]+)\s*=\s*[\{"](.*)[\}"]', line)
        if m:
            k = m.group(1).lower()
            v = m.group(2).rstrip('}",')
            fields[k] = v
            
    title = fields.get('title', '').replace('{', '').replace('}', '')
    author = fields.get('author', '')
    doi = fields.get('doi', '')
    journal = fields.get('journal', '')
    booktitle = fields.get('booktitle', '')
    howpub = fields.get('howpublished', '')
    
    print(f"\n==========================================")
    print(f"KEY: {key} ({entry_type})")
    print(f"  Title: {title}")
    print(f"  Authors: {author}")
    print(f"  Venue: {journal or booktitle or howpub}")
    
    # Check DOI
    if doi:
        ok, msg = check_doi(doi)
        print(f"  [DOI {doi}]: {'PASS' if ok else 'FAIL'} -> {msg}")
    
    # Check arXiv
    arxiv_m = re.search(r'arXiv(?::|\s+preprint\s+arXiv:?)\s*([0-9]+\.[0-9]+)', journal or title or howpub, re.IGNORECASE)
    if arxiv_m:
        aid = arxiv_m.group(1)
        ok, msg = check_arxiv(aid)
        print(f"  [arXiv {aid}]: {'PASS' if ok else 'FAIL'} -> {msg}")
        
    # Check URL
    url_m = re.search(r'\\url\{([^}]+)\}', howpub)
    if url_m:
        u = url_m.group(1)
        ok, msg = check_url(u)
        print(f"  [URL {u}]: {'PASS' if ok else 'FAIL'} -> {msg}")
