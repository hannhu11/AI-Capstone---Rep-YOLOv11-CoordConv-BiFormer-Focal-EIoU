import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET

def fetch_arxiv(aid):
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    req = urllib.request.Request(url, headers={'User-Agent': 'RefAudit/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            if entry is not None:
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
                authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                return {'title': title, 'authors': authors, 'year': published[:4]}
    except Exception as e:
        return {'error': str(e)}
    return None

def fetch_crossref(query):
    q = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query={q}&rows=3"
    req = urllib.request.Request(url, headers={'User-Agent': 'RefAudit/1.0 (mailto:audit@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.loads(resp.read().decode('utf-8'))['message']['items']
            results = []
            for it in items:
                title = it.get('title', [''])[0]
                authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in it.get('author', [])]
                venue = it.get('container-title', [''])[0]
                year = it.get('published', {}).get('date-parts', [[None]])[0][0]
                doi = it.get('DOI', '')
                results.append({'title': title, 'authors': authors, 'venue': venue, 'year': year, 'doi': doi})
            return results
    except Exception as e:
        return [{'error': str(e)}]

arxiv_checks = {
    'wang2024yolov10': '2405.14458',
    'li2023yolov6': '2301.05586',
    'yolo26_2025': '2510.09653',
    'dabfnet2024': '2411.19071',
    'mafyolo2024': '2407.04381',
    'tinyformer2024': '2605.25046',
    'tong2023wise': '2301.10051',
}

for k, aid in arxiv_checks.items():
    print(f"=== ARXIV {k} ({aid}) ===")
    res = fetch_arxiv(aid)
    print(" ", res)
