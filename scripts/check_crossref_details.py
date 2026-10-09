import urllib.request
import urllib.parse
import json

def fetch_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={'User-Agent': 'RefAudit/1.0 (mailto:audit@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read().decode('utf-8'))['message']
            title = d.get('title', [''])[0]
            authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in d.get('author', [])]
            venue = d.get('container-title', [''])[0]
            year = d.get('published', {}).get('date-parts', [[None]])[0][0]
            vol = d.get('volume', '')
            issue = d.get('issue', '')
            page = d.get('page', '')
            art = d.get('article-number', '')
            return {'title': title, 'authors': authors, 'venue': venue, 'year': year, 'vol': vol, 'issue': issue, 'page': page or art}
    except Exception as e:
        return {'error': str(e)}

def search_title(title):
    q = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.title={q}&rows=2"
    req = urllib.request.Request(url, headers={'User-Agent': 'RefAudit/1.0 (mailto:audit@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.loads(resp.read().decode('utf-8'))['message']['items']
            if items:
                it = items[0]
                t = it.get('title', [''])[0]
                authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in it.get('author', [])]
                venue = it.get('container-title', [''])[0]
                year = it.get('published', {}).get('date-parts', [[None]])[0][0]
                doi = it.get('DOI', '')
                return {'title': t, 'authors': authors, 'venue': venue, 'year': year, 'doi': doi}
    except Exception as e:
        return {'error': str(e)}
    return None

doi_items = {
    'ecyolov8_2024': '10.3390/app16104613',
    'yolocbf_2025': '10.3390/electronics14071413',
    'zhao2025yolodcrcf': '10.3390/jimaging11090320',
    'fuYolov8nFADSStudyEnhancing2024': '10.3390/s24123767',
    'zhang2022focal': '10.1016/j.neucom.2022.07.042',
    'gdut_hwd_2021': '10.1016/j.autcon.2019.102894',
}

for k, doi in doi_items.items():
    print(f"=== DOI {k} ({doi}) ===")
    print(" ", fetch_doi(doi))

title_items = {
    'wang2023yolov7': 'YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors',
    'ding2021repvgg': 'RepVGG: Making VGG-style ConvNets Great Again',
    'zhu2023biformer': 'BiFormer: Vision Transformer with Bi-Level Routing Attention',
    'liu2018coordconv': 'An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution',
    'liu2024vmamba': 'VMamba: Visual State Space Model',
    'lin2017feature': 'Feature Pyramid Networks for Object Detection',
    'liu2018path': 'Path Aggregation Network for Instance Segmentation',
    'feng2021tood': 'TOOD: Task-Aligned One-Stage Object Detection',
    'li2020generalized': 'Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection',
    'bodla2017soft': 'Soft-NMS--Improving Object Detection With One Line of Code',
}

for k, title in title_items.items():
    print(f"=== TITLE {k} ===")
    print(" ", search_title(title))
