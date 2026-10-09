import urllib.request
import json
import urllib.parse

def search_crossref(title):
    query = urllib.parse.quote(title)
    url = f'https://api.crossref.org/works?query.title={query}&rows=3'
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicChecker/1.0 (mailto:test@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.loads(resp.read().decode('utf-8'))['message']['items']
            print(f'=== SEARCH: {title} ===')
            for item in items[:2]:
                t = item.get('title', [''])[0]
                authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in item.get('author', [])]
                doi = item.get('DOI', '')
                container = item.get('container-title', [''])[0]
                year = item.get('published', {}).get('date-parts', [[None]])[0][0]
                print(f'  Match: {t}')
                print(f'    Authors: {"; ".join(authors)}')
                print(f'    Venue: {container} ({year}), DOI: {doi}')
    except Exception as e:
        print(f'Error searching {title}: {e}')

titles = [
    "DABFNet: Dual-Attention Bilateral Feature Pyramid Network",
    "MAF-YOLO: Multi-Scale Adaptive Fusion Network",
    "TinyFormer: A Lightweight Transformer for Tiny Object Detection",
    "GDUT-HWD: A Large-Scale Safety Helmet Wearing Dataset",
    "SFCHD: Safety Helmet Face and Head Detection Dataset",
    "RepVGG: Making VGG-style ConvNets Great Again",
    "BiFormer: Vision Transformer with Bi-Level Routing Attention",
    "An intriguing failing of convolutional neural networks and the CoordConv solution",
    "YOLOv7: Trainable bag-of-freebies",
    "YOLOv10: Real-Time End-to-End Object Detection",
    "TOOD: Task-aligned one-stage object detection",
    "Generalized focal loss: Learning qualified and distributed",
    "VMamba: Visual State Space Model",
    "Soft-NMS--Improving Object Detection With One Line of Code",
    "Path aggregation network for instance segmentation"
]

for t in titles:
    search_crossref(t)
