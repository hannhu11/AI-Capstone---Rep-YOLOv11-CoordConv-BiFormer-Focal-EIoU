import urllib.request
import json

def check_doi(doi):
    url = f'https://api.crossref.org/works/{doi}'
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicReferenceChecker/1.0 (mailto:test@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['message']
            title = data.get('title', [''])[0]
            authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in data.get('author', [])]
            container = data.get('container-title', [''])[0]
            vol = data.get('volume', '')
            issue = data.get('issue', '')
            year = data.get('published', {}).get('date-parts', [[None]])[0][0]
            page = data.get('page', '')
            article_number = data.get('article-number', '')
            print(f'DOI: {doi}')
            print(f'  Title: {title}')
            print(f'  Authors: {"; ".join(authors)}')
            print(f'  Container: {container}, Vol: {vol}, Issue: {issue}, Year: {year}, Page: {page}, ArtNo: {article_number}')
    except Exception as e:
        print(f'DOI {doi} error: {e}')

dois = [
    '10.3390/jimaging11090320',
    '10.3390/s24123767',
    '10.3390/app16104613',
    '10.3390/electronics14071413',
    '10.1016/j.neucom.2022.07.042',
    '10.1038/s41598-024-68446-z',
    '10.1109/TII.2021.3116167', # gdut
    '10.3389/fbuil.2023.1288445'
]

for d in dois:
    check_doi(d)
