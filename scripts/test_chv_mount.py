import requests
import json

token = 'KGAT_e12378203b320b6808bb0a4f272aae4e'
headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

nb_content = {
    'cells': [
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                "import os\n",
                "print('=== /kaggle/input full tree inspection ===')\n",
                "for root, dirs, files in os.walk('/kaggle/input'):\n",
                "    depth = root.replace('/kaggle/input', '').count(os.sep)\n",
                "    if depth <= 3:\n",
                "        print(root, 'dirs:', dirs[:5], 'file count:', len(files))\n"
            ]
        }
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.12'}
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

payload = {
    'slug': 'hannhu40022/test-chv-mount',
    'newTitle': 'Test CHV Mount',
    'text': json.dumps(nb_content),
    'language': 'python',
    'kernelType': 'notebook',
    'isPrivate': True,
    'enableGpu': False,
    'enableInternet': True,
    'datasetDataSources': ['tundng111/gdut-hwd', 'dngtrnphmtunse183674/iitu-safety-helmet', 'hannhu4002/chv-dataset']
}

r = requests.post('https://www.kaggle.com/api/v1/kernels/push', json=payload, headers=headers)
print('Push Status:', r.status_code, r.text)
