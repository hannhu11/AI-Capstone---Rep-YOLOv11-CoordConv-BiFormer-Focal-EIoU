import requests
import json

token = 'KGAT_e12378203b320b6808bb0a4f272aae4e'
headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

nb_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "print('Checking /kaggle/input:')\n",
                "if os.path.exists('/kaggle/input'):\n",
                "    print(os.listdir('/kaggle/input'))\n"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

payload = {
    "slug": "hannhu40022/test-dataset-attach",
    "newTitle": "Test Dataset Attach",
    "text": json.dumps(nb_content),
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": False,
    "enableInternet": True,
    "datasetDataSources": ["tundng111/gdut-hwd", "dngtrnphmtunse183674/iitu-safety-helmet"]
}

r = requests.post('https://www.kaggle.com/api/v1/kernels/push', json=payload, headers=headers)
print('Push Status:', r.status_code, r.text)
