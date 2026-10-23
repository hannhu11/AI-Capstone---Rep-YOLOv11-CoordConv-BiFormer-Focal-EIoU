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
                "print('=== /kaggle/input inspection ===')\n",
                "if os.path.exists('/kaggle/input'):\n",
                "    for root, dirs, files in os.walk('/kaggle/input'):\n",
                "        print(root, '-> dirs:', dirs, 'files count:', len(files))\n",
                "        break\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

payload = {
    "slug": "hannhu40022/test-dataset-attach-v2",
    "newTitle": "Test Dataset Attach V2",
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
