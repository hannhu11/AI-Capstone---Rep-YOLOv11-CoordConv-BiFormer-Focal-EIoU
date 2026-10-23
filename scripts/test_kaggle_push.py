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
            "source": ["print('Kaggle API Bearer Token Push Verified Successfully!')\n"]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

payload = {
    "slug": "hannhu40022/test-api-ping",
    "newTitle": "Test API Ping",
    "text": json.dumps(nb_content),
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": False,
    "enableInternet": True,
    "datasetDataSources": []
}

r = requests.post('https://www.kaggle.com/api/v1/kernels/push', json=payload, headers=headers)
print('Push Status:', r.status_code, r.text)
