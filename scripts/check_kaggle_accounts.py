import requests
import json

tokens = {
    "hannhu40022": "KGAT_e12378203b320b6808bb0a4f272aae4e",
    "mnu303": "KGAT_9d55f06fadc38903cb36f8842430875c"
}

for user, tok in tokens.items():
    headers = {"Authorization": "Bearer " + tok}
    r = requests.get(f"https://www.kaggle.com/api/v1/kernels/list?user={user}", headers=headers)
    if r.status_code == 200:
        items = r.json()
        print(f"=== {user} kernels ({len(items)}) ===")
        for it in items:
            print(f"  - ref: {it.get('ref')}, title: {it.get('title')}")
    else:
        print(f"=== {user} kernels error: {r.status_code} {r.text[:100]}")
