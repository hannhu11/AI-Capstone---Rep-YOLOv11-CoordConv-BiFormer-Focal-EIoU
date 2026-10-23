import requests

tokens = {
    "hannhu40022": "KGAT_e12378203b320b6808bb0a4f272aae4e",
    "mnu303": "KGAT_9d55f06fadc38903cb36f8842430875c"
}

# Check mnu303 kernels
headers_mnu = {"Authorization": "Bearer " + tokens["mnu303"]}
r = requests.get("https://www.kaggle.com/api/v1/kernels/list?user=mnu303", headers=headers_mnu)
print("mnu303 kernels:", [(k['ref'], k['title']) for k in r.json()[:5]])

# Check if any bo1 or bo3 exists
for u, tok in tokens.items():
    headers = {"Authorization": "Bearer " + tok}
    for slug in ["dataset-bo1-5class-gdut-hwd-benchmark", "dataset-bo3-5class-iitu-drone-benchmark", "color-helmet-5class-benchmark"]:
        r = requests.get(f"https://www.kaggle.com/api/v1/kernels/status?userName={u}&kernelSlug={slug}", headers=headers)
        if r.status_code == 200:
            print(f"Found {u}/{slug}: {r.json()}")
