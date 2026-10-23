import requests
import os

tokens = {
    "hannhu4002": "",
    "mnu303": "KGAT_9d55f06fadc38903cb36f8842430875c",
    "hannhu40022": "KGAT_e12378203b320b6808bb0a4f272aae4e"
}
with open(os.path.expanduser("~/.kaggle/access_token")) as f:
    tokens["hannhu4002"] = f.read().strip()

fleet = [
    ("hannhu4002", "dataset-bo1-5class-gdut-hwd-benchmark"),
    ("mnu303", "dataset-bo3-5class-iitu-drone-benchmark"),
    ("hannhu40022", "dataset-bo4-7class-unified-ppe-master")
]

for user, slug in fleet:
    headers = {"Authorization": "Bearer " + tokens[user]}
    r = requests.get(f"https://www.kaggle.com/api/v1/kernels/output?userName={user}&kernelSlug={slug}", headers=headers)
    d = r.json()
    log = d.get("logNullable") or ""
    files = d.get("files") or []
    print(f"@{user}/{slug}: log len={len(log)}, files={len(files)}")
    if files:
        for f in files:
            print(f"   file: {f.get('fileName')}")
