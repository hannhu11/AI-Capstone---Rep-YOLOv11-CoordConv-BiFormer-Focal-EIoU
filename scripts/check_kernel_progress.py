import requests
import os

tokens = {
    "hannhu4002": "",
    "mnu303": "KGAT_9d55f06fadc38903cb36f8842430875c",
    "hannhu40022": "KGAT_e12378203b320b6808bb0a4f272aae4e"
}
with open(os.path.expanduser("~/.kaggle/access_token")) as f:
    tokens["hannhu4002"] = f.read().strip()

import json
import requests
import os

tokens = {
    "hannhu4002": "",
    "mnu303": "KGAT_9d55f06fadc38903cb36f8842430875c",
    "hannhu40022": "KGAT_e12378203b320b6808bb0a4f272aae4e"
}
token_path = os.path.expanduser("~/.kaggle/access_token")
if os.path.exists(token_path):
    with open(token_path) as f:
        tokens["hannhu4002"] = f.read().strip()

fleet = [
    ("hannhu4002", "clean-chv-6-class-official-benchmark"),
    ("mnu303", "true-modular-ablation-suite-chv-v2"),
    ("hannhu40022", "conflict-free-7class-chv-gdut-benchmark")
]

for user, slug in fleet:
    headers = {"Authorization": "Bearer " + tokens[user]}
    r = requests.get(f"https://www.kaggle.com/api/v1/kernels/output?userName={user}&kernelSlug={slug}", headers=headers)
    if r.status_code != 200:
        print(f"@{user}/{slug}: HTTP {r.status_code}")
        continue
    d = r.json()
    raw_log = d.get("log") or "[]"
    try:
        entries = json.loads(raw_log)
        stdout_lines = [e.get("data", "") for e in entries if e.get("stream_name") == "stdout"]
        full_text = "".join(stdout_lines)
        last_lines = [l for l in full_text.splitlines() if l.strip()][-8:]
        print(f"\n=== @{user}/{slug} ===")
        for l in last_lines:
            print("  " + l)
    except Exception as ex:
        print(f"@{user}/{slug}: Error parsing log: {ex}")

