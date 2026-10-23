import os
import json
import urllib.request
import urllib.error

TOKEN = "KGAT_e12378203b320b6808bb0a4f272aae4e"
USERNAME = "hannhu40022"
KERNEL_SLUG = "dataset-bo4-7class-unified-ppe-master-benchmark"

def check_status():
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName={USERNAME}&kernelSlug={KERNEL_SLUG}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        return {"error": str(e)}

def download_output():
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName={USERNAME}&kernelSlug={KERNEL_SLUG}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    st = check_status()
    print("STATUS:", st)
    if st.get("status") == "complete":
        out = download_output()
        print("FILES:", [f.get("fileName") for f in out.get("files", [])])
        log = out.get("log", "")
        if log:
            print("LOG TAIL:\n", log[-1500:])
        out_dir = "Output/dataset_bo4_benchmark"
        os.makedirs(out_dir, exist_ok=True)
        for f in out.get("files", []):
            fname = f.get("fileName")
            url = f.get("url")
            if fname and url:
                print(f"Downloading {fname}...")
                target = os.path.join(out_dir, fname)
                os.makedirs(os.path.dirname(target) if os.path.dirname(target) else ".", exist_ok=True)
                urllib.request.urlretrieve(url, target)
        print("Download complete.")
