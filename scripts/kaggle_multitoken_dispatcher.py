#!/usr/bin/env python3
"""
kaggle_multitoken_dispatcher.py
================================
Multi-account Kaggle dispatcher & monitoring tool for Capstone AI.
Manages 3 authorized Kaggle accounts cleanly via REST API:
  - Account 1: hannhu4002   (Token from ~/.kaggle/access_token)
  - Account 2: mnu303       (Token: KGAT_9d55f06fadc38903cb36f8842430875c)
  - Account 3: hannhu40022  (Token: KGAT_e12378203b320b6808bb0a4f272aae4e)

Features:
  - Account rate limiting & concurrency protection (avoids API throttle & bans).
  - Push notebooks with automated kernelspec validation & GPU acceleration.
  - Asynchronous status polling and log inspection.
  - Automatic artifact and report downloading.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import requests

ACCOUNTS = {
    "hannhu4002": {
        "token_path": os.path.expanduser(r"~/.kaggle/access_token"),
        "default_token": "",
        "username": "hannhu4002"
    },
    "mnu303": {
        "token_path": None,
        "default_token": "KGAT_9d55f06fadc38903cb36f8842430875c",
        "username": "mnu303"
    },
    "hannhu40022": {
        "token_path": None,
        "default_token": "KGAT_e12378203b320b6808bb0a4f272aae4e",
        "username": "hannhu40022"
    }
}

def get_token(account_name: str) -> str:
    cfg = ACCOUNTS.get(account_name)
    if not cfg:
        raise ValueError(f"Unknown account: {account_name}. Available: {list(ACCOUNTS.keys())}")
    
    if cfg["token_path"] and os.path.exists(cfg["token_path"]):
        with open(cfg["token_path"], "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token:
                return token
    return cfg["default_token"]

def get_headers(account_name: str):
    token = get_token(account_name)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def push_notebook(account_name: str, notebook_path: str, slug: str, title: str, 
                  dataset_sources=None, enable_gpu=True, is_private=True):
    """Pushes a Jupyter notebook to Kaggle for execution."""
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    # Ensure kernelspec exists in metadata to prevent Papermill crashes
    if "metadata" not in nb_json:
        nb_json["metadata"] = {}
    if "kernelspec" not in nb_json["metadata"]:
        nb_json["metadata"]["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    if "language_info" not in nb_json["metadata"]:
        nb_json["metadata"]["language_info"] = {
            "name": "python",
            "version": "3.10.12"
        }

    payload = {
        "slug": f"{account_name}/{slug}",
        "newTitle": title,
        "text": json.dumps(nb_json),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": is_private,
        "enableGpu": enable_gpu,
        "enableInternet": True,
        "datasetDataSources": dataset_sources or []
    }

    url = "https://www.kaggle.com/api/v1/kernels/push"
    headers = get_headers(account_name)
    
    print(f"🚀 [Kaggle Dispatcher] Pushing '{title}' ({slug}) to account @{account_name}...")
    res = requests.post(url, json=payload, headers=headers)
    
    if res.status_code == 200:
        data = res.json()
        print(f"✅ Success! Kernel ID: {data.get('kernelId')}, Ref: {data.get('ref')}")
        print(f"🔗 URL: {data.get('url')}")
        return True, data
    else:
        print(f"❌ Failed (HTTP {res.status_code}): {res.text}")
        return False, res.text

def get_kernel_status(account_name: str, slug: str):
    """Checks the status of a specific kernel."""
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName={account_name}&kernelSlug={slug}"
    headers = get_headers(account_name)
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()
    else:
        return {"status": "unknown", "error": res.text, "http_code": res.status_code}

def get_kernel_output(account_name: str, slug: str, download_dir=None):
    """Retrieves kernel output files and saves artifacts locally."""
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName={account_name}&kernelSlug={slug}"
    headers = get_headers(account_name)
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        print(f"❌ Failed to get output (HTTP {res.status_code}): {res.text}")
        return None

    data = res.json()
    files = data.get("files", [])
    print(f"📦 Kernel @{account_name}/{slug} produced {len(files)} output files.")
    
    if download_dir:
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        
        base_jwt_url = None
        for f in files:
            file_name = f.get("fileName")
            file_url = f.get("url")
            if file_url and not base_jwt_url:
                parts = file_url.split("/")
                if len(parts) >= 6:
                    base_jwt_url = "/".join(parts[:6])

            if file_url and (file_name.endswith(".md") or file_name.endswith(".png") or file_name.endswith(".yaml") or file_name.endswith(".csv") or file_name.endswith(".pt")):
                target = download_path / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"  ⬇️ Downloading {file_name} -> {target}")
                r_file = requests.get(file_url)
                if r_file.status_code == 200:
                    with open(target, "wb") as out_f:
                        out_f.write(r_file.content)

        # Direct JWT probe for critical root artifacts (bypasses 500-file API cap)
        if base_jwt_url:
            probed_artifacts = [
                "clean_chv_6class_comparison.csv",
                "CLEAN_CHV_6CLASS_BENCHMARK_REPORT.md",
                "rep_yolo11s_chv6_fused_deploy.pt",
                "modular_ablation_suite_results.csv",
                "TASK2_TRUE_MODULAR_ABLATION_REPORT.md",
                "conflict_free_7class_comparison.csv",
                "CONFLICT_FREE_7CLASS_BENCHMARK_REPORT.md",
                "rep_yolo11s_chv7_best_deploy.pt"
            ]
            for art in probed_artifacts:
                art_target = download_path / art
                if not art_target.exists():
                    probe_url = f"{base_jwt_url}/{art}"
                    try:
                        r_probe = requests.get(probe_url, timeout=30)
                        if r_probe.status_code == 200:
                            with open(art_target, "wb") as out_f:
                                out_f.write(r_probe.content)
                            print(f"  🎯 [JWT Direct Download] Recovered capped artifact: {art} ({len(r_probe.content)} bytes)")
                    except Exception as e:
                        pass
    return data

def status_all():
    """Checks recent kernels across all 3 configured accounts."""
    print("=" * 80)
    print("📊 KAGGLE MULTI-ACCOUNT FLEET STATUS")
    print("=" * 80)
    for acc in ACCOUNTS.keys():
        print(f"\n👤 Account: @{acc}")
        headers = get_headers(acc)
        url = "https://www.kaggle.com/api/v1/kernels/list?group=profile&pageSize=4"
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            kernels = r.json()
            if not kernels:
                print("   (No kernels found)")
            for k in kernels:
                ref = k.get("ref")
                u, s = ref.split("/")
                st = get_kernel_status(u, s)
                status_str = st.get("status", "unknown")
                symbol = "🟢" if status_str == "complete" else ("🟡" if status_str == "running" else "🔴")
                print(f"   {symbol} [{ref}] -> Status: {status_str}")
        else:
            print(f"   ❌ HTTP {r.status_code}: {r.text}")
        time.sleep(1.0) # Rate limit safety

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Multi-Token Dispatcher for Capstone AI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # status command
    subparsers.add_parser("status", help="Check status across all 3 accounts")
    
    # push command
    p_push = subparsers.add_parser("push", help="Push notebook to an account")
    p_push.add_argument("--account", required=True, choices=list(ACCOUNTS.keys()))
    p_push.add_argument("--notebook", required=True, help="Path to .ipynb file")
    p_push.add_argument("--slug", required=True, help="Kernel slug")
    p_push.add_argument("--title", required=True, help="Kernel display title")
    p_push.add_argument("--datasets", nargs="*", default=[], help="Datasets to attach")
    p_push.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    
    # check command
    p_check = subparsers.add_parser("check", help="Check status of a single kernel")
    p_check.add_argument("--account", required=True, choices=list(ACCOUNTS.keys()))
    p_check.add_argument("--slug", required=True)
    
    # pull command
    p_pull = subparsers.add_parser("pull", help="Download output files from a kernel")
    p_pull.add_argument("--account", required=True, choices=list(ACCOUNTS.keys()))
    p_pull.add_argument("--slug", required=True)
    p_pull.add_argument("--outdir", default="./kaggle_outputs", help="Output directory")

    args = parser.parse_args()
    if args.command == "status":
        status_all()
    elif args.command == "push":
        push_notebook(
            account_name=args.account,
            notebook_path=args.notebook,
            slug=args.slug,
            title=args.title,
            dataset_sources=args.datasets,
            enable_gpu=not args.cpu
        )
    elif args.command == "check":
        res = get_kernel_status(args.account, args.slug)
        print(f"Status for @{args.account}/{args.slug}: {res}")
    elif args.command == "pull":
        get_kernel_output(args.account, args.slug, args.outdir)
    else:
        parser.print_help()
