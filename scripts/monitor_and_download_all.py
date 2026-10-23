"""
monitor_and_download_all.py
===========================
Continuously monitors all 3 running Kaggle kernels, prints live status,
detects errors, and automatically downloads artifacts upon completion.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.kaggle_multitoken_dispatcher import get_kernel_status, get_kernel_output

FLEET = [
    {
        "account": "hannhu4002",
        "slug": "clean-chv-6-class-official-benchmark",
        "outdir": Path("Output/kaggle_results/hannhu4002_clean_chv6")
    },
    {
        "account": "mnu303",
        "slug": "true-modular-ablation-suite-chv-v2",
        "outdir": Path("Output/kaggle_results/mnu303_modular_ablation")
    },
    {
        "account": "hannhu40022",
        "slug": "conflict-free-7class-chv-gdut-benchmark",
        "outdir": Path("Output/kaggle_results/hannhu40022_conflict_free_7class")
    }
]

def main():
    print("=" * 80)
    print("🛰️ MULTI-ACCOUNT FLEET MONITORING LOOP INITIATED")
    print("=" * 80)

    completed = {}
    for k in FLEET:
        key = f"{k['account']}/{k['slug']}"
        if k['slug'] == 'clean-chv-6-class-official-benchmark' and (k['outdir'] / 'CLEAN_CHV_6CLASS_BENCHMARK_REPORT.md').exists():
            completed[key] = True
            print(f"  [ALREADY DOWNLOADED] @{key}")
        elif k['slug'] == 'conflict-free-7class-chv-gdut-benchmark' and (k['outdir'] / 'CONFLICT_FREE_7CLASS_BENCHMARK_REPORT.md').exists():
            completed[key] = True
            print(f"  [ALREADY DOWNLOADED] @{key}")
        elif k['slug'] == 'true-modular-ablation-suite-chv-v2' and (k['outdir'] / 'TASK2_TRUE_MODULAR_ABLATION_REPORT.md').exists():
            completed[key] = True
            print(f"  [ALREADY DOWNLOADED] @{key}")
        else:
            completed[key] = False

    start_time = time.time()

    while not all(completed.values()):
        elapsed = (time.time() - start_time) / 60
        print(f"\n--- Checkpoint at T+{elapsed:.1f} minutes ---", flush=True)

        for item in FLEET:
            key = f"{item['account']}/{item['slug']}"
            if completed[key]:
                continue

            status_info = get_kernel_status(item["account"], item["slug"])
            status = status_info.get("status", "unknown")
            fail_msg = status_info.get("failureMessage", "")

            if status == "complete":
                print(f"  [COMPLETE] @{key} finished successfully! Downloading artifacts...", flush=True)
                out_data = get_kernel_output(item["account"], item["slug"], download_dir=str(item["outdir"]))
                if out_data:
                    import json
                    log_entries = json.loads(out_data.get('log', '[]'))
                    stdout_text = ''.join([entry.get('data', '') for entry in log_entries if entry.get('stream_name') == 'stdout'])
                    with open(item["outdir"] / 'full_stdout.txt', 'w', encoding='utf-8') as f:
                        f.write(stdout_text)
                    print(f"  [SAVED] Saved full stdout for @{key}", flush=True)
                completed[key] = True
            elif status == "error":
                print(f"  [ERROR] @{key} failed with message: {fail_msg}")
                # Print output log for diagnostics
                out_info = get_kernel_output(item["account"], item["slug"])
                print(f"  Log preview: {str(out_info)[:500]}")
                sys.exit(1)
            elif status == "running":
                print(f"  [RUNNING] @{key} is running on GPU...")
            elif status == "queued":
                print(f"  [QUEUED] @{key} is queued in Kaggle cluster...")
            else:
                print(f"  [STATUS: {status}] @{key}")

        if not all(completed.values()):
            time.sleep(35)

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL 3 KAGGLE KERNELS HAVE COMPLETED AND ARTIFACTS ARE DOWNLOADED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
