#!/usr/bin/env python3
"""
wait_and_monitor.py
===================
Monitors the 3 dispatched Kaggle benchmark runs until all reach a terminal state
('complete' or 'error'). Automatically downloads the generated reports and metrics.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, 'scripts')
import kaggle_multitoken_dispatcher as d

RUNS = [
    {"account": "hannhu4002", "slug": "dataset-bo1-5class-gdut-hwd-benchmark", "outdir": "./results_bo1"},
    {"account": "mnu303", "slug": "dataset-bo3-5class-iitu-drone-benchmark", "outdir": "./results_bo3"},
    {"account": "hannhu40022", "slug": "dataset-bo4-7class-unified-ppe-master", "outdir": "./results_bo4"}
]

max_wait_seconds = 1800  # 30 minutes maximum
check_interval = 25     # check every 25s
elapsed = 0

print("=" * 80)
print("⏳ MONITORING 3 KAGGLE BENCHMARK RUNS")
print("=" * 80)

completed = set()

while elapsed < max_wait_seconds:
    all_terminal = True
    print(f"\n[+{elapsed:03d}s] Checking status of active kernels...")
    
    for r in RUNS:
        acc = r["account"]
        slug = r["slug"]
        key = f"{acc}/{slug}"
        
        st = d.get_kernel_status(acc, slug)
        status_str = st.get("status", "unknown")
        fail_msg = st.get("failureMessage", "")
        
        symbol = "🟢" if status_str == "complete" else ("🟡" if status_str == "running" else "🔴")
        print(f"  {symbol} {key:<45} -> Status: {status_str} {('(' + fail_msg + ')') if fail_msg else ''}")
        
        if status_str in ["complete", "error"]:
            if key not in completed:
                completed.add(key)
                if status_str == "complete":
                    print(f"  📥 Pulling output artifacts for {key} into {r['outdir']}...")
                    d.get_kernel_output(acc, slug, r["outdir"])
                else:
                    print(f"  ❌ Error encountered in {key}. Downloading error log...")
                    # download log
                    headers = d.get_headers(acc)
                    import requests
                    res = requests.get(f"https://www.kaggle.com/api/v1/kernels/output?userName={acc}&kernelSlug={slug}", headers=headers)
                    if res.status_code == 200:
                        log_data = res.json().get("log", "")
                        Path(r["outdir"]).mkdir(parents=True, exist_ok=True)
                        with open(Path(r["outdir"]) / "error_log.json", "w", encoding="utf-8") as f_log:
                            f_log.write(log_data)
        else:
            all_terminal = False
            
    if all_terminal:
        print("\n🎉 ALL 3 KAGGLE RUNS HAVE REACHED TERMINAL STATE!")
        break
        
    time.sleep(check_interval)
    elapsed += check_interval

print("=" * 80)
print(f"Monitoring cycle finished after {elapsed}s. Completed runs: {len(completed)}/3")
print("=" * 80)
