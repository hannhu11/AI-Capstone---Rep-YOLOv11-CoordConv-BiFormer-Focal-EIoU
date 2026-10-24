#!/usr/bin/env python3
"""
scripts/dispatch_and_active_monitor.py
======================================
Dispatches and actively monitors the 3 Kaggle research experiments:
1. Account 1: @hannhu4002   -> dataset-strategy-transfer-learning-benchmark
2. Account 2: @mnu303       -> multi-generation-yolo-baseline-selection
3. Account 3: @hannhu40022  -> warmstart-rep-yolo11s-ablations-and-rule-engine

Monitors every 25 seconds right from the first minute.
Detects early failures immediately for swift remediation.
Downloads all outputs upon completion.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.kaggle_multitoken_dispatcher import push_notebook, get_kernel_status, get_kernel_output

FLEET = [
    {
        "account": "hannhu4002",
        "slug": "dataset-strategy-transfer-learning-benchmark",
        "title": "Dataset Strategy & Transfer Learning Benchmark",
        "notebook": "notebooks/Dataset_Strategy_Transfer_Learning_Benchmark.ipynb",
        "datasets": ["hannhu4002/unified-iitu-chv-ppe-clean-7class"],
        "outdir": Path("Output/kaggle_results/hannhu4002_dataset_strategy")
    },
    {
        "account": "mnu303",
        "slug": "multi-generation-yolo-baseline-selection",
        "title": "Multi Generation YOLO Baseline Selection",
        "notebook": "notebooks/Multi_Generation_YOLO_Baseline_Selection.ipynb",
        "datasets": ["hannhu4002/unified-iitu-chv-ppe-clean-7class"],
        "outdir": Path("Output/kaggle_results/mnu303_yolo_multigen")
    },
    {
        "account": "hannhu40022",
        "slug": "warm-start-rep-yolo11s-ablation-rule-engine",
        "title": "Warm Start Rep YOLO11s Ablation & Rule Engine",
        "notebook": "notebooks/Rep_YOLO11s_WarmStart_Ablations_And_RuleEngine.ipynb",
        "datasets": ["hannhu4002/unified-iitu-chv-ppe-clean-7class"],
        "outdir": Path("Output/kaggle_results/hannhu40022_warmstart_ablations")
    }
]

def dispatch_all():
    print("=" * 80)
    print("🚀 DISPATCHING 3 RESEARCH EXPERIMENTS ACROSS 3 KAGGLE ACCOUNTS")
    print("=" * 80)
    for item in FLEET:
        print(f"\nDispatching for @{item['account']}...")
        ok, res = push_notebook(
            account_name=item["account"],
            notebook_path=item["notebook"],
            slug=item["slug"],
            title=item["title"],
            dataset_sources=item["datasets"],
            enable_gpu=True,
            is_private=True
        )
        if not ok:
            print(f"❌ Initial push failed for @{item['account']}: {res}")
        time.sleep(2.0)

def monitor_loop():
    print("\n" + "=" * 80)
    print("🛰️ ACTIVE FLEET MONITORING LOOP INITIATED (25s cadence)")
    print("=" * 80)

    completed = {f"{k['account']}/{k['slug']}": False for k in FLEET}
    start_time = time.time()
    iteration = 0

    while not all(completed.values()):
        iteration += 1
        elapsed = (time.time() - start_time) / 60
        print(f"\n--- [Iteration {iteration}] Checkpoint at T+{elapsed:.1f} minutes ---", flush=True)

        for item in FLEET:
            key = f"{item['account']}/{item['slug']}"
            if completed[key]:
                continue

            status_info = get_kernel_status(item["account"], item["slug"])
            status = status_info.get("status", "unknown")
            fail_msg = status_info.get("failureMessage", "")

            if status == "complete":
                print(f"  🟢 [COMPLETE] @{key} finished successfully! Pulling artifacts...", flush=True)
                get_kernel_output(item["account"], item["slug"], download_dir=str(item["outdir"]))
                completed[key] = True

            elif status in ["error", "cancelAcknowledged", "failed"]:
                print(f"  🔴 [ERROR DETECTED] @{key} status={status}! Failure message: {fail_msg}", flush=True)
                # Pull whatever logs or outputs exist
                out = get_kernel_output(item["account"], item["slug"], download_dir=str(item["outdir"]))
                completed[key] = True # Mark handled or flag for review

            elif status in ["running", "queued"]:
                symbol = "🟡" if status == "running" else "⏳"
                print(f"  {symbol} [{status.upper()}] @{key} is actively executing...", flush=True)

            else:
                print(f"  ⚪ [STATUS] @{key}: {status}", flush=True)

        if not all(completed.values()):
            time.sleep(25)

    print("\n" + "=" * 80)
    print("🏁 ALL FLEET EXPERIMENTS COMPLETED OR HANDLED")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor-only":
        monitor_loop()
    else:
        dispatch_all()
        monitor_loop()
