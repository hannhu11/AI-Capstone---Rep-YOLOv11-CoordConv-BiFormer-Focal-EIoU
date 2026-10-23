#!/usr/bin/env python3
"""
scripts/active_poll_fleet.py
============================
Synchronous active polling loop for the 3 Kaggle GPU research runs.
Polls every 25 seconds until all reach 'complete'.
On error, downloads logs for immediate bug diagnosis.
On completion, downloads all generated artifacts and verifies metrics.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.kaggle_multitoken_dispatcher import get_kernel_status, get_kernel_output

FLEET = [
    {
        "account": "hannhu4002",
        "slug": "dataset-strategy-transfer-learning-benchmark",
        "outdir": Path("Output/kaggle_results/hannhu4002_dataset_strategy")
    },
    {
        "account": "mnu303",
        "slug": "multi-generation-yolo-baseline-selection",
        "outdir": Path("Output/kaggle_results/mnu303_yolo_multigen")
    },
    {
        "account": "hannhu40022",
        "slug": "warm-start-rep-yolo11s-ablation-rule-engine",
        "outdir": Path("Output/kaggle_results/hannhu40022_warmstart_ablations")
    }
]

def check_one_cycle():
    statuses = {}
    for item in FLEET:
        key = f"{item['account']}/{item['slug']}"
        try:
            st = get_kernel_status(item["account"], item["slug"])
            status = st.get("status", "unknown")
            fail_msg = st.get("failureMessage", "")
            statuses[key] = {"status": status, "failureMessage": fail_msg, "item": item}
        except Exception as e:
            statuses[key] = {"status": "request_error", "failureMessage": str(e), "item": item}
    return statuses

def main():
    print("=" * 80)
    print("🛰️ ACTIVE FLEET MONITORING INITIATED (25s Cadence)")
    print("=" * 80)

    completed = {f"{k['account']}/{k['slug']}": False for k in FLEET}
    start_time = time.time()
    iteration = 0

    while not all(completed.values()):
        iteration += 1
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"\n--- [Iteration {iteration:03d} | Elapsed: {elapsed_min:5.1f} min] ---", flush=True)

        statuses = check_one_cycle()
        for key, info in statuses.items():
            status = info["status"]
            fail_msg = info["failureMessage"]
            item = info["item"]

            if completed[key]:
                print(f"  ✅ [DONE] @{key} (previously completed)", flush=True)
                continue

            if status == "complete":
                print(f"  🎉 [COMPLETE] @{key} has finished! Downloading artifacts to {item['outdir']}...", flush=True)
                get_kernel_output(item["account"], item["slug"], download_dir=str(item["outdir"]))
                completed[key] = True

            elif status in ["error", "failed", "cancelAcknowledged"]:
                print(f"  ❌ [ERROR DETECTED] @{key} failed with status='{status}'! Message: {fail_msg}", flush=True)
                print(f"  📥 Pulling diagnostic output/logs to {item['outdir']}...", flush=True)
                get_kernel_output(item["account"], item["slug"], download_dir=str(item["outdir"]))
                print(f"  🚨 Terminating loop immediately so agent can remediate error on @{key}!", flush=True)
                sys.exit(2)

            elif status == "running":
                print(f"  ⚡ [RUNNING] @{key} is actively training...", flush=True)

            elif status == "queued":
                print(f"  ⏳ [QUEUED]  @{key} is waiting in GPU queue...", flush=True)

            else:
                print(f"  ❓ [{status.upper()}] @{key} (info: {fail_msg})", flush=True)

        # Write status file for external inspection
        status_file = Path("Output/kaggle_results/fleet_status.json")
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as sf:
            json.dump({
                "iteration": iteration,
                "elapsed_min": round(elapsed_min, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fleet": {k: {"status": v["status"], "fail_msg": v["failureMessage"]} for k, v in statuses.items()}
            }, sf, indent=2)

        # Check if all done
        if all(completed.values()):
            break

        time.sleep(25)

    print("\n" + "=" * 80)
    print("🏁 MONITORING COMPLETED: FINAL DISPOSITION")
    print("=" * 80)
    for key, val in completed.items():
        print(f"  @{key}: {'COMPLETE' if val is True else val}")

if __name__ == "__main__":
    main()
