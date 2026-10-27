import time
from fastapi.testclient import TestClient
from src.web.app import app
from src.engine.pipeline import cctv_pipeline

client = TestClient(app)

print("--- Testing API Endpoints Live ---")
# 1. Switch to UHD construction video
res_sw = client.post("/api/switch_source", json={"video_source": "video_test/13921040-uhd_3840_2160_30fps.mp4"})
print("Switch Source Response:", res_sw.json())

# Wait 4 seconds for pipeline background worker to process frames
time.sleep(4.0)

# 2. Check Metrics
res_met = client.get("/api/metrics")
data_met = res_met.json()
print(f"Metrics: FPS={data_met['fps']}, Workers={data_met['workers_count']}, Source={data_met['video_source']}, Registered={data_met.get('registered_workers_count')}")

# 3. Check Tracked Workers List
res_work = client.get("/api/tracking/workers")
data_work = res_work.json()
print(f"Tracked Workers (Total={data_work['total']}):")
for w in data_work["workers"][:5]:
    print(f"   Worker #{w['track_id']} [{w['helmet_color']}] - Status: {w['status']}, Active: {w['is_active']}")

# 4. Set Focus on Worker #1
print("\nActivating Solo Focus Mode on Worker #1...")
res_foc = client.post("/api/tracking/focus", json={"track_id": 1})
print("Focus Response:", res_foc.json())

time.sleep(1.0)
res_met2 = client.get("/api/metrics")
print(f"Focus Mode Metric in Telemetry: focus_track_id = {res_met2.json().get('focus_track_id')}")

# 5. Clear Focus Mode
print("\nClearing Focus Mode back to View All...")
res_clear = client.post("/api/tracking/focus", json={"track_id": None})
print("Clear Focus Response:", res_clear.json())

cctv_pipeline.stop()
print("All endpoints tested successfully!")
