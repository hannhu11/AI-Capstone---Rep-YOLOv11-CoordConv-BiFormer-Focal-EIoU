#!/usr/bin/env bash
set -e

echo "=============================================================================="
echo "[DOCKER BUILD & DEPLOY] INDUSTRIAL CCTV SAFETY SURVEILLANCE NODE (LINUX/WSL)"
echo "=============================================================================="

# Build image
echo "[*] Building Docker Image: rep-yolo11s-cctv:latest ..."
docker build -t rep-yolo11s-cctv:latest -f Dockerfile .

echo "[*] Starting Container with NVIDIA GPU access on Port 8000..."
docker run --rm -d \
    --name cctv_safety_surveillance_node \
    --gpus all \
    -p 8000:8000 \
    -v "$(pwd)/video_test:/app/video_test:ro" \
    -v "$(pwd)/Output:/app/Output" \
    -v "$(pwd)/snapshots:/app/snapshots" \
    -v "$(pwd)/.env:/app/.env:ro" \
    rep-yolo11s-cctv:latest

echo "=============================================================================="
echo "[*] Surveillance Node is live at: http://localhost:8000"
echo "[*] To view logs: docker logs -f cctv_safety_surveillance_node"
echo "[*] To stop:      docker stop cctv_safety_surveillance_node"
echo "=============================================================================="
