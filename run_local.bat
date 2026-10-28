@echo off
title CCTV Safety Surveillance - Local GPU Mode (Port 8000)
echo ==============================================================================
echo [LOCAL GPU LAUNCHER] Starting CCTV Dashboard directly with host CUDA GPU...
echo Dashboard will be available at: http://localhost:8000
echo ==============================================================================
python scripts/run_local_dashboard.py
pause
