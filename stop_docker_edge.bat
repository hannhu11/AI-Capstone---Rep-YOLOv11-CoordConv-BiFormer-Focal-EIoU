@echo off
title CCTV Safety Edge Simulation (Docker) - Stopping
echo ==============================================================================
echo [DOCKER EDGE LAUNCHER] Stopping CCTV Edge Simulation Container...
echo ==============================================================================
docker compose -f docker-compose.edge-simulation.yml stop
echo.
echo ==============================================================================
echo STATUS: Container stopped successfully. Resources (RAM/GPU) freed!
echo Container remains listed in Docker Desktop Containers tab for instant restart.
echo ==============================================================================
pause
