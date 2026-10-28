@echo off
title CCTV Safety Edge Simulation (Docker) - Starting
echo ==============================================================================
echo [DOCKER EDGE LAUNCHER] Starting CCTV Edge Simulation Container...
echo ==============================================================================
docker compose -f docker-compose.edge-simulation.yml up -d
echo.
echo ==============================================================================
echo STATUS: Container is running!
echo Access Dashboard at: http://localhost:8080
echo ==============================================================================
echo.
echo To view realtime logs, run: docker logs -f cctv_safety_edge_simulation_node
echo To stop container, run stop_docker_edge.bat or docker compose -f docker-compose.edge-simulation.yml stop
pause
