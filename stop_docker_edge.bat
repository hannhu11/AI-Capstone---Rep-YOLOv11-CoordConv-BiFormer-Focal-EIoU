@echo off
setlocal enabledelayedexpansion
title CCTV Safety Edge Simulation (Docker) - Stopping

echo ==============================================================================
echo [DOCKER EDGE LAUNCHER] Stopping CCTV Edge Simulation Container...
echo ==============================================================================

:: Ensure Docker CLI paths are in PATH
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin" (
    set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"
)
if exist "C:\Program Files\Docker\Docker\resources\bin" (
    set "PATH=C:\Program Files\Docker\Docker\resources\bin;%PATH%"
)

docker compose -f "%~dp0docker-compose.edge-simulation.yml" stop
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Docker compose stop had an issue, attempting direct container stop...
    docker stop cctv_safety_edge_simulation_node >nul 2>&1
)

echo.
echo ==============================================================================
echo [SUCCESS] Container stopped successfully. Resources (RAM/GPU) freed!
echo Container remains listed in Docker Desktop for instant restart.
echo To restart, run: start_docker_edge.bat
echo ==============================================================================
pause
