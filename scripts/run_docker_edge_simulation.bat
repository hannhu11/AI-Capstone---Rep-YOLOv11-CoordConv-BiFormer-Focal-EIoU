@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo [DOCKER EDGE SIMULATION] 4-CORE 4GB JETSON/EDGE HARDWARE CONSTRAINT LAUNCHER
echo ==============================================================================

:: Ensure Docker CLI and credential helpers are in PATH unconditionally
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin" (
    set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"
)
if exist "C:\Program Files\Docker\Docker\resources\bin" (
    set "PATH=C:\Program Files\Docker\Docker\resources\bin;%PATH%"
)

:: Find docker executable
set DOCKER_EXE=docker
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker executable could not be found! Please make sure Docker Desktop is running.
    pause
    exit /b 1
)

:: Ensure .env file exists
if not exist "%cd%\.env" (
    if exist "%cd%\.env.example" (
        echo [*] Auto-generating .env from .env.example ...
        copy "%cd%\.env.example" "%cd%\.env" >nul
    )
)

echo [*] Using Docker CLI: !DOCKER_EXE!
echo [*] Launching Edge Simulation Compose (4 vCPUs, 4GB RAM limit) ...
"!DOCKER_EXE!" compose -f docker-compose.edge-simulation.yml up -d --build

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker edge simulation launch failed!
    pause
    exit /b %ERRORLEVEL%
)

if not defined EDGE_SIM_PORT set EDGE_SIM_PORT=8080

echo ==============================================================================
echo [*] Edge Simulation Node is running at: http://localhost:!EDGE_SIM_PORT!
echo [*] Resource Constraints: 4.0 vCPUs, 4096MB Memory limit (Jetson Profile)
echo [*] To view live logs: "!DOCKER_EXE!" logs -f cctv_safety_edge_simulation_node
echo [*] To stop: "!DOCKER_EXE!" compose -f docker-compose.edge-simulation.yml down
echo ==============================================================================
pause
