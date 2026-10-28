@echo off
setlocal enabledelayedexpansion
title CCTV Safety Edge Simulation (Docker) - Starting

echo ==============================================================================
echo [DOCKER EDGE LAUNCHER] Starting CCTV Edge Simulation Container...
echo ==============================================================================

:: Ensure Docker CLI paths are in PATH
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin" (
    set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"
)
if exist "C:\Program Files\Docker\Docker\resources\bin" (
    set "PATH=C:\Program Files\Docker\Docker\resources\bin;%PATH%"
)

:: Verify Docker executable exists
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker CLI not found! Please ensure Docker Desktop is installed.
    echo If Docker Desktop is installed, please add its bin folder to system PATH.
    pause
    exit /b 1
)

:: Verify Docker Daemon is running (auto-launch if inactive)
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Docker daemon is not active. Attempting to launch Docker Desktop...
    if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" (
        start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
    ) else if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    )
    echo [*] Waiting for Docker daemon to initialize...
    set /a d_attempts=0
    :DOCKER_WAIT
    set /a d_attempts+=1
    timeout /t 2 /nobreak >nul
    docker info >nul 2>&1
    if !ERRORLEVEL! equ 0 goto DOCKER_READY
    if !d_attempts! geq 30 (
        echo.
        echo [ERROR] Docker daemon did not initialize in time.
        echo Please ensure Docker Desktop is open and running, then re-run this script.
        pause
        exit /b 1
    )
    <nul set /p=.
    goto DOCKER_WAIT
)
:DOCKER_READY

:: Ensure .env exists
if not exist "%~dp0.env" (
    if exist "%~dp0.env.example" (
        echo [*] Auto-generating .env from .env.example ...
        copy "%~dp0.env.example" "%~dp0.env" >nul
    )
)

:: Start container with Docker Compose
echo [*] Launching container with Docker Compose...
docker compose -f "%~dp0docker-compose.edge-simulation.yml" up -d
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Compose failed to start container. Check output above.
    pause
    exit /b %ERRORLEVEL%
)

:: Automated Healthcheck Wait Loop
echo.
<nul set /p=[*] Waiting for CCTV Safety Engine and PyTorch CUDA warm-up
set /a attempts=0

:WAIT_LOOP
set /a attempts+=1
curl.exe -s -o NUL -w "%%{http_code}" http://localhost:8080/api/metrics > "%TEMP%\edge_hc.txt" 2>NUL
set /p HC_STATUS=<"%TEMP%\edge_hc.txt"
if exist "%TEMP%\edge_hc.txt" del "%TEMP%\edge_hc.txt" >nul 2>&1

if "!HC_STATUS!"=="200" goto READY

if !attempts! geq 60 goto TIMEOUT
timeout /t 1 /nobreak >nul
<nul set /p=.
goto WAIT_LOOP

:TIMEOUT
echo.
echo [WARNING] Healthcheck timed out after 60 seconds.
echo Container might still be warming up or encountered an error.
echo Check container logs with: docker logs -f cctv_safety_edge_simulation_node
echo You can still try opening http://localhost:8080 manually.
goto FINISH

:READY
echo.
echo ==============================================================================
echo [SUCCESS] CCTV Surveillance Engine is FULLY READY on CUDA GPU!
echo [URL] Dashboard: http://localhost:8080
echo.
echo [HUONG DAN CHUYEN DOI MO HINH - HOT-SWAP]:
echo 1. Ngay tren Web UI: An Control Panel -^> Chon Model -^> An 'Swap Weights'.
echo 2. Bang 1-click batch: Chay 'switch_to_gdut.bat' hoac 'switch_to_champion.bat'.
echo 3. Chinh MODEL_PATH trong file .env neu muon doi model mac dinh luc khoi dong container.
echo ==============================================================================
echo [*] Opening Dashboard in default web browser...
start http://localhost:8080

:FINISH
echo.
echo [COMMANDS]:
echo - Live logs: docker logs -f cctv_safety_edge_simulation_node
echo - Restart container: restart_docker_edge.bat
echo - Stop container: stop_docker_edge.bat
pause
