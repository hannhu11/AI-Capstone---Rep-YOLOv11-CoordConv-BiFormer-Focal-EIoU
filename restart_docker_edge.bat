@echo off
setlocal enabledelayedexpansion
title CCTV Safety Edge Simulation (Docker) - Restarting

echo ==============================================================================
echo [DOCKER EDGE LAUNCHER] Restarting CCTV Edge Simulation Container...
echo ==============================================================================

:: Ensure Docker CLI paths are in PATH
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin" (
    set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"
)
if exist "C:\Program Files\Docker\Docker\resources\bin" (
    set "PATH=C:\Program Files\Docker\Docker\resources\bin;%PATH%"
)

:: Restart container with Docker Compose
echo [*] Restarting container (re-reading .env)...
docker compose -f "%~dp0docker-compose.edge-simulation.yml" restart
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Compose restart failed.
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
echo ==============================================================================
echo [*] Opening Dashboard in default web browser...
start http://localhost:8080

:FINISH
echo.
echo [COMMANDS]:
echo - Live logs: docker logs -f cctv_safety_edge_simulation_node
echo - Stop container: stop_docker_edge.bat
pause
