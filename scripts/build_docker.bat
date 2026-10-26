@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo [DOCKER BUILD & DEPLOY] INDUSTRIAL CCTV SAFETY SURVEILLANCE NODE
echo ==============================================================================

:: Find docker executable
set DOCKER_EXE=docker
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" (
        set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
        set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%PATH%"
    ) else if exist "C:\Program Files\Docker\Docker\resources\bin\docker.exe" (
        set "DOCKER_EXE=C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        set "PATH=C:\Program Files\Docker\Docker\resources\bin;%PATH%"
    ) else (
        echo [ERROR] Docker executable could not be found! Please make sure Docker Desktop is running.
        pause
        exit /b 1
    )
)

echo [*] Using Docker CLI: !DOCKER_EXE!
echo [*] Building Docker Image: rep-yolo11s-cctv:latest ...
"!DOCKER_EXE!" build -t rep-yolo11s-cctv:latest -f Dockerfile .
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo [*] Image built successfully!
echo [*] Starting Container with GPU reservation on Port 8000...
"!DOCKER_EXE!" run --rm -d ^
    --name cctv_safety_surveillance_node ^
    --gpus all ^
    -p 8000:8000 ^
    -v "%cd%\video_test:/app/video_test:ro" ^
    -v "%cd%\Output:/app/Output" ^
    -v "%cd%\snapshots:/app/snapshots" ^
    -v "%cd%\.env:/app/.env:ro" ^
    rep-yolo11s-cctv:latest

echo ==============================================================================
echo [*] Surveillance Node is running at: http://localhost:8000
echo [*] To view live logs: "!DOCKER_EXE!" logs -f cctv_safety_surveillance_node
echo [*] To stop container: "!DOCKER_EXE!" stop cctv_safety_surveillance_node
echo ==============================================================================
pause
