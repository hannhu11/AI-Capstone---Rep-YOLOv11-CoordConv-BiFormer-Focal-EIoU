# scripts/make_batch_launchers.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

START_EDGE_BAT = """@echo off
setlocal enabledelayedexpansion
title CCTV Safety Edge Simulation (Docker) - Starting

echo ==============================================================================
echo [DOCKER EDGE LAUNCHER] Starting CCTV Edge Simulation Container...
echo ==============================================================================

:: Ensure Docker CLI paths are in PATH
if exist "%LOCALAPPDATA%\\Programs\\DockerDesktop\\resources\\bin" (
    set "PATH=%LOCALAPPDATA%\\Programs\\DockerDesktop\\resources\\bin;%PATH%"
)
if exist "C:\\Program Files\\Docker\\Docker\\resources\\bin" (
    set "PATH=C:\\Program Files\\Docker\\Docker\\resources\\bin;%PATH%"
)

:: Verify Docker executable exists
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker CLI not found! Please ensure Docker Desktop is installed.
    echo If Docker Desktop is installed, please add its bin folder to system PATH.
    pause
    exit /b 1
)

:: Verify Docker Daemon is running
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker daemon is not running!
    echo Please launch Docker Desktop and wait until the whale icon stops animating, then re-run this script.
    pause
    exit /b 1
)

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
curl.exe -s -o NUL -w "%%{http_code}" http://localhost:8080/api/metrics > "%TEMP%\\edge_hc.txt" 2>NUL
set /p HC_STATUS=<"%TEMP%\\edge_hc.txt"
if exist "%TEMP%\\edge_hc.txt" del "%TEMP%\\edge_hc.txt" >nul 2>&1

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
echo - Stop container: stop_docker_edge.bat
pause
"""

SWITCH_GDUT_BAT = """@echo off
setlocal enabledelayedexpansion
title CCTV Model Hot-Swap -^> GDUT 5-Class (Helmet Only SOTA)

echo ==============================================================================
echo [HOT-SWAP] Chuyen doi mo hinh sang GDUT 5-Class (rep_yolo11s_gdut_a6_deploy.pt)
echo ==============================================================================

:: Check if container API is reachable
curl.exe -s -o NUL -w "%%{http_code}" http://localhost:8080/api/metrics > "%TEMP%\\cctv_check.txt" 2>NUL
set /p API_CODE=<"%TEMP%\\cctv_check.txt"
if exist "%TEMP%\\cctv_check.txt" del "%TEMP%\\cctv_check.txt" >nul 2>&1

if not "!API_CODE!"=="200" (
    echo [ERROR] CCTV Container chua khoi dong hoac chua san sang [HTTP !API_CODE!].
    echo Vui long chay 'start_docker_edge.bat' truoc de khoi dong he thong!
    echo ==============================================================================
    pause
    exit /b 1
)

:: Create temporary payload JSON (immune to Windows CLI quote stripping)
echo {"model_path": "Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt"} > "%TEMP%\\swap_gdut_payload.json"

echo [*] Dang nap trong so GDUT 5-class vao GPU CUDA:0...
curl.exe -s -X POST http://localhost:8080/api/switch_model -H "Content-Type: application/json" -d @"%TEMP%\\swap_gdut_payload.json" > "%TEMP%\\swap_gdut_resp.txt"
if exist "%TEMP%\\swap_gdut_payload.json" del "%TEMP%\\swap_gdut_payload.json" >nul 2>&1

echo.
echo ==============================================================================
echo [KET QUA HOT-SWAP]:
type "%TEMP%\\swap_gdut_resp.txt"
if exist "%TEMP%\\swap_gdut_resp.txt" del "%TEMP%\\swap_gdut_resp.txt" >nul 2>&1
echo.
echo.
echo [DA KICH HOAT]: GDUT 5-Class Helmet-Only Mode!
echo - Classes: 0: blue, 1: white, 2: yellow, 3: red, 4: none
echo - Direct Helmet Violation Mode: Phat hien truc tiep dau khong doi mu bao ho.
echo - Web Dashboard da tu dong dong bo tai: http://localhost:8080
echo ==============================================================================
pause
"""

SWITCH_CHAMPION_BAT = """@echo off
setlocal enabledelayedexpansion
title CCTV Model Hot-Swap -^> Champion 7-Class (Person + PPE SOTA)

echo ==============================================================================
echo [HOT-SWAP] Chuyen doi mo hinh sang Champion 7-Class (rep_yolo11s_chv7_best_deploy.pt)
echo ==============================================================================

:: Check if container API is reachable
curl.exe -s -o NUL -w "%%{http_code}" http://localhost:8080/api/metrics > "%TEMP%\\cctv_check.txt" 2>NUL
set /p API_CODE=<"%TEMP%\\cctv_check.txt"
if exist "%TEMP%\\cctv_check.txt" del "%TEMP%\\cctv_check.txt" >nul 2>&1

if not "!API_CODE!"=="200" (
    echo [ERROR] CCTV Container chua khoi dong hoac chua san sang [HTTP !API_CODE!].
    echo Vui long chay 'start_docker_edge.bat' truoc de khoi dong he thong!
    echo ==============================================================================
    pause
    exit /b 1
)

:: Create temporary payload JSON (immune to Windows CLI quote stripping)
echo {"model_path": "Output/kaggle_results/hannhu40022_warmstart_ablations/rep_yolo11s_chv7_best_deploy.pt"} > "%TEMP%\\swap_champ_payload.json"

echo [*] Dang nap trong so Champion 7-class vao GPU CUDA:0...
curl.exe -s -X POST http://localhost:8080/api/switch_model -H "Content-Type: application/json" -d @"%TEMP%\\swap_champ_payload.json" > "%TEMP%\\swap_champ_resp.txt"
if exist "%TEMP%\\swap_champ_payload.json" del "%TEMP%\\swap_champ_payload.json" >nul 2>&1

echo.
echo ==============================================================================
echo [KET QUA HOT-SWAP]:
type "%TEMP%\\swap_champ_resp.txt"
if exist "%TEMP%\\swap_champ_resp.txt" del "%TEMP%\\swap_champ_resp.txt" >nul 2>&1
echo.
echo.
echo [DA KICH HOAT]: Champion 7-Class Person + PPE Relational Mode!
echo - Classes: 0: person, 1: safety_vest, 2: blue_helmet, 3: green_helmet, 4: red_helmet, 5: white_helmet, 6: yellow_helmet
echo - Spatial Relational ByteTrack MOT: Dinh danh cong nhan va gan ket mu + ao.
echo - Web Dashboard da tu dong dong bo tai: http://localhost:8080
echo ==============================================================================
pause
"""


def write_crlf(path: Path, content: str):
    crlf_content = content.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(path, "wb") as f:
        f.write(crlf_content.encode("utf-8"))
    print(f"Generated CRLF batch file: {path.name}")


if __name__ == "__main__":
    write_crlf(ROOT / "start_docker_edge.bat", START_EDGE_BAT)
    write_crlf(ROOT / "switch_to_gdut.bat", SWITCH_GDUT_BAT)
    write_crlf(ROOT / "switch_to_champion.bat", SWITCH_CHAMPION_BAT)
