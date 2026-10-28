@echo off
setlocal enabledelayedexpansion
title CCTV Model Hot-Swap -^> Champion 7-Class (Person + PPE SOTA)

echo ==============================================================================
echo [HOT-SWAP] Chuyen doi mo hinh sang Champion 7-Class (rep_yolo11s_chv7_best_deploy.pt)
echo ==============================================================================

:: Check if container API is reachable
curl.exe -s -o NUL -w "%%{http_code}" http://localhost:8080/api/metrics > "%TEMP%\cctv_check.txt" 2>NUL
set /p API_CODE=<"%TEMP%\cctv_check.txt"
if exist "%TEMP%\cctv_check.txt" del "%TEMP%\cctv_check.txt" >nul 2>&1

if not "!API_CODE!"=="200" (
    echo [ERROR] CCTV Container chua khoi dong hoac chua san sang [HTTP !API_CODE!].
    echo Vui long chay 'start_docker_edge.bat' truoc de khoi dong he thong!
    echo ==============================================================================
    pause
    exit /b 1
)

:: Create temporary payload JSON (immune to Windows CLI quote stripping)
echo {"model_path": "Output/kaggle_results/hannhu40022_warmstart_ablations/rep_yolo11s_chv7_best_deploy.pt"} > "%TEMP%\swap_champ_payload.json"

echo [*] Dang nap trong so Champion 7-class vao GPU CUDA:0...
curl.exe -s -X POST http://localhost:8080/api/switch_model -H "Content-Type: application/json" -d @"%TEMP%\swap_champ_payload.json" > "%TEMP%\swap_champ_resp.txt"
if exist "%TEMP%\swap_champ_payload.json" del "%TEMP%\swap_champ_payload.json" >nul 2>&1

echo.
echo ==============================================================================
echo [KET QUA HOT-SWAP]:
type "%TEMP%\swap_champ_resp.txt"
if exist "%TEMP%\swap_champ_resp.txt" del "%TEMP%\swap_champ_resp.txt" >nul 2>&1
echo.
echo.
echo [DA KICH HOAT]: Champion 7-Class Person + PPE Relational Mode!
echo - Classes: 0: person, 1: safety_vest, 2: blue_helmet, 3: green_helmet, 4: red_helmet, 5: white_helmet, 6: yellow_helmet
echo - Spatial Relational ByteTrack MOT: Dinh danh cong nhan va gan ket mu + ao.
echo - Web Dashboard da tu dong dong bo tai: http://localhost:8080
echo ==============================================================================
pause
