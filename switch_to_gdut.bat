@echo off
setlocal enabledelayedexpansion
title CCTV Model Hot-Swap -^> GDUT 5-Class (Helmet Only SOTA)

echo ==============================================================================
echo [HOT-SWAP] Chuyen doi mo hinh sang GDUT 5-Class (rep_yolo11s_gdut_a6_deploy.pt)
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
echo {"model_path": "Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt"} > "%TEMP%\swap_gdut_payload.json"

echo [*] Dang nap trong so GDUT 5-class vao GPU CUDA:0...
curl.exe -s -X POST http://localhost:8080/api/switch_model -H "Content-Type: application/json" -d @"%TEMP%\swap_gdut_payload.json" > "%TEMP%\swap_gdut_resp.txt"
if exist "%TEMP%\swap_gdut_payload.json" del "%TEMP%\swap_gdut_payload.json" >nul 2>&1

echo.
echo ==============================================================================
echo [KET QUA HOT-SWAP]:
type "%TEMP%\swap_gdut_resp.txt"
if exist "%TEMP%\swap_gdut_resp.txt" del "%TEMP%\swap_gdut_resp.txt" >nul 2>&1
echo.
echo.
echo [DA KICH HOAT]: GDUT 5-Class Helmet-Only Mode!
echo - Classes: 0: blue, 1: white, 2: yellow, 3: red, 4: none
echo - Direct Helmet Violation Mode: Phat hien truc tiep dau khong doi mu bao ho.
echo - Web Dashboard da tu dong dong bo tai: http://localhost:8080
echo ==============================================================================
pause
