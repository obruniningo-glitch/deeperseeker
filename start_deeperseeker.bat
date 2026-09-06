@echo off
title DeepSeeker Gateway — http://localhost:4000/
cd /d "%~dp0"

echo ============================================================
echo  DeepSeeker Gateway
echo  Listening on : http://localhost:4000/
echo  Auth tokens  : 2 ACTIVE
echo  Models       : instant  ^|  vision  ^|  expert
echo  API key      : dseeker
echo ============================================================
echo.

call deeperseeker_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    echo Expected: %~dp0deeperseeker_env\Scripts\activate.bat
    pause
    exit /b 1
)

python app.py

echo.
echo DeepSeeker stopped. Press any key to close this window.
pause
