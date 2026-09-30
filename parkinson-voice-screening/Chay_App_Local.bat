@echo off
title ParkiScan Desktop
echo ===================================================
echo   ParkiScan - Voice Parkinson Screening (Desktop)
echo ===================================================
echo.

set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"

cd /d "%~dp0"
python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    pause
)
