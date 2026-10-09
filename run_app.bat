@echo off
title ISL Translator Desktop App
echo =======================================================
echo   Starting Indian Sign Language Translator App...
echo   (Loading TensorFlow and models, please wait...)
echo =======================================================
echo.

call "%~dp0venv\Scripts\activate.bat"
"%~dp0venv\Scripts\python.exe" "%~dp0app.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo Error launching application. See message above.
    echo =======================================================
    pause
)
