@echo off
REM Launch the image_to_keychain web UI via WSL, then open it in your browser.
REM Double-click this file; leave the console window open while you use the UI.
setlocal
set "SCRIPT_DIR=%~dp0"
REM Convert C:\... to /mnt/c/... for WSL
set "WSL_DIR=/mnt/c%SCRIPT_DIR:~2%"
set "WSL_DIR=%WSL_DIR:\=/%"
start "" "http://localhost:7860"
wsl -- bash -c "cd '%WSL_DIR%' && python3 app.py"
