@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-BiBaZuShortcuts.ps1" %*
if errorlevel 1 (
  echo Shortcut installation failed. See error above.
  pause
  exit /b 1
)
echo Shortcuts installed.
pause
