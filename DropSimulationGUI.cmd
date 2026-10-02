@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-DropSimulationGUI.ps1"
if errorlevel 1 (
  echo Drop Simulation GUI could not start. See error above.
  pause
  exit /b 1
)
