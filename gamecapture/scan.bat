@echo off
rem Level scanner. Double-click this.
rem
rem Only two things happen out here: finding Python, and making the private
rem environment that keeps this tool's packages away from the system ones.
rem Everything else — working out what is missing, saying what it is for, and
rem asking permission before downloading it — happens in Python, where it can be
rem a dialog instead of a wall of console text.
setlocal
cd /d "%~dp0"
title 3DHeat - level scanner

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"
if not defined PY (
  echo.
  echo   Python was not found, and it is needed to run the scanner.
  echo.
  choice /c YN /m "Install it now with winget"
  if errorlevel 2 (
    echo.
    echo   Nothing was installed. You can get Python from python.org
    echo   and then run this again.
    echo.
    pause
    exit /b 1
  )
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  if errorlevel 1 (
    echo   The install did not finish. Nothing else was changed.
    pause
    exit /b 1
  )
  echo.
  echo   Python installed. Close this window and run scan.bat again so Windows
  echo   picks up the new PATH.
  echo.
  pause
  exit /b 0
)

if not exist ".venv" (
  echo.
  echo   Setting up a private environment for this tool, so nothing is installed
  echo   into your system Python.
  echo.
  %PY% -m venv .venv || (echo Could not create it. & pause & exit /b 1)
  .venv\Scripts\python.exe -m pip install --upgrade pip -q
)

rem From here on Python does the asking.
.venv\Scripts\python.exe -m heat3d_capture.ui
if errorlevel 1 pause
