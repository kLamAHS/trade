@echo off
setlocal
title Fractional Trading Bot
cd /d "%~dp0"

rem ---- find Python 3 -------------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run start.bat again.
  pause
  exit /b 1
)

rem ---- create the virtual environment on first run --------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment ...
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

rem ---- install / update dependencies when needed ----------------------------
"%VPY%" -c "import trading_bot, lightgbm, alpaca, sklearn, statsmodels" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies ^(first run only, this can take a few minutes^) ...
  "%VPY%" -m pip install --upgrade pip >nul
  "%VPY%" -m pip install -e ".[dev,plots]" || goto :fail
)

rem ---- launch the dashboard -------------------------------------------------
echo.
echo Starting the dashboard. Enter your Alpaca API key on the page; keys are saved to settings.json
echo next to this file ^(never committed^). Close this window or press Ctrl+C to stop the bot.
echo.
"%VPY%" -m trading_bot.main gui %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo Something went wrong. Scroll up for the error message.
pause
exit /b 1
