#!/usr/bin/env bash
# Linux / macOS twin of start.bat: creates .venv on first run, installs, launches the dashboard.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment ..."
  "$PY" -m venv .venv
fi
if ! .venv/bin/python -c "import trading_bot, lightgbm, alpaca, sklearn, statsmodels" >/dev/null 2>&1; then
  echo "Installing the pinned dependencies (first run only) ..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.lock
  .venv/bin/python -m pip install -e . --no-deps
fi
echo "Starting the dashboard; API keys are saved to settings.json next to this script (git-ignored)."
exec .venv/bin/python -m trading_bot.main gui "$@"
