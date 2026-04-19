#!/usr/bin/env bash
# Launch the image_to_keychain web UI on macOS.
#
# Double-click this file in Finder. On first run it creates a virtualenv
# and installs dependencies (~2 min). Subsequent launches are instant.
# The console window must stay open while you use the UI; close it to stop
# the server.
#
# If Finder warns "cannot be opened because it is from an unidentified
# developer", right-click the file once -> Open, then Allow in System
# Settings -> Privacy & Security. Every launch after that is one click.

set -e
cd "$(dirname "$0")"

# --- locate python3 ----------------------------------------------------------
PY="$(command -v python3 || true)"
if [[ -z "$PY" ]]; then
    echo
    echo "ERROR: python3 not found."
    echo "Install it from https://www.python.org/downloads/  or via Homebrew:"
    echo "    brew install python"
    echo
    echo "Press any key to close this window..."
    read -n 1
    exit 1
fi
PY_VER=$("$PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PY ($PY_VER)"

# --- virtualenv (one-time setup) --------------------------------------------
VENV=".venv"
if [[ ! -d "$VENV" ]]; then
    echo "Creating virtual environment in $VENV ..."
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- install deps if missing ------------------------------------------------
if ! python -c "import gradio, vtracer, trimesh, shapely, manifold3d" 2>/dev/null; then
    echo "Installing dependencies (this may take ~2 minutes on first run)..."
    python -m pip install -q --upgrade pip
    python -m pip install -q -r requirements.txt
fi

# --- open browser once the server is up -------------------------------------
# Gradio prints "Running on local URL" on startup; we open a tab after a
# short delay rather than waiting for that line so the user gets instant
# feedback.
(sleep 4 && open "http://localhost:7860") &

# --- run the app ------------------------------------------------------------
echo
echo "Server starting at http://localhost:7860 — close this window to stop it."
echo
python app.py
