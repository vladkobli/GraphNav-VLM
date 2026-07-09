#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Rebuild LeRobot venv for Feetech motor control
# Python: 3.12.3
# Venv:   /home/vladkobli/rocon-demos/lerobot/.venv
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python3}"
LEROBOT_DIR="${LEROBOT_DIR:-/home/vladkobli/rocon-demos/lerobot}"
VENV_DIR="${VENV_DIR:-$LEROBOT_DIR/.venv}"

# Set FIX_AUTOSOURCE=1 when running this script if you want it to
# comment out hardcoded auto-source lines from shell config files.
FIX_AUTOSOURCE="${FIX_AUTOSOURCE:-0}"

echo "[INFO] Python:"
$PYTHON_BIN --version

PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$PY_VER" != "3.12" ]]; then
  echo "[ERROR] Expected Python 3.12.x, got Python $PY_VER"
  echo "        Run with: PYTHON_BIN=python3.12 $0"
  exit 1
fi

echo "[INFO] Installing system dependencies..."
sudo apt update
sudo apt install -y \
  python3.12-venv \
  python3.12-dev \
  python3-pip \
  git \
  build-essential \
  cmake \
  pkg-config \
  ffmpeg

echo "[INFO] Creating LeRobot folder if missing:"
echo "       $LEROBOT_DIR"
mkdir -p "$LEROBOT_DIR"

echo "[INFO] Removing old venv:"
echo "       $VENV_DIR"
rm -rf "$VENV_DIR"

echo "[INFO] Creating fresh venv..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo "[INFO] Activating venv..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Upgrading pip only, not setuptools/packaging blindly..."
python -m pip install --upgrade pip

echo "[INFO] Writing dependency constraints..."
CONSTRAINTS_FILE="$LEROBOT_DIR/lerobot_constraints.txt"

cat > "$CONSTRAINTS_FILE" <<'EOF'
setuptools>=71,<81
packaging>=24.2,<26
EOF

echo "[INFO] Installing pinned packaging tools..."
PIP_CONSTRAINT="$CONSTRAINTS_FILE" python -m pip install \
  "setuptools>=71,<81" \
  "packaging>=24.2,<26" \
  wheel

echo "[INFO] Installing LeRobot from pip..."

# Try the feetech extra first. If the extra is ignored or unavailable, normal
# lerobot still contains the imports you used successfully.
PIP_CONSTRAINT="$CONSTRAINTS_FILE" python -m pip install "lerobot[feetech]==0.5.1" || \
PIP_CONSTRAINT="$CONSTRAINTS_FILE" python -m pip install "lerobot==0.5.1"

echo "[INFO] Installing extra packages used by your server scripts..."
PIP_CONSTRAINT="$CONSTRAINTS_FILE" python -m pip install \
  numpy \
  requests \
  pyserial

echo "[INFO] Re-pin setuptools and packaging after all installs..."
python -m pip install --force-reinstall \
  "setuptools>=71,<81" \
  "packaging>=24.2,<26" \
  wheel

echo "[INFO] Running pip check..."
python -m pip check

echo "[INFO] Verifying LeRobot Feetech imports..."
python - <<'PY'
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
import lerobot
import numpy as np
import packaging
import setuptools

print("OK: LeRobot Feetech imports work")
print("lerobot file:", lerobot.__file__)
print("numpy:", np.__version__)
print("packaging:", packaging.__version__)
print("setuptools:", setuptools.__version__)
PY

echo "[INFO] Checking serial access group..."
if groups "$USER" | grep -qw dialout; then
  echo "[OK] User $USER is already in dialout group."
else
  echo "[WARN] User $USER is not in dialout group."
  echo "       Adding now. You may need to log out/in afterwards."
  sudo usermod -aG dialout "$USER"
fi

echo "[INFO] Checking for unwanted auto-activation lines..."
AUTO_SOURCE_PATTERN="$VENV_DIR/bin/activate"

SHELL_CONFIGS=(
  "$HOME/.bashrc"
  "$HOME/.profile"
  "$HOME/.bash_profile"
  "$HOME/.zshrc"
)

FOUND_AUTOSOURCE=0

for cfg in "${SHELL_CONFIGS[@]}"; do
  if [[ -f "$cfg" ]] && grep -q "$AUTO_SOURCE_PATTERN" "$cfg"; then
    FOUND_AUTOSOURCE=1
    echo "[WARN] Found auto-source line in: $cfg"
    grep -n "$AUTO_SOURCE_PATTERN" "$cfg" || true

    if [[ "$FIX_AUTOSOURCE" == "1" ]]; then
      echo "[INFO] Commenting out matching lines in $cfg"
      sed -i.bak "\|$AUTO_SOURCE_PATTERN| s|^|# DISABLED_BY_REBUILD_LEROBOT_VENV: |" "$cfg"
      echo "[INFO] Backup saved as: $cfg.bak"
    fi
  fi
done

if [[ "$FOUND_AUTOSOURCE" == "0" ]]; then
  echo "[OK] No hardcoded auto-source line found in basic shell configs."
else
  if [[ "$FIX_AUTOSOURCE" != "1" ]]; then
    echo "[NOTE] To automatically comment out those lines, rerun:"
    echo "       FIX_AUTOSOURCE=1 $0"
  fi
fi

echo
echo "[DONE] LeRobot venv rebuilt successfully."
echo
echo "To use it manually:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To run your camera sweep server:"
echo "  source $VENV_DIR/bin/activate"
echo "  python $LEROBOT_DIR/lerobot/lerobot_camera_sweep_server.py"
echo
echo "If /dev/ttyACM0 permission still fails, log out/in or run:"
echo "  newgrp dialout"