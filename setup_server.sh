#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "python3.13 is required but was not found."
  echo "Install Python 3.13 on the server, then rerun this script."
  exit 1
fi

if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt was not found in ${ROOT_DIR}."
  exit 1
fi

if [[ ! -d model/PaddleOCR-VL-1.5 ]]; then
  echo "model/PaddleOCR-VL-1.5 is missing."
  echo "This repo needs the local model files before it can run."
  exit 1
fi

python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Environment is ready."
echo "Venv: ${ROOT_DIR}/.venv"
echo "Run with: source .venv/bin/activate && python tools/predict.py"
