#!/bin/bash
set -euo pipefail

sudo apt update
sudo apt-get install -y git-lfs
git lfs install
git lfs pull
sudo apt install -y curl build-essential libssl-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libffi-dev zlib1g-dev libgdbm-dev liblzma-dev tk-dev

curl https://pyenv.run | bash

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
