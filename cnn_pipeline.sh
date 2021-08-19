#!/usr/bin/env bash

sudo apt-get update

python3.7 -m venv env

source .env/bin/activate

pip install --upgrade pip

test -f requirements.txt && pip install requirements.txt

python src/cnn_resnet50.py

deactivate

echo "[INFO] Task completed!"