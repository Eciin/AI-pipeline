# Jugo AI Pipeline

This repository is where we build the pipeline for the **Jugo** project.

The repo currently runs a local PaddleOCR-VL model from the checked-in `model/` directory and includes:

- `tools/predict.py` for direct command-line inference

## Setup

Windows:
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Linux server:
```bash
chmod +x setup_server.sh
./setup_server.sh
source .venv/bin/activate
```

## Pipeline

```text
Practical Architecture

Screenshot
|
v
Vision model (neural network)
|
v
Extract chart data
|
v
Rule engine / classical ML
|
v
Explanation model
```

Other models will be added soon.
