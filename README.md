# Jugo AI Pipeline

This repository is where we build the pipeline for the **Jugo** project.

The repo currently runs a local PaddleOCR-VL model from the checked-in `model/` directory and includes:

- `tools/launcher.py` for an interactive CLI flow
- `tools/predict.py` for direct command-line inference

## Setup

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Start the interactive launcher:

```bash
python tools/launcher.py
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
