# MLOps Project — Loan Approval Prediction

This repository contains an end-to-end MLOps example for a Loan Approval prediction system. The code trains models, logs experiments, saves model artifacts, generates drift reports, and serves predictions via a FastAPI app with Prometheus metrics. Artifacts can be uploaded to the Hugging Face Model Hub and experiment metadata is logged to Weights & Biases (W&B). An MLflow server is also provided as a Docker image to track experiments locally if needed.

Notes:
- All descriptions, commands, and environment variable names below are derived directly from the repository source code.
- Search the repository for more files or details: https://github.com/SameerHandsome/mlops-fastapi-project/search

Table of contents
- Project overview
- Architecture & logic flow
- Tech stack
- Repository layout
- Environment variables
- How to run (training, serving, drift reporting, MLflow)
- API endpoints
- Prometheus / monitoring
- Troubleshooting & notes

---

## Project overview

This project trains and evaluates classification models to predict loan approval. It:
- Loads preprocessed datasets from `data/X.csv` and `data/y.csv`.
- Tunes models (RidgeClassifier and DecisionTree) using Optuna.
- Selects the best model by evaluation metrics (accuracy, F1, ROC AUC).
- Persists model artifacts (model file, encoders, scaler, and feature lists).
- Uploads artifacts to Hugging Face Model Hub (if environment variables are set).
- Logs runs and artifacts to Weights & Biases.
- Generates data-drift reports using Evidently and uploads the report to W&B.
- Serves predictions using a FastAPI app that loads artifacts (from HF Hub or local path), performs preprocessing, returns predictions, and exposes Prometheus metrics.

All behavior and names of files and environment variables are taken from the code (see sections below).

---

## Architecture & logic flow

High-level flows (as implemented in code):

1. Training pipeline (src/train.py)
   - Reads preprocessed features and labels from `data/X.csv` and `data/y.csv`.
   - Splits into train/test.
   - Defines an evaluation routine that computes accuracy, F1, and ROC AUC.
   - Uses Optuna to tune hyperparameters for RidgeClassifier and DecisionTreeClassifier (each optimized over a small number of trials).
   - Fits each model and computes evaluation metrics.
   - Selects the best model (best accuracy is used to select between the two).
   - Logs metrics and hyperparameters to W&B (project `loan-approval-mlops`).
   - Saves artifacts to `models/`:
     - `models/best_model.joblib`
     - `models/encoders.joblib`
     - `models/scaler.joblib`
     - `models/categorical_columns.joblib`
     - `models/boolean_columns.joblib`
     - `models/feature_columns.joblib`
   - Attempts to upload artifacts to a Hugging Face model repository (via `HfApi`) if `HF_TOKEN` and `HF_MODEL_REPO` are provided.

2. Serving pipeline (src/serve.py)
   - FastAPI application that:
     - Loads model artifacts at startup using `huggingface_hub.hf_hub_download(...)` (paths used in code are the same as artifacts saved by training).
     - Provides a health endpoint (`/`) returning `model_loaded`, `features_count`, etc.
     - Provides a prediction endpoint (`/predict`) which:
       - Optionally enforces API-key auth using the `x-api-key` header and `API_KEY` env var.
       - Aligns/fills incoming JSON payload to required `FEATURE_COLUMNS`.
       - Applies boolean mapping for known boolean-like columns.
       - Applies safe label encoding for categorical columns using stored LabelEncoders (handles unseen categories by mapping to the most frequent class).
       - Re-orders columns to training order, scales using stored scaler, and predicts using the loaded model.
       - Responds with JSON giving numeric prediction, a human label (`Approved` / `Rejected`), latency, and features used.
       - Updates Prometheus metrics (counter, histogram, gauge) and optionally pushes metrics to a Prometheus Pushgateway (`PROM_PUSHGATEWAY`).
     - Provides a Prometheus scrape endpoint (`/metrics`) that returns Prometheus plaintext metrics.

3. Drift reporting (src/drift_report.py)
   - Uses Evidently to compute data drift between reference data (`data/X.csv`) and a current batch (if `data/current_batch.csv` exists, otherwise uses a sampled subset).
   - Attempts to handle multiple versions of Evidently APIs (the code has guarded imports to support different versions).
   - Saves a drift report HTML to `reports/drift_report.html`.
   - Uploads the drift report to W&B as an artifact (project name can be taken from `WANDB_PROJECT` env var).

4. MLflow server (mlflow-space)
   - A small Dockerfile exposes a Python 3.10-slim image with MLflow installed.
   - `start_mlflow.sh` runs `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts`.
   - The Dockerfile exposes port 5000 and provides an entrypoint to start the MLflow server.

---

## Tech stack (derived from imports and files)
- Python (Dockerfile uses Python 3.10)
- FastAPI (HTTP server)
- uvicorn (typical ASGI server; use to run FastAPI)
- scikit-learn (RidgeClassifier, DecisionTreeClassifier, scaler)
- Optuna (hyperparameter tuning)
- joblib (model & object persistence)
- pandas, numpy (data handling)
- wandb (Weights & Biases experiment tracking)
- huggingface_hub (download/upload model artifacts)
- prometheus_client (metrics, pushgateway integration)
- requests (pushgateway HTTP requests)
- evidently (data drift / monitoring)
- mlflow (server Dockerfile and start script)
- Docker (mlflow-space image)

---

## Repository layout (relevant files observed in code)
- src/
  - train.py         — training + Optuna + W&B + save & upload artifacts
  - serve.py         — FastAPI server, downloads artifacts, preprocessing, predict endpoint, /metrics
  - drift_report.py  — data drift generation via Evidently and W&B upload
- data/
  - X.csv, y.csv     — preprocessed data expected by training code (must exist)
  - current_batch.csv — optional batch for drift detection
- models/            — generated by training (best_model.joblib, encoders, scaler, lists)
- reports/           — drift report output (drift_report.html)
- mlflow-space/
  - Dockerfile
  - start_mlflow.sh

---

## Environment variables (names and defaults pulled from code)

Training (src/train.py)
- WANDB_API_KEY — required (train exits if missing)
- HF_TOKEN — required for HF uploads (train exits if missing)
- HF_MODEL_REPO — required (train exits if missing). Example default used in serve: `"Sameer-Handsome173/loan-approval-model"` (serve provides that default, but train expects it to be set).

Serving (src/serve.py)
- API_KEY — optional; used to validate `x-api-key` header (comment in code: "set these in HF Space -> Secrets")
- HF_MODEL_REPO — default set in code to "Sameer-Handsome173/loan-approval-model"
- HF_TOKEN — used by huggingface_hub for download if the repo is private
- PROM_PUSHGATEWAY — optional Pushgateway URL (e.g. Grafana Cloud push URL)
- PROM_USERNAME, PROM_API_KEY — optional credentials for pushgateway basic auth
- PROM_PUSHJOB — optional override for job name (default `loan_model` in code)

Drift reporting (src/drift_report.py)
- WANDB_API_KEY — required (drift_report exits if missing)
- WANDB_PROJECT — optional (used for wandb.init, default in code: `loan-approval-mlops`)

MLflow (mlflow-space)
- None required inside Dockerfile; start script uses sqlite local DB and local artifact root.

---

## How to run

All commands below assume a Python environment with required dependencies installed. The exact requirements file is not in the repository, so install packages used in code: fastapi, uvicorn, scikit-learn, optuna, pandas, numpy, joblib, wandb, huggingface_hub, prometheus_client, evidently, requests.

1) Training
- Ensure `data/X.csv` and `data/y.csv` exist.
- Set environment variables:
  - WANDB_API_KEY (required)
  - HF_TOKEN (required if you want to upload to Hugging Face)
  - HF_MODEL_REPO (required)
- Run training:
  - python src/train.py
- Output:
  - Saved artifacts under `models/` (joblib files).
  - Logs sent to W&B and (if configured) model artifacts uploaded to the Hugging Face model repo.

2) Serving (FastAPI)
- Ensure artifacts are available either locally under `models/` or uploaded to a HF model repo referenced by `HF_MODEL_REPO`.
- Set environment variables (if needed):
  - HF_TOKEN (for private HF repos)
  - HF_MODEL_REPO (if not using the default)
  - API_KEY (optional; if set, client must send `x-api-key` header)
  - PROM_PUSHGATEWAY, PROM_USERNAME, PROM_API_KEY (optional push to Prometheus Pushgateway)
- Start server (example using uvicorn):
  - pip install uvicorn
  - uvicorn src.serve:app --host 0.0.0.0 --port 8000
- Endpoints:
  - GET /        — health, model_loaded, counts
  - POST /predict — JSON payload with feature values (see API section below)
  - GET /metrics  — Prometheus metrics text (for scraping)
- Example (cURL):
  - curl http://localhost:8000/          # health
  - curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature_1": 1, "feature_2": "x"}'
    - Note: payload must be a JSON object mapping feature names to values. The model expects the same features and column order used during training; missing features are filled with zero by the server code.

3) Drift report
- Ensure `data/X.csv` (reference) exists.
- Optionally provide `data/current_batch.csv` for current data; otherwise the code samples a subset of `X.csv`.
- Set:
  - WANDB_API_KEY (required)
  - optionally WANDB_PROJECT
- Run:
  - python src/drift_report.py
- Output:
  - `reports/drift_report.html` saved locally and uploaded to W&B as an artifact.

4) MLflow server (local)
- Build and run Dockerfile in `mlflow-space`:
  - docker build -t mlflow-local mlflow-space/
  - docker run -p 5000:5000 mlflow-local
- The container runs:
  - mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
- MLflow UI will be available on port 5000 inside the mapped port.

---

## API endpoints (behavior derived from src/serve.py)

- GET /
  - Returns JSON:
    - status: "ok"
    - model_loaded: bool
    - features_count: int
    - categorical_count: int
    - boolean_count: int

- POST /predict
  - Accepts a JSON object (single record). The server:
    - Fills missing features referenced by `FEATURE_COLUMNS` with zero.
    - Applies boolean and categorical preprocessing using stored encoders.
    - Scales features and runs model .predict(...) returning first item.
  - Response JSON includes:
    - prediction: integer (0/1)
    - prediction_label: "Approved" or "Rejected"
    - latency_seconds: float
    - features_used: list of feature column names
  - Authentication:
    - If `API_KEY` env var is set, the request must include header `x-api-key: <API_KEY>`.

- GET /metrics
  - Returns Prometheus-format metrics for scraping (generated via `prometheus_client.generate_latest()`).

---

## Prometheus / Monitoring

- The server maintains three Prometheus metrics:
  - Counter: `pred_requests_total`
  - Histogram: `pred_request_latency_seconds`
  - Gauge: `latest_prediction`
- If `PROM_PUSHGATEWAY` is provided, the server will attempt to POST the minimal metrics payload to that URL after every prediction. If `PROM_USERNAME` and `PROM_API_KEY` are provided, they are used as HTTP basic auth for the push.

---

## Troubleshooting & notes (from code behavior)

- train.py and drift_report.py exit immediately if required environment variables are missing (WANDB_API_KEY and HF_TOKEN / HF_MODEL_REPO).
- serve.py prints warnings if API_KEY is not set (auth disabled) and prints error traces if any artifact fails to load.
- The serve code includes robust handling for unseen categorical values in label encoders (maps to most frequent class).
- drift_report.py has guarded imports for different Evidently versions and will print information about installed Evidently if it cannot import expected modules.
- All artifact filenames expected by the server are:
  - `best_model.joblib`
  - `models/encoders.joblib`
  - `models/scaler.joblib`
  - `models/categorical_columns.joblib`
  - `models/boolean_columns.joblib`
  - `models/feature_columns.joblib`
  Ensure these artifacts are present locally or present in the configured HF model repo.


I generated the README above by scanning the repository code (train, serve, drift_report, and mlflow-space). If you'd like any section expanded, or want me to produce a requirements file and run instructions for a specific environment (Docker Compose, GitHub Actions, or HF Space), tell me which and I'll produce it next.
