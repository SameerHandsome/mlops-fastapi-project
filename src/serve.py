# src/serve.py
import os
import joblib
import time
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, JSONResponse
from huggingface_hub import hf_hub_download
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry

app = FastAPI(title="Loan Approval Serve")

# ENV / Secrets (set these in HF Space -> Secrets)
API_KEY = os.getenv("API_KEY")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Sameer-Handsome173/loan-approval-model")
HF_TOKEN = os.getenv("HF_TOKEN")
PROM_PUSHGATEWAY = os.getenv("PROM_PUSHGATEWAY")        # e.g. https://prometheus-prod-XX.grafana.net/api/prom/push/YOUR_ID
PROM_USERNAME = os.getenv("PROM_USERNAME")              # optional
PROM_API_KEY = os.getenv("PROM_API_KEY")                # optional
PROM_PUSHJOB = os.getenv("PROM_PUSHJOB", "loan_model")  # pushgateway job name

# Prometheus metrics (global)
REQ_COUNTER = Counter("pred_requests_total", "Total prediction requests")
REQ_LATENCY = Histogram("pred_request_latency_seconds", "Prediction latency seconds")
LATEST_PRED = Gauge("latest_prediction", "Last predicted value")

def safe_download(filename: str):
    """Download a file from HF model repo; return local path or None."""
    try:
        path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename, token=HF_TOKEN)
        print(f"Downloaded: {filename} -> {path}")
        return path
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

# Download artifacts
model_path = safe_download("best_model.joblib")
enc_path = safe_download("models/encoders.joblib")
scaler_path = safe_download("models/scaler.joblib")
cat_path = safe_download("models/categorical_columns.joblib")
bool_path = safe_download("models/boolean_columns.joblib")
feat_path = safe_download("models/feature_columns.joblib")

# Load artifacts (with safe defaults)
MODEL = None
ENCODERS = {}
SCALER = None
CATEGORICAL_COLUMNS = []
BOOLEAN_COLUMNS = []
FEATURE_COLUMNS = []

try:
    if model_path:
        MODEL = joblib.load(model_path)
        print("Model loaded.")
    if enc_path:
        ENCODERS = joblib.load(enc_path) or {}
        print("Encoders loaded:", list(ENCODERS.keys()))
    if scaler_path:
        SCALER = joblib.load(scaler_path)
        print("Scaler loaded.")
    if cat_path:
        CATEGORICAL_COLUMNS = joblib.load(cat_path) or []
        print("Categorical columns:", CATEGORICAL_COLUMNS)
    if bool_path:
        BOOLEAN_COLUMNS = joblib.load(bool_path) or []
        print("Boolean columns:", BOOLEAN_COLUMNS)
    if feat_path:
        FEATURE_COLUMNS = joblib.load(feat_path) or []
        print("Feature columns:", FEATURE_COLUMNS)
except Exception as e:
    print("Artifact load error:", e)

def map_boolean_series(s: pd.Series):
    """Map common boolean-like values to 1/0 and ensure integer dtype."""
    # Normalize strings, booleans, numerics
    def map_val(v):
        if pd.isna(v):
            return 0
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return 1 if v != 0 else 0
        vs = str(v).strip().lower()
        if vs in ("true", "1", "yes", "y", "t"):
            return 1
        if vs in ("false", "0", "no", "n", "f"):
            return 0
        # fallback
        try:
            # try numeric conversion
            nv = float(v)
            return 1 if nv != 0 else 0
        except Exception:
            return 0
    return s.map(map_val).astype(int)

def safe_label_encode_series(series: pd.Series, le):
    """Transform a pandas Series using LabelEncoder 'le' but handle unseen values by mapping to most frequent class."""
    # label encoder classes_
    classes = list(le.classes_)
    # vectorized transform: where value in classes -> transform, else map to classes[0]
    def transform_val(v):
        if pd.isna(v):
            return int(le.transform([classes[0]])[0])
        if v in classes:
            return int(le.transform([v])[0])
        # try to match string representation
        vs = str(v)
        if vs in classes:
            return int(le.transform([vs])[0])
        return int(le.transform([classes[0]])[0])
    return series.astype(str).map(transform_val).astype(int)

@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "features_count": len(FEATURE_COLUMNS),
        "categorical_count": len(CATEGORICAL_COLUMNS),
        "boolean_count": len(BOOLEAN_COLUMNS),
    }

@app.post("/predict")
def predict(payload: dict, x_api_key: str = Header(None)):
    # Auth
    if API_KEY is None:
        print("Warning: API_KEY is not set in environment. Disable auth only for testing.")
    if API_KEY and (x_api_key is None or x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key header")

    if MODEL is None or SCALER is None or len(FEATURE_COLUMNS) == 0:
        raise HTTPException(status_code=503, detail="Model or preprocessors not loaded")

    # Convert and align
    try:
        df = pd.DataFrame([payload])
        # ensure all feature columns exist in the dataframe
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # Apply boolean mapping first
        for bcol in BOOLEAN_COLUMNS:
            if bcol in df.columns:
                df[bcol] = map_boolean_series(df[bcol])

        # Apply label encoding for categorical columns
        for ccol in CATEGORICAL_COLUMNS:
            if ccol in df.columns:
                le = ENCODERS.get(ccol)
                if le is not None:
                    df[ccol] = safe_label_encode_series(df[ccol], le)
                else:
                    # If encoder missing, try to coerce to numeric fallback
                    df[ccol] = df[ccol].astype(str).map(lambda v: 0)

        # Reorder columns to training order
        df = df[FEATURE_COLUMNS]

        # Scale
        df_scaled = SCALER.transform(df)

        # Predict
        start = time.time()
        pred = MODEL.predict(df_scaled)[0]
        latency = time.time() - start

        # Prometheus metrics
        REQ_COUNTER.inc()
        REQ_LATENCY.observe(latency)
        LATEST_PRED.set(int(pred))

        # Push to Pushgateway if provided
        if PROM_PUSHGATEWAY:
            try:
                # Use a fresh registry with the metrics we want to push.
                registry = CollectorRegistry()
                # minimal metrics to push
                tmp_g = Gauge("latest_prediction", "Latest prediction (pushed)", registry=registry)
                tmp_g.set(int(pred))
                # if auth provided, use HTTP basic auth
                auth = None
                if PROM_USERNAME and PROM_API_KEY:
                    auth = (PROM_USERNAME, PROM_API_KEY)
                headers = {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
                # push via HTTP POST to pushgateway URL (if URL includes job etc, just post)
                # requests can handle auth if provided
                resp = requests.post(PROM_PUSHGATEWAY, data=generate_latest(registry), headers=headers, auth=auth, timeout=5)
                if resp.status_code >= 400:
                    print("Pushgateway responded:", resp.status_code, resp.text)
            except Exception as e:
                print("Pushgateway push failed:", e)

        return JSONResponse({
            "prediction": int(pred),
            "prediction_label": ("Approved" if int(pred) == 1 else "Rejected"),
            "latency_seconds": round(latency, 5),
            "features_used": FEATURE_COLUMNS
        })

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/metrics")
def metrics():
    """Return Prometheus metrics text for scraping."""
    data = generate_latest()
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")
