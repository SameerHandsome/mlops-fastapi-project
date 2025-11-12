from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import pandas as pd, os, joblib, time, requests
from huggingface_hub import hf_hub_download
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = FastAPI()

API_KEY = os.getenv("API_KEY")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
PROM_PUSHGATEWAY = os.getenv("PROM_PUSHGATEWAY")

REQS = Counter("pred_requests_total", "Total prediction requests")
LAT = Histogram("pred_request_latency_seconds", "Request latency")
LATEST = Gauge("latest_prediction", "Last predicted value")

try:
    m = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib")
    e = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/encoders.joblib")
    s = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/scaler.joblib")
    f = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/feature_columns.joblib")
    c = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/categorical_columns.joblib")
    b = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/boolean_columns.joblib")

    model = joblib.load(m)
    encoders = joblib.load(e)
    scaler = joblib.load(s)
    feature_columns = joblib.load(f)
    categorical_columns = joblib.load(c)
    boolean_columns = joblib.load(b)
    loaded = True
except Exception as ex:
    print("Model load error:", ex)
    loaded = False
    model = None
    encoders = {}
    scaler = None
    feature_columns = []
    categorical_columns = []
    boolean_columns = []


@app.get("/")
def health():
    return {
        "status": "ok", 
        "model_loaded": loaded, 
        "features": feature_columns,
        "categorical_features": categorical_columns,
        "boolean_features": boolean_columns
    }


@app.post("/predict")
def predict(payload: dict, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([payload])

        for col in boolean_columns:
            if col in df.columns:
                if df[col].dtype == bool:
                    df[col] = df[col].astype(int)
                elif df[col].dtype == 'object':
                   
                    df[col] = df[col].map({
                        'True': 1, 'true': 1, 'TRUE': 1, True: 1, 1: 1, '1': 1,
                        'False': 0, 'false': 0, 'FALSE': 0, False: 0, 0: 0, '0': 0
                    }).fillna(0).astype(int)
                else:
                    df[col] = df[col].astype(int)
        
        # Apply label encoding to categorical features
        for col in categorical_columns:
            if col in df.columns:
                if col in encoders:
                    # Handle unseen categories
                    try:
                        df[col] = encoders[col].transform(df[col])
                    except ValueError:
                        # If category not seen during training, use most frequent class (0)
                        df[col] = 0
        
        # Reorder columns to match training data
        df = df[feature_columns]
        
        # Scale all features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        start = time.time()
        pred = model.predict(df_scaled)[0]
        latency = time.time() - start
        
        # Update Prometheus metrics
        LAT.observe(latency)
        REQS.inc()
        LATEST.set(pred)

        # Push metrics to Prometheus Pushgateway if configured
        if PROM_PUSHGATEWAY:
            try:
                requests.post(
                    f"{PROM_PUSHGATEWAY}/metrics/job/loan_model", 
                    data=generate_latest()
                )
            except Exception as push_error:
                print(f"Failed to push metrics: {push_error}")

        return {
            "prediction": int(pred),
            "prediction_label": "Approved" if pred == 1 else "Rejected",
            "latency_seconds": round(latency, 4),
            "features_used": feature_columns
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
def metrics():
    return generate_latest()