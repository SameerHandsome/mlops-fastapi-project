# src/train.py
import os
import joblib
import pandas as pd
import numpy as np
import wandb
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna

# ENV
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "loan-approval")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Load processed data
X = pd.read_csv("data/X.csv")
y = pd.read_csv("data/y.csv").values.ravel().astype(int)

# Load preprocess artifacts
encoders = joblib.load("models/encoders.joblib")
scaler = joblib.load("models/scaler.joblib")
categorical_cols = joblib.load("models/categorical_columns.joblib")
boolean_cols = joblib.load("models/boolean_columns.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

def evaluate(model):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    return acc, f1, auc

# Optuna objectives
def ridge_opt(trial):
    alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
    model = RidgeClassifier(alpha=alpha)
    model.fit(X_train, y_train)
    acc, _, _ = evaluate(model)
    return 1 - acc

def dt_opt(trial):
    md = trial.suggest_int("max_depth", 2, 20)
    mss = trial.suggest_int("min_samples_split", 2, 10)
    model = DecisionTreeClassifier(max_depth=md, min_samples_split=mss)
    model.fit(X_train, y_train)
    acc, _, _ = evaluate(model)
    return 1 - acc

def run_training():
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT, job_type="train")

    # Ridge tuning
    ridge_study = optuna.create_study(direction="minimize")
    ridge_study.optimize(ridge_opt, n_trials=15)
    best_alpha = ridge_study.best_params["alpha"]

    ridge = RidgeClassifier(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    r_acc, r_f1, r_auc = evaluate(ridge)

    # DT tuning
    dt_study = optuna.create_study(direction="minimize")
    dt_study.optimize(dt_opt, n_trials=15)
    dt_params = dt_study.best_params

    dt = DecisionTreeClassifier(**dt_params)
    dt.fit(X_train, y_train)
    d_acc, d_f1, d_auc = evaluate(dt)

    # Pick best
    if r_acc > d_acc:
        best_model = ridge
        best_name = "RidgeClassifier"
        metrics = {"accuracy": r_acc, "f1": r_f1, "auc": r_auc, "alpha": best_alpha}
    else:
        best_model = dt
        best_name = "DecisionTreeClassifier"
        metrics = {"accuracy": d_acc, "f1": d_f1, "auc": d_auc, **dt_params}

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")

    # Log to wandb
    run.summary.update({"best_model": best_name, **metrics})
    art = wandb.Artifact("best_model", type="model")
    art.add_file("models/best_model.joblib")
    art.add_file("models/encoders.joblib")
    art.add_file("models/scaler.joblib")
    art.add_file("models/categorical_columns.joblib")
    art.add_file("models/boolean_columns.joblib")
    art.add_file("models/feature_columns.joblib")
    run.log_artifact(art)
    run.finish()

    # Upload to HF for serving
    api = HfApi()
    file_map = {
        "models/best_model.joblib": "best_model.joblib",
        "models/encoders.joblib": "models/encoders.joblib",
        "models/scaler.joblib": "models/scaler.joblib",
        "models/categorical_columns.joblib": "models/categorical_columns.joblib",
        "models/boolean_columns.joblib": "models/boolean_columns.joblib",
        "models/feature_columns.joblib": "models/feature_columns.joblib",
    }

    for local, remote in file_map.items():
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )

if __name__ == "__main__":
    run_training()
