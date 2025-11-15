import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
import wandb
from huggingface_hub import HfApi

# Get environment variables
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
HF_MODEL_REPO = os.getenv('HF_MODEL_REPO')

# Check required environment variables
if not WANDB_API_KEY:
    print("❌ ERROR: WANDB_API_KEY environment variable is not set!")
    print("Make sure it's configured in GitHub Actions secrets")
    sys.exit(1)

if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN environment variable is not set!")
    sys.exit(1)

if not HF_MODEL_REPO:
    print("❌ ERROR: HF_MODEL_REPO environment variable is not set!")
    sys.exit(1)

# Login to W&B
print("🔐 Logging into W&B...")
wandb.login(key=WANDB_API_KEY)
print("✅ W&B login successful")

# Load preprocessed data
print("\n📊 Loading data...")
X = pd.read_csv('data/X.csv')
y = pd.read_csv('data/y.csv').values.ravel()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Evaluation function
def evaluate(model):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    try:
        if len(set(y_test)) == 2:
            auc = roc_auc_score(y_test, preds)
        else:
            auc = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    return acc, f1, auc

# Ridge Classifier with Optuna
print("\n🔍 Tuning RidgeClassifier...")
def ridge_objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    model = RidgeClassifier(alpha=alpha).fit(X_train, y_train)
    acc, _, _ = evaluate(model)
    return 1 - acc

ridge_study = optuna.create_study(direction='minimize')
ridge_study.optimize(ridge_objective, n_trials=10, show_progress_bar=True)

ridge_model = RidgeClassifier(alpha=ridge_study.best_params['alpha']).fit(X_train, y_train)
ridge_acc, ridge_f1, ridge_auc = evaluate(ridge_model)
print(f"✅ RidgeClassifier - Acc: {ridge_acc:.4f}, F1: {ridge_f1:.4f}, AUC: {ridge_auc:.4f}")

# Decision Tree with Optuna
print("\n🔍 Tuning DecisionTreeClassifier...")
def dt_objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    ).fit(X_train, y_train)
    acc, _, _ = evaluate(model)
    return 1 - acc

dt_study = optuna.create_study(direction='minimize')
dt_study.optimize(dt_objective, n_trials=10, show_progress_bar=True)

dt_model = DecisionTreeClassifier(**dt_study.best_params, random_state=42).fit(X_train, y_train)
dt_acc, dt_f1, dt_auc = evaluate(dt_model)
print(f"✅ DecisionTreeClassifier - Acc: {dt_acc:.4f}, F1: {dt_f1:.4f}, AUC: {dt_auc:.4f}")

# Select best model
if ridge_acc > dt_acc:
    best_model = ridge_model
    best_name = 'RidgeClassifier'
    best_acc, best_f1, best_auc = ridge_acc, ridge_f1, ridge_auc
    best_params = ridge_study.best_params
else:
    best_model = dt_model
    best_name = 'DecisionTreeClassifier'
    best_acc, best_f1, best_auc = dt_acc, dt_f1, dt_auc
    best_params = dt_study.best_params

print(f"\n🏆 Best Model: {best_name}")
print(f"   Accuracy: {best_acc:.4f}")
print(f"   F1 Score: {best_f1:.4f}")
print(f"   ROC AUC: {best_auc:.4f}")

# Log to W&B
print("\n📊 Logging to W&B...")
wandb.init(project="loan-approval-mlops", name=best_name)
wandb.log({
    "model": best_name,
    "accuracy": best_acc,
    "f1_score": best_f1,
    "roc_auc": best_auc,
    **best_params
})
wandb.finish()
print("✅ Logged to W&B")

# Save model locally
print("\n💾 Saving model artifacts...")
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.joblib')

# Upload to Hugging Face
print("\n☁️  Uploading to Hugging Face...")
api = HfApi()

files_to_upload = [
    ('models/best_model.joblib', 'best_model.joblib'),
    ('models/encoders.joblib', 'models/encoders.joblib'),
    ('models/scaler.joblib', 'models/scaler.joblib'),
    ('models/feature_columns.joblib', 'models/feature_columns.joblib'),
    ('models/categorical_columns.joblib', 'models/categorical_columns.joblib'),
    ('models/boolean_columns.joblib', 'models/boolean_columns.joblib')
]

for local_path, hf_path in files_to_upload:
    if os.path.exists(local_path):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path,
                repo_id=HF_MODEL_REPO,
                repo_type='model',
                token=HF_TOKEN
            )
            print(f"  ✅ Uploaded {hf_path}")
        except Exception as e:
            print(f"  ❌ Failed to upload {hf_path}: {e}")
    else:
        print(f"  ⚠️  {local_path} not found, skipping")

print("\n🎉 Training and deployment complete!")
