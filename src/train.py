import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow, optuna
from huggingface_hub import HfApi

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment('Loan_Approval_Models')

X, y = pd.read_csv('data/X.csv'), pd.read_csv('data/y.csv').values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

def ridge(trial):
    a = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    m = RidgeClassifier(alpha=a).fit(X_train, y_train)
    acc, _, _ = evaluate(m)
    return 1 - acc

r = optuna.create_study(direction='minimize')
r.optimize(ridge, n_trials=10, show_progress_bar=True)
ridge_model = RidgeClassifier(alpha=r.best_params['alpha']).fit(X_train, y_train)
ridge_acc, ridge_f1, ridge_auc = evaluate(ridge_model)

print(f"RidgeClassifier - Acc: {ridge_acc:.4f}, F1: {ridge_f1:.4f}, AUC: {ridge_auc:.4f}")

def dt(trial):
    d = trial.suggest_int('max_depth', 2, 20)
    s = trial.suggest_int('min_samples_split', 2, 10)
    m = DecisionTreeClassifier(max_depth=d, min_samples_split=s, random_state=42).fit(X_train, y_train)
    acc, _, _ = evaluate(m)
    return 1 - acc

t = optuna.create_study(direction='minimize')
t.optimize(dt, n_trials=10, show_progress_bar=True)
dt_model = DecisionTreeClassifier(**t.best_params, random_state=42).fit(X_train, y_train)
dt_acc, dt_f1, dt_auc = evaluate(dt_model)

print(f"DecisionTreeClassifier - Acc: {dt_acc:.4f}, F1: {dt_f1:.4f}, AUC: {dt_auc:.4f}")

if ridge_acc > dt_acc:
    best_model, best_name = ridge_model, 'RidgeClassifier'
    best_acc, best_f1, best_auc = ridge_acc, ridge_f1, ridge_auc
    best_params = r.best_params
else:
    best_model, best_name = dt_model, 'DecisionTreeClassifier'
    best_acc, best_f1, best_auc = dt_acc, dt_f1, dt_auc
    best_params = t.best_params

print(f"\n🏆 Best Model: {best_name}")
print(f"Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")

# Log to MLflow
with mlflow.start_run(run_name=best_name):
    mlflow.log_params({'model': best_name, **best_params})
    mlflow.log_metrics({
        'Accuracy': best_acc,
        'F1_Score': best_f1,
        'ROC_AUC': best_auc
    })
    mlflow.sklearn.log_model(best_model, 'model')

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.joblib')

api = HfApi()
repo, token = os.getenv('HF_MODEL_REPO'), os.getenv('HF_TOKEN')

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
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=repo,
            repo_type='model',
            token=token
        )
        print(f"✅ Uploaded {hf_path}")
    else:
        print(f"⚠️  Warning: {local_path} not found, skipping upload")

print('✅ Model training and upload complete.')