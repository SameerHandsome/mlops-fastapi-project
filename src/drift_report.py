import pandas as pd, mlflow, os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment('Drift_Detection')
ref = pd.read_csv('data/X.csv'); cur = ref.sample(frac=0.2, random_state=42)
rep = Report(metrics=[DataDriftPreset()]); rep.run(reference_data=ref, current_data=cur)
os.makedirs('reports', exist_ok=True); path='reports/drift.html'; rep.save_html(path)
with mlflow.start_run(run_name='drift_check'): mlflow.log_artifact(path)
print('✅ Drift report logged.')
