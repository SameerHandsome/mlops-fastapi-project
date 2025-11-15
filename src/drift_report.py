import pandas as pd
import os
import wandb
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

wandb.login(key=os.getenv("WANDB_API_KEY"))

ref = pd.read_csv("data/X.csv")
cur = ref.sample(frac=0.2, random_state=42)


if not os.path.exists(cur_file):
    print("❌ No current batch found.")
else:
    cur = pd.read_csv(cur_file)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")

    run = wandb.init(project=os.getenv("WANDB_PROJECT", "loan-approval"), job_type="drift")
    art = wandb.Artifact("drift_report", type="report")
    art.add_file("reports/drift_report.html")
    run.log_artifact(art)
    run.finish()
