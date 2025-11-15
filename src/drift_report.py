import pandas as pd
import os
import sys
import wandb

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    USE_OLD_EVIDENTLY = True
except ImportError:
    try:
        from evidently.report import Report
        from evidently.metrics import DataDriftTable
        USE_OLD_EVIDENTLY = False
    except ImportError:
        print("❌ ERROR: evidently package not installed correctly")
        sys.exit(1)

# Check environment variables
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    print(" ERROR: WANDB_API_KEY environment variable is not set!")
    sys.exit(1)


print(" Logging into W&B..")
wandb.login(key=WANDB_API_KEY)
print(" W&B login successful")

# Load reference data
print("\n Loading reference data...")
ref = pd.read_csv("data/X.csv")
print(f"Reference data shape: {ref.shape}")

# Check if current batch exists, otherwise use sample
cur_file = "data/current_batch.csv"
if not os.path.exists(cur_file):
    print("No current batch found, using sample from reference data...")
    cur = ref.sample(frac=0.2, random_state=42)
else:
    print(f"📊 Loading current batch from {cur_file}...")
    cur = pd.read_csv(cur_file)

print(f"Current data shape: {cur.shape}")

# Generate drift report
print("\n📈 Generating drift report...")
try:
    if USE_OLD_EVIDENTLY:
        report = Report(metrics=[DataDriftPreset()])
    else:
        report = Report(metrics=[DataDriftTable()])
    
    report.run(reference_data=ref, current_data=cur)
    print("✅ Drift analysis complete")
except Exception as e:
    print(f"❌ Error generating drift report: {e}")
    sys.exit(1)

# Save report
os.makedirs("reports", exist_ok=True)
report_path = "reports/drift_report.html"
report.save_html(report_path)
print(f"✅ Drift report saved to {report_path}")

try:
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "loan-approval-mlops"),
        job_type="drift-monitoring"
    )
    
    # Create and log artifact
    artifact = wandb.Artifact("drift_report", type="report")
    artifact.add_file(report_path)
    run.log_artifact(artifact)
    
    run.finish()
    print("✅ Drift report uploaded to W&B")
except Exception as e:
    print(f"❌ Error uploading to W&B: {e}")
    sys.exit(1)

print("\n🎉 Drift detection complete!")
