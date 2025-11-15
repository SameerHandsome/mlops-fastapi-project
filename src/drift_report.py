import pandas as pd
import os
import sys
import wandb

# Try importing evidently with proper error handling
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    USE_OLD_EVIDENTLY = True
    print("✅ Using evidently with DataDriftPreset")
except ImportError:
    try:
        from evidently.report import Report
        from evidently.metrics import DataDriftTable
        USE_OLD_EVIDENTLY = False
        print("✅ Using evidently with DataDriftTable")
    except ImportError:
        try:
            # Try even newer version
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metrics import DatasetDriftMetric
            USE_OLD_EVIDENTLY = None
            print("✅ Using evidently with DatasetDriftMetric")
        except ImportError as e:
            print(f"❌ ERROR: Could not import evidently: {e}")
            print("\nTrying to check evidently installation:")
            try:
                import evidently
                print(f"Evidently version: {evidently.__version__}")
                print(f"Evidently location: {evidently.__file__}")
                print("\nAvailable modules:")
                import pkgutil
                for importer, modname, ispkg in pkgutil.iter_modules(evidently.__path__):
                    print(f"  - {modname}")
            except Exception as check_e:
                print(f"Error checking evidently: {check_e}")
            sys.exit(1)

# Check environment variables
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    print("❌ ERROR: WANDB_API_KEY environment variable is not set!")
    sys.exit(1)

# Login to W&B
print("🔐 Logging into W&B...")
wandb.login(key=WANDB_API_KEY)
print("✅ W&B login successful")

# Load reference data
print("\n📊 Loading reference data...")
ref = pd.read_csv("data/X.csv")
print(f"Reference data shape: {ref.shape}")

# Check if current batch exists, otherwise use sample
cur_file = "data/current_batch.csv"
if not os.path.exists(cur_file):
    print("⚠️  No current batch found, using sample from reference data...")
    cur = ref.sample(frac=0.2, random_state=42)
else:
    print(f"📊 Loading current batch from {cur_file}...")
    cur = pd.read_csv(cur_file)

print(f"Current data shape: {cur.shape}")

# Generate drift report
print("\n📈 Generating drift report...")
try:
    if USE_OLD_EVIDENTLY == True:
        # Old evidently version
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
    elif USE_OLD_EVIDENTLY == False:
        # Medium evidently version
        report = Report(metrics=[DataDriftTable()])
        report.run(reference_data=ref, current_data=cur)
    else:
        # Newest evidently version
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=ref, current_data=cur)
    
    print("✅ Drift analysis complete")
except Exception as e:
    print(f"❌ Error generating drift report: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save report
os.makedirs("reports", exist_ok=True)
report_path = "reports/drift_report.html"
report.save_html(report_path)
print(f"✅ Drift report saved to {report_path}")

# Upload to W&B
print("\n☁️  Uploading to W&B...")
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
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Drift detection complete!")
